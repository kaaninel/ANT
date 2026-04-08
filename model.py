import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig


# ---------------------------------------------------------------------------
# Static KV Cache — pre-allocated, no torch.cat per step
# ---------------------------------------------------------------------------

class StaticKVCache:
    """Pre-allocated KV cache that writes in-place instead of torch.cat.

    Usage:
        cache = StaticKVCache(n_layers, batch, n_heads, max_seq, head_dim, device)
        # Prefill: writes positions 0..T-1
        logits, halt, cache = model(prompt, kv_cache=cache, cache_position=0)
        # Decode: writes position T, T+1, ...
        logits, halt, cache = model(tok, kv_cache=cache, cache_position=T)
    """
    __slots__ = ('k', 'v', 'pos')

    def __init__(self, n_layers: int, batch: int, n_heads: int,
                 max_seq: int, head_dim: int, device, dtype=None):
        if dtype is None:
            dtype = torch.float32
        shape = (n_layers, batch, n_heads, max_seq, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int, n: int):
        """Write k,v at positions [pos, pos+n) for given layer."""
        self.k[layer_idx, :, :, pos:pos + n, :] = k
        self.v[layer_idx, :, :, pos:pos + n, :] = v

    def read(self, layer_idx: int, end_pos: int):
        """Read k,v for positions [0, end_pos) for given layer."""
        return self.k[layer_idx, :, :, :end_pos, :], self.v[layer_idx, :, :, :end_pos, :]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.weight * x_float * rms).to(dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Returns (cos, sin) each of shape (max_seq_len, head_dim//2)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)           # (max_seq_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x   : (B, n_heads, T, head_dim)
    cos : (T, head_dim//2)
    sin : (T, head_dim//2)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)     # (1, 1, T, half)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


# ---------------------------------------------------------------------------
# SiLU FFN
# ---------------------------------------------------------------------------

class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.up   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


# ---------------------------------------------------------------------------
# AddrNet — Small co-processor for hierarchical address generation
# ---------------------------------------------------------------------------

class AddrNet(nn.Module):
    """Generates a variable-depth hierarchical address from a hidden state.

    Runs `depth` internal clock cycles. Each cycle:
      1. Project current state → logits over n_bins
      2. Pick bin (argmax in forward, Gumbel-softmax for training)
      3. Condition on choice via residual + nonlinearity

    Output: list of `depth` bin indices forming a trie path.
    """
    def __init__(self, d_model: int = 128, hidden_dim: int = 16,
                 n_bins: int = 256, depth: int = 8):
        super().__init__()
        self.depth = depth
        self.n_bins = n_bins
        self.proj_in = nn.Linear(d_model, hidden_dim)
        self.bin_embed = nn.Embedding(n_bins, hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_bins)

    def forward(self, hidden_state: torch.Tensor, temperature: float = 1.0):
        """
        hidden_state: (B, d_model) or (d_model,)
        Returns: (B, depth) int64 tensor of bin indices
        """
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        B = hidden_state.shape[0]
        h = self.proj_in(hidden_state)          # (B, hidden_dim)
        bins = []
        for _ in range(self.depth):
            logits = self.out(h)                # (B, n_bins)
            if self.training and temperature > 0:
                # Gumbel-softmax for differentiable training
                soft = F.gumbel_softmax(logits, tau=temperature, hard=True)
                bin_idx = soft.argmax(dim=-1)   # (B,)
                h = h + (soft @ self.bin_embed.weight)  # differentiable embed
            else:
                bin_idx = logits.argmax(dim=-1) # (B,)
                h = h + self.bin_embed(bin_idx) # (B, hidden_dim)
            h = F.silu(self.mlp(h))
            bins.append(bin_idx)
        return torch.stack(bins, dim=1)         # (B, depth)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        d = cfg.d_model
        self.q = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache=None,
        _layer_idx: int = -1,
        _cache_position: int = -1,
    ) -> tuple:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = self.k(x).view(B, T, H, D).transpose(1, 2)
        v = self.v(x).view(B, T, H, D).transpose(1, 2)

        # RoPE on Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV-cache handling
        if isinstance(kv_cache, StaticKVCache):
            pos = _cache_position
            kv_cache.write(_layer_idx, k, v, pos, T)
            k, v = kv_cache.read(_layer_idx, pos + T)
            new_cache = kv_cache
        elif kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            new_cache = (k, v)
        else:
            # No prior cache (prefill or training) — still return K/V for
            # list-based caching to populate on first call
            new_cache = (k, v)

        # mask=None: no masking (incremental decode)
        # mask="causal": is_causal=True (prefill)
        # Otherwise: additive mask
        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        elif isinstance(mask, str) and mask == "causal":
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True)
        else:
            attn_mask = mask.unsqueeze(0).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0)

        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o(out), new_cache


# ---------------------------------------------------------------------------
# Memory Cross-Attention
# ---------------------------------------------------------------------------

class MemoryAttention(nn.Module):
    """Cross-attention to external key-value memory.

    Full Q/K/V/O projections so each head can specialize in reading
    different aspects of memory (entity names, locations, relations, etc.).

    Includes a learnable inverse-temperature per head for sharper attention
    when entity discrimination is needed.

    Supports top-k sparse attention with straight-through estimator (STE):
    forward uses only top-k slots per query, backward flows through full softmax.
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int, topk: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.topk = topk
        inner = n_heads * head_dim

        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)

        # Per-head learnable inverse temperature (init=1.0, can sharpen)
        self.inv_temp = nn.Parameter(torch.ones(n_heads))

    def forward(self, x: torch.Tensor, mem_keys: torch.Tensor,
                mem_values: torch.Tensor,
                mem_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        S = mem_keys.shape[1]
        H, D = self.n_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)          # (B, H, T, D)
        k = self.k(mem_keys).view(B, S, H, D).transpose(1, 2)   # (B, H, S, D)
        v = self.v(mem_values).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)

        # Apply per-head temperature scaling to queries
        temp = self.inv_temp.view(1, H, 1, 1)  # (1, H, 1, 1)
        q = q * temp

        effective_topk = self.topk
        if effective_topk <= 0 or effective_topk >= S:
            # Standard softmax attention
            attn_mask = None
            if mem_mask is not None:
                attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
                attn_mask.masked_fill_(~mem_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        else:
            # Top-K sparse attention with STE
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,T,S)

            if mem_mask is not None:
                pad_mask = ~mem_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
                scores = scores.masked_fill(pad_mask, float('-inf'))

            # Full softmax (used as gradient surrogate)
            full_attn = F.softmax(scores, dim=-1)

            # Top-k mask: keep only the k highest-scoring slots per query position
            k_clamped = min(effective_topk, S)
            topk_vals, _ = scores.topk(k_clamped, dim=-1)        # (B,H,T,k)
            threshold = topk_vals[..., -1:]                        # (B,H,T,1)
            topk_mask = (scores >= threshold).float()              # (B,H,T,S)

            # Sparse attention: mask out non-top-k, renormalize
            sparse_attn = full_attn * topk_mask
            sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)

            # STE: forward uses sparse_attn, backward flows through full_attn
            attn = full_attn + (sparse_attn - full_attn).detach()

            out = torch.matmul(attn, v)  # (B, H, T, D)

        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o(out)


# ---------------------------------------------------------------------------
# Transformer Block (with tag + memory cross-attention)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Layer order: Self-Attention → Tag-Attention → Memory-Attention → FFN."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = Attention(cfg)

        # Tag cross-attention (persistent context register)
        self.use_tag = getattr(cfg, 'use_tag_system', False)
        if self.use_tag:
            self.norm_tag = RMSNorm(cfg.d_model)
            self.tag_head = nn.Linear(cfg.d_model, cfg.d_model)
            self.tag_gate = nn.Linear(cfg.d_model, 1)

        # Memory cross-attention (trie-retrieved vectors)
        self.use_mem_attn = getattr(cfg, 'use_memory_cross_attention', False)
        if self.use_mem_attn:
            self.norm_mem = RMSNorm(cfg.d_model)
            topk = getattr(cfg, 'memory_topk', 0)
            self.mem_attn = MemoryAttention(
                cfg.d_model, cfg.n_heads, cfg.head_dim, topk=topk)

        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SiLUFFN(cfg.d_model, cfg.ffn_dim)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache=None,
        mem_keys: torch.Tensor | None = None,
        mem_values: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
        tag_register: torch.Tensor | None = None,
        _layer_idx: int = -1,
        _cache_position: int = -1,
    ) -> tuple:
        # 1. Self-attention
        attn_out, new_cache = self.attn(
            self.norm1(x), mask, cos, sin, kv_cache,
            _layer_idx=_layer_idx, _cache_position=_cache_position)
        x = x + self.drop(attn_out)

        # 2. Tag cross-attention (GRU-style gated update from tag register)
        if self.use_tag and tag_register is not None:
            normed = self.norm_tag(x)
            new_tag = torch.tanh(self.tag_head(normed))
            gate = torch.sigmoid(self.tag_gate(normed))
            tag_context = gate * new_tag + (1 - gate) * tag_register.unsqueeze(1)
            x = x + self.drop(tag_context)

        # 3. Memory cross-attention (trie-retrieved vectors)
        if self.use_mem_attn and mem_keys is not None:
            x = x + self.drop(self.mem_attn(
                self.norm_mem(x), mem_keys, mem_values, mem_mask))

        # 4. FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, new_cache


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ANT(nn.Module):
    """ANT — ~892K param byte-level transformer with hierarchical persistent memory."""
    def __init__(self, cfg: ModelConfig, use_checkpoint: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_checkpoint = use_checkpoint

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)

        # Halt head for ACT: bias initialized to [0, 0] (50/50 CONTINUE/HALT)
        self.halt_head = nn.Linear(cfg.d_model, 2, bias=True)
        nn.init.constant_(self.halt_head.bias, 0.0)

        # AddrNets: 3 separate co-processors generating hierarchical addresses
        n_nets = getattr(cfg, 'n_addr_nets', 3)
        hidden = getattr(cfg, 'addr_hidden_dim', 16)
        n_bins = getattr(cfg, 'addr_n_bins', 256)
        depth = getattr(cfg, 'addr_depth', 8)
        self.addr_nets = nn.ModuleList([
            AddrNet(cfg.d_model, hidden, n_bins, depth) for _ in range(n_nets)
        ])

        # V_proj: projects hidden state to stored value format
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # RoPE cache (text positions only — memory is via cross-attention now)
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Causal mask for text-only self-attention
        neg_inf = -1e9
        n_text_max = cfg.max_seq_len
        text_only = torch.triu(
            torch.full((n_text_max, n_text_max), neg_inf), diagonal=1
        )
        self.register_buffer("text_only_mask", text_only)

    def make_cache(self, batch_size: int, max_seq: int = 0,
                   device=None, dtype=None) -> StaticKVCache:
        """Create a pre-allocated static KV cache for generation."""
        if max_seq == 0:
            max_seq = self.cfg.max_seq_len
        if device is None:
            device = self.embed.weight.device
        if dtype is None:
            dtype = self.embed.weight.dtype
        return StaticKVCache(
            self.cfg.n_layers, batch_size, self.cfg.n_heads,
            max_seq, self.cfg.head_dim, device, dtype)

    # ------------------------------------------------------------------
    # Address computation (via AddrNets)
    # ------------------------------------------------------------------

    def compute_addresses(self, hidden_state: torch.Tensor, temperature: float = 1.0):
        """
        hidden_state : (d_model,) or (B, d_model)
        Returns list of N_addr_nets tensors, each (B, depth) int64.
        """
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        return [net(hidden_state, temperature) for net in self.addr_nets]

    def compute_value(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state to memory value format.
        hidden_state: (B, d_model) or (d_model,)
        Returns: same shape as input.
        """
        return self.v_proj(hidden_state)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        memory_keys: torch.Tensor | None = None,
        memory_values: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tag_register: torch.Tensor | None = None,
        return_hidden: bool = False,
        kv_cache: list | None = None,
        cache_position: int = 0,
        bidirectional: bool = False,
    ):
        """
        token_ids      : (B, T_text)
        memory_keys    : (B, S, d_model) — trie-retrieved vectors used as keys
        memory_values  : (B, S, d_model) — trie-retrieved vectors used as values
        memory_mask    : (B, S) bool — True for valid memory slots
        tag_register   : (B, d_model) — persistent tag context vector
        kv_cache       : StaticKVCache or list of (k,v) tuples, or None
        cache_position : position offset for RoPE when using KV-cache
        bidirectional  : if True, use full (non-causal) self-attention mask
        Returns (logits, halt_logits[, hidden][, new_kv_cache]).
        """
        B, T_text = token_ids.shape
        device = token_ids.device

        x = self.embed(token_ids)   # (B, T_text, d_model)

        # Cross-attention memory from trie
        cross_keys = memory_keys
        cross_vals = memory_values
        cross_mask = memory_mask

        T_new = x.shape[1]

        # Determine if this is prefill or incremental
        use_static = isinstance(kv_cache, StaticKVCache)
        is_incremental = kv_cache is not None and cache_position > 0

        if is_incremental:
            cos = self.rope_cos[cache_position:cache_position + T_new]
            sin = self.rope_sin[cache_position:cache_position + T_new]
            mask = None
        else:
            cos = self.rope_cos[:T_new]
            sin = self.rope_sin[:T_new]
            if bidirectional:
                mask = torch.zeros(T_new, T_new, device=device)
            else:
                mask = "causal"

        if use_static:
            for i, layer in enumerate(self.layers):
                if self.use_checkpoint and self.training:
                    x, _ = checkpoint(layer, x, mask, cos, sin, None,
                                      cross_keys, cross_vals, cross_mask,
                                      tag_register,
                                      use_reentrant=False)
                else:
                    x, _ = layer(x, mask, cos, sin, kv_cache=kv_cache,
                                 mem_keys=cross_keys, mem_values=cross_vals,
                                 mem_mask=cross_mask,
                                 tag_register=tag_register,
                                 _layer_idx=i, _cache_position=cache_position)
            kv_cache.pos = cache_position + T_new
            new_kv_cache = kv_cache
        else:
            new_kv_cache = []
            for i, layer in enumerate(self.layers):
                layer_cache = kv_cache[i] if kv_cache is not None and i < len(kv_cache) else None
                if self.use_checkpoint and self.training:
                    x, _ = checkpoint(layer, x, mask, cos, sin, None,
                                      cross_keys, cross_vals, cross_mask,
                                      tag_register,
                                      use_reentrant=False)
                    new_kv_cache.append(None)
                else:
                    x, layer_kv = layer(x, mask, cos, sin, kv_cache=layer_cache,
                                        mem_keys=cross_keys, mem_values=cross_vals,
                                        mem_mask=cross_mask,
                                        tag_register=tag_register)
                    new_kv_cache.append(layer_kv)

        hidden = self.norm(x)
        logits = F.linear(hidden, self.embed.weight)
        halt_logits = self.halt_head(hidden)

        if kv_cache is not None:
            if return_hidden:
                return logits, halt_logits, hidden, new_kv_cache
            return logits, halt_logits, new_kv_cache
        if return_hidden:
            return logits, halt_logits, hidden
        return logits, halt_logits
