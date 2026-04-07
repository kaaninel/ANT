"""MLX port of LoopedLatentController — optimised for autoregressive generation.

Apple Silicon unified-memory, lazy evaluation, and fused kernels
eliminate the per-step dispatch overhead that bottlenecks PyTorch/MPS.

Only the inference path is ported (no training, no gradient checkpointing).
Training stays in PyTorch; weights are converted at load time.
"""

import math
import mlx.core as mx
import mlx.nn as nn

from config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((d_model,))

    def __call__(self, x):
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * x * rms


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, half, dtype=mx.float32) / half))
    t = mx.arange(max_seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, H, T, D)  cos/sin: (T, D//2)"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


# ---------------------------------------------------------------------------
# SiLU FFN
# ---------------------------------------------------------------------------

class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.up(x)))


# ---------------------------------------------------------------------------
# Attention (causal self-attention with KV cache)
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

    def __call__(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = mx.concatenate([cached_k, k], axis=2)
            v = mx.concatenate([cached_v, v], axis=2)
        new_cache = (k, v)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, T, S)
        S = k.shape[2]

        if T > 1:
            # Causal mask for prefill
            mask = mx.triu(mx.full((T, S), float('-inf')), k=S - T + 1)
            scores = scores + mask[None, None, :, :]

        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.o(out), new_cache


# ---------------------------------------------------------------------------
# Memory Cross-Attention (multi-hop)
# ---------------------------------------------------------------------------

class MemoryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, topk: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)
        self.inv_temp = mx.ones((n_heads,))

    def __call__(self, x, mem_keys, mem_values, mem_mask=None):
        B, T, _ = x.shape
        S = mem_keys.shape[1]
        H, D = self.n_heads, self.head_dim

        q = self.q(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k(mem_keys).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        v = self.v(mem_values).reshape(B, S, H, D).transpose(0, 2, 1, 3)

        temp = self.inv_temp.reshape(1, H, 1, 1)
        q = q * temp

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D)
        if mem_mask is not None:
            pad_mask = mx.logical_not(mem_mask)[:, None, None, :]
            scores = mx.where(pad_mask, mx.full(scores.shape, float('-inf')), scores)

        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.o(out)


class MultiHopMemoryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, topk: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.q1 = nn.Linear(d_model, inner, bias=False)
        self.q2 = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)
        self.inv_temp = mx.ones((n_heads,))

    def _attend(self, q, k, v, mask_val):
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask_val is not None:
            scores = scores + mask_val
        weights = mx.softmax(scores, axis=-1)
        return weights @ v

    def __call__(self, x, mem_keys, mem_values, mem_mask=None):
        B, T, _ = x.shape
        S = mem_keys.shape[1]
        H, D = self.n_heads, self.head_dim

        k = self.k(mem_keys).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        v = self.v(mem_values).reshape(B, S, H, D).transpose(0, 2, 1, 3)

        mask_val = None
        if mem_mask is not None:
            pad_mask = mx.logical_not(mem_mask)[:, None, None, :]
            mask_val = mx.where(pad_mask, mx.full((1,), float('-inf')), mx.zeros((1,)))

        temp = self.inv_temp.reshape(1, H, 1, 1)

        q1 = self.q1(x).reshape(B, T, H, D).transpose(0, 2, 1, 3) * temp
        hop1 = self._attend(q1, k, v, mask_val)
        hop1_out = hop1.transpose(0, 2, 1, 3).reshape(B, T, H * D)

        q2 = self.q2(x + hop1_out).reshape(B, T, H, D).transpose(0, 2, 1, 3) * temp
        hop2 = self._attend(q2, k, v, mask_val)
        hop2_out = hop2.transpose(0, 2, 1, 3).reshape(B, T, H * D)

        return self.o(hop2_out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)

        self.use_mem_attn = getattr(cfg, 'use_memory_cross_attention', False)
        if self.use_mem_attn:
            self.norm_mem = RMSNorm(cfg.d_model)
            topk = getattr(cfg, 'memory_topk', 0)
            hops = getattr(cfg, 'memory_hops', 1)
            if hops >= 2:
                self.mem_attn = MultiHopMemoryAttention(
                    cfg.d_model, cfg.n_heads, cfg.head_dim, topk=topk)
            else:
                self.mem_attn = MemoryAttention(
                    cfg.d_model, cfg.n_heads, cfg.head_dim, topk=topk)

        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SiLUFFN(cfg.d_model, cfg.ffn_dim)

    def __call__(self, x, cos, sin, kv_cache=None,
                 mem_keys=None, mem_values=None, mem_mask=None):
        attn_out, new_cache = self.attn(self.norm1(x), cos, sin, kv_cache)
        x = x + attn_out

        if self.use_mem_attn and mem_keys is not None:
            x = x + self.mem_attn(self.norm_mem(x), mem_keys, mem_values, mem_mask)

        x = x + self.ffn(self.norm2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class LoopedLatentControllerMLX(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.d_model)

        self.halt_head = nn.Linear(cfg.d_model, 2, bias=True)

        total_pos = cfg.n_mem_positions + cfg.max_seq_len
        cos, sin = precompute_rope(cfg.head_dim, total_pos, cfg.rope_theta)
        self.rope_cos = cos
        self.rope_sin = sin

    def __call__(self, token_ids, kv_cache=None, cache_position=0,
                 memory_keys=None, memory_values=None, memory_mask=None):
        """
        token_ids: (B, T) int32
        kv_cache: list of (k, v) per layer or None
        Returns: (logits, halt_logits, new_kv_cache)
        """
        B, T = token_ids.shape
        x = self.embed(token_ids)

        is_incremental = kv_cache is not None and cache_position > 0
        if is_incremental:
            cos = self.rope_cos[cache_position:cache_position + T]
            sin = self.rope_sin[cache_position:cache_position + T]
        else:
            cos = self.rope_cos[:T]
            sin = self.rope_sin[:T]

        new_kv = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, lkv = layer(x, cos, sin, kv_cache=layer_cache,
                           mem_keys=memory_keys, mem_values=memory_values,
                           mem_mask=memory_mask)
            new_kv.append(lkv)

        hidden = self.norm(x)
        logits = hidden @ self.embed.weight.T
        halt_logits = self.halt_head(hidden)
        return logits, halt_logits, new_kv

    def generate(self, prompt, gen_len=64):
        """Autoregressive generation — the path we're optimising."""
        # Prefill
        logits, _, kv = self(prompt)
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        tokens = [tok]
        pos = prompt.shape[1]

        # Decode loop
        for _ in range(gen_len - 1):
            logits, _, kv = self(tok, kv_cache=kv, cache_position=pos)
            tok = mx.argmax(logits[:, -1:, :], axis=-1)
            tokens.append(tok)
            pos += 1

        mx.eval(tokens[-1])  # force evaluation of full graph
        return mx.concatenate([prompt] + tokens, axis=1)


# ---------------------------------------------------------------------------
# Weight conversion: PyTorch → MLX
# ---------------------------------------------------------------------------

def convert_torch_to_mlx(torch_model) -> dict:
    """Convert PyTorch state_dict to MLX-compatible weight dict."""
    import torch as th
    sd = torch_model.state_dict()
    mlx_weights = {}

    key_map = {
        'embed.weight': 'embed.weight',
        'norm.weight': 'norm.weight',
        'halt_head.weight': 'halt_head.weight',
        'halt_head.bias': 'halt_head.bias',
    }

    for k, v in sd.items():
        np_val = v.detach().cpu().float().numpy()

        # Skip buffers we recompute
        if any(skip in k for skip in ['rope_cos', 'rope_sin', 'text_only_mask',
                                       'mem_text_mask']):
            continue

        # Map layer keys
        mlx_key = k
        # addr_heads not needed for inference gen
        if 'addr_heads' in k or 'temporal_emb' in k:
            continue

        # MemoryAttention inv_temp is a Parameter in torch, plain array in MLX
        if 'inv_temp' in k:
            mlx_weights[mlx_key] = mx.array(np_val)
            continue

        mlx_weights[mlx_key] = mx.array(np_val)

    return mlx_weights


def load_from_torch(torch_model, cfg: ModelConfig):
    """Create MLX model and load weights from a PyTorch model."""
    mlx_model = LoopedLatentControllerMLX(cfg)
    weights = convert_torch_to_mlx(torch_model)

    # Load weights into MLX model
    flat = {}
    for k, v in weights.items():
        flat[k] = v

    # Only load weights that exist in the model (skip recomputed buffers)
    model_keys = set()
    for k, _ in mlx_model.parameters().items():
        model_keys.add(k)
    # Flatten nested params to dot-separated keys
    def _flatten(prefix, obj):
        keys = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys |= _flatten(f"{prefix}.{k}" if prefix else k, v)
        else:
            keys.add(prefix)
        return keys
    model_keys = _flatten("", mlx_model.parameters())

    filtered = [(k, v) for k, v in flat.items() if k in model_keys]
    mlx_model.load_weights(filtered, strict=False)
    return mlx_model


def load_from_checkpoint(path: str, cfg: ModelConfig):
    """Load MLX model from a PyTorch checkpoint file."""
    import torch
    from model import LoopedLatentController

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['model'] if 'model' in ckpt else ckpt

    torch_model = LoopedLatentController(cfg, use_checkpoint=False)
    torch_model.load_state_dict(sd, strict=False)

    return load_from_torch(torch_model, cfg)
