from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    ANT — ~892K parameter byte-level transformer with persistent hierarchical memory.

    Pure 256-byte vocabulary (token ID = raw byte value).
    Special tokens mapped to ASCII control characters:
        PAD=NUL(0x00)  BOS=STX(0x02)  EOS=ETX(0x03)  ANS=ENQ(0x05)
        MEM_START=SOH(0x01)  MEM_END=EOT(0x04)  NOOP=ACK(0x06)  UNK=SUB(0x1A)
    """
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    head_dim: int = 32
    ffn_dim: int = 256
    n_layers: int = 4
    max_seq_len: int = 192
    dropout: float = 0.0
    rope_theta: float = 10000.0
    pad_id: int = 0x00                # NUL
    eos_id: int = 0x03                # ETX
    bos_id: int = 0x02                # STX
    unk_id: int = 0x1A                # SUB
    mem_start_id: int = 0x01          # SOH
    mem_end_id: int = 0x04            # EOT
    noop_id: int = 0x06               # ACK
    # Cross-attention memory
    use_memory_cross_attention: bool = True
    memory_topk: int = 0              # 0=softmax, >0=top-k sparse attention
    # AddrNet: 3 separate co-processors generating hierarchical addresses
    n_addr_nets: int = 3              # number of address paths
    addr_hidden_dim: int = 16         # internal hidden dim of each AddrNet
    addr_n_bins: int = 256            # bins per trie level
    addr_depth: int = 8               # max address depth (hard cap)
    # Tag system: persistent context register
    use_tag_system: bool = True
    # Memory cross-attention slot count (3 paths × (depth+1) levels)
    n_mem_slots: int = 25             # 3 × 9 - 2 shared roots = ~25


@dataclass
class MemoryConfig:
    """Hierarchical trie memory with float32 mmap storage."""
    data_path: str = "data_cache/memory"
    ema_alpha_base: float = 0.1       # base EMA momentum for leaf writes
    ema_alpha_min: float = 0.001      # minimum EMA alpha
    depth_cap: int = 8                # hard cap on address depth
    coarse_depth: int = 2             # depth 0..coarse_depth kept in RAM
    n_addr_nets: int = 3              # must match ModelConfig.n_addr_nets
    d_model: int = 128                # vector dimensionality
    n_bins: int = 256                 # bins per trie level
    page_size: int = 16384            # OS page size (16KB on M4)
    flush_interval: int = 1000        # flush dirty pages every N writes

