# Architecture — ANT: Addressable Neural Transformer (828K params)

## Overview

A looping transformer with persistent external memory accessed via cross-attention.
The model operates on raw bytes (256 vocab), uses a sliding window to encode passages
into memory, and retrieves stored information through learned address heads.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ANT — Addressable Neural Transformer (828,306 params)       │
│                                                                             │
│  Vocabulary: 256 (raw bytes, no tokenizer)                                  │
│  Embedding:  256 × 128 = 32,768 params                                     │
│  Layers:     4 × TransformerBlock = 786,960 params                          │
│  Heads:      Halt (258) + 3×Address (3,072) + Temporal (4,096)              │
│  Positions:  RoPE (θ=10000, up to 203 positions)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Full Data Flow

```
  Input: raw UTF-8 bytes
  ┌─────────────────┐
  │ "John is in the │    tokenize() = identity
  │  kitchen"       │    [74, 111, 104, 110, ...]
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐     ┌──────────────────────────────────┐
  │   Byte Embedding │     │  Sliding Window Encoder           │
  │   256 × 128      │     │  (encodes passage into memory)    │
  │   (32,768 params) │     │                                   │
  └────────┬────────┘     │  chunk_size=8, stride=8            │
           │              │  Each chunk → 2 memory vectors:    │
           ▼              │    • mean(hidden_states)            │
  ┌─────────────────────┐ │    • last hidden state             │
  │                     │ │  + temporal embedding (position)   │
  │  Transformer Stack  │◄┤                                   │
  │  (4 layers)         │ │  Vectors quantized to int8,       │
  │                     │ │  stored in TrieIndex by address    │
  │  ┌───────────────┐  │ └──────────────────────────────────┘
  │  │ Layer 0       │  │
  │  │  ┌──────────┐ │  │     ┌──────────────────────────────┐
  │  │  │ RMSNorm  │ │  │     │                              │
  │  │  │ Self-Attn│ │  │     │  Persistent External Memory  │
  │  │  │ (causal) │ │  │     │  ┌────────────────────────┐  │
  │  │  └──────────┘ │  │     │  │ TrieIndex              │  │
  │  │  ┌──────────┐ │  │     │  │  3 addr heads × 8 dims │  │
  │  │  │ RMSNorm  │ │  │     │  │  ±1 neighbor search    │  │
  │  │  │ Mem Cross│◄┼──┼─────┤  │  int8 quantized vecs   │  │
  │  │  │ Attention│ │  │     │  │  EMA write blending     │  │
  │  │  └──────────┘ │  │     │  └────────────────────────┘  │
  │  │  ┌──────────┐ │  │     │                              │
  │  │  │ RMSNorm  │ │  │     │  Read: addr_heads(hidden)    │
  │  │  │ FFN(SiLU)│ │  │     │    → 3 addresses → lookup    │
  │  │  └──────────┘ │  │     │    → 9 memory vectors        │
  │  └───────────────┘  │     │    → cross-attend             │
  │  ┌───────────────┐  │     │                              │
  │  │ Layer 1       │  │     │  Write: encoder hidden        │
  │  │  (same arch)  │  │     │    → quantize to int8         │
  │  └───────────────┘  │     │    → EMA blend at address     │
  │  ┌───────────────┐  │     └──────────────────────────────┘
  │  │ Layer 2       │  │
  │  │  (same arch)  │  │
  │  └───────────────┘  │
  │  ┌───────────────┐  │
  │  │ Layer 3       │  │
  │  │  (same arch)  │  │
  │  └───────────────┘  │
  │                     │
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────┐     ┌──────────────────┐
  │ Final RMSNorm   │     │ Halt Head        │
  └────────┬────────┘     │ 128→2 (cont/halt)│
           │              └──────────────────┘
           ▼
  ┌─────────────────┐     ┌──────────────────────────────────┐
  │ LM Head         │     │ 3 × Address Heads                │
  │ (embed.weight^T)│     │ each: Linear(128→8, no bias)     │
  │ tied weights    │     │ output: int8 address for trie     │
  └────────┬────────┘     └──────────────────────────────────┘
           │
           ▼
     logits [B, T, 256]
```

## Transformer Block Detail

```
  ┌──────────────────────────────────────────────────────┐
  │  TransformerBlock (196,740 params each × 4 layers)    │
  │                                                       │
  │  Input: x [B, T, 128]                                │
  │    │                                                  │
  │    ├──► RMSNorm ──► Self-Attention ──► + residual     │
  │    │    (128)        Q,K,V,O: 128×128                 │
  │    │                 4 heads × 32 dim                  │
  │    │                 RoPE positions                    │
  │    │                 Causal mask                       │
  │    │                                                  │
  │    ├──► RMSNorm ──► Memory Cross-Attention ──► + res  │
  │    │    (128)        Q,K,V,O: 128×128                 │
  │    │                 4 heads × 32 dim                  │
  │    │                 Learned inv_temp per head         │
  │    │                 Keys/Values from memory vectors   │
  │    │                 No causal mask (full attention)   │
  │    │                                                  │
  │    └──► RMSNorm ──► SiLU FFN ──► + residual           │
  │         (128)        up:  128→256 (gate + value)      │
  │                      SiLU activation on gate           │
  │                      down: 256→128                     │
  └──────────────────────────────────────────────────────┘
```

## Sliding Window Encoding

The encoder processes passages through a sliding window, compressing each chunk
into memory vectors that the decoder later retrieves via cross-attention.

```
  Passage: "John went to the kitchen. Mary went to the garden."

  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
  │ chunk 0 │ chunk 1 │ chunk 2 │ chunk 3 │ chunk 4 │ chunk 5 │
  │ 8 bytes │ 8 bytes │ 8 bytes │ 8 bytes │ 8 bytes │ 8 bytes │
  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘
       │         │         │         │         │         │
       ▼         ▼         ▼         ▼         ▼         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │           Causal Transformer Forward Pass                    │
  │           (same model weights, sliding window)               │
  └────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┘
       │    │    │    │    │    │    │    │    │    │    │    │
       ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
  │mean_0││last_0││mean_1││last_1││mean_2││last_2│ ...
  │ +t_0 ││ +t_0 ││ +t_1 ││ +t_1 ││ +t_2 ││ +t_2 │
  └──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘
     │       │       │       │       │       │
     ▼       ▼       ▼       ▼       ▼       ▼
  ┌─────────────────────────────────────────────────┐
  │  Memory Vector Bank (float → int8 quantized)     │
  │  Indexed by address heads for later retrieval     │
  │  EMA blending: v_new = α·v_write + (1-α)·v_old  │
  └─────────────────────────────────────────────────┘
```

## Memory System

```
  ┌──────────────────────────────────────────────────────┐
  │  TrieIndex — Persistent External Memory               │
  │                                                       │
  │  Address Space: 3 heads × 8 dims × int8 = 24 bytes   │
  │  Vector Size:   128 dims × int8 (quantized from f32)  │
  │                                                       │
  │  WRITE:                                               │
  │    hidden_state ──► addr_head(h) ──► 8-byte address   │
  │    hidden_state ──► quantize(h * 127) ──► int8 vec    │
  │    trie[addr] = EMA_blend(old_vec, new_vec)           │
  │                                                       │
  │  READ:                                                │
  │    query_hidden ──► addr_head(h) ──► 3 addresses      │
  │    for each address:                                  │
  │      exact_match = trie.get(addr)                     │
  │      neighbors   = trie.get(addr ± 1)  (±1 per dim)  │
  │    collect up to 9 vectors (n_mem_slots)              │
  │    return as [1, 9, 128] tensor for cross-attention   │
  │                                                       │
  │  Properties:                                          │
  │    • Persistent across training steps                  │
  │    • Content-addressed (not position-addressed)        │
  │    • Neighbor search enables concept clustering        │
  │    • EMA writes prevent catastrophic overwriting       │
  │    • Decoupled from model size (can grow indefinitely) │
  └──────────────────────────────────────────────────────┘
```

## Special Tokens (ASCII Control Characters)

```
  Hex   ASCII   Name          Role
  ────  ─────   ────────────  ──────────────────────────
  0x00  NUL     Null          PAD token
  0x01  SOH     Start of Hdr  MEM_START (memory block open)
  0x02  STX     Start of Text BOS (beginning of sequence)
  0x03  ETX     End of Text   EOS (end of sequence)
  0x04  EOT     End of Xmit   MEM_END (memory block close)
  0x05  ENQ     Enquiry       ANS (answer marker in QA)
  0x06  ACK     Acknowledge   NOOP (no output token)
  0x1A  SUB     Substitute    UNK (unknown/fallback)

  All other byte values (0x07-0x19, 0x1B-0xFF) = printable/UTF-8 data
```

## Parameter Breakdown

```
  Component                    Parameters    % of Total
  ─────────────────────────    ──────────    ──────────
  Byte Embedding (256×128)       32,768        4.0%
  Layer 0 (Self+Mem+FFN)        196,740       23.8%
  Layer 1 (Self+Mem+FFN)        196,740       23.8%
  Layer 2 (Self+Mem+FFN)        196,740       23.8%
  Layer 3 (Self+Mem+FFN)        196,740       23.8%
  Final RMSNorm                      128        0.0%
  Halt Head (128→2)                  258        0.0%
  Address Heads (3×128→8)          3,072        0.4%
  Temporal Embedding (32×128)      4,096        0.5%
  LM Head                    (tied with embed)
  ─────────────────────────    ──────────    ──────────
  TOTAL                          828,306      100.0%

  Per-layer breakdown:
    Self-Attention (Q,K,V,O)    65,536  (4 × 128×128)
    Memory Cross-Attn (Q,K,V,O) 65,536  (4 × 128×128)
    Memory inv_temp                  4  (learned per head)
    FFN (up + down)             65,536  (128×256 + 256×128)
    RMSNorm × 3                    384  (3 × 128)
    Subtotal per layer:        196,740
```

## Training Modes

### QA-Only (Sliding Window + Memory)

```
  Passage ──► Sliding Window Encode ──► Memory Write
                                            │
  Question ──► Causal Forward + MemCrossAttn ──► Answer
                       ▲                              │
                       └──── Memory Read (9 vecs) ◄───┘

  Curriculum:
    Phase A  (500 steps):  warmup, frozen encoder, passage in context
    Phase D1 (30% steps):  no context, frozen encoder → forces memory use
    Phase D2 (70% steps):  no context, differentiable encoder → end-to-end
```

### Multi-Task (LM + QA)

```
  ┌─── LM Batch ───────────────────────────────────────┐
  │  Shell commands + Wikipedia text                     │
  │  Standard causal forward pass (no sliding window)    │
  │  Loss: cross-entropy on next byte prediction         │
  └─────────────────────────────────────────────────────┘
        ↕  alternating batches
  ┌─── QA Batch ───────────────────────────────────────┐
  │  bAbI memory-recall tasks (1/2/3 supporting facts)  │
  │  Sliding window encode → memory → cross-attention    │
  │  Loss: cross-entropy on answer tokens only           │
  └─────────────────────────────────────────────────────┘

  Combined loss = lm_weight × LM_loss + qa_weight × QA_loss
```

## Configuration

```python
ModelConfig(
    vocab_size   = 256,       # raw bytes
    d_model      = 128,       # hidden dimension
    n_heads      = 4,         # attention heads
    head_dim     = 32,        # per-head dimension
    ffn_dim      = 256,       # FFN intermediate (2× expansion)
    n_layers     = 4,         # transformer blocks
    max_seq_len  = 192,       # context window
    n_mem_slots  = 9,         # memory vectors per read
    n_addr_heads = 3,         # parallel address probes
    addr_dim     = 8,         # address dimensionality
    chunk_size   = 8,         # sliding window chunk size
    slots_per_chunk = 2,      # memory entries per chunk
    max_temporal_chunks = 32, # temporal embedding capacity
)
```
