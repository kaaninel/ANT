# ANT — Architecture (828,306 params)

## Overview

A byte-level transformer with persistent external memory accessed via cross-attention.
The model operates on raw bytes (256 vocab), uses a causal sliding window for unlimited
context, and retrieves stored information through learned address heads. User messages
are encoded into memory for chat, solving the receptive field limitation of tiny windows.

```
+-----------------------------------------------------------------------------+
|                          ANT (828,306 params)                               |
|                                                                             |
|  Vocabulary: 256 (raw bytes, no tokenizer)                                  |
|  Embedding:  256 x 128 = 32,768 params                                     |
|  Layers:     4 x TransformerBlock = 787,984 params                          |
|  Heads:      Halt (258) + 3 x Address (3,330) + Temporal (4,096)            |
|  Positions:  RoPE (theta=10000, up to 203 positions)                        |
|  Memory:     9 slots x 128-dim via cross-attention (no distance limit)      |
+-----------------------------------------------------------------------------+
```

## Full Data Flow

```
  Input: raw UTF-8 bytes
  +------------------+
  | "John is in the  |    tokenize() = identity
  |  kitchen"        |    [74, 111, 104, 110, ...]
  +--------+---------+
           |
           v
  +------------------+     +-----------------------------------+
  |  Byte Embedding  |     |  Sliding Window Encoder            |
  |  256 x 128       |     |  (encodes passage into memory)     |
  |  (32,768 params) |     |                                    |
  +--------+---------+     |  chunk_size=8, stride=1, passes=4  |
           |               |  Each sentence --> memory vectors:  |
           v               |    * last hidden state per sentence |
  +----------------------+ |    + temporal embedding (position)  |
  |                      | |                                    |
  |  Transformer Stack   |<+  Vectors stored as cross-attention  |
  |  (4 layers)          | |  keys/values for later retrieval   |
  |                      | +-----------------------------------+
  |  +----------------+  |
  |  | Layer 0        |  |
  |  |  +-----------+ |  |     +-------------------------------+
  |  |  | RMSNorm   | |  |     |                               |
  |  |  | Self-Attn | |  |     |  Persistent External Memory   |
  |  |  | (causal)  | |  |     |  +-------------------------+  |
  |  |  +-----------+ |  |     |  | 9 memory slots          |  |
  |  |  +-----------+ |  |     |  | 128-dim vectors (float)  |  |
  |  |  | RMSNorm   | |  |     |  | Indexed by 3 addr heads  |  |
  |  |  | Mem Cross |<+--+-----+  | No distance limitation   |  |
  |  |  | Attention | |  |     |  +-------------------------+  |
  |  |  +-----------+ |  |     |                               |
  |  |  +-----------+ |  |     |  Read: addr_heads(hidden)     |
  |  |  | RMSNorm   | |  |     |    --> 3 addresses --> lookup  |
  |  |  | FFN(SiLU) | |  |     |    --> 9 memory vectors        |
  |  |  +-----------+ |  |     |    --> cross-attend             |
  |  +----------------+  |     |                               |
  |  +----------------+  |     |  Write: encoder hidden         |
  |  | Layer 1        |  |     |    --> temporal embedding       |
  |  |  (same arch)   |  |     |    --> memory key/value pair    |
  |  +----------------+  |     +-------------------------------+
  |  +----------------+  |
  |  | Layer 2        |  |
  |  |  (same arch)   |  |
  |  +----------------+  |
  |  +----------------+  |
  |  | Layer 3        |  |
  |  |  (same arch)   |  |
  |  +----------------+  |
  |                      |
  +--------+-------------+
           |
           v
  +------------------+     +-------------------+
  | Final RMSNorm    |     | Halt Head         |
  +--------+---------+     | 128->2 (cont/halt)|
           |               +-------------------+
           v
  +------------------+     +-----------------------------------+
  | LM Head          |     | 3 x Address Heads                 |
  | (embed.weight^T) |     | each: Linear(128->8, no bias)     |
  | tied weights     |     | output: 8-dim address for lookup   |
  +--------+---------+     +-----------------------------------+
           |
           v
     logits [B, T, 256]
```

## Transformer Block Detail

```
  +-------------------------------------------------------+
  |  TransformerBlock (196,996 params each x 4 layers)     |
  |                                                        |
  |  Input: x [B, T, 128]                                 |
  |    |                                                   |
  |    +---> RMSNorm --> Self-Attention --> + residual      |
  |    |    (128)        Q,K,V,O: 128x128                  |
  |    |                 4 heads x 32 dim                   |
  |    |                 RoPE positions                     |
  |    |                 Causal mask                        |
  |    |                                                   |
  |    +---> RMSNorm --> Memory Cross-Attention --> + res   |
  |    |    (128)        Q,K,V,O: 128x128                  |
  |    |                 4 heads x 32 dim                   |
  |    |                 Learned inv_temp per head          |
  |    |                 Keys/Values from memory vectors    |
  |    |                 No causal mask (full attention)    |
  |    |                                                   |
  |    +---> RMSNorm --> SiLU FFN --> + residual            |
  |         (128)        up:  128->256 (gate + value)      |
  |                      SiLU activation on gate            |
  |                      down: 256->128                     |
  +-------------------------------------------------------+
```

## Sliding Window Encoding

The encoder processes input through a causal sliding window with multiple passes.
Each pass sees the full sequence but offset, creating an expanding receptive field.

```
  With W=8 bytes, stride=1:

  Pass 1:  [........]                      --> 5 bytes visible per token
  Pass 2:     [........]                   --> 9 bytes visible per token
  Pass 3:        [........]                --> 13 bytes visible per token
  Pass 4:           [........]             --> 17 bytes visible per token

  For memory encoding (sentences):
    "John went to the kitchen." --> model() forward --> last hidden state
    Each sentence = 1 memory slot (key + value, 128-dim each)
    + temporal embedding for position within passage
```

## Memory-Based Chat Architecture

The key architectural insight: with W=8 and passes=4, each token sees only 17 bytes
behind it. But agent tags are 45 bytes — so the model literally cannot see the user's
question through the sliding window. Memory cross-attention has NO distance limit.

```
  TRAINING (Phase 3):

    User: "How do I sort a list?"
         |
         v
    Frozen Encoder (model forward, no gradients)
         |
         v
    Memory: [key_0, val_0], [key_1, val_1], ...  (up to 9 slots)
         |
         +----------------------------------------+
                                                  |
    Agent: "localhost/ant/chat@ts: You can use..." |
         |                                        |
         v                                        v
    Sliding Window Encode (W=8, 4 passes) + Cross-Attention
         |
         v
    Chat Loss (CE on agent response tokens)


  INFERENCE:

    Last 6 user messages --> join with ". " --> truncate to 190 tokens
         |
         v
    Frozen Encoder --> chat memory slots
         |
    /mem add facts --> Frozen Encoder --> passage memory slots
         |
         +-- merge (concat along slot dim) --+
                                             |
    Prompt: [BOS] + agent_tag                |
         |                                   v
    Sliding Generate + Cross-Attention to merged memory
         |
         v
    Unlimited streaming output (window slides forward)
```

## Memory System

```
  +-------------------------------------------------------+
  |  Cross-Attention Memory (Training)                     |
  |                                                        |
  |  Encoder produces per-sentence hidden states:          |
  |    sentence --> model() --> hidden[-1] + temporal_emb  |
  |    = memory key (128-dim) + memory value (128-dim)     |
  |                                                        |
  |  At each transformer layer, cross-attention reads:     |
  |    Q = self-attn output (from current token)           |
  |    K = memory keys [1, N_slots, 128]                   |
  |    V = memory values [1, N_slots, 128]                 |
  |    Attention: softmax(Q @ K^T * inv_temp) @ V          |
  |                                                        |
  |  Properties:                                           |
  |    * No distance limit (every token sees all slots)    |
  |    * Learned inverse temperature per head               |
  |    * Up to 9 slots (configurable)                       |
  |    * Frozen encoder during chat training                |
  |    * Differentiable encoder during QA training          |
  +-------------------------------------------------------+
```

```
  +-------------------------------------------------------+
  |  TrieIndex Memory (Persistent / Inference)             |
  |                                                        |
  |  Address Space: 3 heads x 8 dims x int8 = 24 bytes    |
  |  Vector Size:   128 dims x int8 (quantized from f32)   |
  |                                                        |
  |  WRITE:                                                |
  |    hidden_state --> addr_head(h) --> 8-byte address     |
  |    hidden_state --> quantize(h * 127) --> int8 vec      |
  |    trie[addr] = EMA_blend(old_vec, new_vec)             |
  |                                                        |
  |  READ:                                                 |
  |    query_hidden --> addr_head(h) --> 3 addresses        |
  |    for each address:                                   |
  |      exact_match = trie.get(addr)                      |
  |      neighbors   = trie.get(addr +/- 1) per dim        |
  |    collect up to 9 vectors (n_mem_slots)               |
  |    return as [1, 9, 128] tensor for cross-attention    |
  |                                                        |
  |  Properties:                                           |
  |    * Persistent across sessions                         |
  |    * Content-addressed (not position-addressed)         |
  |    * Neighbor search enables concept clustering         |
  |    * EMA writes prevent catastrophic overwriting        |
  |    * Decoupled from model size (can grow indefinitely)  |
  +-------------------------------------------------------+
```

## Special Tokens (ASCII Control Characters)

```
  Hex   ASCII   Name          Role
  ----  -----   ------------  --------------------------
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
  -------------------------    ----------    ----------
  Byte Embedding (256x128)       32,768        4.0%
  Layer 0 (Self+Mem+FFN)        196,996       23.8%
  Layer 1 (Self+Mem+FFN)        196,996       23.8%
  Layer 2 (Self+Mem+FFN)        196,996       23.8%
  Layer 3 (Self+Mem+FFN)        196,996       23.8%
  Final RMSNorm                      128        0.0%
  Halt Head (128->2)                 258        0.0%
  Address Heads (3x128->8)        3,072        0.4%
  Temporal Embedding (32x128)      4,096        0.5%
  LM Head                    (tied with embed)
  -------------------------    ----------    ----------
  TOTAL                          828,306      100.0%

  Per-layer breakdown:
    Self-Attention (Q,K,V,O)    65,536  (4 x 128x128)
    Memory Cross-Attn (Q,K,V,O) 65,536  (4 x 128x128)
    Memory inv_temp                  4  (learned per head)
    FFN (up + down)             65,536  (128x256 + 256x128)
    RMSNorm x 3                    384  (3 x 128)
    Subtotal per layer:        196,996
```

## Training Phases

### Phase 1 — Language Model (Sliding Window)

```
  Wiki + Shell text --> Sliding Window Encode (W=8, 4 passes) --> Causal LM loss
```

### Phase 2 — LM + QA (Memory Recall)

```
  +--- LM Batch -------------------------------------------+
  |  Shell commands + Wikipedia text                        |
  |  Sliding window causal forward pass                     |
  |  Loss: cross-entropy on next byte prediction            |
  +--------------------------------------------------------+
        |  alternating batches
  +--- QA Batch -------------------------------------------+
  |  bAbI memory-recall tasks (1/2/3 supporting facts)     |
  |  Sliding window encode --> memory --> cross-attention   |
  |  Loss: cross-entropy on answer tokens only              |
  +--------------------------------------------------------+

  Curriculum:
    Phase D1 (30%):  no context, frozen encoder --> forces memory reading
    Phase D2 (70%):  no context, differentiable --> end-to-end gradients
```

### Phase 3 — Full (LM + QA + Memory-Based Chat)

```
  +--- Chat Batch -----------------------------------------+
  |                                                        |
  |  User message --> Frozen Encoder --> Memory Slots       |
  |                                        |               |
  |  Agent tag --> Sliding Window + CrossAttn --> Loss      |
  |                       ^                                |
  |                       +-- 9 memory vectors <-----------+
  |                                                        |
  |  Key insight: user question is in memory, not in       |
  |  the sliding window. Cross-attention has no distance   |
  |  limit, so every generated token can "see" the         |
  |  question regardless of window size.                   |
  +--------------------------------------------------------+

  Combined: LM loss + QA loss + Chat loss (weighted)
```

## Spatiotemporal Tags

All data is tagged with a URI-style spatiotemporal header:

```
  host/agent/dataplane@ISO-timestamp: content

  Examples:
    localhost/user/chat@2026-04-08T12:00:00Z: Hello, how are you?
    localhost/ant/chat@2026-04-08T12:00:00Z: I'm doing well!
    shell/root@2026-04-01T14:22:33Z: ls -la /home/user
    wiki/article@2025-06-15T08:30:00Z: The French Revolution began in 1789.
```

The model reads and writes this format natively. Tags enable:
- Multi-agent conversations (user, ant, system)
- Dataplane routing (chat, shell, wiki, qa)
- Temporal ordering without positional encoding

## Configuration

```python
ModelConfig(
    vocab_size   = 256,       # raw bytes
    d_model      = 128,       # hidden dimension
    n_heads      = 4,         # attention heads
    head_dim     = 32,        # per-head dimension
    ffn_dim      = 256,       # FFN intermediate (2x expansion)
    n_layers     = 4,         # transformer blocks
    max_seq_len  = 192,       # context window per chunk
    n_mem_slots  = 9,         # memory vectors per read
    n_addr_heads = 3,         # parallel address probes
    addr_dim     = 8,         # address dimensionality
    chunk_size   = 8,         # sliding window chunk size
    slots_per_chunk = 2,      # memory entries per chunk
    max_temporal_chunks = 32, # temporal embedding capacity
)
```
