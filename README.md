# 🐜 ANT

> **828,306 parameters. Persistent memory. Unlimited context. Pure bytes.**

ANT is an experimental language model that fits under 1 million parameters while supporting unlimited input/output through a causal sliding window and persistent external memory accessed via cross-attention. Every byte is a token — no tokenizer, no vocabulary limits, no subword artifacts.

```
  ┌─────────────────────────────────────────────────────┐
  │                    🐜 ANT                            │
  │                                                     │
  │    828K params · 4 layers · 128-dim · byte-level    │
  │                                                     │
  │    Input ──► Sliding Window ──► Memory Write         │
  │                                    │                 │
  │    Query ──► Cross-Attention ◄─────┘                 │
  │                  │                                   │
  │                  ▼                                   │
  │              Response (unlimited length)              │
  └─────────────────────────────────────────────────────┘
```

## Why ANT?

Most small language models are crippled by fixed context windows — they forget everything outside their last ~200 tokens. ANT solves this with **persistent external memory**: information is encoded into memory slots via a sliding window, then retrieved via cross-attention during generation. The model's context window is tiny (8 bytes), but memory cross-attention has **no distance limit**.

| Feature | ANT | Typical Small LM |
|---------|-----|-------------------|
| Parameters | 828K | 1M–10M |
| Context | Unlimited (sliding window + memory) | Fixed (512–2048 tokens) |
| Tokenizer | None (raw bytes) | BPE/SentencePiece |
| Memory | Persistent cross-attention | None |
| QA Accuracy | 100% (bAbI 1/2/3-fact) | N/A |

## Quick Start

```bash
pip install -r requirements.txt

# Train (overnight, M4/MPS)
PYTHONUNBUFFERED=1 python3 train_overnight.py 2>&1 | tee training.log

# Train (A100 GPU, Colab)
# Open train_colab.ipynb

# Interactive chat
python3 chat.py --checkpoint checkpoints/overnight/checkpoint_best.pt
```

## Architecture

```
828K params │ 4 layers │ d_model=128 │ 4 heads │ 256 vocab (raw bytes)
```

Each transformer block:

```
  Input
    │
    ├──► RMSNorm → Self-Attention (causal, RoPE) → + residual
    │
    ├──► RMSNorm → Memory Cross-Attention ──────── + residual
    │                  ▲
    │                  │ 9 memory vectors (persistent)
    │
    └──► RMSNorm → SiLU FFN (128→256→128) ──────── + residual
```

### Memory-Based Chat

User messages are encoded into memory via a frozen sliding window encoder, then the model generates responses with cross-attention to those memory slots. This means:

- **User question is always visible** — regardless of tag overhead or window size
- **Multiple turns accumulate** — recent conversation history lives in memory
- **Facts from `/mem add` merge** — passage memory and chat memory combine

### Sliding Window

The causal sliding window processes input in overlapping 8-byte chunks across multiple passes. Each pass sees the full sequence but offset, creating an expanding receptive field:

```
  passes=1:   5 bytes visible per token
  passes=4:  17 bytes visible per token
  passes=8:  33 bytes visible per token
```

For generation, the window slides indefinitely — **no output length limit**.

### Persistent Memory

```
  Encode: passage → sliding window → per-sentence hidden states → memory slots
  Read:   3 address heads × 8 dims → trie lookup → 9 vectors → cross-attend
  Write:  EMA blending (won't overwrite, merges with existing)
```

Memory is separate from model weights. It can grow without increasing parameter count.

## Results

| Task | Accuracy | Notes |
|------|----------|-------|
| bAbI 1-fact QA | 100% | Single supporting fact recall |
| bAbI 2-fact QA | 100% | Multi-hop entity→attribute |
| bAbI 3-fact QA | 100% | Three-step reasoning chain |
| Chat | Training in progress | Memory-based architecture |
| LM (wiki+shell) | loss ~1.5 | Byte-level next-token prediction |

## Training

ANT uses a 3-phase curriculum:

1. **Phase 1 — Language Model**: Causal LM on Wikipedia + shell commands (sliding window)
2. **Phase 2 — LM + QA**: Add bAbI memory-recall tasks with cross-attention
3. **Phase 3 — Full**: Add memory-based chat (user→memory, agent→sliding generation)

The training pipeline is self-contained in `train_micro.py` (~4000 lines). Data sources:

- **Wikipedia**: Sentence-level text for language modeling
- **Shell**: Synthetic command patterns
- **QA**: bAbI 1/2/3-fact memory tasks (synthetic)
- **Chat**: 30K pairs from HuggingFace (UltraChat, SmolTalk)

## Files

```
train_micro.py      Self-contained training pipeline (~4000 lines)
                    Tokenizer, datasets, encoders, training loops, evaluation
train_overnight.py  M4 MPS overnight training script (3-phase)
train_colab.ipynb   A100 GPU training notebook (HuggingFace Hub integration)
config.py           ModelConfig (828K) + MemoryConfig
model.py            ANT transformer with cross-attention memory
model_mlx.py        Apple Silicon optimized inference (MLX)
memory.py           TrieIndex — persistent int8 vector store
chat.py             Interactive terminal chat CLI
benchmark.py        Inference + training performance benchmarks
```

## Key Design Decisions

- **Pure byte vocabulary** — Token ID = raw byte value. No BPE, no subword tokenizer. Special tokens use ASCII control characters (NUL=PAD, STX=BOS, ETX=EOS)
- **Cross-attention memory** — Dedicated memory attention layer per block, separate from self-attention. No distance limit on memory retrieval
- **Memory-based chat** — User messages encoded into memory via frozen encoder. Solves the receptive field problem (tiny sliding window can't see past tags to the question)
- **Sliding window generation** — Output streams indefinitely. No fixed output cap
- **Weight-tied LM head** — Embedding matrix reused for output projection. Saves 32K params
- **Spatiotemporal tags** — All data tagged as `host/agent/dataplane@timestamp: content`. Model learns to read/write this format natively

## Chat Interface

```
$ python3 chat.py --checkpoint checkpoints/overnight/checkpoint_best.pt

  🐜 ANT Terminal Canvas
  ──────────────────────────────────────────
  Model: 0.83M params | Memory-based chat
  Type /help for commands, !cmd for shell

kaaninel@mac> Hello
ant@mac/chat Hello! How can I help you?

kaaninel@mac> /mem add The capital of France is Paris
  ✓ Encoded 1 sentence into memory

kaaninel@mac> What is the capital of France?
ant@mac/chat The capital of France is Paris.
```

## License

Research prototype. Not yet licensed for production use.