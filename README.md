# ANT — Addressable Neural Transformer

An 828K parameter looping transformer with persistent external memory and cross-attention memory retrieval. Pure 256-byte vocabulary — token ID equals raw byte value.

**ANT** stands for **Addressable Neural Transformer** — the defining trait is that memory slots are written and retrieved via learned 3-head × 8-dim addresses, enabling content-addressed persistent storage separate from the context window.

## Quick Start

```bash
pip install -r requirements.txt

# QA-only training (bAbI memory recall, ~10 min)
python train_micro.py --chunk_size 16

# Multi-task training: LM (shell+wiki) + QA (bAbI) (~25 min)
python train_micro.py --chunk_size 16 --multitask

# Interactive chat
python chat.py
```

## Architecture

```
828K params │ 4 layers │ 128 d_model │ 4 heads │ 256 vocab (raw bytes)
```

Each transformer block: **Self-Attention → Memory Cross-Attention → FFN**

The model reads from a persistent trie-indexed external memory via learned address heads. A sliding window encoder processes input in overlapping chunks, writing compressed representations to memory for later retrieval.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture diagram and component details.

## Results

| Task | Accuracy | Notes |
|------|----------|-------|
| bAbI 1-fact QA | 98.9% | Memory recall with sliding window |
| bAbI 2-fact QA | 100% | Multi-hop entity→attribute |
| bAbI 3-fact QA | 100% | Three-step reasoning chain |
| **Overall QA** | **99.5%** | With simultaneous LM training |
| Shell+Wiki LM | loss 2.49 | Still improving at early stop |

## Files

```
train_micro.py    Self-contained training pipeline (~3000 lines)
                  Tokenizer, datasets, encoders, training loops, evaluation
config.py         ModelConfig (828K) + MemoryConfig
model.py          ANT, TransformerBlock, Attention, MemoryAttention
model_mlx.py      ANTMLX — Apple Silicon optimized inference
memory.py         MemorySystem — trie-indexed persistent int8 vector store
chat.py           Interactive terminal chat CLI
benchmark.py      Inference + training performance benchmarks
train_colab.ipynb A100 GPU training notebook (Hugging Face Hub integration)
```

## Key Design Decisions

- **Pure byte vocabulary**: Token ID = raw byte value. No BPE, no subword tokenizer. Special tokens use ASCII control characters (NUL=PAD, STX=BOS, ETX=EOS, etc.)
- **Cross-attention memory**: Each block has a dedicated memory cross-attention layer (not prepended to sequence). Enables clean separation of context window and memory.
- **Sliding window encoding**: Input processed in overlapping chunks. Each chunk produces memory vectors (mean + last hidden state) tagged with temporal embeddings.
- **Trie-indexed storage**: 3 address heads × 8 dimensions. Addresses learned end-to-end. Neighbor search (±1) enables implicit concept clustering.
- **Multi-task training**: Causal LM on shell commands + Wikipedia markdown, interleaved with sliding-window QA on bAbI. Different forward paths for speed (1.2 it/s on MPS).

## Training Curriculum

1. **Phase A** — Warmup: frozen encoder, context-only QA (500 steps)
2. **Phase D1** — No-context QA: forces memory reading from step 1 (30% of steps)
3. **Phase D2** — Memory-only QA: passages only in memory, not context (70% of steps)
4. **Multi-task** — Interleave LM batches with QA batches

## License

Research prototype. Not yet licensed for production use.