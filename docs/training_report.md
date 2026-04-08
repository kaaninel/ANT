# ANT — Training Report

## Architecture State

ANT is a 937K parameter byte-level transformer where ALL knowledge lives in
a persistent hierarchical trie. The model weights store zero knowledge — they
learn only how to read and write the trie.

### What's Verified
- **Model architecture**: 937K params, AddrNet × 3, V_proj, tag system, MemoryAttention
- **Forward pass**: works with and without memory vectors
- **Memory system**: HierarchicalTrie with EMA, binary serialization, 34μs write, 7μs read
- **Trie bridge**: trie_write, trie_read, trie_write_all_tokens tested end-to-end
- **Sliding window**: multi-pass encode with optional memory cross-attention
- **Inference**: terminal chat with per-token trie read/write

### Codebase (clean rewrite)
- **data.py** (699 lines) — all data pipelines, tokenizer, generators, datasets
- **train.py** (1035 lines) — Phase A/B/C curriculum with trie integration
- **inference.py** (298 lines) — terminal chat with per-token trie read/write
- Old files (train_micro.py, train_overnight.py, chat.py) deleted

## Training Data

| Source | Count | Description |
|--------|-------|-------------|
| Wikipedia | 5,000 sentences | Filtered 30–300 chars, from HuggingFace |
| Shell | 1,000 commands | Synthetic template grammar (8 pattern types) |
| QA (bAbI) | 5,000 examples | 1/2/3-fact location tracking |
| Chat | 30,000 pairs | UltraChat + SmolTalk from HuggingFace |

All text tagged with spatiotemporal provenance:
```
shell/root@2026-04-01T14:22:33Z: ls -la /home
wiki/article@2025-06-15T08:30:00Z: The French Revolution began in 1789.
```

## Training Curriculum

### Phase A — Base Language Model
- Wiki + shell → sliding window → causal LM loss
- Memory OFF — model learns language patterns and byte embeddings
- Establishes baseline LM loss

### Phase B — Memory Training
- Freeze all base weights (embed, layers, norm, halt_head)
- Train only: 3 × AddrNet, V_proj, tag system
- Losses:
  - Contrastive address loss (similar content → nearby addresses)
  - Quadratic depth cost (shallow addresses for common concepts)
  - Retrieval accuracy (write then read back)
- Purpose: learn a stable address space before unfreezing

### Phase C — End-to-End
- Unfreeze base model
- Keep frozen: AddrNet, V_proj (prevents address space drift)
- Every forward pass: trie READ + WRITE
- LM + QA + Chat losses through memory

## Known Limitations

1. **LM output is incoherent without memory.** 937K params with byte-level
   tokenization cannot produce fluent English from weights alone. This is
   by design — the model's strength is memory recall.

2. **Old checkpoints incompatible.** Previous checkpoints used different model
   architecture. Training must restart from scratch with new code.

## Performance Benchmarks

```
  Memory:   34μs/write, 7μs/read, 3ms load (9K nodes)
  Model:    ~0.7 it/s on M4 MPS (batch=8)
  Storage:  ~500 bytes/node, 4.7MB for 1000 writes
```

## Next Steps

1. Run Phase A training to establish baseline LM loss
2. Run Phase B training for stable address space
3. Run Phase C end-to-end training with trie
4. Validate on bAbI with trie architecture (target: 100% QA)
5. Update train_colab.ipynb for Colab training
