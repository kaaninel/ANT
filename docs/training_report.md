# ANT — Training Report

## Architecture State

ANT is a 937K parameter byte-level transformer where ALL knowledge lives in
a persistent hierarchical trie. The model weights store zero knowledge — they
learn only how to read and write the trie.

### What's Verified
- **Model architecture**: 937K params, AddrNet × 3, V_proj, tag system, MemoryAttention
- **Forward pass**: works with and without memory vectors
- **Memory system**: HierarchicalTrie with EMA, binary serialization, 34μs write, 7μs read
- **QA accuracy**: 100% on bAbI 1/2/3-fact (achieved with old architecture, before trie rewrite)

### What Needs Work
- **Training pipeline**: Memory is NOT wired into the training loop. AddrNets and
  V_proj exist but are never called during training. Training runs without any
  trie reads or writes.
- **train_micro.py**: ~2955 lines of dead code from old architecture iterations.
  Memory API calls reference methods that no longer exist.
- **chat.py**: References removed temporal_emb. Needs full rewrite for duplex streaming.
- **Checkpoints**: Old checkpoints incompatible with new model (state dict mismatch).

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

## Training Curriculum (Target)

### Phase A — Base Language Model
- Wiki + shell → sliding window → causal LM loss
- Memory OFF — model learns language patterns and byte embeddings
- Establishes baseline LM loss

### Phase B — Memory Training
- Freeze all base weights
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

## Known Issues

1. **Memory integration is broken in training code.** The MemorySystem constructor
   signature changed but train_micro.py was never updated. Calls to
   `.read_memory_batch()`, `.write_memory()`, `.rebuild_index()` reference
   methods that don't exist on the current MemorySystem class.

2. **LM output is incoherent.** Loss ~1.5 = ~4.5 bits/byte. With byte-level
   tokenization, 937K params cannot produce fluent English. This is expected —
   the model's strength is memory recall, not raw language modeling.

3. **Chat training crashed.** Phase 3 (chat) fails due to broken memory API
   and empty chat dataset construction. Must be fixed before resuming training.

## Performance Benchmarks

```
  Memory:   34μs/write, 7μs/read, 3ms load (9K nodes)
  Model:    ~0.7 it/s on M4 MPS (batch=8)
  Storage:  ~500 bytes/node, 4.7MB for 1000 writes
```

## Next Steps

1. Clean dead code from train_micro.py (~2955 lines)
2. Wire MemorySystem into training loop (Phase A/B/C curriculum)
3. Implement contrastive address loss for Phase B
4. Rewrite chat.py for duplex streaming with per-token trie access
5. Validate on bAbI with new architecture (target: 100% QA)
