# Looped Latent Controller with Shared Memory

A ~34M parameter decoder-only transformer that operates as a token stream processor with persistent shared memory, adaptive computation time, and multi-agent orchestration.

## Quick Start (Google Colab)

```bash
pip install -r requirements.txt
python run_colab.py --phase 1 --checkpoint_dir ./checkpoints
```

## Architecture

- **Model**: 34M param decoder-only transformer (d_model=512, 8 heads, 8 layers, RoPE, RMSNorm, SiLU)
- **Memory**: Trie-indexed flat file of 512-byte int8 vectors, 3-probe addressing, adaptive EMA writes
- **ACT**: Adaptive computation time with differentiable soft halting (training) / hard halting (inference)
- **Agent**: Token stream processor — one method: `process_token(token_id) → output_token_id | None`

## Training Phases

1. **Phase 1**: Baseline language model on TinyStories (~6-8h on T4)
2. **Phase 2**: Address head contrastive pretraining (~30min on T4)
3. **Phase 3**: Memory integration with live memory building (~12-15h on T4)
4. **Phase 4**: Adaptive computation time with ponder curriculum (~24-30h on T4)

## Usage

```python
from orchestrator import Orchestrator
from model import LoopedLatentController
from memory import MemorySystem
from config import ModelConfig, MemoryConfig

model = LoopedLatentController(ModelConfig())
memory = MemorySystem('./memory', MemoryConfig())
orch = Orchestrator(model, memory, tokenizer, device='cuda')

response = orch.query("Once upon a time there was a little bear")
```