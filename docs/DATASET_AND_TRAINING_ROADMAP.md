# Dataset Generation & Training Roadmap Report

## The Goal

Train a 34M-parameter looping transformer where **weights = DNA** (intelligence,
reasoning, how to think) and **memory = lived experience** (facts, context, knowledge).
The model must evolve through continuous training across increasingly complex capabilities
while memory is periodically rebuilt to match the evolved weights.

This report covers: what data to train on, in what order, how to test each stage,
when to rebuild memory, and how LoRA adapters enable specialization without corrupting
the shared address space.

---

## Current State

```
Phase 1: ✅ Complete (TinyStories, 1.5B tokens, PPL ~2.x, on HuggingFace)
Phase 2: ✅ Complete (Address head contrastive pretraining, 10K steps)
Phase 3: 🔄 Running (Memory integration, 2B tokens, A100 40GB)
Phase 4: ⏳ Queued (ACT halting, 1B tokens)
Phase 5: ⏳ Queued (Unified streaming, 2B tokens)

Model: 34M params, 512-d, 8 layers, 8 heads, vocab 16,512
Context: 512 tokens + 9 memory vectors (11 memory positions)
Hardware: A100 40GB (Colab)
```

**Hard limits that affect dataset choices:**
- **512 token context** — all training samples must fit within this
- **16,512 vocab** — English BPE only; multilingual needs tokenizer rebuild
- **34M params** — can't absorb too many capabilities at once
- **9 memory slots** — memory bandwidth is fixed per forward pass

---

## Part 1: The Evolution Model

### Philosophy: Generational Training

Each "generation" is a full cycle:

```
Generation N:
  1. Train weights on new/harder data (Phases 1-5 or subset)
  2. Rebuild memory from scratch using updated weights
  3. Evaluate: did the weights improve? Does memory help more?
  4. If yes → Generation N+1 with harder data
  5. If no  → diagnose, adjust data mix, retry
```

This mirrors biological evolution:
- **Weights (DNA)** improve across generations through training
- **Memory (experience)** is rebuilt each generation using the new DNA
- **LoRA adapters (epigenetics)** allow specialization without changing DNA

### Why Rebuild Memory Between Generations?

When weights change, the address heads produce different addresses for the same content.
Old memory was indexed by old addresses — it becomes unreachable or maps to wrong
content. Rebuilding memory re-indexes everything under the new address space.

```
Generation 1 weights: "quantum physics" → address [23, -5, 100, 42, 10, -3, 55, 0]
Generation 2 weights: "quantum physics" → address [25, -3, 98, 44, 8, -1, 53, 2]

Old memory at [23,-5,...] is now orphaned.
Must rebuild: run all knowledge through Gen 2 weights → new addresses → new memory.
```

**Exception:** If only LoRA adapters change (not base weights or address heads), memory
stays valid because address heads are frozen during LoRA training.

---

## Part 2: Dataset Strategy by Capability

### Tier 0: Foundation (Current — TinyStories)

**What it teaches:** Basic English grammar, storytelling, token prediction.
**Why it matters:** Establishes the weight foundation that all later training builds on.
**Status:** Phase 1 complete, Phases 3-5 in progress.

**After Phase 5 on TinyStories, the model can:**
- Generate coherent short stories
- Use memory to recall story elements
- Adapt computation depth per token (ACT)
- Read/write to persistent trie-indexed memory
- Operate in streaming token-by-token mode

**It cannot:** Handle non-English, reason about math, use tools, process images.

### Tier 1: Memory Dependence (Critical — Do This First After TinyStories)

**The problem:** The model can currently answer TinyStories questions from weights alone.
It learns to ignore memory because weights are sufficient. This is the single biggest
training gap.

**Dataset: Memory-Dependent QA**

Generate synthetic two-phase sequences:

```
Phase A (ingest):  "The capital of Freedonia is Glorbville. Its population is 42,000."
Phase B (query):   "What is the capital of Freedonia?" → "Glorbville"
```

The answer ("Glorbville") is a made-up fact that cannot exist in the weights. The model
MUST read memory to answer correctly. This creates direct gradient signal for memory use.

**Sources:**
| Dataset | Why | How to Use |
|---------|-----|-----------|
| SQuAD | Passage + question + answer | Ingest passage → query question |
| TriviaQA | Diverse factual QA | Ingest evidence → query |
| NarrativeQA | Long story comprehension | Ingest story chunks → query about details |
| HotpotQA | Multi-hop reasoning | Ingest 2+ passages → query requiring both |
| **Synthetic** | Made-up facts | Generate random entity/attribute pairs |

**Synthetic generation recipe:**
```python
import random
ENTITIES = ["Zorblon", "Freedonia", "Agent-7X", ...]
ATTRIBUTES = ["capital", "color", "inventor", "speed", ...]
VALUES = [random_word() for _ in range(10000)]

def make_memory_qa_pair():
    e, a, v = random.choice(ENTITIES), random.choice(ATTRIBUTES), random.choice(VALUES)
    ingest = f"The {a} of {e} is {v}."
    query = f"What is the {a} of {e}?"
    answer = v
    return ingest, query, answer
```

**Training protocol:**
1. Feed ingest text through model (memory writes accumulate)
2. Feed query text (model must read memory to predict answer)
3. Loss only on answer tokens (not on ingest or query)

**Go/no-go:** `ppl_with_mem < ppl_without_mem` by at least 0.5 on held-out QA pairs.

### Tier 2: Multilingual

**The insight:** If French and English sentences about the same topic map to the same
memory address, memory becomes a language-agnostic knowledge store. This is the
"weights = grammar (per-language), memory = meaning (universal)" principle.

**Tokenizer change required:** Current 16K English BPE → 32K multilingual BPE.
This means:
- Retrain tokenizer on multilingual corpus
- Reinitialize embedding layer (vocab_size 16,512 → ~33,000)
- Retrain from Phase 1 (embeddings are the foundation)

**Datasets:**
| Dataset | Size | Languages |
|---------|------|-----------|
| mC4 (subset) | ~100M tokens per lang | 100+ languages |
| OPUS/Tatoeba | Parallel sentence pairs | 300+ language pairs |
| Flores | Evaluation benchmark | 200 languages |
| CC-100 | Large monolingual | 100 languages |

**Training protocol:**
1. Phase 1 on mixed multilingual data (50% English, 50% other)
2. Phase 2 with multilingual hidden states (address heads learn cross-lingual mapping)
3. Phase 3 with parallel sentence pairs:
   - Ingest English sentence → memory write
   - Feed French equivalent → model should read same memory region
   - Loss on next-token prediction (French) — memory must help
4. Phase 5 with mixed-language sequences

**Go/no-go:** Same-meaning sentences in different languages produce cosine similarity
> 0.8 in address space. Cross-lingual memory retrieval improves perplexity.

### Tier 3: Reasoning & Tool Use

**What to teach:** Step-by-step reasoning, memory as scratchpad, structured action→result.

**Datasets:**
| Dataset | Capability | Format |
|---------|-----------|--------|
| GSM8K | Math reasoning | Question → chain-of-thought → answer |
| ARC | Science reasoning | Multiple choice with evidence |
| ToolBench | API tool calling | Action/observation pairs |
| Code search + execution | Code tools | Query → code → result |
| MATH | Advanced math | Problem → solution steps |

**Tool use training format:**
```
Input:  "What is the weather in Paris?"
Target: "<THINK>I need to check weather data</THINK>
         <TOOL:weather>Paris</TOOL>
         <RESULT>15°C, partly cloudy</RESULT>
         The weather in Paris is 15°C and partly cloudy."
```

Where `<THINK>` sections are NOOP-targeted (written to memory, not emitted) and
`<TOOL:*>` tokens trigger external execution.

**Memory as scratchpad training:**
```
Step 1: "23 × 47 = ?"
Step 2: Model writes intermediate: "23 × 40 = 920" to memory
Step 3: Model writes: "23 × 7 = 161" to memory
Step 4: Model reads both, outputs: "920 + 161 = 1081"
```

The key is that intermediate results live in memory, not in the 512-token context.
This teaches the model that memory extends its working space.

**Go/no-go:** GSM8K accuracy > 20% (baseline for 34M model with memory advantage).
Tool action tokens appear in correct positions. Memory scratchpad usage is measurable.

### Tier 4: Large Context Consumption & Analytics

**The challenge:** 512-token context is tiny. But memory is infinite. The model must
learn to ingest long documents chunk-by-chunk, storing knowledge in memory, then
answer questions that require information from multiple chunks.

**Training approach — Chunked Ingestion:**
```
Document: 5000 tokens
Chunk 1: tokens[0:500]    → ingest, memory writes
Chunk 2: tokens[500:1000] → ingest, memory writes (reads chunk 1 context from memory)
...
Chunk 10: tokens[4500:5000] → ingest
Query: "Summarize the main argument"  → model reads memory spanning all 10 chunks
```

**Datasets:**
| Dataset | Use Case |
|---------|----------|
| BookCorpus | Long narrative ingestion |
| arXiv abstracts+papers | Technical document consumption |
| Wikipedia articles | Encyclopedic knowledge ingestion |
| Financial reports (SEC EDGAR) | Analytics / structured data |
| Time-series descriptions | Temporal pattern recognition |

**Time-series training:**
```
Input: "Revenue Q1: $10M, Q2: $12M, Q3: $11M, Q4: $15M"
       → model writes quarterly data to memory
Query: "What is the trend?" → "Revenue grew 50% over the year with a Q3 dip"
```

The model learns to store sequential data in memory and retrieve patterns across time.

**Go/no-go:** QA accuracy on questions requiring information from chunks > 2 apart.
Summary quality on multi-chunk documents (ROUGE against reference summaries).

### Tier 5: Multimodality

**Architecture changes needed:**
1. Small image encoder (e.g., tiny ViT or CLIP projection) → 512-d vectors
2. Image vectors written to memory via address heads (same as text)
3. New special tokens: `<IMG>`, `</IMG>`, `<IMG_PATCH_N>`
4. Extend vocab for image description tokens

**The elegant part:** Images become memory entries. The model doesn't "see" images
during forward pass — it reads image-derived embeddings from memory slots, same as text
memories. No model architecture change needed beyond the encoder.

```
Ingestion: image → ViT encoder → 512-d patches → write to memory
Query: "What's in the image?" → model reads image memory → describes
```

**Datasets:**
| Dataset | Task |
|---------|------|
| MS-COCO | Image captioning |
| VQAv2 | Visual question answering |
| TextVQA | OCR + reasoning |
| Conceptual Captions | Image-text pairs |

**Go/no-go:** Caption BLEU > baseline. VQA accuracy measurable improvement over
text-only. Image memory vectors cluster by visual similarity in address space.

### Tier 6: Emergent Swarm Behavior

**This is NOT a dataset task — it's an architecture task.** Swarm behavior emerges from:
1. Multiple agents sharing the same memory
2. Agents with same base weights (same address space)
3. LoRA specialization per agent role

**Training protocol:**
```
Phase 6a: Two agents, shared memory, cooperative task
  Agent A ingests document → writes to memory
  Agent B answers questions → reads from memory
  Loss: B's answer quality (A has no direct loss)

Phase 6b: Agent specialization via LoRA
  Agent A: LoRA adapter for "research" (deep reading, many memory writes)
  Agent B: LoRA adapter for "synthesis" (memory reading, coherent output)
  Both share base weights and address heads

Phase 6c: Scaling to N agents
  Orchestrator creates agents for sub-tasks
  Natural EMA blending means more-active agents have more memory influence
  No explicit coordination — emergent from shared address space
```

**Go/no-go:** Agent B's accuracy on questions about Agent A's document > random.
Multi-agent pipeline produces higher quality output than single agent.

---

## Part 3: LoRA Implementation

### Design from Conversations

**Core principles:**
1. **Address heads are NEVER LoRA-adapted** — must stay shared across all agents
2. **LoRA on attention Q,V projections** — rank 4-8
3. **Base weights frozen during LoRA training** — only adapters update
4. **Adapters stored per-agent** — swappable at runtime

### Implementation Sketch

```python
class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for existing Linear layers."""
    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)  # Freeze base
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out


def apply_lora(model, rank=4, alpha=1.0):
    """Apply LoRA to all attention Q and V projections."""
    for layer in model.layers:
        layer.attn.q_proj = LoRALinear(layer.attn.q_proj, rank, alpha)
        layer.attn.v_proj = LoRALinear(layer.attn.v_proj, rank, alpha)
    # NEVER touch addr_heads
    for head in model.addr_heads:
        head.weight.requires_grad_(False)
    return model


def save_lora(model, path):
    """Save only LoRA parameters."""
    lora_state = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(lora_state, path)


def load_lora(model, path):
    """Load LoRA adapter into model."""
    lora_state = torch.load(path)
    model.load_state_dict(lora_state, strict=False)
```

### LoRA Parameter Budget

```
Per Q/V projection: d_model × rank × 2 = 512 × 4 × 2 = 4,096 params
Per layer: 2 projections × 4,096 = 8,192 params
Total (8 layers): 8 × 8,192 = 65,536 params (~66K)

That's 0.19% of the 34M base model. Tiny, swappable, storable.
```

### LoRA Training Protocol ("Sleep")

```
1. Agent runs inference, accumulates experience in memory
2. Periodically (every N interactions), enter "sleep" mode:
   a. Freeze base weights
   b. Train LoRA on replay buffer of recent interactions
   c. Evaluate: did perplexity improve on held-out set?
   d. If yes → commit adapter. If no → rollback.
3. Resume inference with updated adapter
```

### Adapter Versioning

```
adapters/
  agent_research_v1.pt    # 66K params
  agent_research_v2.pt    # After sleep cycle 2
  agent_synthesis_v1.pt   # Different specialization
  agent_code_v1.pt        # Code-focused adapter
```

Hot-swappable: load different adapter in ~1ms (just matrix addition).

---

## Part 4: Realistic Roadmap & Timeline

### Generation 0: TinyStories Foundation (Current)

```
PHASE 3  ───→  PHASE 4  ───→  PHASE 5  ───→  EVALUATE
Memory         ACT Halting    Unified         Go/No-Go
Integration    1B tokens      Streaming       Decision
2B tokens      ~2-3 hours     2B tokens
~5 hours       A100           ~5 hours
A100                          A100
```

**What to expect:**
- Phase 3: Loss should drop to ~1.5-2.0 per step. Memory ppl should beat no-mem ppl
  by step 5000+. If not, memory integration isn't working.
- Phase 4: Halt histogram should show variance (not all tokens halt at step 1).
  Average halt depth ~2-3 for simple text, ~4-5 for complex.
- Phase 5: Loss should match or beat Phase 3 levels. Memory re-reads between ACT
  steps should further improve ppl.

**Testing Generation 0:**
```python
# After Phase 5 completion:

# Test 1: Generation quality
agent = Agent(model, memory)
for prompt in ["Once upon a time", "The cat sat on", "A brave knight"]:
    output = orchestrator.generate(agent, prompt, max_tokens=100)
    print(output)  # Should be coherent English stories

# Test 2: Memory utility
ppl_no_mem = evaluate(model, val_set, memory=None)
ppl_with_mem = evaluate(model, val_set, memory=trained_memory)
assert ppl_with_mem < ppl_no_mem, "Memory should help!"
print(f"Memory benefit: {ppl_no_mem - ppl_with_mem:.2f} ppl reduction")

# Test 3: ACT behavior
for token in tokenize("The quick brown fox"):
    steps = agent.process_token_with_stats(token)
    print(f"Token '{token}': {steps} ACT steps")
# Function words ("the") should use fewer steps than content words ("fox")

# Test 4: Memory persistence
agent.process_tokens("The wizard's name is Glorpax.")
# ... 100 tokens later ...
output = orchestrator.generate(agent, "What was the wizard's name?", max_tokens=20)
assert "Glorpax" in output  # Memory recall test
```

**Decision point:** If all 4 tests pass → proceed to Generation 1.
If memory doesn't help (Test 2 fails) → need memory-dependent QA data (Tier 1).

### Generation 1: Memory Dependence (Post-TinyStories)

```
REBUILD MEMORY  ───→  PHASE 5a: Memory QA  ───→  EVALUATE
(empty start)         Synthetic + SQuAD          Memory must
                      ~1B tokens                 demonstrably help
                      ~3 hours A100
```

**Data mix:** 50% TinyStories (prevent forgetting) + 50% memory-dependent QA.

**What to expect:**
- First 1000 steps: loss spikes (new data distribution)
- Steps 1000-5000: loss drops, memory ppl starts beating no-mem ppl
- By step 10000: model reliably stores and retrieves synthetic facts
- Failure mode: model memorizes QA patterns in weights instead of using memory.
  Fix: increase synthetic entity diversity (100K+ unique entities).

**Testing Generation 1:**
```python
# Held-out QA pairs the model has NEVER seen:
for entity, attribute, value in held_out_qa:
    agent.process_tokens(f"The {attribute} of {entity} is {value}.")
    output = orchestrator.generate(agent, f"What is the {attribute} of {entity}?")
    accuracy = value in output

# Target: > 60% accuracy on held-out synthetic QA
# (Pure weight memorization would give ~0% on novel entities)
```

### Generation 2: Tokenizer Upgrade & Multilingual

**This is a major reset.** New tokenizer = new embeddings = retrain from Phase 1.

```
RETRAIN TOKENIZER  ───→  PHASE 1  ───→  PHASE 2  ───→  PHASES 3-5
32K multilingual          Multilingual    Address heads   Memory + ACT +
BPE on mC4 + CC100       foundation      (cross-lingual) Streaming
                          ~2B tokens      10K steps       ~3B tokens
                          ~8 hours A100   ~30 min         ~12 hours A100
```

**Config changes:**
```python
vocab_size: 33_024  # 32,768 BPE + 256 reserved
# Everything else stays the same
```

**Data mix for Phase 1:** 40% English + 10% each for 6 target languages.
**Data mix for Phase 3:** Parallel sentence pairs (OPUS) + monolingual.
**Data mix for Phase 5:** Mixed-language sequences with memory-dependent QA.

**What to expect:**
- Phase 1 will take ~2× longer (larger vocab = harder prediction task)
- Address heads should learn cross-lingual mapping by Phase 3
  (French "chat" and English "cat" → nearby addresses)
- Memory perplexity benefit should be even stronger (cross-lingual retrieval
  provides "free" information the model can't get from weights alone)

**Testing Generation 2:**
```python
# Cross-lingual memory transfer:
agent.process_tokens("Le chat est sur le tapis.")  # French: "The cat is on the mat"
output = orchestrator.generate(agent, "What animal was mentioned?")
assert "cat" in output.lower()  # English answer from French memory

# Address space alignment:
addr_en = model.compute_addresses(encode("The cat sleeps"))
addr_fr = model.compute_addresses(encode("Le chat dort"))
cosine_sim = cosine(addr_en, addr_fr)
assert cosine_sim > 0.7  # Same meaning → similar addresses
```

### Generation 3: Reasoning & Tool Use

```
REBUILD MEMORY  ───→  PHASE 5b: Reasoning  ───→  PHASE 5c: Tools  ───→  EVALUATE
                      GSM8K + HotpotQA           ToolBench + Synthetic
                      ~1B tokens                 ~1B tokens
                      ~4 hours A100              ~4 hours A100
```

**Requires new special tokens (from reserved pool):**
```
<THINK>  = ID 7   (internal reasoning, NOOP-targeted)
</THINK> = ID 8
<TOOL>   = ID 9   (tool invocation marker)
</TOOL>  = ID 10
<RESULT> = ID 11  (tool result injection)
</RESULT>= ID 12
```

**Data mix:** 30% TinyStories + 20% Memory QA + 25% Reasoning + 25% Tool Use.

**What to expect:**
- Model learns to use `<THINK>` blocks for intermediate reasoning (NOOP → memory)
- Tool tokens appear at correct positions in output
- GSM8K accuracy starts low (~5%) but should reach ~20% with memory scratchpad
- Memory usage should increase dramatically (more writes per sequence)

**Testing Generation 3:**
```python
# Math with memory scratchpad:
output = orchestrator.query(agent, "What is 23 × 47?", think_budget=6)
# Should show: internal reasoning steps in memory, final answer emitted

# Tool invocation:
output = orchestrator.generate(agent, "Search for: latest news about AI")
assert "<TOOL>" in output  # Model produces tool tokens

# Multi-hop reasoning:
agent.process_tokens("Alice is taller than Bob. Bob is taller than Charlie.")
output = orchestrator.query(agent, "Who is tallest?", think_budget=4)
assert "Alice" in output  # Requires reading + reasoning over memory
```

### Generation 4: Long Context & Analytics

```
REBUILD MEMORY  ───→  PHASE 5d: Chunked Ingestion  ───→  EVALUATE
                      Wikipedia + arXiv + Financial
                      ~2B tokens
                      ~8 hours A100
```

**Training on chunked documents:**
```python
def chunk_and_train(document, model, memory, chunk_size=450):
    chunks = split_into_chunks(document, chunk_size)
    for chunk in chunks:
        # Ingest: model processes chunk, writes to memory
        for token in tokenize(chunk):
            agent.process_token(token)
    # Query: ask about content spanning multiple chunks
    question = generate_question_about(document)
    loss = compute_loss(model, question, answer, memory)
```

**What to expect:**
- Memory grows large (10K+ entries per training run)
- Model learns to write "summary" vectors that compress chunk information
- Cross-chunk retrieval accuracy improves steadily
- Time-series data is stored as sequential memory writes with temporal addressing

### Generation 5: Multimodality

```
ADD IMAGE ENCODER  ───→  PHASE 5e: Vision  ───→  EVALUATE
ViT-Tiny → 512-d          COCO + VQA
projection                 ~1B tokens
                           ~6 hours A100
```

**Architecture addition:**
```python
class ImageEncoder(nn.Module):
    def __init__(self, d_model=512):
        self.vit = TinyViT(...)  # Or CLIP image encoder
        self.proj = nn.Linear(vit_dim, d_model)

    def encode(self, image) -> torch.Tensor:
        """Returns (N_patches, d_model) embeddings."""
        patches = self.vit(image)
        return self.proj(patches)
```

Image patches → 512-d vectors → written to memory via address heads → model reads them
as standard memory slots. **No change to the core transformer.**

### Generation 6+: Continuous Evolution

```
LOOP:
  1. Identify weakest capability (eval suite)
  2. Generate/collect targeted training data
  3. Train Phase 5 on data mix (old capabilities + new focus)
  4. Evaluate all capabilities (regression test)
  5. If improvement without regression → commit weights
  6. If regression → reduce new data proportion, retry
  7. Rebuild memory from updated weights
  8. Optional: Train LoRA adapters for specialized roles
  9. GOTO 1
```

---

## Part 5: Data Mix Proportions Over Time

The key to continuous training without catastrophic forgetting is careful data mixing.
The old data acts as a "replay buffer" that prevents the model from forgetting.

```
Generation 0 (Foundation):
  100% TinyStories

Generation 1 (Memory):
  50% TinyStories + 50% Memory QA

Generation 2 (Multilingual):
  30% English stories + 20% Memory QA + 50% Multilingual

Generation 3 (Reasoning):
  20% Stories + 15% Memory QA + 15% Multilingual + 25% Reasoning + 25% Tools

Generation 4 (Long Context):
  15% Stories + 10% Memory QA + 10% Multilingual + 15% Reasoning +
  15% Tools + 35% Long Documents

Generation 5 (Multimodal):
  10% Stories + 10% Memory QA + 10% Multilingual + 10% Reasoning +
  10% Tools + 20% Long Docs + 30% Image+Text

Generation 6+ (Maintenance):
  Equal mix of all, overweight weakest capability by 2×
```

**Rule of thumb:** Never drop any capability below 10% of the training mix.
Never add a new capability at more than 50% (overwhelms existing knowledge).

---

## Part 6: Evaluation Framework

### Per-Generation Test Suite

Every generation must pass ALL previous tests plus new ones:

```python
class EvalSuite:
    def __init__(self, model, memory):
        self.tests = []

    # Generation 0
    def test_story_coherence(self):
        """Generate stories, score grammaticality and coherence 0-3."""

    def test_memory_utility(self):
        """ppl_with_mem < ppl_without_mem"""

    def test_act_variance(self):
        """Halt histogram std > 1.0"""

    # Generation 1
    def test_memory_recall(self):
        """> 60% accuracy on novel synthetic QA"""

    # Generation 2
    def test_crosslingual_memory(self):
        """Same-meaning cross-language address similarity > 0.7"""

    # Generation 3
    def test_math_reasoning(self):
        """GSM8K accuracy > 20%"""

    def test_tool_tokens(self):
        """Tool markers appear in correct positions"""

    # Generation 4
    def test_crosschunk_qa(self):
        """QA requiring info from chunks > 2 apart: > 40% accuracy"""

    # Generation 5
    def test_image_description(self):
        """COCO caption BLEU > baseline"""

    def run_all(self):
        results = {}
        for test in self.tests:
            results[test.__name__] = test()
        regressions = [k for k, v in results.items() if not v]
        if regressions:
            print(f"❌ REGRESSIONS: {regressions}")
        return results
```

### Go/No-Go Criteria Per Phase

| Transition | Required | Metric |
|-----------|----------|--------|
| Phase 3 → Phase 4 | Memory helps | `ppl_mem < ppl_no_mem` by > 0.1 |
| Phase 4 → Phase 5 | ACT varies | Halt histogram `std > 1.0` |
| Phase 5 → Gen 1 | All Phase tests pass | Story coherence > 2.0/3.0 |
| Gen 1 → Gen 2 | Memory recall works | > 60% on synthetic QA |
| Gen 2 → Gen 3 | Multilingual aligned | Cross-lingual cos sim > 0.7 |
| Gen 3 → Gen 4 | Reasoning works | GSM8K > 20%, tools appear |
| Gen 4 → Gen 5 | Long context works | Cross-chunk QA > 40% |

---

## Part 7: Scaling Considerations

### Model Size

34M params is sufficient through Generation 2. At Generation 3 (reasoning + tools),
you'll likely hit capacity limits. Two paths:

**Path A: Grow the model**
```
Gen 0-2:  34M  (512-d, 8 layers, 8 heads)
Gen 3-4:  ~80M (768-d, 12 layers, 12 heads)
Gen 5+:  ~150M (1024-d, 16 layers, 16 heads)
```
Each growth requires retraining from Phase 1. Memory rebuilds automatically.

**Path B: LoRA specialization (keep 34M base)**
```
Base: 34M (general capabilities)
+ Research LoRA: +66K (deep reading)
+ Math LoRA: +66K (reasoning)
+ Code LoRA: +66K (programming)
+ Translation LoRA: +66K (multilingual)
```
Each LoRA is 0.19% of base. Swappable in ~1ms. No Phase 1 retrain needed.

**Recommendation:** Use Path B as long as possible. Only grow the base when LoRA
adapters stop being sufficient (probably around Generation 4-5).

### Memory Scale

```
Gen 0 (TinyStories):   ~100K entries × 512 bytes = ~50MB
Gen 1 (QA):            ~500K entries = ~250MB
Gen 2 (Multilingual):  ~2M entries = ~1GB
Gen 3 (Reasoning):     ~5M entries = ~2.5GB
Gen 4 (Long Context):  ~20M entries = ~10GB
Gen 5 (Multimodal):    ~50M entries = ~25GB
```

All fits in RAM for A100 (89 GB system memory). For larger scales, will need
memory sharding or eviction policy (LRU based on write_count).

### Compute Budget Per Generation

```
                    Phase 1    Phase 2    Phase 3-5    Total
Gen 0 (TinyStories):  6h        30m        10h         ~17h
Gen 1 (Memory QA):    —         —          3h          ~3h
Gen 2 (Multilingual): 8h        30m        12h         ~21h  (full retrain)
Gen 3 (Reasoning):    —         —          8h          ~8h
Gen 4 (Long Context): —         —          8h          ~8h
Gen 5 (Multimodal):   —         —          6h          ~6h
────────────────────────────────────────────────────────────
Total through Gen 5:                                   ~63h A100
```

At Colab A100 rates, this is manageable across weeks of training sessions.

---

## Part 8: Inline Editing (Backspace Capability)

The architecture supports backspace tokens (`<bs1>` through `<bs8>` from reserved pool)
but they're not yet trained. Training requires:

**Dataset:** Text with intentional corrections:
```
"The capital of France is Lond<bs4>Paris."
→ Teaches: emit "Lond", realize mistake, backspace 4, emit "Paris"
```

**Training signal:** Generate correction pairs from common errors:
```python
def make_correction_pair(text):
    # Insert a plausible wrong start, then correct
    words = text.split()
    target_idx = random.randint(1, len(words)-1)
    wrong_word = get_similar_word(words[target_idx])  # "London" for "Paris"
    prefix = wrong_word[:random.randint(2, len(wrong_word)-1)]
    bs_count = len(tokenize(prefix))
    return f"{prefix}<bs{bs_count}>{words[target_idx]}"
```

**When to add:** Generation 3 or 4. The model needs solid generation first before
learning to self-correct.

---

## Part 9: Summary — What to Expect at Each Stage

| Stage | Duration | After Completion |
|-------|----------|-----------------|
| **Phase 3** (now) | ~5h | Model reads/writes memory. Loss ~1.5-2.0/step |
| **Phase 4** | ~3h | Adaptive compute depth. Variable halt steps |
| **Phase 5** | ~5h | Full streaming agent. Memory re-reads in ACT loop |
| **Gen 0 eval** | 30m | Coherent stories, memory helps, ACT varies |
| **Gen 1** | ~3h | Stores novel facts, recalls them. Memory is essential |
| **Gen 2** | ~21h | Multilingual. Same meaning = same memory address |
| **Gen 3** | ~8h | Math reasoning, tool tokens, memory scratchpad |
| **Gen 4** | ~8h | Processes long documents via chunked ingestion |
| **Gen 5** | ~6h | Sees images via memory-encoded patches |
| **Gen 6+** | ongoing | Continuous refinement. LoRA specialization. Swarm |

**The model at Generation 3** will be the first version that feels like a real agent:
it can think (ACT), remember (memory), reason (chain-of-thought in memory), and
act (tool tokens). Everything before that is building the foundation.

**The model at Generation 5** will be a multimodal, multilingual agent with persistent
memory, adaptive computation, tool use, and self-correction — all in 34M parameters
plus LoRA adapters. The weights are the DNA; memory is everything it has ever learned.
