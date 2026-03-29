# Memory Linking & Concept Navigation Report

## The Question

How can information in one memory cell lead the agent to discover related information
in another cell? Should this be an explicit mechanism (pointers, graph edges) or can
it emerge from the current architecture? What would "wormholes" between memory regions
look like?

---

## 1. What Already Exists: Implicit Concept Chaining

The current architecture has **three built-in mechanisms** that create concept linking
without any explicit pointer infrastructure.

### 1a. Address Drift via ACT Loop (Phase 5)

This is the most powerful existing linking mechanism. During streaming inference:

```
ACT Step 0:  hidden_state_0  →  addresses_0  →  read memory  →  9 physics vectors
             ↓ (attention transforms hidden state)
ACT Step 1:  hidden_state_1  →  addresses_1  →  read memory  →  9 math vectors
             ↓ (physics + math context shifts thinking)
ACT Step 2:  hidden_state_2  →  addresses_2  →  read memory  →  9 engineering vectors
```

Each ACT step produces a **new hidden state** that computes **new addresses**. The model
literally walks through concept space, and each step's memory read influences where it
goes next. This is already a learned "wormhole" — memory content at address A causes the
model to compute address B on the next step.

**Strength:** Fully learned, no engineering needed.
**Weakness:** One hop per ACT step. With `max_act_steps=8`, the agent can follow chains
of at most ~8 links per token. And ACT budget is shared with thinking — using it all
for memory traversal leaves nothing for computation.

### 1b. Three Independent Address Heads = Parallel Probes

The model doesn't read from one location — it reads from **three simultaneously**:

```
Head 0: "What topic am I thinking about?"      → physics region
Head 1: "What context was this mentioned in?"   → story/narrative region
Head 2: "What's the structural pattern here?"   → syntax/template region
```

Each head can independently point to a different region of concept space. This means
a single read already creates a **cross-reference** between three different semantic
axes. The model sees all three regions' content simultaneously and can synthesize.

**Strength:** Three simultaneous probes is like reading three different books at once.
**Weakness:** Fixed at 3 heads × 3 results = 9 vectors. Can't dynamically decide
"I need 7 vectors from physics and 2 from math."

### 1c. Neighborhood Search as Local Exploration

The ±1 search on fine dimensions (4-7) means each head doesn't just hit one cell:

```
Head 0 exact address: [23, -5, 100, 42, | 10, -3, 55, 0]
                       ─── coarse (exact) ─── fine (±1) ──

Searches:  [23,-5,100,42, 9,-4,54,-1]  ← nearby cell
           [23,-5,100,42, 10,-3,55, 0]  ← exact cell
           [23,-5,100,42, 11,-2,56, 1]  ← nearby cell
           ... up to 3^4 = 81 candidates in fine dimensions
```

This creates a **local bubble** around each address. If two related concepts were
written to nearby fine addresses (which happens naturally because similar hidden
states produce similar addresses), they'll be co-retrieved.

**Strength:** Automatic clustering of related concepts.
**Weakness:** Radius-1 is very local. Concepts that are "related but distant" in
address space won't be found this way.

---

## 2. The Gap: What Can't Emerge Naturally

Despite the three mechanisms above, there's a fundamental limitation:

### The Locality Bottleneck

All current linking is **mediated by the model's hidden state**. The chain is always:

```
memory content → hidden state change → new address → new memory
```

This means:
1. **No long-range jumps in one step.** If "quantum physics" is at address region A
   and "cooking recipes" is at region Z, there's no way for the model to jump from A
   to Z in one read — even if a specific physics fact is directly relevant to a cooking
   technique. The model would need multiple ACT steps to drift there.

2. **No content-triggered teleportation.** A memory cell can't say "after reading me,
   go look at address X." The link is always implicit through hidden state transformation.

3. **Combinatorial explosion in ACT.** Following a chain of 5 concepts costs 5 ACT steps.
   With `max_act_steps=8`, that's most of the budget gone on navigation alone.

### When This Matters

For TinyStories (current training data), this gap is probably irrelevant — stories are
short and self-contained. But for real-world knowledge:

- A legal document references a statute → that statute references a precedent → that
  precedent references a constitutional clause. That's 3 hops minimum.
- A codebase: function A calls function B which uses library C. Understanding A requires
  reaching C.
- Reasoning chains: premise → inference → inference → conclusion.

---

## 3. Five Approaches to Explicit Concept Linking

### Approach A: Multi-Hop Reads (Two-Pass Retrieval)

**Idea:** After reading 9 memory vectors, compute addresses *from those vectors* and
read again. The second read follows wherever the first read's content points.

```python
# Current (single-hop):
mem_vecs = read_memory(compute_addresses(hidden))  # 9 vectors

# Multi-hop:
mem_vecs_1 = read_memory(compute_addresses(hidden))        # First hop: 9 vectors
# Use retrieved content to compute new addresses
hop2_hidden = mean_pool(mem_vecs_1)                         # Aggregate first hop
mem_vecs_2 = read_memory(compute_addresses(hop2_hidden))    # Second hop: 9 more
mem_vecs = interleave(mem_vecs_1, mem_vecs_2)               # Combine (still 9 slots)
```

| Pros | Cons |
|------|------|
| Follows chains without spending ACT budget | Doubles memory read latency |
| Pure architecture change, no new params | How to combine 18→9 vectors? Learned selector? |
| Works with existing trie structure | Second hop is noisy (mean-pool is lossy) |
| The model can learn to "plant breadcrumbs" | Training must teach two-hop utility |

**Verdict:** Clean and powerful. The key insight is that `compute_addresses()` already
works on any 512-d vector — retrieved memory vectors ARE 512-d vectors. The address
heads can naturally compute "what does this memory content remind me of?" This is the
closest thing to a real wormhole.

### Approach B: Stored Back-Links (Pointer Fields in Memory)

**Idea:** Each memory cell stores not just a 512-d vector but also K explicit addresses
of related cells. When you read cell A, you also get "A thinks you should look at B, C."

```python
# Memory record: 512 bytes data + 3×8 bytes back-links = 536 bytes
class MemoryRecord:
    data: np.ndarray        # (512,) int8 — the embedding
    links: List[bytes]      # 3 addresses this cell was co-written with
```

When writing to cell A with context that also addresses cells B and C (from the 3 heads),
store B and C's addresses as back-links in A's record:

```python
def write_memory(addresses, vec):
    for i, addr in enumerate(addresses):
        record = lookup_or_create(addr)
        record.data = ema_blend(record.data, vec)
        # Store the OTHER two heads' addresses as back-links
        record.links = [addresses[j] for j in range(3) if j != i]
```

On read: follow back-links for one extra hop.

| Pros | Cons |
|------|------|
| Explicit graph structure | 24 bytes overhead per record (+4.7%) |
| Co-written concepts are always linked | Links are static (set at write time) |
| Zero compute cost to follow links | Not learned — fixed co-occurrence heuristic |
| Creates natural clustering | Explosion: each link leads to more links |

**Verdict:** Simple but rigid. The links aren't learned — they're just "what was co-active
when this was written." Could be a useful bootstrap for Approach A.

### Approach C: Cross-Slot Attention (Memory Vectors Talk to Each Other)

**Idea:** Before prepending the 9 memory vectors to the input, let them attend to each
other through a small attention layer. This lets information from slot 3 (physics) flow
into slot 7 (math) before the main transformer ever sees them.

```python
class MemoryCrossAttention(nn.Module):
    """1-layer self-attention over the 9 retrieved memory slots."""
    def __init__(self, d_model=512, n_heads=4):
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mem_vecs):  # (B, 9, 512)
        out = self.attn(mem_vecs, mem_vecs, mem_vecs)[0]
        return self.norm(mem_vecs + out)  # Residual + norm
```

```
Before: [physics_0, physics_1, story_0, story_1, syntax_0, ...]
         ↓ (cross-slot attention)
After:  [physics_0+story_context, physics_1+syntax_hints, ...]
```

| Pros | Cons |
|------|------|
| Fully learned combination | Extra parameters (~1M for 4-head attention) |
| Rich interaction between memory slots | Only works within the 9 retrieved vectors |
| Cheap: 9-token attention is trivial | Doesn't create new links, just enriches existing reads |
| Compatible with everything else | May be redundant — main transformer already mixes them |

**Verdict:** Elegant but limited. This is "better fusion" not "new links." The main
transformer's 8 layers already let memory slots interact with each other through
attention. A dedicated cross-attention might help early layers but becomes redundant
by layer 4-5. **Low priority.**

### Approach D: Dynamic Neighborhood Radius

**Idea:** Instead of fixed ±1 on fine dimensions, let the model control the search
radius. When confident, search narrow. When exploring, search wide.

```python
def compute_addresses_with_radius(self, hidden_state):
    addresses = []
    radii = []
    for head in self.addr_heads:
        raw = head(hidden_state)                    # (8,) float
        addr = quantize_to_int8(raw)                # (8,) int8
        # Radius from hidden state — how wide should the search be?
        r = self.radius_head(hidden_state)          # scalar, 0-3
        addresses.append(addr)
        radii.append(r)
    return addresses, radii
```

Search with radius R=2 on fine dims: candidates = 5^4 = 625 (vs 81 for R=1).
Search with radius R=0: exact match only (3 vectors from 3 heads).

| Pros | Cons |
|------|------|
| Model controls exploration vs exploitation | Larger radius = slower search (exponential) |
| Natural curriculum: start narrow, go wide | R=3 means 7^4 = 2401 candidates per head |
| Simple addition (one scalar head per addr head) | Diminishing returns — most neighbors are noise |
| Works with existing trie | Harder to train (radius gradient is discrete) |

**Verdict:** Interesting for exploration behavior but the exponential cost is
concerning. R=2 is probably the sweet spot. Could be scheduled rather than learned:
start narrow in early training, widen later. **Medium priority.**

### Approach E: Write-Time Similarity Index (Background Clustering)

**Idea:** Maintain a secondary index that groups memory cells by content similarity
(not address similarity). When reading, optionally also retrieve the K most similar
cells by content, regardless of address.

```python
class SimilarityIndex:
    """Approximate nearest neighbor index over memory content."""
    def __init__(self):
        self.vectors = []      # list of (record_num, centroid)
        self.clusters = {}     # cluster_id → [record_nums]

    def add(self, record_num, vector):
        # Assign to nearest cluster (or create new)
        ...

    def query(self, vector, k=3):
        # Return k most similar records by content
        ...
```

This decouples "what address did the model compute?" from "what content is actually
similar?" The address heads learn a *task-specific* addressing scheme, while the
similarity index provides a *content-based* fallback.

| Pros | Cons |
|------|------|
| Content-similar cells always findable | Extra memory + compute for index |
| Addresses can be wrong — fallback catches it | Not differentiable (like the trie) |
| Enables discovery of unexpected connections | Similarity in int8 space is crude |
| Useful for seeding new agents | Second retrieval path = complexity |

**Verdict:** Powerful for inference-time exploration but heavy. Better suited as an
offline tool (e.g., "find all cells similar to X" for debugging or agent initialization)
than as a per-token operation. **Low priority for training, high for tooling.**

---

## 4. Comparison Matrix

```
                        Hop    Compute  New      Learned  Complexity
                        Depth  Cost     Params   or Fixed
─────────────────────────────────────────────────────────────────────
Current (ACT drift)     1/step low      0        learned  none
A. Multi-Hop Reads      2      medium   0*       learned  low
B. Stored Back-Links    2      low      0        fixed    low
C. Cross-Slot Attention 1      low      ~1M      learned  medium
D. Dynamic Radius       1      var      ~small   learned  low
E. Similarity Index     ∞      high     0        fixed    high
─────────────────────────────────────────────────────────────────────
* Multi-hop reuses existing address heads — no new parameters
```

---

## 5. Can Linking Be Learned? (The Core Question)

**Yes — and it partially already is.** Here's the argument:

### What the Model Already Learns

During Phase 3/5 training, the model learns:
1. **What to store:** Hidden states capture relevant information
2. **Where to store:** Address heads learn a "concept geography"
3. **What to retrieve:** Address computation from current context
4. **How to use retrieval:** Attention over memory slots

The critical piece is (3). When the model reads a memory vector about "wave-particle
duality" and its hidden state shifts toward "quantum measurement," the address heads
produce new addresses that point toward measurement-related cells. **This IS learned
linking.** The model learns: "after seeing content X, look at address Y."

### What Can't Be Learned (Currently)

The problem is that this linking is **bottlenecked by the model's hidden state
transformation**. The link "physics → math" must be encoded in the transformer weights.
For the model to discover that cell A is related to cell B, it must:

1. Read cell A's content
2. Transform its hidden state through 8 layers of attention
3. Compute new addresses from the transformed state
4. Hope those addresses land near cell B

Step 2-3 is where the "link" lives — in the weights. For a 34M parameter model on
TinyStories, this works fine. For a model that needs to navigate millions of memory
cells across diverse domains, the weight capacity becomes the bottleneck.

### The Multi-Hop Read Solves This

With Approach A (multi-hop), the chain becomes:

1. Read cell A's content (512-d vector)
2. Compute addresses *directly from cell A's content* (no full transformer pass)
3. Read whatever cell A's content addresses

This is fundamentally different. The link is now **in the memory content itself**, not
just in the weights. Cell A's vector, when passed through the address heads, naturally
points to related cells — because similar content was written from similar hidden states,
which produce similar addresses.

The "wormhole" emerges: cell A's content → address heads → cell B. No full forward
pass needed. No ACT step consumed.

---

## 6. Recommendation: Phased Implementation

### Phase 3-5 (Current Training): No Changes Needed

The current architecture's implicit linking through ACT drift is sufficient for
TinyStories. Adding complexity now risks training instability with no measurable gain.
The model needs to first learn basic memory utility before learning to chain-navigate.

### Post-TinyStories Enhancement #1: Multi-Hop Reads

**This is the highest-value, lowest-cost enhancement.** Implementation:

```python
def read_memory_multihop(self, hidden_state, memory_system, cfg, device):
    """Two-hop memory read: follow where retrieved content points."""
    # Hop 1: standard read
    addrs_1 = self.compute_addresses(hidden_state)
    vecs_1 = memory_system.read_memory(addrs_1)              # 9 vectors

    # Hop 2: compute addresses from retrieved content
    hop2_input = torch.tensor(vecs_1, dtype=torch.float32, device=device) / 127.0
    hop2_hidden = hop2_input.mean(dim=0)                      # (512,)
    addrs_2 = self.compute_addresses(hop2_hidden)
    vecs_2 = memory_system.read_memory(addrs_2)               # 9 more vectors

    # Combine: keep top-K by address head confidence or dedup
    # Simple: interleave 5 from hop1 + 4 from hop2 = 9 total
    combined = deduplicate(vecs_1[:5] + vecs_2[:4])
    return pad_to_9(combined)
```

No new parameters. Reuses existing address heads. The "wormhole" is free.

### Post-TinyStories Enhancement #2: Learned Hop Gating

Let the model decide whether to take a second hop:

```python
self.hop_gate = nn.Linear(d_model, 1)  # Sigmoid → probability of second hop

def should_multihop(self, hidden_state):
    return torch.sigmoid(self.hop_gate(hidden_state)) > 0.5
```

During training, the model learns when multi-hop is useful (complex queries) vs
wasteful (simple lookups). This keeps single-hop fast for easy cases.

### Long-Term Enhancement #3: Recursive Reads with Depth Budget

Generalize multi-hop to N hops with a learned stopping criterion:

```python
def read_recursive(self, hidden, memory, max_depth=4):
    all_vecs = []
    for depth in range(max_depth):
        vecs = memory.read(self.compute_addresses(hidden))
        all_vecs.extend(vecs)
        # Should we go deeper?
        fused = aggregate(vecs)
        if self.stop_gate(fused) > 0.5:
            break
        hidden = fused  # Chase the chain
    return select_top_9(all_vecs)  # Best 9 from all hops
```

This is the full "wormhole navigation" — the agent can follow chains of arbitrary
length, limited only by the depth budget. Combined with ACT, this gives:

```
Per token: up to max_depth memory hops × max_act_steps computation steps
         = 4 hops × 8 ACT steps = 32 potential memory accesses per token
```

---

## 7. Key Insight: The Address Heads ARE the Wormholes

The deepest insight is this: **the address heads are already wormhole generators.**

When you compute `address = addr_head(vector)` for ANY 512-d vector — whether it's a
hidden state, a retrieved memory cell, or a mean of multiple cells — you get a
valid address that points somewhere meaningful in memory space.

A retrieved memory vector IS a 512-d vector. Passing it through the address heads
gives you "where does this memory cell's content point to?" That's a wormhole.

The only thing missing is the **plumbing** to actually follow these pointers. The
model can already compute them — it just doesn't have a codepath to act on them
outside the ACT loop.

Multi-hop reads add that codepath. Everything else is already learned.

---

## 8. Analogy Summary

| Concept | Memory System Equivalent |
|---------|--------------------------|
| Hyperlink | Multi-hop address computation from retrieved content |
| Wormhole | Address heads applied to memory vectors (teleportation in concept space) |
| Neighborhood | ±1 fine-dimension search (local cluster) |
| Breadcrumb trail | ACT-step address drift (sequential navigation) |
| Graph edge | Co-occurrence during write (back-links, Approach B) |
| Search engine | Similarity index (Approach E) |
| GPS coordinates | The 8-dim int8 address itself |
| Following your nose | The current system — hidden state transformation guides next read |

---

## 9. Summary

**Can concept linking be learned with the current architecture?**
Yes, through ACT-step address drift. The model learns to navigate concept space
by letting each memory read influence its next address computation.

**Is it sufficient?**
For TinyStories, yes. For complex knowledge navigation, no — the ACT budget is
too expensive for pure navigation, and single-hop reads can't teleport across
distant concept regions.

**What's the best enhancement?**
Multi-hop reads (Approach A). Zero new parameters, reuses existing address heads,
and creates true wormholes where memory content directly points to related memory.
The address heads already know how to compute "where does this content relate to?"
— we just need to let them do it on retrieved vectors, not only on hidden states.

**When to implement?**
After current training (Phases 3-5) completes on TinyStories. The model needs to
first learn basic memory read/write utility before we ask it to chain-navigate.
Multi-hop on untrained memory is just noise following noise.
