# Tool-Calling for LatentController: Architecture Report

## Problem Statement

LatentController is a ~34M parameter streaming token processor with a 512-token context window, persistent trie-indexed memory, and adaptive computation time (ACT). We need to add tool-calling capabilities that align with the core design principle: **"An agent is a stream processor. Tokens go in, tokens come out."**

The challenge is unique: unlike 70B+ models that can reason about JSON schemas and multi-step plans in-context, our model operates with extreme constraints—tiny context, tiny parameters, but compensates with persistent memory and iterative computation. Tool-calling must work *within* this paradigm, not fight against it.

---

## Architectural Constraints

| Constraint | Value | Impact on Tool-Calling |
|---|---|---|
| Parameters | ~34M | Can't learn complex JSON schemas or multi-step tool plans in-weights |
| Context window | 512 tokens | Can't fit tool schemas + input + output in one pass |
| Memory | 9 × 512-byte int8 vectors | Can store tool state, but not verbose tool outputs |
| ACT steps | 1-6 iterations | Can "think longer" about tool decisions |
| Vocab | 16,512 (128 reserved special) | 121 unused special token slots (7-127) |
| Processing | Streaming, per-token | Tool calls must be expressible as token sequences |
| NOOP | Token ID 6 (untrained) | Absorption mechanism exists but isn't active yet |
| Address space | 3 heads × 8-dim int8 | Learned, continuous — model discovers its own addressing |

---

## Approaches Evaluated

### 1. Memory-Mapped Tool I/O (★★★★★ Recommended — Primary Mechanism)

**The PCI/MMIO Analogy:**

In hardware, certain memory addresses don't go to RAM — they go to devices. The CPU doesn't know (or care) whether address `0xFFFF0000` is RAM or a GPU register. The memory controller intercepts and routes. The CPU learns through its firmware/OS which addresses do what.

Applied to LatentController: **the agent's learned address computation IS the bus, and memory IS the address space.** Tool "devices" live at specific locations in that address space. The agent discovers them the same way it discovers any useful memory — by learning that certain address patterns produce useful readbacks.

```
Agent's process_token() loop:
  ┌─────────────────────────────────────────────────┐
  │  1. h = current hidden state                     │
  │  2. addrs = compute_addresses(h)  ← LEARNED      │
  │  3. vecs = memory.read(addrs)     ← THE BUS      │
  │         ↓                                        │
  │    ┌── Memory Bus Controller ──┐                 │
  │    │                           │                 │
  │    ↓                           ↓                 │
  │  RAM (trie)              Tool Devices            │
  │  (normal vectors)        (intercepted I/O)       │
  │                                                   │
  │  4. forward(tokens, vecs) → logits, hidden       │
  │  5. memory.write(addrs, hidden) ← ALSO THE BUS  │
  │         ↓                                        │
  │    Write intercepted if addr matches device       │
  │    → triggers tool execution                      │
  │    → result written to device readback buffer     │
  │    → agent reads it on next cycle                 │
  └─────────────────────────────────────────────────┘
```

**The key insight you identified: the address space is learned, not designed.** The model's 3 address heads (each projecting d_model→8 int8) learn to organize information. We can't just say "address range 0x00-0x0F = bash tool" because the agent's address heads might naturally use those addresses for something else.

**Solution: Device Registration via the Trie Itself**

Instead of reserving address ranges, we register tool devices as **trie entries with special metadata.** The existing `TrieNode` already has a `flags` field (currently unused, always 0). We use it:

```python
class TrieNode:
    __slots__ = ("children", "record_number", "write_count", "flags")
    # flags: 0 = normal memory, 1 = tool device
```

When the agent reads from an address that resolves to a flagged node, the MemorySystem doesn't return a stored vector — it calls the registered device handler. When the agent writes to that address, the write is intercepted and routed to the device.

**How devices get discovered:**

During training (Phase 6), tool device vectors are pre-seeded at deterministic addresses (hashed from tool name). The ±1 neighborhood search in `find_nearest()` means the agent doesn't need to hit the exact address — it just needs to get *close*. Over training, the address heads learn that certain regions of the address space produce useful "tool-flavored" readbacks, and writing certain patterns there triggers external effects.

This is exactly how a CPU learns to use MMIO — the firmware (our training data) teaches which addresses are devices, and the CPU (our model) learns the access patterns.

**Implementation — MemorySystem as Memory Bus Controller:**

```python
class MemorySystem:
    def __init__(self, data_path, cfg):
        ...
        self._devices: Dict[int, ToolDevice] = {}  # record_number → device

    def register_device(self, addresses: List[bytes], device: 'ToolDevice'):
        """Mount a tool device at given addresses in the trie."""
        # Allocate a record slot for this device
        device_vec = device.descriptor_vector()  # (512,) int8 tool description
        rec_num = self._append_record(device_vec)
        for head_idx, addr in enumerate(addresses):
            self.indexes[head_idx].insert(addr, rec_num, 0)
            # Mark as device in trie
            node = self.indexes[head_idx].lookup(addr)
            node.flags = 1  # DEVICE flag
        self._devices[rec_num] = device

    def _read_record(self, record_number: int) -> np.ndarray:
        """Read — intercepts device records."""
        if record_number in self._devices:
            return self._devices[record_number].read()  # device readback
        return self._data_cache[record_number].copy()    # normal RAM

    def write_memory(self, addresses: List[bytes], vector: np.ndarray):
        """Write — intercepts if any resolved address is a device."""
        for head_idx, addr in enumerate(addresses):
            node = self.indexes[head_idx].lookup(addr)
            if node is not None and node.flags == 1:
                # DEVICE WRITE — route to tool handler
                device = self._devices[node.record_number]
                device.write(vector)  # triggers tool execution
                return  # device handles it, skip normal EMA blend
        # Normal memory write (existing logic)
        ...
```

**The ToolDevice interface:**

```python
class ToolDevice:
    """Base class for memory-mapped tool devices."""

    def descriptor_vector(self) -> np.ndarray:
        """512-byte int8 vector describing this tool. Pre-seeded into memory.
        The agent reads this to understand what the device does."""
        ...

    def read(self) -> np.ndarray:
        """Device readback — returns current state/result as 512-byte vector.
        Called when agent reads from this device's address."""
        ...

    def write(self, vector: np.ndarray):
        """Device write — agent sends a command/query as a hidden state vector.
        Triggers tool execution. Result available on next read()."""
        ...

class BashDevice(ToolDevice):
    def descriptor_vector(self):
        # Encode "I am a bash shell. Write hidden states to execute commands."
        return encode_description("bash: shell command execution")

    def write(self, vector: np.ndarray):
        # Decode the hidden state into a command
        # (requires a trained decoder or nearest-neighbor lookup)
        command = self.decode_intent(vector)
        result = subprocess.run(command, capture_output=True, text=True)
        self._result_vec = encode_result(result.stdout[:1000])

    def read(self) -> np.ndarray:
        return self._result_vec if hasattr(self, '_result_vec') else self.descriptor_vector()
```

**Why this is architecturally perfect:**

1. **Zero model changes.** The model already reads/writes memory. Tool devices are just memory locations that behave differently. The forward pass, address computation, ACT loop — nothing changes.

2. **The agent discovers tools organically.** During training, address heads learn that certain memory regions produce useful information (tool descriptors) and that writing to them causes the readback to change (tool results). This is identical to how a CPU discovers PCI devices during bus enumeration.

3. **Fully compatible with existing training.** Phases 1-5 train memory read/write. Phase 6 just adds device-backed entries. The agent uses the same mechanism it already learned — it just encounters new "types" of memory.

4. **Neighborhood search = device discovery.** The `find_nearest()` with ±1 on fine dims means the agent doesn't need perfect address targeting. Getting close to a device address brings it into the 9 memory slots. The agent sees the descriptor vector and learns what to do with it.

5. **No special tokens needed for tool invocation.** The model doesn't emit `<TOOL_CALL>` — it writes a hidden state to a device address. The tool executes. The result appears in memory on the next read. No token protocol overhead.

6. **ACT enables tool polling.** The ACT loop re-reads memory between iterations. After writing to a tool device, the agent can do another ACT step, re-read memory, and see the tool result — all within processing a single token.

**The deep implication: tools become part of the agent's learned world model.** The agent doesn't learn "when I need information, emit `<TOOL_CALL>`". It learns "when I need information, I know where to look in memory, and certain memory locations have answers that change based on what I write to them." This is fundamentally more powerful — it's how biological systems work with sensory organs.

**The challenge you correctly identified: decoding intent from hidden states.**

When the agent writes to a tool device, it writes a 512-d hidden state vector. How does the tool know what the agent *wants*? This is the inverse problem — going from latent representation back to executable intent.

**Options for intent decoding:**

**Option A: Train a decoder head (recommended for complex tools)**
```python
class ToolDecoder(nn.Module):
    """Small network trained alongside the model to decode hidden states into tool actions."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def decode(self, hidden: torch.Tensor) -> str:
        logits = self.proj(hidden)
        tokens = logits.argmax(dim=-1)  # greedy decode
        return detokenize(tokens)
```
Train this during Phase 6: when the model writes to a tool device, supervise the decoder with the intended command text.

**Option B: Cosine similarity to known commands (no training needed)**
```python
class CommandLookup:
    """Nearest-neighbor lookup in a bank of pre-encoded commands."""
    def __init__(self, commands: List[str], encoder):
        self.commands = commands
        self.embeddings = [encoder(cmd) for cmd in commands]

    def decode(self, vector: np.ndarray) -> str:
        sims = [cosine_sim(vector, emb) for emb in self.embeddings]
        return self.commands[np.argmax(sims)]
```

**Option C: The hidden state IS the query (for retrieval tools)**
For search/retrieval tools, the hidden state doesn't need decoding — it IS the query vector. Write it to the search device, the device does nearest-neighbor search in its index, returns results as a 512-byte summary vector.

**Option D: Hybrid — structured device protocol**
The 512-byte write vector is partitioned:
- Bytes 0-7: action code (int8 → 256 possible actions per device)
- Bytes 8-511: payload (the actual query/command content)

The device interprets the action code and payload. The agent learns to structure its writes this way through training.

---

### 2. Special Token Protocol (★★★★☆ Alternative — Simpler but Less Elegant)

**How it works:** The model emits special tokens that the orchestrator intercepts and routes to tool handlers. Tool results are re-injected as token sequences that flow through normal processing.

```
Model output stream:  ... the answer is <TOOL_CALL> <TOOL_BASH> ls -la <TOOL_END>
                                         ↓ (orchestrator intercepts)
                                         ↓ executes: bash("ls -la")
                                         ↓ 
Model input stream:   <TOOL_RESULT> file1.txt file2.txt <TOOL_END> ...
                      ↓ (model processes result tokens normally)
                      ↓ reads memory, ACT loop, writes memory
Model output stream:  There are two files...
```

**Token allocation (from reserved slots 7-127):**
```
 7: <TOOL_CALL>        — begins a tool invocation
 8: <TOOL_END>         — ends tool call / tool result
 9: <TOOL_RESULT>      — begins injected tool output
10: <TOOL_BASH>        — shell command tool
11: <TOOL_READ>        — file read tool
12: <TOOL_WRITE>       — file write tool
13: <TOOL_SEARCH>      — web/doc search tool
14: <TOOL_PYTHON>      — python eval tool
15: <TOOL_ERROR>       — tool execution failed
16-31: reserved for future tools
```

**Why it's simpler:**
- Tool arguments are explicit text tokens — no hidden-state decoding needed
- Easy to debug (you can read the tool call in the token stream)
- Straightforward training data (annotated text sequences)

**Why it's less elegant:**
- Consumes context window budget for tool syntax (`<TOOL_CALL> <TOOL_BASH> ... <TOOL_END>`)
- Breaks the streaming paradigm — model must emit a full command before execution
- Model needs to learn a "protocol" separate from natural language
- Doesn't leverage the existing memory system at all

**Verdict:** Good as a fallback or for tools that need explicit text arguments (bash commands). But the memory-mapped approach is more architecturally aligned.

---

### 3. Hybrid: Memory-Mapped Devices + Token Fallback (★★★★★ Best of Both)

**The real answer is both.** Memory-mapped I/O for learned, implicit tool use. Token protocol for explicit, user-visible tool invocations.

```
              ┌─────────── Agent ───────────┐
              │                              │
              │  Hidden state → Addresses    │
              │       ↓                      │
              │  Memory Read (9 slots)       │
              │    ├── normal memory vecs     │
              │    └── device readbacks ←──── Tool results (implicit)
              │       ↓                      │
              │  Transformer + ACT           │
              │       ↓                      │
              │  Two output paths:           │
              │    ├── Token emission ───────→ Text stream (may include
              │    │                            <TOOL_CALL> for explicit tools)
              │    └── Memory Write ────────→ May hit device address
              │                                (triggers implicit tools)
              └──────────────────────────────┘
```

**When memory-mapped tools fire:**
- Agent is processing text, its address computation drifts toward a tool device
- The device descriptor appears in the 9 memory slots (agent "sees" the tool)
- If the agent's hidden state carries relevant intent, it writes to the device address
- Tool executes, result appears on next memory read
- The user never sees this — it's internal cognition

**When token-protocol tools fire:**
- Agent explicitly emits `<TOOL_CALL> <TOOL_BASH> ls <TOOL_END>`
- Orchestrator intercepts, executes, injects result
- The user sees the tool call in the output stream
- Good for transparency, auditability, user-facing actions

**This maps to how humans use tools:**
- **Implicit** (memory-mapped): You don't announce "I'm going to use my visual cortex to read this text." Your brain routes information to the right processing centers automatically through learned neural pathways.
- **Explicit** (token protocol): "Let me look that up." You verbalize the intent, perform the action, report the result.

---

### 4. MCP as External Backend (★★★★☆ Orthogonal — Infrastructure Layer)

MCP is the right protocol for the *orchestrator-to-tool-servers* connection, regardless of how the model triggers tools. It's invisible to the model.

```
┌── Model ──┐    ┌── Memory Bus ──┐    ┌── Orchestrator ──┐    ┌── MCP ──┐
│            │    │                 │    │                   │    │         │
│ write(h,a) │───→│ device.write(h) │───→│ decode + dispatch │───→│ JSON-RPC│
│            │    │                 │    │                   │    │         │
│ read(a)  ←─│────│ device.read()  ←│────│ encode result    ←│────│ response│
└────────────┘    └─────────────────┘    └───────────────────┘    └─────────┘
```

The model never sees MCP. The tool device handler inside the MemorySystem translates between 512-byte int8 vectors and MCP JSON-RPC. This is exactly like how a device driver translates between memory-mapped registers and hardware protocols.

---

### 5. Bash Pipelining (★★★☆☆ One Implementation of a Device)

Bash pipelining is now just **one type of ToolDevice** — a `BashDevice` mounted at a learned address. The model doesn't know it's "calling bash" — it writes a hidden state to a memory address, and useful information comes back.

### 6. WASM Sandboxing (★★☆☆☆ Execution Backend)

WASM can sandbox the tool device handlers. The model and memory system don't care whether `BashDevice.write()` runs bash natively or in a WASM sandbox. Deferred to production.

---

## Recommended Architecture: Memory-Mapped Tool I/O

### The Complete Design

```
┌──────────────────────────────────────────────────────────┐
│                    MemorySystem (Bus Controller)           │
│                                                            │
│  Address Space (3 heads × 8-dim int8 trie):               │
│                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  RAM Region  │  │ Bash Device │  │Search Device│  ...  │
│  │  (learned)   │  │ flags=1     │  │ flags=1     │       │
│  │              │  │             │  │             │       │
│  │  Normal trie │  │ Pre-seeded  │  │ Pre-seeded  │       │
│  │  records     │  │ at known    │  │ at known    │       │
│  │              │  │ hash addrs  │  │ hash addrs  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                            │
│  read_memory(addrs):                                       │
│    For each addr → trie lookup → check flags               │
│      flags=0: return cached vector (RAM)                   │
│      flags=1: return device.read() (tool readback)         │
│                                                            │
│  write_memory(addrs, vec):                                 │
│    For each addr → trie lookup → check flags               │
│      flags=0: normal EMA blend (RAM)                       │
│      flags=1: device.write(vec) → trigger tool execution   │
│                                                            │
│  register_device(name, handler):                           │
│    hash(name) → deterministic addresses                    │
│    Insert into trie with flags=1                           │
│    Store descriptor vector                                 │
│                                                            │
│  Neighborhood search naturally discovers devices:          │
│    Agent doesn't need exact address → ±1 search finds it  │
└──────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Tools are memory devices, not vocabulary tokens (primary path).**

The agent already has a rich learned interface to memory. Tool invocation through memory writes is zero-overhead — no new tokens, no context budget, no protocol to learn. The model just needs to learn that certain memory regions respond to writes differently. This is a much smaller learning task than learning a token-based tool protocol.

**2. The address space is learned — devices are discovered, not hardcoded.**

We seed device addresses using deterministic hashes of tool names, but the agent's address heads are free to evolve. During training, the agent learns that addresses near the bash device produce useful results. The ±1 neighborhood search means approximate addressing works fine.

**3. Intent decoding is a separate concern from the model.**

The model writes a 512-d hidden state. A small decoder network (or cosine lookup) translates that into an executable command. This decoder is trained alongside the model in Phase 6 but is not part of the transformer itself.

**4. Token protocol remains available for explicit/auditable tool use.**

Some use cases need visible tool calls (e.g., user asks "search for X", agent should show it's searching). The `<TOOL_CALL>` token protocol handles this. Both mechanisms can coexist.

**5. ACT loop enables synchronous tool use within a single token.**

```
process_token(token_id):
  ACT step 1: read memory → see tool descriptor → write intent to device
  ACT step 2: re-read memory → device now returns result → process result
  ACT step 3: incorporate result → decide output → halt
```

The tool executes *between ACT steps*, within a single call to `process_token()`. No need to wait for the next token. This is like how a CPU can do a memory-mapped I/O read and use the result in the same instruction sequence.

---

## Implementation Plan

### Phase 6a: Memory Bus Infrastructure (no training needed)

1. Add `flags` field support to `TrieNode` (already exists, just unused)
2. Add `ToolDevice` base class and `register_device()` to `MemorySystem`
3. Modify `_read_record()` to check flags → route to device
4. Modify `write_memory()` to check flags → route to device
5. Implement `BashDevice`, `FileDevice`, `SearchDevice`
6. Intent decoder: start with cosine-similarity lookup

### Phase 6b: Tool-Aware Training Data

1. Generate sequences where the model writes to device addresses and reads useful results
2. Supervision signal: after device write, the model's next output should reflect the tool result
3. Intent decoder supervision: when model writes to bash device, the decoder's output should match the intended command

### Phase 6c: Train with Devices Active

1. Phase 6 training with tool devices mounted in memory
2. Model learns which memory regions are "active" (devices) vs "passive" (RAM)
3. Address heads learn to target device addresses when tool use would help
4. The decoder learns to translate hidden states → commands

### Phase 6d: MCP Backend + Token Protocol Fallback

1. Connect device handlers to MCP servers
2. Add token-protocol support for explicit tool calls
3. WASM sandboxing for production

---

## Comparison Matrix

| Approach | Architecture Alignment | Impl. Complexity | Training Cost | Transparency | Extensibility |
|---|---|---|---|---|---|
| **Memory-Mapped I/O** | ★★★★★ Perfect | Medium | Medium | Low (implicit) | ★★★★★ |
| **Special Tokens** | ★★★★☆ Good | Low | Medium | ★★★★★ High | ★★★★☆ |
| **Hybrid (both)** | ★★★★★ Perfect | Medium | Medium | Tunable | ★★★★★ |
| **MCP Backend** | N/A (orthogonal) | Medium | None | N/A | ★★★★★ |
| **Bash Pipelining** | ★★★☆☆ (one device) | Low | Medium | Medium | ★★☆☆☆ |
| **WASM Sandbox** | N/A (execution layer) | High | None | N/A | ★★★★☆ |

---

## Recommendation

**Memory-mapped tool I/O as the primary mechanism, with token protocol as the explicit fallback.** This:

1. Requires zero changes to the model architecture (uses existing memory read/write)
2. The agent discovers tools the same way it discovers useful memory — organically
3. ACT loop enables synchronous tool use within a single token
4. The TrieNode.flags field is already there, unused, waiting for this
5. Neighborhood search provides fuzzy device discovery (no exact addressing needed)
6. MCP servers as external backends, WASM sandboxing for production — both orthogonal to the model

**The PCI metaphor is the right metaphor.** The memory bus IS the tool-calling interface. The model doesn't need to learn a new protocol — it already speaks memory.

---

## Evolution: Daemon Agents as Device Controllers

The MMIO model above uses simple callback functions as device handlers. But the user identified a deeper insight: **what if the device controllers are themselves agents?**

### The Problem with Simple Callbacks

A `BashDevice.write(vector)` callback faces an unsolved problem: *how do you decode a 512-d latent vector into a bash command?* Options like cosine-similarity lookup or trained decoders are fragile — they assume the model's hidden state has a clean, decodable relationship to text. In practice, hidden states encode far more than the "intended command" — they encode context, uncertainty, alternatives.

### The Insight: Agents Already Speak Hidden State

LatentController agents process tokens and produce hidden states. They already *understand* the latent space — it's their native language. So instead of a dumb callback trying to decode a vector, **use another agent that shares the same memory and naturally reads the regions it cares about.**

### The Architecture: Daemon Agents on the Memory Bus

```
┌─────────────────── SHARED MEMORY (Trie) ──────────────────────┐
│                                                                 │
│  All agents share one memory. Each reads/writes freely.         │
│  No reserved regions. No flags. No MMIO.                        │
│  The address space is fully learned and emergent.               │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ region A  │  │ region B  │  │ region C  │  │ region D  │  ... │
│  │ (emerged  │  │ (emerged  │  │ (emerged  │  │ (emerged  │      │
│  │  during   │  │  during   │  │  during   │  │  during   │      │
│  │ training) │  │ training) │  │ training) │  │ training) │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│       ↑↓              ↑↓              ↑↓              ↑↓        │
│    Main Agent    Bash Daemon     Search Daemon   Code Daemon    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**No flags. No reserved addresses. No routing.** Every agent just reads and writes memory using its own learned address heads. The "tool invocation" emerges from shared addressing patterns that develop during multi-agent training.

### How It Works

```
Main Agent (general purpose):
  1. Processing user input: "what files are in /tmp?"
  2. Writes hidden state to memory (normal process_token flow)
     → This hidden state ENCODES the intent "list files in /tmp"
     → The address heads place it at addresses the agent learned for
       "things I need help with" / "file system queries"

Bash Daemon (specialist, runs in parallel):
  3. Continuously reads memory at its own learned addresses
     → Its address heads were trained to target the SAME regions
       that the main agent writes "shell-relevant" thoughts to
  4. Sees the hidden state, processes it through its own transformer
     → It was trained on (hidden_state, bash_command, result) triples
     → Outputs: "ls /tmp" → executes → gets result
  5. Writes result embedding back to memory
     → Addresses overlap with main agent's "file system" region

Main Agent (continues):
  6. Next token processing: reads memory again
     → The Bash Daemon's result is now in the 9 memory slots
       (because their address heads learned overlapping regions)
  7. Incorporates result, outputs: "There are two files in /tmp..."
```

### Why This Is Better Than MMIO Callbacks

| | MMIO Callbacks | Daemon Agents |
|---|---|---|
| **Intent decoding** | Hard (vector → text) | Natural (agent reads latent space) |
| **Discovery** | Pre-seeded addresses | Emergent from training |
| **Extensibility** | Register new devices | Train new daemons |
| **Complexity** | Need flags, routing | Just more agents sharing memory |
| **Memory layout** | Some reserved (devices) | Fully learned, no reserved regions |
| **Training** | Phase 6 only | Multi-agent from Phase 5 onward |
| **Communication** | Vector ↔ callback | Hidden state ↔ hidden state (native) |

### The Key Properties

**1. No orchestrator-level routing needed.**

The main agent doesn't emit `<TOOL_CALL>`. It doesn't write to a flagged address. It just *thinks*, and its thoughts land in memory. The daemon agents read from overlapping memory regions and react. The "routing" is learned — address heads of cooperating agents converge to shared regions during training.

**2. Communication is through the latent space, not tokens.**

Agents don't need to tokenize and detokenize to communicate. They share 512-d hidden state vectors through memory. This is far richer than text — a single vector can encode intent, context, confidence, and alternatives simultaneously. It's like the difference between two people talking vs. two brain regions sharing neural activation patterns.

**3. Tool output flows back through the same channel.**

The bash daemon doesn't need to "inject tokens" into the main agent's stream. It writes its result as a hidden state vector to memory. The main agent reads it naturally on the next cycle. No special injection mechanism. No context window consumed.

**4. ACT provides the synchronization primitive.**

```
Main Agent's process_token() on "what files are in /tmp?":
  ACT step 1: Read memory → no tool results yet → write "need file listing" to memory
  ACT step 2: Re-read memory → bash daemon wrote result → incorporate result
  ACT step 3: Output token with file listing information → halt
```

The ACT loop is the polling loop. The main agent keeps iterating until it has what it needs (or runs out of steps). The daemon agents run asynchronously, writing results that the main agent picks up on subsequent ACT steps or on subsequent tokens.

**5. Daemons can be heterogeneous.**

- **Same model, different training**: A LatentController fine-tuned on code
- **Same model, different emit_threshold**: A "quiet thinker" (emit_threshold=1.1) that only writes to memory
- **Different model entirely**: A small MLP that maps hidden states to API calls
- **Not a neural model at all**: A rule-based system that watches for address patterns and executes commands

### The Training Protocol

**Phase 6a: Multi-Agent Memory Communication**
```
Train two agents jointly:
  Agent A processes text, writes hidden states to memory
  Agent B reads from the same memory, must predict what A was processing
  Loss: B's predictions match A's context

This teaches address heads to converge — agents learn to write to
regions that other agents can find and read.
```

**Phase 6b: Daemon Specialization**
```
Train daemon agents on tool-specific data:
  - Bash daemon: (input_context_embedding, shell_command, output_embedding)
  - Search daemon: (query_embedding, search_results_embedding)
  - Code daemon: (specification_embedding, code_embedding)

Each daemon learns to:
  1. Read memory for "requests" relevant to its specialty
  2. Execute the tool (bash, search, etc.)
  3. Write useful result embeddings back to memory
```

**Phase 6c: Joint Multi-Agent Training**
```
Full pipeline:
  Main agent processes text → writes to memory
  Daemons read, execute tools, write results
  Main agent reads results → produces better output

End-to-end loss through the main agent's output quality.
Daemons are rewarded when their results improve main agent's output.
```

### Scheduling: How Daemons Run

**Option A: Round-Robin (simplest)**
```python
class DaemonScheduler:
    def tick(self):
        # After each main agent token, give each daemon one read-write cycle
        for daemon in self.daemons:
            daemon.scan_memory()  # read relevant regions
            daemon.maybe_act()    # execute tool if warranted
            daemon.write_results() # write back to shared memory
```

**Option B: Interrupt-on-Write (efficient)**
```python
class MemorySystem:
    def write_memory(self, addresses, vector):
        # Normal write
        ...
        # Notify subscribed daemons
        for daemon in self._watchers:
            if daemon.is_interested(addresses):
                daemon.on_memory_write(addresses, vector)
```

**Option C: Async with ACT Sync Points (most natural)**
```python
# Daemons run in background threads, writing results whenever ready.
# Main agent's ACT loop is the sync point — each re-read picks up
# any new results that daemons have written.

# Main agent:
for act_step in range(max_act_steps):
    mem_vecs = memory.read(my_addresses)  # picks up daemon results
    logits, halt, hidden = model(tokens, mem_vecs)
    memory.write(my_addresses, hidden)     # may trigger daemons
    if should_halt: break
    # Daemons may write results between now and next read
```

### Example: Full Token Processing Cycle

```
Input token: "files" (in context: "what files are in /tmp?")

Main Agent process_token("files"):
  ├─ Read memory: 9 slots, mostly general context
  ├─ Append "files" to context buffer
  ├─ ACT step 1:
  │   ├─ Forward pass → hidden state h₁ encodes "question about files in /tmp"
  │   ├─ h₁ has high activation in "file system" subspace
  │   ├─ Address heads compute addresses that land near "file system" region
  │   ├─ Write h₁ to memory → [async: bash daemon wakes up]
  │   ├─ halt_prob = 0.2 → continue
  │   └─ Re-read memory for next step
  │
  ├─ [Meanwhile: Bash daemon]
  │   ├─ Reads memory, finds h₁ in its interest region
  │   ├─ Its own forward pass interprets h₁ as "ls /tmp"
  │   ├─ Executes: subprocess.run(["ls", "/tmp"])
  │   ├─ Encodes result as hidden state h_result
  │   └─ Writes h_result to memory at overlapping addresses
  │
  ├─ ACT step 2:
  │   ├─ Re-read memory → h_result is now in the 9 slots!
  │   ├─ Forward pass with h_result in memory → hidden state h₂
  │   ├─ h₂ now encodes "the files are file1.txt, file2.txt"
  │   ├─ halt_prob = 0.7 → halt
  │   └─ Write h₂ to memory
  │
  └─ Emit: top token from h₂'s logits
```

### Connection to Existing Architecture

This builds on what already exists:

| Existing | Extension |
|---|---|
| `Agent.process_token()` | Unchanged — daemons are just more agents |
| `MemorySystem` shared between agents | Already works — `Orchestrator.pipe()` proves it |
| `Agent._read_memory()` / `_write_memory()` | Daemons use the same methods |
| `ACT loop` with memory re-reads | Natural sync point for daemon results |
| `orchestrator.background_consolidation()` | Already runs two agents on shared memory |
| `orchestrator.pipe(source, dest)` | Daemon-to-main-agent communication is similar |

The infrastructure is 80% there. What's missing:
1. **DaemonScheduler** — runs daemons in parallel with main agent
2. **Multi-agent training** — joint loss across main + daemon agents
3. **Tool execution layer** — daemons need actual bash/search/etc. backends
4. **Intent encoding/decoding** — daemons need to interpret and produce hidden states that other agents can understand (this is what Phase 6a trains)
