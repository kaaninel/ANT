# Source Provenance — Spatiotemporal Tags

## Tag Format

All data processed by ANT is tagged with a spatiotemporal header:

```
host/agent/dataplane@ISO-timestamp: content
```

Examples:
```
localhost/user/chat@2026-04-08T12:00:00Z: Hello, how are you?
localhost/ant/chat@2026-04-08T12:00:01Z: I'm doing well!
shell/root@2026-04-01T14:22:33Z: ls -la /home/user
wiki/article@2025-06-15T08:30:00Z: The French Revolution began in 1789.
```

## How Tags Interact with Memory

Tags are processed as raw bytes — no special embedding or structured parsing.
The tag is just more input to the byte-level transformer. The model learns
tag structure from data.

At each token (including tag bytes):
1. AddrNets generate 3 addresses → trie WRITE (tag content gets its own addresses)
2. Trie READ → cross-attention (previous knowledge informs interpretation)

Tag bytes and content bytes naturally get different trie addresses because
their hidden states differ. The model learns to:
- Route tag bytes to "metadata" regions of the address space
- Route content bytes to "fact" regions
- Correlate tags with content through cross-attention

### Tag System in TransformerBlock

Each layer has a persistent `tag_register` (GRU-style gated update):
```
new_tag = tanh(tag_head(x))
gate = sigmoid(tag_gate(x))
tag_register = gate * new_tag + (1 - gate) * tag_register
x = x + tag_register
```

This tracks the current speaker/context/mode across tokens. When a tag
boundary is detected (e.g., speaker change), the gate opens and updates
the register. Between tags, the gate stays mostly closed, preserving context.

## Byte Budget

```
Component         Bytes    Example
─────────────     ─────    ───────────────────────────────
Host              5-12     localhost, shell, wiki
Slash             1        /
Agent             3-15     user, ant, root, article
Slash             1        /
Dataplane         3-8      chat, shell, wiki, qa
At-sign           1        @
ISO 8601          20       2026-03-06T12:32:17Z
Colon+space       2        ": "
─────────────     ─────
TOTAL             36-60    typical ~45 bytes
```

Since the sliding window is 8 bytes with 4 passes (~16 bytes local context),
tags are too long for local context alone. This is by design — tag information
is stored in the trie and recalled via cross-attention, which has no distance
limit.

## Design Decisions

### Why text prefix (not structured embedding)?
1. No architecture changes — prefix is just more bytes
2. Self-supervised — model learns tag structure from data
3. Composable — tag format can evolve without retraining
4. Inspectable — can read raw bytes from memory
5. Consistent with byte-level design

### Why store all versions?
1. Truth is context-dependent
2. Temporal reasoning (newer isn't always better)
3. Source reliability (let queries judge credibility)
4. Contradictions contain more information than consensus

### Future: Contradiction Resolution
When contradicting facts exist in the trie (same entity, different values
from different sources), the model should:
- Store ALL versions at different addresses
- Use tag context to select the appropriate version
- Accept multiple valid answers when source is unspecified
