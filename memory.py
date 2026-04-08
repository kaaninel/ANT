"""
Hierarchical trie-based persistent memory system.

Architecture:
  - Each trie node stores an EMA value vector (d_model × float32)
  - 256 bins per level, up to depth_cap levels (default 8)
  - Writes propagate EMA to all ancestor levels with decay 1/√(depth_diff+1)
  - Reads collect the full ancestor path (root→leaf) = up to depth+1 vectors

Storage: single flat binary file.
  Header (12 bytes): n_nodes(u32), d_model(u32), n_records(u32)
  Values: n_nodes × d_model × float32
  Write counts: n_nodes × uint32
  Adjacency: per node → n_children(u16) + n_children × (bin_id(u8) + child_idx(u32))
"""

import math
import os
import struct
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import MemoryConfig


# ---------------------------------------------------------------------------
# TrieNode — each node stores an EMA value vector
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ("children", "value", "write_count")

    def __init__(self, d_model: int = 128):
        self.children: Dict[int, 'TrieNode'] = {}
        self.value: Optional[np.ndarray] = None  # float32, shape (d_model,)
        self.write_count: int = 0


# ---------------------------------------------------------------------------
# HierarchicalTrie — the core memory structure
# ---------------------------------------------------------------------------

class HierarchicalTrie:
    """Trie with EMA value vectors at every node.

    Each node stores an accumulated representation of all writes that
    pass through it. Leaf nodes store exact values; ancestor nodes store
    decayed EMA summaries of their descendants.
    """

    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.root = TrieNode(cfg.d_model)
        self.root.value = np.zeros(cfg.d_model, dtype=np.float32)
        self.n_records = 0

    def write(self, address: np.ndarray, value: np.ndarray,
              alpha_base: float = 0.1, alpha_min: float = 0.001):
        """Write a value at the given address, propagating EMA to ancestors.

        address: (depth,) int array of bin indices [0..255]
        value:   (d_model,) float32 vector (already projected by V_proj)
        """
        depth = len(address)
        if depth == 0:
            return

        # Collect the path from root to leaf, creating nodes as needed
        path = [self.root]
        node = self.root
        for bin_idx in address:
            b = int(bin_idx)
            if b not in node.children:
                node.children[b] = TrieNode(self.d_model)
                node.children[b].value = np.zeros(self.d_model, dtype=np.float32)
            node = node.children[b]
            path.append(node)

        # Write leaf with full weight
        leaf = path[-1]
        leaf.write_count += 1
        alpha_leaf = max(alpha_base / (1.0 + 0.01 * leaf.write_count), alpha_min)
        if leaf.write_count == 1:
            leaf.value = value.copy()
        else:
            leaf.value = (1 - alpha_leaf) * leaf.value + alpha_leaf * value

        # Propagate EMA to ancestors with decay 1/√(depth_diff+1)
        # path[0] = root, path[-1] = leaf at depth D
        for i in range(len(path) - 1):  # i=0 is root, i=D-1 is parent of leaf
            ancestor = path[i]
            depth_diff = depth - i  # distance from this ancestor to leaf
            decay = 1.0 / math.sqrt(depth_diff + 1)

            # Coarser levels use slower EMA (smaller effective alpha)
            coarse_factor = 1.0 / (i + 1)  # root gets smallest alpha
            alpha = max(alpha_base * decay * coarse_factor, alpha_min)

            ancestor.write_count += 1
            ancestor.value = (1 - alpha) * ancestor.value + alpha * value

        self.n_records += 1

    def read_path(self, address: np.ndarray) -> List[np.ndarray]:
        """Read the full ancestor path for an address.

        Returns list of vectors from root to the deepest matching node.
        If the exact address doesn't exist, returns path to deepest existing prefix.
        """
        vectors = []
        if self.root.value is not None:
            vectors.append(self.root.value)

        node = self.root
        for bin_idx in address:
            b = int(bin_idx)
            child = node.children.get(b)
            if child is None:
                break
            node = child
            if node.value is not None:
                vectors.append(node.value)

        return vectors

    def total_nodes(self) -> int:
        """Count total nodes in the trie (for diagnostics)."""
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count

    def depth_stats(self) -> Dict[int, int]:
        """Count nodes at each depth level."""
        stats: Dict[int, int] = {}
        stack: List[Tuple[TrieNode, int]] = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            stats[depth] = stats.get(depth, 0) + 1
            for child in node.children.values():
                stack.append((child, depth + 1))
        return stats


# ---------------------------------------------------------------------------
# MemorySystem — high-level interface wrapping the trie
# ---------------------------------------------------------------------------

class MemorySystem:
    """Persistent hierarchical memory for ANT.

    Provides read/write operations that interface with the model's AddrNets
    and V_proj. Handles serialization and mmap'd storage.
    """

    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.trie = HierarchicalTrie(cfg)
        self._lock = threading.Lock()
        self._write_count = 0

        os.makedirs(cfg.data_path, exist_ok=True)
        self._bin_path = os.path.join(cfg.data_path, "memory.bin")

        if os.path.exists(self._bin_path):
            self._load()

    def write(self, addresses: List[np.ndarray], value: np.ndarray):
        """Write a value to multiple address paths (one per AddrNet).

        addresses: list of N_addr_nets arrays, each (depth,) int
        value:     (d_model,) float32 vector from V_proj
        """
        with self._lock:
            for addr in addresses:
                self.trie.write(
                    addr, value,
                    alpha_base=self.cfg.ema_alpha_base,
                    alpha_min=self.cfg.ema_alpha_min
                )
            self._write_count += 1

            if self._write_count % self.cfg.flush_interval == 0:
                self._save()

    def write_batch(self, batch_addresses: List[List[np.ndarray]],
                    batch_values: np.ndarray):
        """Batch write: B items, each with N addr paths.

        batch_addresses: [B][N_addr_nets] arrays of (depth,) int
        batch_values:    (B, d_model) float32
        """
        with self._lock:
            for b in range(len(batch_addresses)):
                value = batch_values[b]
                for addr in batch_addresses[b]:
                    self.trie.write(
                        addr, value,
                        alpha_base=self.cfg.ema_alpha_base,
                        alpha_min=self.cfg.ema_alpha_min
                    )
            self._write_count += len(batch_addresses)

    def read(self, addresses: List[np.ndarray],
             max_vectors: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """Read memory for multiple address paths.

        Collects full ancestor path from each AddrNet, deduplicates root,
        pads/truncates to max_vectors.

        Returns:
            vectors: (max_vectors, d_model) float32
            mask:    (max_vectors,) bool — True for valid slots
        """
        all_vectors = []

        # Collect paths from all address nets
        root_added = False
        for addr in addresses:
            path = self.trie.read_path(addr)
            if not root_added and len(path) > 0:
                all_vectors.extend(path)  # include root from first path
                root_added = True
            elif len(path) > 1:
                all_vectors.extend(path[1:])  # skip root (already added)
            elif len(path) == 1 and not root_added:
                all_vectors.extend(path)
                root_added = True

        # Pad or truncate
        n = len(all_vectors)
        vectors = np.zeros((max_vectors, self.cfg.d_model), dtype=np.float32)
        mask = np.zeros(max_vectors, dtype=bool)

        for i in range(min(n, max_vectors)):
            vectors[i] = all_vectors[i]
            mask[i] = True

        return vectors, mask

    def read_batch(self, batch_addresses: List[List[np.ndarray]],
                   max_vectors: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """Batch read: B items.

        Returns:
            vectors: (B, max_vectors, d_model) float32
            mask:    (B, max_vectors) bool
        """
        B = len(batch_addresses)
        vectors = np.zeros((B, max_vectors, self.cfg.d_model), dtype=np.float32)
        mask = np.zeros((B, max_vectors), dtype=bool)

        for b in range(B):
            v, m = self.read(batch_addresses[b], max_vectors)
            vectors[b] = v
            mask[b] = m

        return vectors, mask

    def reset(self):
        """Clear all memory."""
        with self._lock:
            self.trie = HierarchicalTrie(self.cfg)
            self._write_count = 0

    def total_entries(self) -> int:
        return self.trie.n_records

    def total_nodes(self) -> int:
        return self.trie.total_nodes()

    def flush(self):
        """Force save to disk."""
        with self._lock:
            self._save()

    # ------------------------------------------------------------------
    # Binary serialization
    # ------------------------------------------------------------------
    # Format: Header(12B) | Values(N×D×4B) | WriteCounts(N×4B) | Adjacency
    # Header: n_nodes(u32) d_model(u32) n_records(u32)
    # Adjacency per node: n_children(u16) + n_children × (bin_id(u8) child_idx(u32))

    _HEADER = struct.Struct('<III')  # 3 × uint32 = 12 bytes

    def _save(self):
        """Serialize trie to a single flat binary file."""
        node_list: List[TrieNode] = []
        child_map: List[List[Tuple[int, int]]] = []  # [(bin_id, child_idx)]

        def _collect(node: TrieNode) -> int:
            idx = len(node_list)
            node_list.append(node)
            child_map.append([])  # placeholder
            for b in sorted(node.children.keys()):
                ci = _collect(node.children[b])
                child_map[idx].append((b, ci))
            return idx

        _collect(self.trie.root)

        n = len(node_list)
        d = self.cfg.d_model

        with open(self._bin_path, 'wb') as f:
            # Header
            f.write(self._HEADER.pack(n, d, self.trie.n_records))

            # Values: contiguous float32 block
            vals = np.zeros((n, d), dtype=np.float32)
            for i, node in enumerate(node_list):
                if node.value is not None:
                    vals[i] = node.value
            f.write(vals.tobytes())

            # Write counts
            wc = np.array([node.write_count for node in node_list], dtype=np.uint32)
            f.write(wc.tobytes())

            # Adjacency
            for i in range(n):
                children = child_map[i]
                f.write(struct.pack('<H', len(children)))
                for bin_id, ci in children:
                    f.write(struct.pack('<BI', bin_id, ci))

    def _load(self):
        """Deserialize trie from flat binary file."""
        try:
            with open(self._bin_path, 'rb') as f:
                # Header
                hdr = f.read(self._HEADER.size)
                if len(hdr) < self._HEADER.size:
                    return
                n, d, n_records = self._HEADER.unpack(hdr)
                if n == 0 or d != self.cfg.d_model:
                    return

                # Values
                val_bytes = f.read(n * d * 4)
                vals = np.frombuffer(val_bytes, dtype=np.float32).reshape(n, d).copy()

                # Write counts
                wc_bytes = f.read(n * 4)
                wc = np.frombuffer(wc_bytes, dtype=np.uint32)

                # Create nodes
                nodes: List[TrieNode] = []
                for i in range(n):
                    tn = TrieNode(d)
                    tn.value = vals[i]
                    tn.write_count = int(wc[i])
                    nodes.append(tn)

                # Adjacency
                for i in range(n):
                    nc_bytes = f.read(2)
                    nc = struct.unpack('<H', nc_bytes)[0]
                    for _ in range(nc):
                        entry = f.read(5)  # 1 byte bin_id + 4 byte child_idx
                        bin_id, ci = struct.unpack('<BI', entry)
                        nodes[i].children[bin_id] = nodes[ci]

                self.trie.root = nodes[0]
                self.trie.n_records = n_records
        except (OSError, struct.error):
            return
