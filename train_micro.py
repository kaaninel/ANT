#!/usr/bin/env python3
"""
Micro Prototype — ~1M param memory-recall experiment.

Self-contained: builds tokenizer, generates dataset, trains, evaluates.
Proves memory architecture works on extractive QA before scaling.
Runs on MPS/CPU in ~30 minutes.

Usage:
    python train_micro.py                    # full pipeline
    python train_micro.py --eval_only        # eval last checkpoint
    python train_micro.py --device cpu       # force CPU
"""

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import MicroModelConfig, MemoryConfig
from model import LoopedLatentController
from memory import MemorySystem
from agent import addr_bytes, memory_vecs_to_tensor

# ============================================================================
# Micro Tokenizer — simple word-level, ~200 tokens
# ============================================================================

SPECIAL_TOKENS = {
    "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
    "<mem_start>": 4, "<mem_end>": 5, "<noop>": 6,
}

NAMES = [
    "mary", "john", "daniel", "sandra", "fred",
    "bill", "julie", "emma", "bob", "alice",
]
LOCATIONS = [
    "garden", "kitchen", "office", "bedroom", "bathroom",
    "hallway", "park", "school", "cinema", "library",
]
VERBS = ["went", "moved", "journeyed", "travelled", "ran", "walked"]
PREPOSITIONS = ["to", "the", "in", "is", "a"]
QUESTION_WORDS = ["where", "?"]
PUNCTUATION = ["."]
CONNECTORS = ["then", "after", "that"]
ANSWER_MARKER = ["<ans>"]  # marks start of answer region

def build_vocab():
    """Build word→id and id→word mappings."""
    vocab = dict(SPECIAL_TOKENS)
    idx = len(vocab)
    for group in [NAMES, LOCATIONS, VERBS, PREPOSITIONS,
                  QUESTION_WORDS, PUNCTUATION, CONNECTORS, ANSWER_MARKER]:
        for w in group:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab

VOCAB = build_vocab()
ID2WORD = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def tokenize(text: str) -> list[int]:
    """Simple whitespace+punctuation tokenizer."""
    # Split keeping punctuation as separate tokens
    text = text.replace(".", " .").replace("?", " ?")
    words = text.lower().split()
    return [VOCAB.get(w, VOCAB["<unk>"]) for w in words]


def detokenize(ids: list[int]) -> str:
    words = [ID2WORD.get(i, "<unk>") for i in ids
             if i not in (VOCAB["<pad>"], VOCAB["<bos>"], VOCAB["<eos>"],
                          VOCAB["<noop>"], VOCAB["<ans>"])]
    text = " ".join(words)
    text = text.replace(" .", ".").replace(" ?", "?")
    return text


# ============================================================================
# Dataset Generator — bAbI-style extractive QA
# ============================================================================

@dataclass
class QAExample:
    """A single memory-recall QA example."""
    passage: str       # "Mary went to the garden . John went to the kitchen ."
    question: str      # "Where is Mary ?"
    answer: str        # "garden"
    answer_entity: str # the name being asked about
    facts: dict        # {"mary": "garden", "john": "kitchen"}


def generate_single_fact() -> QAExample:
    """One person, one location."""
    name = random.choice(NAMES)
    loc = random.choice(LOCATIONS)
    verb = random.choice(VERBS)
    passage = f"{name} {verb} to the {loc} ."
    question = f"where is {name} ?"
    return QAExample(passage, question, loc, name, {name: loc})


def generate_two_facts() -> QAExample:
    """Two people, ask about one."""
    names = random.sample(NAMES, 2)
    locs = random.sample(LOCATIONS, 2)
    verb1, verb2 = random.choice(VERBS), random.choice(VERBS)
    passage = f"{names[0]} {verb1} to the {locs[0]} . {names[1]} {verb2} to the {locs[1]} ."
    target = random.randint(0, 1)
    question = f"where is {names[target]} ?"
    return QAExample(passage, question, locs[target], names[target],
                     dict(zip(names, locs)))


def generate_three_facts() -> QAExample:
    """Three people, ask about one."""
    names = random.sample(NAMES, 3)
    locs = random.sample(LOCATIONS, 3)
    parts = []
    for n, l in zip(names, locs):
        v = random.choice(VERBS)
        parts.append(f"{n} {v} to the {l} .")
    passage = " ".join(parts)
    target = random.randint(0, 2)
    question = f"where is {names[target]} ?"
    return QAExample(passage, question, locs[target], names[target],
                     dict(zip(names, locs)))


def generate_temporal() -> QAExample:
    """Person moves twice — answer is LAST location."""
    name = random.choice(NAMES)
    loc1, loc2 = random.sample(LOCATIONS, 2)
    v1, v2 = random.sample(VERBS, 2)
    passage = f"{name} {v1} to the {loc1} . then {name} {v2} to the {loc2} ."
    question = f"where is {name} ?"
    return QAExample(passage, question, loc2, name, {name: loc2})


def generate_distractor() -> QAExample:
    """Two people, one moves twice. Ask about either."""
    n1, n2 = random.sample(NAMES, 2)
    l1, l2, l3 = random.sample(LOCATIONS, 3)
    v1, v2, v3 = random.choices(VERBS, k=3)
    passage = (f"{n1} {v1} to the {l1} . {n2} {v2} to the {l2} . "
               f"then {n1} {v3} to the {l3} .")
    facts = {n1: l3, n2: l2}
    target = random.choice([n1, n2])
    question = f"where is {target} ?"
    return QAExample(passage, question, facts[target], target, facts)


GENERATORS = [
    (generate_single_fact, 0.15),
    (generate_two_facts, 0.30),
    (generate_three_facts, 0.20),
    (generate_temporal, 0.20),
    (generate_distractor, 0.15),
]


def generate_dataset(n: int, seed: int = 42) -> list[QAExample]:
    """Generate n QA examples with weighted type distribution."""
    random.seed(seed)
    gens, weights = zip(*GENERATORS)
    examples = []
    for _ in range(n):
        gen = random.choices(gens, weights=weights, k=1)[0]
        examples.append(gen())
    return examples


# ============================================================================
# PyTorch Datasets
# ============================================================================

class ContextQADataset(Dataset):
    """
    For warmup/LM training: passage + question + answer all in context.
    Shifted autoregressive format:
      full = <bos> passage <ans> question answer <eos>
      inp  = full[:-1]
      tgt  = [PAD...context...] answer <eos>  (PAD=ignored in loss)
    """
    def __init__(self, examples: list[QAExample], max_len: int = 128):
        self.samples = []
        pad = VOCAB["<pad>"]
        bos = VOCAB["<bos>"]
        eos = VOCAB["<eos>"]
        ans_marker = VOCAB["<ans>"]

        for ex in examples:
            passage_ids = tokenize(ex.passage)
            question_ids = tokenize(ex.question)
            answer_ids = tokenize(ex.answer)

            full_seq = [bos] + passage_ids + [ans_marker] + question_ids + answer_ids + [eos]
            inp_ids = full_seq[:-1]
            # Only compute loss on answer tokens — use pad_id for context so they're ignored
            n_context = 1 + len(passage_ids) + 1 + len(question_ids)
            tgt_ids = [pad] * (n_context - 1) + answer_ids + [eos]

            assert len(inp_ids) == len(tgt_ids), f"{len(inp_ids)} != {len(tgt_ids)}"

            if len(inp_ids) > max_len:
                inp_ids = inp_ids[:max_len]
                tgt_ids = tgt_ids[:max_len]
            while len(inp_ids) < max_len:
                inp_ids.append(pad)
                tgt_ids.append(pad)

            self.samples.append((
                torch.tensor(inp_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MemoryQADataset(Dataset):
    """
    For memory training: passage is ONLY in memory, not in context.
    Shifted autoregressive format:
      full    = <bos> <ans> question answer <eos>
      inp     = full[:-1]
      tgt     = [PAD...context...] answer <eos>  (PAD=ignored in loss)
      passage = padded passage tokens (for memory feeding)
    """
    def __init__(self, examples: list[QAExample], max_len: int = 128,
                 max_passage_len: int = 64):
        self.samples = []
        pad = VOCAB["<pad>"]
        bos = VOCAB["<bos>"]
        eos = VOCAB["<eos>"]
        ans_marker = VOCAB["<ans>"]

        for ex in examples:
            passage_ids = tokenize(ex.passage)
            question_ids = tokenize(ex.question)
            answer_ids = tokenize(ex.answer)

            full_seq = [bos, ans_marker] + question_ids + answer_ids + [eos]
            inp_ids = full_seq[:-1]
            n_context = 2 + len(question_ids)
            tgt_ids = [pad] * (n_context - 1) + answer_ids + [eos]

            assert len(inp_ids) == len(tgt_ids)

            if len(inp_ids) > max_len:
                inp_ids = inp_ids[:max_len]
                tgt_ids = tgt_ids[:max_len]
            while len(inp_ids) < max_len:
                inp_ids.append(pad)
                tgt_ids.append(pad)

            # Pad passage to fixed length for batching
            p_ids = passage_ids[:max_passage_len]
            while len(p_ids) < max_passage_len:
                p_ids.append(pad)

            self.samples.append((
                torch.tensor(inp_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
                torch.tensor(p_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Helpers
# ============================================================================

def hidden_to_int8(hidden: torch.Tensor) -> np.ndarray:
    arr = hidden.detach().float().cpu().numpy()
    scale = np.abs(arr).max()
    if scale < 1e-6:
        return np.zeros(arr.shape, dtype=np.int8)
    return np.clip(np.round(arr / scale * 127.0), -128, 127).astype(np.int8)


def batch_read_memory(model, hidden_states, memory, device):
    """Read memory for a batch using hidden states to compute addresses."""
    B = hidden_states.size(0)
    d = model.cfg.d_model
    addr_heads = model.compute_addresses_batch(hidden_states)
    addr_cpu = [h.cpu().numpy() for h in addr_heads]
    batch_addresses = []
    for b in range(B):
        sample_addrs = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
        batch_addresses.append(sample_addrs)
    mem_np = memory.read_memory_batch(batch_addresses)
    mem_tensor = torch.from_numpy(
        mem_np.astype(np.float32) / 127.0
    ).to(device, non_blocking=True)
    return mem_tensor


def write_memory_batch(model, hidden, memory, positions=None):
    """Write hidden states at specified positions to memory."""
    B, T, D = hidden.shape
    if positions is None:
        positions = [T - 1]  # default: write last position only

    n_writes = 0
    for pos in positions:
        h_batch = hidden[:, pos, :].detach()
        vecs_np = h_batch.float().cpu().numpy()
        addr_heads = model.compute_addresses_batch(h_batch)
        addr_cpu = [h.cpu().numpy() for h in addr_heads]
        for b in range(B):
            ab = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
            memory.write_memory(ab, vecs_np[b])
            n_writes += 1
    return n_writes


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Phase A: Warmup LM on QA patterns (passage in context)
# ============================================================================

def train_phase_a(model, cfg, device, examples, steps=500, lr=3e-4,
                  batch_size=64, max_len=128):
    """Train LM on QA format: passage + question → answer. No memory."""
    print("\n" + "=" * 60)
    print("  Phase A: Warmup LM (QA patterns, no memory)")
    print("=" * 60)

    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    loader_iter = iter(loader)

    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp, tgt = inp.to(device), tgt.to(device)
        lr_now = get_lr(step, 50, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        logits, halt_logits = model(inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  [A {step}/{steps}] loss={loss.item():.4f} lr={lr_now:.2e} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Phase A done in {elapsed:.0f}s, final loss={loss.item():.4f}")
    return loss.item()


# ============================================================================
# Phase B: Address head contrastive training
# ============================================================================

def train_phase_b(model, cfg, device, examples, steps=300, lr=1e-3,
                  batch_size=128, max_len=128):
    """
    Train address heads to produce distinct addresses for different entities.
    Same entity → similar address, different entity → different address.
    """
    print("\n" + "=" * 60)
    print("  Phase B: Address Head Contrastive Training")
    print("=" * 60)

    # Generate hidden states for various entities
    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Only train address heads
    for p in model.parameters():
        p.requires_grad_(False)
    for head in model.addr_heads:
        for p in head.parameters():
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for head in model.addr_heads for p in head.parameters()],
        lr=lr, weight_decay=0.01,
    )

    model.train()
    loader_iter = iter(loader)
    t0 = time.time()

    for step in range(1, steps + 1):
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp = inp.to(device)

        with torch.no_grad():
            _, _, hidden = model(inp, return_hidden=True)

        # Use last position hidden states
        h = hidden[:, -1, :]  # (B, d_model)
        B = h.size(0)

        # Compute addresses from all heads
        loss = torch.tensor(0.0, device=device)
        for head in model.addr_heads:
            raw = head(h)  # (B, addr_dim)
            # Contrastive: pull same-batch neighbors apart (diversity),
            # but encourage spread across address space
            # Use pairwise distance — maximize average distance
            dists = torch.cdist(raw.unsqueeze(0), raw.unsqueeze(0)).squeeze(0)  # (B, B)
            # Encourage large pairwise distances (negative = push apart)
            margin = 4.0
            spread_loss = F.relu(margin - dists).mean()

            # Entropy: encourage each dimension to use full range
            dim_std = raw.std(dim=0)
            target_std = 15.0
            entropy_loss = F.mse_loss(dim_std, torch.full_like(dim_std, target_std))

            loss = loss + spread_loss + 0.1 * entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for head in model.addr_heads for p in head.parameters()], 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  [B {step}/{steps}] loss={loss.item():.4f} ({elapsed:.0f}s)")

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad_(True)

    elapsed = time.time() - t0
    print(f"  Phase B done in {elapsed:.0f}s")


# ============================================================================
# Phase C: ACT Curriculum (learn when to think more)
# ============================================================================

def streaming_act_forward(model, inp, memory_system, mem_tensor,
                          max_steps, temperature, device):
    """ACT forward with soft halting and optional memory re-read."""
    B, T = inp.shape
    HALT = 1

    remaining = torch.ones(B, T, device=device)
    weighted_logits = None
    expected_steps = torch.zeros(B, T, device=device)
    halt_counts = Counter()

    for i in range(max_steps):
        logits, halt_logits, hidden = model(
            inp, memory_vectors=mem_tensor, return_hidden=True
        )
        halt_prob = F.softmax(halt_logits / max(temperature, 1e-6), dim=-1)[..., HALT]

        if i < max_steps - 1:
            w = remaining * halt_prob
        else:
            w = remaining

        if weighted_logits is None:
            weighted_logits = w.unsqueeze(-1) * logits
        else:
            weighted_logits = weighted_logits + w.unsqueeze(-1) * logits

        expected_steps = expected_steps + (i + 1) * w
        remaining = (remaining - w).clamp(min=0.0)
        halt_counts[i + 1] += (halt_prob > 0.5).sum().item()

        # Re-read memory between ACT steps
        if i < max_steps - 1 and memory_system is not None:
            h_last = hidden[:, -1, :].detach()
            mem_tensor = batch_read_memory(model, h_last, memory_system, device)

    return weighted_logits, expected_steps, halt_counts, hidden


def train_phase_c(model, cfg, device, examples, steps=500, lr=1e-4,
                  batch_size=64, max_len=128):
    """
    ACT curriculum: ramp max_steps and ponder cost.
    Uses context QA (passage in context) so model can learn to halt properly.
    """
    print("\n" + "=" * 60)
    print("  Phase C: ACT Curriculum")
    print("=" * 60)

    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    loader_iter = iter(loader)
    t0 = time.time()

    # Curriculum: (step_threshold, max_act, ponder_weight, temperature)
    curriculum = [
        (0,   2, 0.0,   1.0),
        (100, 2, 0.001, 1.0),
        (200, 4, 0.002, 0.5),
        (350, 4, 0.005, 0.1),
    ]

    def get_curriculum_params(step):
        max_act, pw, temp = 2, 0.0, 1.0
        for thresh, ma, p, t in curriculum:
            if step >= thresh:
                max_act, pw, temp = ma, p, t
        return max_act, pw, temp

    for step in range(1, steps + 1):
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp, tgt = inp.to(device), tgt.to(device)
        max_act, ponder_w, temperature = get_curriculum_params(step)

        lr_now = get_lr(step, 50, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        weighted_logits, expected_steps, halt_counts, _ = streaming_act_forward(
            model, inp, None, None, max_act, temperature, device
        )

        lm_loss = F.cross_entropy(
            weighted_logits.reshape(-1, weighted_logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )
        ponder_loss = ponder_w * expected_steps.mean() if ponder_w > 0 else 0.0
        loss = lm_loss + ponder_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            total_halt = sum(halt_counts.values())
            avg_halt = sum(k * v for k, v in halt_counts.items()) / max(total_halt, 1)
            print(f"  [C {step}/{steps}] loss={loss.item():.4f} "
                  f"act={max_act} halt={avg_halt:.1f} pw={ponder_w:.3f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Phase C done in {elapsed:.0f}s")


# ============================================================================
# Phase D: Memory QA — direct float injection (no trie, no int8)
# ============================================================================

def encode_passages_to_memory(model, passages, device, n_slots=9):
    """
    Encode passages → hidden states → sample positions as float memory vectors.
    Bypasses trie/int8 entirely — direct teacher-forcing for training.

    passages: (B, T_passage) padded token IDs
    Returns: (B, n_slots, d_model) float memory vectors
    """
    B = passages.size(0)
    d = model.cfg.d_model
    pad_id = VOCAB["<pad>"]

    # Add BOS prefix for proper model input
    bos = torch.full((B, 1), VOCAB["<bos>"], dtype=torch.long, device=device)
    p_inp = torch.cat([bos, passages.to(device)], dim=1)  # (B, T+1)

    with torch.no_grad():
        _, _, p_hidden = model(p_inp, return_hidden=True)  # (B, T+1, d_model)

    mem_vecs = torch.zeros(B, n_slots, d, device=device)

    for b in range(B):
        # Find non-pad positions (skip BOS at 0)
        p_ids = passages[b]
        content_len = (p_ids != pad_id).sum().item()
        if content_len == 0:
            continue

        # Sample evenly from content positions (offset +1 for BOS)
        content_positions = list(range(1, content_len + 1))
        if len(content_positions) >= n_slots:
            step = len(content_positions) / n_slots
            selected = [content_positions[int(i * step)] for i in range(n_slots)]
        else:
            # Repeat positions to fill all slots
            selected = content_positions * (n_slots // len(content_positions) + 1)
            selected = selected[:n_slots]

        for j, pos in enumerate(selected):
            mem_vecs[b, j] = p_hidden[b, pos].detach()

    return mem_vecs


def train_phase_d(model, cfg, device, train_examples, val_examples,
                  memory_dir, steps=3000, lr=1e-4, batch_size=32,
                  max_len=128, eval_interval=200):
    """
    Memory-dependent QA with direct float injection.
    Passage hidden states → memory vectors (no trie, no int8).
    Curriculum: warmup with context+memory → memory-only.
    """
    print("\n" + "=" * 60)
    print("  Phase D: Memory QA (direct injection, no trie)")
    print("=" * 60)

    d_model = cfg.d_model
    n_mem_slots = 9

    train_ds = MemoryQADataset(train_examples, max_len=max_len)
    val_ds = MemoryQADataset(val_examples, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    model.train()
    loader_iter = iter(train_loader)
    t0 = time.time()
    best_acc = 0.0

    # Curriculum phases:
    # D1 (0-30%): passage in both context AND memory (learn to attend to mem)
    # D2 (30-100%): memory only (force reliance on memory)
    context_fade_start = int(steps * 0.3)

    for step in range(1, steps + 1):
        try:
            inp, tgt, passages = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inp, tgt, passages = next(loader_iter)

        inp, tgt, passages = inp.to(device), tgt.to(device), passages.to(device)

        lr_now = get_lr(step, 200, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Encode passages → float memory vectors (no trie)
        model.eval()
        mem_vecs = encode_passages_to_memory(model, passages, device, n_mem_slots)
        model.train()

        # D1: Use ContextQA format (passage in context) + memory vectors
        # D2: Memory only (current inp already has no passage)
        if step < context_fade_start:
            # Build context+memory inputs: [BOS] passage [ANS] question answer [EOS]
            # We need to rebuild inp with passage in context
            ctx_inp_list, ctx_tgt_list = [], []
            bos_id, ans_id, eos_id, pad_id = (
                VOCAB["<bos>"], VOCAB["<ans>"], VOCAB["<eos>"], VOCAB["<pad>"]
            )
            for b in range(passages.size(0)):
                p_ids = [x for x in passages[b].tolist() if x != pad_id]
                # Extract question and answer from the memory-QA format inp/tgt
                # inp format: [BOS] [ANS] question... [pad...]
                # tgt format: [PAD] [PAD]... answer [EOS] [pad...]
                inp_b = inp[b].tolist()
                tgt_b = tgt[b].tolist()
                # Find answer tokens (non-pad in tgt)
                ans_tokens = [t for t in tgt_b if t != pad_id]
                # Find question tokens (between ANS marker and padding in inp)
                q_start = 2  # after BOS, ANS
                q_end = q_start
                while q_end < len(inp_b) and inp_b[q_end] != pad_id:
                    q_end += 1
                q_tokens = inp_b[q_start:q_end]

                ctx_seq = [bos_id] + p_ids + [ans_id] + q_tokens + ans_tokens
                ctx_inp = ctx_seq[:-1]
                n_ctx = 1 + len(p_ids) + 1 + len(q_tokens)
                ctx_tgt = [pad_id] * (n_ctx - 1) + ans_tokens
                assert len(ctx_inp) == len(ctx_tgt), \
                    f"len mismatch: {len(ctx_inp)} vs {len(ctx_tgt)}"

                # Pad/truncate
                if len(ctx_inp) > max_len:
                    ctx_inp = ctx_inp[:max_len]
                    ctx_tgt = ctx_tgt[:max_len]
                while len(ctx_inp) < max_len:
                    ctx_inp.append(pad_id)
                    ctx_tgt.append(pad_id)

                ctx_inp_list.append(ctx_inp)
                ctx_tgt_list.append(ctx_tgt)

            inp_d = torch.tensor(ctx_inp_list, dtype=torch.long, device=device)
            tgt_d = torch.tensor(ctx_tgt_list, dtype=torch.long, device=device)
        else:
            inp_d = inp
            tgt_d = tgt

        # Forward with memory vectors
        logits, halt_logits, hidden = model(
            inp_d, memory_vectors=mem_vecs, return_hidden=True
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_d.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            phase = "D1(ctx+mem)" if step < context_fade_start else "D2(mem-only)"
            print(f"  [{phase} {step}/{steps}] loss={loss.item():.4f} "
                  f"lr={lr_now:.2e} ({elapsed:.0f}s)")

        # Evaluation
        if step % eval_interval == 0 or step == steps:
            acc = evaluate_memory_qa(model, cfg, val_examples, device,
                                     max_examples=200)
            print(f"  [D Eval @ {step}] Accuracy: {acc:.1%} (best: {best_acc:.1%})")
            if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(memory_dir, "best_model.pt")
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "accuracy": acc,
                    "vocab": VOCAB,
                }, save_path)
                print(f"  ✓ New best! Saved to {save_path}")

    elapsed = time.time() - t0
    print(f"  Phase D done in {elapsed:.0f}s, best accuracy={best_acc:.1%}")
    return best_acc


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_memory_qa(model, cfg, examples, device, max_examples=200):
    """
    Evaluate memory recall with direct float injection (no trie).
    For each example: encode passage → hidden states → memory vectors → predict.
    """
    model.eval()
    d_model = cfg.d_model
    n_mem_slots = 9
    pad_id = VOCAB["<pad>"]

    correct = 0
    total = 0
    type_correct = Counter()
    type_total = Counter()

    for i, ex in enumerate(examples[:max_examples]):
        passage_ids = tokenize(ex.passage)

        # Encode passage → memory vectors
        p_inp = torch.tensor([[VOCAB["<bos>"]] + passage_ids],
                              dtype=torch.long, device=device)
        _, _, p_hidden = model(p_inp, return_hidden=True)

        # Sample positions from passage hidden states
        content_len = len(passage_ids)
        if content_len == 0:
            mem_vecs = torch.zeros(1, n_mem_slots, d_model, device=device)
        else:
            positions = list(range(1, content_len + 1))
            if len(positions) >= n_mem_slots:
                step = len(positions) / n_mem_slots
                selected = [positions[int(j * step)] for j in range(n_mem_slots)]
            else:
                selected = positions * (n_mem_slots // len(positions) + 1)
                selected = selected[:n_mem_slots]
            mem_vecs = torch.stack(
                [p_hidden[0, pos] for pos in selected]
            ).unsqueeze(0)  # (1, n_mem_slots, d_model)

        # Build question input
        question_ids = tokenize(ex.question)
        inp_ids = [VOCAB["<bos>"], VOCAB["<ans>"]] + question_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        # Forward with memory
        logits, halt_logits, _ = model(inp, memory_vectors=mem_vecs,
                                        return_hidden=True)

        pred_id = logits[0, -1, :].argmax().item()
        expected_id = VOCAB.get(ex.answer, -1)
        is_correct = pred_id == expected_id

        if is_correct:
            correct += 1
        total += 1

        n_facts = len(ex.facts)
        type_total[n_facts] += 1
        if is_correct:
            type_correct[n_facts] += 1

    accuracy = correct / max(total, 1)

    for n_facts in sorted(type_total.keys()):
        t_corr = type_correct[n_facts]
        t_tot = type_total[n_facts]
        print(f"    {n_facts}-fact: {t_corr}/{t_tot} = {t_corr/max(t_tot,1):.1%}")

    model.train()
    return accuracy


@torch.no_grad()
def evaluate_context_qa(model, cfg, examples, device, max_examples=200):
    """Evaluate with passage in context (no memory needed). Baseline test."""
    model.eval()
    correct = 0
    total = 0

    for ex in examples[:max_examples]:
        passage_ids = tokenize(ex.passage)
        question_ids = tokenize(ex.question)

        bos = VOCAB["<bos>"]
        ans_marker = VOCAB["<ans>"]
        inp_ids = [bos] + passage_ids + [ans_marker] + question_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        logits, _ = model(inp)
        pred_id = logits[0, -1, :].argmax().item()

        expected_id = VOCAB.get(ex.answer, -1)
        if pred_id == expected_id:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    model.train()
    return accuracy


def detailed_eval(model, cfg, examples, device, n=10):
    """Print detailed examples with direct float injection."""
    model.eval()
    d_model = cfg.d_model
    n_mem_slots = 9
    print("\n" + "-" * 60)
    print("  Detailed Examples")
    print("-" * 60)

    for i, ex in enumerate(examples[:n]):
        passage_ids = tokenize(ex.passage)

        # Encode passage → memory vectors
        p_inp = torch.tensor([[VOCAB["<bos>"]] + passage_ids],
                              dtype=torch.long, device=device)

        with torch.no_grad():
            _, _, p_hidden = model(p_inp, return_hidden=True)

        content_len = len(passage_ids)
        if content_len == 0:
            mem_vecs = torch.zeros(1, n_mem_slots, d_model, device=device)
        else:
            positions = list(range(1, content_len + 1))
            if len(positions) >= n_mem_slots:
                step = len(positions) / n_mem_slots
                selected = [positions[int(j * step)] for j in range(n_mem_slots)]
            else:
                selected = positions * (n_mem_slots // len(positions) + 1)
                selected = selected[:n_mem_slots]
            mem_vecs = torch.stack(
                [p_hidden[0, pos] for pos in selected]
            ).unsqueeze(0)

        # Question input
        question_ids = tokenize(ex.question)
        inp_ids = [VOCAB["<bos>"], VOCAB["<ans>"]] + question_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            # With memory
            logits_mem, halt_logits, _ = model(
                inp, memory_vectors=mem_vecs, return_hidden=True)
            # Without memory
            logits_no_mem, _ = model(inp)

        pred_mem = logits_mem[0, -1, :].argmax().item()
        pred_no_mem = logits_no_mem[0, -1, :].argmax().item()

        # Top-5 with memory
        top5_vals, top5_ids = logits_mem[0, -1, :].topk(5)
        top5_probs = F.softmax(top5_vals, dim=0)
        top5_words = [(ID2WORD.get(idx.item(), "?"), f"{p.item():.3f}")
                      for idx, p in zip(top5_ids, top5_probs)]

        halt_prob = F.softmax(halt_logits[0, -1, :], dim=-1)[1].item()

        # Mem vector norms
        mem_norms = mem_vecs[0].norm(dim=-1)
        avg_norm = mem_norms.mean().item()

        expected = ex.answer
        mark = "✓" if ID2WORD.get(pred_mem, "") == expected else "✗"

        print(f"\n  {mark} Example {i+1}:")
        print(f"    Passage:  {ex.passage}")
        print(f"    Question: {ex.question}")
        print(f"    Expected: {expected}")
        print(f"    With mem: {ID2WORD.get(pred_mem, '?')} | "
              f"No mem: {ID2WORD.get(pred_no_mem, '?')}")
        print(f"    Top-5:    {top5_words}")
        print(f"    P(halt):  {halt_prob:.3f} | Mem norm: {avg_norm:.2f}")

    model.train()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Micro Prototype Training")
    parser.add_argument("--device", default=None, help="Force device (cpu/mps/cuda)")
    parser.add_argument("--output_dir", default="./checkpoints/micro",
                        help="Where to save checkpoints")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--n_train", type=int, default=20000,
                        help="Number of training examples")
    parser.add_argument("--n_val", type=int, default=1000,
                        help="Number of validation examples")
    parser.add_argument("--phase_a_steps", type=int, default=500)
    parser.add_argument("--phase_b_steps", type=int, default=300)
    parser.add_argument("--phase_c_steps", type=int, default=500)
    parser.add_argument("--phase_d_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("  MICRO PROTOTYPE — Memory Recall Experiment")
    print("=" * 60)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = MicroModelConfig()
    cfg.vocab_size = VOCAB_SIZE  # override with actual vocab size
    print(f"  Device:     {device}")
    print(f"  Vocab:      {VOCAB_SIZE} words")
    print(f"  d_model:    {cfg.d_model}")
    print(f"  n_layers:   {cfg.n_layers}")
    print(f"  n_heads:    {cfg.n_heads}")
    print(f"  max_seq:    {cfg.max_seq_len}")

    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)
    n_params = count_params(model)
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Generate data
    print(f"\n  Generating {args.n_train} train + {args.n_val} val examples...")
    train_examples = generate_dataset(args.n_train, seed=args.seed)
    val_examples = generate_dataset(args.n_val, seed=args.seed + 1)

    # Show data stats
    type_counts = Counter()
    for ex in train_examples[:1000]:
        type_counts[len(ex.facts)] += 1
    print(f"  Type distribution (first 1000): {dict(type_counts)}")
    print(f"  Example: {train_examples[0].passage}")
    print(f"           {train_examples[0].question} → {train_examples[0].answer}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save vocab
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(VOCAB, f, indent=2)

    if args.eval_only:
        ckpt_path = os.path.join(args.output_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ERROR: No checkpoint at {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint (step={ckpt['step']}, acc={ckpt['accuracy']:.1%})")

        print("\n  Context QA (baseline, no memory needed):")
        ctx_acc = evaluate_context_qa(model, cfg, val_examples, device)
        print(f"  → {ctx_acc:.1%}")

        print("\n  Memory QA (passage from memory only):")
        mem_acc = evaluate_memory_qa(model, cfg, val_examples, device)
        print(f"  → {mem_acc:.1%}")

        detailed_eval(model, cfg, val_examples, device, n=10)
        return

    # ===== Full Training Pipeline =====
    t_start = time.time()

    # Phase A: Warmup LM
    loss_a = train_phase_a(model, cfg, device, train_examples,
                           steps=args.phase_a_steps, batch_size=64)

    # Quick context QA check
    print("\n  Context QA after Phase A:")
    ctx_acc_a = evaluate_context_qa(model, cfg, val_examples, device)
    print(f"  → {ctx_acc_a:.1%}")

    # Phase B: Address heads
    train_phase_b(model, cfg, device, train_examples,
                  steps=args.phase_b_steps, batch_size=128)

    # Phase C: ACT curriculum
    train_phase_c(model, cfg, device, train_examples,
                  steps=args.phase_c_steps, batch_size=64)

    # Quick context QA check
    print("\n  Context QA after Phase C:")
    ctx_acc_c = evaluate_context_qa(model, cfg, val_examples, device)
    print(f"  → {ctx_acc_c:.1%}")

    # Phase D: Memory QA (the real test)
    mem_dir = os.path.join(args.output_dir, "memory")
    best_acc = train_phase_d(
        model, cfg, device, train_examples, val_examples, mem_dir,
        steps=args.phase_d_steps, lr=1e-4, batch_size=32,
        eval_interval=200,
    )

    # Final report
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"  Total time:      {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Context QA:      {ctx_acc_c:.1%} (passage in context)")
    print(f"  Memory QA:       {best_acc:.1%} (passage from memory)")
    print(f"  Target:          >80% memory QA")
    print(f"  ACT fix:         halt bias [0,0] (50/50 init)")

    # Detailed eval
    detailed_eval(model, cfg, val_examples, device, n=10)

    # Save final report
    report = {
        "total_time_s": total_time,
        "n_params": n_params,
        "context_qa_acc": ctx_acc_c,
        "memory_qa_acc": best_acc,
        "phases": {
            "A": {"steps": args.phase_a_steps, "final_loss": loss_a},
            "B": {"steps": args.phase_b_steps},
            "C": {"steps": args.phase_c_steps, "context_qa": ctx_acc_c},
            "D": {"steps": args.phase_d_steps, "best_acc": best_acc},
        },
        "config": {
            "d_model": cfg.d_model, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "vocab_size": VOCAB_SIZE,
        },
        "device": device,
    }
    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
