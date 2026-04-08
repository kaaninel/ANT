#!/usr/bin/env python3
"""
ANT — Inference and terminal chat interface.

Duplex streaming with per-token trie read/write.
Every generated token reads from trie (knowledge) and writes to trie (learning).

Usage:
    python inference.py                              # interactive chat
    python inference.py --checkpoint path/to/ckpt.pt # specific checkpoint
    python inference.py --prompt "What is Python?"   # single query
"""

import argparse
import os
import sys
import time
import re

import numpy as np
import torch

from config import ModelConfig, MemoryConfig
from model import ANT
from memory import MemorySystem
from data import (
    VOCAB, VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, ANS_ID, NOOP_ID,
    tokenize, detokenize, _random_timestamp,
    _TAG_REGISTRY, _DOMAIN_MAP,
)
from train import sliding_window_encode, trie_write, trie_read, generate


# ============================================================================
# ANSI Terminal Colors
# ============================================================================

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


# ============================================================================
# Chat Session
# ============================================================================

class ChatSession:
    """Interactive chat session with persistent trie memory.

    Every user message is encoded and written to trie.
    Every response token reads from trie for context.
    """

    def __init__(self, model: ANT, memory: MemorySystem, device: str,
                 window_size: int = 8, num_passes: int = 4):
        self.model = model
        self.memory = memory
        self.device = device
        self.window_size = window_size
        self.num_passes = num_passes
        self.turn_count = 0

    def respond(self, user_text: str, stream: bool = True,
                max_tokens: int = 512, temperature: float = 0.8) -> str:
        """Process user input and generate response.

        Flow:
          1. Encode user text → hidden states → trie WRITE
          2. Use user representation → trie READ → memory vectors
          3. Generate response with sliding window + memory cross-attention
          4. Write each generated token to trie (knowledge accumulates)
        """
        self.turn_count += 1
        self.model.eval()

        # Tag the input
        ts = _random_timestamp()
        user_tag = f"localhost/user/chat@{ts}"
        agent_tag = f"localhost/ant/chat@{ts}"

        tagged_input = f"{user_tag}: {user_text}"
        user_ids = tokenize(tagged_input)

        # 1. Encode user input → write to trie
        user_tensor = torch.tensor([user_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            user_hidden = sliding_window_encode(
                self.model, user_tensor,
                self.window_size, self.num_passes, causal=True)
            trie_write(self.model, self.memory, user_hidden, temperature=0.01)

        # 2. Read from trie using user representation
        user_pooled = user_hidden[:, -1, :]
        mem_k, mem_v, mem_m = trie_read(
            self.model, self.memory, user_pooled, self.device)

        # 3. Generate response with memory
        prompt_ids = tokenize(f"{agent_tag}: ")
        prompt = [BOS_ID] + prompt_ids

        collected = []

        def on_token(tok):
            if stream and tok >= 0x20:
                sys.stdout.write(chr(tok))
                sys.stdout.flush()
            collected.append(tok)

        gen_tokens = generate(
            self.model, prompt, max_new=max_tokens,
            temperature=temperature, top_k=40, top_p=0.9,
            repetition_penalty=1.1,
            window_size=self.window_size, num_passes=self.num_passes,
            mem_keys=mem_k, mem_vals=mem_v, mem_mask=mem_m,
            stop_strings=["\nlocalhost/", "\n\n"],
            callback=on_token,
        )

        if stream:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # 4. Write response to trie
        resp_ids = gen_tokens[len(prompt):]
        if resp_ids:
            resp_tensor = torch.tensor([resp_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                resp_hidden = sliding_window_encode(
                    self.model, resp_tensor,
                    self.window_size, self.num_passes, causal=True)
                trie_write(self.model, self.memory, resp_hidden, temperature=0.01)

        response = detokenize(resp_ids)

        # Strip the agent tag prefix if it leaked into the response
        if response.startswith(agent_tag):
            response = response[len(agent_tag):].lstrip(": ")

        return response

    def stats(self) -> str:
        return (f"Turns: {self.turn_count}  "
                f"Trie: {self.memory.total_entries()} entries, "
                f"{self.memory.total_nodes()} nodes")


# ============================================================================
# Interactive Chat Loop
# ============================================================================

def chat_loop(model, memory, device, window_size=8, num_passes=4):
    session = ChatSession(model, memory, device, window_size, num_passes)

    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  ANT — Interactive Chat{RESET}")
    print(f"{DIM}  Memory: {memory.total_entries()} entries, "
          f"{memory.total_nodes()} nodes{RESET}")
    print(f"{DIM}  Commands: /quit /stats /reset /save{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}\n")

    while True:
        try:
            user_input = input(f"{CYAN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                memory.flush()
                print(f"{DIM}Memory saved. Goodbye!{RESET}")
                break
            elif cmd == "/stats":
                print(f"{DIM}{session.stats()}{RESET}")
                continue
            elif cmd == "/reset":
                memory.reset()
                session.turn_count = 0
                print(f"{DIM}Memory cleared.{RESET}")
                continue
            elif cmd == "/save":
                memory.flush()
                print(f"{DIM}Memory flushed to disk.{RESET}")
                continue
            else:
                print(f"{DIM}Unknown command: {cmd}{RESET}")
                continue

        # Generate response
        sys.stdout.write(f"{GREEN}ant>{RESET} ")
        sys.stdout.flush()
        t0 = time.time()
        response = session.respond(user_input, stream=True)
        dt = time.time() - t0
        print(f"{DIM}  [{dt:.1f}s, {session.stats()}]{RESET}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ANT — Inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--memory_dir", type=str, default=None,
                        help="Path to memory directory (default: alongside checkpoint)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--num_passes", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
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

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            "checkpoints/train/checkpoint_best.pt",
            "checkpoints/train/checkpoint_latest.pt",
            "checkpoints/overnight/checkpoint_best.pt",
            "checkpoints/micro/chat/checkpoint_best.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"{RED}No checkpoint found. Train first: python train.py{RESET}")
        sys.exit(1)

    print(f"{DIM}Loading checkpoint: {ckpt_path}{RESET}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Build model from checkpoint config
    cfg = ModelConfig()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    model = ANT(cfg, use_checkpoint=False).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{DIM}Model: {n_params:,} params on {device}{RESET}")
    if "step" in ckpt:
        print(f"{DIM}Trained: step {ckpt['step']}, "
              f"QA={ckpt.get('qa_accuracy', ckpt.get('best_qa', 0)):.1%}{RESET}")

    # Memory
    mem_cfg = MemoryConfig()
    if args.memory_dir:
        mem_cfg.data_path = args.memory_dir
    else:
        mem_cfg.data_path = os.path.join(os.path.dirname(ckpt_path), "memory")
    memory = MemorySystem(mem_cfg)
    print(f"{DIM}Memory: {memory.total_entries()} entries at {mem_cfg.data_path}{RESET}")

    if args.prompt:
        # Single-shot mode
        session = ChatSession(model, memory, device,
                              args.window_size, args.num_passes)
        response = session.respond(args.prompt, stream=False,
                                   max_tokens=args.max_tokens,
                                   temperature=args.temperature)
        print(response)
        memory.flush()
    else:
        # Interactive mode
        chat_loop(model, memory, device, args.window_size, args.num_passes)


if __name__ == "__main__":
    main()
