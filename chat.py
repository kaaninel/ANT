#!/usr/bin/env python3
"""
ANT Terminal Canvas — Collaborative tagged event stream interface.

Every line is a spatiotemporal event: host/user/path@timestamp: content
The model reads and writes this stream natively.

Usage:
    python chat.py                              # auto-detect checkpoint
    python chat.py --checkpoint path/to/best.pt # specific checkpoint
    python chat.py --agent ant                  # set agent name
    python chat.py --host myserver              # set host name

Modes:
    Default text   → tagged as host/user/chat@now: content
    !command       → real shell execution, output tagged as host/shell/cwd@now:
    /commands      → system commands (help, config, memory, etc.)
"""

import argparse
import os
import platform
import re
import readline  # noqa: F401 — enables arrow keys in input()
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from config import ModelConfig, MemoryConfig
from model import ANT, StaticKVCache

from train_micro import (
    VOCAB, VOCAB_SIZE, ID2WORD, PAD_ID, BOS_ID, EOS_ID, ANS_ID,
    tokenize, detokenize,
    encode_sentence_frozen, sliding_lm_encode, sliding_generate,
    tag_text, tag_passage, _TAG_REGISTRY, _DOMAIN_MAP,
)


# ── Tag parsing ──────────────────────────────────────────────────────────────

_TAG_RE = re.compile(
    r'^([^/]+)/([^/]+)/([^@]*)@(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z): '
)

# ANSI codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"

_HOST_COLORS = {
    "localhost": "\033[35m",  # magenta
    "server1":  "\033[36m",  # cyan
    "wiki":     "\033[32m",  # green
    "news":     "\033[33m",  # yellow
    "cam1":     "\033[34m",  # blue
}

_USER_COLORS = {
    "shell": "\033[31m",     # red for shell
    "root":  "\033[31m",
}


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_tag(host: str, user: str, path: str) -> str:
    """Build a full tag string: host/user/path@timestamp: """
    return f"{host}/{user}/{path}@{_now_tag()}: "


def format_tagged_line(line: str) -> str:
    """Parse a tag and return a color-formatted line for display."""
    m = _TAG_RE.match(line)
    if not m:
        return line
    host, user, path, ts = m.group(1), m.group(2), m.group(3), m.group(4)
    content = line[m.end():]
    hcolor = _HOST_COLORS.get(host, "\033[37m")
    ucolor = _USER_COLORS.get(user, hcolor)
    return f"{_BOLD}{ucolor}{user}{_RESET}{_DIM}@{host}/{path}{_RESET} {content}"


def render_markdown_line(text: str) -> str:
    """Basic ANSI markdown rendering for a single line."""
    # Headers
    if text.startswith("### "):
        return f"{_BOLD}{text[4:]}{_RESET}"
    if text.startswith("## "):
        return f"{_BOLD}\033[4m{text[3:]}{_RESET}"
    if text.startswith("# "):
        return f"{_BOLD}\033[4m\033[1m{text[2:]}{_RESET}"
    # Code fences
    if text.startswith("```"):
        return f"{_DIM}{'─' * 40}{_RESET}"
    # Bullet lists
    if text.startswith("- ") or text.startswith("* "):
        return f"  {_BOLD}•{_RESET} {text[2:]}"
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', f'{_BOLD}\\1{_RESET}', text)
    # Inline code
    text = re.sub(r'`(.+?)`', f'{_DIM}\\1{_RESET}', text)
    return text


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    """Load ANT model from checkpoint."""
    print(f"  Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ModelConfig()
    cfg.vocab_size = VOCAB_SIZE

    if "window_size" in ckpt:
        cfg.chunk_size = ckpt["window_size"]
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    model = ANT(cfg, use_checkpoint=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    step = ckpt.get("step", "?")
    acc = ckpt.get("accuracy", 0)
    mode = ckpt.get("mode", "unknown")
    print(f"  Model: step={step}, QA={acc:.0%}, mode={mode}, "
          f"d={cfg.d_model}, L={cfg.n_layers}, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    return model, cfg, ckpt


# ── Generation ───────────────────────────────────────────────────────────────

def sample_next(logits, generated, temperature=0.8, top_k=50, top_p=0.9,
                repetition_penalty=1.2):
    """Sample next token from logits with configurable strategies."""
    next_logits = logits[0, -1, :].float()

    if repetition_penalty != 1.0:
        for prev_id in set(generated[-50:]):
            if next_logits[prev_id] > 0:
                next_logits[prev_id] /= repetition_penalty
            else:
                next_logits[prev_id] *= repetition_penalty

    if temperature > 0:
        next_logits = next_logits / temperature

    if top_k > 0:
        indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
        next_logits[indices_to_remove] = float('-inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_logits[indices_to_remove] = float('-inf')

    if temperature > 0:
        probs = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, 1).item()
    return next_logits.argmax().item()


@torch.no_grad()
def generate(model, cfg, prompt_ids, device, callback=None,
             max_new_tokens=2048, temperature=0.8, top_k=50, top_p=0.9,
             repetition_penalty=1.2,
             mem_keys=None, mem_vals=None, mem_mask=None,
             window_size=8, num_passes=4,
             # kept for API compatibility — always sliding, this param ignored
             use_sliding=True):
    """Generate tokens via causal sliding window and call callback(token_id) for each.

    ANT is trained exclusively with causal sliding windows — there is no
    non-sliding path. Output length is bounded only by max_new_tokens;
    the rolling context buffer makes generation O(1) per token.

    Returns the list of newly generated token IDs (not including prompt).
    """
    model.eval()

    # Stop if the model starts generating a new tag line (role hallucination)
    _STOP_STRINGS = ["\nlocalhost/", "\nkaaninel@", "\nant@", "\nuser@"]

    generated_so_far = []

    def _cb(tok):
        if tok != EOS_ID:
            generated_so_far.append(tok)
            if callback:
                callback(tok)

    all_ids = sliding_generate(
        model, prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        window_size=window_size,
        num_passes=num_passes,
        mem_keys=mem_keys,
        mem_vals=mem_vals,
        mem_mask=mem_mask,
        stop_token=EOS_ID,
        stop_strings=_STOP_STRINGS,
        callback=_cb,
    )
    return all_ids[len(prompt_ids):]


def encode_passage_to_memory(model, passage_text, device, tagged=False):
    """Encode a passage into memory vectors for QA."""
    if tagged:
        passage_text = tag_passage(passage_text)
    passage_ids = tokenize(passage_text)
    p_tensor = torch.tensor([passage_ids], dtype=torch.long, device=device)
    mem_keys, mem_vals, mem_mask = encode_sentence_frozen(model, p_tensor, device)
    return mem_keys, mem_vals, mem_mask


# ── Streaming renderer ───────────────────────────────────────────────────────

class StreamRenderer:
    """Accumulates generated bytes and renders tagged lines with ANSI formatting.

    Buffers output until a newline is seen (to detect and format tags),
    then flushes with format_tagged_line + render_markdown_line.
    """

    def __init__(self):
        self.line_buf = []
        self.all_bytes = []
        self._first_line = True

    def feed(self, token_id: int):
        """Process one generated byte token."""
        self.all_bytes.append(token_id)

        if token_id == ord('\n'):
            self._flush_line(newline=True)
        else:
            self.line_buf.append(token_id)
            # Flush after tag detection window (~50 chars) if no tag found
            if len(self.line_buf) > 55:
                text = bytes(self.line_buf).decode('utf-8', errors='replace')
                if not _TAG_RE.match(text):
                    # No tag — flush as raw content
                    rendered = render_markdown_line(text)
                    print(rendered, end='', flush=True)
                    self.line_buf = []

    def _flush_line(self, newline=True):
        if not self.line_buf:
            if newline:
                print(flush=True)
            return
        line = bytes(self.line_buf).decode('utf-8', errors='replace')
        self.line_buf = []

        # Parse tag, format, then render markdown on content
        m = _TAG_RE.match(line)
        if m:
            formatted = format_tagged_line(line)
            # Apply markdown rendering to the content portion
            content_start = line[m.end():]
            tag_display = formatted[:formatted.index(content_start)] if content_start in formatted else ""
            rendered_content = render_markdown_line(content_start)
            print(f"{tag_display}{rendered_content}", end='\n' if newline else '', flush=True)
        else:
            print(render_markdown_line(line), end='\n' if newline else '', flush=True)

    def finish(self):
        """Flush any remaining buffered content."""
        if self.line_buf:
            self._flush_line(newline=True)
        elif self.all_bytes and self.all_bytes[-1] != ord('\n'):
            print(flush=True)

    def get_text(self) -> str:
        return bytes(self.all_bytes).decode('utf-8', errors='replace')


# ── Terminal Canvas ──────────────────────────────────────────────────────────

HELP_TEXT = f"""
{_BOLD}ANT Terminal Canvas{_RESET} — Collaborative tagged event stream

{_BOLD}Input modes:{_RESET}
  text          Chat with the agent (tagged automatically)
  !command      Execute shell command (output fed to agent context)

{_BOLD}Commands:{_RESET}
  /help         Show this help
  /quit         Exit
  /config       Show current settings
  /temp <f>     Set temperature (0=greedy, default=0.7)
  /topk <n>     Set top-k (default=50)
  /topp <f>     Set top-p (default=0.9)
  /maxlen <n>   Set max generation length (default=2048, no hard cap)
  /rep <f>      Set repetition penalty (default=1.3)
  /remember <t> Store passage in memory
  /forget       Clear memory
  /memory       Show stored passages
  /ask <q>      Ask a question about stored passages
  /context      Show conversation context (last N events)
  /clear        Clear conversation context
  /agent <name> Switch agent persona

{_BOLD}Architecture:{_RESET}
  Causal sliding window + memory cross-attention. Your messages are encoded
  into memory for cross-attention access. Factual passages (via /remember)
  are also stored in memory. Output length is unbounded.
"""


class TerminalCanvas:
    """Main application state and loop."""

    def __init__(self, model, cfg, ckpt, device, host, username, agent_name,
                 window_size, num_passes, tagged=True):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.host = host
        self.username = username
        self.agent = agent_name
        self.window_size = window_size
        self.num_passes = num_passes
        self.tagged = tagged

        # Sampling params
        self.temperature = 0.7
        self.top_k = 50
        self.top_p = 0.9
        self.max_tokens = 2048   # unlimited output — sliding window handles any length
        self.rep_penalty = 1.3

        # Conversation context (list of tagged event strings)
        # Keep recent turns for prompt construction; sliding_generate's rolling
        # buffer handles generation context. 4096 bytes ≈ 20 conversation turns.
        self.context: list[str] = []
        self.max_context_bytes = 4096

        # Memory
        self.passages: list[str] = []
        self.mem_keys = None
        self.mem_vals = None
        self.mem_mask = None

        # Shell
        self.cwd = os.getcwd()

    def _user_tag(self, path="chat") -> str:
        return make_tag(self.host, self.username, path)

    def _agent_tag(self, path="chat") -> str:
        return make_tag(self.host, self.agent, path)

    def _shell_tag(self) -> str:
        # Use relative path from home, or absolute
        try:
            relpath = os.path.relpath(self.cwd, os.path.expanduser("~"))
            if relpath.startswith(".."):
                path = self.cwd.lstrip("/")
            else:
                path = relpath
        except ValueError:
            path = self.cwd.lstrip("/")
        return make_tag(self.host, "shell", path)

    def _context_ids(self) -> list[int]:
        """Build token IDs from recent conversation context."""
        joined = "\n".join(self.context)
        # Trim to fit context budget
        while len(joined.encode('utf-8')) > self.max_context_bytes and self.context:
            self.context.pop(0)
            joined = "\n".join(self.context)
        return tokenize(joined)

    def _generate_response(self, prompt_suffix="", qa_mode=False):
        """Generate agent response given current context.

        Always uses causal sliding window — ANT has no non-sliding path.
        qa_mode: if True, uses [BOS, ANS] + question format (matching training).

        For chat mode, the user's recent messages are encoded into memory
        via cross-attention, solving the receptive field limitation. The
        sliding window handles local coherence; memory provides global
        understanding of what was asked.
        """
        agent_tag = self._agent_tag()

        if qa_mode:
            last = self.context[-1] if self.context else ""
            m = _TAG_RE.match(last)
            question = last[m.end():] if m else last
            prompt_ids = [BOS_ID, ANS_ID] + tokenize(question)
            # QA uses passage memory (from /mem add)
            mem_keys = self.mem_keys
            mem_vals = self.mem_vals
            mem_mask = self.mem_mask
        else:
            # Chat mode: encode recent user messages into memory
            # Extract user content from the last few context turns
            user_contents = []
            for line in self.context[-6:]:
                m = _TAG_RE.match(line)
                if m and '/user/' in line[:60]:
                    content = line[m.end():]
                    if content.strip():
                        user_contents.append(content.strip())
            chat_context = ". ".join(user_contents) if user_contents else "hello"

            # Encode into memory for cross-attention
            # Truncate to 190 tokens: encode_sentence_frozen uses model()
            # which has RoPE precomputed for 203 positions max
            ctx_tok = tokenize(chat_context)[:190]
            ctx_ids = torch.tensor([ctx_tok],
                                   dtype=torch.long, device=self.device)
            with torch.no_grad():
                chat_mk, chat_mv, chat_mm = encode_sentence_frozen(
                    self.model, ctx_ids, self.device)

            # Merge with any passage memory (from /mem add)
            if self.mem_keys is not None:
                mem_keys = torch.cat([self.mem_keys, chat_mk], dim=1)
                mem_vals = torch.cat([self.mem_vals, chat_mv], dim=1)
                mem_mask = torch.cat([self.mem_mask, chat_mm], dim=1)
            else:
                mem_keys, mem_vals, mem_mask = chat_mk, chat_mv, chat_mm

            # Prompt: agent tag only (user question is in memory)
            prompt_ids = [BOS_ID] + tokenize(agent_tag + prompt_suffix)

        # Print agent label before streaming content
        agent_label = format_tagged_line(agent_tag + "...")
        agent_label = agent_label.rsplit("...", 1)[0]
        print(agent_label, end='', flush=True)

        renderer = StreamRenderer()

        t0 = time.time()
        tokens = generate(
            self.model, self.cfg, prompt_ids, self.device,
            callback=renderer.feed,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k, top_p=self.top_p,
            repetition_penalty=self.rep_penalty,
            mem_keys=mem_keys, mem_vals=mem_vals,
            mem_mask=mem_mask,
            window_size=self.window_size,
            num_passes=self.num_passes)
        renderer.finish()
        elapsed = time.time() - t0

        response_text = renderer.get_text()
        self.context.append(agent_tag + response_text)

        n = len(tokens)
        print(f"{_DIM}  [{n} tok, {elapsed:.1f}s, {n/max(elapsed,0.001):.0f} tok/s]{_RESET}")

    def handle_shell(self, cmd: str):
        """Execute a shell command and feed output into context."""
        shell_tag = self._shell_tag()
        print(f"{format_tagged_line(shell_tag + cmd)}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=self.cwd)
            output = result.stdout
            if result.stderr:
                output += result.stderr
            output = output.strip()
        except subprocess.TimeoutExpired:
            output = "[command timed out after 30s]"
        except Exception as e:
            output = f"[error: {e}]"

        # Handle cd commands
        if cmd.strip().startswith("cd "):
            target = cmd.strip()[3:].strip()
            target = os.path.expanduser(target)
            new_cwd = os.path.join(self.cwd, target)
            if os.path.isdir(new_cwd):
                self.cwd = os.path.realpath(new_cwd)

        # Display and record output
        if output:
            for line in output.split('\n')[:50]:  # cap at 50 lines
                tagged_line = self._shell_tag() + line
                print(format_tagged_line(tagged_line))
                self.context.append(tagged_line)
        self.context.append(shell_tag + cmd)

    def handle_command(self, cmd: str, arg: str):
        """Handle /slash commands."""
        if cmd == "/help":
            print(HELP_TEXT)
        elif cmd == "/quit":
            print(f"\n{_DIM}Goodbye.{_RESET}")
            sys.exit(0)
        elif cmd == "/config":
            print(f"  Host:        {self.host}")
            print(f"  User:        {self.username}")
            print(f"  Agent:       {self.agent}")
            print(f"  Mode:        sliding window (always on)")
            print(f"  Window:      {self.window_size}  Passes: {self.num_passes}")
            print(f"  Temperature: {self.temperature}")
            print(f"  Top-K:       {self.top_k}")
            print(f"  Top-P:       {self.top_p}")
            print(f"  Max tokens:  {self.max_tokens}  (no hard cap)")
            print(f"  Rep penalty: {self.rep_penalty}")
            print(f"  Tagged:      {self.tagged}")
            print(f"  Context:     {len(self.context)} events")
            print(f"  CWD:         {self.cwd}")
            print(f"  Passages:    {len(self.passages)}")
        elif cmd == "/temp":
            try:
                self.temperature = float(arg)
                print(f"  Temperature → {self.temperature}")
            except ValueError:
                print("  Usage: /temp <float>")
        elif cmd == "/topk":
            try:
                self.top_k = int(arg)
                print(f"  Top-K → {self.top_k}")
            except ValueError:
                print("  Usage: /topk <int>")
        elif cmd == "/topp":
            try:
                self.top_p = float(arg)
                print(f"  Top-P → {self.top_p}")
            except ValueError:
                print("  Usage: /topp <float>")
        elif cmd == "/maxlen":
            try:
                self.max_tokens = int(arg)
                print(f"  Max tokens → {self.max_tokens}")
            except ValueError:
                print("  Usage: /maxlen <int>")
        elif cmd == "/rep":
            try:
                self.rep_penalty = float(arg)
                print(f"  Repetition penalty → {self.rep_penalty}")
            except ValueError:
                print("  Usage: /rep <float>")
        elif cmd == "/remember":
            if not arg:
                print("  Usage: /remember <passage text>")
            else:
                self.passages.append(arg)
                full_passage = " . ".join(self.passages) + " ."
                self.mem_keys, self.mem_vals, self.mem_mask = \
                    encode_passage_to_memory(self.model, full_passage,
                                             self.device, tagged=self.tagged)
                n_valid = self.mem_mask[0].sum().item() if self.mem_mask is not None else 0
                print(f"  ✓ Stored. {len(self.passages)} passage(s), "
                      f"{n_valid} memory slots active.")
        elif cmd == "/forget":
            self.passages = []
            self.mem_keys = self.mem_vals = self.mem_mask = None
            print("  ✓ Memory cleared.")
        elif cmd == "/memory":
            if not self.passages:
                print("  No passages stored. Use /remember <text>")
            else:
                for i, p in enumerate(self.passages, 1):
                    print(f"  [{i}] {p}")
                n_valid = self.mem_mask[0].sum().item() if self.mem_mask is not None else 0
                print(f"  → {n_valid} memory slots active")
        elif cmd == "/ask":
            if not arg:
                print("  Usage: /ask <question>")
            elif self.mem_keys is None:
                print("  No passages stored. Use /remember first.")
            else:
                question = arg if arg.endswith("?") else arg + " ?"
                q_tagged = self._user_tag() + question
                self.context.append(q_tagged)
                print(format_tagged_line(q_tagged))
                # Generate QA response with ANS_ID trigger
                self._generate_response(qa_mode=True)
        elif cmd == "/context":
            n = int(arg) if arg and arg.isdigit() else 10
            events = self.context[-n:]
            print(f"  {_DIM}── Last {len(events)} events ──{_RESET}")
            for ev in events:
                print(f"  {format_tagged_line(ev)}")
        elif cmd == "/clear":
            self.context = []
            print("  ✓ Context cleared.")
        elif cmd == "/agent":
            if arg:
                self.agent = arg
                print(f"  Agent → {self.agent}")
            else:
                print(f"  Current agent: {self.agent}")
        elif cmd == "/sliding":
            print("  Sliding window is always enabled — ANT has no non-sliding path.")
        else:
            print(f"  Unknown command: {cmd}. Type /help for commands.")

    def run(self):
        """Main input loop."""
        prompt_color = _HOST_COLORS.get(self.host, "\033[37m")

        while True:
            try:
                prompt = f"{_BOLD}{prompt_color}{self.username}{_RESET}{_DIM}@{self.host}{_RESET}> "
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_DIM}Goodbye.{_RESET}")
                break

            if not user_input:
                continue

            # Shell execution
            if user_input.startswith("!"):
                cmd = user_input[1:].strip()
                if cmd:
                    self.handle_shell(cmd)
                    # Let agent react to shell output
                    self._generate_response()
                continue

            # Slash commands
            if user_input.startswith("/"):
                parts = user_input.split(None, 1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                self.handle_command(cmd, arg)
                continue

            # Chat: tag user input, generate agent response
            user_event = self._user_tag() + user_input
            self.context.append(user_event)
            print(format_tagged_line(user_event))
            self._generate_response()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ANT Terminal Canvas")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--device", default=None,
                        help="Force device (cpu/mps/cuda)")
    parser.add_argument("--host", default=None,
                        help="Host name for tags (default: hostname)")
    parser.add_argument("--user", default=None,
                        help="Username for tags (default: system user)")
    parser.add_argument("--agent", default="ant",
                        help="Agent persona name (default: ant)")
    parser.add_argument("--no-tags", action="store_true",
                        help="Disable source tagging")
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Auto-detect host/user
    host = args.host or platform.node().split('.')[0].lower() or "localhost"
    username = args.user or os.environ.get("USER", "user").lower()

    # Auto-detect checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            "checkpoints/overnight/checkpoint_best.pt",
            "checkpoints/chat/best_model.pt",
            "checkpoints/tag_test/multitask/best_multitask.pt",
            "checkpoints/local_test/multitask/best_multitask.pt",
            "checkpoints/micro/multitask/best_multitask.pt",
            "checkpoints/micro/sliding_lm/best_model.pt",
            "checkpoints/micro/memory/best_model.pt",
            "checkpoints/micro/best_model.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break
        if ckpt_path is None:
            print("ERROR: No checkpoint found. Train first or use --checkpoint.")
            sys.exit(1)

    # Banner
    print(f"\n{_BOLD}  ANT Terminal Canvas{_RESET}")
    print(f"  {'─' * 42}")

    model, cfg, ckpt = load_model(ckpt_path, device)
    window_size = ckpt.get("window_size", cfg.chunk_size)
    # Respect the checkpoint's num_passes — the model was trained with a specific
    # pass count and using a different value produces garbage outputs.
    num_passes = ckpt.get("num_passes", 1)

    canvas = TerminalCanvas(
        model, cfg, ckpt, device,
        host=host, username=username, agent_name=args.agent,
        window_size=window_size, num_passes=num_passes,
        tagged=not args.no_tags)

    print(f"  {_DIM}Device: {device} | Agent: {args.agent} | "
          f"Window: W={window_size} passes={num_passes} | Memory-based chat{_RESET}")
    print(f"  {_DIM}Type /help for commands, !cmd for shell{_RESET}\n")

    canvas.run()


if __name__ == "__main__":
    main()