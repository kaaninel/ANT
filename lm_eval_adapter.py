"""
lm-evaluation-harness adapter for LoopedLatentController.

Enables standard LLM benchmarks (HellaSwag, ARC, PIQA, etc.) and direct
comparison with any HuggingFace model (Qwen3, Llama, Phi, …).

Usage (CLI):
    pip install lm-eval

    # Benchmark LoopedLatentController:
    lm-eval run --include_path . --model latent_controller \
        --model_args checkpoint_dir=./checkpoints,data_dir=./data_cache \
        --tasks hellaswag,arc_easy,piqa,boolq \
        --device cuda:0

    # Same benchmarks on Qwen3-0.5B:
    lm-eval run --model hf \
        --model_args pretrained=Qwen/Qwen3-0.5B,dtype=bfloat16,trust_remote_code=true \
        --tasks hellaswag,arc_easy,piqa,boolq \
        --device cuda:0

Usage (Python API — see run_standard_benchmarks.py for full comparison):
    import lm_eval
    import lm_eval_adapter          # registers "latent_controller"
    results = lm_eval.simple_evaluate(
        model="latent_controller",
        model_args="checkpoint_dir=./checkpoints,data_dir=./data_cache",
        tasks=["hellaswag", "arc_easy"],
        device="cuda:0",
    )
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Tuple

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance

from model import LoopedLatentController
from config import ModelConfig, MemoryConfig
from memory import MemorySystem
from tokenizer_utils import load_tokenizer
from utils import load_checkpoint


@register_model("latent_controller")
class LatentControllerLM(LM):
    """Wrap LoopedLatentController for EleutherAI lm-evaluation-harness."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        data_dir: str = "./data_cache",
        use_memory: str = "false",
        device: str = "cuda",
        batch_size: int | str = 1,
        max_gen_toks: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.cfg = ModelConfig()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)
        self._use_memory = str(use_memory).lower() in ("true", "1", "yes")

        # --- tokenizer ---
        tok_path = os.path.join(data_dir, "tokenizer.json")
        self.tokenizer = load_tokenizer(tok_path)

        # --- model ---
        self.model = LoopedLatentController(self.cfg, use_checkpoint=False).to(self._device)
        loaded_phase = self._load_best_checkpoint(checkpoint_dir)
        self.model.eval()
        print(f"[latent_controller] Phase {loaded_phase} | {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params | {self._device}")

        # --- optional memory ---
        self.memory = None
        if self._use_memory:
            self.memory = MemorySystem(MemoryConfig(), self.cfg)
            print("[latent_controller] Memory system enabled")

    # ----- required LM properties -----

    @property
    def eot_token_id(self):
        return self.cfg.eos_id

    @property
    def max_length(self):
        return self.cfg.max_seq_len          # 512

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    # ----- tokenizer helpers -----

    def tok_encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.cfg.bos_id] + ids
        return ids

    def tok_decode(self, tokens: List[int]) -> str:
        tokens = [t for t in tokens if t >= 7]  # skip special 0-6
        return self.tokenizer.decode(tokens) if tokens else ""

    # ----- core evaluation methods -----

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results: List[Tuple[float, bool]] = []
        for req in requests:
            ctx_str, cont_str = req.args

            # Tokenize jointly to handle BPE boundary correctly
            ctx_ids = self.tok_encode(ctx_str)
            full_ids = self.tok_encode(ctx_str + cont_str)

            if full_ids[:len(ctx_ids)] == ctx_ids:
                cont_ids = full_ids[len(ctx_ids):]
                all_ids = full_ids
            else:
                cont_ids = self.tok_encode(cont_str)
                all_ids = ctx_ids + cont_ids

            if not cont_ids:
                results.append((0.0, True))
                continue

            cont_len = len(cont_ids)

            # Truncate from left to fit context window
            if len(all_ids) > self.max_length:
                all_ids = all_ids[-self.max_length:]
                cont_len = min(cont_len, len(all_ids))

            cont_start = len(all_ids) - cont_len

            logits = self._forward(torch.tensor([all_ids], device=self._device))
            log_probs = F.log_softmax(logits[0], dim=-1)

            total_ll = 0.0
            is_greedy = True
            for j in range(cont_len):
                pred_pos = cont_start + j - 1          # logits[t] predicts t+1
                if pred_pos < 0:
                    continue
                target = all_ids[cont_start + j]
                total_ll += log_probs[pred_pos, target].item()
                if logits[0, pred_pos].argmax().item() != target:
                    is_greedy = False

            results.append((total_ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float]]:
        results: List[Tuple[float]] = []
        for req in requests:
            (text,) = req.args
            ids = self.tok_encode(text)

            if len(ids) <= 1:
                results.append((0.0,))
                continue

            total_ll = 0.0
            for start in range(0, len(ids), self.max_length):
                chunk = ids[start : start + self.max_length]
                if len(chunk) <= 1:
                    break
                logits = self._forward(torch.tensor([chunk], device=self._device))
                log_probs = F.log_softmax(logits[0], dim=-1)
                for j in range(len(chunk) - 1):
                    total_ll += log_probs[j, chunk[j + 1]].item()

            results.append((total_ll,))
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results: List[str] = []
        for req in requests:
            ctx_str, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_toks = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = gen_kwargs.get("do_sample", False)

            ids = self.tok_encode(ctx_str)
            if len(ids) > self.max_length - max_toks:
                ids = ids[-(self.max_length - max_toks):]

            generated: List[int] = []
            current = list(ids)

            for _ in range(max_toks):
                window = current[-self.max_length:]
                logits = self._forward(torch.tensor([window], device=self._device))
                last_logits = logits[0, -1]

                if do_sample and temperature > 0:
                    probs = F.softmax(last_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                else:
                    next_id = last_logits.argmax().item()

                if next_id == self.eot_token_id:
                    break

                generated.append(next_id)
                current.append(next_id)

                gen_text = self.tok_decode(generated)
                stopped = False
                for stop in until:
                    idx = gen_text.find(stop)
                    if idx != -1:
                        gen_text = gen_text[:idx]
                        stopped = True
                        break
                if stopped:
                    results.append(gen_text)
                    break
            else:
                results.append(self.tok_decode(generated))

        return results

    # ----- internals -----

    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            amp_on = self._device.type == "cuda"
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                logits, _ = self.model(input_ids)
        return logits.float()

    def _load_best_checkpoint(self, checkpoint_dir: str) -> int:
        for phase in [5, 4, 3, 1]:
            ckpt = os.path.join(checkpoint_dir, f"phase{phase}", "best.pt")
            if not os.path.exists(ckpt):
                continue
            load_checkpoint(self.model, None, ckpt, self._device)
            if phase >= 3:
                p2 = os.path.join(checkpoint_dir, "phase2", "best.pt")
                if os.path.exists(p2):
                    p2_data = torch.load(p2, map_location=self._device, weights_only=False)
                    if "addr_heads" in p2_data:
                        for i, h in enumerate(self.model.addr_heads):
                            h.load_state_dict(p2_data["addr_heads"][i])
            return phase
        raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}")
