#!/usr/bin/env python3
"""
Run standard LLM benchmarks on LoopedLatentController and display results
alongside published reference scores from Qwen3 and other models.

Uses EleutherAI's lm-evaluation-harness under the hood.

Examples:
    # Quick run (~5 min on A100): 3 tasks, 200 examples each
    python run_standard_benchmarks.py --quick

    # Full run (~30 min): 6 tasks, full datasets
    python run_standard_benchmarks.py

    # Custom tasks + limit
    python run_standard_benchmarks.py --tasks hellaswag arc_easy --limit 500

    # Include memory system
    python run_standard_benchmarks.py --quick --use_memory
"""

import argparse
import gc
import json
import os
import sys

TASKS_FULL = ["hellaswag", "arc_easy", "piqa", "boolq", "lambada_openai", "winogrande"]
TASKS_QUICK = ["hellaswag", "arc_easy", "piqa"]

# Published reference scores (0-shot accuracy %)
# Sources: Qwen3 Technical Report (arXiv:2505.09388), Open LLM Leaderboard
REFERENCE_SCORES = {
    "hellaswag":       {"Random": 25.0, "Qwen3-0.6B": 42.6, "Qwen3-4B": 71.4, "Llama3-8B": 82.0},
    "arc_easy":        {"Random": 25.0, "Qwen3-0.6B": 48.0, "Qwen3-4B": 75.1, "Llama3-8B": 80.4},
    "piqa":            {"Random": 50.0, "Qwen3-0.6B": 64.4, "Qwen3-4B": 81.1, "Llama3-8B": 82.3},
    "boolq":           {"Random": 50.0, "Qwen3-0.6B": 56.1, "Qwen3-4B": 77.6, "Llama3-8B": 83.1},
    "lambada_openai":  {"Random":  0.0, "Qwen3-0.6B": 32.7, "Qwen3-4B": 61.4, "Llama3-8B": 76.3},
    "winogrande":      {"Random": 50.0, "Qwen3-0.6B": 50.2, "Qwen3-4B": 73.0, "Llama3-8B": 78.5},
}

REF_MODELS = ["Random", "Qwen3-0.6B", "Qwen3-4B", "Llama3-8B"]


def run_eval(model_args, tasks, device, limit, num_fewshot=0):
    """Run lm-eval on LoopedLatentController and return results."""
    import lm_eval

    out = lm_eval.simple_evaluate(
        model="latent_controller",
        model_args=model_args,
        tasks=tasks,
        device=device,
        limit=limit,
        batch_size=1,
        num_fewshot=num_fewshot,
    )
    return out["results"]


def get_accuracy(task_result: dict):
    """Extract the best accuracy metric from a task result dict."""
    for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
        if key in task_result:
            return task_result[key]
    return None


def print_results(lc_results: dict, tasks: list, limit: int | None):
    """Print results table with LC scores alongside published references."""
    col_w = 14
    ref_cols = REF_MODELS
    all_cols = ["LC-34M"] + ref_cols
    sep_len = 22 + col_w * len(all_cols)

    print()
    print("=" * sep_len)
    title = "STANDARD LLM BENCHMARKS — 0-shot accuracy %"
    if limit:
        title += f"  (limit={limit})"
    print(f"  {title}")
    print("=" * sep_len)

    header = f"{'Task':<22}"
    for col in all_cols:
        header += f"{col:>{col_w}}"
    print(header)
    print("-" * sep_len)

    lc_avg = []
    ref_avgs = {m: [] for m in ref_cols}

    for task in tasks:
        row = f"{task:<22}"

        # LC score
        acc = get_accuracy(lc_results.get(task, {}))
        if acc is not None:
            row += f"{acc * 100:>{col_w - 1}.1f}%"
            lc_avg.append(acc * 100)
        else:
            row += f"{'N/A':>{col_w}}"

        # Reference scores
        refs = REFERENCE_SCORES.get(task, {})
        for m in ref_cols:
            val = refs.get(m)
            if val is not None:
                row += f"{val:>{col_w - 1}.1f}%"
                ref_avgs[m].append(val)
            else:
                row += f"{'—':>{col_w}}"

        print(row)

    print("-" * sep_len)
    row = f"{'AVERAGE':<22}"
    row += f"{sum(lc_avg) / len(lc_avg) if lc_avg else 0:>{col_w - 1}.1f}%"
    for m in ref_cols:
        vals = ref_avgs[m]
        row += f"{sum(vals) / len(vals) if vals else 0:>{col_w - 1}.1f}%"
    print(row)
    print()
    print("  Reference scores: Qwen3 Technical Report (arXiv:2505.09388), Open LLM Leaderboard")
    print("  LC-34M: LoopedLatentController 34M params, trained on TinyStories")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run standard LLM benchmarks on LoopedLatentController"
    )
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help=f"Tasks to run (default: {' '.join(TASKS_FULL)})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples per task")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 200 examples, 3 tasks",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_memory", action="store_true", help="Enable memory system")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Few-shot examples (0 recommended for 512 context window)")
    parser.add_argument("--output", default="./benchmark_results.json")
    args = parser.parse_args()

    if args.quick:
        args.limit = args.limit or 200
        tasks = args.tasks or TASKS_QUICK
    else:
        tasks = args.tasks or TASKS_FULL

    # Register the latent_controller model with lm-eval
    import lm_eval_adapter  # noqa: F401

    print("=" * 60)
    print("  Benchmarking: LoopedLatentController (34M)")
    print("=" * 60)

    model_args = f"checkpoint_dir={args.checkpoint_dir},data_dir={args.data_dir}"
    if args.use_memory:
        model_args += ",use_memory=true"

    results = run_eval(model_args, tasks, args.device, args.limit, args.num_fewshot)

    # Print comparison table
    print_results(results, tasks, args.limit)

    # Save raw results
    out = {"lc_results": results, "reference_scores": REFERENCE_SCORES}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Raw results saved → {args.output}")


if __name__ == "__main__":
    main()
