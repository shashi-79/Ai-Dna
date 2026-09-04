"""
Local Text Benchmark Runner & Model Card Widget Updater (Step 3 & Step 4).

Features:
  1. Step 3 (Benchmarking):
     - Evaluates local model folders (e.g. ./my_llm_folder) across core reasoning dimensions:
       * MMLU (Multi-domain knowledge & abstract reasoning)
       * GSM8K (Step-by-step mathematical reasoning)
       * IFEval (Strict instruction following & constraints)
       * ARC / Science (Scientific reasoning & facts)
     - Saves detailed metrics and logs into `./bench_results/`.

  2. Step 4 (Model Page Score Widget):
     - Automatically parses `./bench_results/benchmark_summary.json`
     - Injects official `model-index:` YAML frontmatter into `README.md`
       so benchmark scores render as official widgets on Hugging Face model pages.

Usage:
    # Run benchmarks on ./my_llm_folder and auto-update README.md
    python run_local_benchmarks.py --model-dir "./my_llm_folder" --update-readme

    # Run specific tasks with custom limit
    python run_local_benchmarks.py --model-dir "./my_llm_folder" --tasks mmlu gsm8k --limit 10

    # Only update README.md from an existing bench_results folder
    python run_local_benchmarks.py --update-readme-only --results-file "./bench_results/benchmark_summary.json"
"""

import os
import sys
import json
import argparse
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ai_dna.evaluation import (
    evaluate_hf_model,
    update_readme_model_index,
    format_benchmark_markdown_table,
)


def main():
    parser = argparse.ArgumentParser(
        description="Local Open-Source Benchmark Runner & Model Card Widget Updater for AI-DNA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="./my_llm_folder",
        help="Path to local Hugging Face model directory.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["mmlu", "gsm8k", "arc", "ifeval"],
        choices=["mmlu", "gsm8k", "arc", "ifeval"],
        help="List of benchmark tasks to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions per task (useful for quick local testing, e.g. --limit 5).",
    )
    parser.add_argument(
        "--output-dir",
        default="./bench_results",
        help="Directory to save benchmark results JSON and metrics.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to execute benchmark on ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        default=True,
        help="Automatically inject benchmark scores into README.md model card widget.",
    )
    parser.add_argument(
        "--update-readme-only",
        action="store_true",
        help="Only update README.md from an existing benchmark results file without re-running evaluation.",
    )
    parser.add_argument(
        "--results-file",
        default="./bench_results/benchmark_summary.json",
        help="Path to existing benchmark results JSON (used with --update-readme-only).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    readme_path = os.path.join(args.model_dir, "README.md")
    model_name = os.path.basename(os.path.abspath(args.model_dir))

    if args.update_readme_only:
        if not os.path.exists(args.results_file):
            print(f"[ERROR] Results file not found: {args.results_file}")
            sys.exit(1)
        with open(args.results_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        update_readme_model_index(readme_path, model_name, summary)
        print(f"[+] Updated model index in {readme_path}")
        return

    # Step 3: Run Benchmark Evaluation using modular engine
    summary = evaluate_hf_model(
        model_dir=args.model_dir,
        tasks=args.tasks,
        limit=args.limit,
        device=args.device,
        verbose=True,
    )

    # Save Results JSON
    results_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Saved Benchmark Summary to: {os.path.abspath(results_path)}")

    # Step 4: Update Model Card
    if args.update_readme and os.path.exists(readme_path):
        update_readme_model_index(readme_path, model_name, summary)
        print(f"[+] Updated Hugging Face model-index frontmatter in: {readme_path}")

    print("\n" + "=" * 80)
    print("  [BENCHMARK SUMMARY]")
    avg = summary.get("summary", {}).get("average_accuracy", 0.0)
    print(f"  Overall Score: {avg:.1f}%")
    for tk, tv in summary.get("tasks", {}).items():
        print(f"    - {tk.upper():<10}: {tv['accuracy']:.1f}% ({tv['correct']}/{tv['total']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
