"""
Parent vs Fused Child Benchmark Comparison
==========================================
Runs the same MMLU / GSM8K / ARC benchmark questions on:
  - Each parent model  (opt-125m, qwen2.5-0.5b, smollm2-360m, tinyllama-1.1b)
  - The fused child    (my_llm_folder)

Produces a side-by-side accuracy table and saves full JSON.

Usage (run in YOUR cmd terminal — needs internet for datasets):
  .venv\\Scripts\\python.exe compare_parent_vs_child.py
  .venv\\Scripts\\python.exe compare_parent_vs_child.py --limit 200
  .venv\\Scripts\\python.exe compare_parent_vs_child.py --tasks mmlu gsm8k
  .venv\\Scripts\\python.exe compare_parent_vs_child.py --limit 100 --tasks arc
"""

import os
import sys
import re
import json
import time
import argparse
import ctypes
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

# ─────────────────────────────────────────────────────────────────────
# Model Registry  (all locally available)
# ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "opt-125m":       os.path.join(BASE, "modal", "text_models", "opt-125m"),
    "smollm2-360m":   os.path.join(BASE, "modal", "text_models", "smollm2-360m"),
    "qwen2.5-0.5b":   os.path.join(BASE, "modal", "text_models", "qwen2.5-0.5b"),
    "tinyllama-1.1b": os.path.join(BASE, "modal", "text_models", "tinyllama-1.1b"),
    "fused-child":    os.path.join(BASE, "my_llm_folder"),
}

from ai_dna.evaluation import (
    auto_batch,
    TASK_LOADERS,
    TASK_SCORERS,
    MMLU_SUBJECTS_SAMPLE,
)


def load_dataset_questions(task: str, limit: Optional[int], subject: Optional[str] = None) -> List[Dict]:
    """Loads benchmark questions using the unified ai_dna.evaluation task loaders."""
    if task == "mmlu":
        subjects = [subject] if subject else MMLU_SUBJECTS_SAMPLE
        return TASK_LOADERS["mmlu"](limit=limit, subjects=subjects)
    elif task in TASK_LOADERS:
        return TASK_LOADERS[task](limit=limit)
    else:
        print(f"[!] Unknown task: {task}")
        return []


def score(response: str, q: Dict) -> bool:
    """Scores response using the unified ai_dna.evaluation task scorers."""
    task = q.get("task", "")
    scorer = TASK_SCORERS.get(task)
    if scorer:
        return scorer(response, q)
    return False



# ─────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    model_path: str,
    questions_by_task: Dict[str, List[Dict]],
    device: str,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*70}")
    print(f"  MODEL: {name}")
    print(f"  Path : {model_path}")
    print(f"{'='*70}")

    if not os.path.exists(model_path):
        print(f"  [!] Path does not exist — skipping.")
        return {"name": name, "error": "path not found", "tasks": {}}

    # Load
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"  [!] Failed to load: {e}")
        return {"name": name, "error": str(e), "tasks": {}}

    n_params = sum(p.numel() for p in model.parameters())
    batch_size = auto_batch(model_path, device)
    print(f"  Params: {n_params:,}  |  Batch: {batch_size}  |  Load: {time.time()-t0:.1f}s")

    pad_id = tokenizer.pad_token_id or 0

    def run_batch(prompts: List[str]) -> List[str]:
        formatted = []
        for p in prompts:
            if getattr(tokenizer, "chat_template", None):
                try:
                    p = tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    pass
            formatted.append(p)
        enc = tokenizer(formatted, return_tensors="pt", padding=True,
                        truncation=True, max_length=512, return_attention_mask=True).to(device)
        iln = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=48, do_sample=False, pad_token_id=pad_id,
            )
        return [tokenizer.decode(seq[iln:], skip_special_tokens=True).strip() for seq in out]

    task_results = {}
    grand_correct = 0; grand_total = 0

    for task, questions in questions_by_task.items():
        if not questions:
            continue
        correct = 0; sample_responses = []
        t_task = time.time()
        import math
        n_batches = math.ceil(len(questions) / batch_size)

        print(f"\n  [{task.upper()}] {len(questions)} questions | {n_batches} batches")

        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            batch_q = questions[sl]
            try:
                responses = run_batch([q["prompt"] for q in batch_q])
            except Exception as e:
                print(f"    [!] Batch {b} error: {e}")
                responses = [""] * len(batch_q)

            for q, resp in zip(batch_q, responses):
                ok = score(resp, q)
                if ok: correct += 1
                if len(sample_responses) < 3:
                    sample_responses.append({
                        "prompt": q["prompt"][:80],
                        "response": resp[:100],
                        "correct": ok,
                    })

            done = min((b + 1) * batch_size, len(questions))
            elapsed = time.time() - t_task
            acc = correct / done * 100
            qps = done / elapsed if elapsed else 0
            eta = (len(questions) - done) / qps if qps else 0
            bar = "█" * int(30 * done / len(questions)) + "░" * (30 - int(30 * done / len(questions)))
            print(f"  [{bar}] {done:>5}/{len(questions)}  {acc:5.1f}%  {qps:5.1f} q/s  ETA {eta:.0f}s",
                  end="\r", flush=True)

        elapsed_task = time.time() - t_task
        acc = correct / len(questions) * 100 if questions else 0
        print()
        print(f"  {task.upper()} DONE: {acc:.2f}%  ({correct}/{len(questions)})  {elapsed_task:.1f}s")

        task_results[task] = {
            "accuracy": round(acc, 2),
            "correct": correct,
            "total": len(questions),
            "elapsed_seconds": round(elapsed_task, 1),
            "qps": round(len(questions) / elapsed_task, 1) if elapsed_task else 0,
            "samples": sample_responses,
        }
        grand_correct += correct; grand_total += len(questions)

    # Free GPU memory before loading next model
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    overall = grand_correct / grand_total * 100 if grand_total else 0.0
    return {
        "name": name,
        "params": n_params,
        "batch_size": batch_size,
        "overall_accuracy": round(overall, 2),
        "total_evaluated": grand_total,
        "tasks": task_results,
    }


# ─────────────────────────────────────────────────────────────────────
# Comparison Table
# ─────────────────────────────────────────────────────────────────────

def print_comparison_table(all_results: Dict[str, Any], tasks: List[str]) -> None:
    print("\n" + "=" * 90)
    print("  PARENT vs FUSED CHILD — BENCHMARK COMPARISON")
    print("=" * 90)

    # Header
    col_w = 14
    task_cols = "".join(f"  {t.upper():<{col_w}}" for t in tasks)
    print(f"  {'Model':<22}  {'Params':>10}  {'Overall':>8}{task_cols}")
    print("  " + "─" * 85)

    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<22}  {'ERROR':>10}  {'—':>8}")
            continue

        params_str = f"{res['params']/1e6:.0f}M" if res.get("params") else "—"
        overall_str = f"{res['overall_accuracy']:.2f}%"
        task_str = ""
        for t in tasks:
            acc = res["tasks"].get(t, {}).get("accuracy", None)
            task_str += f"  {f'{acc:.2f}%' if acc is not None else '—':<{col_w}}"

        # Highlight fused child
        marker = "◀ FUSED" if name == "fused-child" else ""
        print(f"  {name:<22}  {params_str:>10}  {overall_str:>8}{task_str}  {marker}")

    print("=" * 90)

    # Delta section: fused vs each parent
    if "fused-child" in all_results and "error" not in all_results["fused-child"]:
        child = all_results["fused-child"]
        print("\n  DELTA  (Fused Child − Parent)")
        print("  " + "─" * 70)
        for name, res in all_results.items():
            if name == "fused-child" or "error" in res: continue
            delta_overall = child["overall_accuracy"] - res["overall_accuracy"]
            sign = "+" if delta_overall >= 0 else ""
            task_deltas = ""
            for t in tasks:
                c_acc = child["tasks"].get(t, {}).get("accuracy", 0)
                p_acc = res["tasks"].get(t, {}).get("accuracy", 0)
                d = c_acc - p_acc
                s = "+" if d >= 0 else ""
                task_deltas += f"  {f'{s}{d:.2f}%':<14}"
            print(f"  vs {name:<20}  {sign}{delta_overall:.2f}%{task_deltas}")
        print()


# ─────────────────────────────────────────────────────────────────────
# Sample output comparison
# ─────────────────────────────────────────────────────────────────────

def print_sample_comparison(all_results: Dict[str, Any], tasks: List[str]) -> None:
    print("\n" + "=" * 90)
    print("  SAMPLE RESPONSE COMPARISON (first question per task)")
    print("=" * 90)

    for task in tasks:
        print(f"\n  ── {task.upper()} ──")
        prompt_shown = False
        for name, res in all_results.items():
            if "error" in res: continue
            samples = res["tasks"].get(task, {}).get("samples", [])
            if not samples: continue
            s = samples[0]
            if not prompt_shown:
                print(f"  Q: {s['prompt'][:100]}...")
                prompt_shown = True
            marker = "✓" if s["correct"] else "✗"
            print(f"  {marker} {name:<20}: {s['response'][:80]}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare parent models vs fused child on MMLU / GSM8K / ARC"
    )
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k", "arc"],
                        choices=["mmlu", "gsm8k", "arc"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions per task per model (None = full)")
    parser.add_argument("--subject", type=str, default=None,
                        help="Restrict MMLU to a single subject")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Which models to include (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="./bench_results")
    args = parser.parse_args()

    # Print resource info
    has_cuda = torch.cuda.is_available()
    print("\n" + "=" * 70)
    print("  [COMPARISON BENCHMARK — AUTO RESOURCE DETECTION]")
    print("=" * 70)
    if has_cuda:
        free_v, total_v = torch.cuda.mem_get_info(0)
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {free_v/1e9:.2f} GB free / {total_v/1e9:.2f} GB total")
    print(f"  Device: {args.device.upper()}")
    print(f"  Tasks : {', '.join(args.tasks)}")
    print(f"  Limit : {'ALL' if not args.limit else args.limit} per task")
    print("=" * 70)

    # Load datasets once (shared across all models)
    print("\n[+] Loading benchmark datasets ...")
    questions_by_task: Dict[str, List[Dict]] = {}
    for task in args.tasks:
        questions_by_task[task] = load_dataset_questions(
            task, limit=args.limit, subject=args.subject
        )

    # Evaluate each model
    all_results: Dict[str, Any] = {}
    for model_name in args.models:
        if model_name not in MODELS:
            print(f"[!] Unknown model '{model_name}', skipping.")
            continue
        result = evaluate_model(
            name=model_name,
            model_path=MODELS[model_name],
            questions_by_task=questions_by_task,
            device=args.device,
        )
        all_results[model_name] = result

    # Print comparison table + sample responses
    print_comparison_table(all_results, args.tasks)
    print_sample_comparison(all_results, args.tasks)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "comparison_results.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": args.tasks,
        "limit": args.limit,
        "device": args.device,
        "models": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Full comparison saved -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
