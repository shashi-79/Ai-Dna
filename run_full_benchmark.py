"""
Full Dataset Benchmark Runner — Parallel / Batched GPU Inference
================================================================
Auto-detects available VRAM and RAM, computes optimal batch size,
and runs batched generation on CUDA for maximum throughput.

Strategy:
  - Model ~256 MB bfloat16 on CUDA
  - 7+ GB free VRAM -> batch_size auto-tuned up to 128
  - Multiple threads feed the GPU from pre-tokenised DataLoader
  - Live progress: q/s, ETA, running accuracy

Datasets (download requires internet — run in YOUR terminal, not sandbox):
  - MMLU         : 14,042 questions, 57 subjects
  - GSM8K        : 1,319 math problems
  - ARC-Challenge: 1,172 science questions
  - IFEval       : 541 instruction-following prompts

Usage:
  .venv\\Scripts\\python.exe run_full_benchmark.py --model-dir ./my_llm_folder
  .venv\\Scripts\\python.exe run_full_benchmark.py --model-dir ./my_llm_folder --limit 500
  .venv\\Scripts\\python.exe run_full_benchmark.py --model-dir ./my_llm_folder --tasks gsm8k arc
  .venv\\Scripts\\python.exe run_full_benchmark.py --model-dir ./my_llm_folder --batch-size 64
"""

import os
import sys
import time
import json
import re
import math
import argparse
import ctypes
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ai_dna.dna.serialization import load_genotype
from convert_aidna_to_safetensors import extract_weights_from_genotype

# =====================================================================
# Modular Task Loaders, Scorers & Resource Detection (ai_dna.evaluation)
# =====================================================================
from ai_dna.evaluation import (
    auto_detect_resources as detect_resources,
    TASK_LOADERS,
    TASK_SCORERS,
    check_mc,
    check_gsm8k,
    check_ifeval,
    load_mmlu as load_mmlu_dataset,
    load_gsm8k as load_gsm8k_dataset,
    load_arc as load_arc_dataset,
    load_ifeval as load_ifeval_dataset,
)


def print_resource_banner(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("  [AUTO RESOURCE DETECTION]")
    print("=" * 70)
    if res["device"] == "cuda":
        print(f"  GPU      : {res['gpu_name']}")
        print(f"  VRAM     : {res['free_vram_gb']:.2f} GB free / {res['total_vram_gb']:.2f} GB total")
    print(f"  RAM      : {res['free_ram_gb']:.2f} GB free")
    print(f"  CPU cores: {res['cpu_count']}")
    print(f"  Model    : {res['model_size_mb']} MB")
    print(f"  ─── Auto-selected settings ───")
    print(f"  Device   : {res['device'].upper()}")
    print(f"  Batch sz : {res['batch_size']}  (questions processed simultaneously)")
    print(f"  DL workers: {res['dataloader_workers']}")
    print("=" * 70)


# =====================================================================
# Batched Inference Engine
# =====================================================================

class BatchedInferenceEngine:
    """
    Wraps a HF model + tokenizer and runs parallel batched generation.
    Automatically pads/truncates and handles variable-length outputs.
    """

    def __init__(self, model, tokenizer, device: str, batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        # Set pad_token if missing and ensure left padding for decoder-only generation
        self.tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    def _apply_template(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def infer_batch(self, prompts: List[str], max_new: int = 64) -> List[str]:
        """Tokenise, pad, run generation, decode — for a list of prompts."""
        formatted = [self._apply_template(p) for p in prompts]

        enc = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
        ).to(self.device)

        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.pad_id,
            )

        responses = []
        for seq in out:
            new_tok = seq[input_len:]
            text = self.tokenizer.decode(new_tok, skip_special_tokens=True).strip()
            responses.append(text)
        return responses

    def run_dataset(
        self,
        questions: List[Dict],
        task: str,
        max_new: int = 64,
    ) -> Tuple[int, int, List[Dict]]:
        """
        Evaluate all questions in batches.
        Returns (correct, total, sample_list).
        """
        correct = 0
        total = len(questions)
        samples: List[Dict] = []
        n_batches = math.ceil(total / self.batch_size)
        t0 = time.time()

        print(
            f"\n  Batch size: {self.batch_size} | "
            f"Batches: {n_batches} | "
            f"Questions: {total:,}"
        )
        print(f"  {'─'*60}")

        for b_idx in range(n_batches):
            start = b_idx * self.batch_size
            end   = min(start + self.batch_size, total)
            batch = questions[start:end]
            prompts = [q["prompt"] for q in batch]

            responses = self.infer_batch(prompts, max_new=max_new)

            for q, resp in zip(batch, responses):
                if task in ("mmlu", "arc"):
                    ok = check_mc(resp, q["answer"])
                elif task == "gsm8k":
                    ok = check_gsm8k(resp, q["expected"])
                elif task == "ifeval":
                    ok = check_ifeval(resp, q.get("instructions", []))
                else:
                    ok = False
                if ok:
                    correct += 1
                if len(samples) < 100:
                    samples.append({
                        "index": start + len(samples) + 1,
                        "prompt": q["prompt"][:100],
                        "response": resp[:120],
                        "correct": ok,
                    })

            # Progress every batch
            elapsed = time.time() - t0
            done = end
            acc = correct / done * 100
            qps = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / qps if qps > 0 else 0
            bar_len = 30
            filled = int(bar_len * done / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"  [{bar}] {done:>6}/{total}  "
                f"Acc: {acc:5.1f}%  |  {qps:5.1f} q/s  |  ETA: {eta:>5.0f}s",
                end="\r", flush=True
            )

        elapsed_total = time.time() - t0
        final_acc = correct / total * 100 if total else 0
        qps_final = total / elapsed_total if elapsed_total > 0 else 0
        print()  # newline after \r
        print(
            f"\n  {'='*60}\n"
            f"  {task.upper()} DONE:  {final_acc:.2f}%  "
            f"({correct}/{total})  |  {qps_final:.1f} q/s avg  |  {elapsed_total:.1f}s\n"
            f"  {'='*60}"
        )
        return correct, total, samples, elapsed_total


# =====================================================================
# Main
# =====================================================================

def load_model_from_aidna(aidna_path: str, fallback_dir: str, device: str):
    """Loads model weights and architecture directly from an .aidna genetic container."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    g = load_genotype(aidna_path)
    weights, _ = extract_weights_from_genotype(g, device=torch.device("cpu"))

    cfg = None
    if fallback_dir and os.path.exists(fallback_dir):
        try:
            cfg = AutoConfig.from_pretrained(fallback_dir)
        except Exception:
            cfg = None

    if cfg is None:
        for k, v in g.sensory_assets.items():
            if k.startswith("config."):
                if isinstance(v, bytes):
                    cfg_dict = json.loads(v.decode("utf-8"))
                elif isinstance(v, dict):
                    cfg_dict = v
                else:
                    cfg_dict = None
                if cfg_dict:
                    try:
                        cfg = AutoConfig.for_model(**cfg_dict)
                        break
                    except Exception:
                        pass

    if cfg is None:
        raise ValueError(f"Could not resolve architecture config for {aidna_path}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    mdl = AutoModelForCausalLM.from_config(cfg).to(dtype=dtype)
    model_state = mdl.state_dict()
    filtered_weights = {}
    for k, v in weights.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_weights[k] = v

    mdl.load_state_dict(filtered_weights, strict=False)
    mdl.to(device)
    mdl.eval()

    tok_dir = fallback_dir if os.path.exists(fallback_dir) else "my_llm_folder"
    tok = AutoTokenizer.from_pretrained(tok_dir)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    return mdl, tok, g


def run_full_benchmark(
    model_dir: str,
    aidna_path: Optional[str],
    tasks: List[str],
    limit: Optional[int],
    batch_size: Optional[int],
    device: Optional[str],
    subject: Optional[str],
    output_dir: str,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    target_source = aidna_path if (aidna_path and os.path.exists(aidna_path)) else model_dir

    # 1. Resource detection
    res = detect_resources(model_dir if os.path.isdir(target_source) else os.path.dirname(target_source))
    if device:
        res["device"] = device
    if batch_size:
        res["batch_size"] = batch_size
    print_resource_banner(res)

    chosen_device = res["device"]
    chosen_batch = res["batch_size"]

    print("\n" + "=" * 70)
    print("  [AI-DNA FULL DATASET BENCHMARK — PARALLEL GPU INFERENCE]")
    print(f"  Target : {os.path.abspath(target_source)}")
    print(f"  Tasks  : {', '.join(tasks)}")
    print(f"  Limit  : {'FULL DATASET' if not limit else f'{limit} per task'}")
    print(f"  Device : {chosen_device.upper()}  |  Batch: {chosen_batch}")
    print("=" * 70)

    # 2. Load model
    print("\n[+] Loading model & tokenizer ...", flush=True)
    t0 = time.time()
    if aidna_path and os.path.exists(aidna_path):
        model, tokenizer, g = load_model_from_aidna(aidna_path, model_dir, chosen_device)
        print(f"[+] Loaded {len(g.dna_instinct.genetic_parameters)} tensors directly from .aidna: {os.path.basename(aidna_path)}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        dtype = torch.bfloat16 if chosen_device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(chosen_device)
        model.eval()

    # Optional: torch.compile for extra speed on CUDA (PyTorch 2.0+)
    compiled = False
    if chosen_device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            compiled = True
            print("[+] torch.compile enabled (reduce-overhead mode)")
        except Exception:
            pass

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Loaded {n_params:,} params in {time.time()-t0:.2f}s"
          + ("  [compiled]" if compiled else ""))

    engine = BatchedInferenceEngine(model, tokenizer, chosen_device, chosen_batch)

    # 3. Evaluate each task
    all_results: Dict[str, Any] = {}
    grand_correct = 0
    grand_total   = 0

    loaders = {
        "mmlu":   lambda: load_mmlu_dataset(subject=subject, limit=limit),
        "gsm8k":  lambda: load_gsm8k_dataset(limit=limit),
        "arc":    lambda: load_arc_dataset(limit=limit),
        "ifeval": lambda: load_ifeval_dataset(limit=limit),
    }

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*70}")

        if task not in loaders:
            print(f"  [!] Unknown task '{task}', skipping."); continue

        print(f"  Loading {task.upper()} dataset ...")
        questions = loaders[task]()
        if not questions:
            print(f"  [!] No questions for {task}. Skipping."); continue

        correct, total, samples, elapsed = engine.run_dataset(
            questions=questions, task=task, max_new=64
        )

        accuracy = correct / total * 100 if total else 0.0
        all_results[task] = {
            "name": task.upper(),
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total": total,
            "elapsed_seconds": round(elapsed, 1),
            "qps": round(total / elapsed, 2) if elapsed else 0,
            "batch_size_used": chosen_batch,
            "samples": samples,
        }
        grand_correct += correct
        grand_total   += total

    # 4. Summary
    overall = grand_correct / grand_total * 100 if grand_total else 0.0
    summary = {
        "model_dir": os.path.abspath(model_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_accuracy": round(overall, 2),
        "total_evaluated": grand_total,
        "hardware": {
            "device": res["device"],
            "gpu": res["gpu_name"],
            "vram_free_gb": res["free_vram_gb"],
            "ram_free_gb": res["free_ram_gb"],
            "batch_size": chosen_batch,
        },
        "tasks": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "full_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Final table
    print("\n" + "=" * 80)
    print("  FINAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  {'Task':<22} {'Acc':>9}  {'Correct':>8}  {'Total':>8}  {'q/s':>6}  {'Time':>7}")
    print("  " + "─" * 68)
    for tk, tv in all_results.items():
        print(
            f"  {tv['name']:<22} {tv['accuracy']:>8.2f}%"
            f"  {tv['correct']:>8}  {tv['total']:>8}"
            f"  {tv['qps']:>6.1f}  {tv['elapsed_seconds']:>5.0f}s"
        )
    print("  " + "─" * 68)
    print(f"  {'OVERALL':<22} {overall:>8.2f}%  {grand_correct:>8}  {grand_total:>8}")
    print("=" * 80)
    print(f"\n[+] Results saved -> {os.path.abspath(out_path)}")

    # Inject model-index into README
    readme_path = os.path.join(model_dir, "README.md")
    if os.path.exists(readme_path):
        _inject_model_index(readme_path, os.path.basename(os.path.abspath(model_dir)), summary)
        print("[+] README.md updated with HF model-index widget scores!")

    return summary


# =====================================================================
# README injection
# =====================================================================

def _inject_model_index(readme_path: str, model_name: str, summary: Dict) -> None:
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    dataset_map = {
        "mmlu": "cais/mmlu", "gsm8k": "openai/gsm8k",
        "arc": "ai2_arc",    "ifeval": "google/IFEval",
    }
    lines = ["model-index:", f"- name: {model_name}", "  results:"]
    for tk, ti in summary.get("tasks", {}).items():
        lines += [
            "  - task:",
            "      type: text-generation",
            "      name: Text Generation",
            "    dataset:",
            f"      name: {ti['name']}",
            f"      type: {dataset_map.get(tk, tk)}",
            "    metrics:",
            "    - name: Accuracy",
            "      type: accuracy",
            f"      value: {ti['accuracy']}",
        ]
    block = "\n".join(lines) + "\n"

    content = re.sub(r"model-index:.*?(?=\n---|\Z)", "", content, flags=re.DOTALL)
    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx != -1:
            content = content[:end_idx].rstrip() + "\n" + block + content[end_idx:]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI-DNA Full Benchmark — Auto RAM/VRAM detection + batched GPU inference"
    )
    parser.add_argument("--model-dir", default="./my_llm_folder",
                        help="Path to HF model folder")
    parser.add_argument("--aidna-path", default=None,
                        help="Direct path to .aidna file (e.g. modal/parent_text_smollm2_360m.aidna or modal/fused_text_child.aidna)")
    parser.add_argument("--tasks", nargs="+",
                        default=["mmlu", "gsm8k", "arc"],
                        choices=["mmlu", "gsm8k", "arc", "ifeval"],
                        help="Tasks to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions per task (None = full dataset)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override auto batch size")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: 'cuda' or 'cpu'")
    parser.add_argument("--subject", type=str, default=None,
                        help="Restrict MMLU to one subject (e.g. anatomy)")
    parser.add_argument("--output-dir", default="./bench_results",
                        help="Folder for results JSON")
    args = parser.parse_args()

    run_full_benchmark(
        model_dir=args.model_dir,
        aidna_path=args.aidna_path,
        tasks=args.tasks,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        subject=args.subject,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
