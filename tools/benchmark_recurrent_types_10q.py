"""
Benchmarking All 6 Recurrent Depth Architectures vs 24-Layer Baseline.
Evaluates 500 questions per category (2,500 questions per model):
  1. Mathematics (GSM8K, 500 Qs)
  2. Python Coding (AST Exec, 500 Qs)
  3. Science & Natural Laws (500 Qs)
  4. World History & Geography (500 Qs)
  5. Language, Grammar & Logic (500 Qs)

Collects for each model:
  - Accuracy per category and overall
  - Parameter count (Total and Active per step)
  - Disk size in MB
  - RAM & VRAM memory footprint
  - Evaluation latency & throughput
"""

import os
import sys
import gc
import time
import json
import math
from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ai_dna.models.recurrent_causal_lm import RecurrentQwenForCausalLM
from tools.benchmark_parallel_batched_vram import (
    load_gsm8k_math_subset,
    generate_coding_subset,
    generate_science_subset,
    generate_history_geo_subset,
    generate_language_logic_subset,
    eval_code_submission,
    eval_math_submission,
    eval_fact_submission,
    BATCH_SIZE,
    DEVICE,
)

OUTPUT_REPORT_PATH = os.path.join(WORKSPACE_ROOT, "outputs", "recurrent_types_10q_comparison_report.json")


def get_process_memory_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def get_vram_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def evaluate_model_large_batch_unified(
    model: Any,
    tokenizer: Any,
    model_name: str,
    categories: Dict[str, List[Dict[str, Any]]],
    is_recurrent: bool = False,
) -> Dict[str, Any]:
    print(f"\n{'='*80}\n  EVALUATING: {model_name} (Recurrent={is_recurrent})\n{'='*80}")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results_by_cat = {}
    total_passed = 0
    total_evaluated = 0
    sample_audits = []
    t0_start = time.time()

    for cat_name, q_list in categories.items():
        t_cat_start = time.time()
        cat_passed = 0
        cat_total = len(q_list)

        # Batch iteration
        for b_start in range(0, cat_total, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, cat_total)
            batch_qs = q_list[b_start:b_end]

            prompts = [q["prompt"] for q in batch_qs]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)

            if cat_name == "coding":
                max_tokens_cat = 32
            elif cat_name == "math":
                max_tokens_cat = 20
            else:
                max_tokens_cat = 12

            with torch.no_grad():
                gen_tokens = model.generate(
                    **enc,
                    max_new_tokens=max_tokens_cat,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            out_tokens = gen_tokens[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)

            for idx_in_batch, (item, gen_str) in enumerate(zip(batch_qs, decoded)):
                target = item["answer"].strip()
                q_type = item.get("type", "fact")

                if q_type == "code":
                    tpl = item.get("template_name", "square")
                    passed, rationale = eval_code_submission(gen_str, tpl)
                elif q_type == "math":
                    passed, rationale = eval_math_submission(gen_str, target)
                else:
                    passed, rationale = eval_fact_submission(gen_str, target)

                if passed:
                    cat_passed += 1

                if b_start == 0 and idx_in_batch < 2:
                    sample_audits.append({
                        "category": cat_name,
                        "prompt": item["prompt"].strip()[:100],
                        "expected": target,
                        "raw_output": gen_str.strip()[:120],
                        "passed": passed,
                        "rationale": rationale,
                    })

        cat_elapsed = time.time() - t_cat_start
        cat_acc = (cat_passed / cat_total) * 100.0 if cat_total > 0 else 0.0
        results_by_cat[cat_name] = {
            "passed": cat_passed,
            "total": cat_total,
            "accuracy_pct": round(cat_acc, 2),
            "elapsed_seconds": round(cat_elapsed, 1),
        }
        total_passed += cat_passed
        total_evaluated += cat_total
        print(f"  - {cat_name.capitalize():<12}: {cat_passed}/{cat_total} ({cat_acc:5.1f}%) in {cat_elapsed:4.1f}s")

    total_elapsed = time.time() - t0_start
    total_acc = (total_passed / total_evaluated) * 100.0 if total_evaluated > 0 else 0.0
    print(f"  TOTAL ACCURACY: {total_passed}/{total_evaluated} ({total_acc:.2f}%) in {total_elapsed:.1f}s")

    return {
        "model_label": model_name,
        "is_recurrent": is_recurrent,
        "total_passed": total_passed,
        "total_evaluated": total_evaluated,
        "total_accuracy_pct": round(total_acc, 2),
        "total_time_seconds": round(total_elapsed, 1),
        "categories": results_by_cat,
        "sample_audits": sample_audits,
    }


def main():
    print("=" * 80)
    print(" 10Q BENCHMARK SUITE: ALL RECURRENT DEPTH TYPES VS BASELINE ")
    print(f" Device: {DEVICE}")
    print("=" * 80)

    # 1. Load 10 questions per category
    print("\nLoading Datasets (10 Questions x 5 Categories = 50 total) ...")
    categories = {
        "math": load_gsm8k_math_subset(10),
        "coding": generate_coding_subset(10),
        "science": generate_science_subset(10),
        "history_geo": generate_history_geo_subset(10),
        "logic": generate_language_logic_subset(10),
    }
    for k, v in categories.items():
        print(f"  - {k.capitalize():<12}: {len(v)} questions")

    models_to_test = [
        {
            "id": 0,
            "name": "Baseline: Full 24-Layer Qwen2.5-0.5B",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "text_models", "qwen2.5-0.5b"),
            "is_recurrent": False,
        },
        {
            "id": 1,
            "name": "Type 1: Step-LoRA (Middle-Band SVD, r=16)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "type_1"),
            "is_recurrent": True,
        },
        {
            "id": 2,
            "name": "Type 2: Step-LoRA (All-Layer SVD, r=16)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "type_2"),
            "is_recurrent": True,
        },
        {
            "id": 3,
            "name": "Type 3: Step-LoRA (Layer 12 Anchor, r=16)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "type_3"),
            "is_recurrent": True,
        },
        {
            "id": 4,
            "name": "Type 4: Pure Recurrent (Middle-Band SVD, r=0)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "type_4"),
            "is_recurrent": True,
        },
        {
            "id": 5,
            "name": "Type 5: Pure Recurrent (All-Layer SVD, r=0)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "type_5"),
            "is_recurrent": True,
        },
        {
            "id": 6,
            "name": "Type 6: Pure Recurrent (Layer 12 Anchor, r=0)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "type_6"),
            "is_recurrent": True,
        },
        {
            "id": 7,
            "name": "Type 7: Fixed Step-LoRA (Layer 0 Anchor, r=896 / 2048 cap)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "test_fixed_layer0"),
            "is_recurrent": True,
        },
        {
            "id": 8,
            "name": "Type 8: Fixed Step-LoRA (Layer 0 Anchor, r=128)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "test_fixed_layer0_r128"),
            "is_recurrent": True,
        },
    ]

    all_results = {}
    if os.path.exists(OUTPUT_REPORT_PATH):
        try:
            with open(OUTPUT_REPORT_PATH, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    for m_info in models_to_test:
        m_id = m_info["id"]
        m_name = m_info["name"]
        m_path = m_info["path"]
        is_rec = m_info["is_recurrent"]

        if m_name in all_results:
            print(f"\n[CACHE HIT] Model '{m_name}' already evaluated. Skipping...")
            continue

        if not os.path.exists(m_path):
            print(f"\n[SKIP] Path does not exist: {m_path}")
            continue

        # Measure disk size
        st_file = os.path.join(m_path, "model.safetensors")
        disk_size_mb = os.path.getsize(st_file) / (1024 * 1024) if os.path.exists(st_file) else 0.0

        # Memory before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ram_before = get_process_memory_mb()
        vram_before = get_vram_memory_mb()

        print(f"\n[{m_id + 1}/{len(models_to_test)}] Loading {m_name} from {m_path} ...")
        t_load_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(m_path)
        if is_rec:
            model = RecurrentQwenForCausalLM.from_pretrained(
                m_path,
                device=str(DEVICE),
                dtype=torch.float32,
            )
            from safetensors import safe_open
            with safe_open(st_file, framework="pt", device="cpu") as f:
                total_params = sum(f.get_tensor(k).numel() for k in f.keys())
        else:
            model = AutoModelForCausalLM.from_pretrained(
                m_path,
                torch_dtype=torch.float32,
            ).to(DEVICE)
            model.eval()
            total_params = sum(p.numel() for p in model.parameters())

        load_elapsed = time.time() - t_load_start

        # Memory after loading
        ram_after = get_process_memory_mb()
        vram_after = get_vram_memory_mb()
        ram_footprint = max(0.0, ram_after - ram_before)
        vram_footprint = max(0.0, vram_after - vram_before)

        print(f"  Loaded in {load_elapsed:.2f}s | Params: {total_params:,} | Disk: {disk_size_mb:.2f} MB | VRAM: {vram_footprint:.1f} MB | RAM: {ram_footprint:.1f} MB")

        # Run 500Q Evaluation across 5 categories
        eval_metrics = evaluate_model_large_batch_unified(
            model=model,
            tokenizer=tokenizer,
            model_name=m_name,
            categories=categories,
            is_recurrent=is_rec,
        )

        eval_metrics["parameters"] = total_params
        eval_metrics["disk_size_mb"] = round(disk_size_mb, 2)
        eval_metrics["ram_footprint_mb"] = round(ram_footprint, 2)
        eval_metrics["vram_footprint_mb"] = round(vram_footprint, 2)
        eval_metrics["load_time_seconds"] = round(load_elapsed, 2)
        all_results[m_name] = eval_metrics

        os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
        with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # Cleanup memory
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save complete report
    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved full 500Q comparison report to: {OUTPUT_REPORT_PATH}")

    # Print Final Comparison Table
    print("\n" + "=" * 130)
    print("  FINAL BENCHMARK COMPARISON: ALL RECURRENT TYPES VS 24-LAYER BASELINE (10Q / CATEGORY)")
    print("=" * 130)
    header = f"{'Model Configuration':<55} | {'Math':<6} | {'Code':<6} | {'Sci':<6} | {'Hist':<6} | {'Logic':<6} | {'Overall':<8} | {'Params':<8} | {'Disk MB':<8} | {'VRAM MB'}"
    print(header)
    print("-" * len(header))

    for m_name, d in all_results.items():
        cats = d["categories"]
        m_acc = f"{cats['math']['accuracy_pct']:.1f}%"
        c_acc = f"{cats['coding']['accuracy_pct']:.1f}%"
        s_acc = f"{cats['science']['accuracy_pct']:.1f}%"
        h_acc = f"{cats['history_geo']['accuracy_pct']:.1f}%"
        l_acc = f"{cats['logic']['accuracy_pct']:.1f}%"
        tot_acc = f"{d['total_accuracy_pct']:.1f}%"
        params_m = f"{d['parameters'] / 1e6:.1f}M"
        disk_mb = f"{d['disk_size_mb']:.1f}"
        vram_mb = f"{d['vram_footprint_mb']:.1f}"
        print(f"{m_name[:55]:<55} | {m_acc:>6} | {c_acc:>6} | {s_acc:>6} | {h_acc:>6} | {l_acc:>6} | {tot_acc:>8} | {params_m:>8} | {disk_mb:>8} | {vram_mb:>7}")

    print("=" * 130)


if __name__ == "__main__":
    main()
