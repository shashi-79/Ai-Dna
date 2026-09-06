"""
Verification and 10Q Benchmark for Two-Stage Layer-First Recurrent Fusion.
Fuses all 3-4 text models:
  - Primary: modal/text_models/qwen2.5-0.5b (24 layers, 896 dim)
  - Donor 1: modal/text_models/smollm2-360m (32 layers, code/algorithm specialist)
  - Donor 2: modal/text_models/tinyllama-1.1b (22 layers, knowledge specialist)
  - Donor 3: modal/text_model (30 layers, general specialist)
Then executes 10 questions per category across Math, Code, Science, History/Geo, Logic.
"""

import os
import sys
import gc
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ai_dna.evolution.fusion import build_recurrent_depth_model
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
    DEVICE,
)

FUSED_4MODEL_DIR = os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "fused_4model_layer_first_type7")
OUTPUT_REPORT_PATH = os.path.join(WORKSPACE_ROOT, "outputs", "two_stage_layer_first_fusion_10q_report.json")


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


def evaluate_model_10q(
    model,
    tokenizer,
    model_name: str,
    categories: dict,
    is_recurrent: bool = False,
) -> dict:
    print(f"\n{'='*75}\n[BENCHMARK] Evaluating: {model_name} (Recurrent={is_recurrent})\n{'='*75}")
    category_results = {}
    sample_audits = []
    total_passed = 0
    total_tested = 0
    t_start = time.time()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for cat_name, q_list in categories.items():
        cat_t0 = time.time()
        cat_passed = 0
        cat_tested = len(q_list)
        prompts = [q["prompt"] for q in q_list]
        tokenizer.padding_side = "left"

        if cat_name == "coding":
            max_tokens_cat = 32
        elif cat_name == "math":
            max_tokens_cat = 20
        else:
            max_tokens_cat = 12

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)

        with torch.no_grad():
            if is_recurrent:
                gen_ids = model.generate(
                    input_ids=enc["input_ids"],
                    max_new_tokens=max_tokens_cat,
                    pad_token_id=pad_id,
                )
            else:
                gen_ids = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_tokens_cat,
                    pad_token_id=pad_id,
                    do_sample=False,
                )

        in_lens = [len(ids) for ids in enc["input_ids"]]
        raw_outputs = []
        for i, full_out in enumerate(gen_ids):
            out_tokens = full_out[in_lens[i]:]
            text = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
            raw_outputs.append(text)

        for i, q in enumerate(q_list):
            pred_text = raw_outputs[i]
            target = q.get("answer", "").strip()
            q_type = q.get("type", "fact")

            if q_type == "code":
                tpl = q.get("template_name", "square")
                is_correct, rationale = eval_code_submission(pred_text, tpl)
            elif q_type == "math":
                is_correct, rationale = eval_math_submission(pred_text, target)
            else:
                is_correct, rationale = eval_fact_submission(pred_text, target)

            if is_correct:
                cat_passed += 1

            if i < 2:
                sample_audits.append({
                    "category": cat_name,
                    "prompt": q["prompt"][:80],
                    "target": str(target)[:40],
                    "generated": pred_text[:80].replace("\n", "\\n"),
                    "is_correct": bool(is_correct),
                    "rationale": str(rationale)[:80],
                })

        cat_acc = (cat_passed / cat_tested) * 100.0 if cat_tested > 0 else 0.0
        cat_time = time.time() - cat_t0
        category_results[cat_name] = {
            "passed": cat_passed,
            "total": cat_tested,
            "accuracy_pct": round(cat_acc, 1),
            "elapsed_seconds": round(cat_time, 2),
        }
        total_passed += cat_passed
        total_tested += cat_tested
        print(f"  -> {cat_name.upper():<12} | Accuracy: {cat_acc:5.1f}% ({cat_passed}/{cat_tested}) in {cat_time:.1f}s")

    total_acc = (total_passed / total_tested) * 100.0 if total_tested > 0 else 0.0
    total_elapsed = time.time() - t_start

    return {
        "model_label": model_name,
        "is_recurrent": is_recurrent,
        "total_passed": total_passed,
        "total_evaluated": total_tested,
        "total_accuracy_pct": round(total_acc, 1),
        "total_time_seconds": round(total_elapsed, 1),
        "categories": category_results,
        "sample_audits": sample_audits,
    }


def step1_fuse_all_4_models():
    print("\n" + "=" * 80)
    print("STEP 1: FUSING ALL 4 TEXT MODELS VIA TWO-STAGE LAYER-FIRST RECURRENT FUSION")
    print("=" * 80)

    if os.path.exists(os.path.join(FUSED_4MODEL_DIR, "model.safetensors")):
        print(f"  [CACHE HIT] Fused model already exists at {FUSED_4MODEL_DIR}. Skipping fusion.")
        manifest_path = os.path.join(FUSED_4MODEL_DIR, "recurrent_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                print(f"  Existing Manifest:\n{json.dumps(manifest, indent=2)}")
                return manifest
        return {}

    primary_dir = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "qwen2.5-0.5b")
    donors = [
        {
            "path": os.path.join(WORKSPACE_ROOT, "modal", "text_models", "smollm2-360m"),
            "weight": 0.02,
            "specialization": "code",
        },
        {
            "path": os.path.join(WORKSPACE_ROOT, "modal", "text_models", "tinyllama-1.1b"),
            "weight": 0.015,
            "specialization": "knowledge",
        },
        {
            "path": os.path.join(WORKSPACE_ROOT, "modal", "text_model"),
            "weight": 0.01,
            "specialization": "general",
        },
    ]

    print(f"  Primary Backbone: {primary_dir}")
    print(f"  Donors ({len(donors)} models):")
    for d in donors:
        print(f"    - {d['path']} (weight={d['weight']}, spec={d['specialization']})")
    print(f"  Output Directory: {FUSED_4MODEL_DIR}")

    t0 = time.time()
    manifest = build_recurrent_depth_model(
        primary_dir=primary_dir,
        output_dir=FUSED_4MODEL_DIR,
        donors=donors,
        rank=2048,
        donor_rank=16,
        alpha=0.015,
        outlier_threshold=6.0,
        device="cpu",
    )
    t_elapsed = time.time() - t0
    print(f"[Fusion Complete] Took {t_elapsed:.2f}s. Manifest:\n{json.dumps(manifest, indent=2)}")
    return manifest


def step2_benchmark_all():
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING 10Q BENCHMARK (MATH, CODE, SCI, HIST, LOGIC) ACROSS MODELS")
    print("=" * 80)

    limit = 10
    categories = {
        "math": load_gsm8k_math_subset(limit=limit),
        "coding": generate_coding_subset(limit=limit),
        "science": generate_science_subset(limit=limit),
        "history_geo": generate_history_geo_subset(limit=limit),
        "logic": generate_language_logic_subset(limit=limit),
    }

    models_to_test = [
        {
            "id": 0,
            "name": "Baseline: Full 24-Layer Qwen2.5-0.5B",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "text_models", "qwen2.5-0.5b"),
            "is_recurrent": False,
        },
        {
            "id": 1,
            "name": "Type 7: Single-Model Recurrent (Layer 0, r=896)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types", "test_fixed_layer0"),
            "is_recurrent": True,
        },
        {
            "id": 2,
            "name": "Two-Stage 4-Model Fused Recurrent (Qwen+SmolLM2+TinyLlama+SmolLM)",
            "path": FUSED_4MODEL_DIR,
            "is_recurrent": True,
        },
    ]

    all_results = {}
    from safetensors import safe_open

    for m_info in models_to_test:
        m_id = m_info["id"]
        m_name = m_info["name"]
        m_path = m_info["path"]
        is_rec = m_info["is_recurrent"]

        if not os.path.exists(m_path):
            print(f"\n[SKIP] Path not found: {m_path}")
            continue

        st_file = os.path.join(m_path, "model.safetensors")
        disk_size_mb = os.path.getsize(st_file) / (1024 * 1024) if os.path.exists(st_file) else 0.0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ram_before = get_process_memory_mb()
        vram_before = get_vram_memory_mb()

        print(f"\n[{m_id + 1}/{len(models_to_test)}] Loading {m_name} ...")
        t_load_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(m_path)
        if is_rec:
            model = RecurrentQwenForCausalLM.from_pretrained(
                m_path,
                device=str(DEVICE),
                dtype=torch.float32,
            )
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
        ram_after = get_process_memory_mb()
        vram_after = get_vram_memory_mb()
        ram_footprint = max(0.0, ram_after - ram_before)
        vram_footprint = max(0.0, vram_after - vram_before)

        print(f"  Loaded in {load_elapsed:.2f}s | Params: {total_params:,} | Disk: {disk_size_mb:.2f} MB | VRAM: {vram_footprint:.1f} MB | RAM: {ram_footprint:.1f} MB")

        eval_metrics = evaluate_model_10q(
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

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 125)
    print("  TWO-STAGE LAYER-FIRST 4-MODEL FUSED RECURRENT BENCHMARK COMPARISON TABLE")
    print("=" * 125)
    print(f"{'Model Configuration':<58} | {'Math':<6} | {'Code':<6} | {'Sci':<6} | {'Hist':<6} | {'Logic':<6} | {'Overall':<8} | {'Params':<8} | {'Disk MB':<8} | {'VRAM MB':<8}")
    print("-" * 125)
    for name, r in all_results.items():
        cats = r["categories"]
        m_acc = cats.get("math", {}).get("accuracy_pct", 0.0)
        c_acc = cats.get("coding", {}).get("accuracy_pct", 0.0)
        s_acc = cats.get("science", {}).get("accuracy_pct", 0.0)
        h_acc = cats.get("history_geo", {}).get("accuracy_pct", 0.0)
        l_acc = cats.get("logic", {}).get("accuracy_pct", 0.0)
        ov = r["total_accuracy_pct"]
        p_str = f"{r['parameters']/1e6:.1f}M"
        d_str = f"{r['disk_size_mb']:.1f}"
        v_str = f"{r['vram_footprint_mb']:.1f}"
        print(f"{name:<58} | {m_acc:5.1f}% | {c_acc:5.1f}% | {s_acc:5.1f}% | {h_acc:5.1f}% | {l_acc:5.1f}% | {ov:7.1f}% | {p_str:>8} | {d_str:>8} | {v_str:>8}")
    print("=" * 125)


if __name__ == "__main__":
    step1_fuse_all_4_models()
    step2_benchmark_all()
