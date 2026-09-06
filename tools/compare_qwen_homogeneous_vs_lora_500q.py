"""
Compares Homogeneous Fusion vs Asymmetric Layer-Depth Decoupled LoRA Fusion
when BOTH models use Qwen2.5-0.5B as the primary host/backbone.
Evaluates 500 questions per category (2,500 questions each, 5,000 evaluations total):
  1. Mathematics (GSM8K/Math, 500 Qs)
  2. Python Coding (AST + Sandboxed Exec, 500 Qs)
  3. Science & Natural Laws (500 Qs)
  4. World History & Geography (500 Qs)
  5. Language, Grammar & Logic (500 Qs)
"""

import os
import sys
import time
import json
import torch
from typing import Dict, Any

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.benchmark_parallel_batched_vram import (
    load_gsm8k_math_subset,
    generate_coding_subset,
    generate_science_subset,
    generate_history_geo_subset,
    generate_language_logic_subset,
    evaluate_model_large_batch,
    DEVICE,
)

OUTPUT_REPORT_PATH = os.path.join(WORKSPACE_ROOT, "outputs", "qwen_homogeneous_vs_lora_500q_report.json")


def main():
    print(f"Loading Benchmark Datasets (500 Qs per Category = 2,500 Qs total) ...")
    categories = {
        "math": load_gsm8k_math_subset(500),
        "coding": generate_coding_subset(500),
        "science": generate_science_subset(500),
        "history_geo": generate_history_geo_subset(500),
        "logic": generate_language_logic_subset(500),
    }
    for k, v in categories.items():
        print(f"  - {k.capitalize()}: {len(v)} questions")

    models_to_test = [
        {
            "name": "Qwen Host + Homogeneous Fusion (Uniform Convex Averaging across all 24 layers)",
            "path": os.path.join(WORKSPACE_ROOT, "modal", "fused_qwen_homogeneous"),
            "method": "Homogeneous Lineage Convex Averaging (Method 4 style on Qwen)",
        },
        {
            "name": "Qwen Host + Asymmetric Layer-Depth Decoupled LoRA Instinct Fusion (`my_llm_folder`)",
            "path": os.path.join(WORKSPACE_ROOT, "my_llm_folder"),
            "method": "Asymmetric Layer-Depth Decoupled LoRA (Canonical Standard)",
        },
    ]

    all_results = {}

    for m_info in models_to_test:
        m_name = m_info["name"]
        m_path = m_info["path"]
        print(f"\nEvaluating: {m_name} from {m_path} ...")
        
        tokenizer = AutoTokenizer.from_pretrained(m_path)
        model = AutoModelForCausalLM.from_pretrained(
            m_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(DEVICE)
        model.eval()

        res = evaluate_model_large_batch(model, tokenizer, m_name, categories)
        res["method"] = m_info["method"]
        all_results[m_name] = res

        # Clear VRAM between runs
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 90)
    print("  FINAL 500Q HEAD-TO-HEAD COMPARISON (BOTH WITH QWEN AS HOST / PARENT)")
    print("=" * 90)
    header = f"{'Model Configuration':<45} | {'Math':<7} | {'Code':<7} | {'Sci':<7} | {'Hist':<7} | {'Logic':<7} | {'TOTAL':<11} | {'Accuracy'}"
    print(header)
    print("-" * len(header))

    for m_name, data in all_results.items():
        cats = data["categories"]
        m_str = f"{cats['math']['passed']}/{cats['math']['total']}"
        c_str = f"{cats['coding']['passed']}/{cats['coding']['total']}"
        s_str = f"{cats['science']['passed']}/{cats['science']['total']}"
        h_str = f"{cats['history_geo']['passed']}/{cats['history_geo']['total']}"
        l_str = f"{cats['logic']['passed']}/{cats['logic']['total']}"
        tot_str = f"{data['total_passed']}/{data['total_evaluated']}"
        acc_str = f"{data['total_accuracy_pct']:.2f}%"

        short_name = m_name[:44]
        print(f"{short_name:<45} | {m_str:<7} | {c_str:<7} | {s_str:<7} | {h_str:<7} | {l_str:<7} | {tot_str:<11} | {acc_str}")

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed comparison report to: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
