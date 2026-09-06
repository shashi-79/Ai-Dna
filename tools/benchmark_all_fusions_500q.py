"""
Comprehensive Multi-Parent Fusion Benchmark Suite (500 Questions per Category).
Evaluates all 4 fusion methodologies, their hybrid combination, and parent baselines across:
  - Math (500 Qs)
  - Coding (500 Qs)
  - Science (500 Qs)
  - History/Geo (500 Qs)
  - Language/Logic (500 Qs)
Total: 2,500 questions per model on CUDA using parallel batching.
"""

import os
import sys
import time
import json
import re
from typing import List, Dict, Any, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from ai_dna.models.moe_child import build_fused_moe_model
except ImportError:
    build_fused_moe_model = None

NUM_PER_CATEGORY = 500
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_JSON_PATH = os.path.join(WORKSPACE_ROOT, "outputs", "all_fusions_500q_report.json")


# =========================================================================
# 1. Dataset Generation (500 Questions per Category)
# =========================================================================
def load_gsm8k_math_subset(limit: int = 500) -> List[Dict[str, Any]]:
    questions = []
    seen = set()
    data_files = [
        os.path.join(WORKSPACE_ROOT, "ai-dna-data", "adaptation", "gsm8k", "gsm8k_train.jsonl"),
        os.path.join(WORKSPACE_ROOT, "ai-dna-data", "adaptation", "math", "math_train.jsonl"),
    ]
    for fp in data_files:
        if os.path.exists(fp) and len(questions) < limit:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        q = obj.get("question") or obj.get("problem")
                        a = obj.get("answer") or obj.get("solution")
                        if q and a:
                            m = re.findall(r"####\s*(-?\d+[\d,]*)", a)
                            ans = m[-1].replace(",", "") if m else a.strip().split()[-1]
                            if q not in seen:
                                seen.add(q)
                                questions.append({"prompt": f"Solve this math problem: {q}\nAnswer:", "answer": ans, "type": "math"})
                                if len(questions) >= limit:
                                    break
                    except Exception:
                        continue

    # Synthetic generator if fewer than limit
    idx = 0
    while len(questions) < limit:
        idx += 1
        a_val = (idx * 17) % 499 + 11
        b_val = (idx * 23) % 499 + 7
        ans = a_val + b_val
        questions.append({
            "prompt": f"Solve this math problem: What is {a_val} plus {b_val}?\nAnswer:",
            "answer": str(ans),
            "type": "math",
        })
    return questions[:limit]


def generate_coding_subset(limit: int = 500) -> List[Dict[str, Any]]:
    questions = []
    templates = [
        ("Write a python function to compute the square of x:", "def square(x):\n    return x * x", ["def square", "return"]),
        ("Write a python function to check if a number x is even:", "def is_even(x):\n    return x % 2 == 0", ["def is_even", "return"]),
        ("Write a python function to calculate the length of list lst:", "def get_length(lst):\n    return len(lst)", ["def get_length", "return len"]),
        ("Write a python function to reverse a string s:", "def reverse_string(s):\n    return s[::-1]", ["def reverse_string", "return"]),
        ("Write a python function to find maximum of a and b:", "def find_max(a, b):\n    return max(a, b)", ["def find_max", "return"]),
        ("Write a python function to double all numbers in list nums:", "def double_nums(nums):\n    return [x * 2 for x in nums]", ["def double_nums", "return"]),
        ("Write a python function to return the first element of lst:", "def first_elem(lst):\n    return lst[0]", ["def first_elem", "return"]),
        ("Write a python function to check if x is positive:", "def is_positive(x):\n    return x > 0", ["def is_positive", "return"]),
        ("Write a python function to calculate cube of n:", "def cube(n):\n    return n ** 3", ["def cube", "return"]),
        ("Write a python function to join strings a and b:", "def concat(a, b):\n    return a + b", ["def concat", "return"]),
    ]
    for i in range(limit):
        prompt_txt, solution_txt, check_keys = templates[i % len(templates)]
        questions.append({
            "prompt": f"{prompt_txt}\n",
            "answer": solution_txt,
            "check_keys": check_keys,
            "type": "code",
        })
    return questions


def generate_science_subset(limit: int = 500) -> List[Dict[str, Any]]:
    elements = [
        ("Hydrogen", "1"), ("Helium", "2"), ("Lithium", "3"), ("Beryllium", "4"),
        ("Boron", "5"), ("Carbon", "6"), ("Nitrogen", "7"), ("Oxygen", "8"),
        ("Fluorine", "9"), ("Neon", "10"), ("Sodium", "11"), ("Magnesium", "12"),
        ("Aluminum", "13"), ("Silicon", "14"), ("Phosphorus", "15"), ("Sulfur", "16"),
        ("Chlorine", "17"), ("Argon", "18"), ("Potassium", "19"), ("Calcium", "20"),
        ("Iron", "26"), ("Copper", "29"), ("Zinc", "30"), ("Silver", "47"), ("Gold", "79"),
    ]
    questions = []
    for i in range(limit):
        elem, num = elements[i % len(elements)]
        questions.append({
            "prompt": f"What is the atomic number of {elem}?\nAnswer:",
            "answer": num,
            "type": "exact",
        })
    return questions


def generate_history_geo_subset(limit: int = 500) -> List[Dict[str, Any]]:
    facts = [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"),
        ("Italy", "Rome"), ("Spain", "Madrid"), ("Canada", "Ottawa"),
        ("Australia", "Canberra"), ("Brazil", "Brasilia"), ("India", "New Delhi"),
        ("Egypt", "Cairo"), ("Russia", "Moscow"), ("China", "Beijing"),
        ("United Kingdom", "London"), ("South Korea", "Seoul"), ("Mexico", "Mexico City"),
        ("Argentina", "Buenos Aires"), ("Turkey", "Ankara"), ("Greece", "Athens"),
        ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Poland", "Warsaw"),
    ]
    questions = []
    for i in range(limit):
        country, cap = facts[i % len(facts)]
        questions.append({
            "prompt": f"What is the capital city of {country}?\nAnswer:",
            "answer": cap,
            "type": "exact",
        })
    return questions


def generate_language_logic_subset(limit: int = 500) -> List[Dict[str, Any]]:
    antonyms = [
        ("hot", "cold"), ("up", "down"), ("fast", "slow"), ("happy", "sad"),
        ("light", "dark"), ("big", "small"), ("high", "low"), ("rich", "poor"),
        ("strong", "weak"), ("good", "bad"), ("early", "late"), ("hard", "soft"),
        ("true", "false"), ("win", "lose"), ("clean", "dirty"), ("young", "old"),
    ]
    questions = []
    for i in range(limit):
        word, opp = antonyms[i % len(antonyms)]
        questions.append({
            "prompt": f"What is the exact antonym of '{word}'?\nAnswer:",
            "answer": opp,
            "type": "exact",
        })
    return questions


# =========================================================================
# 2. Evaluation Harness
# =========================================================================
def evaluate_model_on_500q(
    model: Any,
    tokenizer: Any,
    model_name: str,
    categories: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    print(f"\n{'='*75}\n  RUNNING 2,500-Q BENCHMARK (500 x 5): {model_name}\n{'='*75}")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results_by_cat = {}
    total_passed = 0
    total_evaluated = 0
    t0_start = time.time()

    for cat_name, q_list in categories.items():
        t_cat_start = time.time()
        cat_passed = 0
        cat_total = len(q_list)

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

            with torch.no_grad():
                gen_tokens = model.generate(
                    **enc,
                    max_new_tokens=24,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Strip input tokens
            out_tokens = gen_tokens[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)

            for item, gen_str in zip(batch_qs, decoded):
                target = item["answer"].strip()
                q_type = item.get("type", "exact")

                if q_type == "math":
                    m = re.findall(r"-?\d+", gen_str)
                    passed = (m and m[0] == target) or (target in gen_str.strip())
                elif q_type == "code":
                    check_keys = item.get("check_keys", ["def", "return"])
                    passed = all(k in gen_str for k in check_keys)
                else:
                    gen_clean = re.sub(r"[^\w\s]", "", gen_str.lower())
                    target_clean = re.sub(r"[^\w\s]", "", target.lower())
                    passed = target_clean in gen_clean

                if passed:
                    cat_passed += 1

        dt_cat = time.time() - t_cat_start
        acc_pct = (cat_passed / cat_total) * 100.0
        results_by_cat[cat_name] = {
            "passed": cat_passed,
            "total": cat_total,
            "accuracy_pct": round(acc_pct, 2),
            "elapsed_seconds": round(dt_cat, 1),
        }
        total_passed += cat_passed
        total_evaluated += cat_total
        print(f"  [{cat_name.upper():<14}] {cat_passed:>3}/{cat_total:<3} ({acc_pct:>5.1f}%) in {dt_cat:>5.1f}s")

    total_time = time.time() - t0_start
    total_acc = (total_passed / total_evaluated) * 100.0
    print(f"  TOTAL ACCURACY: {total_passed}/{total_evaluated} ({total_acc:.2f}%) in {total_time:.1f}s")

    return {
        "model_label": model_name,
        "total_passed": total_passed,
        "total_evaluated": total_evaluated,
        "total_accuracy_pct": round(total_acc, 2),
        "total_time_seconds": round(total_time, 1),
        "categories": results_by_cat,
    }


# =========================================================================
# 3. Main Benchmark Runner
# =========================================================================
def main():
    print("\n" + "=" * 80)
    print("  AI-DNA MULTI-PARENT FUSION BENCHMARK (500 QUESTIONS PER CATEGORY)")
    print(f"  Device: {DEVICE} | Batch Size: {BATCH_SIZE} | Total Per Model: 2,500 Questions")
    print("=" * 80)

    # 1. Prepare Question Categories (500 each)
    categories = {
        "Math": load_gsm8k_math_subset(NUM_PER_CATEGORY),
        "Coding": generate_coding_subset(NUM_PER_CATEGORY),
        "Science": generate_science_subset(NUM_PER_CATEGORY),
        "History/Geo": generate_history_geo_subset(NUM_PER_CATEGORY),
        "Language/Logic": generate_language_logic_subset(NUM_PER_CATEGORY),
    }

    report = []

    # ---------------------------------------------------------------------
    # 1. Parent 1 Baseline: SmolLM2-360M
    # ---------------------------------------------------------------------
    path_smol = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "smollm2-360m")
    tok_smol = AutoTokenizer.from_pretrained(path_smol)
    m_smol = AutoModelForCausalLM.from_pretrained(path_smol, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_smol = evaluate_model_on_500q(m_smol, tok_smol, "Parent 1: SmolLM2-360M", categories)
    report.append(rep_smol)
    del m_smol
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 2. Parent 2 Baseline: Qwen2.5-0.5B
    # ---------------------------------------------------------------------
    path_qwen = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "qwen2.5-0.5b")
    tok_qwen = AutoTokenizer.from_pretrained(path_qwen)
    m_qwen = AutoModelForCausalLM.from_pretrained(path_qwen, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_qwen = evaluate_model_on_500q(m_qwen, tok_qwen, "Parent 2: Qwen2.5-0.5B", categories)
    report.append(rep_qwen)
    del m_qwen
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 3. Method 1: AI-DNA MoE Fused Child (Dual-Expert Architecture)
    # ---------------------------------------------------------------------
    m_moe, tok_moe = build_fused_moe_model(path_qwen, path_smol, device=DEVICE, is_hybrid=False)
    rep_moe = evaluate_model_on_500q(m_moe, tok_moe, "Method 1: AI-DNA MoE Fused Child", categories)
    report.append(rep_moe)
    del m_moe
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 4. Method 2: LoRA Instinct-Filter Fused Child
    # ---------------------------------------------------------------------
    path_lora = os.path.join(WORKSPACE_ROOT, "modal", "fused_lora_child")
    m_lora = AutoModelForCausalLM.from_pretrained(path_lora, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_lora = evaluate_model_on_500q(m_lora, tok_qwen, "Method 2: LoRA Instinct Fused Child", categories)
    report.append(rep_lora)
    del m_lora
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 5. Method 3: Dense Continuous SVD Energy Fused Child (my_llm_folder)
    # ---------------------------------------------------------------------
    path_m3 = os.path.join(WORKSPACE_ROOT, "my_llm_folder")
    m_m3 = AutoModelForCausalLM.from_pretrained(path_m3, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_m3 = evaluate_model_on_500q(m_m3, tok_qwen, "Method 3: Dense SVD Energy Fused Child (my_llm_folder)", categories)
    report.append(rep_m3)
    del m_m3
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 6. Method 4: Homogeneous Lineage Fused Child (SmolLM2-135M + 360M)
    # ---------------------------------------------------------------------
    path_homo = os.path.join(WORKSPACE_ROOT, "modal", "fused_homogeneous_smollm2")
    m_homo = AutoModelForCausalLM.from_pretrained(path_homo, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_homo = evaluate_model_on_500q(m_homo, tok_smol, "Method 4: Homogeneous Lineage Fused Child", categories)
    report.append(rep_homo)
    del m_homo
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 7. Method 5: Combined Hybrid (MoE + Outlier Attention Blend)
    # ---------------------------------------------------------------------
    m_hybrid, tok_hybrid = build_fused_moe_model(path_qwen, path_smol, device=DEVICE, is_hybrid=True, hybrid_alpha=0.03)
    rep_hybrid = evaluate_model_on_500q(m_hybrid, tok_hybrid, "Method 5: Combined Hybrid (MoE + Outlier Attention Blend)", categories)
    report.append(rep_hybrid)
    del m_hybrid
    torch.cuda.empty_cache()

    # Save complete JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 85)
    print("  COMPLETE 500-Q/CATEGORY (2,500 TOTAL) COMPARATIVE PERFORMANCE MATRIX")
    print("=" * 85)
    header = f"| {'Model':<35} | {'Math':<8} | {'Coding':<8} | {'Science':<8} | {'Hist/Geo':<8} | {'Logic':<8} | {'TOTAL':<10} |"
    print(header)
    print("|" + "-" * 37 + "|" + ("-" * 10 + "|") * 5 + "-" * 12 + "|")
    for r in report:
        c = r["categories"]
        m_str = f"{c['Math']['accuracy_pct']:.1f}%"
        co_str = f"{c['Coding']['accuracy_pct']:.1f}%"
        s_str = f"{c['Science']['accuracy_pct']:.1f}%"
        h_str = f"{c['History/Geo']['accuracy_pct']:.1f}%"
        l_str = f"{c['Language/Logic']['accuracy_pct']:.1f}%"
        tot_str = f"{r['total_accuracy_pct']:.2f}%"
        print(f"| {r['model_label']:<35} | {m_str:<8} | {co_str:<8} | {s_str:<8} | {h_str:<8} | {l_str:<8} | {tot_str:<10} |")
    print("=" * 85)
    print(f"Report saved to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
