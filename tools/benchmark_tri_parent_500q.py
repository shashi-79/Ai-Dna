"""
Comprehensive Tri-Parent Fusion Benchmark Suite (500 Questions per Category).
100% NON-REGEX DETERMINISTIC EVALUATION:
  - Code: Python AST parse + dynamic sandbox execution with input/output unit tests
  - Math: Numeric token parsing + floating-point mathematical equivalence
  - Science / Facts: Token-level exact phrase sequence matching
  - History / Geo: Token-level exact capital matching
  - Logic / Antonyms: Token-level exact antonym matching
Evaluates 6 Models across 5 Categories (500 Qs each = 2,500 Qs per model, 15,000 evaluations total):
  1. Parent 1: SmolLM2-360M (Coding specialist)
  2. Parent 2: Qwen2.5-0.5B (Reasoning foundation)
  3. Parent 3: TinyLlama-1.1B (LLaMA conversational)
  4. Dual-Parent Champion: Method 2 (LoRA Instinct Fused Child)
  5. Tri-Parent LoRA Fused Child: Dense Safetensors in modal/fused_tri_parent_lora
  6. Tri-Parent MoE Fused Child: 3-Expert Mixture-of-Experts
"""

import os
import sys
import time
import json
import ast
from typing import List, Dict, Any, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from ai_dna.models.tri_moe_child import build_tri_fused_moe_model
except ImportError:
    build_tri_fused_moe_model = None

NUM_PER_CATEGORY = 500
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_JSON_PATH = os.path.join(WORKSPACE_ROOT, "outputs", "tri_parent_500q_report.json")


# =========================================================================
# 1. Non-Regex Deterministic Evaluators
# =========================================================================
def extract_pure_function_block(code_str: str, func_name: str) -> str:
    lines = code_str.split("\n")
    func_lines = []
    found_def = False
    for line in lines:
        stripped = line.strip()
        if not found_def:
            if stripped.startswith(f"def {func_name}(") or stripped.startswith("def "):
                found_def = True
                func_lines.append(line)
        else:
            if stripped.startswith("```") or stripped.startswith(">>>") or stripped.startswith("print(") or stripped.startswith("# Test") or stripped.startswith("assert ") or stripped.startswith("if __name__"):
                break
            if line and not line[0].isspace() and not stripped.startswith("#"):
                break
            func_lines.append(line)
    return "\n".join(func_lines) if func_lines else code_str.split("```")[0].strip()


def eval_code_submission(code_str: str, template_name: str) -> Tuple[bool, str]:
    """Evaluates Python code via AST parsing and sandbox execution without regex."""
    clean_code = extract_pure_function_block(code_str, template_name)

    try:
        ast.parse(clean_code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    sandbox = {}
    try:
        exec(clean_code, sandbox)
    except Exception as e:
        return False, f"ExecError: {e}"

    tests = {
        "square": ("square", [(4, 16), (-3, 9), (0, 0)]),
        "is_even": ("is_even", [(8, True), (7, False), (0, True)]),
        "get_length": ("get_length", [([1, 2, 3], 3), ([], 0), ([10], 1)]),
        "reverse_string": ("reverse_string", [("hello", "olleh"), ("", ""), ("a", "a")]),
        "find_max": ("find_max", [(3, 7, 7), (10, 2, 10), (-5, -1, -1)]),
        "double_nums": ("double_nums", [([1, 2], [2, 4]), ([], [])]),
        "first_elem": ("first_elem", [([10, 20], 10), (["a"], "a")]),
        "is_positive": ("is_positive", [(5, True), (-2, False), (0, False)]),
        "cube": ("cube", [(3, 27), (-2, -8), (0, 0)]),
        "concat": ("concat", [("foo", "bar", "foobar"), ("", "x", "x")]),
    }

    if template_name not in tests:
        return True, "AST parsed successfully."

    func_name, cases = tests[template_name]
    if func_name not in sandbox or not callable(sandbox[func_name]):
        return False, f"Function '{func_name}' not defined or not callable."

    fn = sandbox[func_name]
    for case in cases:
        try:
            if len(case) == 2:
                arg, expected = case
                res = fn(arg)
            else:
                arg1, arg2, expected = case
                res = fn(arg1, arg2)
            if res != expected:
                return False, f"Failed on input {case[:-1]}: expected {expected}, got {res}"
        except Exception as e:
            return False, f"Runtime error on input {case[:-1]}: {e}"

    return True, f"Passed all {len(cases)} unit tests for {func_name}()"


def eval_math_submission(response_str: str, expected_num_str: str) -> Tuple[bool, str]:
    """Evaluates mathematical answer by multi-candidate numeric parsing without regex."""
    try:
        expected = float(expected_num_str)
    except ValueError:
        return False, f"Invalid expected value: {expected_num_str}"

    candidates = []

    # 1. Look for numbers in first non-empty line (direct answer location)
    lines = [ln.strip() for ln in response_str.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    if lines:
        clean_first = lines[0].replace("$", " ").replace(",", "").replace("=", " ").replace(":", " ")
        for t in clean_first.split():
            t_c = t.strip(".:;!?()[]\"'")
            if t_c.lstrip("-").replace(".", "", 1).isdigit():
                try:
                    candidates.append(float(t_c))
                except ValueError:
                    pass

    # 2. Look for \boxed{...}
    if "\\boxed{" in response_str:
        boxed_part = response_str.split("\\boxed{", 1)[1].split("}", 1)[0]
        b_clean = boxed_part.replace("$", " ").replace(",", "").strip()
        for t in b_clean.split():
            t_c = t.strip(".:;!?()[]\"'")
            if t_c.lstrip("-").replace(".", "", 1).isdigit():
                try:
                    candidates.append(float(t_c))
                except ValueError:
                    pass

    # 3. Look for "final answer:", "answer:", "is:"
    for kw in ["final answer:", "answer:", "is:"]:
        if kw in response_str.lower():
            after_kw = response_str.lower().split(kw, 1)[1]
            for t in after_kw.split()[:5]:
                t_c = t.strip(".:;!?()[]\"'")
                if t_c.lstrip("-").replace(".", "", 1).isdigit():
                    try:
                        candidates.append(float(t_c))
                    except ValueError:
                        pass

    # 4. Look across all tokens
    clean_all = response_str.replace("$", " ").replace(",", "").replace("=", " ").replace(":", " ")
    for t in clean_all.split():
        t_c = t.strip(".:;!?()[]\"'")
        if t_c.lstrip("-").replace(".", "", 1).isdigit():
            try:
                candidates.append(float(t_c))
            except ValueError:
                pass

    for c in candidates:
        if abs(c - expected) < 1e-4:
            return True, f"Parsed answer {c} == expected {expected}"

    if candidates:
        return False, f"Parsed answer {candidates[-1]} != expected {expected} (tested: {candidates[:3]})"
    return False, "No numeric tokens found in response"


def eval_fact_submission(response_str: str, target_phrase: str) -> Tuple[bool, str]:
    """Evaluates exact word/phrase presence without regex."""
    words = [w.strip(".,;:!?()[]\"'").lower() for w in response_str.split()]
    target_words = target_phrase.lower().split()

    if not words or not target_words:
        return False, "Empty response or target"

    n_w = len(words)
    n_t = len(target_words)
    for i in range(n_w - n_t + 1):
        if words[i:i + n_t] == target_words:
            return True, f"Found exact phrase '{target_phrase}' at token index {i}"
    return False, f"Phrase '{target_phrase}' not found in token sequence: {words[:12]}..."


# =========================================================================
# 2. Dataset Generation (500 Questions per Category)
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
                            tokens = a.strip().split()
                            ans = tokens[-1].replace(",", "").strip(".$#")
                            if q not in seen:
                                seen.add(q)
                                questions.append({
                                    "prompt": f"Solve this math problem step by step: {q}\nFinal Answer:",
                                    "answer": ans,
                                    "type": "math",
                                })
                                if len(questions) >= limit:
                                    break
                    except Exception:
                        continue

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
        ("Write a clean python function `square(x)` to compute the square of x:", "square"),
        ("Write a clean python function `is_even(x)` to check if a number x is even:", "is_even"),
        ("Write a clean python function `get_length(lst)` to calculate the length of list lst:", "get_length"),
        ("Write a clean python function `reverse_string(s)` to reverse a string s:", "reverse_string"),
        ("Write a clean python function `find_max(a, b)` to find maximum of a and b:", "find_max"),
        ("Write a clean python function `double_nums(nums)` to double all numbers in list nums:", "double_nums"),
        ("Write a clean python function `first_elem(lst)` to return the first element of lst:", "first_elem"),
        ("Write a clean python function `is_positive(x)` to check if x is positive:", "is_positive"),
        ("Write a clean python function `cube(n)` to calculate cube of n:", "cube"),
        ("Write a clean python function `concat(a, b)` to join strings a and b:", "concat"),
    ]
    for i in range(limit):
        prompt_txt, template_name = templates[i % len(templates)]
        questions.append({
            "prompt": f"{prompt_txt}\n```python\n",
            "answer": template_name,
            "template_name": template_name,
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
            "type": "fact",
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
            "type": "fact",
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
            "type": "fact",
        })
    return questions


# =========================================================================
# 3. Model Evaluation with Non-Regex Deterministic Scorers
# =========================================================================
def evaluate_model_on_500q(
    model: Any,
    tokenizer: Any,
    model_name: str,
    categories: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    print(f"\n{'='*75}\n  RUNNING 2,500-Q BENCHMARK (NON-REGEX EVALUATION): {model_name}\n{'='*75}")
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
                    max_new_tokens=36,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            out_tokens = gen_tokens[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)

            for idx_in_batch, (item, gen_str) in enumerate(zip(batch_qs, decoded)):
                target = item["answer"].strip()
                q_type = item.get("type", "fact")

                # Non-regex deterministic evaluation
                if q_type == "code":
                    tpl = item.get("template_name", "square")
                    passed, rationale = eval_code_submission(gen_str, tpl)
                elif q_type == "math":
                    passed, rationale = eval_math_submission(gen_str, target)
                else:
                    passed, rationale = eval_fact_submission(gen_str, target)

                if passed:
                    cat_passed += 1

                # Record audit samples for manual inspection (first 3 per category)
                if b_start == 0 and idx_in_batch < 3:
                    sample_audits.append({
                        "category": cat_name,
                        "prompt": item["prompt"].strip(),
                        "expected": target,
                        "raw_output": gen_str.strip()[:150],
                        "passed": passed,
                        "rationale": rationale,
                    })

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
        "sample_audits": sample_audits,
    }


# =========================================================================
# 4. Main Benchmark Runner
# =========================================================================
def main():
    print("\n" + "=" * 80)
    print("  AI-DNA TRI-PARENT FUSION BENCHMARK (100% NON-REGEX DETERMINISTIC EVAL)")
    print(f"  Device: {DEVICE} | Batch Size: {BATCH_SIZE} | Total Per Model: 2,500 Questions")
    print("=" * 80)

    categories = {
        "Math": load_gsm8k_math_subset(NUM_PER_CATEGORY),
        "Coding": generate_coding_subset(NUM_PER_CATEGORY),
        "Science": generate_science_subset(NUM_PER_CATEGORY),
        "History/Geo": generate_history_geo_subset(NUM_PER_CATEGORY),
        "Language/Logic": generate_language_logic_subset(NUM_PER_CATEGORY),
    }

    report = []

    path_smol = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "smollm2-360m")
    path_qwen = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "qwen2.5-0.5b")
    path_tiny = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "tinyllama-1.1b")
    path_lora_dual = os.path.join(WORKSPACE_ROOT, "modal", "fused_lora_child")
    path_lora_tri = os.path.join(WORKSPACE_ROOT, "modal", "fused_tri_parent_lora")

    tok_qwen = AutoTokenizer.from_pretrained(path_qwen)

    # 1. Parent 1 Baseline: SmolLM2-360M
    tok_smol = AutoTokenizer.from_pretrained(path_smol)
    m_smol = AutoModelForCausalLM.from_pretrained(path_smol, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_smol = evaluate_model_on_500q(m_smol, tok_smol, "Parent 1: SmolLM2-360M", categories)
    report.append(rep_smol)
    del m_smol
    torch.cuda.empty_cache()

    # 2. Parent 2 Baseline: Qwen2.5-0.5B
    m_qwen = AutoModelForCausalLM.from_pretrained(path_qwen, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_qwen = evaluate_model_on_500q(m_qwen, tok_qwen, "Parent 2: Qwen2.5-0.5B", categories)
    report.append(rep_qwen)
    del m_qwen
    torch.cuda.empty_cache()

    # 3. Parent 3 Baseline: TinyLlama-1.1B
    tok_tiny = AutoTokenizer.from_pretrained(path_tiny)
    m_tiny = AutoModelForCausalLM.from_pretrained(path_tiny, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_tiny = evaluate_model_on_500q(m_tiny, tok_tiny, "Parent 3: TinyLlama-1.1B", categories)
    report.append(rep_tiny)
    del m_tiny
    torch.cuda.empty_cache()

    # 4. Dual-Parent Champion: Method 2 (LoRA Instinct Fused Child)
    m_dual = AutoModelForCausalLM.from_pretrained(path_lora_dual, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_dual = evaluate_model_on_500q(m_dual, tok_qwen, "Dual-Parent Champion (LoRA Instinct)", categories)
    report.append(rep_dual)
    del m_dual
    torch.cuda.empty_cache()

    # 5. Tri-Parent LoRA Fused Child (Dense Safetensors)
    m_tri_lora = AutoModelForCausalLM.from_pretrained(path_lora_tri, torch_dtype=torch.bfloat16).to(DEVICE)
    rep_tri_lora = evaluate_model_on_500q(m_tri_lora, tok_qwen, "Tri-Parent LoRA Fused Child (Dense)", categories)
    report.append(rep_tri_lora)
    del m_tri_lora
    torch.cuda.empty_cache()

    # 6. Tri-Parent MoE Fused Child (3-Expert MoE Architecture)
    m_tri_moe, tok_tri_moe = build_tri_fused_moe_model(path_qwen, path_smol, path_tiny, device=DEVICE)
    rep_tri_moe = evaluate_model_on_500q(m_tri_moe, tok_tri_moe, "Tri-Parent MoE Fused Child (3-Expert)", categories)
    report.append(rep_tri_moe)
    del m_tri_moe
    torch.cuda.empty_cache()

    # Save complete JSON report
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 95)
    print("  TRI-PARENT MULTI-MODEL COMPARATIVE PERFORMANCE MATRIX (100% NON-REGEX EVAL)")
    print("=" * 95)
    header = f"| {'Model':<40} | {'Math':<8} | {'Coding':<8} | {'Science':<8} | {'Hist/Geo':<8} | {'Logic':<8} | {'TOTAL':<10} |"
    print(header)
    print("|" + "-" * 42 + "|" + ("-" * 10 + "|") * 5 + "-" * 12 + "|")
    for r in report:
        c = r["categories"]
        m_str = f"{c['Math']['accuracy_pct']:.1f}%"
        co_str = f"{c['Coding']['accuracy_pct']:.1f}%"
        s_str = f"{c['Science']['accuracy_pct']:.1f}%"
        h_str = f"{c['History/Geo']['accuracy_pct']:.1f}%"
        l_str = f"{c['Language/Logic']['accuracy_pct']:.1f}%"
        tot_str = f"{r['total_accuracy_pct']:.2f}%"
        print(f"| {r['model_label']:<40} | {m_str:<8} | {co_str:<8} | {s_str:<8} | {h_str:<8} | {l_str:<8} | {tot_str:<10} |")
    print("=" * 95)
    print(f"Report saved to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
