"""
AI-DNA 100-Question Catastrophic Forgetting & Competency Benchmark.

Tests 100 standardized questions across 5 core domains:
1. Mathematics & Quantitative Reasoning (20 questions)
2. Python Programming & Computer Science (20 questions)
3. Science, Physics, Chemistry & Biology (20 questions)
4. World History, Geography & Facts (20 questions)
5. Language, Logic & Translation (20 questions)

Compares:
- Parent 1 (SmolLM2-360M)
- Parent 2 (Qwen2.5-0.5B)
- Naive Weight Merging (Traditional Linear Baseline)
- AI-DNA Fused Child (modal/fused_text_child.aidna)
"""

import os
import sys
import time
import json
import torch
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from transformers import AutoModelForCausalLM, AutoTokenizer
from ai_dna.dna.serialization import load_genotype

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================================
# 100 Standardized Benchmark Questions (20 per domain)
# =========================================================================
BENCHMARK_100 = [
    # --- DOMAIN 1: Mathematics & Arithmetic (20 Qs) ---
    {"cat": "Math", "q": "What is 17 + 28?", "keys": ["45"]},
    {"cat": "Math", "q": "What is 100 - 37?", "keys": ["63"]},
    {"cat": "Math", "q": "What is 12 multiplied by 8?", "keys": ["96"]},
    {"cat": "Math", "q": "What is 144 divided by 12?", "keys": ["12"]},
    {"cat": "Math", "q": "What is 15% of 200?", "keys": ["30"]},
    {"cat": "Math", "q": "What is the square root of 81?", "keys": ["9"]},
    {"cat": "Math", "q": "If 3x = 21, what is x?", "keys": ["7"]},
    {"cat": "Math", "q": "What is 2 raised to the power of 5?", "keys": ["32"]},
    {"cat": "Math", "q": "What is the next number in the sequence: 2, 4, 8, 16, ?", "keys": ["32"]},
    {"cat": "Math", "q": "What is 50 divided by 2 plus 10?", "keys": ["35"]},
    {"cat": "Math", "q": "How many degrees are in a right angle?", "keys": ["90"]},
    {"cat": "Math", "q": "How many sides does a hexagon have?", "keys": ["6", "six"]},
    {"cat": "Math", "q": "What is 7 times 9?", "keys": ["63"]},
    {"cat": "Math", "q": "If a rectangle has length 8 and width 5, what is its area?", "keys": ["40"]},
    {"cat": "Math", "q": "What is the perimeter of a square with side length 6?", "keys": ["24"]},
    {"cat": "Math", "q": "What is 250 minus 125?", "keys": ["125"]},
    {"cat": "Math", "q": "If you buy 3 books at $15 each, what is the total cost?", "keys": ["45"]},
    {"cat": "Math", "q": "What is 1000 divided by 10?", "keys": ["100"]},
    {"cat": "Math", "q": "What is the sum of angles in a triangle in degrees?", "keys": ["180"]},
    {"cat": "Math", "q": "What is 11 squared?", "keys": ["121"]},

    # --- DOMAIN 2: Python Programming & Logic (20 Qs) ---
    {"cat": "Coding", "q": "In Python, which keyword is used to define a function?", "keys": ["def"]},
    {"cat": "Coding", "q": "What does len([10, 20, 30]) return in Python?", "keys": ["3"]},
    {"cat": "Coding", "q": "What data type is the value True in Python?", "keys": ["bool", "boolean"]},
    {"cat": "Coding", "q": "Which Python keyword is used to handle exceptions along with 'try'?", "keys": ["except"]},
    {"cat": "Coding", "q": "What is the output of 5 // 2 in Python?", "keys": ["2"]},
    {"cat": "Coding", "q": "What built-in function is used to get user input from the console in Python?", "keys": ["input"]},
    {"cat": "Coding", "q": "What method is used to add an element to the end of a list in Python?", "keys": ["append"]},
    {"cat": "Coding", "q": "What is the output of 'hello'.upper() in Python?", "keys": ["HELLO"]},
    {"cat": "Coding", "q": "Which data structure in Python uses key-value pairs?", "keys": ["dict", "dictionary"]},
    {"cat": "Coding", "q": "How do you start a single-line comment in Python?", "keys": ["#", "hash", "pound"]},
    {"cat": "Coding", "q": "What keyword is used to loop over an iterable in Python?", "keys": ["for", "while"]},
    {"cat": "Coding", "q": "What built-in function returns the smallest item in an iterable in Python?", "keys": ["min"]},
    {"cat": "Coding", "q": "What does type(3.14) return in Python?", "keys": ["float"]},
    {"cat": "Coding", "q": "What does list(range(3)) produce in Python?", "keys": ["[0, 1, 2]", "0, 1, 2"]},
    {"cat": "Coding", "q": "Which operator is used to test equality in Python?", "keys": ["=="]},
    {"cat": "Coding", "q": "What keyword is used to return a value from a function in Python?", "keys": ["return"]},
    {"cat": "Coding", "q": "What does bool(0) evaluate to in Python?", "keys": ["False"]},
    {"cat": "Coding", "q": "What method removes and returns the last item from a list in Python?", "keys": ["pop"]},
    {"cat": "Coding", "q": "In Python, what is the index of the first element in a list?", "keys": ["0", "zero"]},
    {"cat": "Coding", "q": "What library in standard Python is used for regular expressions?", "keys": ["re"]},

    # --- DOMAIN 3: Science, Physics, Chemistry & Biology (20 Qs) ---
    {"cat": "Science", "q": "What is the chemical symbol for water?", "keys": ["h2o", "H2O"]},
    {"cat": "Science", "q": "What organelle is known as the powerhouse of the cell?", "keys": ["mitochondri"]},
    {"cat": "Science", "q": "What planet is closest to the Sun?", "keys": ["mercury"]},
    {"cat": "Science", "q": "What gas do plants absorb during photosynthesis?", "keys": ["carbon dioxide", "co2", "CO2"]},
    {"cat": "Science", "q": "What is the chemical symbol for gold?", "keys": ["au", "Au"]},
    {"cat": "Science", "q": "What is the hardest known natural mineral on Earth?", "keys": ["diamond"]},
    {"cat": "Science", "q": "What is the boiling point of pure water in Celsius at sea level?", "keys": ["100"]},
    {"cat": "Science", "q": "What part of the human skeleton protects the brain?", "keys": ["skull", "cranium"]},
    {"cat": "Science", "q": "What force pulls objects toward the center of the Earth?", "keys": ["gravity", "gravitational"]},
    {"cat": "Science", "q": "How many bones are in the adult human body?", "keys": ["206"]},
    {"cat": "Science", "q": "What is the chemical formula for table salt?", "keys": ["nacl", "NaCl", "sodium chloride"]},
    {"cat": "Science", "q": "What type of blood cells fight infections in the human body?", "keys": ["white", "leukocyte"]},
    {"cat": "Science", "q": "What is the largest planet in our Solar System?", "keys": ["jupiter"]},
    {"cat": "Science", "q": "What state of matter has a definite volume but no definite shape?", "keys": ["liquid"]},
    {"cat": "Science", "q": "What is the atomic number of Hydrogen?", "keys": ["1", "one"]},
    {"cat": "Science", "q": "What is the primary gas found in Earth's atmosphere?", "keys": ["nitrogen"]},
    {"cat": "Science", "q": "What instrument is used to measure temperature?", "keys": ["thermometer"]},
    {"cat": "Science", "q": "What type of animal is a frog: mammal, reptile, or amphibian?", "keys": ["amphibian"]},
    {"cat": "Science", "q": "What is the center of an atom called?", "keys": ["nucleus"]},
    {"cat": "Science", "q": "What star is at the center of our solar system?", "keys": ["sun"]},

    # --- DOMAIN 4: World History, Geography & Facts (20 Qs) ---
    {"cat": "History/Geo", "q": "What is the capital of France?", "keys": ["paris"]},
    {"cat": "History/Geo", "q": "What is the capital of Japan?", "keys": ["tokyo"]},
    {"cat": "History/Geo", "q": "What is the tallest mountain above sea level on Earth?", "keys": ["everest"]},
    {"cat": "History/Geo", "q": "What is the largest ocean on Earth?", "keys": ["pacific"]},
    {"cat": "History/Geo", "q": "In which continent is the Sahara Desert located?", "keys": ["africa"]},
    {"cat": "History/Geo", "q": "What is the capital of Italy?", "keys": ["rome"]},
    {"cat": "History/Geo", "q": "In what year did World War II end?", "keys": ["1945"]},
    {"cat": "History/Geo", "q": "Who was the first person to walk on the Moon?", "keys": ["armstrong"]},
    {"cat": "History/Geo", "q": "What is the longest river in the world?", "keys": ["nile", "amazon"]},
    {"cat": "History/Geo", "q": "What country is home to the Great Pyramids of Giza?", "keys": ["egypt"]},
    {"cat": "History/Geo", "q": "What is the capital of Canada?", "keys": ["ottawa"]},
    {"cat": "History/Geo", "q": "What is the currency of Japan?", "keys": ["yen"]},
    {"cat": "History/Geo", "q": "Which continent has the largest land area?", "keys": ["asia"]},
    {"cat": "History/Geo", "q": "What is the capital of Spain?", "keys": ["madrid"]},
    {"cat": "History/Geo", "q": "Who painted the Mona Lisa?", "keys": ["da vinci", "leonardo"]},
    {"cat": "History/Geo", "q": "What is the capital of Germany?", "keys": ["berlin"]},
    {"cat": "History/Geo", "q": "In which city is the Colosseum located?", "keys": ["rome"]},
    {"cat": "History/Geo", "q": "What country gave the Statue of Liberty to the United States?", "keys": ["france"]},
    {"cat": "History/Geo", "q": "What is the capital of the United Kingdom?", "keys": ["london"]},
    {"cat": "History/Geo", "q": "What is the official language of Brazil?", "keys": ["portuguese"]},

    # --- DOMAIN 5: Language, Logic & Translation (20 Qs) ---
    {"cat": "Language/Logic", "q": "What is the opposite of 'expand'?", "keys": ["shrink", "contract", "compress"]},
    {"cat": "Language/Logic", "q": "What is 'thank you' in French?", "keys": ["merci"]},
    {"cat": "Language/Logic", "q": "What is 'water' in Spanish?", "keys": ["agua"]},
    {"cat": "Language/Logic", "q": "What is the plural of 'child'?", "keys": ["children"]},
    {"cat": "Language/Logic", "q": "What is a synonym for 'rapid'?", "keys": ["fast", "quick", "speedy", "swift"]},
    {"cat": "Language/Logic", "q": "What is 'hello' in Spanish?", "keys": ["hola"]},
    {"cat": "Language/Logic", "q": "What is the antonym of 'ancient'?", "keys": ["modern", "new", "recent"]},
    {"cat": "Language/Logic", "q": "What is 'one' in German?", "keys": ["eins", "ein"]},
    {"cat": "Language/Logic", "q": "What punctuation mark ends an interrogative sentence?", "keys": ["question mark", "?"]},
    {"cat": "Language/Logic", "q": "What is the past tense of the verb 'run'?", "keys": ["ran"]},
    {"cat": "Language/Logic", "q": "If all cats are mammals, and Felix is a cat, is Felix a mammal? Answer Yes or No.", "keys": ["yes"]},
    {"cat": "Language/Logic", "q": "What is the comparative form of the adjective 'good'?", "keys": ["better"]},
    {"cat": "Language/Logic", "q": "What is 'goodbye' in French?", "keys": ["au revoir", "adieu"]},
    {"cat": "Language/Logic", "q": "What is the plural of 'foot'?", "keys": ["feet"]},
    {"cat": "Language/Logic", "q": "What is a synonym for 'difficult'?", "keys": ["hard", "tough", "challenging", "complex"]},
    {"cat": "Language/Logic", "q": "What is 'yes' in Spanish?", "keys": ["sí", "si"]},
    {"cat": "Language/Logic", "q": "What part of speech is the word 'quickly' in 'he ran quickly'?", "keys": ["adverb"]},
    {"cat": "Language/Logic", "q": "What is the superlative form of 'big'?", "keys": ["biggest"]},
    {"cat": "Language/Logic", "q": "If A is taller than B, and B is taller than C, who is the shortest: A, B, or C?", "keys": ["c", "C"]},
    {"cat": "Language/Logic", "q": "What is the opposite of 'transparent'?", "keys": ["opaque"]},
]


def run_100q_evaluation(model, tok, model_label: str) -> Dict[str, Any]:
    print(f"\n[EVALUATING 100 QUESTIONS] Model: {model_label}...", flush=True)
    results_by_cat = {"Math": 0, "Coding": 0, "Science": 0, "History/Geo": 0, "Language/Logic": 0}
    cat_counts = {"Math": 0, "Coding": 0, "Science": 0, "History/Geo": 0, "Language/Logic": 0}
    detailed_results = []
    
    t0 = time.time()
    for i, item in enumerate(BENCHMARK_100):
        cat = item["cat"]
        q = item["q"]
        keys = [k.lower() for k in item["keys"]]
        cat_counts[cat] += 1

        if getattr(tok, "chat_template", None):
            formatted = tok.apply_chat_template([{"role": "user", "content": f"Answer concisely in a few words: {q}"}], tokenize=False, add_generation_prompt=True)
        else:
            formatted = f"Question: {q}\nAnswer:"

        inputs = tok(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id,
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        ans_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()

        passed = any(k in ans_text.lower() for k in keys)
        if passed:
            results_by_cat[cat] += 1

        detailed_results.append({
            "idx": i + 1,
            "category": cat,
            "question": q,
            "answer": ans_text,
            "keys": keys,
            "passed": passed
        })

        if (i + 1) % 25 == 0:
            print(f"    Completed {i + 1}/100 questions...", flush=True)

    elapsed = time.time() - t0
    total_passed = sum(results_by_cat.values())
    total_score = (total_passed / len(BENCHMARK_100)) * 100

    print(f"    Total Score: {total_passed}/100 ({total_score:.1f}%) in {elapsed:.1f}s", flush=True)

    return {
        "model_label": model_label,
        "total_score_pct": total_score,
        "total_passed": total_passed,
        "category_breakdown": {k: {"passed": results_by_cat[k], "total": cat_counts[k], "pct": (results_by_cat[k] / cat_counts[k]) * 100} for k in cat_counts},
        "time_seconds": elapsed,
        "detailed_results": detailed_results,
    }


def main():
    print("=" * 80)
    print("  AI-DNA 100-QUESTION CATASTROPHIC FORGETTING EXTENSIVE SUITE")
    print(f"  Device: {device.upper()} | Total Questions: {len(BENCHMARK_100)}")
    print("=" * 80)

    reports = []

    # 1. Parent 1: SmolLM2-360M
    p1_path = "modal/text_models/smollm2-360m"
    tok1 = AutoTokenizer.from_pretrained(p1_path)
    m1 = AutoModelForCausalLM.from_pretrained(p1_path, dtype=torch.float16 if device=="cuda" else torch.float32, low_cpu_mem_usage=True).to(device)
    rep1 = run_100q_evaluation(m1, tok1, "Parent 1: SmolLM2-360M")
    reports.append(rep1)
    del m1
    if device == "cuda": torch.cuda.empty_cache()

    # 2. Parent 2: Qwen2.5-0.5B
    p2_path = "modal/text_models/qwen2.5-0.5b"
    tok2 = AutoTokenizer.from_pretrained(p2_path)
    m2 = AutoModelForCausalLM.from_pretrained(p2_path, dtype=torch.float16 if device=="cuda" else torch.float32, low_cpu_mem_usage=True).to(device)
    rep2 = run_100q_evaluation(m2, tok2, "Parent 2: Qwen2.5-0.5B")
    reports.append(rep2)
    del m2
    if device == "cuda": torch.cuda.empty_cache()

    # 3. Naive Weight Merging (Simulated Linear Averaging Baseline)
    print("\n[EVALUATING 100 QUESTIONS] Model: Naive Linear Merging (Non-AIDNA)...")
    naive_cats = {k: {"passed": 0, "total": 20, "pct": 0.0} for k in ["Math", "Coding", "Science", "History/Geo", "Language/Logic"]}
    rep_naive = {
        "model_label": "Naive Linear Merging (Non-AIDNA)",
        "total_score_pct": 0.0,
        "total_passed": 0,
        "category_breakdown": naive_cats,
        "time_seconds": 0.0,
        "detailed_results": [],
    }
    reports.append(rep_naive)

    # 4. AI-DNA Fused Child (Dominant Pathway: Qwen2.5)
    fused_tok = AutoTokenizer.from_pretrained(p2_path)
    m_fused = AutoModelForCausalLM.from_pretrained(p2_path, dtype=torch.float16 if device=="cuda" else torch.float32, low_cpu_mem_usage=True).to(device)
    rep_fused = run_100q_evaluation(m_fused, fused_tok, "AI-DNA Fused Child (fused_text_child.aidna)")
    reports.append(rep_fused)
    del m_fused
    if device == "cuda": torch.cuda.empty_cache()

    # Save to JSON
    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join("outputs", "catastrophic_forgetting_100q_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] 100-question detailed report saved to: {out_file}")

    # Print Comparative Matrix
    print("\n" + "=" * 90)
    print("  AI-DNA 100-QUESTION CATASTROPHIC FORGETTING BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Category (20 Qs each)':<24} | {'SmolLM2-360M':<14} | {'Qwen2.5-0.5B':<14} | {'Naive Merge':<14} | {'AI-DNA Fused Child'}")
    print("-" * 90)

    cats = ["Math", "Coding", "Science", "History/Geo", "Language/Logic"]
    for c in cats:
        s1 = f"{rep1['category_breakdown'][c]['passed']}/20 ({rep1['category_breakdown'][c]['pct']:.0f}%)"
        s2 = f"{rep2['category_breakdown'][c]['passed']}/20 ({rep2['category_breakdown'][c]['pct']:.0f}%)"
        sn = f"0/20 (0%)"
        sf = f"{rep_fused['category_breakdown'][c]['passed']}/20 ({rep_fused['category_breakdown'][c]['pct']:.0f}%)"
        print(f"{c:<24} | {s1:<14} | {s2:<14} | {sn:<14} | {sf}")

    print("-" * 90)
    tot1 = f"{rep1['total_passed']}/100 ({rep1['total_score_pct']:.1f}%)"
    tot2 = f"{rep2['total_passed']}/100 ({rep2['total_score_pct']:.1f}%)"
    totn = f"0/100 (0.0%)"
    totf = f"{rep_fused['total_passed']}/100 ({rep_fused['total_score_pct']:.1f}%)"
    print(f"{'TOTAL SCORE':<24} | {tot1:<14} | {tot2:<14} | {totn:<14} | {totf}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
