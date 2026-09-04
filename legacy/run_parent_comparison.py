"""
Direct Parent Models vs AI-DNA Fused Child Benchmark Comparison (Direct from .aidna)
=====================================================================================
Compares all downloaded parent models and the fused child directly from their .aidna genetic containers
on core reasoning tasks:
- MMLU (Knowledge)
- GSM8K (Math reasoning)
- IFEval (Instruction compliance)
"""

import os
import sys
import time
import json
import argparse
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ai_dna.dna.serialization import load_genotype
from convert_aidna_to_safetensors import extract_weights_from_genotype

device = "cuda" if torch.cuda.is_available() else "cpu"

AIDNA_MODELS = [
    ("Parent: SmolLM2-135M [.aidna]", "modal/parent_text.aidna", "modal/text_model"),
    ("Parent: OPT-125M [.aidna]", "modal/parent_text_opt_125m.aidna", "modal/text_models/opt-125m"),
    ("Parent: SmolLM2-360M [.aidna]", "modal/parent_text_smollm2_360m.aidna", "modal/text_models/smollm2-360m"),
    ("Parent: Qwen2.5-0.5B [.aidna]", "modal/parent_text_qwen2.5_0.5b.aidna", "modal/text_models/qwen2.5-0.5b"),
    ("Parent: TinyLlama-1.1B [.aidna]", "modal/parent_text_tinyllama_1.1b.aidna", "modal/text_models/tinyllama-1.1b"),
    ("AI-DNA Fused Child [.aidna]", "modal/fused_text_child.aidna", "my_llm_folder"),
]

HF_MODELS = [
    ("Parent: SmolLM2-135M [HF]", "modal/text_model"),
    ("Parent: OPT-125M [HF]", "modal/text_models/opt-125m"),
    ("Parent: SmolLM2-360M [HF]", "modal/text_models/smollm2-360m"),
    ("Parent: Qwen2.5-0.5B [HF]", "modal/text_models/qwen2.5-0.5b"),
    ("Parent: TinyLlama-1.1B [HF]", "modal/text_models/tinyllama-1.1b"),
    ("AI-DNA Fused Child [Folder]", "my_llm_folder"),
]

EVAL_QUESTIONS = [
    # MMLU / Science & Knowledge
    {
        "type": "mmlu",
        "cat": "Science",
        "q": "Which of the following elements has the highest electronegativity?\nA) Sodium\nB) Chlorine\nC) Fluorine\nD) Oxygen\nAnswer (one letter only):",
        "expected": "C"
    },
    {
        "type": "mmlu",
        "cat": "Computer Science",
        "q": "In computer architecture, what does ALU stand for?\nA) Arithmetic Logic Unit\nB) Asynchronous Linear Utility\nC) Algorithmic Logic Utility\nD) Array Linear Unit\nAnswer (one letter only):",
        "expected": "A"
    },
    {
        "type": "mmlu",
        "cat": "Physics",
        "q": "Which law states that the total entropy of an isolated system always increases over time?\nA) First Law of Thermodynamics\nB) Second Law of Thermodynamics\nC) Third Law of Thermodynamics\nD) Zeroth Law of Thermodynamics\nAnswer (one letter only):",
        "expected": "B"
    },
    {
        "type": "mmlu",
        "cat": "Biology",
        "q": "What primary function does ATP synthase perform in cellular respiration?\nA) Hydrolyzing glucose\nB) Phosphorylating ADP to ATP\nC) Pumping protons out of mitochondria\nD) Oxidizing NADH\nAnswer (one letter only):",
        "expected": "B"
    },
    # GSM8K / Math Reasoning
    {
        "type": "gsm8k",
        "cat": "Math",
        "q": "A bakery sells boxes of donuts for $12 each. If a customer buys 4 boxes and pays with a $100 bill, how much change should they receive?\nAnswer with the final number:",
        "expected": "52"
    },
    {
        "type": "gsm8k",
        "cat": "Math",
        "q": "A train travels at a constant speed of 60 miles per hour for 2.5 hours. How many miles does it travel?\nAnswer with the final number:",
        "expected": "150"
    },
    {
        "type": "gsm8k",
        "cat": "Math",
        "q": "If a rectangle has length 15 cm and width 8 cm, what is its perimeter in cm?\nAnswer with the final number:",
        "expected": "46"
    },
    {
        "type": "gsm8k",
        "cat": "Math",
        "q": "A store offers a 20% discount on an item originally priced at $80. What is the discounted price?\nAnswer with the final number:",
        "expected": "64"
    },
    # IFEval / Instruction Adherence
    {
        "type": "ifeval",
        "cat": "Instruction",
        "q": "Write a 3-word sentence about water.",
        "expected": "3words"
    },
    {
        "type": "ifeval",
        "cat": "Instruction",
        "q": "Write the word CONFIRMED in all uppercase letters.",
        "expected": "CONFIRMED"
    },
]


def load_model_from_aidna(aidna_path: str, fallback_tokenizer_dir: str, device: str):
    """Loads model weights and architecture directly from an .aidna genetic container."""
    g = load_genotype(aidna_path)
    weights, _ = extract_weights_from_genotype(g, device=torch.device("cpu"))

    cfg = None
    if fallback_tokenizer_dir and os.path.exists(fallback_tokenizer_dir):
        try:
            cfg = AutoConfig.from_pretrained(fallback_tokenizer_dir)
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

    # Build model and inject weights directly from .aidna (filtering matching parameter shapes)
    dtype = torch.float16 if device == "cuda" else torch.float32
    mdl = AutoModelForCausalLM.from_config(cfg).to(dtype=dtype)
    model_state = mdl.state_dict()
    filtered_weights = {}
    for k, v in weights.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_weights[k] = v

    mdl.load_state_dict(filtered_weights, strict=False)
    mdl.to(device)
    mdl.eval()

    # Load tokenizer
    tok_dir = fallback_tokenizer_dir if os.path.exists(fallback_tokenizer_dir) else "my_llm_folder"
    tok = AutoTokenizer.from_pretrained(tok_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    return mdl, tok, g


def evaluate(use_aidna: bool = True):
    models = AIDNA_MODELS if use_aidna else HF_MODELS
    mode_str = "DIRECT FROM .AIDNA GENOTYPES" if use_aidna else "FROM HUGGINGFACE FOLDERS"

    print("=" * 92)
    print(f"  AI-DNA: COMPARING ALL PARENT MODELS WITH FUSED CHILD ({mode_str})")
    print(f"  Device: {device.upper()} | Number of Models: {len(models)} | Total Questions: {len(EVAL_QUESTIONS)}")
    print("=" * 92)

    results = {}

    for item in models:
        if use_aidna:
            label, aidna_path, fallback_dir = item
            src_path = aidna_path
        else:
            label, src_path = item
            fallback_dir = src_path

        print(f"\n[+] Testing {label} (source: {src_path})...", flush=True)
        if not os.path.exists(src_path):
            print(f"    [!] Missing path: {src_path}")
            continue

        t0 = time.time()
        try:
            if use_aidna:
                mdl, tok, g = load_model_from_aidna(src_path, fallback_dir, device)
                print(f"    [AI-DNA] Loaded {len(g.dna_instinct.genetic_parameters)} tensors directly from {os.path.basename(src_path)}")
            else:
                tok = AutoTokenizer.from_pretrained(src_path)
                if tok.pad_token_id is None:
                    tok.pad_token_id = tok.eos_token_id or 0
                mdl = AutoModelForCausalLM.from_pretrained(
                    src_path,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                ).to(device)
                mdl.eval()
        except Exception as e:
            print(f"    [ERROR] Failed to load {label}: {e}")
            continue

        params = sum(p.numel() for p in mdl.parameters())
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id or 0

        # Format all prompts for parallel GPU batching
        formatted_prompts = []
        for q_obj in EVAL_QUESTIONS:
            prompt = q_obj["q"]
            if getattr(tok, "chat_template", None):
                try:
                    fp = tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    fp = prompt
            else:
                fp = prompt
            formatted_prompts.append(fp)

        # Run parallel batch inference in a single GPU pass
        enc = tok(formatted_prompts, padding=True, return_tensors="pt").to(device)
        input_len = enc["input_ids"].shape[1]

        with torch.inference_mode():
            out = mdl.generate(
                **enc,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

        responses = [tok.decode(seq[input_len:], skip_special_tokens=True).strip() for seq in out]

        correct = 0
        details = []

        for q_obj, resp in zip(EVAL_QUESTIONS, responses):
            is_correct = False
            if q_obj["type"] == "mmlu":
                exp = q_obj["expected"]
                first_letter = ""
                for c in resp.upper():
                    if c in ["A", "B", "C", "D"]:
                        first_letter = c
                        break
                is_correct = (first_letter == exp) or (f"ANSWER: {exp}" in resp.upper()) or (f"{exp})" in resp.upper())
            elif q_obj["type"] == "gsm8k":
                exp = q_obj["expected"]
                is_correct = exp in resp
            elif q_obj["type"] == "ifeval":
                if q_obj["expected"] == "3words":
                    is_correct = len(resp.strip().split()) in [3, 4]
                elif q_obj["expected"] == "CONFIRMED":
                    is_correct = "CONFIRMED" in resp

            if is_correct:
                correct += 1

            details.append({
                "category": q_obj["cat"],
                "q": q_obj["q"].split("\n")[0],
                "expected": q_obj["expected"],
                "response": resp[:70].replace("\n", " "),
                "correct": is_correct,
            })

        acc = (correct / len(EVAL_QUESTIONS)) * 100.0
        elapsed = time.time() - t0
        results[label] = {
            "source_file": src_path,
            "params": f"{params/1e6:.1f}M",
            "accuracy": round(acc, 1),
            "correct": correct,
            "total": len(EVAL_QUESTIONS),
            "time_seconds": round(elapsed, 2),
            "details": details,
        }
        print(f"    -> Score: {correct}/{len(EVAL_QUESTIONS)} ({acc:.1f}%) in {elapsed:.1f}s", flush=True)

        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()

    # Matrix Table Output
    print("\n" + "=" * 96)
    print(f"  BENCHMARK COMPARISON MATRIX: PARENT MODELS vs AI-DNA FUSED CHILD ({mode_str})")
    print("=" * 96)
    print(f"  {'Model / Source':<32} | {'Params':<9} | {'Overall':<9} | {'Correct':<8} | {'Latency':<9} | Status")
    print("  " + "-" * 92)

    for lbl, r in results.items():
        status = "Child (Merged)" if "Fused" in lbl else "Parent (Origin)"
        print(f"  {lbl:<32} | {r['params']:^9} | {r['accuracy']:>6.1f}%   | {r['correct']:>2}/{r['total']:<2}    | {r['time_seconds']:>6.1f}s   | {status}")

    print("=" * 96)

    # Question-by-Question Comparison
    print("\n" + "=" * 96)
    print("  SAMPLE ANSWERS ACROSS MODELS DIRECTLY FROM .AIDNA")
    print("=" * 96)
    for q_idx, q_item in enumerate(EVAL_QUESTIONS[:4]):
        print(f"\nQ{q_idx+1} [{q_item['cat']}]: {q_item['q'].split(chr(10))[0]}")
        print(f"Expected: {q_item['expected']}")
        print("  " + "-" * 88)
        for lbl, r in results.items():
            dt = r["details"][q_idx]
            tag = "[PASS]" if dt["correct"] else "[FAIL]"
            print(f"  {tag:<6} {lbl:<30}: {dt['response'][:50]}")

    os.makedirs("bench_results", exist_ok=True)
    out_json = "bench_results/parent_vs_child_comparison_aidna.json" if use_aidna else "bench_results/parent_vs_child_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full comparison JSON saved to: {os.path.abspath(out_json)}")


def main():
    parser = argparse.ArgumentParser(description="Direct Parent vs AI-DNA Fused Child Benchmark")
    parser.add_argument("--hf-mode", action="store_true", help="Load from Hugging Face model folders instead of .aidna")
    args = parser.parse_args()

    evaluate(use_aidna=not args.hf_mode)


if __name__ == "__main__":
    main()
