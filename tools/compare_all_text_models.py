"""
Multi-Model, Multi-Category Inference Comparison Suite.
Evaluates all 5 open-source LLMs across 5 core evaluation categories:
1. Math & Step-by-Step Reasoning
2. Python Code Generation
3. World Knowledge & Fact Retrieval
4. Creative & Concise Explanation
5. Multilingual Translation
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

MODELS = [
    ("SmolLM2-135M", "modal/text_model"),
    ("OPT-125M", "modal/text_models/opt-125m"),
    ("SmolLM2-360M", "modal/text_models/smollm2-360m"),
    ("Qwen2.5-0.5B", "modal/text_models/qwen2.5-0.5b"),
    ("TinyLlama-1.1B", "modal/text_models/tinyllama-1.1b"),
]

CATEGORIES = [
    {
        "id": "math_reasoning",
        "title": "Category 1: Math & Logical Reasoning",
        "prompt": "If a train travels at 60 mph for 2.5 hours, how far does it travel? Show the step-by-step calculation.",
    },
    {
        "id": "code_generation",
        "title": "Category 2: Python Code Generation",
        "prompt": "Write a clean Python function `is_palindrome(s: str) -> bool` that returns True if string s is a palindrome, ignoring spaces and case.",
    },
    {
        "id": "world_knowledge",
        "title": "Category 3: World Knowledge & Fact Retrieval",
        "prompt": "What is the capital of Australia, and what is the difference between an asteroid and a meteorite?",
    },
    {
        "id": "concise_explanation",
        "title": "Category 4: Concise Explanation (for a 10-year-old)",
        "prompt": "Explain what an artificial neural network is in two simple, clear sentences for a 10-year-old child.",
    },
    {
        "id": "multilingual",
        "title": "Category 5: Multilingual Translation",
        "prompt": "Translate this sentence into Spanish and French:\n'Technology and artificial intelligence are shaping the future of humanity.'",
    },
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print("  AI-DNA MULTI-MODEL, MULTI-CATEGORY INFERENCE BENCHMARK")
    print(f"  Device: {device.upper()} | Models: {len(MODELS)} | Categories: {len(CATEGORIES)}")
    print("=" * 85)

    all_results: Dict[str, Dict[str, Any]] = {cat["id"]: {"title": cat["title"], "prompt": cat["prompt"], "models": {}} for cat in CATEGORIES}

    for model_name, model_dir in MODELS:
        print(f"\n[+] Loading {model_name} from {model_dir}...", flush=True)
        t_load_start = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)
            load_time = time.time() - t_load_start
            print(f"    Loaded in {load_time:.2f}s.", flush=True)
        except Exception as e:
            print(f"    [ERROR] Failed to load {model_name}: {e}")
            continue

        for cat in CATEGORIES:
            cat_id = cat["id"]
            user_prompt = cat["prompt"]

            # Apply chat template if model is an instruction-tuned model
            if getattr(tok, "chat_template", None):
                formatted_input = tok.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_input = f"Instruction: {user_prompt}\nAnswer:\n"

            inputs = tok(formatted_input, return_tensors="pt").to(device)
            input_token_count = inputs.input_ids.shape[1]

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=95,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id,
                )

            if device == "cuda":
                torch.cuda.synchronize()
            gen_time = time.time() - t0

            output_ids = outputs[0][input_token_count:]
            new_tokens = len(output_ids)
            decoded_text = tok.decode(output_ids, skip_special_tokens=True).strip()

            tok_per_sec = (new_tokens / gen_time) if gen_time > 0 else 0

            all_results[cat_id]["models"][model_name] = {
                "generated_text": decoded_text,
                "gen_time_ms": gen_time * 1000,
                "new_tokens": new_tokens,
                "tokens_per_sec": tok_per_sec,
            }

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Save output to JSON
    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join("outputs", "text_models_multi_category_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Full structured comparison results saved to: {out_file}")

    # Print Formatted Comparison Summary to stdout
    print("\n" + "=" * 90)
    print("  MULTI-CATEGORY INFERENCE COMPARISON RESULTS")
    print("=" * 90)

    for cat_id, data in all_results.items():
        print("\n" + "#" * 90)
        print(f"  {data['title'].upper()}")
        print(f"  PROMPT: \"{data['prompt']}\"")
        print("#" * 90)

        for m_name, res in data["models"].items():
            first_few_lines = res["generated_text"].replace("\r", "").strip()
            print(f"\n  --- [{m_name}] ({res['gen_time_ms']:.1f}ms | {res['new_tokens']} tokens | {res['tokens_per_sec']:.1f} tok/s) ---")
            print(f"  {first_few_lines}")
    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
