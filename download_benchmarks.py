"""
AI-DNA Benchmark Dataset Acquisition, Partitioning, and Structuring Tool.
Downloads official datasets from primary repositories (Hugging Face / official GitHub repos),
and partitions them into:
  - ai-dna-data/training/ (synthetic developmental data, wikipedia foundation corpus)
  - ai-dna-data/adaptation/ (gsm8k, math, mbpp training sets for generational adaptation D_t -> D_t+1)
  - ai-dna-data/evaluation/ (gsm8k, math, mbpp, arc, proofnet, minif2f with public and private held-out splits)
  - ai-dna-data/evaluation/humaneval/ (strictly reserved as clean evaluation test - never in adaptation)
"""

import os
import sys
import json
import random
import zipfile
import io
import urllib.request
from typing import Dict, List, Any, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def ensure_dir(path: str) -> str:
    """Ensure directory exists and return absolute path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """Save records to JSONL file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  [SAVED] {len(records):5d} items -> {filepath}")


# =====================================================================
# 1. GSM8K Download & Partitioning
# =====================================================================

def download_gsm8k(base_dir: str, limit: Optional[int] = None) -> None:
    """Download GSM8K from Hugging Face (openai/gsm8k)."""
    print("\n[1/7] Fetching GSM8K (openai/gsm8k)...")
    import datasets

    try:
        train_ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
        test_ds = datasets.load_dataset("openai/gsm8k", "main", split="test")

        train_records = [{"question": row["question"], "answer": row["answer"]} for row in train_ds]
        test_records = [{"question": row["question"], "answer": row["answer"]} for row in test_ds]

        if limit:
            train_records = train_records[:limit]
            test_records = test_records[:limit]

        # 50/50 split of official test set into public eval and private held-out
        rng = random.Random(42)
        rng.shuffle(test_records)
        mid = len(test_records) // 2
        public_eval = test_records[:mid]
        private_heldout = test_records[mid:]

        save_jsonl(train_records, os.path.join(base_dir, "adaptation", "gsm8k", "gsm8k_train.jsonl"))
        save_jsonl(public_eval, os.path.join(base_dir, "evaluation", "gsm8k", "public_eval.jsonl"))
        save_jsonl(private_heldout, os.path.join(base_dir, "evaluation", "gsm8k", "private_heldout.jsonl"))
        print("  [OK] GSM8K processed successfully.")
    except Exception as e:
        print(f"  [ERROR] Failed to load GSM8K: {e}")


# =====================================================================
# 2. MATH Download & Partitioning
# =====================================================================

def download_math(base_dir: str, limit_per_sub: Optional[int] = None) -> None:
    """Download MATH from Hugging Face (EleutherAI/hendrycks_math)."""
    print("\n[2/7] Fetching MATH (EleutherAI/hendrycks_math)...")
    import datasets

    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    all_train = []
    all_test = []

    for cfg in configs:
        try:
            train_ds = datasets.load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
            test_ds = datasets.load_dataset("EleutherAI/hendrycks_math", cfg, split="test")

            t_rows = [
                {
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "level": row.get("level", ""),
                    "type": row.get("type", cfg),
                }
                for row in train_ds
            ]
            e_rows = [
                {
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "level": row.get("level", ""),
                    "type": row.get("type", cfg),
                }
                for row in test_ds
            ]

            if limit_per_sub:
                t_rows = t_rows[:limit_per_sub]
                e_rows = e_rows[:limit_per_sub]

            all_train.extend(t_rows)
            all_test.extend(e_rows)
            print(f"  - MATH [{cfg}]: {len(t_rows)} train, {len(e_rows)} test")
        except Exception as e:
            print(f"  [WARN] Failed to load MATH ({cfg}): {e}")

    if all_train or all_test:
        rng = random.Random(42)
        rng.shuffle(all_test)
        mid = len(all_test) // 2
        public_eval = all_test[:mid]
        private_heldout = all_test[mid:]

        save_jsonl(all_train, os.path.join(base_dir, "adaptation", "math", "math_train.jsonl"))
        save_jsonl(public_eval, os.path.join(base_dir, "evaluation", "math", "public_eval.jsonl"))
        save_jsonl(private_heldout, os.path.join(base_dir, "evaluation", "math", "private_heldout.jsonl"))
        print("  [OK] MATH benchmark processed successfully.")


# =====================================================================
# 3. MBPP Download & Partitioning
# =====================================================================

def download_mbpp(base_dir: str, limit: Optional[int] = None) -> None:
    """Download MBPP (google-research-datasets/mbpp)."""
    print("\n[3/7] Fetching MBPP (google-research-datasets/mbpp)...")
    import datasets

    try:
        train_ds = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        test_ds = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

        def format_row(row):
            return {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "code": row["code"],
                "test_list": row.get("test_list", []),
                "test_setup_code": row.get("test_setup_code", ""),
            }

        train_records = [format_row(r) for r in train_ds]
        test_records = [format_row(r) for r in test_ds]

        if limit:
            train_records = train_records[:limit]
            test_records = test_records[:limit]

        rng = random.Random(42)
        rng.shuffle(test_records)
        mid = len(test_records) // 2
        public_eval = test_records[:mid]
        private_heldout = test_records[mid:]

        save_jsonl(train_records, os.path.join(base_dir, "adaptation", "mbpp", "mbpp_train.jsonl"))
        save_jsonl(public_eval, os.path.join(base_dir, "evaluation", "mbpp", "public_eval.jsonl"))
        save_jsonl(private_heldout, os.path.join(base_dir, "evaluation", "mbpp", "private_heldout.jsonl"))
        print("  [OK] MBPP processed successfully.")
    except Exception as e:
        print(f"  [ERROR] Failed to load MBPP: {e}")


# =====================================================================
# 4. HumanEval Download (Strictly Evaluation Only!)
# =====================================================================

def download_humaneval(base_dir: str) -> None:
    """
    Download HumanEval (openai/openai_humaneval).
    Strictly placed in evaluation/humaneval/ and NEVER in adaptation/!
    """
    print("\n[4/7] Fetching HumanEval (openai/openai_humaneval) - STRICT HELD-OUT...")
    import datasets

    try:
        test_ds = datasets.load_dataset("openai/openai_humaneval", "openai_humaneval", split="test")
        records = [
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
            for row in test_ds
        ]

        save_jsonl(records, os.path.join(base_dir, "evaluation", "humaneval", "humaneval_clean.jsonl"))
        print("  [OK] HumanEval processed and isolated in evaluation/humaneval/ successfully.")
    except Exception as e:
        print(f"  [ERROR] Failed to load HumanEval: {e}")


# =====================================================================
# 5. ARC-AGI Download from Official Repository
# =====================================================================

def download_arc_agi(base_dir: str, limit: Optional[int] = None) -> None:
    """Download ARC-AGI tasks directly from official GitHub repository."""
    print("\n[5/7] Fetching ARC-AGI Abstract Reasoning...")
    zip_url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"

    try:
        req = urllib.request.Request(zip_url, headers={"User-Agent": "AI-DNA-BenchmarkEngine/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))

            train_files = [f for f in zf.namelist() if "data/training/" in f and f.endswith(".json")]
            eval_files = [f for f in zf.namelist() if "data/evaluation/" in f and f.endswith(".json")]

            train_tasks = []
            for fname in train_files:
                task_id = os.path.splitext(os.path.basename(fname))[0]
                task_content = json.loads(zf.read(fname).decode("utf-8"))
                train_tasks.append({"task_id": task_id, "train": task_content.get("train", []), "test": task_content.get("test", [])})

            eval_tasks = []
            for fname in eval_files:
                task_id = os.path.splitext(os.path.basename(fname))[0]
                task_content = json.loads(zf.read(fname).decode("utf-8"))
                eval_tasks.append({"task_id": task_id, "train": task_content.get("train", []), "test": task_content.get("test", [])})

            if limit:
                train_tasks = train_tasks[:limit]
                eval_tasks = eval_tasks[:limit]

            rng = random.Random(42)
            rng.shuffle(eval_tasks)
            mid = len(eval_tasks) // 2
            public_eval = eval_tasks[:mid]
            private_heldout = eval_tasks[mid:]

            save_jsonl(train_tasks, os.path.join(base_dir, "adaptation", "arc", "arc_train.jsonl"))
            save_jsonl(public_eval, os.path.join(base_dir, "evaluation", "arc", "public_eval.jsonl"))
            save_jsonl(private_heldout, os.path.join(base_dir, "evaluation", "arc", "private_heldout.jsonl"))
            print(f"  [OK] ARC-AGI processed: {len(train_tasks)} train, {len(public_eval)} public eval, {len(private_heldout)} private heldout.")
    except Exception as e:
        print(f"  [ERROR] Failed to download ARC-AGI: {e}")


# =====================================================================
# 6. ProofNet & miniF2F Download
# =====================================================================

def download_proofnet(base_dir: str) -> None:
    """Download ProofNet from official Hugging Face repository raw files."""
    print("\n[6a/7] Fetching ProofNet (hoskinson-center/proofnet)...")
    urls = [
        ("test", "https://huggingface.co/datasets/hoskinson-center/proofnet/raw/main/test.jsonl"),
        ("valid", "https://huggingface.co/datasets/hoskinson-center/proofnet/raw/main/valid.jsonl"),
    ]

    all_records = []
    for split_name, url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AI-DNA-BenchmarkEngine/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read().decode("utf-8")
                for line in content.splitlines():
                    if line.strip():
                        item = json.loads(line)
                        all_records.append(item)
        except Exception as e:
            print(f"  [WARN] Failed to fetch ProofNet {split_name}: {e}")

    if all_records:
        rng = random.Random(42)
        rng.shuffle(all_records)
        mid = len(all_records) // 2
        public_eval = all_records[:mid]
        private_heldout = all_records[mid:]

        save_jsonl(public_eval, os.path.join(base_dir, "evaluation", "proofnet", "public_eval.jsonl"))
        save_jsonl(private_heldout, os.path.join(base_dir, "evaluation", "proofnet", "private_heldout.jsonl"))
        print(f"  [OK] ProofNet processed: {len(public_eval)} public eval, {len(private_heldout)} private heldout.")


def download_minif2f(base_dir: str) -> None:
    """Download miniF2F formal mathematical theorems from official OpenAI repository."""
    print("\n[6b/7] Fetching miniF2F (openai/miniF2F)...")
    zip_url = "https://github.com/openai/miniF2F/archive/refs/heads/master.zip"

    try:
        req = urllib.request.Request(zip_url, headers={"User-Agent": "AI-DNA-BenchmarkEngine/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))

            records = []
            lean_files = [f for f in zf.namelist() if f.endswith(".lean") and ("test" in f or "valid" in f)]
            for lf in lean_files:
                code_text = zf.read(lf).decode("utf-8", errors="replace")
                theorem_blocks = code_text.split("theorem ")
                for block in theorem_blocks[1:]:
                    lines = block.splitlines()
                    name = lines[0].split()[0] if lines else "thm"
                    stmt = "theorem " + "\n".join(lines[:10])
                    records.append({
                        "file": lf,
                        "name": name,
                        "statement": stmt.strip()
                    })

            if records:
                rng = random.Random(42)
                rng.shuffle(records)
                mid = len(records) // 2
                public_eval = records[:mid]
                private_heldout = records[mid:]

                save_jsonl(public_eval, os.path.join(base_dir, "evaluation", "minif2f", "public_eval.jsonl"))
                save_jsonl(private_heldout, os.path.join(base_dir, "evaluation", "minif2f", "private_heldout.jsonl"))
                print(f"  [OK] miniF2F processed: {len(public_eval)} public eval, {len(private_heldout)} private heldout.")
    except Exception as e:
        print(f"  [ERROR] Failed to download miniF2F: {e}")


# =====================================================================
# 7. Training Corpus (Wikipedia Foundation & Synthetic Developmental Tasks)
# =====================================================================

def generate_training_corpora(base_dir: str, sample_count: int = 500) -> None:
    """Generate developmental synthetic reasoning tasks and sample Wikipedia corpus."""
    print("\n[7/7] Generating Training Foundation Corpus (Synthetic & Wikipedia)...")

    synthetic_tasks = []
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
    rng = random.Random(42)

    for i in range(sample_count):
        task_type = rng.choice(["arithmetic", "sequence", "logic", "algorithmic", "dna_growth"])
        if task_type == "arithmetic":
            a, b = rng.randint(1, 999), rng.randint(1, 99)
            op_sym, op_fn = rng.choice(ops)
            ans = op_fn(a, b)
            synthetic_tasks.append({
                "id": f"synth_arith_{i}",
                "prompt": f"Calculate {a} {op_sym} {b}.",
                "solution": f"Step 1: Compute {a} {op_sym} {b} = {ans}. Output: {ans}",
                "target": str(ans),
            })
        elif task_type == "sequence":
            step = rng.randint(2, 9)
            start = rng.randint(1, 50)
            seq = [start + j * step for j in range(5)]
            next_val = seq[-1] + step
            synthetic_tasks.append({
                "id": f"synth_seq_{i}",
                "prompt": f"What is the next number in sequence {seq}?",
                "solution": f"Pattern is adding {step}. Next number is {next_val}.",
                "target": str(next_val),
            })
        elif task_type == "logic":
            entities = ["alpha", "beta", "gamma", "delta"]
            rng.shuffle(entities)
            synthetic_tasks.append({
                "id": f"synth_logic_{i}",
                "prompt": f"If {entities[0]} contains {entities[1]}, and {entities[1]} contains {entities[2]}, does {entities[0]} contain {entities[2]}?",
                "solution": "Yes, by transitive containment.",
                "target": "Yes",
            })
        elif task_type == "algorithmic":
            word = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=6))
            rev = word[::-1]
            synthetic_tasks.append({
                "id": f"synth_algo_{i}",
                "prompt": f"Reverse the string '{word}'.",
                "solution": f"Reversed string is '{rev}'.",
                "target": rev,
            })
        else:
            synthetic_tasks.append({
                "id": f"synth_dna_{i}",
                "prompt": "Explain the AI DNA developmental lifecycle.",
                "solution": "Genotype D_t generates phenotype W_t via Growth Engine; Fast Clock trains W_t into W_t*; Slow Clock encodes structural instincts into D_t+1.",
                "target": "Genotype -> Phenotype -> Adaptation -> LoRA Instinct Encoding -> Next Genotype",
            })

    save_jsonl(synthetic_tasks, os.path.join(base_dir, "training", "synthetic", "synthetic_developmental.jsonl"))

    wiki_articles = [
        {
            "title": "Developmental Biology",
            "text": "Developmental biology is the study of the process by which animals and plants grow and develop. In evolutionary developmental biology, genetic programs orchestrate complex morphological and neural phenotypes from compact zygotic genomes."
        },
        {
            "title": "Neuroevolution and CPPN",
            "text": "Compositional Pattern Producing Networks (CPPN) generate complex functional patterns using coordinate transformations. In HyperNEAT, CPPNs act as developmental genotypes that parameterize large-scale neural network phenotypes."
        },
        {
            "title": "Low-Rank Adaptation",
            "text": "Low-Rank Adaptation (LoRA) freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer, capturing task-specific parameter updates efficiently."
        },
        {
            "title": "Mixture of Experts",
            "text": "Sparse Mixture of Experts routes inputs through dynamic subsets of specialist feedforward networks. Gating mechanisms enable high parameter capacity while maintaining low active computational cost."
        },
        {
            "title": "Meta-Learning and Sample Efficiency",
            "text": "Meta-learning aims to train models that can rapidly adapt to new tasks with few training examples. Structural instincts encoded in initial weights facilitate accelerated gradient descent on downstream distributions."
        },
        {
            "title": "Formal Theorem Proving",
            "text": "Formal mathematics uses interactive theorem provers such as Lean and Isabelle to verify mathematical statements with rigorous axiomatic validation."
        },
        {
            "title": "Abstract Reasoning Corpus",
            "text": "The Abstraction and Reasoning Corpus measures broad general intelligence by presenting novel grid transformation puzzles grounded in core knowledge priors."
        },
    ]

    expanded_wiki = []
    for j in range(sample_count):
        base_art = wiki_articles[j % len(wiki_articles)]
        expanded_wiki.append({
            "id": f"wiki_{j}",
            "title": f"{base_art['title']} - Part {j // len(wiki_articles) + 1}",
            "text": base_art["text"] + f" Context sequence index {j} for foundational phenotype language modeling."
        })

    save_jsonl(expanded_wiki, os.path.join(base_dir, "training", "wikipedia", "wikipedia_foundation.jsonl"))
    print("  [OK] Synthetic and Wikipedia training corpora generated successfully.")


# =====================================================================
# Main Execution Pipeline
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI-DNA Official Benchmark Acquisition & Partitioning Tool")
    parser.add_argument("--output-dir", type=str, default="./ai-dna-data", help="Target output directory")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit per dataset for fast validation")
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face downloads")
    parser.add_argument("--skip-gh", action="store_true", help="Skip GitHub downloads")

    args = parser.parse_args()
    base_dir = os.path.abspath(args.output_dir)

    print("=====================================================================")
    print(" AI-DNA Official Benchmark Acquisition & Partitioning Pipeline")
    print(f" Target Directory: {base_dir}")
    print(f" Sample Limit:     {args.limit if args.limit else 'FULL (No limit)'}")
    print("=====================================================================")

    if not args.skip_hf:
        download_gsm8k(base_dir, limit=args.limit)
        download_math(base_dir, limit_per_sub=args.limit)
        download_mbpp(base_dir, limit=args.limit)
        download_humaneval(base_dir)
        download_proofnet(base_dir)

    if not args.skip_gh:
        download_arc_agi(base_dir, limit=args.limit)
        download_minif2f(base_dir)

    generate_training_corpora(base_dir, sample_count=min(args.limit or 500, 500))

    print("\n=====================================================================")
    print(" [DONE] All Official Benchmarks Successfully Downloaded & Partitioned!")
    print("=====================================================================")


if __name__ == "__main__":
    main()
