"""
AI-DNA Head-to-Head Benchmark Runner.
Compares SVD + CPPN vs. LoRA + Hypernetwork over 20 generations.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

from ai_dna.dna.structure import Genotype
from ai_dna.growth.engine import GrowthEngine
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.encoding.slow_clock import SlowClockEncoder
from ai_dna.inference.pipeline import InferencePipeline
from data import CustomTextTokenizer


def get_synthetic_data(num_samples=40):
    samples = []
    import random
    categories = ["Arithmetic", "Sequence Pattern", "Transitive Logic"]
    for i in range(num_samples):
        cat = random.choice(categories)
        if cat == "Arithmetic":
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            prompt = f"Calculate {a} + {b}."
            expected = str(a + b)
        elif cat == "Sequence Pattern":
            start = random.randint(5, 50)
            step = random.randint(2, 5)
            seq = [start + j * step for j in range(5)]
            prompt = f"What is the next number in sequence {seq}?"
            expected = str(seq[-1] + step)
        else:
            prompt = "If beta contains delta, and delta contains alpha, does beta contain alpha?"
            expected = "Yes"
        samples.append({"prompt": prompt, "solution": expected})
    return samples


def load_stratified_training_data(data_dir="./ai-dna-data", max_samples=800):
    samples_by_task = {}
    for subfolder in ["training", "adaptation"]:
        path = os.path.join(data_dir, subfolder)
        if not os.path.exists(path):
            continue
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jsonl"):
                    file_path = os.path.join(root, file)
                    task_name = file.replace(".jsonl", "")
                    samples_by_task[task_name] = []
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    samples_by_task[task_name].append(json.loads(line))
                    except Exception:
                        pass

    combined_samples = []
    num_tasks = len(samples_by_task)
    if num_tasks > 0:
        samples_per_task = max_samples // num_tasks
        for task, samples in samples_by_task.items():
            subset = samples[:samples_per_task]
            combined_samples.extend(subset)
        import random
        random.seed(42)
        random.shuffle(combined_samples)

    if not combined_samples:
        print("[!] No real datasets found. Falling back to synthetic datasets...")
        combined_samples = get_synthetic_data(max_samples)
    return combined_samples


def evaluate_on_suite(pipeline, tokenizer, test_suite, device):
    correct = 0
    for item in test_suite:
        prompt = item["prompt"]
        expected = item["expected"]
        prompt_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
        res = pipeline.generate(prompt_ids, modality="text", max_new_tokens=10, temperature=0.1)
        out = tokenizer.decode(res["output"].squeeze(0))[len(prompt):].strip()
        if expected.lower() in out.lower():
            correct += 1
    return (correct / len(test_suite)) * 100


def run_benchmark(num_generations=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print(f" Running Comparative Benchmark: SVD + CPPN vs. LoRA + Hypernetwork ({num_generations} Generations)")
    print(f" Hardware Device: {device}")
    print("=" * 110, flush=True)

    tokenizer = CustomTextTokenizer(vocab_size=256, mode="word")
    growth_engine = GrowthEngine(device=device)
    criterion = nn.CrossEntropyLoss()

    # Load training subset
    train_data = load_stratified_training_data(max_samples=100)
    train_subset = train_data[:40]
    print(f"[+] Loaded {len(train_subset)} training samples.", flush=True)

    test_suite = [
        {"prompt": "Calculate 60 + 5.", "expected": "65"},
        {"prompt": "Calculate 90 + 20.", "expected": "110"},
        {"prompt": "What is the next number in sequence [10, 12, 14, 16]?", "expected": "18"},
        {"prompt": "If beta contains delta, and delta contains alpha, does beta contain alpha?", "expected": "Yes"},
    ]

    # Setup genotype templates
    genotype_ref = Genotype.create_default(genotype_id="baseline")
    genotype_ref.dna_architecture.vocab_size = 256
    genotype_ref.dna_architecture.d_model = 64
    genotype_ref.dna_architecture.num_layers = 2
    genotype_ref.dna_architecture.num_experts = 2
    genotype_ref.dna_architecture.d_expert_hidden = 64
    genotype_ref.dna_architecture.coord_dim = 32

    # -----------------------------------------------------------------
    # CONFIGURATION 1: SVD + CPPN
    # -----------------------------------------------------------------
    print("\n>>> Run 1: Evolving SVD + CPPN Baseline...", flush=True)
    slow_clock_cppn = SlowClockEncoder(rank_ratio=0.5, encoder_steps=20, device=device)
    current_genotype_cppn = genotype_ref.clone("cppn_gen0")

    cppn_history = []

    for g in range(1, num_generations + 1):
        t_start = time.time()

        # Grow model
        model = growth_engine.grow_phenotype_model(current_genotype_cppn)

        # Fast Clock (3 epochs of full training)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        epoch_losses = []
        for epoch in range(3):
            total_loss = 0.0
            count = 0
            for item in train_subset:
                prompt = item.get("prompt", item.get("input", ""))
                target = item.get("solution", item.get("target", item.get("output", "")))
                text_pair = prompt + " " + target
                tokens = tokenizer.encode(text_pair).unsqueeze(0).to(device)
                if tokens.shape[1] > 2:
                    optimizer.zero_grad()
                    h, aux_loss, _, _ = model(tokens[:, :-1], modality="text", is_causal=True)
                    logits = model.ar_head(h)
                    targets = tokens[:, 1:]
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)) + aux_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1
            epoch_losses.append(total_loss / max(1, count))

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        # Evaluation
        pipeline = InferencePipeline(phenotype=model, tokenizer=tokenizer, device=device)
        test_acc = evaluate_on_suite(pipeline, tokenizer, test_suite, device)

        # Slow Clock Encoding
        t_slow_start = time.time()
        next_genotype, slow_summary = slow_clock_cppn.step(
            current_genotype_cppn,
            model.state_dict(),
            phenotype_model=model,
            protect_ancestral=False
        )
        slow_time = time.time() - t_slow_start
        current_genotype_cppn = next_genotype

        genotype_size = current_genotype_cppn.total_parameters()
        recon_loss = slow_summary.get("reconstruction_loss", 0.0)

        gen_time = time.time() - t_start
        print(f"Gen {g:2d} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_acc:5.1f}% | Slow Clock Time: {slow_time:5.2f}s | DNA Size: {genotype_size:5d} params | Recon Loss: {recon_loss:.4f}", flush=True)

        cppn_history.append({
            "generation": g,
            "train_loss": avg_train_loss,
            "test_acc": test_acc,
            "slow_clock_time_sec": slow_time,
            "genotype_size_params": genotype_size,
            "reconstruction_loss": recon_loss
        })

    # -----------------------------------------------------------------
    # CONFIGURATION 2: LoRA + Hypernetwork
    # -----------------------------------------------------------------
    print("\n>>> Run 2: Evolving LoRA + Hypernetwork Alternative...", flush=True)
    slow_clock_lora = SlowClockEncoder(rank_ratio=0.5, encoder_steps=20, device=device)

    current_genotype_lora = genotype_ref.clone("lora_gen0")
    current_genotype_lora.dna_architecture.lora_rank = 4
    current_genotype_lora.dna_instinct.genetic_parameters = {"latent_vector": torch.randn(128)}

    lora_history = []

    for g in range(1, num_generations + 1):
        t_start = time.time()

        # Grow model (with LoRA adapters injected automatically)
        model = growth_engine.grow_phenotype_model(current_genotype_lora)

        # Freeze base layers, optimize ONLY LoRA parameters
        from ai_dna.models.lora import freeze_model_except_lora
        freeze_model_except_lora(model)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1.5e-3, weight_decay=1e-4)

        epoch_losses = []
        for epoch in range(3):
            total_loss = 0.0
            count = 0
            for item in train_subset:
                prompt = item.get("prompt", item.get("input", ""))
                target = item.get("solution", item.get("target", item.get("output", "")))
                text_pair = prompt + " " + target
                tokens = tokenizer.encode(text_pair).unsqueeze(0).to(device)
                if tokens.shape[1] > 2:
                    optimizer.zero_grad()
                    h, aux_loss, _, _ = model(tokens[:, :-1], modality="text", is_causal=True)
                    logits = model.ar_head(h)
                    targets = tokens[:, 1:]
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)) + aux_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1
            epoch_losses.append(total_loss / max(1, count))

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        # Evaluation
        pipeline = InferencePipeline(phenotype=model, tokenizer=tokenizer, device=device)
        test_acc = evaluate_on_suite(pipeline, tokenizer, test_suite, device)

        # Slow Clock Encoding (routes to InverseHypernetworkEncoder automatically)
        t_slow_start = time.time()
        next_genotype, slow_summary = slow_clock_lora.step(
            current_genotype_lora,
            model.state_dict(),
            phenotype_model=model,
            growth_engine=growth_engine,
            protect_ancestral=False
        )
        slow_time = time.time() - t_slow_start
        current_genotype_lora = next_genotype

        genotype_size = current_genotype_lora.total_parameters()
        recon_loss = slow_summary.get("reconstruction_loss", 0.0)

        gen_time = time.time() - t_start
        print(f"Gen {g:2d} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_acc:5.1f}% | Slow Clock Time: {slow_time:5.2f}s | DNA Size: {genotype_size:5d} params | Recon Loss: {recon_loss:.4f}", flush=True)

        lora_history.append({
            "generation": g,
            "train_loss": avg_train_loss,
            "test_acc": test_acc,
            "slow_clock_time_sec": slow_time,
            "genotype_size_params": genotype_size,
            "reconstruction_loss": recon_loss
        })

    # =========================================================================
    # PART C: Generate final comparative report
    # =========================================================================
    print("\n" + "=" * 120)
    print(" COMPARISON SUMMARY TABLE")
    print("=" * 120, flush=True)

    print(f"{'Metric':<25} | {'SVD + CPPN (Baseline)':<30} | {'LoRA + Hypernetwork':<30}")
    print("-" * 95, flush=True)

    avg_cppn_slow_time = sum(h["slow_clock_time_sec"] for h in cppn_history) / num_generations
    avg_lora_slow_time = sum(h["slow_clock_time_sec"] for h in lora_history) / num_generations
    print(f"{'Avg Slow Clock Time':<25} | {avg_cppn_slow_time:26.2f}s | {avg_lora_slow_time:26.2f}s")

    final_cppn_size = cppn_history[-1]["genotype_size_params"]
    final_lora_size = lora_history[-1]["genotype_size_params"]
    print(f"{'Final Genotype Size':<25} | {final_cppn_size:24d} params | {final_lora_size:24d} params")

    avg_cppn_recon_loss = sum(h["reconstruction_loss"] for h in cppn_history) / num_generations
    avg_lora_recon_loss = sum(h["reconstruction_loss"] for h in lora_history) / num_generations
    print(f"{'Avg Reconstruction Loss':<25} | {avg_cppn_recon_loss:28.4f} | {avg_lora_recon_loss:28.4f}")

    final_cppn_acc = cppn_history[-1]["test_acc"]
    final_lora_acc = lora_history[-1]["test_acc"]
    print(f"{'Final Test Suite Acc':<25} | {final_cppn_acc:27.1f}% | {final_lora_acc:27.1f}%")

    report = {
        "num_generations": num_generations,
        "cppn_history": cppn_history,
        "lora_history": lora_history,
        "summary": {
            "cppn": {
                "avg_slow_clock_time_sec": avg_cppn_slow_time,
                "final_genotype_size_params": final_cppn_size,
                "avg_reconstruction_loss": avg_cppn_recon_loss,
                "final_test_acc": final_cppn_acc
            },
            "lora": {
                "avg_slow_clock_time_sec": avg_lora_slow_time,
                "final_genotype_size_params": final_lora_size,
                "avg_reconstruction_loss": avg_lora_recon_loss,
                "final_test_acc": final_lora_acc
            }
        }
    }

    with open("comparison_results_20_generations.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n[+] Detailed comparative report saved to: comparison_results_20_generations.json")


if __name__ == "__main__":
    run_benchmark(num_generations=20)
