"""
Head-to-Head Benchmark Comparison: Normal Continual Training vs AI-DNA Evolutionary Architecture.
Trains two parallel models over 10-12 generations:
  1. Standard Baseline Model (Conventional parameter updates with continuous SGD/AdamW)
  2. AI-DNA Evolved Model (Genotype D_t -> Phenotype W_t -> Fast Clock -> SVD Slow Clock -> D_{t+1})

Evaluates both models at every generation across:
  - Public Benchmark Suite (GSM8K, MATH, MBPP, ARC-AGI, ProofNet, miniF2F)
  - Private Held-Out Benchmark Suite (GSM8K, MATH, MBPP, ARC-AGI, ProofNet, miniF2F)
  - Strict Clean Evaluation Benchmark: HumanEval (Zero exposure)
"""

import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ai_dna.dna.structure import Genotype
from ai_dna.growth.engine import GrowthEngine
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.encoding.slow_clock import SlowClockEncoder
from ai_dna.evolution.mutation import GenotypeMutator
from data import AIDNABenchmarkDataset, CustomTextTokenizer


def train_step_batch(
    model: PhenotypeNeuralNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    steps: int = 15,
) -> float:
    """Executes gradient descent steps on a single batch."""
    model.train()
    total_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        h, aux_loss, _, _ = model(inputs, modality="text", is_causal=True)
        logits = model.ar_head(h)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)) + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(steps, 1)


def evaluate_batch(
    model: PhenotypeNeuralNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Evaluates phenotype on evaluation batch; returns (loss, accuracy %)."""
    model.eval()
    with torch.no_grad():
        h, _, _, _ = model(inputs, modality="text", is_causal=True)
        logits = model.ar_head(h)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item() * 100.0
    return round(loss, 4), round(acc, 2)


def run_comparison_experiment(
    num_generations: int = 10,
    data_dir: str = "./ai-dna-data",
    seq_len: int = 64,
    device_str: Optional[str] = None,
    output_report: str = "comparison_normal_vs_aidna_results.json",
) -> Dict[str, Any]:
    """
    Executes parallel training of Standard Continual Model vs AI-DNA Evolved Model over N generations.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 85)
    print(" ⚔️  AI-DNA vs NORMAL CONTINUAL TRAINING: MULTI-GENERATIONAL BENCHMARK HARNESS")
    print(f" Device:          {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Generations:     {num_generations} (D_0 -> D_{num_generations-1})")
    print(f" Data Directory:  {data_dir}")
    print("=" * 85)

    tokenizer = CustomTextTokenizer(vocab_size=256, mode="word")
    growth_engine = GrowthEngine(device=device)
    slow_clock = SlowClockEncoder(rank_ratio=0.35, encoder_steps=35, device=device)
    mutator = GenotypeMutator()
    criterion = nn.CrossEntropyLoss()

    # 1. Setup Evaluation Batches
    eval_tasks = ["gsm8k", "math", "mbpp", "arc", "proofnet", "minif2f"]
    public_eval_data = {}
    heldout_eval_data = {}
    sample_cap = 30

    print("\n[+] Preloading Benchmark Evaluation Sets...")
    for t in eval_tasks:
        try:
            ds_pub = AIDNABenchmarkDataset(task=t, split="public_eval", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer, max_samples=sample_cap)
            if len(ds_pub) > 0:
                public_eval_data[t] = (
                    torch.stack([item[0] for item in ds_pub]).to(device),
                    torch.stack([item[1] for item in ds_pub]).to(device)
                )
        except Exception as e:
            print(f"  [WARN] Loading public eval {t}: {e}")

        try:
            ds_held = AIDNABenchmarkDataset(task=t, split="private_heldout", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer, max_samples=sample_cap)
            if len(ds_held) > 0:
                heldout_eval_data[t] = (
                    torch.stack([item[0] for item in ds_held]).to(device),
                    torch.stack([item[1] for item in ds_held]).to(device)
                )
        except Exception as e:
            print(f"  [WARN] Loading private heldout {t}: {e}")

    # HumanEval Clean Benchmark (Strictly Isolated)
    ds_he = AIDNABenchmarkDataset(task="humaneval", split="clean_eval", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer, max_samples=sample_cap)
    humaneval_data = (
        torch.stack([item[0] for item in ds_he]).to(device),
        torch.stack([item[1] for item in ds_he]).to(device)
    )
    print(f"  ✓ Clean HumanEval Test: {len(ds_he)} problems loaded (Never in training/adaptation)")

    # 2. Setup Adaptation Batches (Task streams that change/rotate per generation)
    adaptation_data = {}
    for t in ["gsm8k", "math", "mbpp", "arc"]:
        ds_a = AIDNABenchmarkDataset(task=t, split="adaptation", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer, max_samples=sample_cap)
        adaptation_data[t] = (
            torch.stack([item[0] for item in ds_a]).to(device),
            torch.stack([item[1] for item in ds_a]).to(device)
        )
    print(f"  ✓ Adaptation Streams: GSM8K, MATH, MBPP, ARC loaded successfully.\n")

    # 3. Setup Models
    # Model 1: Standard Continual Training Baseline (Persistent parameter updates)
    root_genotype = Genotype.create_default(genotype_id="root_baseline")
    root_genotype.dna_architecture.vocab_size = 256
    root_genotype.dna_architecture.d_model = 64
    root_genotype.dna_architecture.num_layers = 3
    root_genotype.dna_architecture.num_experts = 4
    root_genotype.dna_architecture.d_expert_hidden = 128

    standard_model = PhenotypeNeuralNetwork(root_genotype).to(device)
    standard_optimizer = optim.AdamW(standard_model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Model 2: AI-DNA Architecture (Genotype-Phenotype Developmental Lifecycle)
    aidna_genotype = copy.deepcopy(root_genotype)
    aidna_genotype.genotype_id = "gen_0"

    results = {
        "metadata": {
            "num_generations": num_generations,
            "d_model": root_genotype.dna_architecture.d_model,
            "num_layers": root_genotype.dna_architecture.num_layers,
            "num_experts": root_genotype.dna_architecture.num_experts,
            "device": str(device),
        },
        "generations": [],
    }

    print("=" * 85)
    print(" 🚀 STARTING HEAD-TO-HEAD GENERATIONAL EXPERIMENT")
    print("=" * 85)

    start_time = time.time()

    for g in range(num_generations):
        gen_label = f"Gen_{g}"
        print(f"\n" + "-" * 85)
        print(f"  ▶ GENERATION {g+1}/{num_generations} [D_{g}]")
        print("-" * 85)

        # -------------------------------------------------------------
        # Branch A: AI-DNA Phenotype Growth
        # -------------------------------------------------------------
        aidna_phenotype = PhenotypeNeuralNetwork(aidna_genotype).to(device)
        grown_w = growth_engine.grow_phenotype_weights(aidna_genotype)
        p_state = aidna_phenotype.state_dict()
        for k, v in grown_w.items():
            if k in p_state and p_state[k].shape == v.shape:
                p_state[k] = v
        aidna_phenotype.load_state_dict(p_state)
        aidna_optimizer = optim.AdamW(aidna_phenotype.parameters(), lr=5e-4, weight_decay=1e-4)

        # -------------------------------------------------------------
        # Evaluate Pre-Adaptation Benchmarks for Both Models
        # -------------------------------------------------------------
        # Standard Model Eval
        std_pub_accs, std_pub_losses = {}, {}
        std_held_accs, std_held_losses = {}, {}
        for t, (inps, tgts) in public_eval_data.items():
            l, acc = evaluate_batch(standard_model, inps, tgts, criterion)
            std_pub_losses[t], std_pub_accs[t] = l, acc
        for t, (inps, tgts) in heldout_eval_data.items():
            l, acc = evaluate_batch(standard_model, inps, tgts, criterion)
            std_held_losses[t], std_held_accs[t] = l, acc
        std_he_loss, std_he_acc = evaluate_batch(standard_model, humaneval_data[0], humaneval_data[1], criterion)

        # AI-DNA Model Eval
        dna_pub_accs, dna_pub_losses = {}, {}
        dna_held_accs, dna_held_losses = {}, {}
        for t, (inps, tgts) in public_eval_data.items():
            l, acc = evaluate_batch(aidna_phenotype, inps, tgts, criterion)
            dna_pub_losses[t], dna_pub_accs[t] = l, acc
        for t, (inps, tgts) in heldout_eval_data.items():
            l, acc = evaluate_batch(aidna_phenotype, inps, tgts, criterion)
            dna_held_losses[t], dna_held_accs[t] = l, acc
        dna_he_loss, dna_he_acc = evaluate_batch(aidna_phenotype, humaneval_data[0], humaneval_data[1], criterion)

        # Compute Averages
        std_avg_pub = sum(std_pub_accs.values()) / len(std_pub_accs)
        std_avg_held = sum(std_held_accs.values()) / len(std_held_accs)
        dna_avg_pub = sum(dna_pub_accs.values()) / len(dna_pub_accs)
        dna_avg_held = sum(dna_held_accs.values()) / len(dna_held_accs)

        print(f"  [Pre-Adapt Baseline Benchmark Scores]:")
        print(f"    • Standard Model: Public Avg = {std_avg_pub:5.2f}% | Held-Out Avg = {std_avg_held:5.2f}% | HumanEval Loss = {std_he_loss:.4f} ({std_he_acc:.2f}%)")
        print(f"    • AI-DNA Model  : Public Avg = {dna_avg_pub:5.2f}% | Held-Out Avg = {dna_avg_held:5.2f}% | HumanEval Loss = {dna_he_loss:.4f} ({dna_he_acc:.2f}%)")

        # -------------------------------------------------------------
        # Training Phase on Generational Adaptation Stream
        # -------------------------------------------------------------
        # Rotating focus task per generation to test continual adaptation & forgetting
        focus_task = ["gsm8k", "math", "mbpp", "arc"][g % 4]
        print(f"\n  [Training Stream]: Adapting on Primary Focus Task [{focus_task.upper()}] + Joint Streams...")

        # Standard Model Training (Continuous parameter accumulation)
        t0 = time.time()
        std_loss_f = train_step_batch(standard_model, adaptation_data[focus_task][0], adaptation_data[focus_task][1], standard_optimizer, criterion, steps=20)
        std_train_time = round(time.time() - t0, 3)

        # AI-DNA Fast Clock Training
        t0 = time.time()
        dna_loss_f = train_step_batch(aidna_phenotype, adaptation_data[focus_task][0], adaptation_data[focus_task][1], aidna_optimizer, criterion, steps=20)
        dna_train_time = round(time.time() - t0, 3)

        print(f"    • Standard Model Adapt Loss: {std_loss_f:.4f} ({std_train_time}s)")
        print(f"    • AI-DNA Model Adapt Loss  : {dna_loss_f:.4f} ({dna_train_time}s)")

        # -------------------------------------------------------------
        # Post-Adaptation Benchmark Evaluation
        # -------------------------------------------------------------
        std_post_pub_accs, std_post_pub_losses = {}, {}
        std_post_held_accs, std_post_held_losses = {}, {}
        for t, (inps, tgts) in public_eval_data.items():
            l, acc = evaluate_batch(standard_model, inps, tgts, criterion)
            std_post_losses = l
            std_post_pub_accs[t] = acc
        for t, (inps, tgts) in heldout_eval_data.items():
            l, acc = evaluate_batch(standard_model, inps, tgts, criterion)
            std_post_held_accs[t] = acc
        std_post_he_loss, std_post_he_acc = evaluate_batch(standard_model, humaneval_data[0], humaneval_data[1], criterion)

        dna_post_pub_accs, dna_post_pub_losses = {}, {}
        dna_post_held_accs, dna_post_held_losses = {}, {}
        for t, (inps, tgts) in public_eval_data.items():
            l, acc = evaluate_batch(aidna_phenotype, inps, tgts, criterion)
            dna_post_pub_accs[t] = acc
        for t, (inps, tgts) in heldout_eval_data.items():
            l, acc = evaluate_batch(aidna_phenotype, inps, tgts, criterion)
            dna_post_held_accs[t] = acc
        dna_post_he_loss, dna_post_he_acc = evaluate_batch(aidna_phenotype, humaneval_data[0], humaneval_data[1], criterion)

        std_post_avg_pub = sum(std_post_pub_accs.values()) / len(std_post_pub_accs)
        std_post_avg_held = sum(std_post_held_accs.values()) / len(std_post_held_accs)
        dna_post_avg_pub = sum(dna_post_pub_accs.values()) / len(dna_post_pub_accs)
        dna_post_avg_held = sum(dna_post_held_accs.values()) / len(dna_post_held_accs)

        print(f"\n  [Post-Adaptation Benchmark Results]:")
        print(f"    • Standard Model: Public Avg = {std_post_avg_pub:5.2f}% | Held-Out Avg = {std_post_avg_held:5.2f}% | Clean HumanEval = {std_post_he_loss:.4f} ({std_post_he_acc:.2f}%)")
        print(f"    • AI-DNA Model  : Public Avg = {dna_post_avg_pub:5.2f}% | Held-Out Avg = {dna_post_avg_held:5.2f}% | Clean HumanEval = {dna_post_he_loss:.4f} ({dna_post_he_acc:.2f}%)")

        gen_record = {
            "generation": g,
            "focus_task": focus_task,
            "standard_model": {
                "pre_adapt_public_avg": round(std_avg_pub, 2),
                "pre_adapt_heldout_avg": round(std_avg_held, 2),
                "pre_adapt_humaneval_loss": std_he_loss,
                "post_adapt_public_avg": round(std_post_avg_pub, 2),
                "post_adapt_heldout_avg": round(std_post_avg_held, 2),
                "post_adapt_humaneval_loss": std_post_he_loss,
                "adaptation_loss": round(std_loss_f, 4),
                "public_acc_per_task": std_post_pub_accs,
                "heldout_acc_per_task": std_post_held_accs,
            },
            "aidna_model": {
                "genotype_id": aidna_genotype.genotype_id,
                "pre_adapt_public_avg": round(dna_avg_pub, 2),
                "pre_adapt_heldout_avg": round(dna_avg_held, 2),
                "pre_adapt_humaneval_loss": dna_he_loss,
                "post_adapt_public_avg": round(dna_post_avg_pub, 2),
                "post_adapt_heldout_avg": round(dna_post_avg_held, 2),
                "post_adapt_humaneval_loss": dna_post_he_loss,
                "adaptation_loss": round(dna_loss_f, 4),
                "public_acc_per_task": dna_post_pub_accs,
                "heldout_acc_per_task": dna_post_held_accs,
            }
        }
        results["generations"].append(gen_record)

        # -------------------------------------------------------------
        # Branch B: AI-DNA Slow Clock (Distill W_t* -> D_{t+1})
        # -------------------------------------------------------------
        if g < num_generations - 1:
            learned_state = {k: v.clone() for k, v in aidna_phenotype.state_dict().items()}
            next_genotype, slow_info = slow_clock.step(aidna_genotype, learned_state)
            next_genotype = mutator.mutate(next_genotype)
            next_genotype.genotype_id = f"gen_{g+1}"
            aidna_genotype = next_genotype

    total_time = round(time.time() - start_time, 2)
    results["total_elapsed_sec"] = total_time
    print("\n" + "=" * 85)
    print(f" [DONE] Completed {num_generations} Generations in {total_time}s!")
    print("=" * 85)

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved complete comparison results to: {output_report}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Head-to-Head Comparison: Standard Training vs AI-DNA Evolution")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations (default: 10)")
    parser.add_argument("--data-dir", type=str, default="./ai-dna-data", help="Data directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--report", type=str, default="comparison_normal_vs_aidna_results.json", help="Output report file")

    args = parser.parse_args()
    run_comparison_experiment(
        num_generations=args.generations,
        data_dir=args.data_dir,
        device_str=args.device,
        output_report=args.report,
    )
