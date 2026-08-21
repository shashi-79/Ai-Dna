"""
Experiment 7: Official Benchmark Suite Evaluation across AI-DNA Generations.
Evaluates generational evolution D_0 -> D_1 -> ... -> D_n against official benchmarks:
  - Adaptation Sets: GSM8K, MATH, MBPP (Train splits)
  - Public Evaluation Sets: GSM8K, MATH, MBPP, ARC, ProofNet, miniF2F
  - Private Held-Out Evaluation Sets: GSM8K, MATH, MBPP, ARC, ProofNet, miniF2F
  - Clean Evaluation Benchmark: HumanEval (Strictly reserved, never in adaptation)
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

from ..dna.structure import Genotype
from ..growth.engine import GrowthEngine
from ..models.phenotype import PhenotypeNeuralNetwork
from ..encoding.slow_clock import SlowClockEncoder
from ..evolution.mutation import GenotypeMutator


def train_phenotype_on_batch(
    model: PhenotypeNeuralNetwork,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    steps: int = 15,
) -> float:
    """Trains phenotype on a batch of adaptation tasks."""
    model.train()
    total_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        h, aux_loss, _, _ = model(batch_inputs, modality="text", is_causal=True)
        logits = model.ar_head(h)
        loss = criterion(logits.view(-1, logits.size(-1)), batch_targets.view(-1)) + aux_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(steps, 1)


def evaluate_phenotype_on_batch(
    model: PhenotypeNeuralNetwork,
    batch_inputs: torch.Tensor,
    batch_targets: torch.Tensor,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Evaluates phenotype on evaluation batch; returns (loss, accuracy)."""
    model.eval()
    with torch.no_grad():
        h, _, _, _ = model(batch_inputs, modality="text", is_causal=True)
        logits = model.ar_head(h)
        loss = criterion(logits.view(-1, logits.size(-1)), batch_targets.view(-1)).item()
        preds = logits.argmax(dim=-1)
        correct = (preds == batch_targets).float().mean().item()
    return loss, correct


def run_experiment_7_official_benchmarks(
    data_dir: str = "./ai-dna-data",
    num_generations: int = 3,
    quick: bool = False,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs multi-generational adaptation and evaluates on Public & Private Held-Out suites.
    """
    from data import AIDNABenchmarkDataset

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("\n" + "="*70)
    print(f" [Experiment 7] Official Benchmark Suite Evaluation ({num_generations} Generations)")
    print(f" Device: {device} | Data Dir: {data_dir} | Quick Mode: {quick}")
    print("="*70)

    # 1. Setup Root Genotype D_0
    current_genotype = Genotype.create_default(genotype_id="gen_0")
    current_genotype.dna_architecture.vocab_size = 256
    current_genotype.dna_architecture.d_model = 32
    current_genotype.dna_architecture.num_layers = 2
    current_genotype.dna_architecture.num_experts = 2
    current_genotype.dna_architecture.d_expert_hidden = 64

    growth_engine = GrowthEngine(device=device)
    slow_clock = SlowClockEncoder(rank_ratio=0.25, encoder_steps=25 if quick else 50, device=device)
    mutator = GenotypeMutator()
    criterion = nn.CrossEntropyLoss()

    sample_cap = 20 if quick else 50
    adapt_tasks = ["gsm8k", "math", "mbpp"]
    eval_tasks = ["gsm8k", "math", "mbpp", "arc", "proofnet", "minif2f"]

    # Preload evaluation batches
    public_eval_batches = {}
    heldout_eval_batches = {}

    for task in eval_tasks:
        try:
            ds_pub = AIDNABenchmarkDataset(task=task, split="public_eval", data_dir=data_dir, max_samples=sample_cap)
            if len(ds_pub) > 0:
                inps = torch.stack([item[0] for item in ds_pub]).to(device)
                tgts = torch.stack([item[1] for item in ds_pub]).to(device)
                public_eval_batches[task] = (inps, tgts)
        except Exception as e:
            print(f"  [WARN] Preloading public eval {task}: {e}")

        try:
            ds_held = AIDNABenchmarkDataset(task=task, split="private_heldout", data_dir=data_dir, max_samples=sample_cap)
            if len(ds_held) > 0:
                inps = torch.stack([item[0] for item in ds_held]).to(device)
                tgts = torch.stack([item[1] for item in ds_held]).to(device)
                heldout_eval_batches[task] = (inps, tgts)
        except Exception as e:
            print(f"  [WARN] Preloading heldout eval {task}: {e}")

    # Preload Clean HumanEval Benchmark (Zero-Shot Clean Test)
    humaneval_batch = None
    try:
        ds_he = AIDNABenchmarkDataset(task="humaneval", split="clean_eval", data_dir=data_dir, max_samples=sample_cap)
        if len(ds_he) > 0:
            inps = torch.stack([item[0] for item in ds_he]).to(device)
            tgts = torch.stack([item[1] for item in ds_he]).to(device)
            humaneval_batch = (inps, tgts)
            print(f"  [OK] Loaded Clean HumanEval Benchmark: {len(ds_he)} problems (Strictly isolated)")
    except Exception as e:
        print(f"  [WARN] Preloading HumanEval: {e}")

    results = {
        "generations": [],
        "public_eval_scores": {},
        "private_heldout_scores": {},
        "humaneval_scores": [],
    }

    for g in range(num_generations):
        print(f"\n--- [Generation D_{g}] Lifecycle & Adaptation ---")

        # 1. Grow Phenotype W_g from Genotype D_g
        phenotype = PhenotypeNeuralNetwork(current_genotype).to(device)
        grown_weights = growth_engine.grow_phenotype_weights(current_genotype)
        p_state = phenotype.state_dict()
        for k, v in grown_weights.items():
            if k in p_state and p_state[k].shape == v.shape:
                p_state[k] = v
        phenotype.load_state_dict(p_state)

        # 2. Benchmark Initial Instinct before Adaptation
        gen_eval = {"generation": f"D_{g}", "public": {}, "heldout": {}}
        for task, (inps, tgts) in public_eval_batches.items():
            _, acc = evaluate_phenotype_on_batch(phenotype, inps, tgts, criterion)
            gen_eval["public"][task] = acc

        for task, (inps, tgts) in heldout_eval_batches.items():
            _, acc = evaluate_phenotype_on_batch(phenotype, inps, tgts, criterion)
            gen_eval["heldout"][task] = acc

        if humaneval_batch is not None:
            _, he_acc = evaluate_phenotype_on_batch(phenotype, humaneval_batch[0], humaneval_batch[1], criterion)
            gen_eval["humaneval_clean"] = he_acc
            results["humaneval_scores"].append(he_acc)
            print(f"  > Clean HumanEval Accuracy: {he_acc:.4f}")

        avg_pub = sum(gen_eval["public"].values()) / max(len(gen_eval["public"]), 1)
        avg_held = sum(gen_eval["heldout"].values()) / max(len(gen_eval["heldout"]), 1)
        print(f"  > Gen D_{g} Benchmark Baseline: Public Avg Acc = {avg_pub:.4f} | Held-Out Avg Acc = {avg_held:.4f}")

        results["generations"].append(gen_eval)

        # 3. Fast Clock: Phenotype Learns on Adaptation Tasks (GSM8K, MATH, MBPP)
        print(f"  > Fast Clock: Adapting phenotype on GSM8K, MATH, MBPP...")
        optimizer = torch.optim.Adam(phenotype.parameters(), lr=1e-3)
        for task in adapt_tasks:
            try:
                ds_adapt = AIDNABenchmarkDataset(task=task, split="adaptation", data_dir=data_dir, max_samples=sample_cap)
                if len(ds_adapt) > 0:
                    inps = torch.stack([item[0] for item in ds_adapt]).to(device)
                    tgts = torch.stack([item[1] for item in ds_adapt]).to(device)
                    loss = train_phenotype_on_batch(phenotype, inps, tgts, optimizer, criterion, steps=8 if quick else 15)
                    print(f"    - Adapted on {task.upper()}: Loss = {loss:.4f}")
            except Exception as e:
                print(f"    [WARN] Adaptation on {task}: {e}")

        # 4. Slow Clock: Distill Learned Instinct back into Genotype D_{g+1} via SVD
        print(f"  > Slow Clock: Encoding learned instinct via Truncated SVD into D_{g+1}...")
        learned_state = {k: v.clone() for k, v in phenotype.state_dict().items()}
        next_genotype, _ = slow_clock.step(current_genotype, learned_state)
        next_genotype = mutator.mutate(next_genotype)
        next_genotype.genotype_id = f"gen_{g+1}"
        current_genotype = next_genotype

    print("\n" + "="*70)
    print(" [Experiment 7] Multi-Generational Benchmark Evaluation Completed!")
    print("="*70)
    return results
