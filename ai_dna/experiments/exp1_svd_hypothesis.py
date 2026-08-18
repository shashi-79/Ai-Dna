"""
Experiment 1: SVD Instinct-Filter Hypothesis.
Tests whether dominant singular components of W* transfer useful structural information
to an unseen task T_B better than random low-rank controls W_k^random.

Baselines evaluated:
1. Baseline 1 (Random): W_R
2. Baseline 2 (Full trained model): W*
3. Baseline 3 (SVD reconstruction): W_k^SVD for k in {1%, 5%, 10%, 25%, 50%, 75%, 100%}
4. Baseline 4 (Random low-rank): W_k^random for matching k
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from ..dna.structure import Genotype
from ..models.phenotype import PhenotypeNeuralNetwork
from ..encoding.svd_filter import SVDInstinctFilter
from ..training.fast_clock import FastClockTrainer
from ..training.metrics import EvaluationMetrics


def generate_synthetic_task(
    task_id: str = "task_A",
    num_samples: int = 400,
    seq_len: int = 16,
    vocab_size: int = 100,
    num_classes: int = 10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates structured classification dataset with non-linear feature interactions.
    """
    torch.manual_seed(seed)
    if task_id == "task_A":
        # Task A: High-frequency modulo pattern
        x = torch.randint(0, vocab_size, (num_samples, seq_len))
        y = (x.sum(dim=1) % num_classes).long()
    else:
        # Task B (Unseen): Quadratic parity pattern
        x = torch.randint(0, vocab_size, (num_samples, seq_len))
        y = (((x[:, 0] * x[:, 1]) + x[:, -1]) % num_classes).long()
    return x, y


def train_to_target_accuracy(
    model: PhenotypeNeuralNetwork,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    target_acc: float = 0.70,
    max_steps: int = 100,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float, List[float]]:
    """
    Trains a phenotype model until reaching target validation accuracy or max_steps.
    Returns: (steps_taken, final_val_acc, history)
    """
    trainer = FastClockTrainer(phenotype=model, learning_rate=lr, device=device)
    batch_size = 32
    num_batches = max(1, x_train.shape[0] // batch_size)
    history = []

    steps_to_target = max_steps
    reached_target = False

    for step in range(1, max_steps + 1):
        idx = (step - 1) % num_batches
        batch_x = x_train[idx * batch_size : (idx + 1) * batch_size]
        batch_y = y_train[idx * batch_size : (idx + 1) * batch_size]

        loss, _ = trainer.train_step_classification(batch_x, batch_y, modality="text")

        if step % 5 == 0 or step == max_steps:
            val_acc, _ = trainer.evaluate_classification(x_val, y_val, modality="text")
            history.append(val_acc)
            if val_acc >= target_acc and not reached_target:
                steps_to_target = step
                reached_target = True

    final_acc, _ = trainer.evaluate_classification(x_val, y_val, modality="text")
    return steps_to_target, final_acc, history


def run_experiment_1(quick: bool = False, device_str: str = "cpu") -> Dict[str, Any]:
    """
    Executes Experiment 1 comparing W_R, W*, W_k^SVD, and W_k^random on unseen task T_B.
    """
    device = torch.device(device_str)
    print("=== [Experiment 1] SVD Instinct-Filter Hypothesis ===")

    # 1. Prepare Genotype and Task Data
    genotype = Genotype.create_default(genotype_id="exp1_root")
    genotype.dna_architecture.vocab_size = 100
    genotype.dna_architecture.d_model = 32
    genotype.dna_architecture.num_layers = 2
    genotype.dna_architecture.num_experts = 2
    genotype.dna_architecture.d_expert_hidden = 64

    x_train_a, y_train_a = generate_synthetic_task("task_A", num_samples=300, seed=42)
    x_val_a, y_val_a = generate_synthetic_task("task_A", num_samples=100, seed=43)

    x_train_b, y_train_b = generate_synthetic_task("task_B", num_samples=300, seed=100)
    x_val_b, y_val_b = generate_synthetic_task("task_B", num_samples=100, seed=101)

    max_steps = 40 if quick else 80
    target_acc = 0.40 if quick else 0.60

    # 2. Phase 1: Train Phenotype W_0 -> W* on Task A
    model_a = PhenotypeNeuralNetwork(genotype).to(device)
    print("-> Training Phenotype on Task A to produce W*...")
    steps_a, acc_a, _ = train_to_target_accuracy(
        model_a, x_train_a, y_train_a, x_val_a, y_val_a,
        target_acc=target_acc, max_steps=max_steps, device=device
    )
    learned_state_dict = {k: v.clone() for k, v in model_a.state_dict().items()}
    print(f"   Task A Complete. Steps: {steps_a}, Final Val Acc: {acc_a:.2%}")

    # 3. Baseline 1: Random Initialization W_R on Task B
    print("-> Evaluating Baseline 1 (W_R - Random Initialization) on Task B...")
    model_random = PhenotypeNeuralNetwork(genotype).to(device)
    steps_wr, acc_wr, _ = train_to_target_accuracy(
        model_random, x_train_b, y_train_b, x_val_b, y_val_b,
        target_acc=target_acc, max_steps=max_steps, device=device
    )
    print(f"   Baseline 1 (W_R) - Steps: {steps_wr}, Val Acc: {acc_wr:.2%}")

    # 4. Baseline 2: Full Trained Model W* on Task B
    print("-> Evaluating Baseline 2 (W* - Full Trained Model) on Task B...")
    model_full = PhenotypeNeuralNetwork(genotype).to(device)
    model_full.load_state_dict(learned_state_dict)
    steps_full, acc_full, _ = train_to_target_accuracy(
        model_full, x_train_b, y_train_b, x_val_b, y_val_b,
        target_acc=target_acc, max_steps=max_steps, device=device
    )
    print(f"   Baseline 2 (W*) - Steps: {steps_full}, Val Acc: {acc_full:.2%}")

    # 5. Spectrum of rank ratios: 10%, 25%, 50%, 75%
    rank_ratios = [0.10, 0.25, 0.50] if quick else [0.05, 0.10, 0.25, 0.50, 0.75]
    results_svd = {}
    results_rand_lowrank = {}

    for r in rank_ratios:
        # Baseline 3: SVD Reconstruction W_k^SVD
        svd_state, energies = SVDInstinctFilter.filter_state_dict(learned_state_dict, rank_ratio=r)
        mean_energy = sum(energies.values()) / max(1, len(energies))

        model_svd = PhenotypeNeuralNetwork(genotype).to(device)
        model_svd.load_state_dict(svd_state)
        steps_svd, acc_svd, _ = train_to_target_accuracy(
            model_svd, x_train_b, y_train_b, x_val_b, y_val_b,
            target_acc=target_acc, max_steps=max_steps, device=device
        )
        se_svd = EvaluationMetrics.sample_efficiency(steps_wr, steps_svd)
        results_svd[r] = {
            "steps": steps_svd,
            "acc": acc_svd,
            "mean_energy": mean_energy,
            "sample_efficiency": se_svd,
        }

        # Baseline 4: Random Low-Rank Control W_k^random matching identical rank k
        rand_lr_state = {}
        for k_name, v_param in learned_state_dict.items():
            if v_param.ndim >= 2 and ("weight" in k_name) and not ("norm" in k_name or "ln" in k_name):
                k_rank = max(1, int(min(v_param.shape[0], v_param.shape[1]) * r))
                rand_lr_state[k_name] = SVDInstinctFilter.generate_random_low_rank(v_param, k_rank)
            else:
                rand_lr_state[k_name] = v_param.clone()

        model_rand_lr = PhenotypeNeuralNetwork(genotype).to(device)
        model_rand_lr.load_state_dict(rand_lr_state)
        steps_rand_lr, acc_rand_lr, _ = train_to_target_accuracy(
            model_rand_lr, x_train_b, y_train_b, x_val_b, y_val_b,
            target_acc=target_acc, max_steps=max_steps, device=device
        )
        se_rand_lr = EvaluationMetrics.sample_efficiency(steps_wr, steps_rand_lr)
        results_rand_lowrank[r] = {
            "steps": steps_rand_lr,
            "acc": acc_rand_lr,
            "sample_efficiency": se_rand_lr,
        }

        print(f"   [Rank {r:4.0%}] SVD: S_E={se_svd:4.2f} (Steps: {steps_svd:2d}) vs Random LR: S_E={se_rand_lr:4.2f} (Steps: {steps_rand_lr:2d}) | Energy: {mean_energy:.1%}")

    return {
        "baseline_1_random": {"steps": steps_wr, "acc": acc_wr, "se": 1.0},
        "baseline_2_full": {"steps": steps_full, "acc": acc_full, "se": EvaluationMetrics.sample_efficiency(steps_wr, steps_full)},
        "baseline_3_svd": results_svd,
        "baseline_4_random_lowrank": results_rand_lowrank,
    }
