"""
Experiment 2: Transferability Curve S_E = f(E_k).
Plots and evaluates correlation between retained SVD singular energy E_k
and downstream Sample Efficiency S_E.
"""

import torch
from typing import Dict, Any, List
from .exp1_svd_hypothesis import (
    generate_synthetic_task,
    train_to_target_accuracy,
)
from ..dna.structure import Genotype
from ..models.phenotype import PhenotypeNeuralNetwork
from ..encoding.svd_filter import SVDInstinctFilter
from ..training.metrics import EvaluationMetrics


def run_experiment_2(quick: bool = False, device_str: str = "cpu") -> Dict[str, Any]:
    """
    Executes Experiment 2 measuring S_E = f(E_k) across granular energy retention thresholds.
    """
    device = torch.device(device_str)
    print("=== [Experiment 2] Transferability Curve S_E = f(E_k) ===")

    genotype = Genotype.create_default(genotype_id="exp2_root")
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

    # 1. Train on Task A
    model_a = PhenotypeNeuralNetwork(genotype).to(device)
    train_to_target_accuracy(model_a, x_train_a, y_train_a, x_val_a, y_val_a, target_acc=target_acc, max_steps=max_steps, device=device)
    learned_state = {k: v.clone() for k, v in model_a.state_dict().items()}

    # 2. Random Baseline on Task B
    model_wr = PhenotypeNeuralNetwork(genotype).to(device)
    steps_wr, _, _ = train_to_target_accuracy(model_wr, x_train_b, y_train_b, x_val_b, y_val_b, target_acc=target_acc, max_steps=max_steps, device=device)

    # 3. Energy spectrum evaluation
    ratios = [0.05, 0.15, 0.30, 0.50, 0.80] if not quick else [0.10, 0.30, 0.60]
    curve_data = []

    for r in ratios:
        svd_state, energies = SVDInstinctFilter.filter_state_dict(learned_state, rank_ratio=r)
        mean_energy = sum(energies.values()) / max(1, len(energies))

        model_test = PhenotypeNeuralNetwork(genotype).to(device)
        model_test.load_state_dict(svd_state)
        steps_test, acc_test, _ = train_to_target_accuracy(
            model_test, x_train_b, y_train_b, x_val_b, y_val_b,
            target_acc=target_acc, max_steps=max_steps, device=device
        )
        se = EvaluationMetrics.sample_efficiency(steps_wr, steps_test)
        curve_data.append({
            "rank_ratio": r,
            "retained_energy": mean_energy,
            "sample_efficiency": se,
            "steps_to_target": steps_test,
            "val_accuracy": acc_test,
        })
        print(f"   E_k: {mean_energy:6.2%} -> Sample Efficiency S_E = {se:4.2f} (Steps: {steps_test:2d})")

    return {"curve": curve_data, "baseline_steps": steps_wr}
