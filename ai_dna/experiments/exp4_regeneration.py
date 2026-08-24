"""
Experiment 4: Genotypic Regeneration.
Tests the complete bidirectional lifecycle:
D_0 -> W_0 -> Train(T_A) -> W_A* -> SlowClock -> D_1 -> (Discard W_A*) -> Regenerate W_1 = G(D_1) -> Train(T_B).
Directly measures whether the compact genotype preserves and accelerates downstream learning.
"""

import torch
from typing import Dict, Any, Optional
from .exp1_lora_hypothesis import (
    generate_synthetic_task,
    train_to_target_accuracy,
)
from ..dna.structure import Genotype
from ..growth.engine import GrowthEngine
from ..models.phenotype import PhenotypeNeuralNetwork
from ..encoding.slow_clock import SlowClockEncoder
from ..training.metrics import EvaluationMetrics


def run_experiment_4(quick: bool = False, device_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Experiment 4 validating full genotypic regeneration.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print("=== [Experiment 4] Genotypic Regeneration Lifecycle ===")

    # 1. Initialize root Genotype D_0
    d_0 = Genotype.create_default(genotype_id="D_0")
    d_0.dna_architecture.vocab_size = 100
    d_0.dna_architecture.d_model = 32
    d_0.dna_architecture.num_layers = 2
    d_0.dna_architecture.num_experts = 2
    d_0.dna_architecture.d_expert_hidden = 64

    x_train_a, y_train_a = generate_synthetic_task("task_A", num_samples=300, seed=42)
    x_val_a, y_val_a = generate_synthetic_task("task_A", num_samples=100, seed=43)
    x_train_b, y_train_b = generate_synthetic_task("task_B", num_samples=300, seed=100)
    x_val_b, y_val_b = generate_synthetic_task("task_B", num_samples=100, seed=101)

    max_steps = 35 if quick else 70
    target_acc = 0.40 if quick else 0.55

    # 2. Grow Phenotype W_0 = G(D_0)
    print("-> Growing Phenotype W_0 from D_0...")
    growth_engine = GrowthEngine(device=device)
    w_0 = PhenotypeNeuralNetwork(d_0).to(device)

    # 3. Fast Clock: Train on Task A -> W_A*
    print("-> Fast Clock: Learning on Task A...")
    train_to_target_accuracy(w_0, x_train_a, y_train_a, x_val_a, y_val_a, target_acc=target_acc, max_steps=max_steps, device=device)
    learned_state_a = {k: v.clone() for k, v in w_0.state_dict().items()}

    # 4. Slow Clock: Encode transferable structural instinct into D_1
    print("-> Slow Clock: Distilling structural instinct into D_1...")
    slow_clock = SlowClockEncoder(
        rank_ratio=0.25,
        encoder_steps=50 if quick else 120,
        device=device,
    )
    d_1, summary = slow_clock.step(d_0, learned_state_a)

    # 5. Discard W_A* and Regenerate W_1 = G(D_1)
    print("-> Discarding W_A* and Regenerating Phenotype W_1 = G(D_1)...")
    del w_0, learned_state_a
    w_1 = PhenotypeNeuralNetwork(d_1).to(device)
    grown_weights_1 = growth_engine.grow_phenotype_weights(d_1)
    w1_state = w_1.state_dict()
    for k, v in grown_weights_1.items():
        if k in w1_state and w1_state[k].shape == v.shape:
            w1_state[k] = v
    w_1.load_state_dict(w1_state)

    # 6. Train W_1 on unseen Task B and compare against Baseline 1 (Random)
    print("-> Evaluating regenerated model W_1 on unseen Task B...")
    model_rand = PhenotypeNeuralNetwork(d_0).to(device)
    steps_rand, _, _ = train_to_target_accuracy(model_rand, x_train_b, y_train_b, x_val_b, y_val_b, target_acc=target_acc, max_steps=max_steps, device=device)
    steps_w1, acc_w1, _ = train_to_target_accuracy(w_1, x_train_b, y_train_b, x_val_b, y_val_b, target_acc=target_acc, max_steps=max_steps, device=device)

    se = EvaluationMetrics.sample_efficiency(steps_rand, steps_w1)
    print(f"   Regenerated Model Sample Efficiency: S_E = {se:.2f} (Steps: {steps_w1:2d} vs Random: {steps_rand:2d})")

    return {
        "d_0_id": d_0.genotype_id,
        "d_1_id": d_1.genotype_id,
        "sample_efficiency": se,
        "steps_random": steps_rand,
        "steps_regenerated": steps_w1,
        "slow_clock_summary": summary,
    }
