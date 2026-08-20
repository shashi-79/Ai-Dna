"""
Experiment 6: Multi-Parent Fusion and Specialization Transfer.
Trains parent D_A on Task A and parent D_B on Task B.
Fuses into offspring D_C = F(D_A, D_B).
Evaluates zero-shot and few-shot capabilities across T_A, T_B, and joint T_AB.
"""

import torch
from typing import Dict, Any, Optional
from .exp1_svd_hypothesis import (
    generate_synthetic_task,
    train_to_target_accuracy,
)
from ..dna.structure import Genotype
from ..growth.engine import GrowthEngine
from ..models.phenotype import PhenotypeNeuralNetwork
from ..encoding.slow_clock import SlowClockEncoder
from ..evolution.fusion import MultiParentFusion
from ..training.fast_clock import FastClockTrainer


def run_experiment_6(quick: bool = False, device_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Experiment 6 testing multi-parent fusion and skill combination.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print("=== [Experiment 6] Multi-Parent Specialization Fusion ===")

    # 1. Datasets for Task A, Task B, and Joint Task AB
    x_train_a, y_train_a = generate_synthetic_task("task_A", num_samples=300, seed=42)
    x_val_a, y_val_a = generate_synthetic_task("task_A", num_samples=100, seed=43)

    x_train_b, y_train_b = generate_synthetic_task("task_B", num_samples=300, seed=100)
    x_val_b, y_val_b = generate_synthetic_task("task_B", num_samples=100, seed=101)

    # Joint Task AB: Concatenate validation sets
    x_val_ab = torch.cat([x_val_a, x_val_b], dim=0)
    y_val_ab = torch.cat([y_val_a, y_val_b], dim=0)

    max_steps = 30 if quick else 60
    target_acc = 0.40 if quick else 0.55

    # 2. Train Parent A on Task A
    print("-> Training Parent A on Task A...")
    d_root_a = Genotype.create_default(genotype_id="Parent_A")
    d_root_a.dna_architecture.vocab_size = 100
    d_root_a.dna_architecture.d_model = 32
    d_root_a.dna_architecture.num_layers = 2
    d_root_a.dna_architecture.num_experts = 2
    d_root_a.dna_architecture.d_expert_hidden = 64

    model_a = PhenotypeNeuralNetwork(d_root_a).to(device)
    train_to_target_accuracy(model_a, x_train_a, y_train_a, x_val_a, y_val_a, target_acc=target_acc, max_steps=max_steps, device=device)
    slow_clock = SlowClockEncoder(rank_ratio=0.25, encoder_steps=35 if quick else 80, device=device)
    parent_d_a, _ = slow_clock.step(d_root_a, {k: v.clone() for k, v in model_a.state_dict().items()})
    parent_d_a.genotype_id = "Parent_A"

    # 3. Train Parent B on Task B
    print("-> Training Parent B on Task B...")
    d_root_b = Genotype.create_default(genotype_id="Parent_B")
    d_root_b.dna_architecture.vocab_size = 100
    d_root_b.dna_architecture.d_model = 32
    d_root_b.dna_architecture.num_layers = 2
    d_root_b.dna_architecture.num_experts = 2
    d_root_b.dna_architecture.d_expert_hidden = 64

    model_b = PhenotypeNeuralNetwork(d_root_b).to(device)
    train_to_target_accuracy(model_b, x_train_b, y_train_b, x_val_b, y_val_b, target_acc=target_acc, max_steps=max_steps, device=device)
    parent_d_b, _ = slow_clock.step(d_root_b, {k: v.clone() for k, v in model_b.state_dict().items()})
    parent_d_b.genotype_id = "Parent_B"

    # 4. Multi-Parent Fusion D_c = F(D_A, D_B)
    print("-> Fusing Parent A and Parent B into Child D_C...")
    fusion_engine = MultiParentFusion(min_compatibility=0.5)
    child_d_c = fusion_engine.fuse([parent_d_a, parent_d_b], weights=[0.5, 0.5], child_id="Child_C")

    # 5. Evaluate Zero-Shot and Fine-Tuning Performance
    growth_engine = GrowthEngine(device=device)
    child_model = PhenotypeNeuralNetwork(child_d_c).to(device)
    grown_w = growth_engine.grow_phenotype_weights(child_d_c)
    c_state = child_model.state_dict()
    for k, v in grown_w.items():
        if k in c_state and c_state[k].shape == v.shape:
            c_state[k] = v
    child_model.load_state_dict(c_state)

    trainer_child = FastClockTrainer(child_model, device=device)
    acc_a_zero, _ = trainer_child.evaluate_classification(x_val_a, y_val_a, modality="text")
    acc_b_zero, _ = trainer_child.evaluate_classification(x_val_b, y_val_b, modality="text")
    acc_ab_zero, _ = trainer_child.evaluate_classification(x_val_ab, y_val_ab, modality="text")

    print(f"   Fused Child Zero-Shot Accuracy on Task A:  {acc_a_zero:.2%}")
    print(f"   Fused Child Zero-Shot Accuracy on Task B:  {acc_b_zero:.2%}")
    print(f"   Fused Child Zero-Shot Accuracy on Joint AB: {acc_ab_zero:.2%}")

    return {
        "parent_a_id": parent_d_a.genotype_id,
        "parent_b_id": parent_d_b.genotype_id,
        "child_id": child_d_c.genotype_id,
        "acc_task_a_zero_shot": acc_a_zero,
        "acc_task_b_zero_shot": acc_b_zero,
        "acc_joint_ab_zero_shot": acc_ab_zero,
    }
