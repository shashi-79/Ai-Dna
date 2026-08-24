"""
Experiment 2: Transferability Curve S_E = f(r).
Plots and evaluates correlation between LoRA Rank r
and downstream Sample Efficiency S_E.
"""

import torch
from typing import Dict, Any, List, Optional
from .exp1_lora_hypothesis import (
    generate_synthetic_task,
    train_to_target_accuracy,
)
from ..dna.structure import Genotype
from ..models.phenotype import PhenotypeNeuralNetwork
from ..models.lora import replace_linear_with_lora, freeze_model_except_lora, extract_lora_parameters
from ..training.metrics import EvaluationMetrics


def run_experiment_2(quick: bool = False, device_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Experiment 2, sweeping over LoRA ranks to measure S_E.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=== [Experiment 2] Transferability Curve S_E = f(r) ===")

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

    print("-> Computing Baseline 1 (W_R - Random)...")
    model_random = PhenotypeNeuralNetwork(genotype).to(device)
    steps_wr, _, _ = train_to_target_accuracy(
        model_random, x_train_b, y_train_b, x_val_b, y_val_b,
        target_acc=target_acc, max_steps=max_steps, device=device
    )
    print(f"   W_R Steps: {steps_wr}")

    ranks_to_test = [1, 2, 4] if quick else [1, 2, 4, 8, 16]
    
    # Save base state to ensure we always start from the exact same random base
    base_state_dict = {k: v.clone() for k, v in model_random.state_dict().items()}

    curve_results = {}
    print("-> Sweeping LoRA ranks (r)...")
    for r in ranks_to_test:
        # 1. Train adapters on Task A
        model_a = PhenotypeNeuralNetwork(genotype).to(device)
        model_a.load_state_dict(base_state_dict)
        replace_linear_with_lora(model_a, rank=r)
        freeze_model_except_lora(model_a)
        
        train_to_target_accuracy(
            model_a, x_train_a, y_train_a, x_val_a, y_val_a,
            target_acc=target_acc, max_steps=max_steps, device=device
        )
        lora_adapters = extract_lora_parameters(model_a)

        # 2. Transfer to Task B
        model_b = PhenotypeNeuralNetwork(genotype).to(device)
        model_b.load_state_dict(base_state_dict)
        replace_linear_with_lora(model_b, rank=r)
        freeze_model_except_lora(model_b)
        model_b.load_state_dict(lora_adapters, strict=False)

        steps_b, acc_b, _ = train_to_target_accuracy(
            model_b, x_train_b, y_train_b, x_val_b, y_val_b,
            target_acc=target_acc, max_steps=max_steps, device=device
        )

        s_e = EvaluationMetrics.sample_efficiency(steps_wr, steps_b)
        curve_results[r] = {
            "steps": steps_b,
            "sample_efficiency": s_e,
        }
        print(f"   [Rank {r:2d}] S_E = {s_e:4.2f} (Steps: {steps_b:2d})")

    return {
        "baseline_steps": steps_wr,
        "curve": curve_results,
    }
