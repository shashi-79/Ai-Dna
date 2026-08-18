"""
Experiment 5: Multi-Generational Evolutionary Scaling.
Tests generational iteration D_0 -> D_1 -> D_2 -> ... -> D_n
and measures whether Sample Efficiency progressively scales S_E(D_{n+1}) >= S_E(D_n)
while tracking retention R_old.
"""

import torch
from typing import Dict, Any, List
from .exp1_svd_hypothesis import (
    generate_synthetic_task,
    train_to_target_accuracy,
)
from ..dna.structure import Genotype
from ..growth.engine import GrowthEngine
from ..models.phenotype import PhenotypeNeuralNetwork
from ..encoding.slow_clock import SlowClockEncoder
from ..evolution.mutation import GenotypeMutator
from ..training.metrics import EvaluationMetrics


def run_experiment_5(num_generations: int = 3, quick: bool = False, device_str: str = "cpu") -> Dict[str, Any]:
    """
    Executes Experiment 5 simulating multi-generational evolution.
    """
    device = torch.device(device_str)
    print(f"=== [Experiment 5] Multi-Generational Evolutionary Scaling ({num_generations} gens) ===")

    # 1. Setup root genotype and environment tasks
    current_genotype = Genotype.create_default(genotype_id="gen_0")
    current_genotype.dna_architecture.vocab_size = 100
    current_genotype.dna_architecture.d_model = 32
    current_genotype.dna_architecture.num_layers = 2
    current_genotype.dna_architecture.num_experts = 2
    current_genotype.dna_architecture.d_expert_hidden = 64

    tasks = [
        generate_synthetic_task(f"task_{i}", num_samples=300, seed=42 + i * 10)
        for i in range(num_generations + 1)
    ]
    val_tasks = [
        generate_synthetic_task(f"task_{i}", num_samples=100, seed=100 + i * 10)
        for i in range(num_generations + 1)
    ]

    max_steps = 30 if quick else 60
    target_acc = 0.40 if quick else 0.55

    # Random baseline on unseen task
    model_rand = PhenotypeNeuralNetwork(current_genotype).to(device)
    steps_rand, _, _ = train_to_target_accuracy(
        model_rand, tasks[-1][0], tasks[-1][1], val_tasks[-1][0], val_tasks[-1][1],
        target_acc=target_acc, max_steps=max_steps, device=device
    )

    slow_clock = SlowClockEncoder(rank_ratio=0.25, encoder_steps=35 if quick else 80, device=device)
    mutator = GenotypeMutator()
    growth_engine = GrowthEngine(device=device)

    generation_history = []
    prev_accuracy_task0 = 0.0

    for g in range(num_generations):
        print(f"-> [Generation {g} -> {g+1}] Learning on Task {g}...")
        x_tr, y_tr = tasks[g]
        x_val, y_val = val_tasks[g]

        # 1. Grow Phenotype
        phenotype = PhenotypeNeuralNetwork(current_genotype).to(device)
        grown_w = growth_engine.grow_phenotype_weights(current_genotype)
        p_state = phenotype.state_dict()
        for k, v in grown_w.items():
            if k in p_state and p_state[k].shape == v.shape:
                p_state[k] = v
        phenotype.load_state_dict(p_state)

        # 2. Fast Clock
        steps_g, acc_g, _ = train_to_target_accuracy(
            phenotype, x_tr, y_tr, x_val, y_val, target_acc=target_acc, max_steps=max_steps, device=device
        )
        if g == 0:
            prev_accuracy_task0 = acc_g

        learned_state = {k: v.clone() for k, v in phenotype.state_dict().items()}

        # 3. Slow Clock
        next_genotype, _ = slow_clock.step(current_genotype, learned_state)

        # 4. Stochastic mutation
        next_genotype = mutator.mutate(next_genotype)
        next_genotype.genotype_id = f"gen_{g+1}"

        # 5. Measure transfer on unseen final task
        test_model = PhenotypeNeuralNetwork(next_genotype).to(device)
        steps_test, acc_test, _ = train_to_target_accuracy(
            test_model, tasks[-1][0], tasks[-1][1], val_tasks[-1][0], val_tasks[-1][1],
            target_acc=target_acc, max_steps=max_steps, device=device
        )
        se = EvaluationMetrics.sample_efficiency(steps_rand, steps_test)

        # 6. Measure retention on Task 0
        ret_model = PhenotypeNeuralNetwork(next_genotype).to(device)
        _, acc_task0_new, _ = train_to_target_accuracy(
            ret_model, tasks[0][0], tasks[0][1], val_tasks[0][0], val_tasks[0][1],
            target_acc=target_acc, max_steps=10, device=device
        )
        r_old = EvaluationMetrics.retention_metric(acc_task0_new, prev_accuracy_task0)

        record = {
            "generation": g + 1,
            "sample_efficiency": se,
            "retention_r_old": r_old,
            "steps_to_target": steps_test,
        }
        generation_history.append(record)
        print(f"   Gen {g+1}: Sample Efficiency S_E = {se:.2f} | Retention R_old = {r_old:.2%}")
        current_genotype = next_genotype

    return {"history": generation_history, "baseline_steps": steps_rand}
