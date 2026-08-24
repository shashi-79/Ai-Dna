"""
Experiment 1: LoRA Instinct-Filter Hypothesis (formerly SVD Hypothesis).
Tests whether LoRA adapters trained on a source task (T_A) transfer useful structural information
to an unseen task (T_B) better than randomly initialized LoRA adapters of the same rank.

Baselines evaluated:
1. Baseline 1 (Random): Full model W_R trained on T_B from scratch.
2. Baseline 2 (Full Transfer): Full model W* trained on T_A, transferred and fine-tuned on T_B.
3. Baseline 3 (LoRA Transfer): Base model + LoRA adapters trained on T_A, transferred and fine-tuned on T_B.
4. Baseline 4 (Random LoRA): Base model + Random LoRA adapters trained on T_B.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from ..dna.structure import Genotype
from ..models.phenotype import PhenotypeNeuralNetwork
from ..models.lora import replace_linear_with_lora, freeze_model_except_lora, extract_lora_parameters
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
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    target_acc: float = 0.70,
    max_steps: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[int, float, List[float]]:
    """
    Trains a model until reaching target validation accuracy or max_steps.
    Returns: (steps_taken, final_val_acc, history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = FastClockTrainer(phenotype=model, learning_rate=lr, device=device)
    num_batches = max(1, x_train.shape[0] // batch_size)
    history = []

    steps_to_target = max_steps
    reached_target = False

    indices = torch.randperm(x_train.size(0))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    for step in range(1, max_steps + 1):
        idx = (step - 1) % num_batches
        if idx == 0 and step > 1:
            indices = torch.randperm(x_train.size(0))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

        batch_x = x_train_shuffled[idx * batch_size : (idx + 1) * batch_size]
        batch_y = y_train_shuffled[idx * batch_size : (idx + 1) * batch_size]

        loss, _ = trainer.train_step_classification(batch_x, batch_y, modality="text")

        if step % 5 == 0 or step == max_steps:
            val_acc, _ = trainer.evaluate_classification(x_val, y_val, modality="text")
            history.append(val_acc)
            if val_acc >= target_acc:
                steps_to_target = step
                reached_target = True
                break

    if reached_target:
        final_acc = val_acc
    else:
        final_acc, _ = trainer.evaluate_classification(x_val, y_val, modality="text")
        
    return steps_to_target, final_acc, history


def run_experiment_1(quick: bool = False, device_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Experiment 1 comparing W_R, W*, LoRA Transfer, and Random LoRA on unseen task T_B.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=== [Experiment 1] LoRA Instinct-Filter Hypothesis ===")

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

    # 2. Baseline 1: Random Initialization W_R on Task B
    print("-> Evaluating Baseline 1 (W_R - Random Full Model) on Task B...")
    model_random = PhenotypeNeuralNetwork(genotype).to(device)
    steps_wr, acc_wr, _ = train_to_target_accuracy(
        model_random, x_train_b, y_train_b, x_val_b, y_val_b,
        target_acc=target_acc, max_steps=max_steps, device=device
    )
    print(f"   Baseline 1 (W_R) - Steps: {steps_wr}, Val Acc: {acc_wr:.2%}")

    # 3. Phase 1: Train Full Phenotype W_0 -> W* on Task A
    model_a = PhenotypeNeuralNetwork(genotype).to(device)
    # Save the base untrained weights to ensure fair initialization across runs
    base_state_dict = {k: v.clone() for k, v in model_a.state_dict().items()}
    
    print("-> Training Full Phenotype on Task A to produce W*...")
    steps_a, acc_a, _ = train_to_target_accuracy(
        model_a, x_train_a, y_train_a, x_val_a, y_val_a,
        target_acc=target_acc, max_steps=max_steps, device=device
    )
    learned_full_state = {k: v.clone() for k, v in model_a.state_dict().items()}
    print(f"   Task A Complete. Steps: {steps_a}, Final Val Acc: {acc_a:.2%}")

    # 4. Baseline 2: Full Trained Model W* on Task B
    print("-> Evaluating Baseline 2 (W* - Full Transfer) on Task B...")
    model_full = PhenotypeNeuralNetwork(genotype).to(device)
    model_full.load_state_dict(learned_full_state)
    steps_full, acc_full, _ = train_to_target_accuracy(
        model_full, x_train_b, y_train_b, x_val_b, y_val_b,
        target_acc=target_acc, max_steps=max_steps, device=device
    )
    print(f"   Baseline 2 (W*) - Steps: {steps_full}, Val Acc: {acc_full:.2%}")

    # 5. Spectrum of LoRA ranks
    ranks = [2, 4] if quick else [1, 2, 4, 8]
    results_lora = {}
    results_rand_lora = {}

    for r in ranks:
        # Phase 1 (LoRA): Train LoRA adapters on Task A
        model_lora_a = PhenotypeNeuralNetwork(genotype).to(device)
        model_lora_a.load_state_dict(base_state_dict)
        replace_linear_with_lora(model_lora_a, rank=r)
        freeze_model_except_lora(model_lora_a)
        
        steps_lora_a, _, _ = train_to_target_accuracy(
            model_lora_a, x_train_a, y_train_a, x_val_a, y_val_a,
            target_acc=target_acc, max_steps=max_steps, device=device
        )
        lora_learned_adapters = extract_lora_parameters(model_lora_a)

        # Baseline 3: Transfer learned adapters to Task B
        model_lora_b = PhenotypeNeuralNetwork(genotype).to(device)
        model_lora_b.load_state_dict(base_state_dict)
        replace_linear_with_lora(model_lora_b, rank=r)
        freeze_model_except_lora(model_lora_b)
        
        # Inject learned adapters
        model_lora_b.load_state_dict(lora_learned_adapters, strict=False)
        
        steps_lora, acc_lora, _ = train_to_target_accuracy(
            model_lora_b, x_train_b, y_train_b, x_val_b, y_val_b,
            target_acc=target_acc, max_steps=max_steps, device=device
        )
        se_lora = EvaluationMetrics.sample_efficiency(steps_wr, steps_lora)
        results_lora[r] = {
            "steps": steps_lora,
            "acc": acc_lora,
            "sample_efficiency": se_lora,
        }

        # Baseline 4: Random LoRA adapters of same rank on Task B
        model_rand_lora = PhenotypeNeuralNetwork(genotype).to(device)
        model_rand_lora.load_state_dict(base_state_dict)
        replace_linear_with_lora(model_rand_lora, rank=r)
        freeze_model_except_lora(model_rand_lora)
        
        steps_rand_lora, acc_rand_lora, _ = train_to_target_accuracy(
            model_rand_lora, x_train_b, y_train_b, x_val_b, y_val_b,
            target_acc=target_acc, max_steps=max_steps, device=device
        )
        se_rand_lora = EvaluationMetrics.sample_efficiency(steps_wr, steps_rand_lora)
        results_rand_lora[r] = {
            "steps": steps_rand_lora,
            "acc": acc_rand_lora,
            "sample_efficiency": se_rand_lora,
        }

        print(f"   [Rank {r:2d}] LoRA Transfer: S_E={se_lora:4.2f} (Steps: {steps_lora:2d}) vs Random LoRA: S_E={se_rand_lora:4.2f} (Steps: {steps_rand_lora:2d})")

    return {
        "baseline_1_random": {"steps": steps_wr, "acc": acc_wr, "se": 1.0},
        "baseline_2_full": {"steps": steps_full, "acc": acc_full, "se": EvaluationMetrics.sample_efficiency(steps_wr, steps_full)},
        "baseline_3_lora": results_lora,
        "baseline_4_random_lora": results_rand_lora,
    }

