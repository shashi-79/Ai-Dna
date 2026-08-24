"""
Experiment 3: LoRA-CPPN Encoding and Growth Validation.
Validates W_base -> LoRA -> CPPN -> DNA -> Growth -> W_regen.
Evaluates:
- True Compression Ratio C_R = |theta_model| / (|D| + |theta_G| + |theta_S|)
- Normalized Frobenius Reconstruction Error
- Behavioral Divergence D_KL
- Sample efficiency S_E on unseen task T_B.
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
from ..models.lora import replace_linear_with_lora, freeze_model_except_lora, extract_lora_parameters
from ..encoding.cppn_encoder import InverseCPPNEncoder
from ..training.metrics import EvaluationMetrics


def run_experiment_3(quick: bool = False, device_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes Experiment 3 testing LoRA adapter genetic encoding via CPPN.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print("=== [Experiment 3] LoRA-CPPN Encoding and Growth Engine ===")

    # 1. Setup Genotype and Datasets
    genotype = Genotype.create_default(genotype_id="exp3_root")
    genotype.dna_architecture.vocab_size = 100
    genotype.dna_architecture.d_model = 32
    genotype.dna_architecture.num_layers = 2
    genotype.dna_architecture.num_experts = 2
    genotype.dna_architecture.d_expert_hidden = 64
    genotype.dna_architecture.lora_rank = 4
    genotype.dna_instinct.cppn_hidden_dim = 24
    genotype.dna_instinct.cppn_layers = 2

    # Initialize Base CPPN Parameters
    from ..growth.cppn import CPPNNetwork
    base_cppn = CPPNNetwork(in_features=32, hidden_dim=24, num_layers=2, out_features=1)
    genotype.dna_instinct.genetic_parameters = base_cppn.get_parameter_dict()

    x_train_a, y_train_a = generate_synthetic_task("task_A", num_samples=300, seed=42)
    x_val_a, y_val_a = generate_synthetic_task("task_A", num_samples=100, seed=43)
    x_train_b, y_train_b = generate_synthetic_task("task_B", num_samples=300, seed=100)
    x_val_b, y_val_b = generate_synthetic_task("task_B", num_samples=100, seed=101)

    max_steps = 30 if quick else 60
    target_acc = 0.40 if quick else 0.55

    # 2. Base Model and Train LoRA on Task A
    print("-> Training LoRA Adapters on Task A...")
    model_base = PhenotypeNeuralNetwork(genotype).to(device)
    base_state_dict = {k: v.clone() for k, v in model_base.state_dict().items()}
    
    replace_linear_with_lora(model_base, rank=4)
    freeze_model_except_lora(model_base)

    train_to_target_accuracy(model_base, x_train_a, y_train_a, x_val_a, y_val_a, target_acc=target_acc, max_steps=max_steps, device=device)
    lora_adapters = extract_lora_parameters(model_base)

    # 3. Inverse CPPN Encoding LoRA -> D_1
    print("-> Distilling extracted LoRA adapters into compact Genotype DNA...")
    encoder = InverseCPPNEncoder(
        learning_rate=1e-2,
        max_steps=50 if quick else 120,
        device=device,
    )
    # Note: InverseCPPNEncoder operates on the provided state dict, which is just the LoRA parameters
    genotype_d1, recon_loss, _ = encoder.encode_genotype(genotype, lora_adapters)

    # Prefix encoded parameters with 'adapter.' to follow CL-DNA conventions
    encoded_adapters = {f"adapter.{k}": v for k, v in genotype_d1.dna_instinct.genetic_parameters.items()}
    
    # Create merged genotype with both base CPPN parameters and new adapter CPPN parameters
    merged_genotype = Genotype.create_default(genotype_id="exp3_merged")
    merged_genotype.dna_architecture = genotype.dna_architecture
    merged_genotype.dna_instinct.cppn_hidden_dim = genotype.dna_instinct.cppn_hidden_dim
    merged_genotype.dna_instinct.cppn_layers = genotype.dna_instinct.cppn_layers
    merged_genotype.dna_instinct.genetic_parameters = {
        **base_cppn.get_parameter_dict(),
        **encoded_adapters
    }

    # 4. Growth Engine Regeneration D_1 -> W_regen
    print("-> Regenerating Phenotype from Genotype via Growth Engine...")
    growth_engine = GrowthEngine(device=device)
    model_grown = growth_engine.grow_phenotype_model(merged_genotype)
    model_grown.to(device)

    # 5. Measure Behavioral Divergence D_KL
    with torch.no_grad():
        model_base.eval()
        model_grown.eval()
        h_orig, _, _, _ = model_base(x_val_a[:32].to(device), modality="text")
        logits_orig = model_base.ar_head(h_orig)
        h_grown, _, _, _ = model_grown(x_val_a[:32].to(device), modality="text")
        logits_grown = model_grown.ar_head(h_grown)
        kl_div = EvaluationMetrics.behavioral_divergence(logits_orig, logits_grown)

    # 6. Compute True Compression Ratio C_R
    model_param_count = sum(p.numel() for p in model_base.parameters())
    dna_param_count = merged_genotype.total_parameters()
    c_r = EvaluationMetrics.true_compression_ratio(
        model_parameters=model_param_count,
        dna_parameters=dna_param_count,
    )

    # 7. Evaluate Downstream Transfer to Task B
    print("-> Evaluating Regenerated Model on Unseen Task B...")
    freeze_model_except_lora(model_grown)
    steps_regen, acc_regen, _ = train_to_target_accuracy(
        model_grown, x_train_b, y_train_b, x_val_b, y_val_b,
        target_acc=target_acc, max_steps=max_steps, device=device
    )

    # Get random baseline for S_E calculation
    model_random = PhenotypeNeuralNetwork(genotype).to(device)
    replace_linear_with_lora(model_random, rank=4)
    freeze_model_except_lora(model_random)
    steps_wr, _, _ = train_to_target_accuracy(
        model_random, x_train_b, y_train_b, x_val_b, y_val_b,
        target_acc=target_acc, max_steps=max_steps, device=device
    )

    s_e = EvaluationMetrics.sample_efficiency(steps_wr, steps_regen)
    print(f"   [CPPN DNA] S_E: {s_e:.2f} | D_KL: {kl_div:.4f} | Recon Loss: {recon_loss:.4f} | C_R: {c_r:.2f}x")

    return {
        "true_compression_ratio": c_r,
        "recon_loss": recon_loss,
        "kl_divergence": kl_div,
        "sample_efficiency": s_e,
        "steps_regen": steps_regen,
        "steps_random": steps_wr,
    }
