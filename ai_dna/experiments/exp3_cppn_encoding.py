"""
Experiment 3: CPPN Encoding and Growth Validation.
Validates W_k -> CPPN -> D -> Growth -> W_D.
Evaluates:
- True Compression Ratio C_R = |theta_model| / (|D| + |theta_G| + |theta_S|)
- Normalized Frobenius Reconstruction Error
- Behavioral Divergence D_KL
- Sample efficiency S_E on unseen task T_B.
"""

import torch
from typing import Dict, Any
from .exp1_svd_hypothesis import (
    generate_synthetic_task,
    train_to_target_accuracy,
)
from ..dna.structure import Genotype
from ..growth.engine import GrowthEngine
from ..models.phenotype import PhenotypeNeuralNetwork
from ..encoding.svd_filter import SVDInstinctFilter
from ..encoding.cppn_encoder import InverseCPPNEncoder
from ..training.metrics import EvaluationMetrics


def run_experiment_3(quick: bool = False, device_str: str = "cpu") -> Dict[str, Any]:
    """
    Executes Experiment 3 testing CPPN genetic encoding and regeneration.
    """
    device = torch.device(device_str)
    print("=== [Experiment 3] CPPN Encoding and Growth Engine ===")

    # 1. Setup Genotype and Datasets
    genotype = Genotype.create_default(genotype_id="exp3_root")
    genotype.dna_architecture.vocab_size = 100
    genotype.dna_architecture.d_model = 32
    genotype.dna_architecture.num_layers = 2
    genotype.dna_architecture.num_experts = 2
    genotype.dna_architecture.d_expert_hidden = 64
    genotype.dna_instinct.cppn_hidden_dim = 24
    genotype.dna_instinct.cppn_layers = 2

    x_train_a, y_train_a = generate_synthetic_task("task_A", num_samples=300, seed=42)
    x_val_a, y_val_a = generate_synthetic_task("task_A", num_samples=100, seed=43)
    x_train_b, y_train_b = generate_synthetic_task("task_B", num_samples=300, seed=100)
    x_val_b, y_val_b = generate_synthetic_task("task_B", num_samples=100, seed=101)

    max_steps = 30 if quick else 60
    target_acc = 0.40 if quick else 0.55

    # 2. Train on Task A -> W*
    print("-> Training Phenotype on Task A...")
    model_orig = PhenotypeNeuralNetwork(genotype).to(device)
    train_to_target_accuracy(model_orig, x_train_a, y_train_a, x_val_a, y_val_a, target_acc=target_acc, max_steps=max_steps, device=device)
    orig_state = {k: v.clone() for k, v in model_orig.state_dict().items()}

    # 3. SVD Filtering W* -> W_k
    svd_state, _ = SVDInstinctFilter.filter_state_dict(orig_state, rank_ratio=0.25)

    # 4. Inverse CPPN Encoding W_k -> D_1
    print("-> Distilling extracted instinct into compact Genotype DNA...")
    encoder = InverseCPPNEncoder(
        learning_rate=1e-2,
        max_steps=50 if quick else 120,
        device=device,
    )
    genotype_d1, recon_loss, _ = encoder.encode_genotype(genotype, svd_state)

    # 5. Growth Engine Regeneration D_1 -> W_D
    print("-> Regenerating Phenotype from Genotype via Growth Engine...")
    growth_engine = GrowthEngine(device=device)
    grown_weights = growth_engine.grow_phenotype_weights(genotype_d1)

    model_grown = PhenotypeNeuralNetwork(genotype_d1).to(device)
    # Bind grown weights into executable phenotype
    model_state = model_grown.state_dict()
    for k, v in grown_weights.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    model_grown.load_state_dict(model_state)

    # 6. Measure Behavioral Divergence D_KL
    with torch.no_grad():
        h_orig, _, _, _ = model_orig(x_val_a[:32].to(device), modality="text")
        logits_orig = model_orig.ar_head(h_orig)
        h_grown, _, _, _ = model_grown(x_val_a[:32].to(device), modality="text")
        logits_grown = model_grown.ar_head(h_grown)
        kl_div = EvaluationMetrics.behavioral_divergence(logits_orig, logits_grown)

    # 7. Compute True Compression Ratio C_R
    model_param_count = sum(p.numel() for p in model_orig.parameters())
    dna_param_count = genotype_d1.total_parameters()
    c_r = EvaluationMetrics.true_compression_ratio(
        model_parameters=model_param_count,
        dna_parameters=dna_param_count,
    )

    # 8. Evaluate transfer on unseen Task B
    model_wr = PhenotypeNeuralNetwork(genotype).to(device)
    steps_wr, _, _ = train_to_target_accuracy(model_wr, x_train_b, y_train_b, x_val_b, y_val_b, target_acc=target_acc, max_steps=max_steps, device=device)
    steps_grown, acc_grown, _ = train_to_target_accuracy(model_grown, x_train_b, y_train_b, x_val_b, y_val_b, target_acc=target_acc, max_steps=max_steps, device=device)
    se_grown = EvaluationMetrics.sample_efficiency(steps_wr, steps_grown)

    print(f"   Reconstruction Error L_recon: {recon_loss:.4f}")
    print(f"   Behavioral Divergence D_KL:   {kl_div:.4f}")
    print(f"   True Compression Ratio C_R:   {c_r:.1f}x (|Model|={model_param_count} vs |DNA|={dna_param_count})")
    print(f"   Downstream Sample Efficiency: S_E={se_grown:.2f} (Steps: {steps_grown:2d} vs Random: {steps_wr:2d})")

    return {
        "reconstruction_loss": recon_loss,
        "behavioral_divergence_kl": kl_div,
        "true_compression_ratio": c_r,
        "sample_efficiency": se_grown,
        "model_parameters": model_param_count,
        "dna_parameters": dna_param_count,
    }
