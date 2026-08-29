"""
Unit tests for LoRA adapters and the CPPN-based LoRA genotypic encoding pipeline.
"""

import torch
import torch.nn as nn
from ai_dna.dna.structure import Genotype
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.models.lora import replace_linear_with_lora, freeze_model_except_lora, extract_lora_parameters
from ai_dna.growth.engine import GrowthEngine
from ai_dna.encoding.slow_clock import SlowClockEncoder


def test_lora_injection_and_freezing():
    # Create a small genotype
    d_0 = Genotype.create_default(genotype_id="lora_test")
    d_0.dna_architecture.vocab_size = 20
    d_0.dna_architecture.d_model = 16
    d_0.dna_architecture.num_layers = 1
    d_0.dna_architecture.num_experts = 2
    d_0.dna_architecture.d_expert_hidden = 32

    # Instantiate model
    model = PhenotypeNeuralNetwork(d_0)
    
    # Check that there are no LoRA layers initially
    assert not any(hasattr(module, "lora_A") for module in model.modules())

    # Replace with LoRA
    adapted = replace_linear_with_lora(model, rank=4, alpha=8.0)
    assert len(adapted) > 0
    assert any(hasattr(module, "lora_A") for module in model.modules())

    # Freeze model parameters except LoRA
    freeze_model_except_lora(model, freeze_modalities=True)
    
    # Check parameter grad status
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad


def test_lora_cppn_reconstruction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup small genotype with CPPN config
    d_0 = Genotype.create_default(genotype_id="lora_cppn_test")
    d_0.dna_architecture.vocab_size = 20
    d_0.dna_architecture.d_model = 16
    d_0.dna_architecture.num_layers = 1
    d_0.dna_architecture.num_experts = 2
    d_0.dna_architecture.d_expert_hidden = 32
    d_0.dna_architecture.lora_rank = 4

    # Optimize a dummy base CPPN first to fill baseline parameters
    from ai_dna.growth.cppn import CPPNNetwork
    base_cppn = CPPNNetwork(in_features=32, hidden_dim=32, num_layers=3, out_features=1)
    d_0.dna_instinct.genetic_parameters = base_cppn.get_parameter_dict()

    # 2. Grow model
    growth_engine = GrowthEngine(device=device)
    model = growth_engine.grow_phenotype_model(d_0)

    # Check model has LoRALinear modules
    assert any("lora_A" in k for k in model.state_dict().keys())

    # 3. Create dummy trained LoRA state updates
    dummy_lora_updates = {}
    for k, v in model.state_dict().items():
        if "lora_" in k:
            dummy_lora_updates[k] = torch.randn_like(v) * 0.1

    # 4. Run slow clock step using the LoRA + CPPN pathway
    slow_clock = SlowClockEncoder(rank_ratio=0.5, encoder_steps=5, device=device)
    d_1, summary = slow_clock.step(
        d_0,
        dummy_lora_updates,
        protect_ancestral=False,
        phenotype_model=model,
        growth_engine=growth_engine,
    )

    assert d_1.generation == 1
    # Check that both base and adapter CPPN parameters exist in genetic_parameters
    assert any(k.startswith("adapter.") for k in d_1.dna_instinct.genetic_parameters.keys())
    assert any(not k.startswith("adapter.") for k in d_1.dna_instinct.genetic_parameters.keys())
    
    # 5. Regrow from d_1 and verify LoRA parameters are populated
    model_regen = growth_engine.grow_phenotype_model(d_1)
    
    # Verify that LoRA parameters exist in regenerated model
    lora_params = extract_lora_parameters(model_regen)
    assert len(lora_params) > 0
    non_zero_params = [name for name, param in lora_params.items() if (param != 0).any()]
    assert len(non_zero_params) > 0, "No non-zero LoRA parameters found in regenerated model"
