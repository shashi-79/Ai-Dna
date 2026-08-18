"""
Tests for CPPN Network, Substrate Coordinates, and Growth Engine.
"""

import torch
from ai_dna.dna.structure import Genotype
from ai_dna.growth.cppn import CPPNNetwork
from ai_dna.growth.coordinates import SubstrateCoordinateGenerator
from ai_dna.growth.engine import GrowthEngine


def test_cppn_forward_and_bounds():
    cppn = CPPNNetwork(in_features=5, hidden_dim=16, num_layers=2, out_features=1)
    coords = torch.randn(4, 8, 5)
    out = cppn(coords)

    assert out.shape == (4, 8, 1)
    # Output should be bounded due to scaled tanh
    assert out.abs().max() <= 1.5


def test_substrate_coordinates():
    coords_2d = SubstrateCoordinateGenerator.get_2d_weight_coordinates(
        out_features=32,
        in_features=16,
        layer_idx=1,
        num_layers=4,
    )
    assert coords_2d.shape == (32, 16, 5)
    assert coords_2d[..., 0].min() >= -1.0
    assert coords_2d[..., 0].max() <= 1.0


def test_growth_engine_phenotype_generation():
    genotype = Genotype.create_default(genotype_id="growth_test")
    genotype.dna_architecture.d_model = 32
    genotype.dna_architecture.num_layers = 2
    genotype.dna_architecture.num_experts = 2
    genotype.dna_architecture.d_expert_hidden = 64
    genotype.dna_architecture.kv_latent_dim = 8

    growth_engine = GrowthEngine()
    weights = growth_engine.grow_phenotype_weights(genotype)

    assert "text_encoder.token_emb.weight" in weights
    assert "blocks.0.attn.w_q.weight" in weights
    assert "blocks.0.attn.w_dkv.weight" in weights
    assert "blocks.0.moe.experts.0.up_proj.weight" in weights
    assert weights["blocks.0.attn.w_q.weight"].shape == (32, 32)
    assert weights["blocks.0.attn.w_dkv.weight"].shape == (8, 32)

    # Test full phenotype model generation
    model = growth_engine.grow_phenotype_model(genotype)
    assert model.d_model == 32
