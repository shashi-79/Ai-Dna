"""
Tests for EWC and Slow Clock.
"""

import torch
from ai_dna.dna.structure import Genotype
from ai_dna.encoding.ewc import EWCConsolidator
from ai_dna.encoding.slow_clock import SlowClockEncoder


def test_slow_clock_step():
    genotype = Genotype.create_default(genotype_id="slow_clock_test")
    genotype.dna_architecture.d_model = 16
    genotype.dna_architecture.num_layers = 2
    genotype.dna_architecture.num_experts = 2
    genotype.dna_architecture.d_expert_hidden = 32

    learned_state = {
        "blocks.0.attn.w_q.weight": torch.randn(16, 16),
        "blocks.0.attn.w_dkv.weight": torch.randn(16, 16),
    }

    slow_clock = SlowClockEncoder(rank_ratio=0.5, encoder_steps=5)
    new_genotype, summary = slow_clock.step(genotype, learned_state, protect_ancestral=False)

    assert new_genotype.generation == 1
    assert "mean_retained_energy" in summary
    assert "reconstruction_loss" in summary
