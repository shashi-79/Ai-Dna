"""
Tests for SVD Instinct Filter, EWC, and Slow Clock.
"""

import torch
from ai_dna.dna.structure import Genotype
from ai_dna.encoding.svd_filter import SVDInstinctFilter
from ai_dna.encoding.ewc import EWCConsolidator
from ai_dna.encoding.slow_clock import SlowClockEncoder


def test_svd_instinct_filter():
    # Construct rank-2 matrix
    u = torch.randn(32, 2)
    v = torch.randn(2, 16)
    w = torch.matmul(u, v)

    w_k, k, e_k = SVDInstinctFilter.truncate_rank(w, rank_k=2)
    assert k == 2
    assert e_k >= 0.999  # Rank 2 should capture ~100% energy of rank-2 matrix
    assert torch.allclose(w, w_k, atol=1e-4)


def test_random_low_rank_control():
    w = torch.randn(32, 16)
    w_rand = SVDInstinctFilter.generate_random_low_rank(w, rank_k=4)

    assert w_rand.shape == w.shape
    u, s, vh, _ = SVDInstinctFilter.decompose_matrix(w_rand)
    # Singular values after rank 4 should be negligible
    assert (s[4:] < 1e-4).all()


def test_slow_clock_step():
    genotype = Genotype.create_default(genotype_id="slow_clock_test")
    genotype.dna_architecture.d_model = 16
    genotype.dna_architecture.num_layers = 2

    learned_state = {
        "blocks.0.attn.q_proj.weight": torch.randn(16, 16),
        "blocks.0.attn.k_proj.weight": torch.randn(16, 16),
    }

    slow_clock = SlowClockEncoder(rank_ratio=0.5, encoder_steps=10)
    new_genotype, summary = slow_clock.step(genotype, learned_state)

    assert new_genotype.generation == 1
    assert "mean_retained_energy" in summary
    assert "reconstruction_loss" in summary


def test_extract_cross_modal_instinct():
    vision_delta = torch.randn(64, 48)
    audio_delta = torch.randn(64, 80)
    deltas = {
        "vision_to_text": vision_delta,
        "audio_to_text": audio_delta,
    }
    bases = SVDInstinctFilter.extract_cross_modal_instinct(deltas, rank_k=8)

    assert "vision_to_text" in bases
    assert "audio_to_text" in bases
    u_v, s_v, vh_v = bases["vision_to_text"]
    assert u_v.shape == (64, 8)
    assert s_v.shape == (8,)
    assert vh_v.shape == (8, 48)
