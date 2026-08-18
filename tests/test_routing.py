"""
Tests for Sparse Low-Rank Router and Straight-Through Estimator.
"""

import torch
from ai_dna.routing.low_rank_gate import LowRankExpertGate
from ai_dna.routing.ste import StraightThroughEstimator
from ai_dna.routing.router import GenerativeSparseRouter


def test_low_rank_gate():
    gate = LowRankExpertGate(d_model=32, num_experts=4, rank=4)
    h = torch.randn(2, 8, 32)
    z, p_gate = gate(h)

    assert z.shape == (2, 8, 4)
    assert p_gate.shape == (2, 8, 4)
    assert (p_gate >= 0.0).all() and (p_gate <= 1.0).all()


def test_straight_through_estimator_gradients():
    ste = StraightThroughEstimator(threshold=0.5)
    p_gate = torch.tensor([[[0.2, 0.7, 0.9]]], requires_grad=True)

    m_gate = ste(p_gate)
    # Forward check: 0.2 -> 0.0, 0.7 -> 1.0, 0.9 -> 1.0
    assert torch.allclose(m_gate, torch.tensor([[[0.0, 1.0, 1.0]]]))

    # Backward gradient check
    loss = (m_gate * 2.0).sum()
    loss.backward()

    # Straight-through gradient dm/dp approx 1.0, so dLoss/dp should be 2.0
    assert torch.allclose(p_gate.grad, torch.tensor([[[2.0, 2.0, 2.0]]]))


def test_generative_sparse_router():
    router = GenerativeSparseRouter(d_model=32, num_experts=4)
    h = torch.randn(2, 16, 32)
    p_gate, m_gate, aux_loss = router(h)

    assert p_gate.shape == (2, 16, 4)
    assert m_gate.shape == (2, 16, 4)
    assert aux_loss.item() >= 0.0
