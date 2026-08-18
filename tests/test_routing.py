"""
Tests for Top-K Sparsely-Gated Router and Noisy Load Balancing.
"""

import torch
from ai_dna.routing.topk_gate import TopKNoisyGate
from ai_dna.routing.router import GenerativeSparseRouter
from ai_dna.dna.structure import DNARouting


def test_topk_noisy_gate():
    gate = TopKNoisyGate(d_model=32, num_experts=4, top_k=2, noise_std=1.0)
    h = torch.randn(2, 8, 32)
    gates, indices = gate(h)

    assert gates.shape == (2, 8, 4)
    assert indices.shape == (2, 8, 2)
    # Exactly top_k non-zero per token
    non_zeros = (gates > 0).sum(dim=-1)
    assert (non_zeros <= 2).all()
    # Gates sum to ~1.0 for active tokens
    gate_sums = gates.sum(dim=-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)


def test_topk_gate_backward_gradients():
    gate = TopKNoisyGate(d_model=32, num_experts=4, top_k=2, noise_std=0.0)
    h = torch.randn(2, 4, 32, requires_grad=True)
    gates, _ = gate(h)

    loss = gates.sum()
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


def test_generative_sparse_router():
    dna_routing = DNARouting(top_k_experts=2, routing_noise_std=0.5)
    router = GenerativeSparseRouter(d_model=32, num_experts=4, dna_routing=dna_routing)
    h = torch.randn(2, 16, 32)
    p_gate, m_gate, aux_loss = router(h)

    assert p_gate.shape == (2, 16, 4)
    assert m_gate.shape == (2, 16, 4)
    assert aux_loss.item() >= 0.0
