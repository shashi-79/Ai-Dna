"""
Low-Rank Expert Representation for Sparse Routing.
Computes per-expert scalar: z_{b,s,e} = \\sum_{k=1}^r A_{b,s,e,k} * B_{b,s,e,k}
and gating probability: P_{gate} = \\sigma(z).
"""

import torch
import torch.nn as nn
from typing import Tuple


class LowRankExpertGate(nn.Module):
    """
    Factorizes the gating representation into low-rank tensors A and B of rank r.
    Avoids O(E^2) parameter expansion and produces smooth dynamic gating.
    """
    def __init__(self, d_model: int, num_experts: int, rank: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.rank = rank

        # Linear projections from d_model to (num_experts * rank)
        self.proj_a = nn.Linear(d_model, num_experts * rank, bias=False)
        self.proj_b = nn.Linear(d_model, num_experts * rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: Input hidden state of shape (B, S, D_model)
        Returns:
            z: Logit tensor of shape (B, S, E_max)
            p_gate: Continuous gating probability in (0, 1) of shape (B, S, E_max)
        """
        batch_size, seq_len, _ = h.shape

        # A, B: (B, S, E_max * r) -> reshape to (B, S, E_max, r)
        a = self.proj_a(h).view(batch_size, seq_len, self.num_experts, self.rank)
        b = self.proj_b(h).view(batch_size, seq_len, self.num_experts, self.rank)

        # Per-expert scalar inner product: z_{b,s,e} = \sum_{k=1}^r A_{b,s,e,k} * B_{b,s,e,k}
        z = (a * b).sum(dim=-1) + self.bias  # Shape: (B, S, E_max)

        p_gate = torch.sigmoid(z)
        return z, p_gate
