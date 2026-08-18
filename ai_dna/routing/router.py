"""
Generative Sparse Router with Load Balancing Loss and Operational Telemetry.
Combines Low-Rank expert representation, Straight-Through gating, and auxiliary balancing.
Implements idea.md Section 8.1: X_in = [X_meta || E_modality || (W_proj * h_t)]
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from ..dna.structure import DNARouting
from .low_rank_gate import LowRankExpertGate
from .ste import StraightThroughEstimator


class GenerativeSparseRouter(nn.Module):
    """
    DNA-controlled Sparse Generative Router (Section 8).
    Routes latent states to dynamically activated experts.
    Supports optional operational telemetry (X_meta) per Section 8.1.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        dna_routing: Optional[DNARouting] = None,
        meta_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.dna_routing = dna_routing or DNARouting()
        self.meta_dim = meta_dim

        # Optional X_meta projection (Section 8.1)
        # When meta_dim > 0, the routing input becomes [X_meta || E_modality || W_proj * h_t]
        if meta_dim > 0:
            self.meta_proj = nn.Linear(meta_dim, d_model)
        else:
            self.meta_proj = None

        self.gate = LowRankExpertGate(
            d_model=d_model,
            num_experts=num_experts,
            rank=self.dna_routing.rank,
        )
        self.ste = StraightThroughEstimator(threshold=self.dna_routing.threshold)
        self.load_balance_weight = self.dna_routing.load_balance_weight

    def forward(
        self,
        h: torch.Tensor,
        modality_emb: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        x_meta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: Latent token representation (B, S, D_model)
        modality_emb: Optional modality embedding (B, S, D_model)
        x_meta: Optional operational telemetry tensor (B, S, meta_dim)
                Contains runtime metrics like compute cost, memory utilization, etc.
        Returns:
            p_gate: Continuous gating probabilities (B, S, E_max)
            m_gate: Straight-Through discrete selection mask (B, S, E_max)
            aux_loss: Expert load balancing loss L_bal
        """
        # Build routing input: X_in = [X_meta || E_modality || (W_proj * h_t)] (Section 8.1)
        h_routed = h

        if modality_emb is not None:
            h_routed = h_routed + modality_emb

        if x_meta is not None and self.meta_proj is not None:
            meta_features = self.meta_proj(x_meta)  # (B, S, D_model)
            h_routed = h_routed + meta_features

        _, p_gate = self.gate(h_routed)
        m_gate = self.ste(p_gate, threshold=threshold)

        # Expert Balancing Loss: L_bal = E_max * sum_{e=1}^{E_max} (P_e * f_e) (Section 12.1)
        # P_e: mean routing probability across batch and sequence
        # f_e: actual dispatch fraction
        batch_size, seq_len, num_exp = p_gate.shape
        p_e = p_gate.mean(dim=(0, 1))  # (E_max,)
        f_e = (m_gate > 0.0).float().mean(dim=(0, 1))  # (E_max,)

        l_bal = self.num_experts * (p_e * f_e).sum()
        aux_loss = self.load_balance_weight * l_bal

        return p_gate, m_gate, aux_loss
