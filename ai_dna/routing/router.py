"""
Generative Sparse Router with Top-K Sparsely-Gated Noisy Routing and Load Balancing.
Replaces legacy STE and Low-Rank hard gating.
Implements idea.md Section 8.2 & 8.3.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from ..dna.structure import DNARouting
from .topk_gate import TopKNoisyGate


class GenerativeSparseRouter(nn.Module):
    """
    DNA-controlled Sparse Generative Router (Section 8).
    Routes latent states to dynamically activated Top-K experts with exploration noise and CV^2 load balancing.
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
        if meta_dim > 0:
            self.meta_proj = nn.Linear(meta_dim, d_model)
        else:
            self.meta_proj = None

        # Top-K Noisy Gating module
        self.gate = TopKNoisyGate(
            d_model=d_model,
            num_experts=num_experts,
            top_k=self.dna_routing.top_k_experts,
            noise_std=self.dna_routing.routing_noise_std,
        )
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
        Returns:
            p_gate: Normalized gating weights across experts (B, S, E_max)
            m_gate: Binary activation mask (B, S, E_max)
            aux_loss: Expert load balancing loss L_bal
        """
        h_routed = h

        if modality_emb is not None:
            h_routed = h_routed + modality_emb

        if x_meta is not None and self.meta_proj is not None:
            meta_features = self.meta_proj(x_meta)
            h_routed = h_routed + meta_features

        p_gate, top_indices = self.gate(h_routed)
        m_gate = (p_gate > 0.0).float()

        # Compute auxiliary load balancing loss (Section 8.3 / Section 12.1)
        aux_loss = self.load_balance_weight * self.gate.get_load_balancing_loss()

        return p_gate, m_gate, aux_loss
