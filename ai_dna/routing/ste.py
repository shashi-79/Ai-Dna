"""
Straight-Through Estimator (STE) for discrete gating with differentiable backward pass.
M_gate = M_hard + stop_gradient(P_gate - M_hard)
"""

import torch
import torch.nn as nn


class StraightThroughEstimator(nn.Module):
    """
    Applies hard thresholding on the forward pass while allowing continuous gradients
    to pass through to gate probabilities during backward optimization.
    """
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, p_gate: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """
        p_gate: Continuous gate probabilities in (0, 1) of shape (B, S, E_max)
        returns:
            m_gate: STE tensor (discrete on forward pass, differentiable on backward pass)
        """
        thresh = self.threshold if threshold is None else threshold
        # Binary hard mask: M_hard in {0, 1}
        m_hard = (p_gate > thresh).float()

        # Straight-Through formulation: Forward returns m_hard, Gradient dM/dP approx 1
        m_gate = p_gate + (m_hard - p_gate).detach()
        return m_gate
