"""
Compositional Pattern Producing Network (CPPN) for Genotype-to-Phenotype weight generation.
Maps geometric coordinates (x1, y1, x2, y2, d, r) -> connection weight w.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CPPNActivation(nn.Module):
    """
    Multi-functional activation function combining Sinusoidal, Gaussian, Tanh, Linear, and ReLU
    to generate rich periodic, symmetric, and localized weight patterns.
    """
    def __init__(self, chunk_size: int = 8):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split channels or apply mixed non-linearities
        # 1st quarter: sin(pi * x), 2nd: exp(-x^2), 3rd: tanh(x), 4th: relu(x)
        c1 = torch.sin(math.pi * x[..., :self.chunk_size])
        c2 = torch.exp(-torch.clamp(x[..., self.chunk_size:2*self.chunk_size]**2, max=20.0))
        c3 = torch.tanh(x[..., 2*self.chunk_size:3*self.chunk_size])
        c4 = F.silu(x[..., 3*self.chunk_size:])
        return torch.cat([c1, c2, c3, c4], dim=-1)


class CPPNNetwork(nn.Module):
    """
    CPPN Neural Network acting as the developmental kernel G_D.
    Input: coordinate tensor (..., in_features)
    Output: scalar weight (..., 1) or vector representation
    Supports optional Random Fourier Feature (RFF) / SIREN coordinate projections.
    """
    def __init__(
        self,
        in_features: int = 32,
        hidden_dim: int = 32,
        num_layers: int = 3,
        out_features: int = 1,
        use_rff: bool = False,
        rff_features: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_features = out_features
        self.use_rff = use_rff
        self.rff_features = rff_features

        actual_in_dim = (rff_features * 2) if use_rff else in_features

        layers = []
        # Input projection
        layers.append(nn.Linear(actual_in_dim, hidden_dim))
        layers.append(CPPNActivation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(CPPNActivation())

        # Output projection (scaled tanh for bounded stable initialization)
        self.backbone = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hidden_dim, out_features)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: Tensor of shape (..., in_features)
        returns: Tensor of shape (..., out_features)
        """
        if self.use_rff:
            from .coordinates import SubstrateCoordinateGenerator
            coords = SubstrateCoordinateGenerator.apply_rff_embedding(
                coords, num_fourier_feats=self.rff_features
            )

        features = self.backbone(coords)
        out = self.out_proj(features)
        # Scaled tanh for zero-centered stable weight ranges
        return torch.tanh(out) * 1.5

    def get_parameter_dict(self) -> Dict[str, torch.Tensor]:
        """Exports state dict as a dictionary of detached tensors."""
        return {k: v.clone().detach() for k, v in self.state_dict().items()}

    def load_parameter_dict(self, param_dict: Dict[str, torch.Tensor]):
        """Loads state from parameter dictionary."""
        self.load_state_dict(param_dict)
