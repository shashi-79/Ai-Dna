"""
Substrate Coordinate Generator for Phenotype parameter locations.
Generates geometric coordinates (x1, y1, x2, y2, layer_idx, expert_idx) in normalized [-1, 1] bounds.
"""

import torch
from typing import Tuple, Optional


class SubstrateCoordinateGenerator:
    """
    Generates normalized spatial coordinates for weight matrices and MoE expert slices.
    """

    @staticmethod
    def get_2d_weight_coordinates(
        out_features: int,
        in_features: int,
        layer_idx: int = 0,
        num_layers: int = 4,
        expert_idx: int = 0,
        num_experts: int = 1,
        device: Optional[torch.device] = None,
        coord_dim: int = 5,
    ) -> torch.Tensor:
        """
        Generates coordinate grid for a 2D weight matrix W of shape (out_features, in_features).
        Returns: Tensor of shape (out_features, in_features, coord_dim)
        Coordinates: (x1_src, y1_src, x2_dst, y2_dst, norm_layer_idx, [norm_expert_idx])
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalized source coordinates in [-1, 1]
        x1 = torch.linspace(-1.0, 1.0, in_features, device=device)
        
        # Normalized target coordinates in [-1, 1]
        x2 = torch.linspace(-1.0, 1.0, out_features, device=device)
        
        # Meshgrid of source and target locations
        x2_grid, x1_grid = torch.meshgrid(x2, x1, indexing="ij")
        
        # Layer & expert normalized position in [-1, 1]
        norm_layer = (2.0 * layer_idx / max(1, num_layers - 1)) - 1.0 if num_layers > 1 else 0.0
        norm_expert = (2.0 * expert_idx / max(1, num_experts - 1)) - 1.0 if num_experts > 1 else 0.0
        
        layer_grid = torch.full_like(x1_grid, norm_layer)
        
        if coord_dim >= 6:
            y1_grid = torch.zeros_like(x1_grid)
            y2_grid = torch.zeros_like(x2_grid)
            expert_grid = torch.full_like(x1_grid, norm_expert)
            coords = torch.stack([x1_grid, y1_grid, x2_grid, y2_grid, layer_grid, expert_grid], dim=-1)
        else:
            # If coord_dim is 5, encode expert index in the unused y1 dimension
            # to ensure different experts generate different weights.
            y1_grid = torch.full_like(x1_grid, norm_expert)
            y2_grid = torch.zeros_like(x2_grid)
            coords = torch.stack([x1_grid, y1_grid, x2_grid, y2_grid, layer_grid], dim=-1)
            
        return coords

    @staticmethod
    def get_1d_bias_coordinates(
        features: int,
        layer_idx: int = 0,
        num_layers: int = 4,
        device: Optional[torch.device] = None,
        coord_dim: int = 5,
    ) -> torch.Tensor:
        """
        Generates coordinate vector for a 1D bias vector of shape (features,).
        Returns: Tensor of shape (features, coord_dim)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = torch.linspace(-1.0, 1.0, features, device=device)
        norm_layer = (2.0 * layer_idx / max(1, num_layers - 1)) - 1.0 if num_layers > 1 else 0.0
        
        layer_col = torch.full_like(x, norm_layer)
        zeros = torch.zeros_like(x)
        
        if coord_dim >= 6:
            coords = torch.stack([zeros, zeros, x, zeros, layer_col, zeros], dim=-1)
        else:
            coords = torch.stack([zeros, zeros, x, zeros, layer_col], dim=-1)
        return coords
