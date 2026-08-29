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
    def get_matrix_idx_from_name(name: str) -> int:
        """Extracts a unique matrix index from layer name to avoid coordinate aliasing."""
        is_lora_b = "lora_B" in name
        if "w_q" in name:
            base_idx = 0
        elif "w_dkv" in name:
            base_idx = 1
        elif "w_uk" in name:
            base_idx = 2
        elif "w_uv" in name:
            base_idx = 3
        elif "o_proj" in name:
            base_idx = 4
        elif "up_proj" in name:
            base_idx = 5
        elif "down_proj" in name:
            base_idx = 6
        elif "gate" in name or "router" in name:
            base_idx = 7
        else:
            base_idx = 8
        
        return base_idx * 2 + (1 if is_lora_b else 0)

    @staticmethod
    def get_2d_weight_coordinates(
        out_features: int,
        in_features: int,
        layer_idx: int = 0,
        num_layers: int = 4,
        expert_idx: int = 0,
        num_experts: int = 1,
        matrix_idx: int = 0,
        device: Optional[torch.device] = None,
        coord_dim: int = 32,
    ) -> torch.Tensor:
        """
        Generates coordinate grid for a 2D weight matrix W of shape (out_features, in_features).
        Returns: Tensor of shape (out_features, in_features, coord_dim)
        Coordinates: (x1_src, y1_src, x2_dst, y2_dst, norm_layer_idx, [norm_expert_idx, norm_matrix_idx])
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalized source coordinates in [-1, 1]
        x1 = torch.linspace(-1.0, 1.0, in_features, device=device)
        
        # Normalized target coordinates in [-1, 1]
        x2 = torch.linspace(-1.0, 1.0, out_features, device=device)
        
        # Meshgrid of source and target locations
        x2_grid, x1_grid = torch.meshgrid(x2, x1, indexing="ij")
        
        # Layer, expert, and matrix normalized position in [-1, 1]
        norm_layer = (2.0 * layer_idx / max(1, num_layers - 1)) - 1.0 if num_layers > 1 else 0.0
        norm_expert = (2.0 * expert_idx / max(1, num_experts - 1)) - 1.0 if num_experts > 1 else 0.0
        norm_matrix = (2.0 * matrix_idx / 15.0) - 1.0
        
        layer_grid = torch.full_like(x1_grid, norm_layer)
        expert_grid = torch.full_like(x1_grid, norm_expert)
        matrix_grid = torch.full_like(x1_grid, norm_matrix)
        zeros = torch.zeros_like(x1_grid)
        ones = torch.ones_like(x1_grid)
        
        if coord_dim == 32:
            # 32D Universal Manifold: 16D Source Address + 16D Target Address
            # Embed matrix_grid into y1 (position 1) and y2 (position 17)
            y1_grid = matrix_grid
            y2_grid = matrix_grid
            
            src_16d = [
                x1_grid, y1_grid, layer_grid,
                zeros, zeros,
                ones, zeros, zeros, zeros,
                expert_grid, expert_grid, torch.full_like(x1_grid, 0.5),
                ones, zeros, zeros,
                torch.full_like(x1_grid, 0.1)
            ]
            tgt_16d = [
                x2_grid, y2_grid, layer_grid,
                zeros, zeros,
                ones, zeros, zeros, zeros,
                expert_grid, expert_grid, torch.full_like(x1_grid, 0.5),
                ones, zeros, zeros,
                torch.full_like(x1_grid, 0.1)
            ]
            coords = torch.stack(src_16d + tgt_16d, dim=-1)
        elif coord_dim >= 6:
            y1_grid = matrix_grid
            y2_grid = zeros
            channels = [x1_grid, y1_grid, x2_grid, y2_grid, layer_grid, expert_grid]
            while len(channels) < coord_dim:
                channels.append(zeros)
            coords = torch.stack(channels[:coord_dim], dim=-1)
        else:
            y1_grid = matrix_grid
            y2_grid = zeros
            channels = [x1_grid, y1_grid, x2_grid, y2_grid, layer_grid]
            coords = torch.stack(channels[:coord_dim], dim=-1)
            
        return coords

    @staticmethod
    def get_1d_bias_coordinates(
        features: int,
        layer_idx: int = 0,
        num_layers: int = 4,
        device: Optional[torch.device] = None,
        coord_dim: int = 32,
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
        
        if coord_dim == 32:
            channels = [zeros, zeros, layer_col, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros]
            channels += [x, zeros, layer_col, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros]
            coords = torch.stack(channels, dim=-1)
        elif coord_dim >= 6:
            channels = [zeros, zeros, x, zeros, layer_col, zeros]
            while len(channels) < coord_dim:
                channels.append(zeros)
            coords = torch.stack(channels[:coord_dim], dim=-1)
        else:
            coords = torch.stack([zeros, zeros, x, zeros, layer_col][:coord_dim], dim=-1)
        return coords

    @staticmethod
    def apply_rff_embedding(
        coords: torch.Tensor,
        num_fourier_feats: int = 16,
        sigma: float = 1.0,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Random Fourier Features (RFF) / SIREN high-frequency coordinate embedding.
        Projects raw coordinates c into gamma(c) = [cos(2*pi*B*c), sin(2*pi*B*c)].
        Overcomes spectral bias of standard MLPs for high-frequency weight boundaries.
        """
        in_dim = coords.shape[-1]
        generator = torch.Generator().manual_seed(seed)
        B_mat = torch.randn(in_dim, num_fourier_feats, generator=generator, device=coords.device) * sigma
        
        proj = 2.0 * 3.141592653589793 * torch.matmul(coords, B_mat)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    @staticmethod
    def compute_manifold_isomorphism_order(dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Computes 1D Graph Laplacian spectral embedding order for topologically
        aligning homologous neurons across diverse layer sizes.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Continuous spectral eigenmap coordinates in [-1, 1]
        t = torch.linspace(0.0, 3.141592653589793, dim, device=device)
        return -torch.cos(t)

