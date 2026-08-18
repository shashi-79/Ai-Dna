"""
SVD Instinct-Filter Mechanism with Walsh-Hadamard Rotation Pre-processing and MLA-Aware Targeting.
Performs Singular Value Decomposition on rotationally smoothed weights, computes retained singular energy E_k,
and provides both Truncated SVD reconstruction and Random Low-Rank baseline control.
Implements idea.md Section 14 & 14.5.
"""

import math
import torch
from typing import Dict, Tuple, List, Optional, Union


class SVDInstinctFilter:
    """
    Implements SVD structural extraction with orthogonal rotation pre-processing (TurboQuant-style)
    and low-rank reconstruction for MLA down-projections and dense linear weights.
    """

    @staticmethod
    def _get_orthogonal_rotation(dim: int, device: torch.device) -> torch.Tensor:
        """Generates random orthogonal rotation matrix Pi for outlier smoothing."""
        rand_mat = torch.randn(dim, dim, device=device)
        q, r = torch.linalg.qr(rand_mat)
        d = torch.diag(r)
        ph = d.sign()
        q *= ph
        return q

    @classmethod
    def decompose_matrix(
        cls,
        w: torch.Tensor,
        use_rotation: bool = True,
        return_rotation: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]]:
        """
        Decomposes a 2D weight matrix W into U, S, V^T after optional orthogonal rotation Pi * W.
        Returns 4-tuple (u, s, vh, fro_sq) by default, or 5-tuple (u, s, vh, fro_sq, pi_rot) if return_rotation=True.
        """
        if w.ndim != 2:
            orig_shape = w.shape
            w_2d = w.reshape(orig_shape[0], -1)
        else:
            w_2d = w

        M, N = w_2d.shape
        fro_sq = (w_2d.float() ** 2).sum().item()

        # Apply Walsh-Hadamard / Orthogonal rotation pre-processing to eliminate outlier peaks
        pi_rot = None
        if use_rotation and M > 1:
            pi_rot = cls._get_orthogonal_rotation(M, w_2d.device)
            w_rotated = torch.matmul(pi_rot, w_2d.float())
        else:
            w_rotated = w_2d.float()

        u, s, vh = torch.linalg.svd(w_rotated, full_matrices=False)
        if return_rotation:
            return u, s, vh, fro_sq, pi_rot
        return u, s, vh, fro_sq

    @classmethod
    def truncate_rank(
        cls,
        w: torch.Tensor,
        rank_k: Optional[int] = None,
        rank_ratio: Optional[float] = None,
        energy_threshold: Optional[float] = None,
        use_rotation: bool = True,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Truncates weight matrix W to rank k using rotation-preprocessed SVD.
        Returns:
            w_k: Truncated rank-k reconstruction (same shape as W)
            k: Number of retained singular components
            e_k: Retained singular energy fraction sum(s_1..k^2) / ||W||_F^2
        """
        orig_shape = w.shape
        w_2d = w.reshape(orig_shape[0], -1) if w.ndim != 2 else w
        u, s, vh, fro_sq, pi_rot = cls.decompose_matrix(w_2d, use_rotation=use_rotation, return_rotation=True)

        max_rank = s.shape[0]
        if fro_sq < 1e-12:
            return w.clone(), max_rank, 1.0

        if rank_k is not None:
            k = min(max(1, rank_k), max_rank)
        elif rank_ratio is not None:
            k = min(max(1, math.ceil(max_rank * rank_ratio)), max_rank)
        elif energy_threshold is not None:
            cum_energy = torch.cumsum(s ** 2, dim=0) / fro_sq
            k_indices = torch.nonzero(cum_energy >= energy_threshold)
            k = k_indices[0].item() + 1 if k_indices.numel() > 0 else max_rank
        else:
            k = max_rank

        # Compute retained energy E_k = sum_{i=1}^k (sigma_i^2) / ||W||_F^2
        retained_sq = (s[:k] ** 2).sum().item()
        e_k = retained_sq / fro_sq

        # Reconstruct W_k = U_k * diag(Sigma_k) * V_k^T
        u_k = u[:, :k]
        s_k = s[:k]
        vh_k = vh[:k, :]
        w_k_rotated = torch.matmul(u_k * s_k.unsqueeze(0), vh_k)

        # Invert rotation Pi^T * W_rotated
        if pi_rot is not None:
            w_k_2d = torch.matmul(pi_rot.t(), w_k_rotated)
        else:
            w_k_2d = w_k_rotated

        w_k = w_k_2d.reshape(orig_shape)
        return w_k, k, e_k

    @classmethod
    def filter_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        rank_ratio: float = 0.25,
        target_mla_only: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Applies SVD instinct filtering across weights in a model state_dict.
        If target_mla_only=True, prioritizes w_dkv MLA down-projections and expert weights.
        """
        filtered_state = {}
        energies = {}
        for name, param in state_dict.items():
            is_2d_weight = param.ndim >= 2 and ("weight" in name) and not ("norm" in name or "ln" in name)

            should_filter = is_2d_weight
            if target_mla_only:
                should_filter = is_2d_weight and ("w_dkv" in name or "up_proj" in name or "down_proj" in name)

            if should_filter:
                w_k, k, e_k = cls.truncate_rank(param, rank_ratio=rank_ratio, use_rotation=True)
                filtered_state[name] = w_k
                energies[name] = e_k
            else:
                filtered_state[name] = param.clone()

        return filtered_state, energies

    @classmethod
    def generate_random_low_rank(
        cls,
        w_ref: torch.Tensor,
        rank_k: int,
    ) -> torch.Tensor:
        """
        Baseline 4 Control: Generates a random matrix with IDENTICAL shape and rank k,
        scaled to match the Frobenius energy of w_ref.
        """
        orig_shape = w_ref.shape
        w_2d = w_ref.reshape(orig_shape[0], -1) if w_ref.ndim != 2 else w_ref
        m, n = w_2d.shape
        k = min(max(1, rank_k), min(m, n))

        # Generate random low-rank factors: A in R^{M x K}, B in R^{K x N}
        a = torch.randn(m, k, device=w_ref.device)
        b = torch.randn(k, n, device=w_ref.device)
        w_rand = torch.matmul(a, b)

        # Scale Frobenius norm to match reference
        ref_norm = torch.linalg.norm(w_2d)
        rand_norm = torch.linalg.norm(w_rand)
        if rand_norm > 1e-12:
            w_rand = w_rand * (ref_norm / rand_norm)

        return w_rand.reshape(orig_shape)
