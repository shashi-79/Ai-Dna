"""
SVD Instinct-Filter Mechanism.
Performs Singular Value Decomposition, computes retained singular energy E_k,
and provides both Truncated SVD reconstruction and Random Low-Rank baseline control.
"""

import math
import torch
from typing import Dict, Tuple, List, Optional


class SVDInstinctFilter:
    """
    Implements SVD structural extraction and low-rank reconstruction.
    """

    @staticmethod
    def decompose_matrix(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Decomposes a 2D weight matrix W into U, S, V^T.
        Returns:
            u: (M, K)
            s: (K,)
            vh: (K, N)
            frobenius_norm_sq: ||W||_F^2
        """
        if w.ndim != 2:
            orig_shape = w.shape
            w_2d = w.reshape(orig_shape[0], -1)
        else:
            w_2d = w

        u, s, vh = torch.linalg.svd(w_2d.float(), full_matrices=False)
        fro_sq = (s ** 2).sum().item()
        return u, s, vh, fro_sq

    @classmethod
    def truncate_rank(
        cls,
        w: torch.Tensor,
        rank_k: Optional[int] = None,
        rank_ratio: Optional[float] = None,
        energy_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Truncates weight matrix W to rank k.
        Returns:
            w_k: Truncated rank-k reconstruction (same shape as W)
            k: Number of retained singular components
            e_k: Retained singular energy fraction sum(s_1..k^2) / ||W||_F^2
        """
        orig_shape = w.shape
        w_2d = w.reshape(orig_shape[0], -1) if w.ndim != 2 else w
        u, s, vh, fro_sq = cls.decompose_matrix(w_2d)

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
        w_k_2d = torch.matmul(u_k * s_k.unsqueeze(0), vh_k)

        w_k = w_k_2d.reshape(orig_shape)
        return w_k, k, e_k

    @classmethod
    def filter_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        rank_ratio: float = 0.25,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Applies SVD instinct filtering across all 2D weight matrices in a model state_dict.
        """
        filtered_state = {}
        energies = {}
        for name, param in state_dict.items():
            if param.ndim >= 2 and ("weight" in name) and not ("norm" in name or "ln" in name):
                w_k, k, e_k = cls.truncate_rank(param, rank_ratio=rank_ratio)
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
