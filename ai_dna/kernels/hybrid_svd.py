"""
Hybrid cuSOLVER SVD Engine with Canonical Sign Stabilization.
Combines exact host-side cuSOLVER matrix decomposition with the Bro & Kiers
sign convention to eliminate coordinate inversion glitches when coupling SVD with CPPNs.
Production-grade implementation with numerical safeguards, adaptive rank, and min_rank guarantees.
"""

import math
import torch
from typing import Tuple, Optional, Dict, Any


def stabilize_svd_signs(U: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the Bro & Kiers (2008) Canonical Sign Convention to singular vectors.
    Forces the element with the largest absolute magnitude in each column of U to be positive.
    Synchronously scales V to preserve exact matrix equivalence: U_fixed @ S @ V_fixed.T == U @ S @ V.T.
    
    Eliminates random sign flips (+/- 1) across runs, guaranteeing deterministic spatial
    coordinates for continuous CPPN functional generators.
    """
    if U.dim() != 2 or V.dim() != 2:
        return U, V

    # Find the row index of the maximum absolute value in each column of U
    max_abs_indices = torch.argmax(torch.abs(U), dim=0)
    
    # Extract actual signs at those peak locations
    col_indices = torch.arange(U.shape[1], device=U.device)
    peak_values = U[max_abs_indices, col_indices]
    signs = torch.sign(peak_values)
    # Replace zeros with +1 to prevent zeroing out columns
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)

    # Adjust U and V synchronously
    U_fixed = U * signs.unsqueeze(0)
    V_fixed = V * signs.unsqueeze(0)

    return U_fixed, V_fixed


def exact_cusolver_svd(
    tensor: torch.Tensor,
    rank: Optional[int] = None,
    min_rank: int = 128,
    max_rank_cap: Optional[int] = None,
    energy_threshold: float = 0.999,
    apply_canonical_signs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Executes production-grade exact, deterministic cuSOLVER SVD on active GPU/CPU device.
    
    Features:
      - Canonical Bro & Kiers sign locking (prevents CPPN coordinate mirroring)
      - Cumulative energy-spectrum rank selection (sum sigma_i^2 / sum sigma_j^2 >= energy_threshold)
      - Production min_rank safeguard (default: 128 for ultra-high fidelity)
      - Float64 numerical fallback on ill-conditioned matrices
      - NaN / Inf sanitization
      
    Args:
        tensor: 2D Weight or Activation tensor in R^{M x N}.
        rank: Optional fixed rank truncation. If None, dynamically determined via energy_threshold.
        min_rank: Production lower-bound on rank (default: 128, clamped to min(M, N)).
        max_rank_cap: Optional upper-bound on rank.
        energy_threshold: Cumulative singular energy threshold (default: 0.995 = 99.5%).
        apply_canonical_signs: Whether to apply Bro & Kiers sign locking (default: True).
        
    Returns:
        (U_k, S_k, V_k, optimal_rank)
    """
    if tensor.dim() != 2:
        raise ValueError(f"exact_cusolver_svd requires a 2D tensor, got shape: {tensor.shape}")

    device = tensor.device
    orig_dtype = tensor.dtype
    M, N = tensor.shape
    max_rank = min(M, N)
    effective_min_rank = max(1, min(min_rank, max_rank))

    # Production numerical sanity check
    clean_tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    # Execute exact GPU cuSOLVER decomposition with double-precision fallback
    try:
        U, S, Vh = torch.linalg.svd(clean_tensor, full_matrices=False)
    except Exception:
        # Fallback to float64 for ill-conditioned matrices
        U, S, Vh = torch.linalg.svd(clean_tensor.to(torch.float64), full_matrices=False)
        U = U.to(orig_dtype)
        S = S.to(orig_dtype)
        Vh = Vh.to(orig_dtype)

    V = Vh.T  # V in R^{N x min(M, N)}

    # Apply Bro & Kiers Canonical Sign Stabilization
    if apply_canonical_signs:
        U, V = stabilize_svd_signs(U, V)

    # Determine optimal rank k*
    if rank is not None:
        k = max(1, min(rank, max_rank))
    else:
        # Energy-Spectrum Adaptive Rank: sum_{i=1}^k sigma_i^2 / sum_{j=1}^d sigma_j^2 >= threshold
        total_energy = torch.sum(S ** 2)
        if total_energy <= 1e-12:
            k = max(effective_min_rank, min(16, max_rank))
        else:
            cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
            mask = cum_energy >= energy_threshold
            if mask.any():
                k = int(torch.argmax(mask.to(torch.int64)).item()) + 1
            else:
                k = max_rank

        # Apply production bounds [effective_min_rank, max_rank_cap or max_rank]
        k = max(effective_min_rank, min(k, max_rank))

    if max_rank_cap is not None:
        k = min(k, max(1, min(max_rank_cap, max_rank)))

    # Truncate to top-k dimensions
    U_k = U[:, :k].contiguous()
    S_k = S[:k].contiguous()
    V_k = V[:, :k].contiguous()

    return U_k, S_k, V_k, k
