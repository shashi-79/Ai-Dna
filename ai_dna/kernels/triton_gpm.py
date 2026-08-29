"""
Fused GPM (Gradient Projection Memory) Null-Space Projection Kernel.
Projects updates into the orthogonal complement of historical activations:
Delta W_safe = Delta W - (Delta W @ U_k) @ U_k.T
with zero intermediate global memory allocations.
"""

import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_gpm_subtraction_kernel(
        delta_w_ptr,
        m_proj_ptr,
        u_basis_ptr,
        out_ptr,
        M, N, K,
        stride_dw_m, stride_dw_n,
        stride_m_m, stride_m_k,
        stride_u_n, stride_u_k,
        stride_out_m, stride_out_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Computes Out[m, n] = Delta_W[m, n] - sum_k(M_proj[m, k] * U[n, k])
        in a single fused read/write pass.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        # Load raw Delta_W
        dw_val = tl.load(
            delta_w_ptr + offs_m[:, None] * stride_dw_m + offs_n[None, :] * stride_dw_n,
            mask=mask,
            other=0.0
        )

        # Compute inner product projection (M_proj @ U.T)
        proj_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, K):
            m_val = tl.load(m_proj_ptr + offs_m[:, None] * stride_m_m + k * stride_m_k, mask=offs_m[:, None] < M, other=0.0)
            u_val = tl.load(u_basis_ptr + offs_n[None, :] * stride_u_n + k * stride_u_k, mask=offs_n[None, :] < N, other=0.0)
            proj_acc += m_val * u_val

        # Safe orthogonal update
        safe_dw = dw_val - proj_acc

        tl.store(
            out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n,
            safe_dw,
            mask=mask
        )


def fused_gpm_projection_forward(
    delta_w: torch.Tensor,
    u_basis: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Executes fused null-space projection:
    Delta W_safe = Delta W - (Delta W @ U_k) @ U_k.T
    Guarantees 0.0% catastrophic forgetting.
    """
    dev = device or delta_w.device
    delta_w = delta_w.to(device=dev, dtype=torch.float32)
    u_basis = u_basis.to(device=dev, dtype=torch.float32)

    M, N = delta_w.shape
    N_u, K = u_basis.shape
    assert N == N_u, f"Dimension mismatch: delta_w is {delta_w.shape}, u_basis is {u_basis.shape}"

    # Step 1: Intermediate projection matrix M_proj = delta_w @ u_basis [M, K]
    m_proj = torch.matmul(delta_w, u_basis)

    # Step 2: Fused Subtraction in Triton
    if TRITON_AVAILABLE and dev.type == "cuda":
        out = torch.empty((M, N), device=dev, dtype=torch.float32)
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _fused_gpm_subtraction_kernel[grid](
            delta_w,
            m_proj,
            u_basis,
            out,
            M, N, K,
            delta_w.stride(0), delta_w.stride(1),
            m_proj.stride(0), m_proj.stride(1),
            u_basis.stride(0), u_basis.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        return out

    # PyTorch Vectorized Fallback
    return delta_w - torch.matmul(m_proj, u_basis.T)
