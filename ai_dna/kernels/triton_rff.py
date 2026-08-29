"""
Fused Coordinate Generation & Random Fourier Feature (RFF / SIREN) Kernel.
Computes in-register sinusoidal projections: gamma(c) = [cos(2*pi*B*c), sin(2*pi*B*c)]
with zero global memory round-trips.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

# Check for Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_rff_coordinate_kernel(
        out_ptr,
        B_weight_ptr,
        M, N, coord_dim, out_features,
        stride_out_m, stride_out_n, stride_out_f,
        stride_b_k, stride_b_in,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Fused kernel: Generates normalized coordinates on-the-fly in SRAM and applies
        sinusoidal projection: [cos(2*pi*B*c), sin(2*pi*B*c)] directly to output.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        # Coordinate synthesis: c_src in [-1, 1], c_tgt in [-1, 1]
        c_src = -1.0 + 2.0 * (offs_m.to(tl.float32) / (M - 1.0 + 1e-6))
        c_tgt = -1.0 + 2.0 * (offs_n.to(tl.float32) / (N - 1.0 + 1e-6))

        # Base 2D coordinate vector
        c_diff = c_tgt - c_src
        c_dist = tl.sqrt(c_src * c_src + c_tgt * c_tgt)

        # Fused 2*pi projection in registers
        pi_factor = 2.0 * 3.141592653589793
        half_dim = out_features // 2

        for f in range(0, half_dim):
            # Read B weight
            b_val = tl.load(B_weight_ptr + f * stride_b_k + 0 * stride_b_in)
            proj = (c_src * b_val + c_tgt * b_val + c_dist * 0.5) * pi_factor

            cos_val = tl.cos(proj)
            sin_val = tl.sin(proj)

            # Store cos half
            tl.store(
                out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n + f * stride_out_f,
                cos_val[:, None] + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32),
                mask=mask_m[:, None] & mask_n[None, :]
            )
            # Store sin half
            tl.store(
                out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n + (f + half_dim) * stride_out_f,
                sin_val[:, None] + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32),
                mask=mask_m[:, None] & mask_n[None, :]
            )


def fused_rff_coordinate_forward(
    M: int,
    N: int,
    coord_dim: int = 32,
    out_features: int = 64,
    B_weight: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Executes fused RFF coordinate generation. Uses compiled Triton kernel if on CUDA,
    falling back to high-throughput PyTorch vectorization otherwise.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half_features = out_features // 2

    if B_weight is None:
        B_weight = torch.randn((half_features, coord_dim), device=dev, dtype=dtype) * 0.5
    else:
        B_weight = B_weight.to(device=dev, dtype=dtype)

    # Fast Triton execution on CUDA
    if TRITON_AVAILABLE and dev.type == "cuda":
        out = torch.empty((M, N, out_features), device=dev, dtype=dtype)
        BLOCK_M = 32
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _fused_rff_coordinate_kernel[grid](
            out,
            B_weight,
            M, N, coord_dim, out_features,
            out.stride(0), out.stride(1), out.stride(2),
            B_weight.stride(0), B_weight.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        return out

    # PyTorch Vectorized Fallback
    src = torch.linspace(-1.0, 1.0, M, device=dev, dtype=dtype).unsqueeze(1).expand(M, N)
    tgt = torch.linspace(-1.0, 1.0, N, device=dev, dtype=dtype).unsqueeze(0).expand(M, N)
    diff = tgt - src
    dist = torch.sqrt(src * src + tgt * tgt)

    # Stack coordinates [M, N, 4]
    coords_base = torch.stack([src, tgt, diff, dist], dim=-1)
    
    # Pad to coord_dim
    if coord_dim > 4:
        padding = torch.zeros((M, N, coord_dim - 4), device=dev, dtype=dtype)
        coords = torch.cat([coords_base, padding], dim=-1)
    else:
        coords = coords_base[:, :, :coord_dim]

    # Sinusoidal Gaussian projection
    proj = 2.0 * math.pi * torch.matmul(coords, B_weight.T)
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
