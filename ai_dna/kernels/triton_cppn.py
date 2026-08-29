"""
Fused Tiled CPPN Multi-Activation Synaptic Weight Synthesis Kernel.
Evaluates continuous coordinate MLP with composite non-linearities (Sin, Gaussian, Sigmoid)
directly in GPU SRAM tiles.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_cppn_mlp_kernel(
        coords_ptr,
        w1_ptr, b1_ptr,
        w2_ptr, b2_ptr,
        out_ptr,
        M, N, K_in, H_dim,
        stride_c_m, stride_c_n, stride_c_k,
        stride_w1_h, stride_w1_k,
        stride_w2_o, stride_w2_h,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Fused CPPN Tiled Kernel:
        h = sin(x @ W1.T + b1) * exp(-(x @ W1.T)^2)
        out = h @ W2.T + b2
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        # Accumulator for output weight
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Iterate over hidden neurons
        for h in range(0, H_dim):
            # Accumulate Linear Layer 1
            linear1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(0, K_in):
                c_val = tl.load(
                    coords_ptr + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n + k * stride_c_k,
                    mask=mask,
                    other=0.0
                )
                w1_val = tl.load(w1_ptr + h * stride_w1_h + k * stride_w1_k)
                linear1 += c_val * w1_val

            b1_val = tl.load(b1_ptr + h)
            pre_act = linear1 + b1_val

            # Composite Activation: Sinusoidal + Gaussian Envelope
            sin_act = tl.sin(pre_act)
            gauss_act = tl.exp(-1.0 * pre_act * pre_act)
            act_val = sin_act * gauss_act

            # Layer 2 Projection
            w2_val = tl.load(w2_ptr + 0 * stride_w2_o + h * stride_w2_h)
            acc += act_val * w2_val

        # Add Layer 2 bias
        b2_val = tl.load(b2_ptr + 0)
        final_weight = acc + b2_val

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            final_weight,
            mask=mask
        )


def fused_cppn_synthesis_forward(
    coords: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Synthesizes weight matrix W in R^{M x N} from coordinate features using fused CPPN.
    """
    dev = device or coords.device
    M, N, K_in = coords.shape
    H_dim = w1.shape[0]

    coords = coords.to(device=dev, dtype=torch.float32)
    w1 = w1.to(device=dev, dtype=torch.float32)
    b1 = b1.to(device=dev, dtype=torch.float32)
    w2 = w2.to(device=dev, dtype=torch.float32)
    b2 = b2.to(device=dev, dtype=torch.float32)

    # Triton Execution
    if TRITON_AVAILABLE and dev.type == "cuda":
        out = torch.empty((M, N), device=dev, dtype=torch.float32)
        BLOCK_M = 16
        BLOCK_N = 16
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _fused_cppn_mlp_kernel[grid](
            coords,
            w1, b1,
            w2, b2,
            out,
            M, N, K_in, H_dim,
            coords.stride(0), coords.stride(1), coords.stride(2),
            w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        return out

    # PyTorch Vectorized Fallback
    h_pre = torch.matmul(coords, w1.T) + b1
    h_act = torch.sin(h_pre) * torch.exp(-torch.pow(h_pre, 2))
    out = torch.matmul(h_act, w2.T) + b2
    return out.squeeze(-1)
