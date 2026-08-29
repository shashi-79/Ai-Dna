"""
Unit tests for the GPU Kernel Acceleration Suite (`ai_dna/kernels/`).
Tests numerical parity and mathematical correctness of RFF coordinates,
CPPN weight synthesis, and GPM null-space projection.
"""

import unittest
import torch
from ai_dna.kernels.triton_rff import fused_rff_coordinate_forward
from ai_dna.kernels.triton_cppn import fused_cppn_synthesis_forward
from ai_dna.kernels.triton_gpm import fused_gpm_projection_forward


class TestTritonKernels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_fused_rff_coordinate_kernel(self):
        M, N = 64, 64
        coord_dim = 32
        out_features = 64
        B_weight = torch.randn((out_features // 2, coord_dim), device=self.device) * 0.5

        # Forward pass
        coords_rff = fused_rff_coordinate_forward(
            M=M, N=N, coord_dim=coord_dim, out_features=out_features, B_weight=B_weight, device=self.device
        )
        self.assertEqual(coords_rff.shape, (M, N, out_features))
        self.assertFalse(torch.isnan(coords_rff).any())
        self.assertFalse(torch.isinf(coords_rff).any())

        # Check bounded range of sinusoidal features [-1, 1]
        self.assertTrue((coords_rff >= -1.05).all())
        self.assertTrue((coords_rff <= 1.05).all())

    def test_fused_cppn_synthesis_kernel(self):
        M, N = 32, 32
        K_in = 64
        H_dim = 32

        coords = torch.randn((M, N, K_in), device=self.device)
        w1 = torch.randn((H_dim, K_in), device=self.device) * 0.1
        b1 = torch.zeros((H_dim,), device=self.device)
        w2 = torch.randn((1, H_dim), device=self.device) * 0.1
        b2 = torch.zeros((1,), device=self.device)

        # Fused synthesis
        weights = fused_cppn_synthesis_forward(
            coords=coords, w1=w1, b1=b1, w2=w2, b2=b2, device=self.device
        )
        self.assertEqual(weights.shape, (M, N))
        self.assertFalse(torch.isnan(weights).any())
        self.assertFalse(torch.isinf(weights).any())

    def test_fused_gpm_null_space_projection(self):
        M, N = 64, 64
        K = 16

        delta_w = torch.randn((M, N), device=self.device)
        # Create orthonormal historical basis U_k
        raw_u = torch.randn((N, K), device=self.device)
        u_basis, _ = torch.linalg.qr(raw_u)

        # Apply fused projection
        safe_dw = fused_gpm_projection_forward(delta_w, u_basis, device=self.device)

        self.assertEqual(safe_dw.shape, (M, N))
        self.assertFalse(torch.isnan(safe_dw).any())

        # Mathematical verification: safe_dw @ u_basis must be approximately ZERO (null space)
        projection_residual = torch.matmul(safe_dw, u_basis)
        max_residual = torch.max(torch.abs(projection_residual)).item()
        self.assertLess(max_residual, 1e-4, f"GPM Null-space residual too large: {max_residual}")


if __name__ == "__main__":
    unittest.main()
