"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
Implements random rotation via Walsh-Hadamard Transform + Scalar Quantization
and QJL residual correction for unbiased inner product estimation.
Based on Zandieh et al., 2025 (arXiv:2504.19874v1, ICLR 2026).
Implements idea.md Section 9.2.
"""

import math
import torch
import torch.nn as nn


class TurboQuant:
    """
    TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
    1. Random orthogonal rotation Pi via Walsh-Hadamard with diagonal signs (coordinate variance = 1/D).
    2. Coordinate-wise scalar quantization using Lloyd-Max / scaled centroids.
    3. 1-bit QJL residual correction for unbiased inner product estimation.
    """
    def __init__(self, d_model: int, b_quant: int = 3):
        self.d_model = d_model
        self.b_quant = b_quant
        self.num_levels = 2 ** b_quant

        # Coordinate distribution has std = 1 / sqrt(D)
        # Optimal centroids span [-3 * std, +3 * std]
        std = 1.0 / math.sqrt(d_model)
        scale = 3.0 * std
        self.centroids = torch.linspace(-scale, scale, self.num_levels)

        # Generate orthogonal rotation matrix Pi (Walsh-Hadamard or Haar orthogonal)
        rot_mat = self._build_hadamard_or_haar_matrix(d_model)
        self.register_buffer("rotation_pi", rot_mat)

        # Normalized Gaussian projection matrix for QJL
        self.register_buffer("matrix_s", torch.randn(d_model, d_model) / math.sqrt(d_model))

    @staticmethod
    def _build_hadamard_or_haar_matrix(dim: int) -> torch.Tensor:
        """
        Builds randomized Walsh-Hadamard matrix Pi = H_d * diag(+-1) if dim is power of 2,
        or Haar orthogonal matrix via QR decomposition otherwise.
        """
        # Check if dim is power of 2
        if (dim & (dim - 1) == 0) and dim > 0:
            # Recursive Hadamard construction
            h = torch.tensor([[1.0]], dtype=torch.float32)
            while h.shape[0] < dim:
                top = torch.cat([h, h], dim=1)
                bottom = torch.cat([h, -h], dim=1)
                h = torch.cat([top, bottom], dim=0) / math.sqrt(2.0)
            # Apply random diagonal sign flips: Pi = H_d * diag(d_j)
            signs = torch.randint(0, 2, (dim,)).float() * 2.0 - 1.0
            return h * signs.unsqueeze(0)
        else:
            rand_mat = torch.randn(dim, dim)
            q, r = torch.linalg.qr(rand_mat)
            d = torch.diag(r)
            ph = d.sign()
            q *= ph
            return q

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def quantize(self, x: torch.Tensor) -> dict:
        """
        Compresses tensor x.
        x shape: [..., D]
        Returns dict with:
            idx: quantized indices [..., D] uint8
            norm: vector norms [...]
            gamma: residual norms [...]
            qjl: 1-bit residual corrections [..., D]
        """
        D = self.d_model
        orig_shape = x.shape
        x_flat = x.view(-1, D).float()

        # 1. Norm extraction: x_normalized in S^{D-1}
        norms = torch.norm(x_flat, p=2, dim=-1, keepdim=True) + 1e-8
        x_normalized = x_flat / norms

        # 2. Random Rotation: y = (x / ||x||) * Pi^T
        rot_pi = self.rotation_pi.to(x.device)
        y = torch.matmul(x_normalized, rot_pi.t())  # (N, D)

        # 3. Scalar Quantization per coordinate
        centroids = self.centroids.to(x.device)
        y_expanded = y.unsqueeze(-1)  # (N, D, 1)
        c_expanded = centroids.view(1, 1, -1)  # (1, 1, L)

        dist = (y_expanded - c_expanded).abs()
        idx = torch.argmin(dist, dim=-1)  # (N, D)

        # 4. Reconstruct MSE approximation to compute residual
        y_hat = centroids[idx]  # (N, D)
        x_hat_mse = norms * torch.matmul(y_hat, rot_pi)

        # 5. QJL Residual Correction
        residual = x_flat - x_hat_mse
        gamma = torch.norm(residual, p=2, dim=-1, keepdim=True) + 1e-8
        r_normalized = residual / gamma

        # qjl = sign( (r / gamma) * S^T )
        mat_s = self.matrix_s.to(x.device)
        qjl_proj = torch.matmul(r_normalized, mat_s.t())
        qjl = (qjl_proj >= 0).byte()

        # Pack results
        idx = idx.view(*orig_shape[:-1], D).byte()
        norms = norms.view(*orig_shape[:-1])
        gamma = gamma.view(*orig_shape[:-1])
        qjl = qjl.view(*orig_shape[:-1], D)

        return {"idx": idx, "norm": norms, "gamma": gamma, "qjl": qjl}

    def dequantize(self, quantized_data: dict) -> torch.Tensor:
        """
        Dequantizes data back to float tensor.
        """
        idx = quantized_data["idx"].long()
        norms = quantized_data["norm"].unsqueeze(-1)
        gamma = quantized_data["gamma"].unsqueeze(-1)
        qjl_bit = quantized_data["qjl"]

        D = self.d_model
        orig_shape = idx.shape
        idx_flat = idx.view(-1, D)
        norms_flat = norms.view(-1, 1).float()
        gamma_flat = gamma.view(-1, 1).float()

        centroids = self.centroids.to(idx.device)
        y_hat = centroids[idx_flat]

        rot_pi = self.rotation_pi.to(idx.device)
        x_hat_mse = norms_flat * torch.matmul(y_hat, rot_pi)

        # QJL correction: bit to [-1, 1]
        qjl = qjl_bit.view(-1, D).float() * 2.0 - 1.0
        mat_s = self.matrix_s.to(idx.device)
        scale = math.sqrt(math.pi / (2.0 * D))
        qjl_correction = torch.matmul(qjl, mat_s)
        x_hat_prod = x_hat_mse + gamma_flat * scale * qjl_correction

        return x_hat_prod.view(*orig_shape)
