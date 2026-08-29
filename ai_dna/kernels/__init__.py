"""
GPU Kernel Acceleration Suite for AI-DNA.
Provides high-throughput fused operators for Coordinate Generation (RFF/SIREN),
CPPN Weight Synthesis, and GPM Null-Space Projection.
"""

from .triton_rff import fused_rff_coordinate_forward
from .triton_cppn import fused_cppn_synthesis_forward
from .triton_gpm import fused_gpm_projection_forward
from .hybrid_svd import exact_cusolver_svd, stabilize_svd_signs

__all__ = [
    "fused_rff_coordinate_forward",
    "fused_cppn_synthesis_forward",
    "fused_gpm_projection_forward",
    "exact_cusolver_svd",
    "stabilize_svd_signs",
]
