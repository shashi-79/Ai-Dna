"""
Gen-0 Seed Extraction & Omni-Modal Inference Package.
Distills trained checkpoints into canonical .aidna DNA files and generates multi-modal outputs.
"""

from .extract_gen0_seed import extract_gen0_dna
from .run_gen0_inference import run_gen0_inference

__all__ = ["extract_gen0_dna", "run_gen0_inference"]
