"""
Evolutionary Engine and Multi-Parent Fusion Package.
"""

from .mutation import GenotypeMutator
from .compatibility import CompatibilityChecker, CompatibilityScore, FunctionalNodeMatcher, NodeSimilarityScore
from .fusion import (
    AsymmetricLayerDepthLoRAFusion,
    MultiParentFusion,
    create_asymmetric_depth_fused_model,
    project_sigma_energy_tensor,
    extract_lora_instinct_components,
)
from .fitness import EvolutionaryFitnessEvaluator, GenerationalScalingTracker

__all__ = [
    "GenotypeMutator",
    "CompatibilityChecker",
    "CompatibilityScore",
    "FunctionalNodeMatcher",
    "NodeSimilarityScore",
    "AsymmetricLayerDepthLoRAFusion",
    "MultiParentFusion",
    "create_asymmetric_depth_fused_model",
    "project_sigma_energy_tensor",
    "extract_lora_instinct_components",
    "EvolutionaryFitnessEvaluator",
    "GenerationalScalingTracker",
]
