"""
Omni-Modal AI DNA Architecture: Genotypic Instinct Encoding, Phenotypic Transferability, and Multi-Generational AI Evolution.
"""

__version__ = "0.2.0"

from .dna.structure import (
    Genotype,
    DNAArchitecture,
    DNAInstinct,
    DNARouting,
    DNAMemory,
    DNALearning,
    DNAEvolution,
)
from .dna.innovation import InnovationTracker
from .dna.serialization import save_genotype, load_genotype

from .growth.engine import GrowthEngine
from .growth.cppn import CPPNNetwork

from .routing.router import GenerativeSparseRouter
from .routing.topk_gate import TopKNoisyGate

from .memory.turboquant import TurboQuant
from .memory.hierarchical import HierarchicalMemoryController
from .memory.working_memory import WorkingMemory
from .memory.archive import PagedArchive, CompressedArchive
from .memory.retrieval import GraphRAG, ExternalVectorDatabase, RetrievalLibrary

from .models.rope import RoPE, RoPE2D, RoPE3D
from .models.mla import MultiHeadLatentAttention
from .models.phenotype import PhenotypeNeuralNetwork, PhenotypeTransformerBlock
from .models.modules import (
    TextEncoder,
    VisionEncoder,
    AudioEncoder,
    VideoEncoder,
    ContrastiveAlignmentHead,
    AutoregressiveDecoderHead,
    DiffusionDecoderHead,
    ClassificationHead,
)

from .inference.output_engine import OutputEngine
from .inference.sparse_executor import SparseHardwareExecutor
from .inference.tokenizer import TextTokenizer
from .inference.pipeline import InferencePipeline

from .encoding.svd_filter import SVDInstinctFilter
from .encoding.cppn_encoder import InverseCPPNEncoder
from .encoding.ewc import EWCConsolidator
from .encoding.slow_clock import SlowClockEncoder

from .evolution.mutation import GenotypeMutator
from .evolution.compatibility import CompatibilityChecker, FunctionalNodeMatcher
from .evolution.fusion import MultiParentFusion
from .evolution.fitness import EvolutionaryFitnessEvaluator, GenerationalScalingTracker

from .training.fast_clock import FastClockTrainer
from .training.losses import JointLoss
from .training.metrics import EvaluationMetrics, FalsificationChecker

__all__ = [
    "Genotype",
    "DNAArchitecture",
    "DNAInstinct",
    "DNARouting",
    "DNAMemory",
    "DNALearning",
    "DNAEvolution",
    "InnovationTracker",
    "save_genotype",
    "load_genotype",
    "GrowthEngine",
    "CPPNNetwork",
    "GenerativeSparseRouter",
    "TopKNoisyGate",
    "TurboQuant",
    "HierarchicalMemoryController",
    "WorkingMemory",
    "PagedArchive",
    "CompressedArchive",
    "GraphRAG",
    "ExternalVectorDatabase",
    "RetrievalLibrary",
    "RoPE",
    "RoPE2D",
    "RoPE3D",
    "MultiHeadLatentAttention",
    "PhenotypeNeuralNetwork",
    "PhenotypeTransformerBlock",
    "TextEncoder",
    "VisionEncoder",
    "AudioEncoder",
    "VideoEncoder",
    "ContrastiveAlignmentHead",
    "AutoregressiveDecoderHead",
    "DiffusionDecoderHead",
    "ClassificationHead",
    "OutputEngine",
    "SparseHardwareExecutor",
    "TextTokenizer",
    "InferencePipeline",
    "SVDInstinctFilter",
    "InverseCPPNEncoder",
    "EWCConsolidator",
    "SlowClockEncoder",
    "GenotypeMutator",
    "CompatibilityChecker",
    "FunctionalNodeMatcher",
    "MultiParentFusion",
    "EvolutionaryFitnessEvaluator",
    "GenerationalScalingTracker",
    "FastClockTrainer",
    "JointLoss",
    "EvaluationMetrics",
    "FalsificationChecker",
]
