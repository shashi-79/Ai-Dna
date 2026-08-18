"""
AI DNA Genotypic Representation Package.
"""

from .structure import (
    Genotype,
    DNAArchitecture,
    DNAInstinct,
    DNARouting,
    DNAMemory,
    DNALearning,
    DNAEvolution,
)
from .innovation import InnovationTracker, InnovationRecord
from .serialization import (
    genotype_to_dict,
    dict_to_genotype,
    save_genotype,
    load_genotype,
)

__all__ = [
    "Genotype",
    "DNAArchitecture",
    "DNAInstinct",
    "DNARouting",
    "DNAMemory",
    "DNALearning",
    "DNAEvolution",
    "InnovationTracker",
    "InnovationRecord",
    "genotype_to_dict",
    "dict_to_genotype",
    "save_genotype",
    "load_genotype",
]
