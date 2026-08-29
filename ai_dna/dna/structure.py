"""
Constitutional Definition of AI DNA.
Defines D = (D_architecture, D_instinct, D_routing, D_memory, D_learning, D_evolution)
and the Genotype container.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import torch
import copy
from .innovation import InnovationTracker


@dataclass
class DNAArchitecture:
    """Defines structural topology of phenotype."""
    num_layers: int = 4
    d_model: int = 64
    num_heads: int = 4
    num_experts: int = 4
    d_expert_hidden: int = 128
    vocab_size: int = 1000
    coord_dim: int = 32  # 32D Hardware-Aligned Universal Coordinate Manifold (16D Source + 16D Target)
    active_expert_threshold: float = 0.5
    kv_latent_dim: int = 16    # d_kv (MLA latent dimension)
    rope_theta: float = 10000.0 # Base frequency for RoPE
    innovation_id: int = 1


@dataclass
class DNAInstinct:
    """
    Defines transferable structural developmental parameters.
    Contains weights for Compositional Pattern Producing Network (CPPN) / Growth generator.
    """
    cppn_hidden_dim: int = 32
    cppn_layers: int = 3
    # Dictionary of layer_name -> tensor of genetic parameters
    genetic_parameters: Dict[str, torch.Tensor] = field(default_factory=dict)
    singular_energy_threshold: float = 0.85
    instinct_rank_ratio: float = 0.25
    epigenetic_mask: Optional[Dict[str, torch.Tensor]] = None  # Epigenetic gene silencing/scaling mask
    innovation_id: int = 2


@dataclass
class DNARouting:
    """Defines Top-K Noisy Gating routing behavior."""
    rank: int = 4
    threshold: float = 0.5
    temperature: float = 1.0
    load_balance_weight: float = 0.01
    top_k_experts: int = 2          # Top-K sparse gating
    routing_noise_std: float = 1.0  # Noise exploration scale
    innovation_id: int = 3


@dataclass
class DNAMemory:
    """Defines hierarchical long-context memory policies."""
    chunk_size: int = 32          # C_chunk (local attention window)
    compression_rate: float = 0.25 # c_rate (compression ratio for historical latents)
    num_retrieval: int = 8        # N_retrieval (number of retrieved latent memories)
    kv_quant_bits: int = 3        # TurboQuant scalar quantization bits
    page_size: int = 16           # PagedAttention archive page size
    max_pages: int = 1024         # PagedAttention maximum allowed pages
    cost_alpha: float = 1.0       # Sequential time cost weight
    cost_beta: float = 0.5        # Peak memory cost weight
    cost_delta: float = 0.2       # Total memory cost weight
    innovation_id: int = 4


@dataclass
class DNALearning:
    """Defines Fast Clock plasticity, optimizer hyperparameters, and learning dynamics."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    plasticity_rate: float = 0.1
    optimizer_type: str = "adamw"
    gradient_clip: float = 1.0
    innovation_id: int = 5


@dataclass
class DNAEvolution:
    """Defines evolutionary variation, mutation rates, and fusion compatibility."""
    mutation_rate: float = 0.05
    structural_mutation_rate: float = 0.02
    param_mutation_scale: float = 0.02
    min_compatibility_score: float = 0.6
    innovation_id: int = 6


@dataclass
class Genotype:
    """
    Complete Constitutional AI Genotype:
    D = (D_architecture, D_instinct, D_routing, D_memory, D_learning, D_evolution)
    """
    dna_architecture: DNAArchitecture = field(default_factory=DNAArchitecture)
    dna_instinct: DNAInstinct = field(default_factory=DNAInstinct)
    dna_routing: DNARouting = field(default_factory=DNARouting)
    dna_memory: DNAMemory = field(default_factory=DNAMemory)
    dna_learning: DNALearning = field(default_factory=DNALearning)
    dna_evolution: DNAEvolution = field(default_factory=DNAEvolution)
    
    generation: int = 0
    genotype_id: str = "dna_root"
    parent_ids: List[str] = field(default_factory=list)
    lineage_notes: str = "Initial Generation"
    fitness_history: Dict[str, float] = field(default_factory=dict)
    
    # Persistent Innovation IDs map for fine-grained node tracking
    node_innovation_map: Dict[str, int] = field(default_factory=dict)
    
    # Genotypically Embedded Calibration Anchors (GECA) for zero-dataset calibration
    calibration_anchors: Dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def create_default(cls, tracker: Optional[InnovationTracker] = None, genotype_id: str = "gen_0") -> "Genotype":
        if tracker is None:
            tracker = InnovationTracker.get_global_tracker()
        
        arch = DNAArchitecture(innovation_id=tracker.get_innovation_id("architecture"))
        instinct = DNAInstinct(innovation_id=tracker.get_innovation_id("instinct"))
        routing = DNARouting(innovation_id=tracker.get_innovation_id("routing"))
        memory = DNAMemory(innovation_id=tracker.get_innovation_id("memory"))
        learning = DNALearning(innovation_id=tracker.get_innovation_id("learning"))
        evolution = DNAEvolution(innovation_id=tracker.get_innovation_id("evolution"))
        
        node_map = {
            "arch": arch.innovation_id,
            "instinct": instinct.innovation_id,
            "routing": routing.innovation_id,
            "memory": memory.innovation_id,
            "learning": learning.innovation_id,
            "evolution": evolution.innovation_id,
        }
        
        return cls(
            dna_architecture=arch,
            dna_instinct=instinct,
            dna_routing=routing,
            dna_memory=memory,
            dna_learning=learning,
            dna_evolution=evolution,
            generation=0,
            genotype_id=genotype_id,
            node_innovation_map=node_map,
        )

    def clone(self, new_id: Optional[str] = None) -> "Genotype":
        """Creates a deep copy of the genotype."""
        new_gen = copy.deepcopy(self)
        if new_id:
            new_gen.genotype_id = new_id
            new_gen.parent_ids = [self.genotype_id]
        return new_gen

    def total_parameters(self) -> int:
        """Returns total scalar genetic parameters in D."""
        count = 0
        for tensor in self.dna_instinct.genetic_parameters.values():
            count += tensor.numel()
        return count
