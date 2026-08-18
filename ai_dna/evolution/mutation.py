"""
Stochastic Mutation Operator mu(D_t, xi_t).
Applies parametric and structural mutations across all 6 DNA components:
- DNAArchitecture (layers, experts, kv_latent_dim, rope_theta)
- DNARouting (top_k_experts, noise_std, temperature)
- DNAMemory (chunk_size, compression_rate, kv_quant_bits, page_size)
- DNALearning (learning_rate, plasticity, weight_decay)
- DNAInstinct (CPPN genetic parameters)
- DNAEvolution (mutation rates, compatibility threshold)
"""

import random
import copy
import torch
from typing import Optional
from ..dna.structure import Genotype
from ..dna.innovation import InnovationTracker


class GenotypeMutator:
    """
    Mutator that introduces bounded genetic variations to a Genotype.
    """
    def __init__(self, tracker: Optional[InnovationTracker] = None):
        self.tracker = tracker or InnovationTracker.get_global_tracker()

    def mutate(self, genotype: Genotype, mutation_rate: Optional[float] = None) -> Genotype:
        """
        Applies stochastic mutation to create a new Genotype generation.
        """
        child = genotype.clone(new_id=f"{genotype.genotype_id}_mut")
        child.generation = genotype.generation + 1

        evo = child.dna_evolution
        rate = mutation_rate if mutation_rate is not None else evo.mutation_rate
        scale = evo.param_mutation_scale

        # 1. Mutate Instinct Genetic Parameters (CPPN weights)
        for name, param in child.dna_instinct.genetic_parameters.items():
            if random.random() < rate:
                noise = torch.randn_like(param) * scale
                child.dna_instinct.genetic_parameters[name] = param + noise

        # 2. Mutate Routing Parameters (Top-K, noise std, temperature)
        if random.random() < rate:
            child.dna_routing.top_k_experts = random.choice([1, 2, 4])
        if random.random() < rate:
            child.dna_routing.routing_noise_std = float(
                max(0.0, min(2.0, child.dna_routing.routing_noise_std + (random.random() - 0.5) * 0.2))
            )
        if random.random() < rate:
            child.dna_routing.temperature = float(
                max(0.1, min(5.0, child.dna_routing.temperature + (random.random() - 0.5) * 0.2))
            )

        # 3. Mutate Memory Policies (chunk size, compression rate, quant bits, page size)
        if random.random() < rate:
            child.dna_memory.chunk_size = random.choice([16, 32, 64, 128])
        if random.random() < rate:
            child.dna_memory.compression_rate = float(
                max(0.1, min(0.5, child.dna_memory.compression_rate + (random.random() - 0.5) * 0.05))
            )
        if random.random() < rate:
            child.dna_memory.kv_quant_bits = random.choice([2, 3, 4, 8])
        if random.random() < rate:
            child.dna_memory.page_size = random.choice([8, 16, 32])

        # 4. Mutate Learning Dynamics
        if random.random() < rate:
            child.dna_learning.learning_rate = float(
                max(1e-5, min(1e-2, child.dna_learning.learning_rate * random.choice([0.8, 1.25])))
            )

        # 5. Mutate Architecture Topology & RoPE/MLA parameters
        if random.random() < rate:
            child.dna_architecture.rope_theta = float(random.choice([10000.0, 50000.0, 100000.0]))
        if random.random() < rate:
            child.dna_architecture.kv_latent_dim = random.choice([8, 16, 32])

        # 6. Structural Mutations (Expert additions with new Innovation IDs)
        if random.random() < evo.structural_mutation_rate:
            if random.random() < 0.5 and child.dna_architecture.num_experts < 16:
                child.dna_architecture.num_experts += 1
                new_inn_id = self.tracker.get_innovation_id("expert_node", target_id=child.dna_architecture.num_experts)
                child.node_innovation_map[f"expert_{child.dna_architecture.num_experts}"] = new_inn_id

        return child
