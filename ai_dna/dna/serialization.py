"""
Genotype serialization and deserialization to JSON and torch safetensors/checkpoints.
"""

import json
import os
from typing import Dict, Any
import torch
from .structure import (
    Genotype,
    DNAArchitecture,
    DNAInstinct,
    DNARouting,
    DNAMemory,
    DNALearning,
    DNAEvolution,
)


def genotype_to_dict(genotype: Genotype) -> Dict[str, Any]:
    """Converts a Genotype into a serializable dictionary."""
    # Convert genetic parameters to list format or serialized state
    genetic_params_meta = {}
    for name, tensor in genotype.dna_instinct.genetic_parameters.items():
        genetic_params_meta[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "values": tensor.detach().cpu().flatten().tolist(),
        }

    return {
        "genotype_id": genotype.genotype_id,
        "generation": genotype.generation,
        "parent_ids": genotype.parent_ids,
        "lineage_notes": genotype.lineage_notes,
        "fitness_history": genotype.fitness_history,
        "node_innovation_map": genotype.node_innovation_map,
        "dna_architecture": {
            "num_layers": genotype.dna_architecture.num_layers,
            "d_model": genotype.dna_architecture.d_model,
            "num_heads": genotype.dna_architecture.num_heads,
            "num_experts": genotype.dna_architecture.num_experts,
            "d_expert_hidden": genotype.dna_architecture.d_expert_hidden,
            "vocab_size": genotype.dna_architecture.vocab_size,
            "coord_dim": genotype.dna_architecture.coord_dim,
            "active_expert_threshold": genotype.dna_architecture.active_expert_threshold,
            "innovation_id": genotype.dna_architecture.innovation_id,
        },
        "dna_instinct": {
            "cppn_hidden_dim": genotype.dna_instinct.cppn_hidden_dim,
            "cppn_layers": genotype.dna_instinct.cppn_layers,
            "singular_energy_threshold": genotype.dna_instinct.singular_energy_threshold,
            "instinct_rank_ratio": genotype.dna_instinct.instinct_rank_ratio,
            "innovation_id": genotype.dna_instinct.innovation_id,
            "genetic_parameters": genetic_params_meta,
        },
        "dna_routing": {
            "rank": genotype.dna_routing.rank,
            "threshold": genotype.dna_routing.threshold,
            "temperature": genotype.dna_routing.temperature,
            "load_balance_weight": genotype.dna_routing.load_balance_weight,
            "innovation_id": genotype.dna_routing.innovation_id,
        },
        "dna_memory": {
            "chunk_size": genotype.dna_memory.chunk_size,
            "compression_rate": genotype.dna_memory.compression_rate,
            "num_retrieval": genotype.dna_memory.num_retrieval,
            "cost_alpha": genotype.dna_memory.cost_alpha,
            "cost_beta": genotype.dna_memory.cost_beta,
            "cost_delta": genotype.dna_memory.cost_delta,
            "innovation_id": genotype.dna_memory.innovation_id,
        },
        "dna_learning": {
            "learning_rate": genotype.dna_learning.learning_rate,
            "weight_decay": genotype.dna_learning.weight_decay,
            "plasticity_rate": genotype.dna_learning.plasticity_rate,
            "optimizer_type": genotype.dna_learning.optimizer_type,
            "gradient_clip": genotype.dna_learning.gradient_clip,
            "innovation_id": genotype.dna_learning.innovation_id,
        },
        "dna_evolution": {
            "mutation_rate": genotype.dna_evolution.mutation_rate,
            "structural_mutation_rate": genotype.dna_evolution.structural_mutation_rate,
            "param_mutation_scale": genotype.dna_evolution.param_mutation_scale,
            "min_compatibility_score": genotype.dna_evolution.min_compatibility_score,
            "innovation_id": genotype.dna_evolution.innovation_id,
        },
    }


def dict_to_genotype(data: Dict[str, Any]) -> Genotype:
    """Reconstructs a Genotype from dictionary data."""
    # Rebuild genetic parameters
    genetic_params = {}
    if "genetic_parameters" in data.get("dna_instinct", {}):
        for name, meta in data["dna_instinct"]["genetic_parameters"].items():
            shape = meta["shape"]
            values = meta["values"]
            t = torch.tensor(values, dtype=torch.float32).reshape(shape)
            genetic_params[name] = t

    instinct_data = data.get("dna_instinct", {})
    instinct = DNAInstinct(
        cppn_hidden_dim=instinct_data.get("cppn_hidden_dim", 32),
        cppn_layers=instinct_data.get("cppn_layers", 3),
        singular_energy_threshold=instinct_data.get("singular_energy_threshold", 0.85),
        instinct_rank_ratio=instinct_data.get("instinct_rank_ratio", 0.25),
        innovation_id=instinct_data.get("innovation_id", 2),
        genetic_parameters=genetic_params,
    )

    arch_data = data.get("dna_architecture", {})
    arch = DNAArchitecture(
        num_layers=arch_data.get("num_layers", 4),
        d_model=arch_data.get("d_model", 64),
        num_heads=arch_data.get("num_heads", 4),
        num_experts=arch_data.get("num_experts", 4),
        d_expert_hidden=arch_data.get("d_expert_hidden", 128),
        vocab_size=arch_data.get("vocab_size", 1000),
        coord_dim=arch_data.get("coord_dim", 5),
        active_expert_threshold=arch_data.get("active_expert_threshold", 0.5),
        innovation_id=arch_data.get("innovation_id", 1),
    )

    routing_data = data.get("dna_routing", {})
    routing = DNARouting(
        rank=routing_data.get("rank", 4),
        threshold=routing_data.get("threshold", 0.5),
        temperature=routing_data.get("temperature", 1.0),
        load_balance_weight=routing_data.get("load_balance_weight", 0.01),
        innovation_id=routing_data.get("innovation_id", 3),
    )

    memory_data = data.get("dna_memory", {})
    memory = DNAMemory(
        chunk_size=memory_data.get("chunk_size", 32),
        compression_rate=memory_data.get("compression_rate", 0.25),
        num_retrieval=memory_data.get("num_retrieval", 8),
        cost_alpha=memory_data.get("cost_alpha", 1.0),
        cost_beta=memory_data.get("cost_beta", 0.5),
        cost_delta=memory_data.get("cost_delta", 0.2),
        innovation_id=memory_data.get("innovation_id", 4),
    )

    learning_data = data.get("dna_learning", {})
    learning = DNALearning(
        learning_rate=learning_data.get("learning_rate", 1e-3),
        weight_decay=learning_data.get("weight_decay", 1e-4),
        plasticity_rate=learning_data.get("plasticity_rate", 0.1),
        optimizer_type=learning_data.get("optimizer_type", "adamw"),
        gradient_clip=learning_data.get("gradient_clip", 1.0),
        innovation_id=learning_data.get("innovation_id", 5),
    )

    evolution_data = data.get("dna_evolution", {})
    evolution = DNAEvolution(
        mutation_rate=evolution_data.get("mutation_rate", 0.05),
        structural_mutation_rate=evolution_data.get("structural_mutation_rate", 0.02),
        param_mutation_scale=evolution_data.get("param_mutation_scale", 0.02),
        min_compatibility_score=evolution_data.get("min_compatibility_score", 0.6),
        innovation_id=evolution_data.get("innovation_id", 6),
    )

    return Genotype(
        dna_architecture=arch,
        dna_instinct=instinct,
        dna_routing=routing,
        dna_memory=memory,
        dna_learning=learning,
        dna_evolution=evolution,
        generation=data.get("generation", 0),
        genotype_id=data.get("genotype_id", "gen_0"),
        parent_ids=data.get("parent_ids", []),
        lineage_notes=data.get("lineage_notes", ""),
        fitness_history=data.get("fitness_history", {}),
        node_innovation_map=data.get("node_innovation_map", {}),
    )


def save_genotype(genotype: Genotype, file_path: str):
    """Saves a genotype to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    data = genotype_to_dict(genotype)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_genotype(file_path: str) -> Genotype:
    """Loads a genotype from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict_to_genotype(data)
