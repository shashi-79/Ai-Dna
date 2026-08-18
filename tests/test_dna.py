"""
Tests for AI DNA Genotype Structure, Innovation Tracker, and Serialization.
"""

import tempfile
import os
import torch
from ai_dna.dna.structure import Genotype, DNAArchitecture, DNAInstinct
from ai_dna.dna.innovation import InnovationTracker
from ai_dna.dna.serialization import save_genotype, load_genotype, genotype_to_dict, dict_to_genotype


def test_genotype_creation_and_defaults():
    tracker = InnovationTracker.reset_global_tracker()
    genotype = Genotype.create_default(tracker=tracker, genotype_id="test_gen_0")

    assert genotype.genotype_id == "test_gen_0"
    assert genotype.generation == 0
    assert genotype.dna_architecture.num_layers == 4
    assert genotype.dna_architecture.d_model == 64
    assert genotype.dna_routing.rank == 4
    assert genotype.dna_memory.chunk_size == 32
    assert len(genotype.node_innovation_map) == 6


def test_innovation_tracker_persistence():
    tracker = InnovationTracker(start_id=1)
    id1 = tracker.get_innovation_id("expert_node", source_id=1, target_id=2)
    id2 = tracker.get_innovation_id("expert_node", source_id=1, target_id=2)
    id3 = tracker.get_innovation_id("expert_node", source_id=1, target_id=3)

    assert id1 == id2  # Identical origin should receive identical innovation ID
    assert id1 != id3
    assert tracker.total_innovations() == 2


def test_genotype_serialization_and_deserialization():
    genotype = Genotype.create_default(genotype_id="ser_test")
    genotype.dna_instinct.genetic_parameters["cppn.weight"] = torch.randn(16, 5)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        save_genotype(genotype, temp_path)
        loaded = load_genotype(temp_path)

        assert loaded.genotype_id == genotype.genotype_id
        assert loaded.dna_architecture.d_model == genotype.dna_architecture.d_model
        assert "cppn.weight" in loaded.dna_instinct.genetic_parameters
        assert torch.allclose(
            genotype.dna_instinct.genetic_parameters["cppn.weight"],
            loaded.dna_instinct.genetic_parameters["cppn.weight"],
            atol=1e-5,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
