"""
Tests for Mutation, Compatibility, Multi-Parent Fusion, and Evolutionary Fitness.
"""

import torch
from ai_dna.dna.structure import Genotype
from ai_dna.evolution.mutation import GenotypeMutator
from ai_dna.evolution.compatibility import CompatibilityChecker
from ai_dna.evolution.fusion import MultiParentFusion
from ai_dna.evolution.fitness import EvolutionaryFitnessEvaluator


def test_mutation_operator():
    genotype = Genotype.create_default(genotype_id="mut_test")
    genotype.dna_instinct.genetic_parameters["test_gene"] = torch.zeros(4, 4)

    mutator = GenotypeMutator()
    mutated = mutator.mutate(genotype, mutation_rate=1.0)

    assert mutated.generation == 1
    # Parameters should have changed due to mutation noise
    assert not torch.allclose(mutated.dna_instinct.genetic_parameters["test_gene"], torch.zeros(4, 4))


def test_compatibility_and_fusion():
    p1 = Genotype.create_default(genotype_id="p1")
    p2 = Genotype.create_default(genotype_id="p2")

    p1.dna_instinct.genetic_parameters["w_shared"] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    p2.dna_instinct.genetic_parameters["w_shared"] = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    p2.dna_instinct.genetic_parameters["w_disjoint"] = torch.tensor([[5.0, 5.0]])

    comp = CompatibilityChecker.evaluate(p1, p2)
    assert comp.is_compatible

    fusion = MultiParentFusion()
    child = fusion.fuse([p1, p2], weights=[0.5, 0.5], child_id="child_12")

    assert child.generation == 1
    # Overlapping instinct inherits from parent with higher SVD energy (p2 with 3.0 vs p1 with 1.0)
    assert torch.allclose(child.dna_instinct.genetic_parameters["w_shared"], torch.tensor([[3.0, 3.0], [3.0, 3.0]]))
    # Disjoint node should be inherited
    assert "w_disjoint" in child.dna_instinct.genetic_parameters


def test_fitness_evaluator():
    evaluator = EvolutionaryFitnessEvaluator()
    genotype = Genotype.create_default(genotype_id="fit_test")

    fitness = evaluator.compute_fitness(genotype, sample_efficiency=1.5, compute_cost=5.0)
    assert "overall_fitness" in genotype.fitness_history
