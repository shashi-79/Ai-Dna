"""
Multi-Parent Fusion Engine D_c = F(D_1, D_2, ..., D_n).
Implements idea.md Sections 19-25:
- Compatibility verification (Section 20)
- Innovation ID + Functional node matching (Sections 21-22)
- Shared-node parameter blending (Section 23)
- SVD energy-based disjoint node inheritance (Section 24)
- Child validation (Section 25)
"""

import copy
import torch
from typing import List, Dict, Optional, Tuple, Any, Callable
from ..dna.structure import Genotype, DNAArchitecture, DNAInstinct, DNARouting, DNAMemory, DNALearning, DNAEvolution
from .compatibility import CompatibilityChecker, FunctionalNodeMatcher, CompatibilityScore


class MultiParentFusion:
    """
    Fuses multiple compatible parent Genotypes into a unified offspring Genotype.
    Uses Innovation ID matching, functional node similarity, and SVD energy-based disjoint selection.
    """
    def __init__(self, min_compatibility: float = 0.6, functional_match_threshold: float = 0.6):
        self.min_compatibility = min_compatibility
        self.node_matcher = FunctionalNodeMatcher(match_threshold=functional_match_threshold)

    @staticmethod
    def _compute_svd_energy(tensor: torch.Tensor) -> float:
        """
        Computes total singular energy Sigma for a parameter tensor (Section 24).
        Used to decide disjoint node inheritance: inherit from parent with greater Sigma.
        """
        if tensor.ndim < 2:
            return tensor.float().abs().sum().item()
        try:
            t_2d = tensor.reshape(tensor.shape[0], -1).float()
            _, s, _ = torch.linalg.svd(t_2d, full_matrices=False)
            return (s ** 2).sum().item()
        except Exception:
            return tensor.float().abs().sum().item()

    def fuse(
        self,
        parents: List[Genotype],
        weights: Optional[List[float]] = None,
        child_id: str = "child_gen",
    ) -> Genotype:
        """
        Fuses n parent genotypes: D_c = F(D_1, ..., D_n).
        Uses Innovation IDs for shared nodes, functional similarity for alignment,
        and SVD energy for disjoint node inheritance.
        """
        if len(parents) < 2:
            raise ValueError("At least 2 parent genotypes are required for multi-parent fusion.")

        # 1. Compatibility verification across parent pairs (Section 20)
        for i in range(len(parents) - 1):
            comp = CompatibilityChecker.evaluate(parents[i], parents[i+1], min_score=self.min_compatibility)
            if not comp.is_compatible:
                raise ValueError(f"Incompatible parents ({parents[i].genotype_id}, {parents[i+1].genotype_id}): {comp.reason}")

        # Normalize fusion weights
        n = len(parents)
        if weights is None:
            w_norm = [1.0 / n] * n
        else:
            total_w = sum(weights)
            w_norm = [w / total_w for w in weights]

        # 2. Base Architecture Inheritance (from highest weighted parent)
        best_parent_idx = int(torch.argmax(torch.tensor(w_norm)).item())
        p_primary = parents[best_parent_idx]

        child_arch = copy.deepcopy(p_primary.dna_architecture)
        child_routing = copy.deepcopy(p_primary.dna_routing)
        child_memory = copy.deepcopy(p_primary.dna_memory)
        child_learning = copy.deepcopy(p_primary.dna_learning)
        child_evolution = copy.deepcopy(p_primary.dna_evolution)

        # 3. Classify parameter keys into shared vs disjoint using Innovation IDs (Section 21)
        all_param_keys = set()
        for p in parents:
            all_param_keys.update(p.dna_instinct.genetic_parameters.keys())

        child_genetic_params = {}
        processed_keys = set()

        # 3a. Shared nodes: keys present in ALL parents (Section 23)
        for key in all_param_keys:
            matching_parents = []
            matching_weights = []
            for idx, p in enumerate(parents):
                if key in p.dna_instinct.genetic_parameters:
                    matching_parents.append(p.dna_instinct.genetic_parameters[key])
                    matching_weights.append(w_norm[idx])

            if len(matching_parents) == len(parents):
                # Shared node: theta_shared = sum(w_i * theta_i) (Section 23)
                m_weights_sum = sum(matching_weights)
                blended = torch.zeros_like(matching_parents[0])
                for t, w in zip(matching_parents, matching_weights):
                    blended = blended + t * (w / m_weights_sum)
                child_genetic_params[key] = blended
                processed_keys.add(key)

        # 3b. Disjoint nodes: present in some but not all parents (Section 24)
        disjoint_keys = all_param_keys - processed_keys

        for key in disjoint_keys:
            # Find which parents have this key
            owning_parents = []
            for idx, p in enumerate(parents):
                if key in p.dna_instinct.genetic_parameters:
                    owning_parents.append((idx, p.dna_instinct.genetic_parameters[key]))

            if not owning_parents:
                continue

            # 3b-i. Try functional matching across parents that DON'T have this key (Section 22)
            functionally_matched = False
            for non_owner_idx, p in enumerate(parents):
                if key not in p.dna_instinct.genetic_parameters:
                    matches = self.node_matcher.find_best_matches(
                        {key: owning_parents[0][1]},
                        p.dna_instinct.genetic_parameters,
                    )
                    if key in matches:
                        matched_key, sim_score = matches[key]
                        # Blend the functionally matched nodes
                        owner_tensor = owning_parents[0][1]
                        matched_tensor = p.dna_instinct.genetic_parameters[matched_key]
                        if owner_tensor.shape == matched_tensor.shape:
                            owner_weight = w_norm[owning_parents[0][0]]
                            match_weight = w_norm[non_owner_idx]
                            total_w = owner_weight + match_weight
                            child_genetic_params[key] = (
                                owner_tensor * (owner_weight / total_w)
                                + matched_tensor * (match_weight / total_w)
                            )
                            functionally_matched = True
                            break

            # 3b-ii. SVD energy-based disjoint inheritance (Section 24)
            if not functionally_matched:
                if len(owning_parents) == 1:
                    child_genetic_params[key] = owning_parents[0][1].clone()
                else:
                    # Inherit from parent with greater SVD energy
                    best_owner = max(
                        owning_parents,
                        key=lambda x: self._compute_svd_energy(x[1])
                    )
                    child_genetic_params[key] = best_owner[1].clone()

        child_instinct = DNAInstinct(
            cppn_hidden_dim=p_primary.dna_instinct.cppn_hidden_dim,
            cppn_layers=p_primary.dna_instinct.cppn_layers,
            genetic_parameters=child_genetic_params,
            singular_energy_threshold=p_primary.dna_instinct.singular_energy_threshold,
            instinct_rank_ratio=p_primary.dna_instinct.instinct_rank_ratio,
        )

        # 4. Merge Innovation IDs map (Section 21)
        merged_nodes = {}
        for p in parents:
            merged_nodes.update(p.node_innovation_map)

        max_gen = max(p.generation for p in parents)

        return Genotype(
            dna_architecture=child_arch,
            dna_instinct=child_instinct,
            dna_routing=child_routing,
            dna_memory=child_memory,
            dna_learning=child_learning,
            dna_evolution=child_evolution,
            generation=max_gen + 1,
            genotype_id=child_id,
            parent_ids=[p.genotype_id for p in parents],
            lineage_notes=f"Fused from {len(parents)} parents: {', '.join(p.genotype_id for p in parents)}",
            node_innovation_map=merged_nodes,
        )

    @staticmethod
    def validate_child(
        child_genotype: Genotype,
        growth_engine: Any,
        parent_tasks: Dict[str, Callable],
    ) -> Dict[str, float]:
        """
        Child Validation (Section 25): Evaluates the fused child D_c on parent task distributions.
        Grows W_c = G(D_c), then evaluates on T_A, T_B, and T_AB.

        Args:
            child_genotype: The fused child genotype.
            growth_engine: GrowthEngine to grow phenotype from genotype.
            parent_tasks: Dict mapping task_name -> callable(phenotype_model) -> float (performance score).

        Returns:
            Dict of task_name -> performance score.
        """
        child_model = growth_engine.grow_phenotype_model(child_genotype)
        child_model.eval()

        results = {}
        for task_name, task_eval_fn in parent_tasks.items():
            try:
                score = task_eval_fn(child_model)
                results[task_name] = score
            except Exception as e:
                results[task_name] = 0.0
                results[f"{task_name}_error"] = str(e)

        child_genotype.fitness_history["validation_results"] = sum(results.values()) / max(1, len(results))
        return results
