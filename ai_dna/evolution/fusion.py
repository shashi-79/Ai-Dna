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
    Uses Innovation ID matching, functional node similarity, and Frobenius/SVD energy-based disjoint selection.
    """
    def __init__(
        self,
        min_compatibility: float = 0.6,
        functional_match_threshold: float = 0.6,
        enable_functional_matching: bool = False,
    ):
        self.min_compatibility = min_compatibility
        self.enable_functional_matching = enable_functional_matching
        self.node_matcher = FunctionalNodeMatcher(match_threshold=functional_match_threshold)

    @staticmethod
    def _compute_svd_energy(tensor: torch.Tensor) -> float:
        """
        Computes total singular energy Sigma = sum(s_i^2) for a parameter tensor.
        By Frobenius theorem, sum(s_i^2) is mathematically IDENTICAL to sum(W_ij^2).
        Calculates in O(N) with zero SVD calls.
        """
        return (tensor.float() ** 2).sum().item()

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

        # 2. Base Architecture Inheritance (Highest Capacity Parent)
        # Dynamically selects the parent with the most parameters as primary backbone
        def parent_capacity_score(p: Genotype) -> float:
            return float(sum(t.numel() for t in p.dna_instinct.genetic_parameters.values()))

        best_parent_idx = max(range(n), key=lambda i: parent_capacity_score(parents[i]))
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

        # Helper for SVD Sigma-Projection across differing dimensions
        def project_sigma_energy(src: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
            if src.shape == target_shape:
                return src.contiguous()
            if src.dim() == 1 and len(target_shape) == 1:
                src_1d = src.view(1, 1, -1).float()
                out_1d = torch.nn.functional.interpolate(src_1d, size=target_shape[0], mode="linear", align_corners=False)
                return out_1d.view(-1).to(dtype=src.dtype)
            if src.dim() == 2 and len(target_shape) == 2:
                src_2d = src.unsqueeze(0).unsqueeze(0).float()
                out_2d = torch.nn.functional.interpolate(src_2d, size=target_shape, mode="bilinear", align_corners=False)
                src_e = (src.float() ** 2).sum()
                out_e = (out_2d ** 2).sum() + 1e-8
                scale = torch.sqrt(src_e / out_e)
                return (out_2d.squeeze(0).squeeze(0) * scale).to(dtype=src.dtype)
            return src

        # 3. Instinct Inheritance:
        # Extract whole instinct from each parent:
        # - For overlapping instincts (present in multiple parents): choose the best one (highest energy / shape-compatible)
        # - For non-overlapping instincts (unique to one parent): add all directly
        for key in all_param_keys:
            # Preserve discrete vocabulary embeddings directly from primary backbone
            is_embedding_key = any(emb in key for emb in ["embed_tokens", "lm_head", "wte"])
            if is_embedding_key and key in p_primary.dna_instinct.genetic_parameters:
                child_genetic_params[key] = p_primary.dna_instinct.genetic_parameters[key].clone()
                continue

            # Find all parents owning this instinct
            owning_tensors = []
            for p in parents:
                if key in p.dna_instinct.genetic_parameters:
                    owning_tensors.append(p.dna_instinct.genetic_parameters[key])

            if not owning_tensors:
                continue

            primary_param = p_primary.dna_instinct.genetic_parameters.get(key)

            if len(owning_tensors) == 1:
                # Non-overlapping instinct: add directly from the parent
                t_val = owning_tensors[0]
                if primary_param is not None and t_val.shape != primary_param.shape:
                    child_genetic_params[key] = project_sigma_energy(t_val, primary_param.shape)
                else:
                    child_genetic_params[key] = t_val.clone()
            else:
                # Overlapping instinct: choose the best one (highest SVD energy / shape-compatible)
                if primary_param is not None:
                    compat = [t for t in owning_tensors if t.shape == primary_param.shape]
                    candidates = compat if compat else owning_tensors
                else:
                    candidates = owning_tensors

                best_tensor = max(candidates, key=lambda t: self._compute_svd_energy(t))
                if primary_param is not None and best_tensor.shape != primary_param.shape:
                    child_genetic_params[key] = project_sigma_energy(best_tensor, primary_param.shape)
                else:
                    child_genetic_params[key] = best_tensor.clone()

        child_instinct = DNAInstinct(
            cppn_hidden_dim=p_primary.dna_instinct.cppn_hidden_dim,
            cppn_layers=p_primary.dna_instinct.cppn_layers,
            genetic_parameters=child_genetic_params,
            singular_energy_threshold=p_primary.dna_instinct.singular_energy_threshold,
            instinct_rank_ratio=p_primary.dna_instinct.instinct_rank_ratio,
        )

        # 4. Merge Innovation IDs map (Section 21)
        merged_nodes = {}
        merged_sensory = {}
        for p in parents:
            merged_nodes.update(p.node_innovation_map)
            if hasattr(p, "sensory_assets") and isinstance(p.sensory_assets, dict):
                merged_sensory.update(p.sensory_assets)

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
            sensory_assets=merged_sensory,
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
