"""
DNA Compatibility Metrics and Functional Node Matching.
Validates structural and dimensional compatibility C(D_i, D_j) >= C_min before multi-parent fusion.
Implements functional node similarity (Section 22) for aligning independently evolved structures.
"""

import torch
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
from ..dna.structure import Genotype


@dataclass
class CompatibilityScore:
    is_compatible: bool
    overall_score: float
    c_arch: float
    c_dim: float
    c_modality: float
    reason: str = ""


class CompatibilityChecker:
    """
    Computes compatibility score C = (C_arch, C_dim, C_modality) between genotypes (Section 20).
    """

    @staticmethod
    def evaluate(g1: Genotype, g2: Genotype, min_score: float = 0.6) -> CompatibilityScore:
        a1, a2 = g1.dna_architecture, g2.dna_architecture

        # 1. Architecture compatibility: layer and head compatibility
        layer_diff = abs(a1.num_layers - a2.num_layers) / max(a1.num_layers, a2.num_layers)
        head_diff = abs(a1.num_heads - a2.num_heads) / max(a1.num_heads, a2.num_heads)
        c_arch = max(0.0, 1.0 - (layer_diff * 0.6 + head_diff * 0.4))

        # 2. Dimension compatibility: d_model and coord_dim
        dim_match = 1.0 if a1.d_model == a2.d_model else 0.0
        coord_match = 1.0 if a1.coord_dim == a2.coord_dim else 0.0
        c_dim = 0.7 * dim_match + 0.3 * coord_match

        # 3. Modality compatibility
        c_modality = 1.0 if a1.vocab_size == a2.vocab_size else 0.5

        overall = 0.4 * c_arch + 0.4 * c_dim + 0.2 * c_modality
        is_compat = overall >= min_score and dim_match > 0.0

        reason = "Compatible" if is_compat else f"Score {overall:.2f} < threshold {min_score:.2f} or dimension mismatch"
        return CompatibilityScore(
            is_compatible=is_compat,
            overall_score=overall,
            c_arch=c_arch,
            c_dim=c_dim,
            c_modality=c_modality,
            reason=reason,
        )


@dataclass
class NodeSimilarityScore:
    """Result of functional similarity comparison between two genetic parameter nodes."""
    similarity: float
    sim_type: float
    sim_shape: float
    sim_magnitude: float
    sim_structure: float
    sim_cosine: float


class FunctionalNodeMatcher:
    """
    Implements functional node matching (Section 22) for aligning independently evolved nodes
    that don't share Innovation IDs.

    Sim(n_A, n_B) = w1*Sim_type + w2*Sim_input + w3*Sim_output + w4*Sim_coordinate + w5*Sim_behavior

    Adapted to work with CPPN genetic parameter tensors rather than topology nodes.
    """

    def __init__(
        self,
        w_type: float = 0.15,
        w_shape: float = 0.25,
        w_magnitude: float = 0.15,
        w_structure: float = 0.20,
        w_cosine: float = 0.25,
        match_threshold: float = 0.6,
    ):
        self.w_type = w_type
        self.w_shape = w_shape
        self.w_magnitude = w_magnitude
        self.w_structure = w_structure
        self.w_cosine = w_cosine
        self.match_threshold = match_threshold

    def compute_node_similarity(
        self,
        name_a: str, tensor_a: torch.Tensor,
        name_b: str, tensor_b: torch.Tensor,
    ) -> NodeSimilarityScore:
        """
        Computes functional similarity between two genetic parameter tensors.
        """
        # 1. Type similarity — do the parameter names suggest same role?
        # (e.g., both are "backbone.2.weight" vs "backbone.3.weight")
        sim_type = self._name_similarity(name_a, name_b)

        # 2. Shape similarity — same shape?
        sim_shape = 1.0 if tensor_a.shape == tensor_b.shape else 0.0

        # 3. Magnitude similarity — similar Frobenius norms?
        norm_a = torch.linalg.norm(tensor_a.float()).item()
        norm_b = torch.linalg.norm(tensor_b.float()).item()
        max_norm = max(norm_a, norm_b, 1e-9)
        sim_magnitude = 1.0 - abs(norm_a - norm_b) / max_norm

        # 4. Structural similarity — similar singular value distribution?
        sim_structure = 0.0
        if tensor_a.shape == tensor_b.shape and tensor_a.ndim >= 2:
            sim_structure = self._svd_structure_similarity(tensor_a, tensor_b)

        # 5. Cosine similarity — if same shape, measure directional alignment
        sim_cosine = 0.0
        if tensor_a.shape == tensor_b.shape:
            flat_a = tensor_a.float().flatten()
            flat_b = tensor_b.float().flatten()
            dot = torch.dot(flat_a, flat_b)
            norms = torch.linalg.norm(flat_a) * torch.linalg.norm(flat_b)
            if norms > 1e-9:
                sim_cosine = max(0.0, (dot / norms).item())

        similarity = (
            self.w_type * sim_type
            + self.w_shape * sim_shape
            + self.w_magnitude * sim_magnitude
            + self.w_structure * sim_structure
            + self.w_cosine * sim_cosine
        )

        return NodeSimilarityScore(
            similarity=similarity,
            sim_type=sim_type,
            sim_shape=sim_shape,
            sim_magnitude=sim_magnitude,
            sim_structure=sim_structure,
            sim_cosine=sim_cosine,
        )

    def _name_similarity(self, name_a: str, name_b: str) -> float:
        """Computes structural similarity between parameter names."""
        if name_a == name_b:
            return 1.0
        # Parse layer type from name (e.g., "backbone.0.weight" -> "backbone.weight")
        parts_a = name_a.split(".")
        parts_b = name_b.split(".")
        # Compare non-numeric parts
        type_parts_a = [p for p in parts_a if not p.isdigit()]
        type_parts_b = [p for p in parts_b if not p.isdigit()]
        if type_parts_a == type_parts_b:
            return 0.8
        # Partial match
        common = len(set(type_parts_a) & set(type_parts_b))
        total = max(len(set(type_parts_a) | set(type_parts_b)), 1)
        return 0.5 * (common / total)

    @staticmethod
    def _svd_structure_similarity(t_a: torch.Tensor, t_b: torch.Tensor) -> float:
        """Compares singular value distributions as a measure of structural similarity."""
        try:
            a_2d = t_a.reshape(t_a.shape[0], -1).float()
            b_2d = t_b.reshape(t_b.shape[0], -1).float()
            _, s_a, _ = torch.linalg.svd(a_2d, full_matrices=False)
            _, s_b, _ = torch.linalg.svd(b_2d, full_matrices=False)

            # Normalize singular values
            s_a_norm = s_a / (s_a.sum() + 1e-9)
            s_b_norm = s_b / (s_b.sum() + 1e-9)

            # Truncate to same length
            min_len = min(len(s_a_norm), len(s_b_norm))
            s_a_norm = s_a_norm[:min_len]
            s_b_norm = s_b_norm[:min_len]

            # 1 - L1 distance between normalized spectra
            dist = torch.abs(s_a_norm - s_b_norm).sum().item()
            return max(0.0, 1.0 - dist)
        except Exception:
            return 0.0

    def find_best_matches(
        self,
        params_a: Dict[str, torch.Tensor],
        params_b: Dict[str, torch.Tensor],
    ) -> Dict[str, Tuple[str, float]]:
        """
        For each key in params_a that is NOT in params_b (disjoint nodes),
        finds the best functional match in params_b.
        Returns: {key_a: (best_key_b, similarity_score)}
        """
        matches = {}
        disjoint_a = set(params_a.keys()) - set(params_b.keys())
        available_b = set(params_b.keys()) - set(params_a.keys())

        for key_a in disjoint_a:
            best_key = None
            best_score = 0.0
            for key_b in available_b:
                score = self.compute_node_similarity(
                    key_a, params_a[key_a],
                    key_b, params_b[key_b],
                )
                if score.similarity > best_score:
                    best_score = score.similarity
                    best_key = key_b

            if best_key is not None and best_score >= self.match_threshold:
                matches[key_a] = (best_key, best_score)

        return matches
