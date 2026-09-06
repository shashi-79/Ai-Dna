"""
Generic Asymmetric Layer-Depth Decoupled LoRA Instinct Fusion Engine.
Implements idea.md Sections 14, 20-25:
- Continuous Tensor Sigma-Interpolation Operator (Proj_Sigma, Section 24.3)
- LoRA Instinct-Filter SVD Low-Rank Subspace Extraction (Section 14)
- Generic N-Parent Asymmetric Layer-Depth Decoupling (Anchored Syntax, Knowledge, Algorithmic Execution)
- Generic Multi-Donor Gram-Schmidt Subspace Orthogonalization (Eliminates inter-donor collision)
- Statistical Outlier Vault Isolation (tau >= 6.0 sigma)
- Discrete Vocabulary Invariance (Section 24.2)
- Generic Multi-Parent Genotype & Physical Safetensors Model Fusion (N >= 1)
"""

import os
import copy
import json
import shutil
import math
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

from ..dna.structure import Genotype, DNAArchitecture, DNAInstinct, DNARouting, DNAMemory, DNALearning, DNAEvolution
from .compatibility import CompatibilityChecker, FunctionalNodeMatcher, CompatibilityScore


def project_sigma_energy_tensor(src: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Continuous Tensor Sigma-Interpolation Operator (idea.md Section 24.3).
    Projects weight tensors across disparate dimensional shapes while strictly
    conserving total Frobenius/singular energy.
    """
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


def extract_lora_instinct_components(
    weight: torch.Tensor,
    rank: int = 16,
) -> torch.Tensor:
    """
    Extracts low-rank singular instinct representation from a weight matrix (idea.md Section 14).
    Discards high-frequency parameter noise and preserves low-rank transferable structure.
    """
    if weight.dim() != 2:
        return weight
    w_f = weight.float()
    m, n = w_f.shape
    eff_rank = min(rank, m, n)
    try:
        U, S, Vh = torch.linalg.svd(w_f, full_matrices=False)
        U_r = U[:, :eff_rank]
        S_r = S[:eff_rank]
        Vh_r = Vh[:eff_rank, :]
        low_rank_instinct = (U_r * S_r.unsqueeze(0)) @ Vh_r
        return low_rank_instinct.to(dtype=weight.dtype)
    except Exception:
        return weight


def get_model_layer_count(weights: Dict[str, torch.Tensor]) -> int:
    """
    Dynamically discovers the total number of transformer layers from tensor keys.
    Supports arbitrarily deep architectures (e.g. 16, 22, 24, 32, 40, 64 layers).
    """
    max_idx = -1
    for k in weights.keys():
        if ".layers." in k:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    max_idx = max(max_idx, int(parts[i + 1]))
    return max_idx + 1 if max_idx >= 0 else 1


@dataclass
class DonorSpec:
    """Configuration for an individual donor foundation model in generic fusion."""
    path: str
    weight: float = 0.015
    specialization: str = "auto"  # "auto", "code", "knowledge", "math", "general"


def _infer_specialization(path: str) -> str:
    """Infers the specialization profile of a donor model from its path or identifier."""
    p_lower = path.lower()
    if any(k in p_lower for k in ["code", "coder", "smol", "algorithm", "python"]):
        return "code"
    if any(k in p_lower for k in ["math", "gsm", "reasoning", "deepseek-r1"]):
        return "math"
    if any(k in p_lower for k in ["llama", "tiny", "instruct", "chat", "world", "fact"]):
        return "knowledge"
    return "general"


def parse_donor_specs(
    donors: Optional[Union[str, List[Union[str, Dict[str, Any], DonorSpec]], Dict[str, float]]] = None,
    alpha: float = 0.015,
    donor1_dir: Optional[str] = None,
    donor2_dir: Optional[str] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
) -> List[DonorSpec]:
    """Parses and standardizes donor configurations across single/multiple formats."""
    donor_specs: List[DonorSpec] = []
    if donors is not None:
        if isinstance(donors, str):
            donor_specs.append(DonorSpec(path=donors, weight=alpha, specialization=_infer_specialization(donors)))
        elif isinstance(donors, dict):
            for d_path, d_w in donors.items():
                donor_specs.append(DonorSpec(path=d_path, weight=float(d_w), specialization=_infer_specialization(d_path)))
        elif isinstance(donors, list):
            for d in donors:
                if isinstance(d, str):
                    donor_specs.append(DonorSpec(path=d, weight=alpha, specialization=_infer_specialization(d)))
                elif isinstance(d, DonorSpec):
                    donor_specs.append(d)
                elif isinstance(d, dict):
                    p = d.get("path") or d.get("dir")
                    w = float(d.get("weight", alpha))
                    spec = d.get("specialization") or _infer_specialization(p)
                    donor_specs.append(DonorSpec(path=p, weight=w, specialization=spec))

    if donor1_dir:
        w1 = alpha1 if alpha1 is not None else alpha
        if not any(d.path == donor1_dir for d in donor_specs):
            donor_specs.append(DonorSpec(path=donor1_dir, weight=w1, specialization=_infer_specialization(donor1_dir)))
    if donor2_dir:
        w2 = alpha2 if alpha2 is not None else alpha
        if not any(d.path == donor2_dir for d in donor_specs):
            donor_specs.append(DonorSpec(path=donor2_dir, weight=w2, specialization=_infer_specialization(donor2_dir)))

    for d in donor_specs:
        if d.specialization == "auto":
            d.specialization = _infer_specialization(d.path)
    return donor_specs


def fuse_feedforward_layers_in_memory(
    prim_weights: Dict[str, torch.Tensor],
    donor_specs: List[DonorSpec],
    rank: int = 16,
    outlier_threshold: float = 6.0,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fuses feedforward layers across N donor models into primary backbone weights.
    Operates entirely in-memory with zero intermediate disk write overhead.
    Guarantees:
      - Shallow band (including Layer 0): strictly 100% frozen primary weights.
      - Middle band: knowledge expansion.
      - Deep band: algorithmic/reasoning execution.
      - Gram-Schmidt orthogonalization: prevents inter-donor subspace collision.
      - Outlier Vault (tau >= 6.0 sigma): isolates sensitive parameters.
      - Frobenius conservation: preserves singular variance per tensor.
    """
    primary_layer_count = get_model_layer_count(prim_weights)

    loaded_donors = []
    for idx, d_spec in enumerate(donor_specs):
        print(f"  [Layer Fusion] Loading Donor {idx + 1}/{len(donor_specs)} [{d_spec.specialization.upper()}]: {d_spec.path} (weight={d_spec.weight}) ...")
        d_weights: Dict[str, torch.Tensor] = {}
        st_file = os.path.join(d_spec.path, "model.safetensors")
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                d_weights[k] = f.get_tensor(k)
        d_layers = get_model_layer_count(d_weights)
        print(f"    Detected {d_layers} layers for donor {idx + 1}.")
        loaded_donors.append({
            "spec": d_spec,
            "weights": d_weights,
            "layers": d_layers,
        })

    shallow_bound = max(1, int(round(0.25 * primary_layer_count)))
    deep_bound = max(shallow_bound + 1, int(round(0.67 * primary_layer_count)))
    print(f"  Depth Decoupling Boundaries: Shallow (0..{shallow_bound - 1}) | Middle ({shallow_bound}..{deep_bound - 1}) | Deep ({deep_bound}..{primary_layer_count - 1})")

    fused_weights: Dict[str, torch.Tensor] = {}
    modified_count = 0

    for k, w_prim in prim_weights.items():
        if any(token in k for token in ["embed_tokens", "lm_head", "wte", "layernorm", "norm"]):
            fused_weights[k] = w_prim.clone()
            continue

        if not k.startswith("model.layers.") or w_prim.dim() != 2:
            fused_weights[k] = w_prim.clone()
            continue

        parts = k.split(".")
        try:
            layer_idx = int(parts[2])
            sub_key = ".".join(parts[3:])
        except Exception:
            fused_weights[k] = w_prim.clone()
            continue

        w_prim_f = w_prim.float()

        active_donors = []
        for d_info in loaded_donors:
            d_spec: DonorSpec = d_info["spec"]
            d_weights: Dict[str, torch.Tensor] = d_info["weights"]
            d_layers = d_info["layers"]
            base_alpha = d_spec.weight
            spec_type = d_spec.specialization

            if layer_idx < shallow_bound:
                eff_alpha = 0.0
            elif layer_idx < deep_bound:
                if spec_type in ["knowledge", "general"]:
                    eff_alpha = base_alpha
                elif spec_type in ["code", "math"]:
                    eff_alpha = base_alpha * 0.33
                else:
                    eff_alpha = base_alpha
            else:
                if spec_type in ["code", "math"]:
                    eff_alpha = base_alpha * 1.25
                elif spec_type == "knowledge":
                    eff_alpha = 0.0
                else:
                    eff_alpha = base_alpha * 0.5

            if eff_alpha <= 0.0:
                continue

            mapped_l = min(int(layer_idx * d_layers / primary_layer_count), d_layers - 1)
            donor_key = f"model.layers.{mapped_l}.{sub_key}"

            if donor_key in d_weights:
                w_d = d_weights[donor_key]
                inst_d = extract_lora_instinct_components(w_d, rank=rank)
                proj_d = project_sigma_energy_tensor(inst_d, w_prim.shape).float()
                active_donors.append((eff_alpha, proj_d))

        if not active_donors:
            fused_weights[k] = w_prim.clone()
            continue

        ortho_deltas = []
        eff_weights = []
        for eff_a, delta in active_donors:
            u = delta.clone()
            for prev_u in ortho_deltas:
                inner_prod = (u * prev_u).sum()
                norm_sq = (prev_u * prev_u).sum() + 1e-8
                u = u - (inner_prod / norm_sq) * prev_u
            if torch.norm(u) > 1e-6:
                ortho_deltas.append(u)
                eff_weights.append(eff_a)

        total_donor_delta = sum(w * u for w, u in zip(eff_weights, ortho_deltas))

        mu = w_prim_f.mean()
        std = w_prim_f.std() + 1e-8
        outlier_mask = (w_prim_f - mu).abs() > (outlier_threshold * std)

        w_fused = w_prim_f + total_donor_delta
        orig_norm = torch.norm(w_prim_f)
        new_norm = torch.norm(w_fused) + 1e-8
        w_fused = w_fused * (orig_norm / new_norm)

        w_fused[outlier_mask] = w_prim_f[outlier_mask]
        fused_weights[k] = w_fused.to(dtype=w_prim.dtype)
        modified_count += 1

    depth_meta = {
        "primary_layers": primary_layer_count,
        "shallow_bound": shallow_bound,
        "deep_bound": deep_bound,
        "tensors_fused": modified_count,
        "total_tensors": len(prim_weights),
    }
    return fused_weights, loaded_donors, depth_meta


def create_asymmetric_depth_fused_model(
    primary_dir: str,
    donors: Optional[Union[str, List[Union[str, Dict[str, Any], DonorSpec]], Dict[str, float]]] = None,
    output_dir: str = "my_llm_folder",
    rank: int = 16,
    alpha: float = 0.015,
    outlier_threshold: float = 6.0,
    # Backward compatibility arguments:
    donor1_dir: Optional[str] = None,
    donor2_dir: Optional[str] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generic Asymmetric Layer-Depth Decoupled LoRA Instinct Fusion Engine.
    Fuses arbitrary N donor models (N >= 1, e.g. 1, 2, 3, 5, ...) into the primary backbone:
      - Automatically detects layer depths across differing architectures without hardcoded values.
      - Shallow Band (first ~25% layers): 100% frozen primary backbone (anchors syntax & RoPE).
      - Middle Band (~25% to ~67% layers): Knowledge & conversational expansion.
      - Deep Band (~67% to 100% layers): Algorithmic & reasoning execution (boosted code/math, general donors attenuated).
      - Generic Gram-Schmidt subspace orthogonalization eliminates inter-donor interference across low-rank singular vectors.
      - Outlier Vault (tau >= 6.0 sigma) protects fragile salient circuits.
      - Discrete vocabulary invariance guarantees zero tokenizer mismatch.
    """
    os.makedirs(output_dir, exist_ok=True)
    donor_specs = parse_donor_specs(donors, alpha=alpha, donor1_dir=donor1_dir, donor2_dir=donor2_dir, alpha1=alpha1, alpha2=alpha2)

    print(f"[Generic LoRA Fusion] Loading Primary Backbone: {primary_dir} ...")
    prim_weights: Dict[str, torch.Tensor] = {}
    with safe_open(os.path.join(primary_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            prim_weights[k] = f.get_tensor(k)

    primary_layer_count = get_model_layer_count(prim_weights)
    print(f"  Primary architecture detected: {primary_layer_count} transformer layers.")

    fused_weights, loaded_donors, depth_meta = fuse_feedforward_layers_in_memory(
        prim_weights=prim_weights,
        donor_specs=donor_specs,
        rank=rank,
        outlier_threshold=outlier_threshold,
    )

    out_st = os.path.join(output_dir, "model.safetensors")
    safetensors_save_file(fused_weights, out_st)
    print(f"[Generic LoRA Fusion Complete] Saved {out_st} ({depth_meta['tensors_fused']}/{depth_meta['total_tensors']} tensors fused across {len(loaded_donors)} donors).")

    for fname in os.listdir(primary_dir):
        if fname != "model.safetensors":
            src_f = os.path.join(primary_dir, fname)
            dst_f = os.path.join(output_dir, fname)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, dst_f)

    manifest = {
        "fusion_type": "Generic Asymmetric Layer-Depth Decoupled LoRA Instinct Fusion",
        "primary_backbone": primary_dir,
        "primary_layers": primary_layer_count,
        "donors": [
            {
                "path": d["spec"].path,
                "layers": d["layers"],
                "weight": d["spec"].weight,
                "specialization": d["spec"].specialization,
            }
            for d in loaded_donors
        ],
        "depth_bands": {
            "shallow_frozen": f"0..{depth_meta['shallow_bound'] - 1}",
            "middle_knowledge": f"{depth_meta['shallow_bound']}..{depth_meta['deep_bound'] - 1}",
            "deep_execution": f"{depth_meta['deep_bound']}..{primary_layer_count - 1}",
        },
        "rank": rank,
        "outlier_threshold": outlier_threshold,
        "tensors_fused": depth_meta["tensors_fused"],
        "total_tensors": depth_meta["total_tensors"],
    }
    with open(os.path.join(output_dir, "fusion_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


class AsymmetricLayerDepthLoRAFusion:
    """
    Canonical Generic Asymmetric Layer-Depth Decoupled LoRA Instinct Fusion Engine.
    Fuses multiple compatible parent Genotypes (N >= 2) into a unified offspring Genotype.
    Uses Innovation IDs for shared nodes, functional similarity for alignment,
    and SVD low-rank energy with Outlier Vault protection for disjoint node inheritance.
    """
    def __init__(
        self,
        min_compatibility: float = 0.6,
        functional_match_threshold: float = 0.6,
        enable_functional_matching: bool = False,
        enable_residual_blend: bool = False,
        blend_alpha: float = 0.15,
        outlier_threshold: float = 6.0,
        rank: int = 16,
    ):
        self.min_compatibility = min_compatibility
        self.enable_functional_matching = enable_functional_matching
        self.node_matcher = FunctionalNodeMatcher(match_threshold=functional_match_threshold)
        self.enable_residual_blend = enable_residual_blend
        self.blend_alpha = blend_alpha
        self.outlier_threshold = outlier_threshold
        self.rank = rank

    @staticmethod
    def _compute_svd_energy(tensor: torch.Tensor) -> float:
        """
        Computes total singular energy Sigma = sum(s_i^2) for a parameter tensor.
        By Frobenius theorem, sum(s_i^2) is mathematically identical to sum(W_ij^2).
        Calculated in O(N) with zero SVD calls.
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
        def parent_capacity_score(p: Genotype) -> float:
            return float(sum(t.numel() for t in p.dna_instinct.genetic_parameters.values()))

        best_parent_idx = max(range(n), key=lambda i: parent_capacity_score(parents[i]))
        p_primary = parents[best_parent_idx]

        child_arch = copy.deepcopy(p_primary.dna_architecture)
        child_routing = copy.deepcopy(p_primary.dna_routing)
        child_memory = copy.deepcopy(p_primary.dna_memory)
        child_learning = copy.deepcopy(p_primary.dna_learning)
        child_evolution = copy.deepcopy(p_primary.dna_evolution)

        # 3. Classify parameter keys into shared vs disjoint using Innovation IDs
        all_param_keys = set()
        for p in parents:
            all_param_keys.update(p.dna_instinct.genetic_parameters.keys())

        child_genetic_params = {}

        # 4. Instinct Inheritance
        for key in all_param_keys:
            # Preserve discrete vocabulary embeddings directly from primary backbone (§24.2)
            is_embedding_key = any(emb in key for emb in ["embed_tokens", "lm_head", "wte"])
            if is_embedding_key and key in p_primary.dna_instinct.genetic_parameters:
                child_genetic_params[key] = p_primary.dna_instinct.genetic_parameters[key].clone()
                continue

            # Find all parents owning this instinct
            owning_tensors = []
            donor_tensors = []
            for p in parents:
                if key in p.dna_instinct.genetic_parameters:
                    t = p.dna_instinct.genetic_parameters[key]
                    owning_tensors.append(t)
                    if p is not p_primary:
                        donor_tensors.append(t)

            if not owning_tensors:
                continue

            primary_param = p_primary.dna_instinct.genetic_parameters.get(key)

            if len(owning_tensors) == 1:
                # Non-overlapping instinct: inherit directly with shape projection
                t_val = owning_tensors[0]
                if primary_param is not None and t_val.shape != primary_param.shape:
                    child_genetic_params[key] = project_sigma_energy_tensor(t_val, primary_param.shape)
                else:
                    child_genetic_params[key] = t_val.clone()
            else:
                # Overlapping instinct:
                if self.enable_residual_blend and primary_param is not None and len(donor_tensors) > 0:
                    w_prim = primary_param.float()
                    projected_donors = [project_sigma_energy_tensor(d, primary_param.shape).float() for d in donor_tensors]
                    
                    # Generic Gram-Schmidt orthogonalization across multiple donor instincts
                    ortho_donors = []
                    for d in projected_donors:
                        u = d.clone()
                        for prev_u in ortho_donors:
                            inner = (u * prev_u).sum()
                            denom = (prev_u * prev_u).sum() + 1e-8
                            u = u - (inner / denom) * prev_u
                        if torch.norm(u) > 1e-6:
                            ortho_donors.append(u)

                    total_donor = sum(ortho_donors) if ortho_donors else projected_donors[0]

                    # Statistical Outlier Vault Isolation (tau = 6.0 sigma)
                    mu = w_prim.mean()
                    std = w_prim.std() + 1e-8
                    outlier_mask = (w_prim - mu).abs() > (self.outlier_threshold * std)

                    # Smooth Continuous Blending
                    alpha = self.blend_alpha
                    w_blended = (1.0 - alpha) * w_prim + alpha * total_donor

                    # Energy preservation scaling factor
                    orig_norm = torch.norm(w_prim)
                    new_norm = torch.norm(w_blended) + 1e-8
                    w_blended = w_blended * (orig_norm / new_norm)

                    # Restore Outlier Vault
                    w_blended[outlier_mask] = w_prim[outlier_mask]

                    child_genetic_params[key] = w_blended.to(dtype=primary_param.dtype)
                else:
                    # Winner-Take-All SVD energy selection
                    if primary_param is not None:
                        compat = [t for t in owning_tensors if t.shape == primary_param.shape]
                        candidates = compat if compat else owning_tensors
                    else:
                        candidates = owning_tensors

                    best_tensor = max(candidates, key=lambda t: self._compute_svd_energy(t))
                    if primary_param is not None and best_tensor.shape != primary_param.shape:
                        child_genetic_params[key] = project_sigma_energy_tensor(best_tensor, primary_param.shape)
                    else:
                        child_genetic_params[key] = best_tensor.clone()

        child_instinct = DNAInstinct(
            cppn_hidden_dim=p_primary.dna_instinct.cppn_hidden_dim,
            cppn_layers=p_primary.dna_instinct.cppn_layers,
            genetic_parameters=child_genetic_params,
            singular_energy_threshold=p_primary.dna_instinct.singular_energy_threshold,
            instinct_rank_ratio=p_primary.dna_instinct.instinct_rank_ratio,
        )

        # 5. Merge Innovation IDs map
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


# Canonical alias
MultiParentFusion = AsymmetricLayerDepthLoRAFusion


def build_recurrent_depth_model(
    primary_dir: str,
    output_dir: str,
    donors: Optional[Union[str, List[Union[str, Dict[str, Any], DonorSpec]], Dict[str, float]]] = None,
    rank: int = 2048,
    donor_rank: int = 16,
    alpha: float = 0.015,
    outlier_threshold: float = 6.0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Converts a standard multi-layer foundation model (or a multi-parent fused model)
    into a canonical Type 7 Recurrent Depth (Looped Transformer) model.

    Two-Stage Layer-First Recurrent Fusion Advancement:
      If `donors` is supplied, normal models are first fused across their feedforward
      layers in-memory via Asymmetric Layer-Depth Decoupled LoRA Fusion (§14, 20-25).
      Layer 0 remains strictly 100% frozen primary (preserving the syntax/RoPE anchor).
      The resulting fused layers are then consolidated into the Type 7 recurrent cell.
      This eliminates the catastrophic hidden-state divergence observed when injecting
      disparate donor adapters directly inside the recurrent execution loop.

    Type 7 Architectural Specification:
      - Base Anchor: Exact Layer 0 weights (zero SVD distortion at initial step).
      - Step LoRA Residuals: Full-rank step residuals (A_t, B_t) for each step t in [0, L-1].
        eff_r = min(rank, m, n) (caps at mathematical rank limit min(4864, 896) = 896).
      - Energy Conservation: Exact Frobenius norm rescaling:
          scale = ||Delta_t||_F / ||A_t B_t||_F
      - LayerNorm Residuals: 1D normalized activation deltas for input and post-attention RMSNorm.
      - Zero Step Embeddings: Zero-initialized temporal embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)

    print(f"\n[Recurrent Fusion Engine] Converting {primary_dir} -> {output_dir}")
    print(f"  Architecture: Type 7 Canonical Recurrent Depth (Layer 0 Anchor + Step-Modulated LoRA)")
    print(f"  Max Rank:     {rank} (Dynamically capped at min(m, n))")

    # 1. Load primary source weights
    src_st = os.path.join(primary_dir, "model.safetensors")
    prim_weights: Dict[str, torch.Tensor] = {}
    with safe_open(src_st, framework="pt", device="cpu") as f:
        for k in f.keys():
            prim_weights[k] = f.get_tensor(k)

    # 2. Stage 1: Optional Layer-First Fusion Across Normal Models
    donor_specs = parse_donor_specs(donors, alpha=alpha) if donors is not None else []
    loaded_donors = []
    depth_meta = {}

    if donor_specs:
        print(f"  [Stage 1: Layer-First Fusion] Fusing {len(donor_specs)} donor models across feedforward layers...")
        source_weights, loaded_donors, depth_meta = fuse_feedforward_layers_in_memory(
            prim_weights=prim_weights,
            donor_specs=donor_specs,
            rank=donor_rank,
            outlier_threshold=outlier_threshold,
        )
        print(f"  [Stage 1 Complete] Layer fusion finished. Advancing to Stage 2 Recurrent Consolidation.")
    else:
        source_weights = prim_weights

    L = get_model_layer_count(source_weights)
    print(f"  Detected {L} physical layers in source architecture.")

    # 3. Discover layer subkeys (e.g., self_attn.q_proj.weight, mlp.gate_proj.weight, etc.)
    layer_subkeys = set()
    for k in source_weights.keys():
        if k.startswith("model.layers.0."):
            sub = k[len("model.layers.0."):]
            layer_subkeys.add(sub)

    # 4. Construct Recurrent Weights (Stage 2: Type 7 Consolidation)
    recurrent_weights: Dict[str, torch.Tensor] = {}

    # 4A. Retain global non-layer weights intact (§24.2 Discrete Vocabulary Invariance)
    for k, v in source_weights.items():
        if not k.startswith("model.layers."):
            recurrent_weights[k] = v.clone()

    # 4B. Consolidate Base Recurrent Cell (Layer 0 Exact Anchor)
    print("  Consolidating Base Recurrent Cell (Exact Layer 0 Anchor)...")
    base_tensors: Dict[str, torch.Tensor] = {}
    for sub in layer_subkeys:
        base_tensors[sub] = source_weights[f"model.layers.0.{sub}"].clone()
        recurrent_weights[f"model.layers.0.{sub}"] = base_tensors[sub]

    # 4C. Extract Step LoRA Residuals across all L iteration steps
    print(f"  Extracting Step Residuals across all {L} iteration steps with Frobenius Rescaling...")
    step_adapter_count = 0
    for t in range(L):
        for sub in layer_subkeys:
            w_orig = source_weights[f"model.layers.{t}.{sub}"].float()
            w_base = base_tensors[sub].float()

            if w_orig.dim() == 2 and w_orig.numel() > 1:
                delta = w_orig - w_base
                m, n = delta.shape
                eff_r = min(rank, m, n)
                try:
                    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
                    sqrt_s = torch.sqrt(torch.clamp(S[:eff_r], min=0.0))
                    
                    # Frobenius norm rescaling to conserve 100% of residual energy
                    delta_norm = delta.norm()
                    recon_norm = torch.sqrt((S[:eff_r]**2).sum())
                    scale = (delta_norm / (recon_norm + 1e-8)).item() if recon_norm > 1e-8 else 1.0
                    sqrt_scale = math.sqrt(scale)

                    A_t = (U[:, :eff_r] * (sqrt_s * sqrt_scale).unsqueeze(0)).contiguous()
                    B_t = ((sqrt_s * sqrt_scale).unsqueeze(1) * Vh[:eff_r, :]).contiguous()
                except Exception:
                    A_t = torch.zeros((m, eff_r), dtype=w_orig.dtype)
                    B_t = torch.zeros((eff_r, n), dtype=w_orig.dtype)

                recurrent_weights[f"model.step_adapters.{t}.{sub}.lora_A"] = A_t.to(dtype=w_orig.dtype)
                recurrent_weights[f"model.step_adapters.{t}.{sub}.lora_B"] = B_t.to(dtype=w_orig.dtype)
                step_adapter_count += 2
            else:
                delta_1d = (w_orig - w_base).contiguous()
                recurrent_weights[f"model.step_adapters.{t}.{sub}.delta_1d"] = delta_1d.to(dtype=w_orig.dtype)
                step_adapter_count += 1

    # 4D. Generate Iteration Step Embeddings (strictly zero-initialized)
    d_model = source_weights.get("model.embed_tokens.weight", torch.zeros(1, 896)).shape[1]
    step_embeds = torch.zeros(L, d_model)
    recurrent_weights["model.step_embeddings.weight"] = step_embeds.to(dtype=recurrent_weights["model.embed_tokens.weight"].dtype)

    # 5. Save Safetensors
    out_st = os.path.join(output_dir, "model.safetensors")
    safetensors_save_file(recurrent_weights, out_st)
    
    total_recurrent_params = sum(t.numel() for t in recurrent_weights.values())
    orig_params = sum(t.numel() for t in prim_weights.values())
    disk_size_mb = os.path.getsize(out_st) / (1024 * 1024)
    orig_disk_mb = os.path.getsize(src_st) / (1024 * 1024)
    param_delta_pct = ((total_recurrent_params - orig_params) / orig_params) * 100.0

    print(f"  [+] Saved {out_st}")
    print(f"      Parameters: {total_recurrent_params:,} (Original: {orig_params:,}, {param_delta_pct:+.1f}%)")
    print(f"      Disk Size:  {disk_size_mb:.2f} MB (Original: {orig_disk_mb:.2f} MB)")

    # 6. Copy and update config
    cfg_src = os.path.join(primary_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_src):
        with open(cfg_src, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    cfg["recurrent_depth"] = L
    cfg["recurrent_strategy"] = "type_7"
    cfg["anchor_method"] = "layer_0"
    cfg["recurrent_rank"] = rank
    cfg["architectures"] = ["RecurrentQwenForCausalLM"]
    if donor_specs:
        cfg["two_stage_layer_first_fusion"] = True
        cfg["donors_count"] = len(donor_specs)

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Copy tokenizer and other metadata
    for fname in os.listdir(primary_dir):
        if fname not in ["model.safetensors", "config.json"]:
            src_f = os.path.join(primary_dir, fname)
            dst_f = os.path.join(output_dir, fname)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, dst_f)

    manifest = {
        "status": "success",
        "fusion_type": "Two-Stage Layer-First Recurrent Fusion (Type 7)" if donor_specs else "Recurrent Depth (Type 7: Layer 0 Anchor + Full-Rank Step LoRA)",
        "two_stage_layer_first_fusion": bool(donor_specs),
        "strategy": "type_7",
        "anchor_method": "layer_0",
        "anchor_description": "Direct Layer 0 Exact Anchor (Syntactic Foundation)",
        "recurrent_depth": L,
        "max_rank": rank,
        "donors_fused": [
            {
                "path": d["spec"].path,
                "layers": d["layers"],
                "weight": d["spec"].weight,
                "specialization": d["spec"].specialization,
            }
            for d in loaded_donors
        ] if donor_specs else [],
        "total_parameters": total_recurrent_params,
        "original_parameters": orig_params,
        "parameter_delta_pct": param_delta_pct,
        "disk_size_mb": disk_size_mb,
        "original_disk_mb": orig_disk_mb,
        "recurrent_tensors": len(recurrent_weights),
        "step_adapter_tensors": step_adapter_count,
    }
    with open(os.path.join(output_dir, "recurrent_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Conversion Complete] {output_dir}\n")
    return manifest


# Canonical alias for Two-Stage Layer-First Recurrent Fusion
fuse_and_build_recurrent_model = build_recurrent_depth_model

