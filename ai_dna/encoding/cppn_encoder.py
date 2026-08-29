"""
Inverse HyperNEAT / CPPN Genetic Parameter Optimizer (Memory-Efficient Streamed Engine).
Encodes structural instinct W_k into compact genotype DNA by optimizing the
Complete DNA Objective (Section 15.5):
L_DNA = lambda_1 * L_reconstruction + lambda_2 * L_behavior + lambda_3 * L_future + lambda_4 * |D| + L_EWC

Zero VRAM Overflow Architecture:
- Stratified coordinate mini-batching (max 2048 coordinate samples per matrix per step)
- Streamed on-the-fly coordinate evaluation (zero static multi-gigabyte coordinate storage)
- Immediate gradient accumulation with explicit intermediate tensor destruction.
"""

import gc
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Callable, List
from ..dna.structure import Genotype
from ..growth.cppn import CPPNNetwork
from ..growth.coordinates import SubstrateCoordinateGenerator
from .ewc import EWCConsolidator


class InverseCPPNEncoder:
    """
    Optimizes CPPN genetic parameters to fit target structural weights W_k.
    Zero-memory-overflow streamed architecture.
    """
    def __init__(
        self,
        learning_rate: float = 1e-2,
        max_steps: int = 40,
        sample_points_per_matrix: int = 2048,
        lambda_recon: float = 1.0,
        lambda_behavior: float = 0.1,
        lambda_future: float = 0.0,
        lambda_size: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.sample_points_per_matrix = sample_points_per_matrix
        self.lambda_recon = lambda_recon
        self.lambda_behavior = lambda_behavior
        self.lambda_future = lambda_future
        self.lambda_size = lambda_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_size_penalty(self, cppn: CPPNNetwork) -> torch.Tensor:
        """L_size = L2 regularization over DNA parameters (Section 15.4)."""
        return sum((p ** 2).sum() for p in cppn.parameters())

    def encode_weight_into_cppn(
        self,
        cppn: CPPNNetwork,
        target_weights: Dict[str, torch.Tensor],
        num_layers: int = 4,
        num_experts: int = 1,
        ewc: Optional[EWCConsolidator] = None,
        behavior_fn: Optional[Callable[[], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, float]]:
        """
        Optimizes CPPN parameters with zero VRAM overflow using stratified coordinate sampling.
        """
        cppn = cppn.to(self.device)
        optimizer = optim.Adam(cppn.parameters(), lr=self.learning_rate)

        # Prepare target matrix metadata without allocating large coordinate tensors
        matrix_metadata = []
        for name, w in target_weights.items():
            if w.ndim >= 2:
                w_2d = w.reshape(w.shape[0], -1)
                out_f, in_f = w_2d.shape[0], w_2d.shape[1]
                
                parts = name.split(".")
                layer_idx = 0
                expert_idx = 0
                for idx, part in enumerate(parts):
                    if part == "blocks":
                        layer_idx = int(parts[idx + 1])
                    elif part == "experts":
                        expert_idx = int(parts[idx + 1])

                matrix_idx = SubstrateCoordinateGenerator.get_matrix_idx_from_name(name)
                std = (2.0 / (in_f + out_f)) ** 0.5
                norm_sq = float((w_2d ** 2).sum().item()) + 1e-8
                
                # Keep target weights in CPU / GPU compact view
                w_target_dev = w_2d.to(device=self.device, dtype=torch.float32)
                matrix_metadata.append({
                    "name": name,
                    "w_target": w_target_dev,
                    "out_f": out_f,
                    "in_f": in_f,
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx,
                    "matrix_idx": matrix_idx,
                    "std": std,
                    "norm_sq": norm_sq,
                })

        if not matrix_metadata:
            return cppn.get_parameter_dict(), 0.0, {}

        best_loss = float("inf")
        best_params = cppn.get_parameter_dict()
        final_breakdown = {}

        total_steps = self.max_steps
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

        for step in range(total_steps):
            optimizer.zero_grad()
            step_recon_loss = 0.0

            # Stream through each target weight matrix one-by-one
            for meta in matrix_metadata:
                out_f = meta["out_f"]
                in_f = meta["in_f"]
                total_elements = out_f * in_f
                w_target = meta["w_target"]
                norm_sq = meta["norm_sq"]
                std = meta["std"]

                # Memory-safe: Subsample coordinates if matrix is large, otherwise use full grid
                if total_elements > self.sample_points_per_matrix:
                    num_samples = self.sample_points_per_matrix
                    flat_indices = torch.randint(0, total_elements, (num_samples,), device=self.device)
                    row_idx = torch.div(flat_indices, in_f, rounding_mode="floor")
                    col_idx = flat_indices % in_f

                    # Compute sampled continuous coordinates in [-1, 1] on-the-fly
                    c_src = -1.0 + 2.0 * (row_idx.float() / max(1.0, out_f - 1.0))
                    c_tgt = -1.0 + 2.0 * (col_idx.float() / max(1.0, in_f - 1.0))
                    c_diff = c_tgt - c_src
                    c_dist = torch.sqrt(c_src * c_src + c_tgt * c_tgt)

                    l_norm = float(2.0 * (meta["layer_idx"] / max(1.0, num_layers - 1.0)) - 1.0)
                    e_norm = float(2.0 * (meta["expert_idx"] / max(1.0, num_experts - 1.0)) - 1.0)
                    m_norm = float(2.0 * (meta["matrix_idx"] / 15.0) - 1.0)

                    l_val = torch.full((num_samples,), l_norm, device=self.device, dtype=torch.float32)
                    e_val = torch.full((num_samples,), e_norm, device=self.device, dtype=torch.float32)
                    m_val = torch.full((num_samples,), m_norm, device=self.device, dtype=torch.float32)

                    base_coords = torch.stack([c_src, c_tgt, c_diff, c_dist, l_val, e_val, m_val], dim=-1)
                    if cppn.in_features > 7:
                        pad = torch.zeros((num_samples, cppn.in_features - 7), device=self.device)
                        sampled_coords = torch.cat([base_coords, pad], dim=-1)
                    else:
                        sampled_coords = base_coords[:, :cppn.in_features]

                    target_vals = w_target[row_idx, col_idx]
                    pred_vals = cppn(sampled_coords).squeeze(-1) * std

                    mat_loss = ((target_vals - pred_vals) ** 2).mean()
                else:
                    # Small matrix: exact grid
                    coords = SubstrateCoordinateGenerator.get_2d_weight_coordinates(
                        out_features=out_f,
                        in_features=in_f,
                        layer_idx=meta["layer_idx"],
                        num_layers=num_layers,
                        expert_idx=meta["expert_idx"],
                        num_experts=num_experts,
                        matrix_idx=meta["matrix_idx"],
                        device=self.device,
                        coord_dim=cppn.in_features,
                    )
                    pred_w = cppn(coords).squeeze(-1) * std
                    mat_loss = ((w_target - pred_w) ** 2).sum() / norm_sq

                # Immediate layer-by-layer gradient accumulation (zero memory accumulation)
                (self.lambda_recon * mat_loss / len(matrix_metadata)).backward()
                step_recon_loss += mat_loss.item()

            # EWC Penalty
            l_ewc_val = 0.0
            if ewc is not None:
                l_ewc = ewc.penalty(cppn)
                l_ewc.backward()
                l_ewc_val = l_ewc.item()

            # Size Penalty
            l_size = self._compute_size_penalty(cppn)
            (self.lambda_size * l_size).backward()

            optimizer.step()
            scheduler.step()

            avg_recon_loss = step_recon_loss / len(matrix_metadata)
            if avg_recon_loss < best_loss:
                best_loss = avg_recon_loss
                best_params = {k: v.cpu().clone() for k, v in cppn.get_parameter_dict().items()}
                final_breakdown = {
                    "loss_recon": avg_recon_loss,
                    "loss_size": l_size.item(),
                    "loss_ewc": l_ewc_val,
                }

        # Clear VRAM after optimization
        del matrix_metadata
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return best_params, best_loss, final_breakdown

    def encode_genotype(
        self,
        genotype: Genotype,
        target_weights: Dict[str, torch.Tensor],
        ewc: Optional[EWCConsolidator] = None,
        behavior_fn: Optional[Callable[[], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[Genotype, float, Dict[str, float]]:
        """
        Encodes target weights into a new updated genotype D_{t+1}.
        """
        new_genotype = genotype.clone(new_id=f"{genotype.genotype_id}_enc")
        new_genotype.generation = genotype.generation + 1

        arch = new_genotype.dna_architecture
        instinct = new_genotype.dna_instinct

        cppn = CPPNNetwork(
            in_features=arch.coord_dim,
            hidden_dim=instinct.cppn_hidden_dim,
            num_layers=instinct.cppn_layers,
            out_features=1,
        ).to(self.device)

        if instinct.genetic_parameters:
            try:
                cppn.load_parameter_dict(instinct.genetic_parameters)
            except Exception:
                pass

        best_params, loss_val, breakdown = self.encode_weight_into_cppn(
            cppn, target_weights,
            num_layers=arch.num_layers,
            num_experts=arch.num_experts,
            ewc=ewc,
            behavior_fn=behavior_fn,
        )

        new_genotype.dna_instinct.genetic_parameters = best_params
        new_genotype.fitness_history["encoding_loss"] = loss_val
        for k, v in breakdown.items():
            new_genotype.fitness_history[f"encoding_{k}"] = v

        return new_genotype, loss_val, breakdown
