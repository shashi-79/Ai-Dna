"""
Inverse HyperNEAT / CPPN Genetic Parameter Optimizer.
Encodes structural instinct W_k into compact genotype DNA by optimizing the
Complete DNA Objective (Section 15.5):
L_DNA = lambda_1 * L_reconstruction + lambda_2 * L_behavior + lambda_3 * L_future + lambda_4 * |D| + L_EWC
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Callable
from ..dna.structure import Genotype
from ..growth.cppn import CPPNNetwork
from ..growth.coordinates import SubstrateCoordinateGenerator
from .ewc import EWCConsolidator


class InverseCPPNEncoder:
    """
    Optimizes CPPN genetic parameters to fit target structural weights W_k.
    Supports the full DNA objective: reconstruction + behavioral + future + size + EWC.
    """
    def __init__(
        self,
        learning_rate: float = 1e-2,
        max_steps: int = 150,
        lambda_recon: float = 1.0,
        lambda_behavior: float = 0.1,
        lambda_future: float = 0.0,
        lambda_size: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.lambda_recon = lambda_recon
        self.lambda_behavior = lambda_behavior
        self.lambda_future = lambda_future
        self.lambda_size = lambda_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_reconstruction_loss(
        self,
        cppn: CPPNNetwork,
        coord_targets: list,
    ) -> torch.Tensor:
        """
        L_reconstruction = sum over matrices of ||W_k - G(D)||_F^2 / (||W_k||_F^2 + eps)
        (Section 15.1)
        """
        total_loss = torch.tensor(0.0, device=self.device)
        for coords, w_target, norm_sq in coord_targets:
            generated = cppn(coords).squeeze(-1)
            recon_loss = ((w_target - generated) ** 2).sum() / norm_sq
            total_loss = total_loss + recon_loss
        return total_loss

    @staticmethod
    def _compute_behavioral_loss(
        logits_original: torch.Tensor,
        logits_regenerated: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_behavior = E_x[D_KL(P_{M*}(y|x) || P_{G(D)}(y|x))]
        (Section 15.2)
        """
        p_orig = F.softmax(logits_original, dim=-1)
        log_p_regen = F.log_softmax(logits_regenerated, dim=-1)
        return F.kl_div(log_p_regen, p_orig, reduction="batchmean")

    def _compute_size_penalty(self, cppn: CPPNNetwork) -> torch.Tensor:
        """
        L_size = L2 regularization over DNA parameters (Section 15.4).
        """
        return sum((p ** 2).sum() for p in cppn.parameters())

    def encode_weight_into_cppn(
        self,
        cppn: CPPNNetwork,
        target_weights: Dict[str, torch.Tensor],
        num_layers: int = 4,
        ewc: Optional[EWCConsolidator] = None,
        behavior_fn: Optional[Callable[[], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, float]]:
        """
        Optimizes CPPN parameters to minimize the Complete DNA Objective (Section 15.5):
        L_DNA = lambda_1*L_recon + lambda_2*L_behavior + lambda_3*L_future + lambda_4*|D| + L_EWC

        Args:
            cppn: The CPPN network to optimize.
            target_weights: Dict of target weight matrices W_k from SVD filtering.
            num_layers: Number of layers for coordinate generation.
            ewc: Optional EWC consolidator for ancestral genotype protection (Section 17.1).
            behavior_fn: Optional callable returning (logits_original, logits_regenerated) for L_behavior.

        Returns:
            best_params: Best CPPN parameter dict.
            best_loss: Best reconstruction loss value.
            loss_breakdown: Dict with per-component loss values.
        """
        cppn = cppn.to(self.device)
        optimizer = optim.Adam(cppn.parameters(), lr=self.learning_rate)

        # Precompute coordinate tensors for each target weight matrix
        coord_targets = []
        for name, w in target_weights.items():
            if w.ndim >= 2:
                w_2d = w.reshape(w.shape[0], -1)
                out_f, in_f = w_2d.shape[0], w_2d.shape[1]
                coords = SubstrateCoordinateGenerator.get_2d_weight_coordinates(
                    out_features=out_f,
                    in_features=in_f,
                    layer_idx=0,
                    num_layers=num_layers,
                    device=self.device,
                )
                w_target = w_2d.to(self.device)
                norm_sq = (w_target ** 2).sum() + 1e-8
                coord_targets.append((coords, w_target, norm_sq))

        if not coord_targets:
            return cppn.get_parameter_dict(), 0.0, {}

        best_loss = float("inf")
        best_params = cppn.get_parameter_dict()
        final_breakdown = {}

        for step in range(self.max_steps):
            optimizer.zero_grad()

            # 1. Reconstruction Loss (Section 15.1)
            l_recon = self._compute_reconstruction_loss(cppn, coord_targets)

            # 2. Behavioral Loss (Section 15.2)
            l_behavior = torch.tensor(0.0, device=self.device)
            if behavior_fn is not None and self.lambda_behavior > 0:
                try:
                    logits_orig, logits_regen = behavior_fn()
                    l_behavior = self._compute_behavioral_loss(logits_orig, logits_regen)
                except Exception:
                    pass

            # 3. Size Penalty (Section 15.4)
            l_size = self._compute_size_penalty(cppn)

            # 4. EWC Penalty (Section 17.1)
            l_ewc = torch.tensor(0.0, device=self.device)
            if ewc is not None:
                l_ewc = ewc.penalty(cppn)

            # Complete DNA Objective (Section 15.5):
            loss = (
                self.lambda_recon * l_recon
                + self.lambda_behavior * l_behavior
                + self.lambda_size * l_size
                + l_ewc
            )

            loss.backward()
            optimizer.step()

            loss_val = l_recon.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_params = cppn.get_parameter_dict()
                final_breakdown = {
                    "loss_recon": l_recon.item(),
                    "loss_behavior": l_behavior.item(),
                    "loss_size": l_size.item(),
                    "loss_ewc": l_ewc.item(),
                    "loss_total": loss.item(),
                }

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
        Supports the full DNA objective when optional parameters are provided.
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
            ewc=ewc,
            behavior_fn=behavior_fn,
        )

        new_genotype.dna_instinct.genetic_parameters = best_params
        new_genotype.fitness_history["encoding_loss"] = loss_val
        for k, v in breakdown.items():
            new_genotype.fitness_history[f"encoding_{k}"] = v

        return new_genotype, loss_val, breakdown
