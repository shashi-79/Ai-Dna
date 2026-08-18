"""
Slow Clock Genotypic Encoding Pipeline (Consolidated).
Executes the complete phenotype-to-genotype lifecycle:
W* -> SVD Structural Extraction -> Complete DNA Objective -> E(W*) -> D_{t+1}

Implements idea.md Sections 13, 14, 15 (complete objective), and 17 (EWC retention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Callable
from ..dna.structure import Genotype
from .svd_filter import SVDInstinctFilter
from .cppn_encoder import InverseCPPNEncoder
from .ewc import EWCConsolidator


class SlowClockEncoder:
    """
    Slow Clock Engine that distills learned phenotype weights into the next generation DNA genotype.
    Supports the Complete DNA Objective (Section 15.5):
    L_DNA = lambda_1*L_recon + lambda_2*L_behavior + lambda_3*L_future + lambda_4*|D| + L_EWC
    """
    def __init__(
        self,
        rank_ratio: float = 0.25,
        encoder_lr: float = 1e-2,
        encoder_steps: int = 150,
        lambda_ewc: float = 100.0,
        lambda_recon: float = 1.0,
        lambda_behavior: float = 0.1,
        lambda_future: float = 0.0,
        lambda_size: float = 1e-4,
        device: torch.device = torch.device("cpu"),
    ):
        self.rank_ratio = rank_ratio
        self.device = device
        self.svd_filter = SVDInstinctFilter()
        self.cppn_encoder = InverseCPPNEncoder(
            learning_rate=encoder_lr,
            max_steps=encoder_steps,
            lambda_recon=lambda_recon,
            lambda_behavior=lambda_behavior,
            lambda_future=lambda_future,
            lambda_size=lambda_size,
            device=device,
        )
        self.ewc = EWCConsolidator(lambda_ewc=lambda_ewc)

    def _make_behavior_fn(
        self,
        phenotype_model: Optional[nn.Module],
        growth_engine: Optional[Any],
        genotype: Genotype,
        validation_data: Optional[torch.Tensor],
    ) -> Optional[Callable]:
        """
        Creates a callable that returns (logits_original, logits_regenerated)
        for behavioral loss computation (Section 15.2).
        Requires the original phenotype and the regenerated phenotype from current DNA.
        """
        if phenotype_model is None or validation_data is None or growth_engine is None:
            return None

        def behavior_fn() -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                # Original phenotype output
                val_data = validation_data.to(self.device)
                h_orig, _, _, _ = phenotype_model(val_data, modality="text", is_causal=True)
                logits_orig = phenotype_model.ar_head(h_orig)

                # Regenerated phenotype from current DNA
                regen_model = growth_engine.grow_phenotype_model(genotype)
                regen_model.eval()
                h_regen, _, _, _ = regen_model(val_data, modality="text", is_causal=True)
                logits_regen = regen_model.ar_head(h_regen)

            return logits_orig, logits_regen

        return behavior_fn

    def _compute_future_learning_loss(
        self,
        growth_engine: Any,
        genotype: Genotype,
        future_task_fn: Optional[Callable],
    ) -> float:
        """
        L_future = E_{T ~ T_future}[L_task(Train(G(D), T))]
        (Section 15.3)

        This is expensive meta-learning: grow a model from DNA, train on unseen task,
        measure resulting loss. Only used when future_task_fn is provided.
        """
        if future_task_fn is None:
            return 0.0

        try:
            return future_task_fn(growth_engine, genotype)
        except Exception:
            return 0.0

    def step(
        self,
        genotype_t: Genotype,
        learned_state_dict: Dict[str, torch.Tensor],
        protect_ancestral: bool = True,
        phenotype_model: Optional[nn.Module] = None,
        growth_engine: Optional[Any] = None,
        validation_data: Optional[torch.Tensor] = None,
        future_task_fn: Optional[Callable] = None,
    ) -> Tuple[Genotype, Dict[str, Any]]:
        """
        Executes Slow Clock transition: W_t* -> SVD Filtering -> Complete DNA Objective -> D_{t+1}.

        Args:
            genotype_t: Current generation genotype.
            learned_state_dict: Trained phenotype parameter state dict (W*).
            protect_ancestral: Whether to apply EWC ancestral genotype protection.
            phenotype_model: Optional learned phenotype for behavioral loss (Section 15.2).
            growth_engine: Optional GrowthEngine for behavioral/future loss.
            validation_data: Optional validation tokens for behavioral divergence measurement.
            future_task_fn: Optional callable(growth_engine, genotype) -> float for L_future (Section 15.3).
        """
        # 1. Truncated SVD structural instinct extraction (Section 14)
        filtered_weights, energies = self.svd_filter.filter_state_dict(
            learned_state_dict, rank_ratio=self.rank_ratio
        )

        mean_energy = sum(energies.values()) / max(1, len(energies))

        # 2. Register ancestral genotype for EWC protection (Section 17.1)
        ewc = None
        if protect_ancestral and genotype_t.dna_instinct.genetic_parameters:
            from ..growth.cppn import CPPNNetwork
            arch = genotype_t.dna_architecture
            instinct = genotype_t.dna_instinct
            old_cppn = CPPNNetwork(
                in_features=arch.coord_dim,
                hidden_dim=instinct.cppn_hidden_dim,
                num_layers=instinct.cppn_layers,
                out_features=1,
            ).to(self.device)
            try:
                old_cppn.load_parameter_dict(instinct.genetic_parameters)
                self.ewc.register_ancestral_genotype(old_cppn)
                ewc = self.ewc
            except Exception:
                ewc = None

        # 3. Build behavioral loss function (Section 15.2)
        behavior_fn = self._make_behavior_fn(
            phenotype_model, growth_engine, genotype_t, validation_data
        )

        # 4. Encode extracted structures into new Genotype via Complete DNA Objective (Section 15.5)
        new_genotype, recon_loss, breakdown = self.cppn_encoder.encode_genotype(
            genotype=genotype_t,
            target_weights=filtered_weights,
            ewc=ewc,
            behavior_fn=behavior_fn,
        )

        # 5. Compute future learning loss if available (Section 15.3)
        future_loss = self._compute_future_learning_loss(
            growth_engine, new_genotype, future_task_fn
        )

        new_genotype.fitness_history["mean_retained_energy"] = mean_energy
        new_genotype.fitness_history["reconstruction_loss"] = recon_loss
        new_genotype.fitness_history["future_learning_loss"] = future_loss

        summary = {
            "mean_retained_energy": mean_energy,
            "reconstruction_loss": recon_loss,
            "future_learning_loss": future_loss,
            "num_filtered_matrices": len(energies),
            "generation": new_genotype.generation,
            **breakdown,
        }

        return new_genotype, summary
