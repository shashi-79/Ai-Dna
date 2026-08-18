"""
Growth Engine G(D, C) -> Phenotype Parameters.
Generates full neural phenotypes from compact genotype representations.
Supports Multi-Head Latent Attention (MLA), Top-K Sparsely-Gated MoE, and Contrastive Omni-Modal Encoders.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING
from ..dna.structure import Genotype
from .cppn import CPPNNetwork
from .coordinates import SubstrateCoordinateGenerator

if TYPE_CHECKING:
    from ..models.phenotype import PhenotypeNeuralNetwork


class GrowthEngine:
    """
    Growth Engine that translates a constitutional Genotype D into phenotype weights W_0 = G(D).
    """
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device

    def instantiate_cppn(self, genotype: Genotype) -> CPPNNetwork:
        """Instantiates and loads the CPPN from genotype's genetic parameters."""
        arch = genotype.dna_architecture
        instinct = genotype.dna_instinct

        cppn = CPPNNetwork(
            in_features=arch.coord_dim,
            hidden_dim=instinct.cppn_hidden_dim,
            num_layers=instinct.cppn_layers,
            out_features=1,
        ).to(self.device)

        if instinct.genetic_parameters:
            try:
                model_state = cppn.state_dict()
                matched_state = {}
                for k, v in instinct.genetic_parameters.items():
                    if k in model_state and model_state[k].shape == v.shape:
                        matched_state[k] = v.to(self.device)
                if matched_state:
                    model_state.update(matched_state)
                    cppn.load_state_dict(model_state)
            except Exception:
                pass  # Fall back to native initialization
        else:
            genotype.dna_instinct.genetic_parameters = cppn.get_parameter_dict()

        return cppn

    def grow_weight_matrix(
        self,
        cppn: CPPNNetwork,
        out_features: int,
        in_features: int,
        layer_idx: int = 0,
        num_layers: int = 4,
        expert_idx: int = 0,
        num_experts: int = 1,
        coord_dim: int = 5,
    ) -> torch.Tensor:
        """
        Grows a 2D weight matrix W_ij = G(D, C_ij) of shape (out_features, in_features).
        """
        coords = SubstrateCoordinateGenerator.get_2d_weight_coordinates(
            out_features=out_features,
            in_features=in_features,
            layer_idx=layer_idx,
            num_layers=num_layers,
            expert_idx=expert_idx,
            num_experts=num_experts,
            device=self.device,
            coord_dim=coord_dim,
        )
        with torch.no_grad():
            raw_weights = cppn(coords).squeeze(-1)
            std = (2.0 / (in_features + out_features)) ** 0.5
            scaled_weights = raw_weights * std
        return scaled_weights

    def grow_bias_vector(
        self,
        cppn: CPPNNetwork,
        features: int,
        layer_idx: int = 0,
        num_layers: int = 4,
        coord_dim: int = 5,
    ) -> torch.Tensor:
        """Grows a 1D bias vector of shape (features,)."""
        coords = SubstrateCoordinateGenerator.get_1d_bias_coordinates(
            features=features,
            layer_idx=layer_idx,
            num_layers=num_layers,
            device=self.device,
            coord_dim=coord_dim,
        )
        with torch.no_grad():
            raw_bias = cppn(coords).squeeze(-1)
            scaled_bias = raw_bias * 0.01
        return scaled_bias

    def grow_phenotype_weights(self, genotype: Genotype) -> Dict[str, torch.Tensor]:
        """
        Eager growth: Generates all weight and bias tensors for the target model architecture.
        Aligned with MLA, Top-K MoE, and Contrastive Encoders.
        """
        cppn = self.instantiate_cppn(genotype)
        arch = genotype.dna_architecture
        weights = {}

        d_model = arch.d_model
        num_layers = arch.num_layers
        num_experts = arch.num_experts
        d_expert_hidden = arch.d_expert_hidden
        d_kv_latent = getattr(arch, "kv_latent_dim", max(8, d_model // 4))
        coord_dim = arch.coord_dim

        # 1. Text & Multimodal Encoders
        weights["text_encoder.token_emb.weight"] = self.grow_weight_matrix(
            cppn, arch.vocab_size, d_model, layer_idx=0, num_layers=num_layers + 2, coord_dim=coord_dim
        )
        weights["vision_encoder.patch_proj.weight"] = self.grow_weight_matrix(
            cppn, d_model, 3 * 4 * 4, layer_idx=0, num_layers=num_layers + 2, coord_dim=coord_dim
        )
        weights["audio_encoder.proj.weight"] = self.grow_weight_matrix(
            cppn, d_model, 80, layer_idx=0, num_layers=num_layers + 2, coord_dim=coord_dim
        )
        weights["contrastive_head.proj.weight"] = self.grow_weight_matrix(
            cppn, d_model, d_model, layer_idx=0, num_layers=num_layers + 2, coord_dim=coord_dim
        )

        # 2. Layer Blocks (MLA Attention Projections & MoE Experts)
        for l in range(num_layers):
            # MLA Projections: w_q, w_dkv (down), w_uk (up), w_uv (up), o_proj
            weights[f"blocks.{l}.attn.w_q.weight"] = self.grow_weight_matrix(
                cppn, d_model, d_model, layer_idx=l, num_layers=num_layers, coord_dim=coord_dim
            )
            weights[f"blocks.{l}.attn.w_dkv.weight"] = self.grow_weight_matrix(
                cppn, d_kv_latent, d_model, layer_idx=l, num_layers=num_layers, coord_dim=coord_dim
            )
            weights[f"blocks.{l}.attn.w_uk.weight"] = self.grow_weight_matrix(
                cppn, d_model, d_kv_latent, layer_idx=l, num_layers=num_layers, coord_dim=coord_dim
            )
            weights[f"blocks.{l}.attn.w_uv.weight"] = self.grow_weight_matrix(
                cppn, d_model, d_kv_latent, layer_idx=l, num_layers=num_layers, coord_dim=coord_dim
            )
            weights[f"blocks.{l}.attn.o_proj.weight"] = self.grow_weight_matrix(
                cppn, d_model, d_model, layer_idx=l, num_layers=num_layers, coord_dim=coord_dim
            )

            # MoE Experts: Gate Up & Down projections
            for e in range(num_experts):
                weights[f"blocks.{l}.moe.experts.{e}.up_proj.weight"] = self.grow_weight_matrix(
                    cppn, d_expert_hidden, d_model, layer_idx=l, num_layers=num_layers, expert_idx=e, num_experts=num_experts, coord_dim=coord_dim
                )
                weights[f"blocks.{l}.moe.experts.{e}.down_proj.weight"] = self.grow_weight_matrix(
                    cppn, d_model, d_expert_hidden, layer_idx=l, num_layers=num_layers, expert_idx=e, num_experts=num_experts, coord_dim=coord_dim
                )

        # 3. Output Head
        weights["ar_head.proj.weight"] = self.grow_weight_matrix(
            cppn, arch.vocab_size, d_model, layer_idx=num_layers + 1, num_layers=num_layers + 2, coord_dim=coord_dim
        )
        weights["ar_head.proj.bias"] = self.grow_bias_vector(
            cppn, arch.vocab_size, layer_idx=num_layers + 1, num_layers=num_layers + 2, coord_dim=coord_dim
        )

        return weights

    def grow_phenotype_model(self, genotype: Genotype) -> "PhenotypeNeuralNetwork":
        """
        Creates the complete, instantiated Phenotype Neural Network guided by the DNA.
        """
        from ..models.phenotype import PhenotypeNeuralNetwork

        # 1. Instantiate the architecture
        model = PhenotypeNeuralNetwork(
            arch=genotype.dna_architecture,
            dna_routing=genotype.dna_routing,
            dna_memory=genotype.dna_memory,
        ).to(self.device)

        # 2. Grow weights from CPPN
        grown_weights = self.grow_phenotype_weights(genotype)

        # 3. Load weights into the model
        model.load_state_dict(grown_weights, strict=False)

        return model
