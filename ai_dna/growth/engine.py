"""
Growth Engine G(D, C) -> Phenotype Parameters.
Generates full neural phenotypes from compact genotype representations.
Supports Multi-Head Latent Attention (MLA), Top-K Sparsely-Gated MoE, and Contrastive Omni-Modal Encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def instantiate_cppn(self, genotype: Genotype) -> CPPNNetwork:
        """Instantiates and loads the CPPN from genotype's genetic parameters with dynamic dimensions."""
        arch = genotype.dna_architecture
        instinct = genotype.dna_instinct

        hidden_dim = instinct.cppn_hidden_dim
        num_layers = instinct.cppn_layers

        # Dynamically detect hidden_dim and num_layers from stored parameters
        if instinct.genetic_parameters:
            if "backbone.0.weight" in instinct.genetic_parameters:
                hidden_dim = instinct.genetic_parameters["backbone.0.weight"].shape[0]
                num_layers = len([k for k in instinct.genetic_parameters.keys() if k.endswith(".weight") and k.startswith("backbone.")])

        cppn = CPPNNetwork(
            in_features=arch.coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
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
        matrix_idx: int = 0,
        coord_dim: int = 32,
    ) -> torch.Tensor:
        """
        Grows a 2D weight matrix W_ij = G(D, C_ij) of shape (out_features, in_features)
        using the CPPN genotype decoder with unique matrix coordinate index.
        """
        coords = SubstrateCoordinateGenerator.get_2d_weight_coordinates(
            out_features=out_features,
            in_features=in_features,
            layer_idx=layer_idx,
            num_layers=num_layers,
            expert_idx=expert_idx,
            num_experts=num_experts,
            matrix_idx=matrix_idx,
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
        Eager growth: Generates all weight and bias tensors for the target model architecture
        using the CPPN genotype decoder.
        """
        arch = genotype.dna_architecture
        coord_dim = arch.coord_dim
        cppn = self.instantiate_cppn(genotype)

        weights = {}
        d_model = arch.d_model
        num_layers = arch.num_layers
        num_experts = arch.num_experts
        d_expert_hidden = arch.d_expert_hidden
        d_kv_latent = getattr(arch, "kv_latent_dim", max(8, d_model // 4))

        kwargs = {"coord_dim": coord_dim}

        # 1. Text & Multimodal Encoders
        weights["text_encoder.token_emb.weight"] = self.grow_weight_matrix(
            cppn, arch.vocab_size, d_model, layer_idx=0, num_layers=num_layers + 2, **kwargs
        )
        weights["vision_encoder.patch_proj.weight"] = self.grow_weight_matrix(
            cppn, d_model, 3 * 4 * 4, layer_idx=0, num_layers=num_layers + 2, **kwargs
        )
        weights["audio_encoder.proj.weight"] = self.grow_weight_matrix(
            cppn, d_model, 80, layer_idx=0, num_layers=num_layers + 2, **kwargs
        )
        weights["contrastive_head.proj.weight"] = self.grow_weight_matrix(
            cppn, d_model, d_model, layer_idx=0, num_layers=num_layers + 2, **kwargs
        )

        # 2. Layer Blocks (MLA Attention Projections & MoE Experts)
        for l in range(num_layers):
            weights[f"blocks.{l}.attn.w_q.weight"] = self.grow_weight_matrix(
                cppn, d_model, d_model, layer_idx=l, num_layers=num_layers, matrix_idx=0, **kwargs
            )
            weights[f"blocks.{l}.attn.w_dkv.weight"] = self.grow_weight_matrix(
                cppn, d_kv_latent, d_model, layer_idx=l, num_layers=num_layers, matrix_idx=2, **kwargs
            )
            weights[f"blocks.{l}.attn.w_uk.weight"] = self.grow_weight_matrix(
                cppn, d_model, d_kv_latent, layer_idx=l, num_layers=num_layers, matrix_idx=4, **kwargs
            )
            weights[f"blocks.{l}.attn.w_uv.weight"] = self.grow_weight_matrix(
                cppn, d_model, d_kv_latent, layer_idx=l, num_layers=num_layers, matrix_idx=6, **kwargs
            )
            weights[f"blocks.{l}.attn.o_proj.weight"] = self.grow_weight_matrix(
                cppn, d_model, d_model, layer_idx=l, num_layers=num_layers, matrix_idx=8, **kwargs
            )

            # MoE Experts with SwiGLU Gated Activation
            for e in range(num_experts):
                weights[f"blocks.{l}.moe.experts.{e}.swiglu.gate_proj.weight"] = self.grow_weight_matrix(
                    cppn, d_expert_hidden, d_model, layer_idx=l, num_layers=num_layers, expert_idx=e, num_experts=num_experts, matrix_idx=10, **kwargs
                )
                weights[f"blocks.{l}.moe.experts.{e}.swiglu.up_proj.weight"] = self.grow_weight_matrix(
                    cppn, d_expert_hidden, d_model, layer_idx=l, num_layers=num_layers, expert_idx=e, num_experts=num_experts, matrix_idx=11, **kwargs
                )
                weights[f"blocks.{l}.moe.experts.{e}.swiglu.down_proj.weight"] = self.grow_weight_matrix(
                    cppn, d_model, d_expert_hidden, layer_idx=l, num_layers=num_layers, expert_idx=e, num_experts=num_experts, matrix_idx=12, **kwargs
                )

        # 3. Output Head
        weights["ar_head.proj.weight"] = self.grow_weight_matrix(
            cppn, arch.vocab_size, d_model, layer_idx=num_layers + 1, num_layers=num_layers + 2, matrix_idx=14, **kwargs
        )
        weights["ar_head.proj.bias"] = self.grow_bias_vector(
            cppn, arch.vocab_size, layer_idx=num_layers + 1, num_layers=num_layers + 2, **kwargs
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

        # 3.2 Load preserved modal parameters (token embeddings, AR head, cls_head) if present in genotype
        if genotype.dna_instinct.genetic_parameters:
            modal_dict = {}
            for k, v in genotype.dna_instinct.genetic_parameters.items():
                if k.startswith("modal."):
                    modal_dict[k[len("modal."):]] = v.to(self.device)
            if "cls_head.classifier.1.weight" in modal_dict:
                ckpt_classes = modal_dict["cls_head.classifier.1.weight"].shape[0]
                if model.cls_head.classifier[1].out_features != ckpt_classes:
                    from ..models.modules import ClassificationHead
                    model.cls_head = ClassificationHead(model.d_model, num_classes=ckpt_classes).to(self.device)
            if any(k.startswith("audio_head.") for k in modal_dict):
                if not hasattr(model, "audio_head") or model.audio_head is None:
                    model.audio_head = nn.Sequential(
                        nn.LayerNorm(model.d_model),
                        nn.Linear(model.d_model, model.d_model),
                        nn.GELU(),
                        nn.Linear(model.d_model, 80)
                    ).to(self.device)
            if modal_dict:
                model.load_state_dict(modal_dict, strict=False)

        # 3.5 Check for LoRA Rank in architecture and inject LoRA parameters
        lora_rank = getattr(genotype.dna_architecture, "lora_rank", 0)
        if lora_rank > 0:
            from ..models.lora import replace_linear_with_lora, load_lora_parameters
            replace_linear_with_lora(model, rank=lora_rank)

            # Check if adapter CPPN parameters exist (keys starting with "adapter.")
            adapter_params = {}
            if genotype.dna_instinct.genetic_parameters:
                for k, v in genotype.dna_instinct.genetic_parameters.items():
                    if k.startswith("adapter."):
                        adapter_params[k[len("adapter."):]] = v

            if adapter_params:
                from .cppn import CPPNNetwork
                arch = genotype.dna_architecture
                instinct = genotype.dna_instinct
                
                hidden_dim = getattr(instinct, "adapter_cppn_hidden_dim", instinct.cppn_hidden_dim)
                num_layers = getattr(instinct, "adapter_cppn_layers", instinct.cppn_layers)
                if "backbone.0.weight" in adapter_params:
                    hidden_dim = adapter_params["backbone.0.weight"].shape[0]
                    num_layers = len([k for k in adapter_params.keys() if k.endswith(".weight") and k.startswith("backbone.")])

                # Instantiate adapter CPPN with dynamic dimensions
                cppn = CPPNNetwork(
                    in_features=arch.coord_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    out_features=1,
                ).to(self.device)
                cppn.load_parameter_dict(adapter_params)
                cppn.eval()

                with torch.no_grad():
                    for name, module in model.named_modules():
                        from ..models.lora import LoRALinear
                        if isinstance(module, LoRALinear):
                            parts = name.split(".")
                            layer_idx = 0
                            expert_idx = 0
                            for idx, part in enumerate(parts):
                                if part == "blocks":
                                    layer_idx = int(parts[idx + 1])
                                elif part == "experts":
                                    expert_idx = int(parts[idx + 1])

                            matrix_idx_A = SubstrateCoordinateGenerator.get_matrix_idx_from_name(f"{name}.lora_A")
                            matrix_idx_B = SubstrateCoordinateGenerator.get_matrix_idx_from_name(f"{name}.lora_B")

                            # Grow lora_A
                            lora_A_weight = self.grow_weight_matrix(
                                cppn=cppn,
                                out_features=module.lora_A.shape[0],
                                in_features=module.lora_A.shape[1],
                                layer_idx=layer_idx,
                                num_layers=arch.num_layers,
                                expert_idx=expert_idx,
                                num_experts=arch.num_experts,
                                matrix_idx=matrix_idx_A,
                                coord_dim=arch.coord_dim,
                            )
                            module.lora_A.copy_(lora_A_weight)

                            # Grow lora_B
                            lora_B_weight = self.grow_weight_matrix(
                                cppn=cppn,
                                out_features=module.lora_B.shape[0],
                                in_features=module.lora_B.shape[1],
                                layer_idx=layer_idx,
                                num_layers=arch.num_layers,
                                expert_idx=expert_idx,
                                num_experts=arch.num_experts,
                                matrix_idx=matrix_idx_B,
                                coord_dim=arch.coord_dim,
                            )
                            module.lora_B.copy_(lora_B_weight)

            # 3.6 Load exact LoRA parameters if present (Hybrid DNA mode)
            if genotype.dna_instinct.genetic_parameters:
                exact_lora = {}
                for k, v in genotype.dna_instinct.genetic_parameters.items():
                    if k.startswith("exact_lora."):
                        exact_lora[k[len("exact_lora."):]] = v.to(self.device)
                if exact_lora:
                    load_lora_parameters(model, exact_lora)
            else:
                # Load stored lora weights if present as fallback
                lora_params = {k: v.to(self.device) for k, v in genotype.dna_instinct.genetic_parameters.items() if "lora_" in k}
                if lora_params:
                    load_lora_parameters(model, lora_params)

        # 4. Auto-Calibrate model using genotypically embedded anchors (GECA) if available
        if hasattr(genotype, "calibration_anchors") and genotype.calibration_anchors:
            try:
                self.auto_calibrate_model(model, genotype.calibration_anchors)
            except Exception as e:
                print(f"[Growth Engine Warning]: GECA Auto-calibration failed: {e}")

        return model

    def auto_calibrate_model(self, model: "PhenotypeNeuralNetwork", calibration_anchors: Dict[str, torch.Tensor], steps: int = 15):
        """
        Executes a rapid, dataset-free, zero-shot calibration phase using the
        anchors embedded inside the Genotype.
        """
        d_model = model.d_model
        # Optimize output head and routing gate parameters for vocabulary logit matching
        opt_params = list(model.ar_head.parameters()) + list(model.shared_router.parameters())
        optimizer = torch.optim.AdamW(opt_params, lr=5e-3, weight_decay=1e-4)
        
        model.train()
        for _ in range(steps):
            loss = 0.0
            for modality, anchor_data in calibration_anchors.items():
                anchor_data = anchor_data.to(self.device)
                if anchor_data.size(0) > 0:
                    a_M = anchor_data[:, :d_model] # Anchor key in latent d_model space
                    y_M = anchor_data[:, d_model:] # Aligned output targets
                    
                    logits = model.ar_head(a_M)
                    loss += F.kl_div(F.log_softmax(logits, dim=-1), y_M, reduction="batchmean")
                    
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
