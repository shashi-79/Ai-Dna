"""
Slow Clock Genotypic Encoding Pipeline (Consolidated).
Executes the complete phenotype-to-genotype lifecycle:
W* -> CPPN Structural Encoding -> Complete DNA Objective -> E(W*) -> D_{t+1}

Implements idea.md Sections 13, 15 (complete objective), 17 (EWC retention), and 43 (CL-DNA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Callable
from ..dna.structure import Genotype
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
        device: Optional[torch.device] = None,
    ):
        self.rank_ratio = rank_ratio
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        Executes Slow Clock transition: W_t* -> LoRA Extraction -> Genotypic Encoding -> D_{t+1}.

        Args:
            genotype_t: Current generation genotype.
            learned_state_dict: Trained phenotype parameter state dict (W*).
            protect_ancestral: Whether to apply EWC ancestral genotype protection.
            phenotype_model: Optional learned phenotype for behavioral loss (Section 15.2).
            growth_engine: Optional GrowthEngine for behavioral/future loss.
            validation_data: Optional validation tokens for behavioral divergence measurement.
            future_task_fn: Optional callable(growth_engine, genotype) -> float for L_future (Section 15.3).
        """
        lora_rank = getattr(genotype_t.dna_architecture, "lora_rank", 0)

        # 1. Build behavioral loss function (Section 15.2)
        behavior_fn = self._make_behavior_fn(
            phenotype_model, growth_engine, genotype_t, validation_data
        )

        if lora_rank > 0:
            # LoRA + CPPN Mode: Extract active adapter weights directly
            if phenotype_model is not None:
                from ..models.lora import extract_lora_parameters
                target_weights = extract_lora_parameters(phenotype_model)
            else:
                target_weights = {k: v for k, v in learned_state_dict.items() if "lora_" in k}
            
            mean_energy = 1.0  # SVD is bypassed

            # Extract previous adapter CPPN parameters if they exist
            prev_adapter_params = {}
            if genotype_t.dna_instinct.genetic_parameters:
                for k, v in genotype_t.dna_instinct.genetic_parameters.items():
                    if k.startswith("adapter."):
                        prev_adapter_params[k[len("adapter."):]] = v
            
            # Register ancestral genotype for EWC protection
            ewc = None
            if protect_ancestral and prev_adapter_params:
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
                    old_cppn.load_parameter_dict(prev_adapter_params)
                    self.ewc.register_ancestral_genotype(old_cppn)
                    ewc = self.ewc
                except Exception:
                    ewc = None

            # Optimize the adapter CPPN with dynamic dimensions
            from ..growth.cppn import CPPNNetwork
            arch = genotype_t.dna_architecture
            instinct = genotype_t.dna_instinct

            hidden_dim = getattr(instinct, "adapter_cppn_hidden_dim", max(64, instinct.cppn_hidden_dim * 2))
            num_layers = getattr(instinct, "adapter_cppn_layers", max(4, instinct.cppn_layers + 1))
            if prev_adapter_params and "backbone.0.weight" in prev_adapter_params:
                hidden_dim = prev_adapter_params["backbone.0.weight"].shape[0]
                num_layers = len([k for k in prev_adapter_params.keys() if k.endswith(".weight") and k.startswith("backbone.")])

            cppn = CPPNNetwork(
                in_features=arch.coord_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                out_features=1,
            ).to(self.device)
            if prev_adapter_params:
                try:
                    cppn.load_parameter_dict(prev_adapter_params)
                except Exception:
                    pass

            best_params, recon_loss, breakdown = self.cppn_encoder.encode_weight_into_cppn(
                cppn=cppn,
                target_weights=target_weights,
                num_layers=arch.num_layers,
                num_experts=arch.num_experts,
                ewc=ewc,
                behavior_fn=behavior_fn,
            )

            # Dynamic Capacity Expansion (DCE) for Adapter CPPN if reconstruction loss is high
            if recon_loss > 0.05 and hidden_dim < 256:
                new_hidden_dim = hidden_dim + 32
                print(f"[Slow Clock Adapter DCE]: Expanding adapter CPPN hidden_dim {hidden_dim} -> {new_hidden_dim}...")
                expanded_cppn = CPPNNetwork(
                    in_features=arch.coord_dim,
                    hidden_dim=new_hidden_dim,
                    num_layers=num_layers,
                    out_features=1,
                ).to(self.device)
                
                # Copy overlapping parameter weights
                expanded_state = expanded_cppn.state_dict()
                for k, v in best_params.items():
                    if k in expanded_state:
                        s = [slice(0, min(d1, d2)) for d1, d2 in zip(expanded_state[k].shape, v.shape)]
                        expanded_state[k][tuple(s)] = v[tuple(s)].to(self.device)
                expanded_cppn.load_state_dict(expanded_state)
                
                # Re-optimize with expanded capacity
                best_params, recon_loss, breakdown = self.cppn_encoder.encode_weight_into_cppn(
                    cppn=expanded_cppn,
                    target_weights=target_weights,
                    num_layers=arch.num_layers,
                    num_experts=arch.num_experts,
                    ewc=ewc,
                    behavior_fn=behavior_fn,
                )
                print(f"[Slow Clock Adapter DCE]: Re-encoding complete. New recon_loss={recon_loss:.4f}")

            # Build final new_genotype
            new_genotype = genotype_t.clone(new_id=f"{genotype_t.genotype_id}_enc")
            new_genotype.generation = genotype_t.generation + 1
            
            # Combine base CPPN parameters, adapter CPPN parameters, and learned modal parameters
            combined_params = {}
            if genotype_t.dna_instinct.genetic_parameters:
                for k, v in genotype_t.dna_instinct.genetic_parameters.items():
                    if not k.startswith("adapter."):
                        combined_params[k] = v.clone() if hasattr(v, "clone") else v

            for k, v in best_params.items():
                combined_params[f"adapter.{k}"] = v

            # Preserve learned modal parameters (embeddings, prediction head, norm) and exact LoRA residuals
            if phenotype_model is not None:
                for name, param in phenotype_model.named_parameters():
                    if any(m in name for m in ["text_encoder", "embeddings", "ar_head", "ln_final", "ln1", "ln2"]):
                        combined_params[f"modal.{name}"] = param.clone().detach()
                    elif "lora_" in name:
                        combined_params[f"exact_lora.{name}"] = param.clone().detach()
            elif learned_state_dict:
                for name, param in learned_state_dict.items():
                    if any(m in name for m in ["text_encoder", "embeddings", "ar_head", "ln_final", "ln1", "ln2"]):
                        combined_params[f"modal.{name}"] = param.clone().detach() if hasattr(param, "clone") else param
                    elif "lora_" in name:
                        combined_params[f"exact_lora.{name}"] = param.clone().detach() if hasattr(param, "clone") else param

            new_genotype.dna_instinct.genetic_parameters = combined_params
        else:
            # Standard CPPN path (direct weight fitting)
            target_weights = {k: v for k, v in learned_state_dict.items() if "weight" in k or "bias" in k}
            mean_energy = 1.0  # SVD is bypassed

            # Register ancestral genotype for EWC protection
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

            # Encode extracted structures into new Genotype via Complete DNA Objective (Section 15.5)
            new_genotype, recon_loss, breakdown = self.cppn_encoder.encode_genotype(
                genotype=genotype_t,
                target_weights=target_weights,
                ewc=ewc,
                behavior_fn=behavior_fn,
            )

            # Dynamic CPPN Capacity Expansion (DCE) if reconstruction limit is hit
            if recon_loss > 0.04 and new_genotype.dna_instinct.cppn_hidden_dim < 128:
                old_dim = new_genotype.dna_instinct.cppn_hidden_dim
                new_dim = old_dim + 16
                print(f"[Slow Clock DCE]: Saturation detected (recon_loss={recon_loss:.4f} > 0.04). Expanding CPPN hidden_dim {old_dim} -> {new_dim}...")
                
                # Net2Net expand the genotype parameter state
                self._expand_cppn_genotype(new_genotype, delta_dim=16)
                
                # Re-run encoder to optimize the new dimensions
                new_genotype, recon_loss, breakdown = self.cppn_encoder.encode_genotype(
                    genotype=new_genotype,
                    target_weights=target_weights,
                    ewc=ewc,
                    behavior_fn=behavior_fn,
                )
                # Correct generation count (prevent double increment)
                new_genotype.generation = genotype_t.generation + 1
                print(f"[Slow Clock DCE]: Re-encoding complete. New recon_loss={recon_loss:.4f}")

        # 5. Compute future learning loss if available (Section 15.3)
        future_loss = self._compute_future_learning_loss(
            growth_engine, new_genotype, future_task_fn
        )

        # 5.5 Extract Genotypically Embedded Calibration Anchors (GECA)
        if phenotype_model is not None:
            try:
                new_genotype.calibration_anchors = self.extract_calibration_anchors(phenotype_model)
            except Exception as e:
                print(f"[Slow Clock Warning]: Failed to extract GECA: {e}")

        new_genotype.fitness_history["mean_retained_energy"] = mean_energy
        new_genotype.fitness_history["reconstruction_loss"] = recon_loss
        new_genotype.fitness_history["future_learning_loss"] = future_loss

        num_filtered = len(target_weights)

        summary = {
            "mean_retained_energy": mean_energy,
            "reconstruction_loss": recon_loss,
            "future_learning_loss": future_loss,
            "num_filtered_matrices": num_filtered,
            "generation": new_genotype.generation,
            **breakdown,
        }

        return new_genotype, summary

    def extract_calibration_anchors(self, phenotype_model: nn.Module, K: int = 8) -> Dict[str, torch.Tensor]:
        """
        Extracts high-density calibration anchors for Text, Vision, and Audio modalities
        directly from the trained phenotype model using SVD decomposition of projection layers.
        """
        anchors = {}
        with torch.no_grad():
            # Get output head weights for target logit mapping
            ar_head = phenotype_model.ar_head.proj if hasattr(phenotype_model.ar_head, "proj") else phenotype_model.ar_head
            w_out = ar_head.weight # [vocab_size, d_model]

            # 1. Text Modality Anchors (from Token Embedding)
            if hasattr(phenotype_model, "text_encoder") and hasattr(phenotype_model.text_encoder, "token_emb"):
                w_text = phenotype_model.text_encoder.token_emb.weight # [vocab_size, d_model]
                # SVD of embedding space to get principal components
                U, S, V = torch.svd(w_text)
                a_text = V[:, :K].t() * S[:K].unsqueeze(-1) # [K, d_model]
                y_text = F.softmax(a_text @ w_out.t(), dim=-1) # [K, vocab_size]
                anchors["text"] = torch.cat([a_text, y_text], dim=-1) # Pack keys and targets together

            # 2. Vision Modality Anchors (from Patch Projection)
            if hasattr(phenotype_model, "vision_encoder") and hasattr(phenotype_model.vision_encoder, "patch_proj"):
                w_vis = phenotype_model.vision_encoder.patch_proj.weight # [d_model, patch_dim]
                U, S, V = torch.svd(w_vis.t())
                a_vis = V[:, :K].t() * S[:K].unsqueeze(-1)
                y_vis = F.softmax(a_vis @ w_out.t(), dim=-1)
                anchors["vision"] = torch.cat([a_vis, y_vis], dim=-1)

            # 3. Audio Modality Anchors (from Audio Projection)
            if hasattr(phenotype_model, "audio_encoder") and hasattr(phenotype_model.audio_encoder, "proj"):
                w_aud = phenotype_model.audio_encoder.proj.weight # [d_model, in_dim]
                U, S, V = torch.svd(w_aud.t())
                a_aud = V[:, :K].t() * S[:K].unsqueeze(-1)
                y_aud = F.softmax(a_aud @ w_out.t(), dim=-1)
                anchors["audio"] = torch.cat([a_aud, y_aud], dim=-1)

        return anchors

    def _expand_cppn_genotype(self, genotype: Genotype, delta_dim: int):
        """
        Dynamically expands the hidden dimension of the CPPN parameters inside the Genotype
        using Net2Net function-preserving network expansion.
        """
        instinct = genotype.dna_instinct
        old_dim = instinct.cppn_hidden_dim
        new_dim = old_dim + delta_dim
        
        # Update metadata field
        instinct.cppn_hidden_dim = new_dim
        
        # Expand actual parameters in the genetic_parameters dictionary
        params = instinct.genetic_parameters
        if not params:
            return
            
        def pad_zeros(tensor: torch.Tensor, dim_idx: int, size: int) -> torch.Tensor:
            sizes = list(tensor.shape)
            sizes[dim_idx] = size
            zeros = torch.zeros(sizes, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, zeros], dim=dim_idx)
            
        new_params = {}
        for name, p in params.items():
            # 1. Input layer (backbone.0)
            if name == "backbone.0.weight":
                # shape: [hidden_dim, in_features] -> expand output dimension (add rows)
                new_params[name] = pad_zeros(p, 0, delta_dim)
            elif name == "backbone.0.bias":
                # shape: [hidden_dim] -> expand output dimension
                new_params[name] = pad_zeros(p, 0, delta_dim)
                
            # 2. Hidden layers (backbone.2, backbone.4, ...)
            elif name.startswith("backbone.") and name.endswith(".weight") and name != "backbone.0.weight":
                # shape: [hidden_dim, hidden_dim] -> expand both input and output dimensions
                # Expand output first (add rows)
                temp = pad_zeros(p, 0, delta_dim)
                # Expand input next (add columns)
                new_params[name] = pad_zeros(temp, 1, delta_dim)
            elif name.startswith("backbone.") and name.endswith(".bias") and name != "backbone.0.bias":
                # shape: [hidden_dim] -> expand output dimension
                new_params[name] = pad_zeros(p, 0, delta_dim)
                
            # 3. Output layer (out_proj)
            elif name == "out_proj.weight":
                # shape: [out_features, hidden_dim] -> expand input dimension (columns)
                new_params[name] = pad_zeros(p, 1, delta_dim)
            elif name == "out_proj.bias":
                # shape: [out_features] -> remains unchanged
                new_params[name] = p.clone()
            else:
                new_params[name] = p.clone()
                
        instinct.genetic_parameters = new_params
