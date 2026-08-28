"""
LoRA (Low-Rank Adaptation) module for Phenotype Neural Networks.
Implements parameter-efficient fine-tuning wrappers and injection/extraction helpers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with trainable low-rank adapters.
    W = W_base + (lora_B @ lora_A) * (alpha / rank)
    """
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base layer is frozen
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False

        # Put lora Parameters on the same device as the base weight
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes A with Kaiming uniform and B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base projection
        base_out = self.linear(x)
        # Low-rank projection
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def replace_linear_with_lora(
    parent_module: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    prefix: str = "",
    adapted_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Recursively scans and replaces Linear layers in target submodules (attention and experts)
    with LoRALinear wrappers.
    """
    if adapted_names is None:
        adapted_names = []

    for name, child in parent_module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            # Adapt attention and MoE expert linear layers
            if "attn" in full_name or "moe" in full_name:
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                device = child.weight.device
                lora_layer.to(device)
                setattr(parent_module, name, lora_layer)
                adapted_names.append(full_name)
        else:
            replace_linear_with_lora(child, rank=rank, alpha=alpha, prefix=full_name, adapted_names=adapted_names)

    return adapted_names


def freeze_model_except_lora(model: nn.Module, freeze_modalities: bool = False):
    """
    Freezes all base parameters of the network except for active LoRA parameters.
    If freeze_modalities is False, keeps text_encoder (token embeddings) and ar_head (output projection)
    trainable so the network can map and decode vocabulary tokens properly.
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        elif not freeze_modalities and any(k in name for k in ["text_encoder", "embeddings", "ar_head", "ln_final", "ln1", "ln2"]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def extract_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts all active lora_A and lora_B parameter tensors from the model."""
    lora_params = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params[f"{name}.lora_A"] = module.lora_A.clone().detach()
            lora_params[f"{name}.lora_B"] = module.lora_B.clone().detach()
    return lora_params


def load_lora_parameters(model: nn.Module, lora_params: Dict[str, torch.Tensor]):
    """Loads a state dict of LoRA parameters back into the model."""
    model_state = model.state_dict()
    for k, v in lora_params.items():
        if k in model_state and model_state[k].shape == v.shape:
            # We want to write directly to parameter tensors
            parts = k.split(".")
            submod = model
            for part in parts[:-1]:
                submod = getattr(submod, part)
            param = getattr(submod, parts[-1])
            with torch.no_grad():
                param.copy_(v)
