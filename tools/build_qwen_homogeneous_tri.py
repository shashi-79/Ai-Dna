"""
Builds a Qwen-hosted Homogeneous Lineage Fusion model.
Uses Qwen2.5-0.5B as the primary host backbone, with SmolLM2-360M and TinyLlama-1.1B
fused using the exact Homogeneous Lineage algorithm:
  - Full-rank tensor projection (Proj_Sigma)
  - Uniform linear convex weight averaging across ALL layers (0 to 23) without layer decoupling
  - Preserved discrete vocabulary (Qwen tokenizer & embeddings)
  - Energy conservation and Outlier Vault (>6.0 sigma)
"""

import os
import sys
import shutil
import torch
from typing import Dict, Any

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file
from ai_dna.evolution.fusion import project_sigma_energy_tensor


def build_qwen_homogeneous_fused_model(
    primary_dir: str = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "qwen2.5-0.5b"),
    donor1_dir: str = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "smollm2-360m"),
    donor2_dir: str = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "tinyllama-1.1b"),
    output_dir: str = os.path.join(WORKSPACE_ROOT, "modal", "fused_qwen_homogeneous"),
    blend_alpha1: float = 0.05,
    blend_alpha2: float = 0.05,
    outlier_threshold: float = 6.0,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Primary Qwen Backbone: {primary_dir} ...")
    prim_weights: Dict[str, torch.Tensor] = {}
    with safe_open(os.path.join(primary_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            prim_weights[k] = f.get_tensor(k)

    print(f"Loading Donor 1 (SmolLM2-360M): {donor1_dir} ...")
    d1_weights: Dict[str, torch.Tensor] = {}
    with safe_open(os.path.join(donor1_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            d1_weights[k] = f.get_tensor(k)

    print(f"Loading Donor 2 (TinyLlama-1.1B): {donor2_dir} ...")
    d2_weights: Dict[str, torch.Tensor] = {}
    with safe_open(os.path.join(donor2_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            d2_weights[k] = f.get_tensor(k)

    fused_weights: Dict[str, torch.Tensor] = {}
    qwen_layers = 24
    smol_layers = 32
    tiny_layers = 22
    modified_count = 0

    for k, w_prim in prim_weights.items():
        # Discrete vocabulary invariance
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

        smol_l = min(int(layer_idx * smol_layers / qwen_layers), smol_layers - 1)
        tiny_l = min(int(layer_idx * tiny_layers / qwen_layers), tiny_layers - 1)

        k_smol = f"model.layers.{smol_l}.{sub_key}"
        k_tiny = f"model.layers.{tiny_l}.{sub_key}"

        w_prim_f = w_prim.float()

        # Homogeneous Lineage: UNIFORM FULL-RANK BLEND ACROSS ALL 24 LAYERS
        # Does NOT anchor layers 0-5, does NOT extract low-rank SVD components
        d1_proj = None
        d2_proj = None

        if k_smol in d1_weights:
            d1_proj = project_sigma_energy_tensor(d1_weights[k_smol], w_prim.shape).float()

        if k_tiny in d2_weights:
            d2_proj = project_sigma_energy_tensor(d2_weights[k_tiny], w_prim.shape).float()

        # Convex Linear Weight Interpolation:
        a1 = blend_alpha1 if d1_proj is not None else 0.0
        a2 = blend_alpha2 if d2_proj is not None else 0.0
        a_prim = 1.0 - a1 - a2

        w_blend = a_prim * w_prim_f
        if d1_proj is not None:
            w_blend += a1 * d1_proj
        if d2_proj is not None:
            w_blend += a2 * d2_proj

        # Outlier Vault Isolation
        mu = w_prim_f.mean()
        std = w_prim_f.std() + 1e-8
        outlier_mask = (w_prim_f - mu).abs() > (outlier_threshold * std)

        # Conserve Energy
        orig_norm = torch.norm(w_prim_f)
        new_norm = torch.norm(w_blend) + 1e-8
        w_blend = w_blend * (orig_norm / new_norm)

        # Restore Outliers
        w_blend[outlier_mask] = w_prim_f[outlier_mask]

        fused_weights[k] = w_blend.to(dtype=w_prim.dtype)
        modified_count += 1

    out_st = os.path.join(output_dir, "model.safetensors")
    safetensors_save_file(fused_weights, out_st)

    for fname in os.listdir(primary_dir):
        if fname != "model.safetensors":
            s = os.path.join(primary_dir, fname)
            d = os.path.join(output_dir, fname)
            if os.path.isfile(s):
                shutil.copy2(s, d)

    print(f"[Qwen Homogeneous Fusion Complete] Saved to: {output_dir}")
    print(f"  Modified {modified_count}/{len(prim_weights)} tensors uniformly across all 24 layers.")
    return {"output_dir": output_dir, "modified_count": modified_count}


if __name__ == "__main__":
    build_qwen_homogeneous_fused_model()
