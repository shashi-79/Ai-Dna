import os
import sys
import torch

WORKSPACE_ROOT = os.getcwd()
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from safetensors import safe_open

def get_safetensors_info(dir_path):
    st_path = os.path.join(dir_path, "model.safetensors")
    if not os.path.exists(st_path):
        return 0, 0
    size_bytes = os.path.getsize(st_path)
    total_params = 0
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            total_params += t.numel()
    return total_params, size_bytes

models = {
    "Parent 1: SmolLM2-360M": "modal/text_models/smollm2-360m",
    "Parent 2: Qwen2.5-0.5B": "modal/text_models/qwen2.5-0.5b",
    "Parent 3: TinyLlama-1.1B": "modal/text_models/tinyllama-1.1b",
    "Method 2: LoRA Instinct Fused Child": "modal/fused_lora_child",
    "Method 3: Dense SVD Energy Blend": "modal/fused_method3_svd",
    "Method 4: Homogeneous SmolLM2": "modal/fused_homogeneous_smollm2",
    "Tri-Parent LoRA Fused Child (my_llm_folder)": "my_llm_folder",
}

print(f"{'Model Name':<42} | {'Params':<12} | {'Disk Size':<14} | {'Precision'}")
print("-" * 80)
for name, rel_path in models.items():
    p, b = get_safetensors_info(os.path.join(WORKSPACE_ROOT, rel_path))
    mb = b / (1024 * 1024)
    gb = b / (1024 * 1024 * 1024)
    print(f"{name:<42} | {p/1e6:8.2f}M    | {mb:7.2f} MB ({gb:.2f}GB) | bfloat16")

# MoE parameter calculation
from ai_dna.models.moe_child import build_fused_moe_model
from ai_dna.models.tri_moe_child import build_tri_fused_moe_model

m_dual_moe, _ = build_fused_moe_model("modal/text_models/qwen2.5-0.5b", "modal/text_models/smollm2-360m", device="cpu")
dual_moe_params = sum(p.numel() for p in m_dual_moe.parameters())
dual_moe_active = 494000000  # Top-1 routing executes 1 expert per layer
print(f"{'Method 1: AI-DNA MoE Fused Child (Dual)':<42} | {dual_moe_params/1e6:8.2f}M    | {dual_moe_params*2/(1024*1024):7.2f} MB ({dual_moe_params*2/(1024**3):.2f}GB) | bfloat16 (Active: 494M)")

m_tri_moe, _ = build_tri_fused_moe_model("modal/text_models/qwen2.5-0.5b", "modal/text_models/smollm2-360m", "modal/text_models/tinyllama-1.1b", device="cpu")
tri_moe_params = sum(p.numel() for p in m_tri_moe.parameters())
print(f"{'Method 7: Tri-Parent MoE Child (3-Expert)':<42} | {tri_moe_params/1e6:8.2f}M    | {tri_moe_params*2/(1024*1024):7.2f} MB ({tri_moe_params*2/(1024**3):.2f}GB) | bfloat16 (Active: 494M)")
