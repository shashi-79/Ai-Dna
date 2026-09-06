import os
import torch
from safetensors.torch import load_file

def verify_tri_parent_layers():
    fused_st = "my_llm_folder/model.safetensors"
    parent_st = "modal/text_models/qwen2.5-0.5b/model.safetensors"
    
    f_sd = load_file(fused_st)
    p_sd = load_file(parent_st)
    
    print("===============================================================")
    print("TRI-PARENT LAYER-DEPTH DECOUPLING VERIFICATION")
    print("===============================================================")
    
    layer_stats = {}
    for i in range(24):
        layer_stats[i] = {"total_tensors": 0, "modified_tensors": 0, "max_diff": 0.0, "mean_diff": 0.0, "mod_names": []}
        
    for k in p_sd.keys():
        if "model.layers." in k:
            parts = k.split(".")
            layer_idx = int(parts[2])
            
            p_w = p_sd[k].float()
            f_w = f_sd[k].float()
            diff = torch.abs(f_w - p_w)
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            
            layer_stats[layer_idx]["total_tensors"] += 1
            if max_d > 1e-7:
                layer_stats[layer_idx]["modified_tensors"] += 1
                layer_stats[layer_idx]["max_diff"] = max(layer_stats[layer_idx]["max_diff"], max_d)
                layer_stats[layer_idx]["mean_diff"] += mean_d
                layer_stats[layer_idx]["mod_names"].append(parts[-2] + "." + parts[-1])
                
    print(f"{'Layer Range':<22} | {'Role in Tri-Parent':<26} | {'Mod Tensors':<12} | {'Max Delta':<10} | {'Status'}")
    print("-" * 85)
    
    # Range 0-5
    mod_0_5 = sum(layer_stats[i]["modified_tensors"] for i in range(6))
    tot_0_5 = sum(layer_stats[i]["total_tensors"] for i in range(6))
    max_0_5 = max(layer_stats[i]["max_diff"] for i in range(6))
    print(f"{'Layers 0 - 5 (Shallow)':<22} | {'Pure Qwen (Grammar Anchor)':<26} | {f'{mod_0_5}/{tot_0_5}':<12} | {max_0_5:<10.6f} | {'100% IDENTICAL (Intended)' if mod_0_5 == 0 else 'Modified'}")
    
    # Range 6-15
    mod_6_15 = sum(layer_stats[i]["modified_tensors"] for i in range(6, 16))
    tot_6_15 = sum(layer_stats[i]["total_tensors"] for i in range(6, 16))
    max_6_15 = max(layer_stats[i]["max_diff"] for i in range(6, 16))
    print(f"{'Layers 6 - 15 (Middle)':<22} | {'TinyLlama (World/Hist/Geo)':<26} | {f'{mod_6_15}/{tot_6_15}':<12} | {max_6_15:<10.6f} | {'FUSED (Active Delta)' if mod_6_15 > 0 else 'Unchanged'}")
    
    # Range 16-23
    mod_16_23 = sum(layer_stats[i]["modified_tensors"] for i in range(16, 24))
    tot_16_23 = sum(layer_stats[i]["total_tensors"] for i in range(16, 24))
    max_16_23 = max(layer_stats[i]["max_diff"] for i in range(16, 24))
    print(f"{'Layers 16 - 23 (Deep)':<22} | {'SmolLM2 (Code/Algorithmic)':<26} | {f'{mod_16_23}/{tot_16_23}':<12} | {max_16_23:<10.6f} | {'FUSED (Active Delta)' if mod_16_23 > 0 else 'Unchanged'}")
    print("-" * 85)

if __name__ == "__main__":
    verify_tri_parent_layers()
