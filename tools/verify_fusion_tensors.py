import os
import sys
import torch
from safetensors.torch import load_file

def analyze_tensor_diffs(name, fused_path, parent_path):
    print(f"\n=======================================================")
    print(f"Checking: {name}")
    print(f"Fused Path: {fused_path}")
    print(f"Parent Path: {parent_path}")
    print(f"=======================================================")
    
    fused_st = os.path.join(fused_path, "model.safetensors")
    parent_st = os.path.join(parent_path, "model.safetensors")
    
    if not os.path.exists(fused_st) or not os.path.exists(parent_st):
        print(f"Error: Missing model.safetensors in {fused_path} or {parent_path}")
        return
        
    fused_sd = load_file(fused_st)
    parent_sd = load_file(parent_st)
    
    total_tensors = len(parent_sd)
    identical_tensors = 0
    modified_tensors = 0
    
    diff_norms = []
    modified_names = []
    
    for k in parent_sd.keys():
        if k not in fused_sd:
            print(f"Tensor {k} missing in fused model!")
            continue
        p_weight = parent_sd[k].float()
        f_weight = fused_sd[k].float()
        
        diff = torch.abs(f_weight - p_weight)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        norm_diff = torch.norm(f_weight - p_weight).item()
        
        if max_diff < 1e-7:
            identical_tensors += 1
        else:
            modified_tensors += 1
            diff_norms.append((k, max_diff, mean_diff, norm_diff, p_weight.numel()))
            if len(modified_names) < 10:
                modified_names.append((k, max_diff, mean_diff))
                
    pct_modified = (modified_tensors / total_tensors) * 100
    total_params = sum(p.numel() for p in parent_sd.values())
    modified_params = sum(x[4] for x in diff_norms)
    pct_params_modified = (modified_params / total_params) * 100
    
    print(f"Total Tensors: {total_tensors}")
    print(f"Identical Tensors: {identical_tensors}")
    print(f"Modified Tensors: {modified_tensors} ({pct_modified:.2f}%)")
    print(f"Total Parameters: {total_params:,}")
    print(f"Modified Parameters: {modified_params:,} ({pct_params_modified:.2f}%)")
    
    if diff_norms:
        max_entry = max(diff_norms, key=lambda x: x[1])
        avg_mean_diff = sum(x[2] for x in diff_norms) / len(diff_norms)
        print(f"Max Delta Across All Tensors: {max_entry[1]:.6f} (in {max_entry[0]})")
        print(f"Average Mean Delta Across Modified Tensors: {avg_mean_diff:.6f}")
        print("\nSample Modified Tensors:")
        for k, md, meand in modified_names[:5]:
            print(f"  - {k}: max_diff={md:.6f}, mean_diff={meand:.6f}")
    else:
        print("WARNING: MODEL IS 100% IDENTICAL TO PARENT 2!")

def main():
    parent_qwen = "modal/text_models/qwen2.5-0.5b"
    
    targets = [
        ("Method 2: LoRA Instinct Fused Child", "modal/fused_lora_child"),
        ("Method 3: Dense SVD Energy Blend Child", "modal/fused_method3_svd"),
        ("Tri-Parent LoRA Fused Child (Dense)", "modal/fused_tri_parent_lora"),
        ("my_llm_folder (Tri-Parent LoRA)", "my_llm_folder"),
    ]
    
    for name, path in targets:
        analyze_tensor_diffs(name, path, parent_qwen)

if __name__ == "__main__":
    main()
