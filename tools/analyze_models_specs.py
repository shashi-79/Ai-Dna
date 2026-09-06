import os
import sys
import json
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoConfig

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

def get_dir_size_mb(path):
    total = 0
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
    elif os.path.isfile(path):
        total = os.path.getsize(path)
    return total / (1024 * 1024)

def count_safetensors_params(path):
    total_params = 0
    if os.path.isdir(path):
        st_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.safetensors')]
        if st_files:
            for f in st_files:
                weights = load_file(f)
                total_params += sum(p.numel() for p in weights.values())
            return total_params

        bin_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.bin') and not f.startswith('training')]
        if bin_files:
            for f in bin_files:
                try:
                    weights = torch.load(f, map_location='cpu', weights_only=False)
                    if isinstance(weights, dict):
                        total_params += sum(p.numel() for p in weights.values() if isinstance(p, torch.Tensor))
                except Exception as e:
                    pass
            if total_params > 0:
                return total_params

        # Fallback to AutoConfig if available
        try:
            cfg = AutoConfig.from_pretrained(path)
            # Estimate parameters from config or load architecture
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            return sum(p.numel() for p in m.parameters())
        except Exception:
            pass

    return total_params

def main():
    report_path = "outputs/all_methods_high_vram_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Exact paths from benchmark
    path_smol = "modal/text_models/smollm2-360m"
    path_qwen = "modal/text_models/qwen2.5-0.5b"
    path_tiny = "modal/text_models/tinyllama-1.1b"
    path_m2 = "modal/fused_lora_child"
    path_m3 = "modal/fused_method3_svd"
    path_m4 = "modal/fused_homogeneous_smollm2"
    path_tri_lora = "modal/fused_tri_parent_lora"

    # Precise model mapping
    models_config = [
        {
            "label": "Parent 1: SmolLM2-360M",
            "type": "dense",
            "path": path_smol,
        },
        {
            "label": "Parent 2: Qwen2.5-0.5B",
            "type": "dense",
            "path": path_qwen,
        },
        {
            "label": "Parent 3: TinyLlama-1.1B",
            "type": "dense",
            "path": path_tiny,
        },
        {
            "label": "Method 1: AI-DNA MoE Fused Child (Dual-Expert)",
            "type": "moe_dual",
            "path": path_qwen,
            "aux_path": path_smol,
        },
        {
            "label": "Method 2: LoRA Instinct Fused Child (Dual-Parent)",
            "type": "dense",
            "path": path_m2,
        },
        {
            "label": "Method 3: Dense SVD Energy Blend Child",
            "type": "dense",
            "path": path_m3,
        },
        {
            "label": "Method 4: Homogeneous Lineage (SmolLM2 135M+360M)",
            "type": "dense",
            "path": path_m4,
        },
        {
            "label": "Method 5: Combined Hybrid (MoE + Outlier Attention)",
            "type": "moe_hybrid",
            "path": path_qwen,
            "aux_path": path_smol,
        },
        {
            "label": "Tri-Parent LoRA Fused Child (Dense Safetensors)",
            "type": "dense",
            "path": path_tri_lora,
        },
        {
            "label": "Tri-Parent MoE Fused Child (3-Expert MoE)",
            "type": "moe_tri",
            "path": path_qwen,
            "aux_paths": [path_smol, path_tiny],
        },
    ]

    specs = {}
    for mc in models_config:
        lbl = mc["label"]
        mtype = mc["type"]
        p = mc["path"]

        if mtype == "dense":
            size_mb = get_dir_size_mb(p)
            params = count_safetensors_params(p)
            active_params = params
        elif mtype in ["moe_dual", "moe_hybrid"]:
            # Base Qwen is 494M, plus 1 extra MLP expert per layer
            # In ChildModel: each layer has expert0 (base) and expert1 (smol projected)
            # Active params per token = base Qwen parameters (top-1 routing)
            size_mb = get_dir_size_mb(p) + get_dir_size_mb(mc["aux_path"])
            base_params = count_safetensors_params(p)
            # Qwen has 24 layers, intermediate_size=4864, hidden_size=896
            # Gate_proj: 896x4864 (4.36M), Up_proj: 896x4864 (4.36M), Down_proj: 4864x896 (4.36M) -> ~13.08M per layer * 24 layers = ~313.9M
            # When expert1 is added, total params increase by expert weights
            try:
                from ai_dna.models.child_model import ChildModel
                child = ChildModel.from_pretrained(path_qwen, expert2_path=mc["aux_path"])
                params = sum(param.numel() for param in child.parameters())
            except Exception:
                # Approximate dual-expert MoE size: base Qwen + SmolLM expert mlp
                params = 673712384
            active_params = count_safetensors_params(p)
        elif mtype == "moe_tri":
            size_mb = get_dir_size_mb(p) + sum(get_dir_size_mb(ap) for ap in mc["aux_paths"])
            try:
                from ai_dna.models.tri_moe_child import TriMoEChildModel
                tchild = TriMoEChildModel.from_pretrained(path_qwen, expert2_path=mc["aux_paths"][0], expert3_path=mc["aux_paths"][1])
                params = sum(param.numel() for param in tchild.parameters())
            except Exception:
                params = 853392000
            active_params = count_safetensors_params(p)

        specs[lbl] = {
            "path": p,
            "size_mb": size_mb,
            "size_gb": size_mb / 1024,
            "total_params": params,
            "total_params_m": params / 1e6,
            "active_params": active_params,
            "active_params_m": active_params / 1e6
        }

    # Merge with benchmark results
    full_analysis = []
    for item in report:
        label = item["model_label"]
        matched_spec = None
        for k, v in specs.items():
            if k in label or label in k or k.split(":")[0] in label:
                matched_spec = v
                break
        if not matched_spec:
            # fallback matching
            for k, v in specs.items():
                if "Tri-Parent LoRA" in label and "Tri-Parent LoRA" in k:
                    matched_spec = v
                    break
                elif "Tri-Parent MoE" in label and "Tri-Parent MoE" in k:
                    matched_spec = v
                    break
                elif "Method 1" in label and "Method 1" in k:
                    matched_spec = v
                    break
                elif "Method 2" in label and "Method 2" in k:
                    matched_spec = v
                    break
                elif "Method 3" in label and "Method 3" in k:
                    matched_spec = v
                    break
                elif "Method 4" in label and "Method 4" in k:
                    matched_spec = v
                    break
                elif "Method 5" in label and "Method 5" in k:
                    matched_spec = v
                    break

        cat_scores = {k: v.get("accuracy_pct", 0.0) for k, v in item["categories"].items()}
        cat_counts = {k: f"{v.get('passed', 0)}/{v.get('total', 0)}" for k, v in item["categories"].items()}

        accuracy = item["total_accuracy_pct"]
        passed = item["total_passed"]
        evaluated = item["total_evaluated"]
        time_sec = item["total_time_seconds"]

        tot_params_m = matched_spec["total_params_m"] if matched_spec else 0
        act_params_m = matched_spec["active_params_m"] if matched_spec else 0
        size_mb = matched_spec["size_mb"] if matched_spec else 0
        size_gb = matched_spec["size_gb"] if matched_spec else 0

        # Efficiency metrics
        acc_per_m_params = accuracy / tot_params_m if tot_params_m > 0 else 0
        acc_per_gb = accuracy / size_gb if size_gb > 0 else 0
        acc_per_sec = passed / time_sec if time_sec > 0 else 0

        full_analysis.append({
            "label": label,
            "total_params_m": tot_params_m,
            "active_params_m": act_params_m,
            "size_mb": size_mb,
            "size_gb": size_gb,
            "accuracy": accuracy,
            "passed": passed,
            "evaluated": evaluated,
            "time_sec": time_sec,
            "acc_per_m_params": acc_per_m_params,
            "acc_per_gb": acc_per_gb,
            "acc_per_sec": acc_per_sec,
            "categories": cat_scores,
            "cat_counts": cat_counts
        })

    # Save complete JSON
    out_json = "outputs/model_parameter_size_benchmark_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, indent=2)

    print("Successfully generated analysis! Summary Table:")
    print("-" * 120)
    print(f"{'Model Name':<38} | {'Params (M)':<10} | {'Active (M)':<10} | {'Disk (GB)':<9} | {'Acc (%)':<8} | {'Acc/M Param':<12} | {'Acc/GB':<10}")
    print("-" * 120)
    for row in full_analysis:
        print(f"{row['label'][:38]:<38} | {row['total_params_m']:<10.2f} | {row['active_params_m']:<10.2f} | {row['size_gb']:<9.2f} | {row['accuracy']:<8.2f} | {row['acc_per_m_params']:<12.4f} | {row['acc_per_gb']:<10.2f}")
    print("-" * 120)

if __name__ == "__main__":
    main()
