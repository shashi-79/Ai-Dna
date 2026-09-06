"""
Converts Qwen2.5-0.5B into all 6 Recurrent Depth Architectures.
Output directories: modal/recurrent_types/type_{1..6}
"""

import os
import sys
import json
import time

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ai_dna.evolution.fusion import build_recurrent_depth_model

MODELS_DIR = os.path.join(WORKSPACE_ROOT, "modal", "text_models", "qwen2.5-0.5b")
OUT_BASE = os.path.join(WORKSPACE_ROOT, "modal", "recurrent_types")

CONFIGS = [
    {
        "type_id": 1,
        "name": "Type 1: Step-Modulated LoRA (Middle-Band SVD Centroid)",
        "strategy": "step_lora",
        "anchor_method": "middle_band",
        "rank": 16,
    },
    {
        "type_id": 2,
        "name": "Type 2: Step-Modulated LoRA (All-Layer SVD Centroid)",
        "strategy": "step_lora",
        "anchor_method": "all_layers",
        "rank": 16,
    },
    {
        "type_id": 3,
        "name": "Type 3: Step-Modulated LoRA (Layer 12 Anchor)",
        "strategy": "step_lora",
        "anchor_method": "middle_layer",
        "rank": 16,
    },
    {
        "type_id": 4,
        "name": "Type 4: Pure Recurrent (Middle-Band SVD Centroid)",
        "strategy": "pure_recurrent",
        "anchor_method": "middle_band",
        "rank": 0,
    },
    {
        "type_id": 5,
        "name": "Type 5: Pure Recurrent (All-Layer SVD Centroid)",
        "strategy": "pure_recurrent",
        "anchor_method": "all_layers",
        "rank": 0,
    },
    {
        "type_id": 6,
        "name": "Type 6: Pure Recurrent (Layer 12 Anchor)",
        "strategy": "pure_recurrent",
        "anchor_method": "middle_layer",
        "rank": 0,
    },
]


def main():
    print("=" * 80)
    print(" CONVERTING QWEN2.5-0.5B INTO ALL 6 RECURRENT DEPTH ARCHITECTURES")
    print(f" Source: {MODELS_DIR}")
    print(f" Target Directory: {OUT_BASE}")
    print("=" * 80)

    summary = []
    t_start_all = time.time()

    for cfg in CONFIGS:
        t_id = cfg["type_id"]
        out_dir = os.path.join(OUT_BASE, f"type_{t_id}")
        print(f"\n[{t_id}/6] Building {cfg['name']} -> {out_dir} ...")

        t0 = time.time()
        manifest = build_recurrent_depth_model(
            primary_dir=MODELS_DIR,
            output_dir=out_dir,
            strategy=cfg["strategy"],
            anchor_method=cfg["anchor_method"],
            rank=cfg["rank"],
            outlier_threshold=6.0,
            device="cpu",
        )
        elapsed = time.time() - t0

        summary.append({
            "type_id": t_id,
            "name": cfg["name"],
            "strategy": cfg["strategy"],
            "anchor_method": cfg["anchor_method"],
            "output_dir": out_dir,
            "params": manifest["recurrent_params"],
            "param_reduction_pct": manifest["param_reduction_pct"],
            "disk_size_mb": manifest["recurrent_disk_mb"],
            "elapsed_seconds": elapsed,
        })

    total_elapsed = time.time() - t_start_all
    print("\n" + "=" * 90)
    print(f" ALL 6 CONVERSIONS COMPLETE IN {total_elapsed:.1f}s")
    print("=" * 90)
    print(f"{'Type':<8} | {'Strategy':<16} | {'Anchor':<15} | {'Parameters':<14} | {'Disk Size':<12} | {'Reduction'}")
    print("-" * 90)
    for s in summary:
        print(f"Type {s['type_id']:<3} | {s['strategy']:<16} | {s['anchor_method']:<15} | {s['params']:,<14} | {s['disk_size_mb']:>7.2f} MB   | -{s['param_reduction_pct']:.1f}%")
    print("-" * 90)

    # Save summary report
    sum_path = os.path.join(OUT_BASE, "all_types_conversion_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved conversion summary to: {sum_path}")


if __name__ == "__main__":
    main()
