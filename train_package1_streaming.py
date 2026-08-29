"""
CLI Entry Point: Train Package 1 ("Instinct Seeding & Quick Proof-of-Concept") in Streaming Mode.
Features 0 GB disk pre-download, auto-resumable stateful checkpointing, and graceful interrupt handling.
"""

import os
import argparse
import torch
from ai_dna.dna.structure import Genotype, DNAArchitecture, DNAInstinct, DNARouting
from ai_dna.growth.engine import GrowthEngine
from training.dataset_manager import StreamDatasetManager
from training.checkpoint_manager import FailproofCheckpointManager
from training.trainer import MultiModalStreamingTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="AI-DNA Package 1 Multi-Modal Streaming Training")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per streaming step")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence context length")
    parser.add_argument("--d-model", type=int, default=128, help="Latent model dimension")
    parser.add_argument("--max-steps", type=int, default=120000, help="Total training steps (defaults to 120,000 for full pre-training)")
    parser.add_argument("--max-hours", type=float, default=None, help="Maximum training duration in hours (e.g. 5.0)")
    parser.add_argument("--save-every-steps", type=int, default=250, help="Atomic checkpoint saving interval")
    parser.add_argument("--log-every-steps", type=int, default=25, help="Metric logging interval")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/package1_streaming", help="Directory for stateful checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (defaults strictly to cuda)")
    parser.add_argument("--no-resume", action="store_true", help="Force starting fresh from step 0 instead of resuming")
    parser.add_argument("--mock-fallback", action="store_true", help="Use offline synthetic streams (for testing without internet)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Intelligent GPU Device Selection
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("\n" + "!" * 75)
            print("[NOTICE]: CUDA GPU was requested by default, but no CUDA GPU is active on this system.")
            print("   To train with maximum speed on your NVIDIA GPU (e.g. RTX 4060/4090), install CUDA PyTorch:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("   Temporarily running on CPU.")
            print("!" * 75 + "\n")
            device = torch.device("cpu")
            device_str = "CPU (Install CUDA PyTorch for NVIDIA GPU Acceleration)"
        else:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            gpu_name = torch.cuda.get_device_name(device)
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            torch.backends.cudnn.benchmark = True
            device_str = f"NVIDIA GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)"
    else:
        device = torch.device(args.device)
        device_str = f"{args.device.upper()}"

    print("\n" + "=" * 75)
    print("[AI-DNA] OMNI-MODAL FOUNDATION TRAINING (PACKAGE 1: INSTINCT SEEDING)")
    print(f"   Modality Mix: Text, Code, Math, Vision (VQA), Audio, Video, Diffusion")
    print(f"   Mode: Zero-Disk Streaming (streaming=True)")
    print(f"   Primary Hardware Device: {device_str}")
    print(f"   Checkpoints Directory: {args.checkpoint_dir}")
    print("=" * 75 + "\n")

    # 1. Initialize DNA Genotype Blueprint
    genotype = Genotype(
        dna_architecture=DNAArchitecture(
            d_model=args.d_model,
            num_layers=4,
            num_heads=4,
            num_experts=4,
            d_expert_hidden=args.d_model * 2,
            vocab_size=8192,
            kv_latent_dim=args.d_model // 4,
        ),
        dna_instinct=DNAInstinct(
            cppn_hidden_dim=32,
            cppn_layers=3,
        ),
        dna_routing=DNARouting(
            top_k_experts=2,
        )
    )

    # 2. Grow Phenotype Neural Network from DNA Kernel
    print("[GROWTH]: Synthesizing neural phenotype layers from continuous CPPN genome...")
    growth_engine = GrowthEngine()
    model = growth_engine.grow_phenotype_model(genotype)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   [+] Phenotype Grown Successfully ({total_params:,} Parameters | {args.d_model}d Latent Substrate)")

    # 3. Initialize Checkpoint Manager
    checkpoint_manager = FailproofCheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n=3,
    )

    # 4. Initialize Zero-Disk Streaming Dataset Manager on Target Device
    dataset_manager = StreamDatasetManager(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        device=device,
        use_mock_fallback=args.mock_fallback,
    )

    # 5. Initialize Failproof Multi-Modal Streaming Trainer on Target Device
    trainer = MultiModalStreamingTrainer(
        model=model,
        genotype=genotype,
        dataset_manager=dataset_manager,
        checkpoint_manager=checkpoint_manager,
        lr=args.lr,
        max_steps=args.max_steps,
        max_duration_hours=args.max_hours,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps,
        device=device,
        auto_resume=not args.no_resume,
    )

    # 6. Execute Training Loop
    trainer.train_loop()


if __name__ == "__main__":
    main()
