"""
Strict 95% Train / 5% Test Multi-Dataset Audio-to-Audio Parallel Training & Benchmark.
Continuous Generative Acoustic Transformation & Restoration:
1. Speech Denoising & Acoustic Enhancement (Noisy Speech -> Clean Speech)
2. Acoustic Inpainting & Temporal Packet Restoration (Masked Audio -> Restored Audio)
3. Harmonic Super-Resolution & Bandwidth Extension (Low-Band Audio -> Full-Band Audio)
4. Synthetic Acoustic Filter Inversion & Resonance Logic (Phase Distorted -> Clean Harmonic)
"""

import os
os.environ["AI_DNA_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.abspath("."))

from ai_dna.dna.structure import Genotype
from ai_dna.growth.engine import GrowthEngine
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.models.lora import replace_linear_with_lora, freeze_model_except_lora, extract_lora_parameters
from ai_dna.encoding.slow_clock import SlowClockEncoder
from data import CustomAudioProcessor, HuggingFaceAudioDataset


class AudioToAudioDataset(Dataset):
    """Memory-efficient PyTorch Dataset for Continuous Audio-to-Audio Spectrograms."""
    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        inp_spec, tgt_spec, task_name = self.samples[idx]
        return inp_spec, tgt_spec, task_name


def collate_audio2audio_fn(batch):
    inps = torch.stack([b[0] for b in batch], dim=0)
    tgts = torch.stack([b[1] for b in batch], dim=0)
    tasks = [b[2] for b in batch]
    return inps, tgts, tasks


def audio_spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combines Multi-Scale Spectral MSE with Cosine Harmonic Alignment."""
    mse_loss = F.mse_loss(pred, target)
    p_flat = pred.view(pred.size(0), -1)
    t_flat = target.view(target.size(0), -1)
    cos_sim = F.cosine_similarity(p_flat, t_flat, dim=-1).mean()
    cos_loss = 1.0 - cos_sim
    return mse_loss + 0.5 * cos_loss


def evaluate_audio2audio_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    batches = 0

    with torch.no_grad():
        for inps, tgts, _ in loader:
            inps, tgts = inps.to(device), tgts.to(device)
            h, _, _, _ = model(inps, modality="audio")
            preds = model.audio_head(h)

            loss = audio_spectral_loss(preds, tgts)
            mse = F.mse_loss(preds, tgts)
            cos = F.cosine_similarity(preds.view(preds.size(0), -1), tgts.view(tgts.size(0), -1), dim=-1).mean()

            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()
            batches += 1

    avg_loss = total_loss / max(1, batches)
    avg_mse = total_mse / max(1, batches)
    avg_cos = (total_cos / max(1, batches)) * 100.0  # Cosine Fidelity %
    return avg_loss, avg_mse, avg_cos


def build_audio2audio_datasets():
    """Builds paired (degraded_input, clean_target) spectrograms across multi-task domains."""
    print("  [*] Generating Multi-Task Audio-to-Audio Paired Spectrograms...")
    
    # 1. Speech Commands -> Speech Denoising
    ds_speech = HuggingFaceAudioDataset(dataset_name="speech_commands", subset="v0.02", split="train", max_samples=3000, seq_len=16, n_mels=80)
    denoise_samples = []
    for s in ds_speech.samples:
        clean = s[0].clone()
        noise = torch.randn_like(clean) * 0.4
        noisy = clean + noise
        denoise_samples.append((noisy, clean, "Speech Denoising"))

    # 2. ESC-50 -> Acoustic Inpainting / Restoration (Missing Temporal Chunks)
    ds_esc = HuggingFaceAudioDataset(dataset_name="ashraq/esc50", subset=None, split="train", max_samples=1000, seq_len=16, n_mels=80)
    inpaint_samples = []
    for s in ds_esc.samples:
        clean = s[0].clone()
        masked = clean.clone()
        # Drop 4 continuous temporal time-slices (packet loss simulation)
        t_start = random.randint(2, 10)
        masked[t_start : t_start + 4, :] = 0.0
        inpaint_samples.append((masked, clean, "Acoustic Inpainting"))

    # 3. GTZAN -> Harmonic Super-Resolution (Bandwidth Extension)
    ds_gtzan = HuggingFaceAudioDataset(dataset_name="marsyas/gtzan", subset=None, split="train", max_samples=500, seq_len=16, n_mels=80)
    superres_samples = []
    for s in ds_gtzan.samples:
        clean = s[0].clone()
        lowband = clean.clone()
        # Suppress high-frequency mel bands (frequencies > 40)
        lowband[:, 40:] = 0.0
        superres_samples.append((lowband, clean, "Harmonic Super-Resolution"))

    # 4. Synthetic Acoustic Filter Inversion
    synth_samples = []
    for i in range(500):
        cls_idx = i % 5
        clean = torch.randn(16, 80) * 0.05
        f1 = cls_idx * 15 + 5
        f2 = min(79, f1 + 20)
        clean[:, f1 : f1 + 6] += 2.5
        clean[:, f2 : f2 + 6] += 1.5
        
        # Phase distortion
        distorted = clean.clone()
        distorted = distorted * torch.cos(torch.linspace(0, math.pi * 3, 16)).unsqueeze(1) + torch.randn_like(clean) * 0.2
        synth_samples.append((distorted, clean, "Synthetic Acoustic Filter Inversion"))

    all_named = {
        "Speech Denoising": denoise_samples,
        "Acoustic Inpainting": inpaint_samples,
        "Harmonic Super-Resolution": superres_samples,
        "Synthetic Acoustic Inversion": synth_samples,
    }

    train_splits = {}
    test_splits = {}
    all_train = []

    for name, raw in all_named.items():
        random.seed(42)
        random.shuffle(raw)
        split_idx = int(len(raw) * 0.95)
        train_p = raw[:split_idx]
        test_p = raw[split_idx:]
        train_splits[name] = train_p
        test_splits[name] = test_p
        all_train.extend(train_p)
        print(f"  [+] {name:<32}: Total={len(raw):5d} | Train (95%)={len(train_p):5d} | Test (5%)={len(test_p):4d}")

    random.seed(42)
    random.shuffle(all_train)
    total_test = sum(len(v) for v in test_splits.values())
    print(f"\n[+] Total Unified Training Audio-to-Audio Pairs (95%): {len(all_train):,}")
    print(f"[+] Total Isolated Test Audio-to-Audio Pairs (5%):     {total_test:,}")

    return all_train, test_splits


def run_audio_to_audio_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print(" STRICT 95% TRAIN / 5% TEST MULTI-DATASET AUDIO-TO-AUDIO PARALLEL BENCHMARK")
    print(f" Execution Device: {device} | CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 110, flush=True)

    all_train_samples, test_splits = build_audio2audio_datasets()
    batch_size = 64

    unified_train_ds = AudioToAudioDataset(all_train_samples)
    train_loader = DataLoader(unified_train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_audio2audio_fn, pin_memory=True)

    test_loaders = {}
    for name, test_items in test_splits.items():
        ds = AudioToAudioDataset(test_items)
        test_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_audio2audio_fn, pin_memory=True)

    # 2. Setup Architectures
    genotype_template = Genotype.create_default(genotype_id="audio2audio_multitask_root")
    genotype_template.dna_architecture.d_model = 128
    genotype_template.dna_architecture.num_layers = 4
    genotype_template.dna_architecture.num_heads = 4
    genotype_template.dna_architecture.num_experts = 4
    genotype_template.dna_architecture.d_expert_hidden = 256
    genotype_template.dna_architecture.lora_rank = 8

    # MODEL 1: Standard Baseline Phenotype
    growth_engine = GrowthEngine(device=device)
    model_std = growth_engine.grow_phenotype_model(genotype_template).to(device)
    model_std.audio_head = nn.Sequential(
        nn.LayerNorm(128),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 80)
    ).to(device)

    # MODEL 2: AI-DNA Phenotype
    model_lora_evolved = growth_engine.grow_phenotype_model(genotype_template).to(device)
    model_lora_evolved.audio_head = nn.Sequential(
        nn.LayerNorm(128),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 80)
    ).to(device)

    optimizer_std = torch.optim.AdamW(model_std.parameters(), lr=1e-3, weight_decay=1e-4)

    # =========================================================================
    # >>> 3. TRAIN MODEL 1: Standard Baseline Phenotype Model
    # =========================================================================
    print("\n" + "=" * 110)
    print(" >>> TRAINING MODEL 1: Standard Baseline Model (Full Weights on Audio-to-Audio)")
    print("=" * 110, flush=True)

    time_std_start = time.time()
    for epoch in range(1, 5):
        model_std.train()
        total_loss, total_mse, total_items = 0.0, 0.0, 0
        t0 = time.time()

        for inps, tgts, _ in train_loader:
            inps, tgts = inps.to(device), tgts.to(device)
            optimizer_std.zero_grad()

            h, aux_loss, _, _ = model_std(inps, modality="audio")
            preds = model_std.audio_head(h)
            loss = audio_spectral_loss(preds, tgts) + 0.01 * aux_loss
            loss.backward()
            optimizer_std.step()

            total_loss += loss.item() * tgts.size(0)
            total_mse += F.mse_loss(preds, tgts).item() * tgts.size(0)
            total_items += tgts.size(0)

        ep_loss = total_loss / max(1, total_items)
        ep_mse = total_mse / max(1, total_items)
        print(f"  Standard Model | Epoch {epoch}/4 | Train Loss: {ep_loss:.4f} | Spec MSE: {ep_mse:.6f} | Epoch Time: {time.time()-t0:.2f}s", flush=True)

    time_std_train = time.time() - time_std_start

    # =========================================================================
    # >>> 4. TRAIN MODEL 2: LoRA + CPPN AI-DNA Evolution (5 Generations)
    # =========================================================================
    print("\n" + "=" * 110)
    print(" >>> TRAINING MODEL 2: LoRA + CPPN AI-DNA Evolution (5 Generations on Audio-to-Audio)")
    print("=" * 110, flush=True)

    slow_clock = SlowClockEncoder(device=device, encoder_steps=120)

    # Gen 0: Base Audio-to-Audio Foundation Initiation
    print("\n  [Generation 0: Initiation] Training base audio-to-audio foundation unfrozen...")
    opt_init = torch.optim.AdamW(model_lora_evolved.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    model_lora_evolved.train()
    total_loss, total_mse, total_items = 0.0, 0.0, 0
    for inps, tgts, _ in train_loader:
        inps, tgts = inps.to(device), tgts.to(device)
        opt_init.zero_grad()
        h, aux_loss, _, _ = model_lora_evolved(inps, modality="audio")
        preds = model_lora_evolved.audio_head(h)
        loss = audio_spectral_loss(preds, tgts) + 0.01 * aux_loss
        loss.backward()
        opt_init.step()
        total_loss += loss.item() * tgts.size(0)
        total_mse += F.mse_loss(preds, tgts).item() * tgts.size(0)
        total_items += tgts.size(0)

    gen0_loss = total_loss / max(1, total_items)
    gen0_mse = total_mse / max(1, total_items)
    print(f"  [Gen 0 Initiation] Loss: {gen0_loss:.4f} | Spec MSE: {gen0_mse:.6f} | Time: {time.time()-t0:.2f}s", flush=True)

    # Encode Base Foundation into Genotype D_0
    print("  [Gen 0 Initiation] Encoding foundation weights into Base Genotype D_0 via Slow Clock...", flush=True)
    genotype_d0, _ = slow_clock.step(
        genotype_template,
        model_lora_evolved.state_dict(),
        phenotype_model=model_lora_evolved,
        growth_engine=growth_engine,
        protect_ancestral=False,
    )
    current_genotype_lora = genotype_d0.clone("audio2audio_dna_gen0")

    num_generations = 5
    lora_history = []
    total_lora_train_time = 0.0
    total_slow_clock_time = 0.0

    for gen in range(1, num_generations + 1):
        print(f"\n  --- [Generation {gen}/{num_generations}] LoRA Fast Adaptation + Slow Clock Distillation ---")

        # 1. Regrow phenotype model for this generation
        model_gen = growth_engine.grow_phenotype_model(current_genotype_lora).to(device)
        model_gen.train()
        freeze_model_except_lora(model_gen)
        
        # Keep audio_encoder and audio_head active for continuous spectral adaptation
        for p in model_gen.audio_encoder.parameters():
            p.requires_grad = True
        for p in model_gen.audio_head.parameters():
            p.requires_grad = True

        optimizer_gen = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_gen.parameters()),
            lr=2e-3,
            weight_decay=1e-4,
        )

        t_fast_start = time.time()
        total_loss, total_mse, total_items = 0.0, 0.0, 0
        for inps, tgts, _ in train_loader:
            inps, tgts = inps.to(device), tgts.to(device)
            optimizer_gen.zero_grad()
            h, aux_loss, _, _ = model_gen(inps, modality="audio")
            preds = model_gen.audio_head(h)
            loss = audio_spectral_loss(preds, tgts) + 0.01 * aux_loss
            loss.backward()
            optimizer_gen.step()
            total_loss += loss.item() * tgts.size(0)
            total_mse += F.mse_loss(preds, tgts).item() * tgts.size(0)
            total_items += tgts.size(0)

        gen_loss = total_loss / max(1, total_items)
        gen_mse = total_mse / max(1, total_items)
        fast_time = time.time() - t_fast_start
        total_lora_train_time += fast_time

        # 2. Slow Clock Distillation
        t_slow_start = time.time()
        next_genotype, slow_summary = slow_clock.step(
            current_genotype_lora,
            model_gen.state_dict(),
            phenotype_model=model_gen,
            growth_engine=growth_engine,
            protect_ancestral=(gen < num_generations),
        )
        current_genotype_lora = next_genotype
        slow_clock_time = time.time() - t_slow_start
        total_slow_clock_time += slow_clock_time

        dna_param_count = current_genotype_lora.total_parameters()
        recon_loss = slow_summary.get("recon_loss", 0.0) if isinstance(slow_summary, dict) else 0.0

        lora_history.append({
            "generation": gen,
            "train_loss": gen_loss,
            "train_mse": gen_mse,
            "fast_time_sec": fast_time,
            "slow_clock_time_sec": slow_clock_time,
            "dna_params": dna_param_count,
            "recon_loss": recon_loss,
        })

        print(f"  Generation {gen} Complete | Train Loss: {gen_loss:.4f} | Spec MSE: {gen_mse:.6f} | Fast Time: {fast_time:.1f}s | Slow Time: {slow_clock_time:.1f}s | DNA Size: {dna_param_count} params", flush=True)

    # 4. Regrow Final Phenotype W_5 from Genotype D_5
    print("\n[+] Regrowing Final 5th Generation Audio-to-Audio Phenotype W_5 from Genotype D_5...")
    model_lora_evolved = growth_engine.grow_phenotype_model(current_genotype_lora).to(device)

    # =========================================================================
    # >>> 5. HELD-OUT 5% TEST EVALUATION PER DOMAIN
    # =========================================================================
    print("\n" + "=" * 110)
    print(" 5% HELD-OUT TEST EVALUATION PER AUDIO-TO-AUDIO TASK (Standard vs. LoRA+CPPN Gen 5)")
    print("=" * 110, flush=True)

    results_table = []
    summary_data = {
        "total_train_samples": len(all_train_samples),
        "total_test_samples": sum(len(v) for v in test_splits.values()),
        "standard_model": {"datasets": {}, "train_time_sec": time_std_train},
        "lora_cppn_model": {
            "datasets": {},
            "train_time_sec": total_lora_train_time,
            "slow_clock_time_sec": total_slow_clock_time,
            "num_generations": num_generations,
            "generational_history": lora_history,
        },
    }

    for name, tloader in test_loaders.items():
        loss_std, mse_std, cos_std = evaluate_audio2audio_model(model_std, tloader, device)
        loss_lora, mse_lora, cos_lora = evaluate_audio2audio_model(model_lora_evolved, tloader, device)

        summary_data["standard_model"]["datasets"][name] = {"loss": loss_std, "mse": mse_std, "cosine_fidelity": cos_std}
        summary_data["lora_cppn_model"]["datasets"][name] = {"loss": loss_lora, "mse": mse_lora, "cosine_fidelity": cos_lora}

        results_table.append({
            "dataset": name,
            "std_loss": loss_std,
            "std_mse": mse_std,
            "std_cos": cos_std,
            "lora_loss": loss_lora,
            "lora_mse": mse_lora,
            "lora_cos": cos_lora,
            "test_samples": len(test_splits[name]),
        })

    print(f"\n{'Task Domain':<32} | {'Test Size':<10} | {'Std Loss':<11} | {'Std Cos %':<11} | {'DNA Loss':<11} | {'DNA Cos %':<11}")
    print("-" * 95)
    for r in results_table:
        print(f"{r['dataset']:<32} | {r['test_samples']:<10d} | {r['std_loss']:<11.4f} | {r['std_cos']:<10.1f}% | {r['lora_loss']:<11.4f} | {r['lora_cos']:<10.1f}%")

    # 6. Side-by-Side Sample Spectral Reconstruction Quality
    print("\n" + "=" * 110)
    print(" SIDE-BY-SIDE SPECTRAL RECONSTRUCTION ON UNSEEN 5% TEST AUDIO")
    print("=" * 110, flush=True)

    model_std.eval()
    model_lora_evolved.eval()

    sample_count = 0
    with torch.no_grad():
        for name, tloader in test_loaders.items():
            for inps, tgts, tasks in tloader:
                inps, tgts = inps.to(device), tgts.to(device)
                h_s, _, _, _ = model_std(inps, modality="audio")
                p_s = model_std.audio_head(h_s)

                h_l, _, _, _ = model_lora_evolved(inps, modality="audio")
                p_l = model_lora_evolved.audio_head(h_l)

                for i in range(min(1, inps.size(0))):
                    sample_count += 1
                    err_s = F.mse_loss(p_s[i], tgts[i]).item()
                    err_l = F.mse_loss(p_l[i], tgts[i]).item()
                    cos_s = F.cosine_similarity(p_s[i].flatten(), tgts[i].flatten(), dim=0).item() * 100.0
                    cos_l = F.cosine_similarity(p_l[i].flatten(), tgts[i].flatten(), dim=0).item() * 100.0
                    
                    print(f"\n[Sample #{sample_count} | Domain: {tasks[i]}]")
                    print(f"  Standard Model Error (MSE): {err_s:.6f} | Cosine Fidelity: {cos_s:.2f}%")
                    print(f"  AI-DNA (W5) Regrowth (MSE) : {err_l:.6f} | Cosine Fidelity: {cos_l:.2f}%")
                break

    # 7. Parameter & Storage Footprint
    std_params = sum(p.numel() for p in model_std.parameters())
    dna_params = current_genotype_lora.total_parameters()
    c_r = std_params / max(1, dna_params)

    summary_data["compression_ratio"] = c_r
    summary_data["std_params"] = std_params
    summary_data["dna_params"] = dna_params

    print("\n" + "=" * 110)
    print(" AUDIO-TO-AUDIO STORAGE & PARAMETER COMPRESSION SUMMARY")
    print("=" * 110)
    print(f"  Standard Phenotype Parameters: {std_params:,} parameters ({std_params*2/1024/1024:.2f} MB in FP16)")
    print(f"  Genotype AI-DNA Parameters:    {dna_params:,} parameters ({dna_params*4/1024:.2f} KB)")
    print(f"  True Compression Ratio (C_R):  {c_r:.2f}x compression")
    print(f"  Training Time: Standard={time_std_train:.1f}s | LoRA (5 Gens)={total_lora_train_time:.1f}s | SlowClock={total_slow_clock_time:.1f}s")
    print("=" * 110)

    # Save Results & Checkpoints
    torch.save(model_std.state_dict(), "checkpoint_audio2audio_standard.pt")
    torch.save(current_genotype_lora, "checkpoint_audio2audio_aidna.pt")
    with open("results_audio_to_audio_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n[+] Complete audio-to-audio benchmark results saved to results_audio_to_audio_benchmark.json", flush=True)


if __name__ == "__main__":
    run_audio_to_audio_benchmark()
