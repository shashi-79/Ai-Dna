"""
Strict 95% Train / 5% Test Multi-Dataset Parallel Audio Training & Evaluation Benchmark.
Ingests audio datasets across diverse acoustic domains:
1. Speech Commands (Keyword Spotting & Spoken Audio - 35 classes)
2. ESC-50 (Environmental Sound & Acoustic Events - 50 classes)
3. GTZAN (Music Harmonic & Genre Recognition - 10 classes)
4. Synthetic Acoustic Logic (Frequency Sweep & Resonance Logic - 5 classes)
Total: 100 Unified Audio Classes
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
from ai_dna.models.modules import ClassificationHead
from ai_dna.models.lora import replace_linear_with_lora, freeze_model_except_lora, extract_lora_parameters
from ai_dna.encoding.slow_clock import SlowClockEncoder
from data import CustomAudioProcessor, HuggingFaceAudioDataset


class UnifiedAudioDataset(Dataset):
    """Memory-efficient PyTorch Dataset for Multi-Task Audio Spectrograms."""
    def __init__(self, samples: List[Tuple[torch.Tensor, int, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        spec, label, task_name = self.samples[idx]
        return spec, torch.tensor(label, dtype=torch.long), task_name


def collate_audio_fn(batch):
    specs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    tasks = [b[2] for b in batch]
    return specs, labels, tasks


def evaluate_audio_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batches = 0

    with torch.no_grad():
        for specs, labels, _ in loader:
            specs, labels = specs.to(device), labels.to(device)
            # Forward pass with audio modality
            h, _, _, _ = model(specs, modality="audio")
            logits = model.cls_head(h)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            batches += 1

    avg_loss = total_loss / max(1, batches)
    acc = (total_correct / max(1, total_samples)) * 100.0
    return avg_loss, acc


def generate_synthetic_acoustic_data(num_samples: int = 1000, seq_len: int = 16, n_mels: int = 80, class_offset: int = 95):
    """Generates synthetic acoustic patterns with distinct frequency harmonics."""
    samples = []
    for i in range(num_samples):
        cls_idx = i % 5
        spec = torch.randn(seq_len, n_mels) * 0.05
        # Harmonic resonance bands
        f1 = cls_idx * 15 + 5
        f2 = min(79, f1 + 20)
        spec[:, f1 : f1 + 6] += 2.5
        spec[:, f2 : f2 + 6] += 1.5
        # Temporal modulation
        t_mod = torch.sin(torch.linspace(0, (cls_idx + 1) * math.pi, seq_len)).unsqueeze(1)
        spec = spec + t_mod * 0.8
        samples.append((spec, class_offset + cls_idx, "Synthetic Acoustic Logic"))
    return samples


def run_audio_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print(" STRICT 95% TRAIN / 5% TEST MULTI-DATASET AUDIO PARALLEL TRAINING & BENCHMARK")
    print(f" Execution Device: {device} | CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 110, flush=True)

    # 1. Ingest Audio Datasets
    print("\n[+] 1. Ingesting Multi-Task Audio Datasets (95% Train / 5% Test)...")
    dataset_splits = {}
    all_train_samples = []
    
    # A. Speech Commands (35 spoken classes)
    print("  [*] Loading Speech Commands (Keyword Spotting)...")
    try:
        ds_speech = HuggingFaceAudioDataset(dataset_name="speech_commands", subset="v0.02", split="train", max_samples=4000, seq_len=16, n_mels=80)
    except Exception:
        ds_speech = HuggingFaceAudioDataset(dataset_name="speech_commands", subset="v0.02", split="train", max_samples=1000, seq_len=16, n_mels=80)
    speech_samples = [(s[0], int(s[1]), "Speech Commands (Words)") for s in ds_speech.samples]

    # B. ESC-50 (50 environmental classes, offset by +35)
    print("  [*] Loading ESC-50 (Environmental Sound Classification)...")
    try:
        ds_esc = HuggingFaceAudioDataset(dataset_name="ashraq/esc50", subset=None, split="train", max_samples=1000, seq_len=16, n_mels=80)
    except Exception:
        ds_esc = HuggingFaceAudioDataset(dataset_name="ashraq/esc50", subset=None, split="train", max_samples=500, seq_len=16, n_mels=80)
    esc_samples = [(s[0], int(s[1]) % 50 + 35, "ESC-50 (Environment)") for s in ds_esc.samples]

    # C. GTZAN (10 music genre classes, offset by +85)
    print("  [*] Loading GTZAN (Music Genre Classification)...")
    try:
        ds_gtzan = HuggingFaceAudioDataset(dataset_name="marsyas/gtzan", subset=None, split="train", max_samples=500, seq_len=16, n_mels=80)
    except Exception:
        ds_gtzan = HuggingFaceAudioDataset(dataset_name="marsyas/gtzan", subset=None, split="train", max_samples=250, seq_len=16, n_mels=80)
    gtzan_samples = [(s[0], int(s[1]) % 10 + 85, "GTZAN (Music Genres)") for s in ds_gtzan.samples]

    # D. Synthetic Acoustic Developmental Patterns (5 classes, offset by +95)
    print("  [*] Generating Synthetic Acoustic Logic Patterns...")
    synth_samples = generate_synthetic_acoustic_data(num_samples=500, seq_len=16, n_mels=80, class_offset=95)

    all_named_datasets = {
        "Speech Commands (Words)": speech_samples,
        "ESC-50 (Environment)": esc_samples,
        "GTZAN (Music Genres)": gtzan_samples,
        "Synthetic Acoustic Logic": synth_samples,
    }

    test_splits = {}
    for name, raw_samples in all_named_datasets.items():
        random.seed(42)
        random.shuffle(raw_samples)
        split_idx = int(len(raw_samples) * 0.95)
        train_p = raw_samples[:split_idx]
        test_p = raw_samples[split_idx:]
        dataset_splits[name] = train_p
        test_splits[name] = test_p
        all_train_samples.extend(train_p)
        print(f"  [+] {name:<30}: Total={len(raw_samples):5d} | Train (95%)={len(train_p):5d} | Test (5%)={len(test_p):4d}")

    random.seed(42)
    random.shuffle(all_train_samples)
    total_test_samples = sum(len(v) for v in test_splits.values())
    print(f"\n[+] Total Unified Training Audio Samples (95%): {len(all_train_samples):,}")
    print(f"[+] Total Isolated Test Audio Samples (5%):     {total_test_samples:,}")

    batch_size = 64
    unified_train_ds = UnifiedAudioDataset(all_train_samples)
    train_loader = DataLoader(unified_train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_audio_fn, pin_memory=True)

    test_loaders = {}
    for name, test_items in test_splits.items():
        ds = UnifiedAudioDataset(test_items)
        test_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_audio_fn, pin_memory=True)

    # 2. Setup Architectures
    num_classes = 100
    genotype_template = Genotype.create_default(genotype_id="audio_multitask_root")
    genotype_template.dna_architecture.d_model = 128
    genotype_template.dna_architecture.num_layers = 4
    genotype_template.dna_architecture.num_heads = 4
    genotype_template.dna_architecture.num_experts = 4
    genotype_template.dna_architecture.d_expert_hidden = 256
    genotype_template.dna_architecture.lora_rank = 8
    genotype_template.dna_architecture.num_classes = num_classes

    # MODEL 1: Standard Baseline
    growth_engine = GrowthEngine(device=device)
    model_std = growth_engine.grow_phenotype_model(genotype_template).to(device)
    model_std.cls_head = ClassificationHead(d_model=128, num_classes=num_classes).to(device)

    # MODEL 2: AI-DNA Phenotype
    model_lora_evolved = growth_engine.grow_phenotype_model(genotype_template).to(device)
    model_lora_evolved.cls_head = ClassificationHead(d_model=128, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_std = torch.optim.AdamW(model_std.parameters(), lr=1e-3, weight_decay=1e-4)

    # =========================================================================
    # >>> 3. TRAIN MODEL 1: Standard Baseline Phenotype Model
    # =========================================================================
    print("\n" + "=" * 110)
    print(" >>> TRAINING MODEL 1: Standard Baseline Model (Full Weights on Unified Audio)")
    print("=" * 110, flush=True)

    time_std_start = time.time()
    for epoch in range(1, 5):
        model_std.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        t0 = time.time()

        for specs, labels, _ in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer_std.zero_grad()

            h, aux_loss, _, _ = model_std(specs, modality="audio")
            logits = model_std.cls_head(h)
            loss = criterion(logits, labels) + 0.01 * aux_loss
            loss.backward()
            optimizer_std.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_items += labels.size(0)

        ep_loss = total_loss / max(1, total_items)
        ep_acc = (total_correct / max(1, total_items)) * 100.0
        print(f"  Standard Model | Epoch {epoch}/4 | Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc:.1f}% | Epoch Time: {time.time()-t0:.2f}s", flush=True)

    time_std_train = time.time() - time_std_start

    # =========================================================================
    # >>> 4. TRAIN MODEL 2: LoRA + CPPN AI-DNA Evolution (5 Generations)
    # =========================================================================
    print("\n" + "=" * 110)
    print(" >>> TRAINING MODEL 2: LoRA + CPPN AI-DNA Evolution (5 Generations on Unified Audio)")
    print("=" * 110, flush=True)

    slow_clock = SlowClockEncoder(device=device, encoder_steps=120)

    # Gen 0: Base Foundation Initiation
    print("\n  [Generation 0: Initiation] Training base audio foundation unfrozen...")
    opt_init = torch.optim.AdamW(model_lora_evolved.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    model_lora_evolved.train()
    total_loss, total_correct, total_items = 0.0, 0, 0
    for specs, labels, _ in train_loader:
        specs, labels = specs.to(device), labels.to(device)
        opt_init.zero_grad()
        h, aux_loss, _, _ = model_lora_evolved(specs, modality="audio")
        logits = model_lora_evolved.cls_head(h)
        loss = criterion(logits, labels) + 0.01 * aux_loss
        loss.backward()
        opt_init.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (torch.argmax(logits, dim=-1) == labels).sum().item()
        total_items += labels.size(0)

    gen0_loss = total_loss / max(1, total_items)
    gen0_acc = (total_correct / max(1, total_items)) * 100.0
    print(f"  [Gen 0 Initiation] Loss: {gen0_loss:.4f} | Acc: {gen0_acc:.1f}% | Time: {time.time()-t0:.2f}s", flush=True)

    # Encode Base Foundation into Genotype D_0
    print("  [Gen 0 Initiation] Encoding foundation weights into Base Genotype D_0 via Slow Clock...", flush=True)
    genotype_d0, _ = slow_clock.step(
        genotype_template,
        model_lora_evolved.state_dict(),
        phenotype_model=model_lora_evolved,
        growth_engine=growth_engine,
        protect_ancestral=False,
    )
    current_genotype_lora = genotype_d0.clone("audio_dna_gen0")

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
        
        # Keep audio_encoder and cls_head active for adaptation
        for p in model_gen.audio_encoder.parameters():
            p.requires_grad = True
        for p in model_gen.cls_head.parameters():
            p.requires_grad = True

        optimizer_gen = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_gen.parameters()),
            lr=2e-3,
            weight_decay=1e-4,
        )

        t_fast_start = time.time()
        total_loss, total_correct, total_items = 0.0, 0, 0
        for specs, labels, _ in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer_gen.zero_grad()
            h, aux_loss, _, _ = model_gen(specs, modality="audio")
            logits = model_gen.cls_head(h)
            loss = criterion(logits, labels) + 0.01 * aux_loss
            loss.backward()
            optimizer_gen.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (torch.argmax(logits, dim=-1) == labels).sum().item()
            total_items += labels.size(0)

        gen_loss = total_loss / max(1, total_items)
        gen_acc = (total_correct / max(1, total_items)) * 100.0
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
            "train_acc": gen_acc,
            "fast_time_sec": fast_time,
            "slow_clock_time_sec": slow_clock_time,
            "dna_params": dna_param_count,
            "recon_loss": recon_loss,
        })

        print(f"  Generation {gen} Complete | Train Loss: {gen_loss:.4f} | Train Acc: {gen_acc:.1f}% | Fast Time: {fast_time:.1f}s | Slow Time: {slow_clock_time:.1f}s | DNA Size: {dna_param_count} params", flush=True)

    # 4. Regrow Final Phenotype W_5 from Genotype D_5
    print("\n[+] Regrowing Final 5th Generation Audio Phenotype W_5 from Genotype D_5...")
    model_lora_evolved = growth_engine.grow_phenotype_model(current_genotype_lora).to(device)

    # =========================================================================
    # >>> 5. HELD-OUT 5% TEST EVALUATION PER DATASET
    # =========================================================================
    print("\n" + "=" * 110)
    print(" 5% HELD-OUT TEST EVALUATION PER AUDIO DATASET (Standard vs. LoRA+CPPN Gen 5)")
    print("=" * 110, flush=True)

    results_table = []
    summary_data = {
        "total_train_samples": len(all_train_samples),
        "total_test_samples": total_test_samples,
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
        loss_std, acc_std = evaluate_audio_model(model_std, tloader, criterion, device)
        loss_lora, acc_lora = evaluate_audio_model(model_lora_evolved, tloader, criterion, device)

        summary_data["standard_model"]["datasets"][name] = {"loss": loss_std, "acc": acc_std}
        summary_data["lora_cppn_model"]["datasets"][name] = {"loss": loss_lora, "acc": acc_lora}

        results_table.append({
            "dataset": name,
            "std_loss": loss_std,
            "std_acc": acc_std,
            "lora_loss": loss_lora,
            "lora_acc": acc_lora,
            "test_samples": len(test_splits[name]),
        })

    print(f"\n{'Dataset Name':<30} | {'Test Size':<10} | {'Standard Loss':<14} | {'Standard Acc':<13} | {'LoRA+DNA Loss':<14} | {'LoRA+DNA Acc':<13}")
    print("-" * 105)
    for r in results_table:
        print(f"{r['dataset']:<30} | {r['test_samples']:<10d} | {r['std_loss']:<14.4f} | {r['std_acc']:<12.1f}% | {r['lora_loss']:<14.4f} | {r['lora_acc']:<12.1f}%")

    # 6. Side-by-Side Inference Predictions on Sample Test Spectrograms
    print("\n" + "=" * 110)
    print(" SIDE-BY-SIDE AUDIO PREDICTION ON 5% HELD-OUT SAMPLES")
    print("=" * 110, flush=True)

    model_std.eval()
    model_lora_evolved.eval()

    sample_count = 0
    with torch.no_grad():
        for name, tloader in test_loaders.items():
            for specs, labels, tasks in tloader:
                specs, labels = specs.to(device), labels.to(device)
                h_s, _, _, _ = model_std(specs, modality="audio")
                p_s = torch.argmax(model_std.cls_head(h_s), dim=-1)

                h_l, _, _, _ = model_lora_evolved(specs, modality="audio")
                p_l = torch.argmax(model_lora_evolved.cls_head(h_l), dim=-1)

                for i in range(min(2, specs.size(0))):
                    sample_count += 1
                    print(f"\n[Sample #{sample_count} | Domain: {tasks[i]}]")
                    print(f"  Target Class ID    : {labels[i].item()}")
                    print(f"  Standard Model Pred: {p_s[i].item()} {'[CORRECT]' if p_s[i].item() == labels[i].item() else '[MISS]'}")
                    print(f"  AI-DNA (W5) Pred   : {p_l[i].item()} {'[CORRECT]' if p_l[i].item() == labels[i].item() else '[MISS]'}")
                break

    # 7. Parameter & Storage Compression
    std_params = sum(p.numel() for p in model_std.parameters())
    dna_params = current_genotype_lora.total_parameters()
    c_r = std_params / max(1, dna_params)

    summary_data["compression_ratio"] = c_r
    summary_data["std_params"] = std_params
    summary_data["dna_params"] = dna_params

    print("\n" + "=" * 110)
    print(" AUDIO STORAGE & PARAMETER COMPRESSION SUMMARY")
    print("=" * 110)
    print(f"  Standard Phenotype Parameters: {std_params:,} parameters ({std_params*2/1024/1024:.2f} MB in FP16)")
    print(f"  Genotype AI-DNA Parameters:    {dna_params:,} parameters ({dna_params*4/1024:.2f} KB)")
    print(f"  True Compression Ratio (C_R):  {c_r:.2f}x compression")
    print(f"  Training Time: Standard={time_std_train:.1f}s | LoRA (5 Gens)={total_lora_train_time:.1f}s | SlowClock={total_slow_clock_time:.1f}s")
    print("=" * 110)

    # Save Results & Checkpoints
    torch.save(model_std.state_dict(), "checkpoint_audio_standard.pt")
    torch.save(current_genotype_lora, "checkpoint_audio_aidna.pt")
    with open("results_audio_multitask_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n[+] Complete audio benchmark results saved to results_audio_multitask_benchmark.json", flush=True)


if __name__ == "__main__":
    run_audio_benchmark()
