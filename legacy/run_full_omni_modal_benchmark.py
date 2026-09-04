"""
Strict 95% Train / 5% Test Omni-Modal Parallel Training & Evaluation Benchmark with SwiGLU.
Trains a Universal Omni-Modal Architecture on ALL Types of Inputs and Outputs:
Inputs:  [Text, Code, Vision (Images), Video (Spatio-Temporal), Audio (Spectrograms), Tabular (Features)]
Outputs: [Autoregressive Text/Tokens, Continuous Audio Spectrograms, Continuous Latent Diffusion, Categorical Decisions]

7 Canonical Omni-Modal Tasks:
1. Text -> Text (Math & Algorithmic Reasoning)
2. Vision -> Text (Image Perception & Captioning)
3. Video -> Text (Spatio-Temporal Action Recognition)
4. Audio -> Text (Speech Keyword Transcription)
5. Audio -> Audio (Continuous Acoustic Restoration & Denoising with SwiGLU)
6. Text -> Diffusion (Continuous Denoising / Latent Feature Generation with SwiGLU)
7. Tabular -> Decision (Structured Numerical Classification)
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


class TaskBatchDataset(Dataset):
    """Batched dataset for a specific modality task."""
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, modality_in: str, task_type: str, task_name: str):
        self.inputs = inputs
        self.targets = targets
        self.modality_in = modality_in
        self.task_type = task_type
        self.task_name = task_name

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def build_omni_task_loaders(batch_size: int = 64):
    """Builds batched DataLoaders for all 7 canonical omni-modal tasks (95% Train / 5% Test)."""
    print("  [*] Synthesizing Multi-Modal Batched Datasets across 7 Canonical Tasks with SwiGLU...")
    random.seed(42)
    torch.manual_seed(42)

    tasks_raw = {}

    # Task 1: Text -> Text (Math Reasoning)
    t1_inps, t1_tgts = [], []
    for _ in range(1200):
        a, b = random.randint(1, 50), random.randint(1, 50)
        c = a + b
        t1_inps.append([10, (a % 100) + 50, 11, (b % 100) + 50, 12])
        t1_tgts.append([(a % 100) + 50, 11, (b % 100) + 50, 12, (c % 100) + 50])
    tasks_raw["Text -> Text (Reasoning)"] = (
        torch.tensor(t1_inps, dtype=torch.long),
        torch.tensor(t1_tgts, dtype=torch.long),
        "text", "ar_text"
    )

    # Task 2: Vision -> Text (Image Captioning)
    t2_inps, t2_tgts = [], []
    for i in range(800):
        img = torch.randn(3, 32, 32) * 0.1
        shape_type = i % 4
        if shape_type == 0: img[:, 12:20, :] += 2.5; tgt = [20, 21, 22]
        elif shape_type == 1: img[:, :, 12:20] += 2.5; tgt = [23, 24, 25]
        elif shape_type == 2: img[:, 10:22, 10:22] += 3.0; tgt = [26, 27, 28]
        else: img[:, :10, :10] += 3.5; tgt = [29, 30, 31]
        t2_inps.append(img)
        t2_tgts.append(tgt)
    tasks_raw["Vision -> Text (Captioning)"] = (
        torch.stack(t2_inps),
        torch.tensor(t2_tgts, dtype=torch.long),
        "vision", "ar_text"
    )

    # Task 3: Video -> Text (Spatio-Temporal Action)
    t3_inps, t3_tgts = [], []
    for i in range(600):
        video = torch.randn(3, 4, 32, 32) * 0.1
        action_type = i % 3
        if action_type == 0:
            for f in range(4): video[:, f, :, f*7 : f*7 + 8] += 2.5
            tgt = [35, 36, 37]
        elif action_type == 1:
            for f in range(4): video[:, f, f*7 : f*7 + 8, :] += 2.5
            tgt = [38, 39, 40]
        else:
            for f in range(4):
                s = (f + 1) * 3
                video[:, f, 16-s : 16+s, 16-s : 16+s] += 2.0
            tgt = [41, 42, 43]
        t3_inps.append(video)
        t3_tgts.append(tgt)
    tasks_raw["Video -> Text (Action)"] = (
        torch.stack(t3_inps),
        torch.tensor(t3_tgts, dtype=torch.long),
        "video", "ar_text"
    )

    # Task 4: Audio -> Text (Speech Transcription)
    t4_inps, t4_tgts = [], []
    for i in range(800):
        spec = torch.randn(16, 80) * 0.1
        w_id = i % 5
        spec[:, w_id*15 + 10 : w_id*15 + 18] += 3.5
        t4_inps.append(spec)
        t4_tgts.append([100 + w_id, 105 + w_id, 110 + w_id])
    tasks_raw["Audio -> Text (Transcription)"] = (
        torch.stack(t4_inps),
        torch.tensor(t4_tgts, dtype=torch.long),
        "audio", "ar_text"
    )

    # Task 5: Audio -> Audio (Continuous Restoration with SwiGLU)
    t5_inps, t5_tgts = [], []
    for i in range(800):
        clean = torch.randn(16, 80) * 0.1
        band = (i % 4) * 18 + 5
        clean[:, band : band + 10] += 2.5
        noisy = clean + torch.randn_like(clean) * 0.4
        t5_inps.append(noisy)
        t5_tgts.append(clean)
    tasks_raw["Audio -> Audio (Restoration)"] = (
        torch.stack(t5_inps),
        torch.stack(t5_tgts),
        "audio", "audio_spec"
    )

    # Task 6: Text -> Diffusion (Continuous Denoising with SwiGLU)
    t6_inps, t6_tgts = [], []
    for i in range(600):
        p_id = i % 4
        cond = [200 + p_id, 205 + p_id, 210 + p_id]
        tgt_latent = torch.zeros(len(cond), 64)
        tgt_latent[:, p_id * 15 : p_id * 15 + 12] = 2.0
        t6_inps.append(cond)
        t6_tgts.append(tgt_latent)
    tasks_raw["Text -> Diffusion (Latent)"] = (
        torch.tensor(t6_inps, dtype=torch.long),
        torch.stack(t6_tgts),
        "text", "diffusion"
    )

    # Task 7: Tabular -> Decision (Multi-Class Classification)
    t7_inps, t7_tgts = [], []
    for i in range(600):
        feat = torch.randn(16)
        c_idx = i % 10
        feat[c_idx] += 3.5
        t7_inps.append(feat)
        t7_tgts.append(c_idx)
    tasks_raw["Tabular -> Decision (Class)"] = (
        torch.stack(t7_inps),
        torch.tensor(t7_tgts, dtype=torch.long),
        "tabular", "classification"
    )

    train_loaders = {}
    test_loaders = {}
    total_train = 0
    total_test = 0

    for name, (inps, tgts, mod_in, t_type) in tasks_raw.items():
        n = len(inps)
        split = int(n * 0.95)
        train_in, test_in = inps[:split], inps[split:]
        train_tg, test_tg = tgts[:split], tgts[split:]

        tr_ds = TaskBatchDataset(train_in, train_tg, mod_in, t_type, name)
        te_ds = TaskBatchDataset(test_in, test_tg, mod_in, t_type, name)

        train_loaders[name] = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loaders[name] = DataLoader(te_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

        total_train += len(train_in)
        total_test += len(test_in)
        print(f"  [+] {name:<35}: Total={n:5d} | Train (95%)={len(train_in):5d} | Test (5%)={len(test_in):4d}")

    print(f"\n[+] Total Unified Training Omni-Modal Samples (95%): {total_train:,}")
    print(f"[+] Total Isolated Test Omni-Modal Samples (5%):     {total_test:,}")

    return train_loaders, test_loaders


def train_omni_batch_step(
    model: PhenotypeNeuralNetwork,
    inps: torch.Tensor,
    tgts: torch.Tensor,
    modality_in: str,
    task_type: str,
    device: torch.device,
) -> Tuple[torch.Tensor, float, bool]:
    inps, tgts = inps.to(device), tgts.to(device)

    if task_type == "ar_text":
        if modality_in == "text":
            h, aux_loss, _, _ = model(inps, modality="text", is_causal=True)
            logits = model.ar_head(h)  # (B, S, vocab)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgts.view(-1)) + 0.01 * aux_loss
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == tgts).float().mean().item() * 100.0
            return loss, acc, True
        else:
            # Cross-modal (Vision / Video / Audio -> Text Tokens)
            h, aux_loss, _, _ = model(inps, modality=modality_in)
            logits = model.ar_head(h)  # (B, S_in, vocab)
            if logits.size(1) != tgts.size(1):
                logits = F.adaptive_avg_pool1d(logits.permute(0, 2, 1), tgts.size(1)).permute(0, 2, 1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgts.view(-1)) + 0.01 * aux_loss
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == tgts).float().mean().item() * 100.0
            return loss, acc, True

    elif task_type == "audio_spec":
        h, aux_loss, _, _ = model(inps, modality="audio")
        pred_spec = model.audio_head(h)  # (B, 16, 80)
        mse = F.mse_loss(pred_spec, tgts)
        p_f = pred_spec.view(pred_spec.size(0), -1)
        t_f = tgts.view(tgts.size(0), -1)
        cos_sim = F.cosine_similarity(p_f, t_f, dim=-1).mean()
        loss = mse + 0.5 * (1.0 - cos_sim) + 0.01 * aux_loss
        return loss, cos_sim.item() * 100.0, False

    elif task_type == "diffusion":
        h, aux_loss, _, _ = model(inps, modality="text")
        noise = torch.randn_like(tgts) * 0.5
        noisy_x = tgts + noise
        t = torch.randint(1, 20, (inps.size(0),), device=device)
        pred_noise = model.diff_head(noisy_x, t, h)
        loss = F.mse_loss(pred_noise, noise) + 0.01 * aux_loss
        cos_sim = F.cosine_similarity(pred_noise.view(pred_noise.size(0), -1), noise.view(noise.size(0), -1), dim=-1).mean()
        return loss, cos_sim.item() * 100.0, False

    elif task_type == "classification":
        h, aux_loss, _, _ = model(inps, modality="tabular")
        logits = model.cls_head(h)  # (B, num_classes)
        loss = F.cross_entropy(logits, tgts) + 0.01 * aux_loss
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == tgts).float().mean().item() * 100.0
        return loss, acc, True

    return torch.tensor(0.0, device=device), 0.0, True


def evaluate_omni_task_loader(model: PhenotypeNeuralNetwork, loader: DataLoader, device: torch.device) -> Tuple[float, float, str]:
    model.eval()
    total_loss = 0.0
    total_score = 0.0
    batches = 0
    is_acc = True

    with torch.no_grad():
        for inps, tgts in loader:
            loss, score, is_acc_flag = train_omni_batch_step(
                model, inps, tgts, loader.dataset.modality_in, loader.dataset.task_type, device
            )
            total_loss += loss.item()
            total_score += score
            is_acc = is_acc_flag
            batches += 1

    avg_loss = total_loss / max(1, batches)
    avg_score = total_score / max(1, batches)
    unit = "Acc %" if is_acc else "Cos %"
    return avg_loss, avg_score, unit


def run_omni_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print(" STRICT 95% TRAIN / 5% TEST OMNI-MODAL ALL-INPUTS ALL-OUTPUTS BENCHMARK (WITH SWIGLU)")
    print(f" Execution Device: {device} | CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 110, flush=True)

    train_loaders, test_loaders = build_omni_task_loaders(batch_size=64)

    # Setup Omni Architecture Template with SwiGLU
    genotype_template = Genotype.create_default(genotype_id="omni_modal_swiglu_root")
    genotype_template.dna_architecture.d_model = 128
    genotype_template.dna_architecture.num_layers = 4
    genotype_template.dna_architecture.num_heads = 4
    genotype_template.dna_architecture.num_experts = 4
    genotype_template.dna_architecture.d_expert_hidden = 256
    genotype_template.dna_architecture.lora_rank = 8
    genotype_template.dna_architecture.vocab_size = 512
    genotype_template.dna_architecture.num_classes = 10

    # MODEL 1: Standard Baseline Omni-Phenotype with SwiGLU
    growth_engine = GrowthEngine(device=device)
    model_std = growth_engine.grow_phenotype_model(genotype_template).to(device)

    # MODEL 2: Omni AI-DNA Evolution with SwiGLU
    model_lora_evolved = growth_engine.grow_phenotype_model(genotype_template).to(device)

    optimizer_std = torch.optim.AdamW(model_std.parameters(), lr=1e-3, weight_decay=1e-4)

    # =========================================================================
    # >>> 3. TRAIN MODEL 1: Standard Baseline Omni Model (Full Weights)
    # =========================================================================
    print("\n" + "=" * 110)
    print(" >>> TRAINING MODEL 1: Standard Baseline Model (Full Weights with SwiGLU on All Modalities)")
    print("=" * 110, flush=True)

    time_std_start = time.time()
    for epoch in range(1, 5):
        model_std.train()
        total_loss, total_score, batches = 0.0, 0.0, 0
        t0 = time.time()

        for name, loader in train_loaders.items():
            for inps, tgts in loader:
                optimizer_std.zero_grad()
                loss, score, _ = train_omni_batch_step(
                    model_std, inps, tgts, loader.dataset.modality_in, loader.dataset.task_type, device
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_std.parameters(), 1.0)
                optimizer_std.step()

                total_loss += loss.item()
                total_score += score
                batches += 1

        ep_loss = total_loss / max(1, batches)
        ep_score = total_score / max(1, batches)
        print(f"  Standard Omni Model | Epoch {epoch}/4 | Train Loss: {ep_loss:.4f} | Omni Score: {ep_score:.1f}% | Epoch Time: {time.time()-t0:.2f}s", flush=True)

    time_std_train = time.time() - time_std_start

    # =========================================================================
    # >>> 4. TRAIN MODEL 2: LoRA + CPPN Omni AI-DNA Evolution (5 Generations)
    # =========================================================================
    print("\n" + "=" * 110)
    print(" >>> TRAINING MODEL 2: LoRA + CPPN AI-DNA Evolution (5 Generations with SwiGLU)")
    print("=" * 110, flush=True)

    slow_clock = SlowClockEncoder(device=device, encoder_steps=120)

    # Gen 0: Base Omni Foundation Initiation
    print("\n  [Generation 0: Initiation] Training base omni foundation with SwiGLU unfrozen...")
    opt_init = torch.optim.AdamW(model_lora_evolved.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    model_lora_evolved.train()
    total_loss, total_score, batches = 0.0, 0.0, 0

    for name, loader in train_loaders.items():
        for inps, tgts in loader:
            opt_init.zero_grad()
            loss, score, _ = train_omni_batch_step(
                model_lora_evolved, inps, tgts, loader.dataset.modality_in, loader.dataset.task_type, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_lora_evolved.parameters(), 1.0)
            opt_init.step()

            total_loss += loss.item()
            total_score += score
            batches += 1

    gen0_loss = total_loss / max(1, batches)
    gen0_score = total_score / max(1, batches)
    print(f"  [Gen 0 Initiation] Loss: {gen0_loss:.4f} | Omni Score: {gen0_score:.1f}% | Time: {time.time()-t0:.2f}s", flush=True)

    # Encode Base Foundation into Genotype D_0
    print("  [Gen 0 Initiation] Encoding foundation weights into Base Genotype D_0 via Slow Clock...", flush=True)
    genotype_d0, _ = slow_clock.step(
        genotype_template,
        model_lora_evolved.state_dict(),
        phenotype_model=model_lora_evolved,
        growth_engine=growth_engine,
        protect_ancestral=False,
    )
    current_genotype_lora = genotype_d0.clone("omni_dna_swiglu_gen0")

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

        # Unfreeze sensory intake encoders & output heads for omni adaptation
        for enc in [model_gen.text_encoder, model_gen.vision_encoder, model_gen.video_encoder, model_gen.audio_encoder, model_gen.tabular_proj]:
            for p in enc.parameters():
                p.requires_grad = True
        for head in [model_gen.ar_head, model_gen.diff_head, model_gen.cls_head, model_gen.audio_head]:
            for p in head.parameters():
                p.requires_grad = True

        optimizer_gen = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_gen.parameters()),
            lr=2e-3,
            weight_decay=1e-4,
        )

        t_fast_start = time.time()
        total_loss, total_score, batches = 0.0, 0.0, 0

        for name, loader in train_loaders.items():
            for inps, tgts in loader:
                optimizer_gen.zero_grad()
                loss, score, _ = train_omni_batch_step(
                    model_gen, inps, tgts, loader.dataset.modality_in, loader.dataset.task_type, device
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_gen.parameters(), 1.0)
                optimizer_gen.step()

                total_loss += loss.item()
                total_score += score
                batches += 1

        gen_loss = total_loss / max(1, batches)
        gen_score = total_score / max(1, batches)
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
            "train_score": gen_score,
            "fast_time_sec": fast_time,
            "slow_clock_time_sec": slow_clock_time,
            "dna_params": dna_param_count,
            "recon_loss": recon_loss,
        })

        print(f"  Generation {gen} Complete | Train Loss: {gen_loss:.4f} | Omni Score: {gen_score:.1f}% | Fast Time: {fast_time:.1f}s | Slow Time: {slow_clock_time:.1f}s | DNA Size: {dna_param_count} params", flush=True)

    # 4. Regrow Final Phenotype W_5 from Genotype D_5
    print("\n[+] Regrowing Final 5th Generation Omni Phenotype W_5 with SwiGLU from Genotype D_5...")
    model_lora_evolved = growth_engine.grow_phenotype_model(current_genotype_lora).to(device)

    # =========================================================================
    # >>> 5. HELD-OUT 5% TEST EVALUATION PER MODALITY TASK
    # =========================================================================
    print("\n" + "=" * 110)
    print(" 5% HELD-OUT TEST EVALUATION PER OMNI TASK (Standard vs. LoRA+CPPN Gen 5 with SwiGLU)")
    print("=" * 110, flush=True)

    results_table = []
    summary_data = {
        "total_train_samples": sum(len(l.dataset) for l in train_loaders.values()),
        "total_test_samples": sum(len(l.dataset) for l in test_loaders.values()),
        "standard_model": {"tasks": {}, "train_time_sec": time_std_train},
        "lora_cppn_model": {
            "tasks": {},
            "train_time_sec": total_lora_train_time,
            "slow_clock_time_sec": total_slow_clock_time,
            "num_generations": num_generations,
            "generational_history": lora_history,
        },
    }

    for name, tloader in test_loaders.items():
        loss_std, score_std, unit = evaluate_omni_task_loader(model_std, tloader, device)
        loss_lora, score_lora, _ = evaluate_omni_task_loader(model_lora_evolved, tloader, device)

        summary_data["standard_model"]["tasks"][name] = {"loss": loss_std, "score": score_std, "unit": unit}
        summary_data["lora_cppn_model"]["tasks"][name] = {"loss": loss_lora, "score": score_lora, "unit": unit}

        results_table.append({
            "task": name,
            "std_loss": loss_std,
            "std_score": score_std,
            "lora_loss": loss_lora,
            "lora_score": score_lora,
            "unit": unit,
            "test_samples": len(tloader.dataset),
        })

    print(f"\n{'Omni Task':<35} | {'Test Size':<10} | {'Std Loss':<11} | {'Std Score':<12} | {'DNA Loss':<11} | {'DNA Score':<12}")
    print("-" * 102)
    for r in results_table:
        print(f"{r['task']:<35} | {r['test_samples']:<10d} | {r['std_loss']:<11.4f} | {r['std_score']:<5.1f} {r['unit']:<5} | {r['lora_loss']:<11.4f} | {r['lora_score']:<5.1f} {r['unit']:<5}")

    # 6. Side-by-Side Inference Demonstrations Across Modalities
    print("\n" + "=" * 110)
    print(" SIDE-BY-SIDE INFERENCE DEMONSTRATION ACROSS ALL 7 OMNI MODALITIES (SWIGLU)")
    print("=" * 110, flush=True)

    model_std.eval()
    model_lora_evolved.eval()

    sample_num = 0
    with torch.no_grad():
        for name, tloader in test_loaders.items():
            sample_num += 1
            for inps, tgts in tloader:
                inps_sub = inps[:1]
                tgts_sub = tgts[:1]
                loss_s, score_s, _ = train_omni_batch_step(
                    model_std, inps_sub, tgts_sub, tloader.dataset.modality_in, tloader.dataset.task_type, device
                )
                loss_l, score_l, _ = train_omni_batch_step(
                    model_lora_evolved, inps_sub, tgts_sub, tloader.dataset.modality_in, tloader.dataset.task_type, device
                )

                print(f"\n[Omni Modality #{sample_num}: {name}]")
                print(f"  Input Modality: {tloader.dataset.modality_in.upper():<8} -> Output Head: {tloader.dataset.task_type.upper()}")
                print(f"  Standard Model Quality (SwiGLU): Loss={loss_s.item():.4f} | Score={score_s:.1f}%")
                print(f"  AI-DNA (W5) Quality (SwiGLU)   : Loss={loss_l.item():.4f} | Score={score_l:.1f}%")
                break

    # 7. Parameter & Storage Compression
    std_params = sum(p.numel() for p in model_std.parameters())
    dna_params = current_genotype_lora.total_parameters()
    c_r = std_params / max(1, dna_params)

    summary_data["compression_ratio"] = c_r
    summary_data["std_params"] = std_params
    summary_data["dna_params"] = dna_params

    print("\n" + "=" * 110)
    print(" OMNI-MODAL STORAGE & PARAMETER COMPRESSION SUMMARY (WITH SWIGLU)")
    print("=" * 110)
    print(f"  Standard Phenotype Parameters: {std_params:,} parameters ({std_params*2/1024/1024:.2f} MB in FP16)")
    print(f"  Genotype AI-DNA Parameters:    {dna_params:,} parameters ({dna_params*4/1024:.2f} KB)")
    print(f"  True Compression Ratio (C_R):  {c_r:.2f}x compression")
    print(f"  Training Time: Standard={time_std_train:.1f}s | LoRA (5 Gens)={total_lora_train_time:.1f}s | SlowClock={total_slow_clock_time:.1f}s")
    print("=" * 110)

    # Save Checkpoints & Results
    torch.save(model_std.state_dict(), "checkpoint_omni_standard.pt")
    torch.save(current_genotype_lora, "checkpoint_omni_aidna.pt")
    with open("results_omni_modal_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n[+] Complete Omni-Modal SwiGLU benchmark results saved to results_omni_modal_benchmark.json", flush=True)


if __name__ == "__main__":
    run_omni_benchmark()
