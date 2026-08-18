"""
Omni-Modal AI DNA: Unified Multi-Modal Training & Evolution Pipeline.
Consolidates all training workflows into a single master training pipeline with:
  1. Parallel Multi-Modality Specialist Training across CPU/GPU threads
  2. Multi-Parent DNA Fusion (D_omni = F(D_text, D_vision, D_audio, D_video, D_code, D_bio, D_tabular))
  3. Live Training Progress Bars with Loss, Throughput, and live GPU VRAM tracking
  4. Memory Optimization (automatic GPU cache clearing, DMA page-locked transfers, garbage collection)
  5. Whole Dataset Support (--all-data / max_samples=None)
"""

import os
import sys
import gc
import time
import math
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_dna import (
    Genotype,
    GrowthEngine,
    InferencePipeline,
    PhenotypeNeuralNetwork,
    FastClockTrainer,
    SlowClockEncoder,
    MultiParentFusion,
    CompatibilityChecker,
    EvaluationMetrics,
    TextTokenizer,
    save_genotype,
    load_genotype,
)

from data import (
    DataType,
    get_dataset,
    get_dataloader,
    get_text_dataset,
    get_vision_dataset,
    get_audio_dataset,
    get_video_dataset,
    get_code_dataset,
    get_bio_dataset,
    get_tabular_dataset,
    get_multimodal_dataset,
    CustomTextTokenizer,
    tqdm,
)


# =====================================================================
# 1. Helper Utilities & Device Memory Management
# =====================================================================

def setup_device(device_str: str = "auto") -> torch.device:
    """Selects target computing device and enables GPU performance optimizations."""
    if device_str in ["auto", "cuda"] and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU Setup] CUDA GPU Acceleration Active: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        return device
    elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[GPU Setup] Apple MPS Acceleration Active")
        return torch.device("mps")
    else:
        if device_str == "cuda":
            print("[GPU Setup] Notice: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")


def clear_memory(device: torch.device):
    """Frees cached GPU memory and forces Python garbage collection."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def get_vram_info(device: torch.device) -> Dict[str, str]:
    """Returns formatted VRAM allocation if CUDA is active."""
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        return {"vram": f"{alloc:.1f}MB", "res": f"{reserved:.1f}MB"}
    return {}


def ensure_dir(path: str):
    """Ensures directory exists."""
    os.makedirs(path, exist_ok=True)


def print_banner(title: str, subtitle: Optional[str] = None):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70)


def create_base_genotype(
    genotype_id: str = "Genesis_DNA",
    d_model: int = 64,
    num_layers: int = 2,
    num_experts: int = 4,
    vocab_size: int = 256,
) -> Genotype:
    """Creates a configured baseline genotype."""
    genotype = Genotype.create_default(genotype_id=genotype_id)
    genotype.dna_architecture.vocab_size = vocab_size
    genotype.dna_architecture.d_model = d_model
    genotype.dna_architecture.num_layers = num_layers
    genotype.dna_architecture.num_experts = num_experts
    genotype.dna_architecture.d_expert_hidden = d_model * 2
    genotype.dna_memory.chunk_size = 16
    genotype.dna_instinct.cppn_hidden_dim = 32
    genotype.dna_instinct.cppn_layers = 2
    return genotype


# =====================================================================
# 2. Pipeline 1: Single Modality Specialist Training
# =====================================================================

def run_single_modality_training(
    modality: str = "text",
    dataset_name: Optional[str] = None,
    subset: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    num_samples: Optional[int] = 500,
    streaming: bool = False,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "./checkpoints/single_modality",
) -> Tuple[Genotype, nn.Module]:
    """
    Trains specialist on a specific modality (text, vision, audio, code, bio, etc.)
    with live progress bar, memory optimization, and Slow Clock SVD distillation.
    """
    print_banner(
        f"Single Modality Specialist Pipeline: [{modality.upper()}]",
        f"Dataset: {dataset_name or 'Default'} | Epochs: {epochs} | Batch Size: {batch_size}",
    )
    ensure_dir(save_dir)

    # Initialize dynamic BPE tokenizer co-evolving with DNA
    from data import CustomTextTokenizer
    vocab_size = 256
    tokenizer_path = os.path.join(save_dir, f"tokenizer_{modality}.json")
    if os.path.exists(tokenizer_path):
        print(f"[*] Loading tokenizer to evolve from: {tokenizer_path}")
        my_tokenizer = CustomTextTokenizer(vocab_size=vocab_size, checkpoint_dir=save_dir)
        my_tokenizer.vocab_size += 32  # Evolve by 32 new token merges each generation
    else:
        print("[*] Initializing new dynamic tokenizer (Gen 0)...")
        my_tokenizer = CustomTextTokenizer(vocab_size=vocab_size)

    # 1. Load Hugging Face Data via data.py & Train BPE Tokenizer
    print(f"\n[1/4] Loading Hugging Face Dataset & Training Tokenizer for modality [{modality}]...")
    loader = get_dataloader(
        data_type=modality,
        dataset_name=dataset_name,
        subset=subset,
        split="train",
        batch_size=batch_size,
        max_samples=num_samples,
        streaming=streaming,
        tokenizer=my_tokenizer,
    )
    total_samples = len(loader.dataset)
    print(f"      Total Training Samples: {total_samples:,}")
    print(f"      Final Evolved Vocab Size: {my_tokenizer.vocab_size}")

    # 2. Create Specialist Genotype & Grow Phenotype
    print("\n[2/4] Creating Genotype and Growing Neural Phenotype...")
    genotype = create_base_genotype(
        genotype_id=f"D_{modality}_specialist",
        vocab_size=my_tokenizer.vocab_size,
    )
    growth_engine = GrowthEngine(device=device)
    model = PhenotypeNeuralNetwork(genotype).to(device)

    # Apply developmental weights
    grown_weights = growth_engine.grow_phenotype_weights(genotype)
    p_state = model.state_dict()
    for k, v in grown_weights.items():
        if k in p_state and p_state[k].shape == v.shape:
            p_state[k] = v
    model.load_state_dict(p_state)

    total_params = sum(p.numel() for p in model.parameters())
    dna_params = genotype.total_parameters()
    comp_ratio = EvaluationMetrics.true_compression_ratio(total_params, dna_params)
    print(f"      Phenotype Parameters: {total_params:,}")
    print(f"      DNA Parameters:       {dna_params:,}")
    print(f"      Compression Ratio:    {comp_ratio:.1f}x")

    # 3. Fast Clock Training Loop with Live Progress Bar
    print(f"\n[3/4] Starting Fast Clock Parametric Optimization ({epochs} Epochs)...")
    trainer = FastClockTrainer(model, learning_rate=learning_rate, device=device)
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(loader, desc=f"  Epoch {epoch:2d}/{epochs:2d}", unit="batch", leave=False) as pbar:
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    bx, by = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                else:
                    bx = (batch["vision"] if modality == "vision" else batch["text"]).to(device, non_blocking=True)
                    by = batch["label"].to(device, non_blocking=True)

                if modality in ["text", "code", "bio"] and by.dim() == bx.dim():
                    loss, _ = trainer.train_step_autoregressive(bx, by)
                else:
                    loss, _ = trainer.train_step_classification(bx, by, modality=modality)

                epoch_loss += loss
                num_batches += 1

                postfix = {"loss": f"{loss:.4f}"}
                postfix.update(get_vram_info(device))
                pbar.set_postfix(**postfix)

        avg_loss = epoch_loss / max(1, num_batches)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            vram_str = f" | VRAM: {get_vram_info(device).get('vram', 'N/A')}" if device.type == "cuda" else ""
            print(f"      Epoch {epoch:2d}/{epochs:2d} | Avg Loss: {avg_loss:.4f}{vram_str}")

        clear_memory(device)

    train_time = (time.perf_counter() - t_start) * 1000.0
    print(f"      Fast Clock Completed in {train_time:.1f} ms | Final Loss: {avg_loss:.4f}")

    # 4. Slow Clock Distillation: W* -> D_new
    print(f"\n[4/4] Slow Clock Distillation: Encoding Learned Phenotype into Genotype...")
    encoder = SlowClockEncoder(rank_ratio=0.25, encoder_steps=50, device=device)
    evolved_genotype, dist_metrics = encoder.step(genotype, model.state_dict())
    print(f"      Distillation Retained Energy: {dist_metrics.get('mean_retained_energy', 1.0):.2%}")

    # Save outputs
    dna_path = os.path.join(save_dir, f"genotype_{modality}.json")
    weights_path = os.path.join(save_dir, f"phenotype_{modality}.pt")
    save_genotype(evolved_genotype, dna_path)
    torch.save(model.state_dict(), weights_path)
    print(f"      [+] Saved Genotype DNA to:   {dna_path}")
    print(f"      [+] Saved Phenotype Model to: {weights_path}")

    # Save co-evolved BPE tokenizer state
    my_tokenizer.bpe.save(tokenizer_path)
    print(f"      [+] Saved Evolving Tokenizer state to: {tokenizer_path}")

    clear_memory(device)
    return evolved_genotype, model


# =====================================================================
# 3. Pipeline 2: Multi-Parent DNA Fusion
# =====================================================================

def run_multimodal_fusion_pipeline(
    epochs_specialist: int = 4,
    epochs_joint: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    num_samples: Optional[int] = 400,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "./checkpoints/multimodal_fusion",
) -> Genotype:
    """Executes 3-Parent DNA Fusion (Text + Vision + Audio)."""
    return run_parallel_all_modalities_fusion_pipeline(
        modalities=["text", "vision", "audio"],
        epochs_specialist=epochs_specialist,
        epochs_joint=epochs_joint,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_samples=num_samples,
        max_workers=3,
        device=device,
        save_dir=save_dir,
    )


def run_parallel_all_modalities_fusion_pipeline(
    modalities: Optional[List[str]] = None,
    epochs_specialist: int = 4,
    epochs_joint: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    num_samples: Optional[int] = 400,
    max_workers: int = 4,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "./checkpoints/omni_parallel_fusion",
) -> Genotype:
    """
    1. Trains ALL modality specialist parents in parallel using multi-worker execution.
    2. Performs multi-parent DNA fusion across all specialist parents:
       D_omni = F(D_text, D_vision, D_audio, D_video, D_code, D_bio, D_tabular)
    3. Grows the unified Omni-Modal Phenotype with Sparse Mixture of Experts routing.
    4. Jointly fine-tunes the omni-modal child on multi-modal paired data on GPU.
    """
    if modalities is None:
        modalities = ["text", "vision", "audio", "video", "code", "bio", "tabular"]

    print_banner(
        f"Parallel Omni-Modal Training & DNA Fusion ({len(modalities)} Modalities)",
        f"Parallel Workers: {max_workers} | Specialist Epochs: {epochs_specialist} | Joint Epochs: {epochs_joint}",
    )
    ensure_dir(save_dir)

    print(f"\n--- STEP 1: Concurrently Training {len(modalities)} Specialist Parents in Parallel ---")
    print(f"    Target Modalities: {modalities}")

    def _train_single_parent(mod: str) -> Tuple[str, Genotype, nn.Module]:
        dna, model = run_single_modality_training(
            modality=mod,
            dataset_name=None,
            epochs=epochs_specialist,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_samples=num_samples,
            device=device,
            save_dir=os.path.join(save_dir, "parents", mod),
        )
        return mod, dna, model

    parent_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_train_single_parent, mod) for mod in modalities]
        for f in futures:
            mod, dna, model = f.result()
            parent_results[mod] = (dna, model)

    parents = [parent_results[m][0] for m in modalities]
    weights = [1.0 / len(parents)] * len(parents)

    # 2. Check Compatibility & Multi-Parent DNA Fusion
    print(f"\n--- STEP 2: Multi-Parent DNA Fusion ({len(parents)} Parents) ---")
    for i in range(len(parents) - 1):
        comp = CompatibilityChecker.evaluate(parents[i], parents[i+1], min_score=0.5)
        print(f"      Compatibility ({parents[i].genotype_id} <-> {parents[i+1].genotype_id}): {comp.overall_score:.2%} ({'PASS' if comp.is_compatible else 'FAIL'})")

    fusion = MultiParentFusion(min_compatibility=0.5)
    omni_genotype = fusion.fuse(
        parents=parents,
        weights=weights,
        child_id="D_Omni_MultiParent_Fused",
    )
    print(f"\n      [+] Created Fused Omni-Modal Child DNA: {omni_genotype.genotype_id}")
    print(f"      Total Merged Innovation Nodes: {len(omni_genotype.node_innovation_map)}")

    # 3. Grow Omni-Modal Phenotype from Fused DNA
    print("\n--- STEP 3: Growing Omni-Modal Phenotype from Fused DNA ---")
    growth_engine = GrowthEngine(device=device)
    omni_model = PhenotypeNeuralNetwork(omni_genotype).to(device)
    grown_weights = growth_engine.grow_phenotype_weights(omni_genotype)
    p_state = omni_model.state_dict()
    for k, v in grown_weights.items():
        if k in p_state and p_state[k].shape == v.shape:
            p_state[k] = v
    omni_model.load_state_dict(p_state)

    total_params = sum(p.numel() for p in omni_model.parameters())
    dna_params = omni_genotype.total_parameters()
    comp_ratio = EvaluationMetrics.true_compression_ratio(total_params, dna_params)
    print(f"      Fused Phenotype Parameters: {total_params:,}")
    print(f"      Fused DNA Parameters:       {dna_params:,}")
    print(f"      Compression Ratio:          {comp_ratio:.1f}x")

    # 4. Joint Multi-Modal Fine-Tuning
    print(f"\n--- STEP 4: Joint Omni-Modal Fine-Tuning ({epochs_joint} Epochs) ---")
    mm_loader = get_dataloader(
        data_type="multimodal",
        batch_size=batch_size,
        max_samples=num_samples,
    )
    omni_trainer = FastClockTrainer(omni_model, learning_rate=learning_rate * 0.5, device=device)

    for epoch in range(1, epochs_joint + 1):
        epoch_loss = 0.0
        batches = 0
        with tqdm(mm_loader, desc=f"  Joint Epoch {epoch:2d}/{epochs_joint:2d}", unit="batch", leave=False) as pbar:
            for batch in pbar:
                txt, vis, aud, lbl = (
                    batch["text"].to(device, non_blocking=True),
                    batch["vision"].to(device, non_blocking=True),
                    batch["audio"].to(device, non_blocking=True),
                    batch["label"].to(device, non_blocking=True),
                )
                l_t, _ = omni_trainer.train_step_classification(txt, lbl, modality="text")
                l_v, _ = omni_trainer.train_step_classification(vis, lbl, modality="vision")
                l_a, _ = omni_trainer.train_step_classification(aud, lbl, modality="audio")
                step_loss = (l_t + l_v + l_a) / 3.0
                epoch_loss += step_loss
                batches += 1
                pbar.set_postfix(loss=f"{step_loss:.4f}", **get_vram_info(device))

        avg_loss = epoch_loss / max(1, batches)
        print(f"      Joint Epoch {epoch:2d}/{epochs_joint:2d} | Tri-Modal Loss: {avg_loss:.4f}")
        clear_memory(device)

    # 5. Save Fused DNA and Joint Model
    fused_dna_path = os.path.join(save_dir, "genotype_omni_fused.json")
    fused_model_path = os.path.join(save_dir, "phenotype_omni_fused.pt")
    save_genotype(omni_genotype, fused_dna_path)
    torch.save(omni_model.state_dict(), fused_model_path)
    print(f"\n[+] Successfully saved Fused Omni-Modal DNA to: {fused_dna_path}")
    print(f"[+] Successfully saved Fused Phenotype to:       {fused_model_path}")

    clear_memory(device)
    return omni_genotype


# =====================================================================
# 4. Pipeline 3: Multi-Generational Evolutionary Lifecycle
# =====================================================================

def run_evolutionary_lifecycle(
    num_generations: int = 4,
    epochs_per_gen: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    num_samples: Optional[int] = 500,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "./checkpoints/evolutionary_lifecycle",
) -> Genotype:
    """
    Runs multi-generational developmental cycle:
    D_0 -> W_0 -> W_0* -> D_1 -> W_1 -> ... -> D_k
    """
    print_banner(
        "Multi-Generational Evolutionary Lifecycle",
        f"Generations: {num_generations} | Epochs per Gen: {epochs_per_gen}",
    )
    ensure_dir(save_dir)

    current_genotype = create_base_genotype(genotype_id="D_Gen0_Genesis")
    growth_engine = GrowthEngine(device=device)

    loader = get_dataloader(
        data_type="text",
        dataset_name="wikitext",
        batch_size=batch_size,
        max_samples=num_samples,
    )

    for gen in range(num_generations):
        print(f"\n{'='*30} GENERATION {gen} {'='*30}")
        print(f" Genotype ID: {current_genotype.genotype_id}")

        # 1. Grow Phenotype
        model = PhenotypeNeuralNetwork(current_genotype).to(device)
        grown_w = growth_engine.grow_phenotype_weights(current_genotype)
        p_state = model.state_dict()
        for k, v in grown_w.items():
            if k in p_state and p_state[k].shape == v.shape:
                p_state[k] = v
        model.load_state_dict(p_state)

        # 2. Fast Clock Learning with Progress Bar
        trainer = FastClockTrainer(model, learning_rate=learning_rate, device=device)
        for epoch in range(1, epochs_per_gen + 1):
            epoch_loss = 0.0
            batches = 0
            with tqdm(loader, desc=f"  Gen {gen} Ep {epoch:2d}/{epochs_per_gen:2d}", unit="batch", leave=False) as pbar:
                for bx, by in pbar:
                    bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                    loss, _ = trainer.train_step_autoregressive(bx, by)
                    epoch_loss += loss
                    batches += 1
                    pbar.set_postfix(loss=f"{loss:.4f}", **get_vram_info(device))
            clear_memory(device)

        print(f"      Gen {gen} Final Fast Clock Loss: {epoch_loss / max(1, batches):.4f}")

        # 3. Slow Clock Distillation
        encoder = SlowClockEncoder(rank_ratio=0.25, encoder_steps=50, device=device)
        evolved_dna, metrics = encoder.step(current_genotype, model.state_dict())
        evolved_dna.genotype_id = f"D_Gen{gen+1}_Evolved"
        evolved_dna.generation = gen + 1

        # Save generation checkpoint
        gen_dna_path = os.path.join(save_dir, f"genotype_gen{gen+1}.json")
        save_genotype(evolved_dna, gen_dna_path)
        print(f"      [+] Distilled Gen {gen+1} DNA saved to: {gen_dna_path}")
        current_genotype = evolved_dna
        clear_memory(device)

    print("\n[+] Evolutionary Lifecycle Complete!")
    return current_genotype


# =====================================================================
# 5. Pipeline 4: Fluent Causal Language Modeling
# =====================================================================

def run_fluent_language_training(
    dataset_name: str = "wikitext",
    subset: Optional[str] = "wikitext-2-raw-v1",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    num_samples: Optional[int] = 500,
    streaming: bool = False,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "./checkpoints/fluent_language",
) -> Genotype:
    """
    Trains AI DNA on natural language sequences with Perplexity and VRAM tracking.
    """
    print_banner(
        "Fluent Natural Language Training Pipeline",
        f"Dataset: {dataset_name} | Epochs: {epochs} | Batch Size: {batch_size}",
    )
    ensure_dir(save_dir)

    tokenizer = CustomTextTokenizer(vocab_size=256, mode="word")
    genotype = create_base_genotype(genotype_id="D_Fluent_Genesis", vocab_size=256)
    growth_engine = GrowthEngine(device=device)

    pipeline = InferencePipeline(genotype=genotype, growth_engine=growth_engine, device=device)
    model = pipeline.phenotype

    loader = get_dataloader(
        data_type="text",
        dataset_name=dataset_name,
        subset=subset,
        batch_size=batch_size,
        max_samples=num_samples,
        streaming=streaming,
    )

    trainer = FastClockTrainer(model, learning_rate=learning_rate, device=device)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        with tqdm(loader, desc=f"  Epoch {epoch:2d}/{epochs:2d}", unit="batch", leave=False) as pbar:
            for bx, by in pbar:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                loss, _ = trainer.train_step_autoregressive(bx, by)
                epoch_loss += loss
                batches += 1
                ppl = math.exp(min(loss, 20.0))
                pbar.set_postfix(loss=f"{loss:.4f}", ppl=f"{ppl:.1f}", **get_vram_info(device))

        avg_loss = epoch_loss / max(1, batches)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"      Epoch {epoch:2d}/{epochs:2d} | Perplexity: {math.exp(min(avg_loss, 20.0)):.2f} | Loss: {avg_loss:.4f}")
        clear_memory(device)

    # Slow Clock distillation
    encoder = SlowClockEncoder(rank_ratio=0.25, encoder_steps=50, device=device)
    fluent_dna, _ = encoder.step(genotype, model.state_dict())
    fluent_dna.genotype_id = "D_Fluent_Language_Evolved"

    dna_path = os.path.join(save_dir, "genotype_fluent.json")
    weights_path = os.path.join(save_dir, "phenotype_fluent.pt")
    save_genotype(fluent_dna, dna_path)
    torch.save(model.state_dict(), weights_path)

    print(f"\n[+] Fluent Language Training Complete!")
    print(f"    Saved Genotype:  {dna_path}")
    print(f"    Saved Phenotype: {weights_path}")

    clear_memory(device)
    return fluent_dna


# =====================================================================
# 6. Master CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Omni-Modal AI DNA Master Training Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "fusion", "parallel_fusion", "omni_fusion", "multimodal", "evolution", "language"],
        help="Training execution mode (e.g. single, fusion, parallel_fusion, evolution, language)",
    )
    parser.add_argument("--modality", type=str, default="text", choices=[d.value for d in DataType], help="Specialist modality")
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name")
    parser.add_argument("--subset", type=str, default=None, help="Hugging Face dataset config/subset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Fast Clock learning rate")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of dataset samples (-1 or 0 for whole dataset)")
    parser.add_argument("--all-data", action="store_true", help="Train on the entire dataset without truncation")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads for parallel specialist training")
    parser.add_argument("--generations", type=int, default=4, help="Number of evolutionary generations")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming for large datasets")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda)")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()
    device = setup_device(args.device)
    effective_samples = None if (args.all_data or args.num_samples <= 0) else args.num_samples

    print("\n" + "#" * 70)
    print("  OMNI-MODAL AI DNA: MASTER TRAINING & EVOLUTION ENGINE")
    print(f"  Execution Mode: {args.mode.upper()} | Device: {device}")
    print(f"  Dataset Scope:  {'WHOLE DATASET (No limit)' if effective_samples is None else f'{effective_samples:,} samples'}")
    print("#" * 70)

    if args.mode == "single":
        run_single_modality_training(
            modality=args.modality,
            dataset_name=args.dataset,
            subset=args.subset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_samples=effective_samples,
            streaming=args.streaming,
            device=device,
            save_dir=os.path.join(args.save_dir, f"single_{args.modality}"),
        )
    elif args.mode in ["fusion", "multimodal"]:
        run_multimodal_fusion_pipeline(
            epochs_specialist=max(4, args.epochs // 2),
            epochs_joint=max(4, args.epochs // 2),
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_samples=effective_samples,
            device=device,
            save_dir=os.path.join(args.save_dir, "fusion"),
        )
    elif args.mode in ["parallel_fusion", "omni_fusion"]:
        run_parallel_all_modalities_fusion_pipeline(
            modalities=["text", "vision", "audio", "video", "code", "bio", "tabular"],
            epochs_specialist=max(4, args.epochs // 2),
            epochs_joint=max(4, args.epochs // 2),
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_samples=effective_samples,
            max_workers=args.workers,
            device=device,
            save_dir=os.path.join(args.save_dir, "omni_parallel_fusion"),
        )
    elif args.mode == "evolution":
        run_evolutionary_lifecycle(
            num_generations=args.generations,
            epochs_per_gen=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_samples=effective_samples,
            device=device,
            save_dir=os.path.join(args.save_dir, "evolution"),
        )
    elif args.mode == "language":
        run_fluent_language_training(
            dataset_name=args.dataset or "wikitext",
            subset=args.subset or "wikitext-2-raw-v1",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_samples=effective_samples,
            streaming=args.streaming,
            device=device,
            save_dir=os.path.join(args.save_dir, "language"),
        )


if __name__ == "__main__":
    main()
