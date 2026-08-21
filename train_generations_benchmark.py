"""
AI-DNA Multi-Generational Evolutionary Training & Benchmark Suite Harness.
Executes generational evolution D_0 -> D_1 -> ... -> D_N with:
  1. Foundation Phenotype Pre-Learning (Wikipedia & Synthetic developmental reasoning)
  2. Zero-Shot / Developmental Instinct Benchmark Evaluation at each generation:
     - Public Evaluation Suite (GSM8K, MATH, MBPP, ARC-AGI, ProofNet, miniF2F)
     - Private Held-Out Evaluation Suite (GSM8K, MATH, MBPP, ARC-AGI, ProofNet, miniF2F)
     - Clean Evaluation Benchmark: HumanEval (Strictly isolated zero-shot test)
  3. Fast Clock Generational Adaptation on GSM8K, MATH, and MBPP
  4. Post-Adaptation Benchmark Evaluation
  5. Slow Clock: SVD Instinct Filter & CPPN Encoding into D_{t+1}
  6. Comprehensive Benchmark Metric Logging across all generations
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ai_dna.dna.structure import Genotype
from ai_dna.growth.engine import GrowthEngine
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.encoding.slow_clock import SlowClockEncoder
from ai_dna.evolution.mutation import GenotypeMutator
from data import AIDNABenchmarkDataset, CustomTextTokenizer


def train_epoch(
    model: PhenotypeNeuralNetwork,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Trains phenotype on a streaming DataLoader batch."""
    model.train()
    total_loss = 0.0
    count = 0
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        h, aux_loss, _, _ = model(inputs, modality="text", is_causal=True)
        logits = model.ar_head(h)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)) + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


def evaluate_dataset(
    model: PhenotypeNeuralNetwork,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 25,
) -> Tuple[float, float, float]:
    """
    Evaluates phenotype on a dataset.
    Returns: (loss, token_accuracy, sequence_exact_match_proxy)
    """
    model.eval()
    total_loss = 0.0
    total_token_acc = 0.0
    total_seq_match = 0.0
    count = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)

            h, aux_loss, _, _ = model(inputs, modality="text", is_causal=True)
            logits = model.ar_head(h)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)) + aux_loss

            preds = logits.argmax(dim=-1)
            token_correct = (preds == targets).float().mean().item()
            seq_correct = (preds == targets).all(dim=-1).float().mean().item()

            total_loss += loss.item()
            total_token_acc += token_correct
            total_seq_match += seq_correct
            count += 1

    if count == 0:
        return 0.0, 0.0, 0.0
    return total_loss / count, total_token_acc / count, total_seq_match / count


def run_multi_generational_training(
    num_generations: int = 4,
    data_dir: str = "./ai-dna-data",
    batch_size: int = 16,
    seq_len: int = 64,
    fast_clock_steps: int = 30,
    device_str: Optional[str] = None,
    output_report: str = "benchmark_results_generations.json",
) -> Dict[str, Any]:
    """
    Main driver executing the complete multi-generational lifecycle and benchmarking.
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    data_dir = os.path.abspath(data_dir)
    print("=" * 80)
    print(" 🧬 AI-DNA OMNI-MODAL MULTI-GENERATIONAL EVOLUTION & BENCHMARK HARNESS")
    print(f" Device:          {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Generations:     {num_generations} (D_0 -> D_{num_generations-1})")
    print(f" Data Directory:  {data_dir}")
    print(f" Batch Size:      {batch_size} | Seq Len: {seq_len}")
    print("=" * 80)

    tokenizer = CustomTextTokenizer(vocab_size=256, mode="word")
    growth_engine = GrowthEngine(device=device)
    slow_clock = SlowClockEncoder(rank_ratio=0.30, encoder_steps=60, device=device)
    mutator = GenotypeMutator()
    criterion = nn.CrossEntropyLoss()

    # Pre-build DataLoaders for all evaluation benchmarks
    eval_benchmark_tasks = ["gsm8k", "math", "mbpp", "arc", "proofnet", "minif2f"]
    public_eval_loaders = {}
    heldout_eval_loaders = {}

    print("\n[+] Initializing Benchmark Evaluation DataLoaders...")
    for t in eval_benchmark_tasks:
        try:
            ds_pub = AIDNABenchmarkDataset(task=t, split="public_eval", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer)
            if len(ds_pub) > 0:
                public_eval_loaders[t] = torch.utils.data.DataLoader(ds_pub, batch_size=batch_size, shuffle=False)
                print(f"  - Public Eval [{t.upper():8s}]: {len(ds_pub):5d} problems")
        except Exception as e:
            print(f"  [WARN] Failed to load public eval {t}: {e}")

        try:
            ds_held = AIDNABenchmarkDataset(task=t, split="private_heldout", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer)
            if len(ds_held) > 0:
                heldout_eval_loaders[t] = torch.utils.data.DataLoader(ds_held, batch_size=batch_size, shuffle=False)
                print(f"  - Private Held-Out [{t.upper():8s}]: {len(ds_held):5d} problems (Unexposed)")
        except Exception as e:
            print(f"  [WARN] Failed to load private heldout {t}: {e}")

    # HumanEval Clean Test (Strictly Isolated Benchmark)
    ds_he = AIDNABenchmarkDataset(task="humaneval", split="clean_eval", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer)
    humaneval_loader = torch.utils.data.DataLoader(ds_he, batch_size=batch_size, shuffle=False)
    print(f"  - Clean Benchmark [HUMANEVAL]: {len(ds_he):5d} problems (ZERO exposure in adaptation)\n")

    # Foundation Corpora
    ds_synth = AIDNABenchmarkDataset(task="synthetic", split="training", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer)
    ds_wiki = AIDNABenchmarkDataset(task="wikipedia", split="training", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer)
    foundation_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([ds_synth, ds_wiki]),
        batch_size=batch_size,
        shuffle=True,
    )

    # Adaptation Corpora
    adaptation_loaders = {}
    for adapt_t in ["gsm8k", "math", "mbpp"]:
        ds_a = AIDNABenchmarkDataset(task=adapt_t, split="adaptation", data_dir=data_dir, seq_len=seq_len, tokenizer=tokenizer)
        adaptation_loaders[adapt_t] = torch.utils.data.DataLoader(ds_a, batch_size=batch_size, shuffle=True)
        print(f"[+] Adaptation Corpus [{adapt_t.upper()}]: {len(ds_a):5d} training items")

    # 1. Initialize Root Genotype D_0
    current_genotype = Genotype.create_default(genotype_id="gen_0")
    current_genotype.dna_architecture.vocab_size = 256
    current_genotype.dna_architecture.d_model = 64
    current_genotype.dna_architecture.num_layers = 3
    current_genotype.dna_architecture.num_experts = 4
    current_genotype.dna_architecture.d_expert_hidden = 128

    master_results = {
        "metadata": {
            "num_generations": num_generations,
            "d_model": current_genotype.dna_architecture.d_model,
            "num_layers": current_genotype.dna_architecture.num_layers,
            "num_experts": current_genotype.dna_architecture.num_experts,
            "device": str(device),
        },
        "generations": [],
    }

    start_total_time = time.time()

    # Multi-Generational Evolution Loop D_0 -> D_1 -> ... -> D_N
    for g in range(num_generations):
        gen_name = f"D_{g}"
        gen_start_time = time.time()
        print("\n" + "#" * 80)
        print(f"  🌟 GENERATION {gen_name} (Genotype ID: {current_genotype.genotype_id})")
        print("#" * 80)

        # -----------------------------------------------------------------
        # Step A: Developmental Growth Engine W_g = G(D_g)
        # -----------------------------------------------------------------
        print(f"\n[Step 1] Growing Neural Phenotype W_{g} from Genotype {gen_name}...")
        phenotype = PhenotypeNeuralNetwork(current_genotype).to(device)
        grown_w = growth_engine.grow_phenotype_weights(current_genotype)
        p_state = phenotype.state_dict()
        for k, v in grown_w.items():
            if k in p_state and p_state[k].shape == v.shape:
                p_state[k] = v
        phenotype.load_state_dict(p_state)
        total_params = sum(p.numel() for p in phenotype.parameters())
        print(f"  ✓ Phenotype grown successfully: {total_params:,} parameters.")

        # -----------------------------------------------------------------
        # Step B: Foundation Pre-Learning (Public Wikipedia + Synthetic)
        # -----------------------------------------------------------------
        print(f"\n[Step 2] Phenotype Pre-Learning on Foundation Corpus...")
        opt = optim.AdamW(phenotype.parameters(), lr=8e-4, weight_decay=1e-4)
        loss_fnd = train_epoch(phenotype, foundation_loader, opt, criterion, device, max_batches=20)
        print(f"  ✓ Pre-Learning Foundation Loss: {loss_fnd:.4f}")

        # -----------------------------------------------------------------
        # Step C: Milestone Zero-Shot / Instinct Benchmark Evaluation
        # -----------------------------------------------------------------
        print(f"\n[Step 3] Evaluating Generation {gen_name} Instincts Across Benchmark Suite...")

        gen_metrics = {
            "generation": gen_name,
            "genotype_id": current_genotype.genotype_id,
            "pre_adapt_public_eval": {},
            "pre_adapt_private_heldout": {},
            "pre_adapt_humaneval_clean": {},
            "post_adapt_public_eval": {},
            "post_adapt_private_heldout": {},
            "post_adapt_humaneval_clean": {},
            "adaptation_losses": {},
        }

        # Public Eval Suite (Pre-Adaptation)
        for t, ldr in public_eval_loaders.items():
            l, t_acc, s_acc = evaluate_dataset(phenotype, ldr, criterion, device)
            gen_metrics["pre_adapt_public_eval"][t] = {"loss": round(l, 4), "token_acc": round(t_acc * 100, 2), "exact_match": round(s_acc * 100, 2)}

        # Private Held-Out Suite (Pre-Adaptation)
        for t, ldr in heldout_eval_loaders.items():
            l, t_acc, s_acc = evaluate_dataset(phenotype, ldr, criterion, device)
            gen_metrics["pre_adapt_private_heldout"][t] = {"loss": round(l, 4), "token_acc": round(t_acc * 100, 2), "exact_match": round(s_acc * 100, 2)}

        # Clean HumanEval Benchmark (Pre-Adaptation)
        l_he, t_he, s_he = evaluate_dataset(phenotype, humaneval_loader, criterion, device)
        gen_metrics["pre_adapt_humaneval_clean"] = {"loss": round(l_he, 4), "token_acc": round(t_he * 100, 2), "exact_match": round(s_he * 100, 2)}

        print(f"  -- Pre-Adaptation Benchmarks for {gen_name}:")
        print(f"     * Clean HumanEval: Token Acc = {t_he*100:.2f}% | Loss = {l_he:.4f}")
        for t in eval_benchmark_tasks:
            pub_acc = gen_metrics["pre_adapt_public_eval"].get(t, {}).get("token_acc", 0.0)
            pvt_acc = gen_metrics["pre_adapt_private_heldout"].get(t, {}).get("token_acc", 0.0)
            print(f"     * {t.upper():10s}: Public Eval = {pub_acc:5.2f}% | Private Held-Out = {pvt_acc:5.2f}%")

        # -----------------------------------------------------------------
        # Step D: Fast Clock Generational Adaptation on GSM8K, MATH, MBPP
        # -----------------------------------------------------------------
        print(f"\n[Step 4] Fast Clock: Phenotype Generational Adaptation...")
        opt_adapt = optim.AdamW(phenotype.parameters(), lr=5e-4, weight_decay=1e-4)

        for adapt_task_name, ldr in adaptation_loaders.items():
            loss_a = train_epoch(phenotype, ldr, opt_adapt, criterion, device, max_batches=fast_clock_steps)
            gen_metrics["adaptation_losses"][adapt_task_name] = round(loss_a, 4)
            print(f"  ✓ Adapted on {adapt_task_name.upper():8s}: Loss = {loss_a:.4f}")

        # -----------------------------------------------------------------
        # Step E: Post-Adaptation Milestone Benchmark Evaluation
        # -----------------------------------------------------------------
        print(f"\n[Step 5] Evaluating Post-Adaptation Benchmarks for {gen_name}...")

        # Public Eval Suite (Post-Adaptation)
        for t, ldr in public_eval_loaders.items():
            l, t_acc, s_acc = evaluate_dataset(phenotype, ldr, criterion, device)
            gen_metrics["post_adapt_public_eval"][t] = {"loss": round(l, 4), "token_acc": round(t_acc * 100, 2), "exact_match": round(s_acc * 100, 2)}

        # Private Held-Out Suite (Post-Adaptation)
        for t, ldr in heldout_eval_loaders.items():
            l, t_acc, s_acc = evaluate_dataset(phenotype, ldr, criterion, device)
            gen_metrics["post_adapt_private_heldout"][t] = {"loss": round(l, 4), "token_acc": round(t_acc * 100, 2), "exact_match": round(s_acc * 100, 2)}

        # Clean HumanEval Benchmark (Post-Adaptation)
        l_he_post, t_he_post, s_he_post = evaluate_dataset(phenotype, humaneval_loader, criterion, device)
        gen_metrics["post_adapt_humaneval_clean"] = {"loss": round(l_he_post, 4), "token_acc": round(t_he_post * 100, 2), "exact_match": round(s_he_post * 100, 2)}

        print(f"  -- Post-Adaptation Benchmarks for {gen_name}:")
        print(f"     * Clean HumanEval: Token Acc = {t_he_post*100:.2f}% | Loss = {l_he_post:.4f}")
        for t in eval_benchmark_tasks:
            pub_acc = gen_metrics["post_adapt_public_eval"].get(t, {}).get("token_acc", 0.0)
            pvt_acc = gen_metrics["post_adapt_private_heldout"].get(t, {}).get("token_acc", 0.0)
            print(f"     * {t.upper():10s}: Public Eval = {pub_acc:5.2f}% | Private Held-Out = {pvt_acc:5.2f}%")

        gen_metrics["generation_time_sec"] = round(time.time() - gen_start_time, 2)
        master_results["generations"].append(gen_metrics)

        # -----------------------------------------------------------------
        # Step F: Slow Clock SVD Instinct Encoding & Mutation into D_{g+1}
        # -----------------------------------------------------------------
        if g < num_generations - 1:
            print(f"\n[Step 6] Slow Clock: Distilling Phenotype Instinct via Truncated SVD into D_{g+1}...")
            learned_state = {k: v.clone() for k, v in phenotype.state_dict().items()}
            next_genotype, slow_metrics = slow_clock.step(current_genotype, learned_state)
            next_genotype = mutator.mutate(next_genotype)
            next_genotype.genotype_id = f"gen_{g+1}"
            print(f"  ✓ Instinct Distilled: Reconstruction Loss = {slow_metrics.get('reconstruction_loss', 0.0):.4f}")
            print(f"  ✓ Transition Complete: {gen_name} -> D_{g+1}")
            current_genotype = next_genotype

    total_elapsed = time.time() - start_total_time
    master_results["total_elapsed_sec"] = round(total_elapsed, 2)

    # Save Results
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[+] Full multi-generational benchmark results saved to: {output_report}")

    return master_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI-DNA Multi-Generational Evolution Benchmark Harness")
    parser.add_argument("--generations", type=int, default=3, help="Number of evolutionary generations (default: 3)")
    parser.add_argument("--data-dir", type=str, default="./ai-dna-data", help="Directory containing partitioned AI-DNA datasets")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--fast-steps", type=int, default=25, help="Fast clock adaptation steps per dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--report", type=str, default="benchmark_results_generations.json", help="Output JSON report file")

    args = parser.parse_args()
    run_multi_generational_training(
        num_generations=args.generations,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        fast_clock_steps=args.fast_steps,
        device_str=args.device,
        output_report=args.report,
    )
