"""
AI-DNA Inference Comparison Runner.
Generates output from both CPPN (Standard) and LoRA+CPPN (Hybrid) phenotypes for qualitative comparison.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath("."))

from ai_dna.dna.structure import Genotype
from ai_dna.growth.engine import GrowthEngine
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.encoding.slow_clock import SlowClockEncoder
from ai_dna.inference.pipeline import InferencePipeline
from data import CustomTextTokenizer
from run_comparison_benchmark import load_stratified_training_data


def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CustomTextTokenizer(vocab_size=256, mode="word")
    growth_engine = GrowthEngine(device=device)
    criterion = nn.CrossEntropyLoss()

    # Load training subset
    train_data = load_stratified_training_data(max_samples=100)
    train_subset = train_data[:40]

    test_suite = [
        {"prompt": "Calculate 60 + 5.", "expected": "65"},
        {"prompt": "Calculate 90 + 20.", "expected": "110"},
        {"prompt": "What is the next number in sequence [10, 12, 14, 16]?", "expected": "18"},
        {"prompt": "If beta contains delta, and delta contains alpha, does beta contain alpha?", "expected": "Yes"},
    ]

    # Setup genotype templates
    genotype_ref = Genotype.create_default(genotype_id="baseline")
    genotype_ref.dna_architecture.vocab_size = 256
    genotype_ref.dna_architecture.d_model = 64
    genotype_ref.dna_architecture.num_layers = 2
    genotype_ref.dna_architecture.num_experts = 2
    genotype_ref.dna_architecture.d_expert_hidden = 64
    genotype_ref.dna_architecture.coord_dim = 32

    # Run CPPN (Standard) for 3 generations
    print("[+] Training and evolving CPPN (Standard) Model (3 Generations)...", flush=True)
    slow_clock_cppn = SlowClockEncoder(rank_ratio=0.5, encoder_steps=20, device=device)
    current_genotype_cppn = genotype_ref.clone("cppn_gen0")
    for g in range(3):
        model = growth_engine.grow_phenotype_model(current_genotype_cppn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        for epoch in range(150):
            for item in train_subset:
                prompt = item.get("prompt", item.get("input", ""))
                target = item.get("solution", item.get("target", item.get("output", "")))
                text_pair = prompt + " " + target
                tokens = tokenizer.encode(text_pair).unsqueeze(0).to(device)
                if tokens.shape[1] > 2:
                    optimizer.zero_grad()
                    h, aux_loss, _, _ = model(tokens[:, :-1], modality="text", is_causal=True)
                    logits = model.ar_head(h)
                    targets = tokens[:, 1:]
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)) + aux_loss
                    loss.backward()
                    optimizer.step()
        next_genotype, _ = slow_clock_cppn.step(
            current_genotype_cppn, model.state_dict(), phenotype_model=model, protect_ancestral=False
        )
        current_genotype_cppn = next_genotype

    # Run LoRA + CPPN (Hybrid) for 3 generations
    print("[+] Training and evolving LoRA + CPPN (Hybrid) Model (3 Generations)...", flush=True)
    slow_clock_lora = SlowClockEncoder(rank_ratio=0.5, encoder_steps=150, device=device)
    current_genotype_lora = current_genotype_cppn.clone("lora_gen0")
    current_genotype_lora.dna_architecture.lora_rank = 4

    for g in range(3):
        model = growth_engine.grow_phenotype_model(current_genotype_lora)
        from ai_dna.models.lora import freeze_model_except_lora
        freeze_model_except_lora(model)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1.5e-3, weight_decay=1e-4)
        for epoch in range(150):
            for item in train_subset:
                prompt = item.get("prompt", item.get("input", ""))
                target = item.get("solution", item.get("target", item.get("output", "")))
                text_pair = prompt + " " + target
                tokens = tokenizer.encode(text_pair).unsqueeze(0).to(device)
                if tokens.shape[1] > 2:
                    optimizer.zero_grad()
                    h, aux_loss, _, _ = model(tokens[:, :-1], modality="text", is_causal=True)
                    logits = model.ar_head(h)
                    targets = tokens[:, 1:]
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1)) + aux_loss
                    loss.backward()
                    optimizer.step()
        next_genotype, _ = slow_clock_lora.step(
            current_genotype_lora, model.state_dict(), phenotype_model=model, growth_engine=growth_engine, protect_ancestral=False
        )
        current_genotype_lora = next_genotype

    # Final Inference and comparison table
    print("\n[+] Running final inference and generating comparison...", flush=True)
    model_cppn = growth_engine.grow_phenotype_model(current_genotype_cppn)
    model_lora = growth_engine.grow_phenotype_model(current_genotype_lora)

    pipeline_cppn = InferencePipeline(phenotype=model_cppn, tokenizer=tokenizer, device=device)
    pipeline_lora = InferencePipeline(phenotype=model_lora, tokenizer=tokenizer, device=device)

    print("=" * 120)
    print(" INFERENCE OUTPUT COMPARISON")
    print("=" * 120)
    print(f"{'Prompt':<45} | {'Expected':<10} | {'CPPN (Standard) Output':<25} | {'LoRA + CPPN (Hybrid) Output':<25}")
    print("-" * 120)

    for item in test_suite:
        prompt = item["prompt"]
        expected = item["expected"]
        prompt_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)

        # CPPN output
        res_cppn = pipeline_cppn.generate(prompt_ids, modality="text", max_new_tokens=10, temperature=0.1)
        out_cppn = tokenizer.decode(res_cppn["output"].squeeze(0))[len(prompt):].strip()

        # LoRA output
        res_lora = pipeline_lora.generate(prompt_ids, modality="text", max_new_tokens=10, temperature=0.1)
        out_lora = tokenizer.decode(res_lora["output"].squeeze(0))[len(prompt):].strip()

        # Clean newlines/excess spaces for clean table formatting
        out_cppn_clean = out_cppn.replace("\n", " ").strip()
        out_lora_clean = out_lora.replace("\n", " ").strip()

        # Sanitize non-ASCII and replacement characters
        out_cppn_clean = out_cppn_clean.encode('ascii', errors='ignore').decode('ascii')[:23]
        out_lora_clean = out_lora_clean.encode('ascii', errors='ignore').decode('ascii')[:23]

        print(f"{prompt:<45} | {expected:<10} | {out_cppn_clean:<25} | {out_lora_clean:<25}")


if __name__ == "__main__":
    run_comparison()
