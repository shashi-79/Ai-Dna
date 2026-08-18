"""
Interactive & CLI Manual Inference Tool for AI DNA Architecture.
Allows testing inference on trained phenotype weights or dynamically grown genotypes.

Usage:
  1. Interactive REPL Mode:
     python infer.py --interactive

  2. Command-Line Autoregressive Generation:
     python infer.py --prompt "5, 25, 45, 65" --mode autoregressive --max-tokens 12 --temp 0.7

  3. Command-Line Classification:
     python infer.py --prompt "5, 25, 45, 65" --mode classify

  4. Continuous Diffusion Denoising:
     python infer.py --mode diffusion --diff-steps 20
"""

import os
import sys
import argparse
from typing import Tuple, List, Optional
import torch

from ai_dna import (
    Genotype,
    load_genotype,
    GrowthEngine,
    InferencePipeline,
    PhenotypeNeuralNetwork,
    TextTokenizer,
)


def parse_or_encode_prompt(prompt_str: str, tokenizer: TextTokenizer) -> Tuple[torch.Tensor, str]:
    """
    Encodes text string into tokens using TextTokenizer or parses integer IDs.
    Returns (token_tensor, input_mode).
    """
    clean = prompt_str.strip()
    parts = clean.replace(",", " ").split()
    if all(p.isdigit() for p in parts) and len(parts) > 0:
        # Numeric token input
        tokens = [int(p) % tokenizer.vocab_size for p in parts]
        return torch.tensor([tokens], dtype=torch.long), "numeric"
    else:
        # Natural language text string
        tokens = tokenizer.encode(clean)
        return tokens, "text"


def load_inference_pipeline(
    checkpoint_path: str = "./checkpoints/fluent_text/phenotype_fluent.pt",
    genotype_path: str = "./checkpoints/fluent_text/genotype_fluent.json",
    device: torch.device = torch.device("cpu"),
) -> InferencePipeline:
    """Loads inference pipeline from saved genotype and/or learned phenotype weights."""
    if os.path.exists(genotype_path):
        print(f"[*] Loading Genotype DNA from: {genotype_path}")
        genotype = load_genotype(genotype_path)
    else:
        print("[*] Using default Genesis Genotype...")
        genotype = Genotype.create_default()

    checkpoint_dir = os.path.dirname(checkpoint_path) if checkpoint_path else None
    pipeline = InferencePipeline(genotype=genotype, device=device, checkpoint_dir=checkpoint_dir)

    # If full learned weights exist, load them into the phenotype
    if os.path.exists(checkpoint_path):
        print(f"[*] Loading Learned Phenotype Weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        missing_keys, unexpected_keys = pipeline.phenotype.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"    [!] Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"    [!] Warning: Unexpected keys in state dict: {unexpected_keys}")

    pipeline.phenotype.eval()
    return pipeline


def run_manual_inference(
    pipeline: InferencePipeline,
    prompt_str: str = "hello",
    mode: str = "autoregressive",
    max_tokens: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.9,
    diff_steps: int = 20,
):
    device = pipeline.device
    tokenizer = pipeline.tokenizer
    tokens, input_mode = parse_or_encode_prompt(prompt_str, tokenizer=tokenizer)
    tokens = tokens.to(device)

    print("\n" + "="*60)
    print(f" AI DNA INFERENCE (Mode: {mode.upper()})")
    print("="*60)

    if mode == "autoregressive":
        print(f" Input Text Prompt:       \"{tokenizer.decode(tokens, skip_special_tokens=False)}\"")
        print(f" Input Prompt Tokens:     {tokens.tolist()[0]}")
        
        res = pipeline.generate(
            tokens,
            modality="text",
            mode="autoregressive",
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        full_tokens = res["output"].tolist()[0]
        generated_only = full_tokens[tokens.shape[1]:]
        decoded_generated = tokenizer.decode(generated_only, skip_special_tokens=True)
        decoded_full = tokenizer.decode(full_tokens, skip_special_tokens=True)

        print(f" Generated New Tokens:    {generated_only}")
        print(f" Decoded Generated Text:  \"{decoded_generated}\"")
        print(f" Full Decoded Output:     \"{decoded_full}\"")
        
        # Token Breakdown Table
        print("\n" + "-"*60)
        print(" TOKEN BREAKDOWN TABLE")
        print("-"*60)
        print(f" {'Step':<6} | {'ID':<6} | {'Decoded Token':<20} | {'Role':<12}")
        print("-"*60)
        
        # Print prompt tokens
        for idx, (tok_id, tok_str, is_special) in enumerate(tokenizer.decode_verbose(tokens)):
            role = "Special" if is_special else "Prompt"
            # Clean up display of control chars / bytes
            disp_str = tok_str.replace("\n", "\\n").replace("\t", "\\t")
            print(f" {idx:<6} | {tok_id:<6} | {disp_str:<20} | {role:<12}")
            
        # Print generated tokens
        for idx, (tok_id, tok_str, is_special) in enumerate(tokenizer.decode_verbose(generated_only)):
            step_idx = len(tokens[0]) + idx
            role = "Generated"
            disp_str = tok_str.replace("\n", "\\n").replace("\t", "\\t")
            print(f" {step_idx:<6} | {tok_id:<6} | {disp_str:<20} | {role:<12}")
        print("-"*60)

    elif mode == "classify":
        if input_mode == "text":
            print(f" Input Text Prompt:       \"{prompt_str}\"")
        print(f" Input Sequence Tokens:   {tokens.tolist()[0]}")
        res = pipeline.generate(tokens, modality="text", mode="classify")
        logits = res["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        pred_class = res["predictions"].item()
        top_prob = probs[pred_class].item()

        print(f" Predicted Class:         Class {pred_class} (Confidence: {top_prob:.2%})")
        print(f" Top-4 Class Probabilities:")
        top_vals, top_idx = torch.topk(probs, min(4, probs.shape[-1]))
        for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
            print(f"   - Class {idx}: {val:6.2%}")
        print(f" Dynamic Memory Latency:  {res['metrics']['t_seq_ms']:.2f} ms")
        print(f" Peak Activation Memory:  {res['metrics']['peak_mem_kb']:.2f} KB")

    elif mode == "diffusion":
        print(f" Continuous Diffusion Denoising across {diff_steps} steps...")
        res = pipeline.generate(
            tokens,
            modality="text",
            mode="diffusion",
            num_diff_steps=diff_steps,
        )
        out_tensor = res["output"]
        print(f" Generated Latent Shape:  {list(out_tensor.shape)}")
        print(f" Latent Norm:             {out_tensor.norm().item():.4f}")
        print(f" Compute Cost (C_comp):   {res['metrics']['c_compute']:.2f}")

    print("="*60 + "\n")


def interactive_repl(pipeline: InferencePipeline):
    """Interactive console REPL for real-time inference testing."""
    print("\n" + "="*60)
    print(" AI DNA Interactive Manual Inference REPL")
    print(" Commands:")
    print("   ar <tokens...>   -> Autoregressive generation (e.g., 'ar 5 25 45')")
    print("   cls <tokens...>  -> Classification (e.g., 'cls 5 25 45')")
    print("   diff             -> Continuous Diffusion generation")
    print("   exit / quit      -> Exit REPL")
    print("="*60 + "\n")

    while True:
        try:
            line = input("AI-DNA> ").strip()
            if not line:
                continue
            if line.lower() in ["exit", "quit", "q"]:
                print("Exiting REPL.")
                break

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg_str = parts[1] if len(parts) > 1 else "1 2 3 4"

            if cmd in ["ar", "autoregressive", "gen"]:
                run_manual_inference(pipeline, prompt_str=arg_str, mode="autoregressive")
            elif cmd in ["cls", "classify", "predict"]:
                run_manual_inference(pipeline, prompt_str=arg_str, mode="classify")
            elif cmd in ["diff", "diffusion"]:
                run_manual_inference(pipeline, prompt_str=arg_str, mode="diffusion")
            else:
                # Default to autoregressive
                run_manual_inference(pipeline, prompt_str=line, mode="autoregressive")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"[Error] {e}")


def main():
    try:
        import sys
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Manual Inference Tool for AI DNA")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive REPL mode")
    parser.add_argument("--prompt", type=str, default="5 25 45 65", help="Input tokens (e.g., '5, 25, 45, 65')")
    parser.add_argument("--mode", type=str, default="classify", choices=["autoregressive", "classify", "diffusion"])
    parser.add_argument("--max-tokens", type=int, default=10, help="Max new tokens for autoregressive mode")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--diff-steps", type=int, default=20, help="Diffusion denoising steps")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/fluent_text/phenotype_fluent.pt", help="Path to weights")
    parser.add_argument("--genotype", type=str, default="./checkpoints/fluent_text/genotype_fluent.json", help="Path to genotype")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', 'cuda')")

    args = parser.parse_args()
    if args.device in ["auto", "cuda"] and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[*] Using GPU: {torch.cuda.get_device_name(0)}")
    elif args.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pipeline = load_inference_pipeline(
        checkpoint_path=args.checkpoint,
        genotype_path=args.genotype,
        device=device,
    )

    if args.interactive:
        interactive_repl(pipeline)
    else:
        run_manual_inference(
            pipeline=pipeline,
            prompt_str=args.prompt,
            mode=args.mode,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            diff_steps=args.diff_steps,
        )


if __name__ == "__main__":
    main()
