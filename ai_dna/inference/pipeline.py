"""
Unified Inference Pipeline for AI DNA Phenotypes.
Supports both Eager (cached phenotype) and Lazy (dynamic CPPN-grown parameter slices) inference.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple
from ..models.phenotype import PhenotypeNeuralNetwork
from .output_engine import OutputEngine
from .sparse_executor import SparseHardwareExecutor
from .tokenizer import TextTokenizer


class InferencePipeline:
    """
    Unified Inference Engine for AI DNA.
    Strictly accepts a fully grown Phenotype Neural Network, NOT a Genotype.
    """
    def __init__(
        self,
        phenotype: Optional[PhenotypeNeuralNetwork] = None,
        tokenizer: Optional[TextTokenizer] = None,
        device: Optional[torch.device] = None,
        genotype: Optional[Any] = None,
        **kwargs,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if tokenizer is None:
            from .tokenizer import TextTokenizer
            tokenizer = TextTokenizer()
        self.tokenizer = tokenizer

        if phenotype is None and genotype is not None:
            from ..growth.engine import GrowthEngine
            growth_engine = GrowthEngine(device=device)
            phenotype = growth_engine.grow_phenotype_model(genotype)
        elif phenotype is None:
            raise ValueError("Must provide either phenotype or genotype to InferencePipeline.")

        self.phenotype = phenotype.to(device)
        self.phenotype.eval()

        # Output Engine for multi-modal decoding
        self.output_engine = OutputEngine(
            ar_head=self.phenotype.ar_head,
            diff_head=self.phenotype.diff_head,
            cls_head=self.phenotype.cls_head,
        )
        self.sparse_executor = SparseHardwareExecutor(num_experts=self.phenotype.num_experts)

    def generate(
        self,
        inputs: torch.Tensor,
        modality: str = "text",
        mode: str = "autoregressive",
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_diff_steps: int = 20,
        stop_on_eos: bool = False,
    ) -> Dict[str, Any]:
        """
        Unified inference entry point.
        Modality intake is handled by PhenotypeNeuralNetwork.encode_input() directly.
        """
        inputs = inputs.to(self.device)
        with torch.no_grad():
            if mode == "autoregressive":
                def forward_fn(t_tokens):
                    return self.phenotype(t_tokens, modality="text", is_causal=True)
                generated_tokens = self.output_engine.generate_autoregressive(
                    forward_fn=forward_fn,
                    prompt_tokens=inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id if stop_on_eos else -1,
                )
                return {"mode": "autoregressive", "output": generated_tokens}

            elif mode == "diffusion":
                h_latents, _, _, mem_metrics = self.phenotype(inputs, modality=modality)
                sampled_continuous = self.output_engine.sample_diffusion(
                    h_context=h_latents,
                    num_steps=num_diff_steps,
                    device=self.device,
                )
                return {
                    "mode": "diffusion",
                    "output": sampled_continuous,
                    "metrics": mem_metrics,
                }

            elif mode == "classify":
                h_latents, _, _, mem_metrics = self.phenotype(inputs, modality=modality)
                logits = self.output_engine.classify(h_latents)
                preds = torch.argmax(logits, dim=-1)
                return {
                    "mode": "classify",
                    "logits": logits,
                    "predictions": preds,
                    "metrics": mem_metrics,
                }
            else:
                raise ValueError(f"Unknown decoding mode: {mode}. Supported: ['autoregressive', 'diffusion', 'classify']")
