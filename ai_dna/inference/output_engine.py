"""
Inference Output Engine.
Supports Autoregressive sequence generation with temperature/top-k/top-p sampling,
Multi-step Diffusion denoising generation, and classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Callable
from ..models.modules import AutoregressiveDecoderHead, DiffusionDecoderHead, ClassificationHead


class OutputEngine(nn.Module):
    """
    Multi-mode dynamic decoding engine.
    """
    def __init__(
        self,
        ar_head: AutoregressiveDecoderHead,
        diff_head: DiffusionDecoderHead,
        cls_head: ClassificationHead,
    ):
        super().__init__()
        self.ar_head = ar_head
        self.diff_head = diff_head
        self.cls_head = cls_head

    def sample_top_p_top_k(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Samples token index from logits with temperature, top-k, and nucleus top-p filtering.
        logits: (B, vocab_size)
        """
        if temperature > 0.0:
            logits = logits / temperature
        else:
            return torch.argmax(logits, dim=-1, keepdim=True)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            sorted_logits[sorted_indices_to_remove] = -float("Inf")
            logits = torch.gather(sorted_logits, 1, torch.argsort(sorted_indices, dim=-1))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def generate_autoregressive(
        self,
        forward_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Any, Any, Any]],
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: int = -1,
    ) -> torch.Tensor:
        """
        Autoregressively generates next tokens given a prompt tensor (B, S).
        Stops early if eos_token_id is sampled (set to -1 to disable).
        """
        curr_tokens = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            # Forward pass through phenotype backbone
            h, _, _, _ = forward_fn(curr_tokens)  # (B, S_curr, D_model)
            logits = self.ar_head(h[:, -1:, :]).squeeze(1).clone()  # (B, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(curr_tokens.shape[0]):
                    unique_tokens = torch.unique(curr_tokens[b])
                    for tok in unique_tokens:
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty

            next_token = self.sample_top_p_top_k(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

            # EOS early stopping
            if eos_token_id >= 0 and (next_token == eos_token_id).all():
                break

        return curr_tokens

    def sample_diffusion(
        self,
        h_context: torch.Tensor,
        num_steps: int = 20,
        shape: Optional[Tuple[int, ...]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Samples continuous modality representations using DDPM denoising.
        h_context: (B, S, D_model)
        """
        batch_size, seq_len, d_model = h_context.shape
        out_shape = shape or (batch_size, seq_len, d_model)
        
        # Start from standard Gaussian noise
        x_t = torch.randn(out_shape, device=device)
        timesteps = torch.linspace(num_steps - 1, 0, num_steps, device=device).long()

        # Define DDPM noise schedule (Linear)
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        for step in timesteps:
            t = torch.full((batch_size,), step.item(), device=device, dtype=torch.long)
            with torch.no_grad():
                eps_pred = self.diff_head(x_t, t, h_context)
            
            alpha_t = alphas[step]
            alpha_bar_t = alphas_cumprod[step]
            
            # Predict x_0 for logging/clipping if needed
            x_0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            
            if step > 0:
                noise = torch.randn_like(x_t)
                mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
                variance = betas[step]
                x_t = mean + torch.sqrt(variance) * noise
            else:
                x_t = x_0

        return x_t

    def classify(self, h: torch.Tensor) -> torch.Tensor:
        """Decodes latent states into class logits."""
        return self.cls_head(h)
