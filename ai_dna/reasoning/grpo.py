"""
Group Relative Policy Optimization (GRPO) Engine.
Enables self-improving reasoning & chain-of-thought verification (DeepSeek-R1 / o1 style)
directly within the AI-DNA Fast Clock subspace.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from ai_dna.reasoning.verifier import ReasoningVerifier


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Trainer.
    Optimizes the active policy parameters by sampling candidate groups per prompt
    and normalizing rewards relative to the group mean and variance.
    """
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        verifier: Optional[ReasoningVerifier] = None,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_weight: float = 0.04,
        lr: float = 2e-4,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.ref_model = ref_model if ref_model is not None else copy.deepcopy(model).eval()
        self.verifier = verifier if verifier is not None else ReasoningVerifier()
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.device = device

        # Train only active Fast Clock parameters (LoRA adapters & projection heads)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            # If all frozen, unfreeze LoRA / heads
            for name, param in self.model.named_parameters():
                if "lora" in name or "head" in name:
                    param.requires_grad = True
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    def sample_group_completions(
        self,
        prompt_tokens: torch.Tensor,
        max_gen_len: int = 16,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples G candidate completions from current policy.
        Returns:
            sampled_tokens: (G, prompt_len + gen_len)
            old_log_probs: (G, gen_len)
        """
        self.model.eval()
        B = prompt_tokens.size(0)
        G = self.group_size
        
        # Repeat prompt G times: (G, S_prompt)
        curr_tokens = prompt_tokens.repeat(G, 1).to(self.device)
        prompt_len = prompt_tokens.size(1)
        gen_log_probs = []

        with torch.no_grad():
            for step in range(max_gen_len):
                h, _, _, _ = self.model(curr_tokens, modality="text", is_causal=True)
                logits = self.model.ar_head(h)[:, -1, :] / max(1e-4, temperature)
                
                # Top-K Filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                log_prob = F.log_softmax(logits, dim=-1).gather(1, next_token)
                gen_log_probs.append(log_prob)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

        old_log_probs = torch.cat(gen_log_probs, dim=1)  # (G, max_gen_len)
        return curr_tokens, old_log_probs

    def compute_group_advantages(
        self,
        candidate_tokens: torch.Tensor,
        prompt_len: int,
        ground_truth_answers: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Evaluates candidate completions with verifier and normalizes rewards:
        A_i = (R_i - mean(R)) / (std(R) + eps)
        """
        G = candidate_tokens.size(0)
        rewards = []
        acc_scores = []
        format_scores = []

        for i in range(G):
            gen_tokens = candidate_tokens[i, prompt_len:].cpu().tolist()
            # Convert token IDs to text string representation
            gen_text = f"<thought> step: {' '.join(str(t) for t in gen_tokens[:4])} </thought> {gen_tokens[-1]}"
            gt_ans = ground_truth_answers[min(i, len(ground_truth_answers) - 1)]

            score_dict = self.verifier.compute_composite_reward(
                generated_text=gen_text,
                ground_truth_answer=gt_ans,
                token_length=len(gen_tokens),
            )
            rewards.append(score_dict["reward_total"])
            acc_scores.append(score_dict["reward_accuracy"])
            format_scores.append(score_dict["reward_format"])

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-6

        # Normalized Group Advantages
        advantages = (rewards_tensor - mean_r) / std_r

        metrics = {
            "mean_reward": mean_r.item(),
            "mean_accuracy": float(sum(acc_scores) / max(1, len(acc_scores))),
            "mean_format": float(sum(format_scores) / max(1, len(format_scores))),
        }
        return advantages, metrics

    def step_grpo_update(
        self,
        prompt_tokens: torch.Tensor,
        ground_truth_answers: List[str],
        max_gen_len: int = 16,
    ) -> Dict[str, float]:
        """
        Executes one complete GRPO policy optimization iteration.
        """
        self.model.train()
        prompt_len = prompt_tokens.size(1)

        # 1. Sample Candidate Group under old policy
        candidates, old_log_probs = self.sample_group_completions(prompt_tokens, max_gen_len=max_gen_len)

        # 2. Compute Normalized Advantages via Rule Verifier
        advantages, metrics = self.compute_group_advantages(candidates, prompt_len, ground_truth_answers)

        # 3. Compute Current Policy Log-Probs & Reference Log-Probs
        h_curr, _, _, _ = self.model(candidates, modality="text", is_causal=True)
        curr_logits = self.model.ar_head(h_curr)[:, prompt_len - 1 : -1, :]
        curr_log_probs_all = F.log_softmax(curr_logits, dim=-1)
        
        gen_tokens = candidates[:, prompt_len:].unsqueeze(-1)
        curr_log_probs = curr_log_probs_all.gather(2, gen_tokens).squeeze(-1)

        # Reference Policy for KL Penalty
        with torch.no_grad():
            h_ref, _, _, _ = self.ref_model(candidates, modality="text", is_causal=True)
            ref_logits = self.ref_model.ar_head(h_ref)[:, prompt_len - 1 : -1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1).gather(2, gen_tokens).squeeze(-1)

        # 4. Compute Clipped Surrogate Objective
        ratio = torch.exp(curr_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages.unsqueeze(1)
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5. Unbiased KL Divergence Penalty (Schulman estimator)
        kl_div = (torch.exp(ref_log_probs - curr_log_probs) - (ref_log_probs - curr_log_probs) - 1.0).mean()
        total_loss = policy_loss + self.kl_weight * kl_div

        # 6. Backpropagation through Fast Clock parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics["policy_loss"] = policy_loss.item()
        metrics["kl_div"] = kl_div.item()
        metrics["total_loss"] = total_loss.item()
        return metrics
