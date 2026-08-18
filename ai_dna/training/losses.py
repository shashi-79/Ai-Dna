"""
Multi-Task Loss Functions.
Joint Training Objective: L_total = lambda_AR * L_AR + lambda_Diff * L_Diff + lambda_bal * L_bal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class JointLoss(nn.Module):
    """
    Computes joint loss across Autoregressive, Diffusion, Classification, and MoE Expert balancing.
    """
    def __init__(
        self,
        lambda_ar: float = 1.0,
        lambda_diff: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_bal: float = 0.01,
    ):
        super().__init__()
        self.lambda_ar = lambda_ar
        self.lambda_diff = lambda_diff
        self.lambda_cls = lambda_cls
        self.lambda_bal = lambda_bal

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.mse_loss = nn.MSELoss()

    def autoregressive_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, S, vocab_size), targets: (B, S)"""
        vocab_size = logits.shape[-1]
        return self.ce_loss(logits.view(-1, vocab_size), targets.view(-1))

    def classification_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, num_classes), targets: (B,)"""
        return self.ce_loss(logits, targets)

    def diffusion_loss(self, eps_pred: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        """eps_pred: (B, S, D), target_noise: (B, S, D)"""
        return self.mse_loss(eps_pred, target_noise)

    def forward(
        self,
        ar_logits: Optional[torch.Tensor] = None,
        ar_targets: Optional[torch.Tensor] = None,
        cls_logits: Optional[torch.Tensor] = None,
        cls_targets: Optional[torch.Tensor] = None,
        diff_pred: Optional[torch.Tensor] = None,
        diff_target: Optional[torch.Tensor] = None,
        aux_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=aux_loss.device if aux_loss is not None else torch.device("cpu"))
        breakdown = {}

        if ar_logits is not None and ar_targets is not None:
            l_ar = self.autoregressive_loss(ar_logits, ar_targets)
            total_loss = total_loss + self.lambda_ar * l_ar
            breakdown["loss_ar"] = l_ar.item()

        if cls_logits is not None and cls_targets is not None:
            l_cls = self.classification_loss(cls_logits, cls_targets)
            total_loss = total_loss + self.lambda_cls * l_cls
            breakdown["loss_cls"] = l_cls.item()

        if diff_pred is not None and diff_target is not None:
            l_diff = self.diffusion_loss(diff_pred, diff_target)
            total_loss = total_loss + self.lambda_diff * l_diff
            breakdown["loss_diff"] = l_diff.item()

        if aux_loss is not None:
            total_loss = total_loss + aux_loss
            breakdown["loss_aux_balance"] = aux_loss.item()

        breakdown["loss_total"] = total_loss.item()
        return total_loss, breakdown
