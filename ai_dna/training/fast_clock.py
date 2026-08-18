"""
Fast Clock: Parametric Learning Loop.
Optimizes phenotype parameters W_0 -> W* while keeping genotype D frozen.
W_{t+1} = Optimizer(W_t, grad_W L_total)
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from ..models.phenotype import PhenotypeNeuralNetwork
from .losses import JointLoss


class FastClockTrainer:
    """
    Manages phenotype learning during the Fast Clock.
    Genotype remains strictly frozen.
    """
    def __init__(
        self,
        phenotype: PhenotypeNeuralNetwork,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        device: torch.device = torch.device("cpu"),
        use_amp: Optional[bool] = None,
    ):
        self.phenotype = phenotype.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.use_amp = (self.device.type == "cuda") if use_amp is None else use_amp
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.optimizer = optim.AdamW(
            self.phenotype.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.loss_fn = JointLoss()

    def train_step_autoregressive(
        self,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Executes single gradient descent step for Autoregressive sequence prediction.
        """
        self.phenotype.train()
        self.optimizer.zero_grad()

        input_tokens = input_tokens.to(self.device, non_blocking=True)
        target_tokens = target_tokens.to(self.device, non_blocking=True)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            h, aux_loss, _, _ = self.phenotype(input_tokens, modality="text", is_causal=True)
            logits = self.phenotype.ar_head(h)
            loss, breakdown = self.loss_fn(ar_logits=logits, ar_targets=target_tokens, aux_loss=aux_loss)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.phenotype.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.phenotype.parameters(), self.gradient_clip)
            self.optimizer.step()

        return loss.item(), breakdown

    def train_step_classification(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        modality: str = "text",
    ) -> Tuple[float, Dict[str, float]]:
        """
        Executes single gradient descent step for Classification task.
        """
        self.phenotype.train()
        self.optimizer.zero_grad()

        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        if targets.dtype != torch.long:
            targets = targets.long()

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            h, aux_loss, _, _ = self.phenotype(inputs, modality=modality)
            logits = self.phenotype.cls_head(h)
            loss, breakdown = self.loss_fn(cls_logits=logits, cls_targets=targets, aux_loss=aux_loss)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.phenotype.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.phenotype.parameters(), self.gradient_clip)
            self.optimizer.step()

        return loss.item(), breakdown

    def evaluate_classification(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        modality: str = "text",
    ) -> Tuple[float, float]:
        """
        Evaluates accuracy and loss.
        """
        self.phenotype.eval()
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        with torch.no_grad():
            h, aux_loss, _, _ = self.phenotype(inputs, modality=modality)
            logits = self.phenotype.cls_head(h)
            loss, _ = self.loss_fn(cls_logits=logits, cls_targets=targets, aux_loss=aux_loss)
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == targets).float().mean().item()

        return accuracy, loss.item()
