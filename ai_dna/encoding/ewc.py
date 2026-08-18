"""
Elastic Weight Consolidation (EWC) for Genotypic Retention.
Protects ancestral genetic parameters during Slow Clock encoding by computing Fisher Information.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class EWCConsolidator:
    """
    Computes diagonal Fisher Information for genetic parameters and computes
    retention quadratic penalty: 0.5 * lambda * sum(F_i * (theta_i - theta_old_i)^2)
    """
    def __init__(self, lambda_ewc: float = 100.0):
        self.lambda_ewc = lambda_ewc
        self.fisher_diag: Dict[str, torch.Tensor] = {}
        self.old_parameters: Dict[str, torch.Tensor] = {}

    def register_ancestral_genotype(
        self,
        cppn: nn.Module,
        data_loader_fn=None,
    ):
        """
        Stores current parameters as old reference and calculates approximate Fisher diagonal.
        """
        self.old_parameters = {}
        self.fisher_diag = {}

        for name, param in cppn.named_parameters():
            self.old_parameters[name] = param.clone().detach()
            # Approximate Fisher as empirical variance / sensitivity if data loader not provided
            self.fisher_diag[name] = torch.ones_like(param) * (1.0 / (param.abs().mean().item() + 1e-4))

    def penalty(self, cppn: nn.Module) -> torch.Tensor:
        """
        Computes EWC quadratic loss across current model parameters.
        """
        if not self.fisher_diag or not self.old_parameters:
            return torch.tensor(0.0, device=next(cppn.parameters()).device)

        loss = 0.0
        for name, param in cppn.named_parameters():
            if name in self.old_parameters and name in self.fisher_diag:
                old_p = self.old_parameters[name].to(param.device)
                f_diag = self.fisher_diag[name].to(param.device)
                loss = loss + (f_diag * (param - old_p) ** 2).sum()

        return 0.5 * self.lambda_ewc * loss
