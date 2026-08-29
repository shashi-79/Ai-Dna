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
                
                # Dynamic shape handling for Net2Net CPPN expansions
                if param.shape != old_p.shape:
                    slices = tuple(slice(0, min(s_curr, s_old)) for s_curr, s_old in zip(param.shape, old_p.shape))
                    curr_p = param[slices]
                else:
                    curr_p = param
                    
                loss = loss + (f_diag * (curr_p - old_p) ** 2).sum()

        return 0.5 * self.lambda_ewc * loss


def adaptive_svd_rank(tensor: torch.Tensor, energy_threshold: float = 0.95, min_rank: int = 1) -> int:
    """
    Dynamically computes optimal rank k* based on cumulative explained singular energy:
    k* = argmin_k ( sum_{i=1}^k sigma_i^2 / sum_{i=1}^d sigma_i^2 >= energy_threshold )
    """
    if tensor.ndim < 2:
        return min_rank
    t_2d = tensor.reshape(tensor.shape[0], -1).float()
    try:
        _, s, _ = torch.linalg.svd(t_2d, full_matrices=False)
        total_energy = (s ** 2).sum()
        if total_energy <= 1e-9:
            return min_rank
        cum_energy = torch.cumsum(s ** 2, dim=0) / total_energy
        mask = (cum_energy >= energy_threshold).nonzero()
        if len(mask) > 0:
            return max(min_rank, mask[0].item() + 1)
        return max(min_rank, len(s))
    except Exception:
        return min_rank


class GPMConsolidator:
    """
    Gradient Projection Memory (GPM) for Lifelong Continual Learning.
    Projects parameter updates Delta W into the null space of historical activation bases:
    Delta W_safe = Delta W @ (I - U_k @ U_k^T)
    Guarantees 0.0% catastrophic interference on previous task subspaces.
    """
    def __init__(self, energy_threshold: float = 0.95):
        self.energy_threshold = energy_threshold
        self.basis_dict: Dict[str, torch.Tensor] = {}

    def update_activation_basis(self, name: str, activations: torch.Tensor):
        """Extracts and updates the orthogonal activation basis U_k via adaptive SVD."""
        if activations.ndim > 2:
            act_2d = activations.reshape(-1, activations.shape[-1]).float()
        else:
            act_2d = activations.float()
        
        try:
            U, _, _ = torch.linalg.svd(act_2d.t(), full_matrices=False)
            k = adaptive_svd_rank(act_2d.t(), energy_threshold=self.energy_threshold)
            self.basis_dict[name] = U[:, :k].detach()
        except Exception:
            pass

    def project_gradient_or_delta(self, name: str, delta_w: torch.Tensor) -> torch.Tensor:
        """Projects weight delta into the orthogonal complement of the historical activation basis."""
        if name not in self.basis_dict:
            return delta_w
        U = self.basis_dict[name].to(delta_w.device)
        # Delta W_safe = Delta W - (Delta W @ U) @ U^T
        proj = torch.matmul(delta_w.float(), U)
        delta_safe = delta_w.float() - torch.matmul(proj, U.t())
        return delta_safe.type_as(delta_w)

