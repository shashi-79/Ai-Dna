import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TopKNoisyGate(nn.Module):
    """
    Top-K Sparsely-Gated Expert Routing with noisy exploration and load balancing.
    Replaces the legacy Straight-Through Estimator (STE) hard routing.
    Based on Shazeer et al., 2017.
    """
    def __init__(self, d_model, num_experts, top_k, noise_std=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        
        # Expert gating weights
        self.w_gate = nn.Linear(d_model, num_experts, bias=True)
        # Noise gating weights for load exploration
        self.w_noise = nn.Linear(d_model, num_experts, bias=True)
        
        # Track for auxiliary loss during training
        self.register_buffer("importance", torch.zeros(num_experts))
        self.register_buffer("load", torch.zeros(num_experts))

    def _cv_squared(self, x):
        """Coefficient of Variation squared: (std / mean)^2"""
        eps = 1e-10
        if x.dim() == 1:
            mean = x.mean()
            var = x.var(unbiased=False)
            return var / (mean**2 + eps)
        return torch.zeros(1, device=x.device)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        
        # Base logits
        logits = self.w_gate(x_flat) # [B*S, E]
        
        # Inject tunable noise during training
        if self.training and self.noise_std > 0.0:
            noise_scale = F.softplus(self.w_noise(x_flat))
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise_scale * noise
        else:
            noisy_logits = logits
            
        # Top-K routing
        top_logits, top_indices = torch.topk(noisy_logits, self.top_k, dim=1) # [B*S, K]
        
        # Softmax over only the selected experts
        top_gates = F.softmax(top_logits, dim=1) # [B*S, K]
        
        # Reconstruct sparse full gates for the output
        gates = torch.zeros_like(logits).scatter_(1, top_indices, top_gates.to(dtype=logits.dtype)) # [B*S, E]
        gates = gates.view(B, S, self.num_experts)
        top_indices = top_indices.view(B, S, self.top_k)
        
        # Calculate load balancing metrics if training
        if self.training:
            # Importance: sum of gate values across tokens
            batch_importance = gates.sum(dim=(0, 1))
            self.importance = 0.9 * self.importance + 0.1 * batch_importance.detach()
            
            # Load: expected tokens per expert
            if self.noise_std > 0.0:
                kth_logits = top_logits[:, -1].unsqueeze(1) # [B*S, 1]
                noise_scale = F.softplus(self.w_noise(x_flat))
                # Normal CDF for probability of exceeding threshold
                norm = torch.distributions.Normal(0, 1)
                z = (logits - kth_logits) / (noise_scale + 1e-5)
                prob = norm.cdf(z)
                batch_load = prob.sum(dim=0)
            else:
                batch_load = (gates > 0).float().sum(dim=0)
                
            self.load = 0.9 * self.load + 0.1 * batch_load.detach()
            
        return gates, top_indices
        
    def get_load_balancing_loss(self):
        """Returns the auxiliary CV^2 balancing loss"""
        if self.importance.sum() == 0:
            return torch.tensor(0.0, device=self.importance.device)
            
        cv2_importance = self._cv_squared(self.importance)
        cv2_load = self._cv_squared(self.load)
        return cv2_importance + cv2_load
