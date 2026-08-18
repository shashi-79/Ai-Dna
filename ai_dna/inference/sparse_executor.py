"""
Sparse Hardware-Aware Executor.
Implements the 3-stage execution pipeline:
Permute -> Grouped Sparse MoE GEMM -> Unpermute.
Bypasses inactive experts to minimize memory traffic and compute latency.
Leverages custom Triton GPU acceleration kernels when available.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from .triton_kernels import TritonSparseMoEExecutor, is_triton_available


class SparseHardwareExecutor(nn.Module):
    """
    Optimizes sparse expert evaluation by grouping active tokens per expert.
    Supports both custom Triton GPU acceleration kernels and pure PyTorch execution.
    """
    def __init__(self, num_experts: int, use_triton_if_available: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.use_triton_if_available = use_triton_if_available
        self.triton_executor = TritonSparseMoEExecutor()

    def execute_sparse_moe(
        self,
        x: torch.Tensor,
        gate_probs: torch.Tensor,
        hard_mask: torch.Tensor,
        expert_modules: nn.ModuleList,
    ) -> torch.Tensor:
        """
        x: (B, S, D_model)
        gate_probs: (B, S, E_max)
        hard_mask: (B, S, E_max)
        expert_modules: ModuleList of expert networks
        Returns:
            y: (B, S, D_model)
        """
        import torch.distributed as dist
        
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)  # (N, D_model) where N = B * S
        flat_mask = hard_mask.view(-1, self.num_experts)  # (N, E_max)
        flat_probs = gate_probs.view(-1, self.num_experts)  # (N, E_max)

        out_flat = torch.zeros_like(flat_x)
        can_use_triton = self.use_triton_if_available and is_triton_available() and x.is_cuda

        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        rank = dist.get_rank() if is_dist else 0

        for e in range(self.num_experts):
            # 1. Permute / Index: Find active tokens for this expert
            active_indices = torch.nonzero(flat_mask[:, e] > 0).squeeze(-1)
            if active_indices.numel() == 0:
                continue  # Inactive expert bypassed completely

            # Extract active tokens
            x_e = flat_x[active_indices]  # (N_active, D_model)
            expert = expert_modules[e]
            expert_owner = e % world_size

            # 2. Distributed Token Dispatching
            if is_dist and world_size > 1:
                # Count active tokens across all ranks
                local_count = torch.tensor([x_e.shape[0]], dtype=torch.long, device=x.device)
                all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
                dist.all_gather(all_counts, local_count)
                
                max_tokens = max([c.item() for c in all_counts])
                
                # Pad tensor to max_tokens for collective communication
                padded_x = torch.zeros((max_tokens, d_model), dtype=x.dtype, device=x.device)
                padded_x[:local_count.item(), :] = x_e
                
                # Gather all tokens to the expert_owner rank
                if rank == expert_owner:
                    gathered_x = [torch.zeros_like(padded_x) for _ in range(world_size)]
                    dist.gather(padded_x, gather_list=gathered_x, dst=expert_owner)
                    
                    # Compute forward pass for all gathered tokens
                    all_y = []
                    for i, t in enumerate(gathered_x):
                        t_active = t[:all_counts[i].item(), :]
                        if t_active.shape[0] > 0:
                            all_y.append(expert(t_active))
                        else:
                            all_y.append(torch.empty((0, d_model), device=x.device))
                            
                    # Scatter outputs back
                    scatter_list = [torch.zeros((max_tokens, d_model), device=x.device) for _ in range(world_size)]
                    for i, y_t in enumerate(all_y):
                        scatter_list[i][:y_t.shape[0], :] = y_t
                        
                    padded_y = torch.zeros_like(padded_x)
                    dist.scatter(padded_y, scatter_list=scatter_list, src=expert_owner)
                    y_e = padded_y[:local_count.item(), :]
                else:
                    dist.gather(padded_x, dst=expert_owner)
                    padded_y = torch.zeros_like(padded_x)
                    dist.scatter(padded_y, src=expert_owner)
                    y_e = padded_y[:local_count.item(), :]
            else:
                # 2. Local Grouped GEMM computation
                if can_use_triton and hasattr(expert, "up_proj") and hasattr(expert, "down_proj"):
                    # Fused Triton accelerated execution
                    h_up = self.triton_executor.triton_gemm(x_e, expert.up_proj.weight.t())
                    h_act = torch.nn.functional.silu(h_up)
                    y_e = self.triton_executor.triton_gemm(h_act, expert.down_proj.weight.t())
                else:
                    y_e = expert(x_e)  # (N_active, D_model)

            # Gate scaling
            probs_e = flat_probs[active_indices, e:e+1]  # (N_active, 1)
            scaled_y = y_e * probs_e

            # 3. Un-permute / Scatter-Add back to original positions
            out_flat.index_add_(0, active_indices, scaled_y)

        return out_flat.view(batch_size, seq_len, d_model)
