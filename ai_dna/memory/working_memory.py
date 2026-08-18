"""
Working Memory Module.
Handles local chunked attention within bounded window size C_chunk.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class WorkingMemory(nn.Module):
    """
    Working Memory window that restricts active dense attention to local chunks of length C_chunk.
    """
    def __init__(self, d_model: int, num_heads: int = 4, chunk_size: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        x: (B, S, D_model)
        Returns: Attention outputs processed in chunks of size chunk_size.
        """
        batch_size, seq_len, d_model = x.shape
        num_chunks = math.ceil(seq_len / self.chunk_size)

        outputs = []
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(seq_len, (i + 1) * self.chunk_size)
            chunk = x[:, start_idx:end_idx, :]  # (B, S_chunk, D_model)
            s_chunk = chunk.shape[1]

            q = self.q_proj(chunk).view(batch_size, s_chunk, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(chunk).view(batch_size, s_chunk, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(chunk).view(batch_size, s_chunk, self.num_heads, self.head_dim).transpose(1, 2)

            try:
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            except Exception:
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if is_causal:
                    mask = torch.tril(torch.ones((s_chunk, s_chunk), device=x.device)).unsqueeze(0).unsqueeze(0)
                    scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
                attn_weights = F.softmax(scores, dim=-1)
                attn_out = torch.matmul(attn_weights, v)

            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, s_chunk, d_model)
            out_chunk = self.out_proj(attn_out)
            outputs.append(chunk + out_chunk)

        return torch.cat(outputs, dim=1)
