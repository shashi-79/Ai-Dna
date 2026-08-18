"""
Working Memory Module.
Handles local chunked attention within bounded window size C_chunk using RoPE and TurboQuant KV cache.
Implements idea.md Section 9.2.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from ..models.rope import RoPE
from .turboquant import TurboQuant


class WorkingMemory(nn.Module):
    """
    Working Memory window that restricts active dense attention to local chunks of length C_chunk.
    Integrates 1D RoPE relative positioning and TurboQuant online KV compression.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        chunk_size: int = 32,
        kv_quant_bits: int = 3,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = d_model // num_heads
        self.kv_quant_bits = kv_quant_bits

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RoPE(self.head_dim, base=rope_base)
        self.turbo_quant = TurboQuant(d_model=d_model, b_quant=kv_quant_bits)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        quantize_kv: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, S, D_model)
        quantize_kv: whether to compress KV cache using TurboQuant during chunk processing
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

            q_raw = self.q_proj(chunk)
            k_raw = self.k_proj(chunk)
            v_raw = self.v_proj(chunk)

            # Optional TurboQuant KV compression (simulates quantized KV store)
            if quantize_kv:
                k_q = self.turbo_quant.quantize(k_raw)
                k_raw = self.turbo_quant.dequantize(k_q)
                v_q = self.turbo_quant.quantize(v_raw)
                v_raw = self.turbo_quant.dequantize(v_q)

            q = q_raw.view(batch_size, s_chunk, self.num_heads, self.head_dim).transpose(1, 2)
            k = k_raw.view(batch_size, s_chunk, self.num_heads, self.head_dim).transpose(1, 2)
            v = v_raw.view(batch_size, s_chunk, self.num_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE to query and key
            q, k = self.rope(q, k)

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
