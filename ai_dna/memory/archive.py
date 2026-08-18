"""
Compressed Memory Archive.
Compresses historical token chunks into dense latent vectors using compression rate c_rate.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional


class CompressedArchive(nn.Module):
    """
    Compresses processed chunks of shape (B, C_chunk, D_model) into (B, num_latents, D_model)
    where num_latents = ceil(C_chunk * c_rate).
    """
    def __init__(self, d_model: int, compression_rate: float = 0.25):
        super().__init__()
        self.d_model = d_model
        self.compression_rate = compression_rate

        self.compressor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def compress_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: (B, S_chunk, D_model)
        Returns: compressed historical latent tensor (B, S_compressed, D_model)
        """
        batch_size, s_chunk, d_model = chunk.shape
        num_compressed = max(1, math.ceil(s_chunk * self.compression_rate))

        # Adaptive pooling across sequence dimension to achieve desired compression rate
        # chunk: (B, D_model, S_chunk)
        chunk_t = chunk.transpose(1, 2)
        pooled = nn.functional.adaptive_avg_pool1d(chunk_t, num_compressed)
        pooled = pooled.transpose(1, 2)  # (B, num_compressed, D_model)

        return self.compressor(pooled)
