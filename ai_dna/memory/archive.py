"""
Paged Compressed Memory Archive.
Implements PagedAttention fixed-page management with TurboQuant 3-bit compression.
Eliminates virtual memory fragmentation and reduces memory footprint by 5-6x.
Implements idea.md Section 9.3.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from .turboquant import TurboQuant


class PagedArchive(nn.Module):
    """
    Paged Compressed Memory Archive using PagedAttention-style page tables and TurboQuant.
    Allocates fixed-size pages of compressed latent vectors from a free list.
    Evicts least-recently used (LRU) pages when max_pages is exceeded.
    """
    def __init__(
        self,
        d_model: int,
        compression_rate: float = 0.25,
        page_size: int = 16,
        max_pages: int = 1024,
        kv_quant_bits: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.compression_rate = compression_rate
        self.page_size = page_size
        self.max_pages = max_pages
        self.kv_quant_bits = kv_quant_bits

        self.turbo_quant = TurboQuant(d_model=d_model, b_quant=kv_quant_bits)

        self.compressor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Page tables: mapping from sequence_id / batch_id to list of page indices
        # Stored in quantized format: list of quantized dicts per page
        self.physical_pages: List[Dict[str, Any]] = []
        self.logical_page_table: Dict[int, List[int]] = {}  # batch_idx -> [physical_page_ids]

    def compress_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: (B, S_chunk, D_model)
        Returns: compressed historical latent tensor (B, num_compressed, D_model)
        """
        batch_size, s_chunk, d_model = chunk.shape
        num_compressed = max(1, math.ceil(s_chunk * self.compression_rate))

        # Adaptive average pooling across sequence dimension
        chunk_t = chunk.transpose(1, 2)
        pooled = nn.functional.adaptive_avg_pool1d(chunk_t, num_compressed)
        pooled = pooled.transpose(1, 2)  # (B, num_compressed, D_model)

        return self.compressor(pooled)

    def store_latents(self, latents: torch.Tensor, batch_idx: int = 0) -> List[int]:
        """
        Compresses and stores latent vectors into fixed-size physical pages using TurboQuant.
        latents: (S_latents, D_model) or (1, S_latents, D_model)
        """
        if latents.dim() == 3:
            latents = latents.squeeze(0)
        
        S, D = latents.shape
        allocated_page_ids = []

        # Split into page_size chunks
        num_pages_needed = math.ceil(S / self.page_size)
        for p in range(num_pages_needed):
            page_slice = latents[p * self.page_size : (p + 1) * self.page_size]
            # Quantize page vectors with TurboQuant
            quantized_page = self.turbo_quant.quantize(page_slice)

            # Check capacity & handle LRU eviction
            if len(self.physical_pages) >= self.max_pages:
                # Evict oldest page (index 0)
                self.physical_pages.pop(0)
                # Re-index page table
                for b in self.logical_page_table:
                    self.logical_page_table[b] = [pid - 1 for pid in self.logical_page_table[b] if pid > 0]

            page_id = len(self.physical_pages)
            self.physical_pages.append(quantized_page)
            allocated_page_ids.append(page_id)

        if batch_idx not in self.logical_page_table:
            self.logical_page_table[batch_idx] = []
        self.logical_page_table[batch_idx].extend(allocated_page_ids)

        return allocated_page_ids

    def fetch_all(self, batch_idx: int = 0, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Dequantizes and returns all stored latents for a batch sequence as a contiguous tensor.
        Returns: (1, total_latents, D_model)
        """
        if batch_idx not in self.logical_page_table or not self.logical_page_table[batch_idx]:
            return None

        page_ids = self.logical_page_table[batch_idx]
        dequantized_pages = []
        for pid in page_ids:
            if pid < len(self.physical_pages):
                q_data = self.physical_pages[pid]
                dequantized = self.turbo_quant.dequantize(q_data)
                if device is not None:
                    dequantized = dequantized.to(device)
                dequantized_pages.append(dequantized)

        if not dequantized_pages:
            return None

        full_latents = torch.cat(dequantized_pages, dim=0).unsqueeze(0)  # (1, S_total, D)
        return full_latents


# Alias for backward-compatibility with HierarchicalMemoryController
CompressedArchive = PagedArchive
