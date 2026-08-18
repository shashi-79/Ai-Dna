"""
Unified Hierarchical Long-Context Memory Controller.
Combines Working Memory with RoPE and TurboQuant KV cache, PagedArchive with PagedAttention,
and GraphRAG-powered Retrieval Library.
Computes memory compute cost: C_compute = alpha * T_seq + beta * M_peak + delta * M_total.
"""

import time
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from ..dna.structure import DNAMemory
from .working_memory import WorkingMemory
from .archive import PagedArchive, CompressedArchive
from .retrieval import RetrievalLibrary, ExternalVectorDatabase


class HierarchicalMemoryController(nn.Module):
    """
    Manages long-context sequence processing by bounding active attention to C_chunk,
    compressing historical contexts into latent archives at c_rate in fixed pages, and retrieving N_retrieval.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dna_memory: Optional[DNAMemory] = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dna_memory = dna_memory or DNAMemory()

        self.working_mem = WorkingMemory(
            d_model=d_model,
            num_heads=num_heads,
            chunk_size=self.dna_memory.chunk_size,
            kv_quant_bits=getattr(self.dna_memory, "kv_quant_bits", 3),
        )
        self.archive = PagedArchive(
            d_model=d_model,
            compression_rate=self.dna_memory.compression_rate,
            page_size=getattr(self.dna_memory, "page_size", 16),
            max_pages=getattr(self.dna_memory, "max_pages", 1024),
            kv_quant_bits=getattr(self.dna_memory, "kv_quant_bits", 3),
        )
        self.retrieval = RetrievalLibrary(
            d_model=d_model,
            num_retrieval=self.dna_memory.num_retrieval,
        )

        self.fusion_gate = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        x: torch.Tensor,
        cached_archive: Optional[torch.Tensor] = None,
        external_db: Optional[Any] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        x: Input token sequence of shape (B, S, D_model)
        cached_archive: Existing compressed archive tensor (B, S_archive, D_model)
        Returns:
            out: Contextualized hidden representation (B, S, D_model)
            new_archive: Updated compressed archive tensor
            metrics: Compute cost metrics (C_compute, peak_mem, total_latents)
        """
        t0 = time.perf_counter()
        batch_size, seq_len, d_model = x.shape

        # 1. Process local working memory chunks with RoPE and TurboQuant
        local_out = self.working_mem(x, is_causal=is_causal)

        # 2. Retrieve relevant historical context from existing archive or GraphRAG external DB
        retrieved = self.retrieval.retrieve(local_out, cached_archive, external_db=external_db)

        if retrieved is not None:
            # Fuse local and retrieved context
            combined = torch.cat([local_out, retrieved], dim=-1)
            out = self.fusion_gate(combined)
        else:
            out = local_out

        # 3. Compress current sequence into new archive latents
        new_latents = self.archive.compress_chunk(x)
        if cached_archive is not None:
            updated_archive = torch.cat([cached_archive, new_latents], dim=1)
        else:
            updated_archive = new_latents

        # Store in PagedArchive
        for b in range(batch_size):
            self.archive.store_latents(new_latents[b], batch_idx=b)

        t1 = time.perf_counter()
        t_seq = (t1 - t0) * 1000.0  # ms

        # Compute cost tracking: C_compute = alpha * T_seq + beta * M_peak + delta * M_total
        active_window = min(seq_len, self.dna_memory.chunk_size) + self.dna_memory.num_retrieval
        m_peak = float(batch_size * active_window * d_model * 4) / 1024.0  # KB
        # Factoring in TurboQuant compression for total memory
        quant_bits = getattr(self.dna_memory, "kv_quant_bits", 3)
        m_total = float(batch_size * updated_archive.shape[1] * d_model * (quant_bits / 8.0)) / 1024.0  # KB

        c_compute = (
            self.dna_memory.cost_alpha * t_seq
            + self.dna_memory.cost_beta * m_peak
            + self.dna_memory.cost_delta * m_total
        )

        metrics = {
            "t_seq_ms": t_seq,
            "peak_mem_kb": m_peak,
            "total_mem_kb": m_total,
            "c_compute": c_compute,
            "archive_length": updated_archive.shape[1],
        }

        return out, updated_archive, metrics

    @classmethod
    def optimize_memory_policy(
        cls,
        d_model: int,
        num_heads: int,
        sample_input: torch.Tensor,
        perf_fn: Optional[Any] = None,
        p_min: float = 0.9,
        chunk_sizes: Optional[list] = None,
        compression_rates: Optional[list] = None,
        num_retrievals: Optional[list] = None,
    ) -> Tuple[DNAMemory, Dict[str, Any]]:
        """
        Memory Policy Optimization (Section 9.5):
        D_memory* = argmin C_compute(D_memory) subject to Perf >= P_min
        """
        chunk_sizes = chunk_sizes or [16, 32, 64, 128]
        compression_rates = compression_rates or [0.1, 0.2, 0.25, 0.35, 0.5]
        num_retrievals = num_retrievals or [4, 8, 12, 16]

        best_cost = float("inf")
        best_policy = DNAMemory()
        all_results = []

        for c_chunk in chunk_sizes:
            for c_rate in compression_rates:
                for n_ret in num_retrievals:
                    dna_mem = DNAMemory(
                        chunk_size=c_chunk,
                        compression_rate=c_rate,
                        num_retrieval=n_ret,
                    )
                    try:
                        controller = cls(d_model=d_model, num_heads=num_heads, dna_memory=dna_mem)
                        with torch.no_grad():
                            _, _, metrics = controller(sample_input)

                        cost = metrics["c_compute"]

                        perf = 1.0
                        if perf_fn is not None:
                            perf = perf_fn(controller)
                        meets_constraint = perf >= p_min

                        result = {
                            "chunk_size": c_chunk,
                            "compression_rate": c_rate,
                            "num_retrieval": n_ret,
                            "c_compute": cost,
                            "performance": perf,
                            "meets_constraint": meets_constraint,
                        }
                        all_results.append(result)

                        if meets_constraint and cost < best_cost:
                            best_cost = cost
                            best_policy = dna_mem

                    except Exception:
                        continue

        search_results = {
            "best_cost": best_cost,
            "num_configs_evaluated": len(all_results),
            "num_feasible": sum(1 for r in all_results if r["meets_constraint"]),
            "all_results": all_results,
        }

        return best_policy, search_results
