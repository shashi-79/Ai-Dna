"""
Tests for Hierarchical Long-Context Memory.
"""

import torch
from ai_dna.memory.working_memory import WorkingMemory
from ai_dna.memory.archive import CompressedArchive
from ai_dna.memory.retrieval import RetrievalLibrary
from ai_dna.memory.hierarchical import HierarchicalMemoryController


def test_working_memory_chunks():
    wm = WorkingMemory(d_model=32, num_heads=4, chunk_size=16)
    x = torch.randn(2, 48, 32)
    out = wm(x)

    assert out.shape == (2, 48, 32)


def test_compressed_archive():
    archive = CompressedArchive(d_model=32, compression_rate=0.25)
    chunk = torch.randn(2, 32, 32)
    latents = archive.compress_chunk(chunk)

    # 32 tokens * 0.25 = 8 compressed latents
    assert latents.shape == (2, 8, 32)


def test_hierarchical_memory_controller():
    hmc = HierarchicalMemoryController(d_model=32, num_heads=4)
    x = torch.randn(2, 64, 32)

    out, new_archive, metrics = hmc(x)
    assert out.shape == (2, 64, 32)
    assert new_archive.shape[0] == 2
    assert "c_compute" in metrics
    assert metrics["c_compute"] > 0.0
