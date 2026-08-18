"""
Tests for Hierarchical Long-Context Memory, TurboQuant, PagedArchive, and GraphRAG.
"""

import torch
from ai_dna.memory.turboquant import TurboQuant
from ai_dna.memory.working_memory import WorkingMemory
from ai_dna.memory.archive import PagedArchive, CompressedArchive
from ai_dna.memory.retrieval import GraphRAG, ExternalVectorDatabase, RetrievalLibrary
from ai_dna.memory.hierarchical import HierarchicalMemoryController


def test_turboquant_quantize_dequantize():
    tq = TurboQuant(d_model=32, b_quant=3)
    x = torch.randn(4, 16, 32)
    q_data = tq.quantize(x)

    assert "idx" in q_data
    assert "norm" in q_data
    assert "gamma" in q_data
    assert "qjl" in q_data
    assert q_data["idx"].shape == (4, 16, 32)

    x_rec = tq.dequantize(q_data)
    assert x_rec.shape == x.shape
    # Check MSE reconstruction distortion is bounded
    mse = torch.mean((x - x_rec) ** 2)
    assert mse.item() < 1.0


def test_working_memory_chunks():
    wm = WorkingMemory(d_model=32, num_heads=4, chunk_size=16, kv_quant_bits=3)
    x = torch.randn(2, 48, 32)
    out = wm(x, quantize_kv=True)

    assert out.shape == (2, 48, 32)


def test_paged_compressed_archive():
    archive = PagedArchive(d_model=32, compression_rate=0.25, page_size=4)
    chunk = torch.randn(2, 32, 32)
    latents = archive.compress_chunk(chunk)

    # 32 tokens * 0.25 = 8 compressed latents
    assert latents.shape == (2, 8, 32)

    # Test Paged store and fetch
    p_ids = archive.store_latents(latents[0], batch_idx=0)
    assert len(p_ids) > 0
    fetched = archive.fetch_all(batch_idx=0)
    assert fetched is not None
    assert fetched.shape[-1] == 32


def test_graph_rag_retrieval():
    rag = GraphRAG(d_model=32, edge_threshold=0.3, num_communities=2, b_quant=3)
    docs = torch.randn(10, 32)
    payloads = [f"doc_{i}" for i in range(10)]
    rag.add_documents(docs, payloads)

    query = torch.randn(1, 2, 32)
    retrieved_vecs, retrieved_payloads = rag.search(query, top_k=3, top_k_communities=2)

    assert retrieved_vecs is not None
    assert retrieved_vecs.shape == (1, 2, 3, 32)
    assert len(retrieved_payloads[0][0]) == 3


def test_hierarchical_memory_controller():
    hmc = HierarchicalMemoryController(d_model=32, num_heads=4)
    x = torch.randn(2, 64, 32)

    out, new_archive, metrics = hmc(x)
    assert out.shape == (2, 64, 32)
    assert new_archive.shape[0] == 2
    assert "c_compute" in metrics
    assert metrics["c_compute"] > 0.0
