"""
Hierarchical Graph Retrieval (GraphRAG) and Retrieval Library.
Implements community-structured hierarchical graph retrieval with TurboQuant compressed indexing.
Based on Edge et al., 2024 (GraphRAG) and Zandieh et al., 2025 (TurboQuant).
Implements idea.md Section 9.4.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
from .turboquant import TurboQuant


class GraphRAG:
    """
    Hierarchical Graph Retrieval Engine (Section 9.4).
    Constructs semantic similarity graphs over documents, detects communities via spectral clustering,
    computes community summary centroids, and performs two-level hierarchical retrieval.
    All indices are compressed using TurboQuant.
    """
    def __init__(
        self,
        d_model: int,
        edge_threshold: float = 0.5,
        num_communities: int = 4,
        b_quant: int = 3,
    ):
        self.d_model = d_model
        self.edge_threshold = edge_threshold
        self.num_communities = num_communities
        self.b_quant = b_quant

        self.turbo_quant = TurboQuant(d_model=d_model, b_quant=b_quant)

        # Raw & Quantized Document storage
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.quantized_docs: Optional[Dict[str, Any]] = None
        self.payloads: List[Any] = []

        # Graph structure
        self.adjacency_matrix: Optional[torch.Tensor] = None
        self.community_assignments: List[int] = []  # doc_idx -> community_id
        self.community_clusters: Dict[int, List[int]] = {}  # community_id -> [doc_indices]
        self.community_summaries: Optional[torch.Tensor] = None  # (C, D)
        self.quantized_summaries: Optional[Dict[str, Any]] = None

    def add_documents(self, embeddings: torch.Tensor, payloads: List[Any]):
        """
        embeddings: (N, D_model) tensor of document vectors.
        payloads: List of N metadata / text objects.
        """
        assert embeddings.shape[-1] == self.d_model, f"Dimension mismatch: expected {self.d_model}, got {embeddings.shape[-1]}"
        embeddings = embeddings.detach().cpu().float()

        if self.doc_embeddings is None:
            self.doc_embeddings = embeddings
        else:
            self.doc_embeddings = torch.cat([self.doc_embeddings, embeddings], dim=0)

        self.payloads.extend(payloads)
        self._build_graph_and_communities()

    def _build_graph_and_communities(self):
        """
        Builds similarity graph, performs spectral clustering to form communities,
        and computes community summary embeddings.
        """
        if self.doc_embeddings is None or self.doc_embeddings.shape[0] == 0:
            return

        N = self.doc_embeddings.shape[0]
        # Quantize all docs with TurboQuant
        self.quantized_docs = self.turbo_quant.quantize(self.doc_embeddings)

        # 1. Similarity Graph Construction: (N, N)
        normed = F.normalize(self.doc_embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(normed, normed.t())  # (N, N) cosine similarities
        
        # Binary adjacency matrix with threshold
        adj = (sim_matrix > self.edge_threshold).float()
        adj.fill_diagonal_(0.0)
        self.adjacency_matrix = adj

        # 2. Community Detection via Spectral Clustering / Graph Laplacian
        num_c = min(self.num_communities, max(1, N))
        if N <= num_c:
            assignments = list(range(N))
        else:
            # Degree matrix D and Laplacian L = D - A
            deg = adj.sum(dim=1)
            deg_mat = torch.diag(deg)
            L = deg_mat - adj

            # Normalized Laplacian: L_sym = D^{-1/2} L D^{-1/2}
            deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-5), -0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            L_sym = torch.matmul(torch.matmul(D_inv_sqrt, L), D_inv_sqrt)

            try:
                # Eigen decomposition of Laplacian
                eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
                # Take smallest k non-trivial eigenvectors
                H = eigenvectors[:, :num_c]
                # K-means clustering on eigen-features
                assignments = self._kmeans(H, k=num_c)
            except Exception:
                # Fallback: chunk-based partitioning
                assignments = [i % num_c for i in range(N)]

        self.community_assignments = assignments
        self.community_clusters = {}
        for doc_idx, c_id in enumerate(assignments):
            if c_id not in self.community_clusters:
                self.community_clusters[c_id] = []
            self.community_clusters[c_id].append(doc_idx)

        # 3. Community Summarization (Mean-pooled centroids)
        summaries = []
        for c_id in sorted(self.community_clusters.keys()):
            indices = self.community_clusters[c_id]
            cluster_vecs = self.doc_embeddings[indices]
            summary_vec = cluster_vecs.mean(dim=0)
            summaries.append(summary_vec)

        self.community_summaries = torch.stack(summaries, dim=0)  # (C, D)
        self.quantized_summaries = self.turbo_quant.quantize(self.community_summaries)

    def _kmeans(self, data: torch.Tensor, k: int, num_iters: int = 10) -> List[int]:
        """Simple deterministic k-means for community clustering."""
        N = data.shape[0]
        # Initialize centroids from distinct evenly spaced samples
        step = max(1, N // k)
        centroids = data[::step][:k].clone()
        if centroids.shape[0] < k:
            centroids = data[:k].clone()

        assignments = torch.zeros(N, dtype=torch.long)
        for _ in range(num_iters):
            # Compute distances: (N, k)
            dists = torch.cdist(data, centroids)
            assignments = torch.argmin(dists, dim=-1)
            # Update centroids
            for c in range(k):
                mask = (assignments == c)
                if mask.any():
                    centroids[c] = data[mask].mean(dim=0)

        return assignments.tolist()

    def search(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        top_k_communities: int = 2,
    ) -> Tuple[Optional[torch.Tensor], List[List[Any]]]:
        """
        Two-level Hierarchical Search:
        1. Find top_k_communities matching the query.
        2. Retrieve top_k leaf documents within the matched communities.
        query: (B, S, D_model)
        Returns:
            retrieved_vectors: (B, S, top_k, D_model)
            retrieved_payloads: List[List[List[Any]]]
        """
        if self.doc_embeddings is None or self.doc_embeddings.shape[0] == 0:
            return None, []

        B, S, D = query.shape
        flat_query = query.reshape(B * S, D).cpu().float()

        # Dequantize community summaries
        summaries = self.turbo_quant.dequantize(self.quantized_summaries)  # (C, D)
        norm_summaries = F.normalize(summaries, p=2, dim=-1)
        norm_query = F.normalize(flat_query, p=2, dim=-1)

        # 1. Match Top-K Communities: (B*S, C)
        comm_sim = torch.matmul(norm_query, norm_summaries.t())
        k_c = min(top_k_communities, summaries.shape[0])
        _, top_comm_indices = torch.topk(comm_sim, k_c, dim=-1)  # (B*S, k_c)

        # 2. Leaf-level Retrieval from matched communities
        # Dequantize document vectors
        docs = self.turbo_quant.dequantize(self.quantized_docs)  # (N, D)
        norm_docs = F.normalize(docs, p=2, dim=-1)

        retrieved_vecs_list = []
        retrieved_payloads_list = []

        for row_idx in range(B * S):
            matched_comms = top_comm_indices[row_idx].tolist()
            candidate_doc_indices = []
            for c_id in matched_comms:
                candidate_doc_indices.extend(self.community_clusters.get(c_id, []))

            if not candidate_doc_indices:
                candidate_doc_indices = list(range(docs.shape[0]))

            candidate_doc_indices = list(set(candidate_doc_indices))
            cand_docs = norm_docs[candidate_doc_indices]  # (N_cand, D)
            q_row = norm_query[row_idx:row_idx+1]  # (1, D)

            # Cosine similarity within candidates
            cand_sim = torch.matmul(q_row, cand_docs.t()).squeeze(0)  # (N_cand,)
            k_doc = min(top_k, len(candidate_doc_indices))
            _, top_cand_k = torch.topk(cand_sim, k_doc, dim=-1)

            selected_doc_indices = [candidate_doc_indices[idx] for idx in top_cand_k.tolist()]
            retrieved_vecs_list.append(docs[selected_doc_indices])
            retrieved_payloads_list.append([self.payloads[idx] for idx in selected_doc_indices])

        # Pad retrieved vectors to uniform top_k if needed
        max_k = max(v.shape[0] for v in retrieved_vecs_list) if retrieved_vecs_list else top_k
        padded_vecs = []
        for v in retrieved_vecs_list:
            if v.shape[0] < max_k:
                pad = torch.zeros(max_k - v.shape[0], D)
                v = torch.cat([v, pad], dim=0)
            padded_vecs.append(v)

        retrieved_tensor = torch.stack(padded_vecs, dim=0).to(query.device)  # (B*S, max_k, D)
        retrieved_tensor = retrieved_tensor.view(B, S, max_k, D)

        # Structure payloads into (B, S, k)
        payloads_nested = []
        for i in range(B):
            b_payloads = []
            for j in range(S):
                b_payloads.append(retrieved_payloads_list[i * S + j])
            payloads_nested.append(b_payloads)

        return retrieved_tensor, payloads_nested


class ExternalVectorDatabase:
    """
    In-memory external factual knowledge database powered by GraphRAG.
    """
    def __init__(self, d_model: int, b_quant: int = 3):
        self.d_model = d_model
        self.graph_rag = GraphRAG(d_model=d_model, b_quant=b_quant)

    def add_documents(self, vectors: torch.Tensor, payloads: List[Any]):
        self.graph_rag.add_documents(vectors, payloads)

    def search(self, query: torch.Tensor, top_k: int = 5) -> Tuple[Optional[torch.Tensor], List[List[Any]]]:
        return self.graph_rag.search(query, top_k=top_k)


class RetrievalLibrary(nn.Module):
    """
    Key-Value Vector Retrieval Library for historical latents and external knowledge.
    Uses multi-head attention over internal archive latents and GraphRAG over external databases.
    """
    def __init__(self, d_model: int, num_retrieval: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_retrieval = num_retrieval

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)

    def retrieve(
        self,
        current_query: torch.Tensor,
        archive_latents: Optional[torch.Tensor],
        external_db: Optional[ExternalVectorDatabase] = None,
        top_k: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        current_query: (B, S_curr, D_model)
        archive_latents: (B, S_archive, D_model)
        Returns: Retrieved contextual latents of shape (B, S_curr, D_model)
        """
        k = top_k if top_k is not None else self.num_retrieval
        q = self.query_proj(current_query)  # (B, S_curr, D_model)

        internal_context = None
        if archive_latents is not None and archive_latents.shape[1] > 0:
            keys = self.key_proj(archive_latents)  # (B, S_archive, D_model)
            vals = self.val_proj(archive_latents)  # (B, S_archive, D_model)

            scores = torch.bmm(q, keys.transpose(1, 2)) / (self.d_model ** 0.5)
            attn = F.softmax(scores, dim=-1)
            internal_context = torch.bmm(attn, vals)

        external_context = None
        if external_db is not None:
            retrieved_vecs, _ = external_db.search(current_query, top_k=k)
            if retrieved_vecs is not None:
                # Average retrieved external vectors
                external_context = retrieved_vecs.mean(dim=2)

        if internal_context is not None and external_context is not None:
            return internal_context + external_context
        elif internal_context is not None:
            return internal_context
        elif external_context is not None:
            return external_context
        return None
