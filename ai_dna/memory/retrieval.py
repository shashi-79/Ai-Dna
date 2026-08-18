"""
Vector Retrieval Library for Long-Context Latents.
Retrieves top-N_retrieval most relevant historical memory latents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Any

class ExternalVectorDatabase:
    """
    In-memory external factual knowledge database.
    Stores dense vectors and their associated contextual payload.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.index = None
        self.payloads = []

    def add_documents(self, vectors: torch.Tensor, payloads: List[Any]):
        """vectors: (N, d_model)"""
        assert vectors.shape[-1] == self.d_model
        if self.index is None:
            self.index = vectors.detach().cpu()
        else:
            self.index = torch.cat([self.index, vectors.detach().cpu()], dim=0)
        self.payloads.extend(payloads)

    def search(self, query: torch.Tensor, top_k: int = 5) -> Tuple[Optional[torch.Tensor], List[List[Any]]]:
        """
        query: (B, S, D_model)
        Returns: 
           retrieved_vectors: (B, S, top_k, D_model)
           retrieved_payloads: List[List[List[Any]]] mapping payloads
        """
        if self.index is None or self.index.shape[0] == 0:
            return None, []
            
        B, S, D = query.shape
        flat_query = query.reshape(B * S, D).cpu()  # (B*S, D)
        
        # Cosine similarity: (B*S, N_docs)
        sim = F.cosine_similarity(flat_query.unsqueeze(1), self.index.unsqueeze(0), dim=-1)
        
        k = min(top_k, self.index.shape[0])
        top_scores, top_indices = torch.topk(sim, k, dim=-1)  # (B*S, k)
        
        # Fetch vectors: (B*S, k, D)
        retrieved = self.index[top_indices].to(query.device)
        retrieved = retrieved.view(B, S, k, D)
        
        # Map payloads
        payloads = []
        flat_indices = top_indices.tolist()
        for i in range(B):
            b_payloads = []
            for j in range(S):
                idx = i * S + j
                b_payloads.append([self.payloads[doc_idx] for doc_idx in flat_indices[idx]])
            payloads.append(b_payloads)
            
        return retrieved, payloads


class RetrievalLibrary(nn.Module):
    """
    Key-Value Vector Retrieval Library for historical latents and external knowledge.
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
            k_int = min(k, archive_latents.shape[1])
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
