"""
Phenotype Neural Network Architecture.
Combines Omni-Modal embeddings, Transformer blocks, and Sparse Low-Rank MoE layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from ..dna.structure import DNAArchitecture, DNARouting, DNAMemory
from ..routing.router import GenerativeSparseRouter
from ..memory.hierarchical import HierarchicalMemoryController
from .modules import (
    TextEncoder,
    VisionEncoder,
    AudioEncoder,
    VideoEncoder,
    AutoregressiveDecoderHead,
    DiffusionDecoderHead,
    ClassificationHead,
)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention block with FlashAttention / SDPA GPU acceleration."""
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Utilize FlashAttention / Scaled Dot Product Attention on CUDA GPU
        try:
            attn_mask = mask.bool() if mask is not None and mask.dtype != torch.bool else mask
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        except Exception:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)


class SparseMoEExpert(nn.Module):
    """Individual MoE Feed-Forward Expert."""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_hidden)
        self.down_proj = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


class SparseMoELayer(nn.Module):
    """
    Sparse MoE Layer with Low-Rank Gate and Straight-Through discrete selection.
    Integrates hardware-aware token grouping and Triton GPU acceleration.
    """
    def __init__(self, d_model: int, num_experts: int, d_expert_hidden: int, router: GenerativeSparseRouter, use_hardware_executor: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.router = router
        self.experts = nn.ModuleList([
            SparseMoEExpert(d_model, d_expert_hidden) for _ in range(num_experts)
        ])
        self.use_hardware_executor = use_hardware_executor
        from ..inference.sparse_executor import SparseHardwareExecutor
        self.hw_executor = SparseHardwareExecutor(num_experts=num_experts)

    def forward(
        self,
        h: torch.Tensor,
        modality_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: (B, S, D_model)
        Returns:
            out: (B, S, D_model)
            aux_loss: Expert balancing loss
        """
        p_gate, m_gate, aux_loss = self.router(h, modality_emb=modality_emb)
        # m_gate: (B, S, E_max)
        if self.use_hardware_executor and (h.is_cuda or not self.training):
            out = self.hw_executor.execute_sparse_moe(h, p_gate, m_gate, self.experts)
        else:
            out = torch.zeros_like(h)
            for e in range(self.num_experts):
                # Expert mask for this expert: (B, S, 1)
                e_mask = m_gate[:, :, e:e+1]
                e_prob = p_gate[:, :, e:e+1]
                if (e_mask > 0).any():
                    expert_out = self.experts[e](h)
                    out = out + (expert_out * e_prob * e_mask)

        return out, aux_loss


class PhenotypeTransformerBlock(nn.Module):
    """Full Transformer block with Self-Attention and Sparse MoE layer."""
    def __init__(self, d_model: int, num_heads: int, num_experts: int, d_expert_hidden: int, router: GenerativeSparseRouter):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = SparseMoELayer(d_model, num_experts, d_expert_hidden, router)

    def forward(
        self,
        x: torch.Tensor,
        modality_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x), mask=mask)
        moe_out, aux_loss = self.moe(self.ln2(x), modality_emb=modality_emb)
        x = x + moe_out
        return x, aux_loss


class PhenotypeNeuralNetwork(nn.Module):
    """
    Complete Executable Phenotype Neural Network $W = G(D)$.
    Supports Omni-Modal inputs, Hierarchical Memory, Dynamic Sparse MoE, and Multi-Mode Decoders.
    """
    def __init__(
        self,
        arch: Any,
        dna_routing: Optional[Any] = None,
        dna_memory: Optional[Any] = None,
    ):
        super().__init__()
        from ..dna.structure import Genotype
        if isinstance(arch, Genotype):
            genotype = arch
            arch = genotype.dna_architecture
            dna_routing = genotype.dna_routing
            dna_memory = genotype.dna_memory
        else:
            from ..dna.structure import DNARouting, DNAMemory
            dna_routing = dna_routing or DNARouting()
            dna_memory = dna_memory or DNAMemory()

        self.d_model = arch.d_model
        self.num_layers = arch.num_layers
        self.num_experts = arch.num_experts

        # 1. Modality Encoders
        self.text_encoder = TextEncoder(arch.vocab_size, self.d_model)
        self.vision_encoder = VisionEncoder(in_channels=3, d_model=self.d_model)
        self.audio_encoder = AudioEncoder(in_dim=80, d_model=self.d_model)
        self.video_encoder = VideoEncoder(in_channels=3, d_model=self.d_model)

        # 2. Hierarchical Memory Controller
        self.memory = HierarchicalMemoryController(
            d_model=self.d_model,
            num_heads=arch.num_heads,
            dna_memory=dna_memory,
        )

        # 3. Router & Transformer MoE Blocks
        self.shared_router = GenerativeSparseRouter(
            d_model=self.d_model,
            num_experts=arch.num_experts,
            dna_routing=dna_routing,
        )
        self.blocks = nn.ModuleList([
            PhenotypeTransformerBlock(
                d_model=self.d_model,
                num_heads=arch.num_heads,
                num_experts=arch.num_experts,
                d_expert_hidden=arch.d_expert_hidden,
                router=self.shared_router,
            )
            for _ in range(arch.num_layers)
        ])
        self.ln_final = nn.LayerNorm(self.d_model)

        self.tabular_proj = nn.Linear(16, self.d_model)

        # 4. Multi-Mode Output Decoders
        self.ar_head = AutoregressiveDecoderHead(self.d_model, arch.vocab_size)
        self.diff_head = DiffusionDecoderHead(self.d_model, out_dim=self.d_model)
        self.cls_head = ClassificationHead(self.d_model, num_classes=10)

    def encode_input(self, x: torch.Tensor, modality: str = "text") -> torch.Tensor:
        if modality in ["text", "code", "bio"]:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.text_encoder(x)
        elif modality == "vision":
            return self.vision_encoder(x)
        elif modality == "audio":
            return self.audio_encoder(x)
        elif modality == "video":
            return self.video_encoder(x)
        elif modality in ["tabular", "scientific"] or x.dim() == 2:
            if x.dim() == 2:
                if x.shape[-1] != self.tabular_proj.in_features:
                    self.tabular_proj = nn.Linear(x.shape[-1], self.d_model).to(x.device)
                return self.tabular_proj(x).unsqueeze(1)
            elif x.dim() == 3:
                return x
            return x.unsqueeze(1)
        else:
            if x.dim() == 2:
                return x.unsqueeze(1)
            return x

    def forward(
        self,
        inputs: torch.Tensor,
        modality: str = "text",
        cached_archive: Optional[torch.Tensor] = None,
        external_db: Optional[Any] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Forward pass through intake, memory, MoE backbone, and returns representations.
        """
        # 1. Intake
        h = self.encode_input(inputs, modality=modality)

        # 2. Hierarchical Memory
        h, new_archive, mem_metrics = self.memory(h, cached_archive=cached_archive, external_db=external_db, is_causal=is_causal)

        # 3. Causal Attention Masking (if autoregressive)
        seq_len = h.shape[1]
        mask = None
        if is_causal:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=h.device)).unsqueeze(0).unsqueeze(0)

        # 4. Backbone Blocks
        total_aux_loss = torch.tensor(0.0, device=h.device)
        for block in self.blocks:
            h, aux_loss = block(h, mask=mask)
            total_aux_loss = total_aux_loss + aux_loss

        h = self.ln_final(h)
        return h, total_aux_loss, new_archive, mem_metrics
