"""
Phenotype Neural Network Architecture.
Combines Omni-Modal Contrastive Intake, Multi-Head Latent Attention (MLA),
Rotary Position Embeddings (RoPE), and Top-K Sparsely-Gated MoE layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from ..dna.structure import DNAArchitecture, DNARouting, DNAMemory
from ..routing.router import GenerativeSparseRouter
from .mla import MultiHeadLatentAttention
from .modules import (
    TextEncoder,
    VisionEncoder,
    AudioEncoder,
    VideoEncoder,
    ContrastiveAlignmentHead,
    AutoregressiveDecoderHead,
    DiffusionDecoderHead,
    ClassificationHead,
    SwiGLU,
)


class SparseMoEExpert(nn.Module):
    """Individual MoE Feed-Forward Expert with SwiGLU Gated Activation."""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.swiglu = SwiGLU(in_features=d_model, hidden_features=d_hidden, out_features=d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class SparseMoELayer(nn.Module):
    """
    Sparse MoE Layer with Top-K Noisy Gating and DeepSeek-V3 Style Shared Base Expert.
    Integrates hardware-aware token grouping and Triton GPU acceleration.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_expert_hidden: int,
        router: GenerativeSparseRouter,
        use_hardware_executor: bool = True,
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.router = router
        self.use_shared_expert = use_shared_expert

        # DeepSeek-V3 Style Shared Base Expert for universal cross-modal representations
        self.shared_expert = SparseMoEExpert(d_model, d_expert_hidden) if use_shared_expert else None

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
                e_mask = m_gate[:, :, e:e+1]
                e_prob = p_gate[:, :, e:e+1]
                if (e_mask > 0).any():
                    expert_out = self.experts[e](h)
                    out = out + (expert_out * e_prob * e_mask)

        # DeepSeek-V3 Style: Add Shared Base Expert output to routed MoE output
        if self.shared_expert is not None:
            out = out + self.shared_expert(h)

        return out, aux_loss


class PhenotypeTransformerBlock(nn.Module):
    """Transformer block with Multi-Head Latent Attention (MLA) and Top-K Sparse MoE."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv_latent: int,
        num_experts: int,
        d_expert_hidden: int,
        router: GenerativeSparseRouter,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv_latent=d_kv_latent,
            rope_base=rope_base,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = SparseMoELayer(d_model, num_experts, d_expert_hidden, router)

    def forward(
        self,
        x: torch.Tensor,
        modality_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x), mask=mask, is_causal=is_causal)
        moe_out, aux_loss = self.moe(self.ln2(x), modality_emb=modality_emb)
        x = x + moe_out
        return x, aux_loss


class PhenotypeNeuralNetwork(nn.Module):
    """
    Complete Executable Phenotype Neural Network $W = G(D)$.
    Supports Omni-Modal inputs, Hierarchical Memory with TurboQuant, MLA Attention with RoPE,
    Dynamic Top-K Sparse MoE, and Multi-Mode Decoders.
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
        self.kv_latent_dim = getattr(arch, "kv_latent_dim", max(8, arch.d_model // 4))
        self.rope_theta = getattr(arch, "rope_theta", 10000.0)

        # 1. Modality Encoders (Contrastive patch projection, no static additive embeddings)
        self.text_encoder = TextEncoder(arch.vocab_size, self.d_model)
        self.vision_encoder = VisionEncoder(in_channels=3, d_model=self.d_model, patch_size=4)
        self.audio_encoder = AudioEncoder(in_dim=80, d_model=self.d_model)
        self.video_encoder = VideoEncoder(in_channels=3, d_model=self.d_model, temporal_patch_size=2, spatial_patch_size=4)
        self.contrastive_head = ContrastiveAlignmentHead(d_model=self.d_model, embed_dim=self.d_model)

        # 2. Hierarchical Memory Controller
        from ..memory.hierarchical import HierarchicalMemoryController
        self.memory = HierarchicalMemoryController(
            d_model=self.d_model,
            num_heads=arch.num_heads,
            dna_memory=dna_memory,
        )

        # 3. Router & Transformer MoE Blocks (MLA + TopK)
        self.shared_router = GenerativeSparseRouter(
            d_model=self.d_model,
            num_experts=arch.num_experts,
            dna_routing=dna_routing,
        )
        self.blocks = nn.ModuleList([
            PhenotypeTransformerBlock(
                d_model=self.d_model,
                num_heads=arch.num_heads,
                d_kv_latent=self.kv_latent_dim,
                num_experts=arch.num_experts,
                d_expert_hidden=arch.d_expert_hidden,
                router=self.shared_router,
                rope_base=self.rope_theta,
            )
            for _ in range(arch.num_layers)
        ])
        self.ln_final = nn.LayerNorm(self.d_model)

        self.tabular_proj = nn.Linear(16, self.d_model)

        # 4. Multi-Mode Output Decoders
        self.ar_head = AutoregressiveDecoderHead(self.d_model, arch.vocab_size)
        self.diff_head = DiffusionDecoderHead(self.d_model, out_dim=64)
        self.cls_head = ClassificationHead(self.d_model, num_classes=getattr(arch, "num_classes", 10))
        self.audio_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            SwiGLU(in_features=self.d_model, hidden_features=self.d_model * 2, out_features=80, bias=True),
        )
        self.contrastive_head = ContrastiveAlignmentHead(self.d_model, embed_dim=self.d_model)
        
        # 5. Modality Segment Embeddings (for cross-attention disambiguation in unified token streams)
        self.modality_embeddings = nn.Embedding(8, self.d_model)
        nn.init.normal_(self.modality_embeddings.weight, std=0.02)
        self.modality_map = {
            "text": 0, "vision": 1, "audio": 2, "video": 3,
            "tabular": 4, "scientific": 5, "code": 6, "bio": 7
        }

    def encode_input(self, x: torch.Tensor, modality: str = "text", add_modality_emb: bool = False) -> torch.Tensor:
        if modality in ["text", "code", "bio", "math"] or x.dtype in [torch.long, torch.int64, torch.int32]:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            h = self.text_encoder(x)
        elif modality == "vision":
            h = self.vision_encoder(x)
        elif modality == "audio":
            h = self.audio_encoder(x)
        elif modality == "video":
            h = self.video_encoder(x)
        elif modality in ["tabular", "scientific"] or x.dim() == 2:
            if x.dim() == 2:
                if x.shape[-1] != self.tabular_proj.in_features:
                    self.tabular_proj = nn.Linear(x.shape[-1], self.d_model).to(x.device)
                h = self.tabular_proj(x).unsqueeze(1)
            elif x.dim() == 3:
                h = x
            else:
                h = x.unsqueeze(1)
        else:
            if x.dim() == 2:
                h = x.unsqueeze(1)
            else:
                h = x

        if add_modality_emb:
            mod_id = torch.tensor(self.modality_map.get(modality, 0), device=h.device)
            h = h + self.modality_embeddings(mod_id)

        return h

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
            h, aux_loss = block(h, mask=mask, is_causal=is_causal)
            total_aux_loss = total_aux_loss + aux_loss

        h = self.ln_final(h)
        return h, total_aux_loss, new_archive, mem_metrics

    def forward_multimodal(
        self,
        text_inputs: Optional[torch.Tensor] = None,
        vision_inputs: Optional[torch.Tensor] = None,
        audio_inputs: Optional[torch.Tensor] = None,
        video_inputs: Optional[torch.Tensor] = None,
        cached_archive: Optional[torch.Tensor] = None,
        external_db: Optional[Any] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Processes a heterogeneous Unified Multimodal Token Stream [Text | Vision | Audio | Video]
        through the shared transformer substrate and genotypic MoE router (idea.md Section 6.7),
        differentiated by learned modality segment indicators.
        """
        token_streams = []
        if text_inputs is not None:
            token_streams.append(self.encode_input(text_inputs, modality="text", add_modality_emb=True))
        if vision_inputs is not None:
            token_streams.append(self.encode_input(vision_inputs, modality="vision", add_modality_emb=True))
        if audio_inputs is not None:
            token_streams.append(self.encode_input(audio_inputs, modality="audio", add_modality_emb=True))
        if video_inputs is not None:
            token_streams.append(self.encode_input(video_inputs, modality="video", add_modality_emb=True))

        if not token_streams:
            raise ValueError("forward_multimodal requires at least one sensory input.")

        # Concatenate along sequence dimension into Unified Multimodal Token Stream
        h_unified = torch.cat(token_streams, dim=1)

        # Pass through memory & transformer backbone
        h, new_archive, mem_metrics = self.memory(h_unified, cached_archive=cached_archive, external_db=external_db, is_causal=is_causal)

        seq_len = h.shape[1]
        mask = None
        if is_causal:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=h.device)).unsqueeze(0).unsqueeze(0)

        total_aux_loss = torch.tensor(0.0, device=h.device)
        for block in self.blocks:
            h, aux_loss = block(h, mask=mask, is_causal=is_causal)
            total_aux_loss = total_aux_loss + aux_loss

        h = self.ln_final(h)
        return h, total_aux_loss, new_archive, mem_metrics
