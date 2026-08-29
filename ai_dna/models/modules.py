"""
Multi-Modal Encoders, Contrastive Alignment, and Dynamic Output Decoders.
Maps heterogeneous inputs into unified latent dimension h_in without static additive position embeddings
(positional information is handled dynamically via RoPE in attention layers).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class TextEncoder(nn.Module):
    """Encodes text token sequences: h_text = E_token(x) * sqrt(D_model). No additive positional embeddings."""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S) integer token ids -> (B, S, D_model)"""
        return self.token_emb(x) * math.sqrt(self.d_model)


class VisionEncoder(nn.Module):
    """
    CLIP-style Contrastive Patch Projection Vision Encoder.
    Divides image into non-overlapping patches, applies linear projection, prepends [CLS] token,
    and applies LayerNorm. Positional information is injected via 2D RoPE in attention.
    """
    def __init__(self, in_channels: int = 3, d_model: int = 64, patch_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, num_patches + 1, D_model)"""
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"Image dims ({H}, {W}) must be divisible by patch size {p}"
        
        # Reshape to patches: (B, C, H//p, p, W//p, p) -> (B, (H//p)*(W//p), C*p*p)
        h_p, w_p = H // p, W // p
        patches = x.view(B, C, h_p, p, w_p, p).permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, h_p * w_p, self.patch_dim)
        
        # Project patches
        projected = self.patch_proj(patches) # (B, N, D_model)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls_tokens, projected], dim=1)
        return self.ln(out)


class AudioEncoder(nn.Module):
    """
    Encodes 1D audio or spectrogram features: h_audio = LayerNorm(Proj(X_audio)).
    No static additive positional embeddings (1D RoPE applied in attention).
    """
    def __init__(self, in_dim: int = 80, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(in_dim, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, in_dim) -> (B, S, D_model)"""
        return self.ln(self.proj(x))


class VideoEncoder(nn.Module):
    """
    Temporal-Spatial Patch Projection Video Encoder.
    Projects spatiotemporal tubes into D_model, prepends [CLS] token, and applies LayerNorm.
    """
    def __init__(self, in_channels: int = 3, d_model: int = 64, temporal_patch_size: int = 2, spatial_patch_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.t_p = temporal_patch_size
        self.s_p = spatial_patch_size
        self.tube_dim = in_channels * temporal_patch_size * spatial_patch_size * spatial_patch_size
        
        self.tube_proj = nn.Linear(self.tube_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, num_tubes + 1, D_model)"""
        B, C, T, H, W = x.shape
        t_p, s_p = self.t_p, self.s_p
        assert T % t_p == 0 and H % s_p == 0 and W % s_p == 0, "Video dimensions must be divisible by tube sizes"
        
        n_t, n_h, n_w = T // t_p, H // s_p, W // s_p
        tubes = x.view(B, C, n_t, t_p, n_h, s_p, n_w, s_p).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        tubes = tubes.view(B, n_t * n_h * n_w, self.tube_dim)
        
        projected = self.tube_proj(tubes)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls_tokens, projected], dim=1)
        return self.ln(out)


class ContrastiveAlignmentHead(nn.Module):
    """
    CLIP/BLIP-style cross-modal contrastive alignment projector and loss (Section 6.5).
    Maps pooled modality representations into a common normalized embedding space.
    """
    def __init__(self, d_model: int, embed_dim: Optional[int] = None, temperature: float = 0.07):
        super().__init__()
        self.embed_dim = embed_dim if embed_dim is not None else d_model
        self.proj = nn.Linear(d_model, self.embed_dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, S, D_model)
        Returns: z: (B, embed_dim) normalized representation
        """
        pooled = h.mean(dim=1)  # Mean pool across sequence
        z = self.proj(pooled)
        return F.normalize(z, p=2, dim=-1)

    def compute_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE loss between batch of modality A and modality B.
        """
        logits = torch.matmul(z_a, z_b.t()) / torch.clamp(self.temperature, min=1e-4)
        labels = torch.arange(z_a.size(0), device=z_a.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a + loss_b)


class AutoregressiveDecoderHead(nn.Module):
    """Decodes latent states into next-token logits: P(y_t | y_<t, x) = Softmax(W_vocab * h_t + b_vocab)."""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, S, D_model) -> (B, S, vocab_size)"""
        return self.proj(h)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU):
    SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
    State-of-the-art bilinear gated activation (LLaMA 3, Mistral, PaLM, DeepSeek-V3).
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None, bias: bool = False):
        super().__init__()
        out_features = out_features or in_features
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DiffusionDecoderHead(nn.Module):
    """Predicts continuous denoising vector with SwiGLU: \hat{\epsilon} = f_theta(x_t, t, h'_t)."""
    def __init__(self, d_model: int, out_dim: int = 64, time_emb_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.net = SwiGLU(in_features=out_dim + d_model, hidden_features=d_model * 2, out_features=out_dim, bias=True)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, noisy_x: torch.Tensor, timesteps: torch.Tensor, h_context: torch.Tensor) -> torch.Tensor:
        """
        noisy_x: (B, S, out_dim)
        timesteps: (B,)
        h_context: (B, S, D_model)
        """
        t_emb = self.time_mlp(self.get_timestep_embedding(timesteps))  # (B, D_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, h_context.shape[1], -1)
        h_conditioned = h_context + t_emb
        combined = torch.cat([noisy_x, h_conditioned], dim=-1)
        return self.net(combined)


class ClassificationHead(nn.Module):
    """Standard classification head for discrete benchmark evaluations."""
    def __init__(self, d_model: int, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Pools sequence dimension and classifies: (B, S, D_model) -> (B, num_classes)."""
        pooled = h.mean(dim=1)
        return self.classifier(pooled)
