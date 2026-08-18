"""
Multi-Modal Encoders and Dynamic Output Decoders.
Maps heterogeneous inputs into unified latent dimension h_in, and decodes representations into diverse modalities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TextEncoder(nn.Module):
    """Encodes text token sequences: h_text = E_token(x) + P_text."""
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S) integer token ids -> (B, S, D_model)"""
        batch_size, seq_len = x.shape
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.token_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos)


class VisionEncoder(nn.Module):
    """Encodes 2D image inputs: h_vision = Flatten(Conv2D(X)) + P_vision."""
    def __init__(self, in_channels: int = 3, d_model: int = 64, patch_size: int = 4, max_patches: int = 256):
        super().__init__()
        self.d_model = d_model
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Embedding(max_patches, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, num_patches, D_model)"""
        # Conv2D -> (B, D_model, H', W')
        feat = self.conv(x)
        batch_size, d_model, h_p, w_p = feat.shape
        num_patches = h_p * w_p
        
        # Flatten spatial dimensions -> (B, num_patches, D_model)
        flat = feat.flatten(2).transpose(1, 2)
        pos = torch.arange(0, num_patches, device=x.device).unsqueeze(0).expand(batch_size, num_patches)
        return flat + self.pos_emb(pos)


class AudioEncoder(nn.Module):
    """Encodes 1D audio or spectrogram features: h_audio = Proj(X_audio) + P_audio."""
    def __init__(self, in_dim: int = 80, d_model: int = 64, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, in_dim) -> (B, S, D_model)"""
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.proj(x) + self.pos_emb(pos)


class VideoEncoder(nn.Module):
    """Encodes 3D spatiotemporal video: h_video = Flatten(Conv3D(X_video)) + P_3D."""
    def __init__(self, in_channels: int = 3, d_model: int = 64, kernel_size: Tuple[int, int, int] = (2, 4, 4), max_tubes: int = 256):
        super().__init__()
        self.d_model = d_model
        self.conv3d = nn.Conv3d(in_channels, d_model, kernel_size=kernel_size, stride=kernel_size)
        self.pos_emb = nn.Embedding(max_tubes, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, num_tubes, D_model)"""
        feat = self.conv3d(x)
        batch_size, d_model, t_p, h_p, w_p = feat.shape
        num_tubes = t_p * h_p * w_p
        flat = feat.flatten(2).transpose(1, 2)
        pos = torch.arange(0, num_tubes, device=x.device).unsqueeze(0).expand(batch_size, num_tubes)
        return flat + self.pos_emb(pos)


class AutoregressiveDecoderHead(nn.Module):
    """Decodes latent states into next-token logits: P(y_t | y_<t, x) = Softmax(W_vocab * h_t + b_vocab)."""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, S, D_model) -> (B, S, vocab_size)"""
        return self.proj(h)


class DiffusionDecoderHead(nn.Module):
    """Predicts continuous denoising vector: \\hat{\\epsilon} = f_theta(x_t, t, h'_t)."""
    def __init__(self, d_model: int, out_dim: int = 64, time_emb_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.net = nn.Sequential(
            nn.Linear(out_dim + d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, out_dim),
        )

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
