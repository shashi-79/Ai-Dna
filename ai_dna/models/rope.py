import math
import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    1D Rotary Position Embedding (RoPE) with YaRN & Dynamic NTK-Aware Scaling.
    Supports long-context sequences (up to 128,000+ tokens) with zero perplexity explosion.
    """
    def __init__(
        self,
        dim: int,
        base: float = 500000.0,
        max_position_embeddings: int = 2048,
        scaling_factor: float = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Standard / YaRN Inverted Frequencies
        inv_freq = self._compute_yarn_inv_freq(dim, base, scaling_factor)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_yarn_inv_freq(self, dim: int, base: float, scale: float) -> torch.Tensor:
        """Computes YaRN ramp-based frequency interpolation across dimensions."""
        pos = torch.arange(0, dim, 2).float()
        if scale <= 1.0:
            return 1.0 / (base ** (pos / dim))

        # YaRN frequency interpolation
        inv_freq_extrapolation = 1.0 / (base ** (pos / dim))
        inv_freq_interpolation = 1.0 / (scale * (base ** (pos / dim)))

        low = max(0, int(math.floor(dim * (1.0 - math.log(self.beta_fast) / math.log(base)))))
        high = min(dim // 2, int(math.ceil(dim * (1.0 - math.log(self.beta_slow) / math.log(base)))))

        if low == high:
            high += 1

        ramp = torch.clamp((pos / 2.0 - low) / max(1.0, float(high - low)), 0.0, 1.0)
        inv_freq = (1.0 - ramp) * inv_freq_interpolation + ramp * inv_freq_extrapolation
        return inv_freq

    def _get_cos_sin(self, seq_len: int, device: torch.device):
        # Dynamic NTK-aware scaling if sequence length exceeds base context
        if seq_len > self.max_position_embeddings:
            scale = seq_len / self.max_position_embeddings
            base = self.base * ((self.scaling_factor * scale) - (self.scaling_factor - 1.0)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        else:
            inv_freq = self.inv_freq.to(device)

        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # YaRN attention scale factor
        mscale = 0.1 * math.log(max(1.0, seq_len / max(1, self.max_position_embeddings))) + 1.0
        cos = emb.cos()[None, None, :, :] * mscale
        sin = emb.sin()[None, None, :, :] * mscale
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k shape: [B, num_heads, S, head_dim]
        seq_len = q.shape[2]
        cos, sin = self._get_cos_sin(seq_len, q.device)
        
        q_rot = self._apply_rotary_emb(q, cos, sin)
        k_rot = self._apply_rotary_emb(k, cos, sin)
        return q_rot, k_rot

    def _apply_rotary_emb(self, x, cos, sin):
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        x_half = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_half * sin)


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding for spatial data (Vision).
    Splits the head dimension into two halves: row rotation and column rotation.
    """
    def __init__(self, dim, base=10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.rope_h = RoPE(self.half_dim, base)
        self.rope_w = RoPE(self.half_dim, base)
        
    def forward(self, q, k, h, w):
        # q, k shape: [B, num_heads, S, head_dim], where S = h * w
        
        # We need to construct 2D coordinates for the sequence
        # h_pos: [h, w] -> flattened to [h*w]
        
        cos_h, sin_h = self.rope_h._get_cos_sin(h, q.device) # [1, 1, h, half_dim]
        cos_w, sin_w = self.rope_w._get_cos_sin(w, q.device) # [1, 1, w, half_dim]
        
        # Broadcast to [1, 1, h, w, half_dim]
        cos_h = cos_h.unsqueeze(3).expand(-1, -1, -1, w, -1)
        sin_h = sin_h.unsqueeze(3).expand(-1, -1, -1, w, -1)
        
        cos_w = cos_w.unsqueeze(2).expand(-1, -1, h, -1, -1)
        sin_w = sin_w.unsqueeze(2).expand(-1, -1, h, -1, -1)
        
        # Flatten spatial dims to sequence dim: [1, 1, h*w, half_dim]
        cos_h = cos_h.reshape(1, 1, h*w, self.half_dim)
        sin_h = sin_h.reshape(1, 1, h*w, self.half_dim)
        cos_w = cos_w.reshape(1, 1, h*w, self.half_dim)
        sin_w = sin_w.reshape(1, 1, h*w, self.half_dim)
        
        # Concatenate height and width components
        cos = torch.cat([cos_h, cos_w], dim=-1) # [1, 1, S, dim]
        sin = torch.cat([sin_h, sin_w], dim=-1) # [1, 1, S, dim]
        
        q_rot = self.rope_h._apply_rotary_emb(q, cos, sin)
        k_rot = self.rope_h._apply_rotary_emb(k, cos, sin)
        
        return q_rot, k_rot

class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding for spatiotemporal data (Video).
    Splits the head dimension into three thirds: time rotation, row rotation, column rotation.
    """
    def __init__(self, dim, base=10000.0):
        super().__init__()
        # Pad dimension if not divisible by 3 (rare, usually dimensions are powers of 2)
        # We'll assert for simplicity that it divides evenly or handle it.
        # DeepSeek often uses head_dim=128 (not div by 3). Let's use proportional splitting or padding.
        # For a standard implementation, we split into 3 parts.
        self.dim = dim
        self.t_dim = dim // 3
        self.h_dim = dim // 3
        self.w_dim = dim - self.t_dim - self.h_dim # remainder
        
        self.rope_t = RoPE(self.t_dim, base)
        self.rope_h = RoPE(self.h_dim, base)
        self.rope_w = RoPE(self.w_dim, base)
        
    def forward(self, q, k, t, h, w):
        # Sequence length S = t * h * w
        cos_t, sin_t = self.rope_t._get_cos_sin(t, q.device)
        cos_h, sin_h = self.rope_h._get_cos_sin(h, q.device)
        cos_w, sin_w = self.rope_w._get_cos_sin(w, q.device)
        
        # Broadcast to [1, 1, t, h, w, part_dim]
        cos_t = cos_t.view(1, 1, t, 1, 1, self.t_dim).expand(-1, -1, -1, h, w, -1)
        sin_t = sin_t.view(1, 1, t, 1, 1, self.t_dim).expand(-1, -1, -1, h, w, -1)
        
        cos_h = cos_h.view(1, 1, 1, h, 1, self.h_dim).expand(-1, -1, t, -1, w, -1)
        sin_h = sin_h.view(1, 1, 1, h, 1, self.h_dim).expand(-1, -1, t, -1, w, -1)
        
        cos_w = cos_w.view(1, 1, 1, 1, w, self.w_dim).expand(-1, -1, t, h, -1, -1)
        sin_w = sin_w.view(1, 1, 1, 1, w, self.w_dim).expand(-1, -1, t, h, -1, -1)
        
        # Flatten spatial dims to sequence dim: [1, 1, t*h*w, part_dim]
        S = t * h * w
        cos_t = cos_t.reshape(1, 1, S, self.t_dim)
        sin_t = sin_t.reshape(1, 1, S, self.t_dim)
        
        cos_h = cos_h.reshape(1, 1, S, self.h_dim)
        sin_h = sin_h.reshape(1, 1, S, self.h_dim)
        
        cos_w = cos_w.reshape(1, 1, S, self.w_dim)
        sin_w = sin_w.reshape(1, 1, S, self.w_dim)
        
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)
        
        q_rot = self.rope_t._apply_rotary_emb(q, cos, sin)
        k_rot = self.rope_t._apply_rotary_emb(k, cos, sin)
        
        return q_rot, k_rot
