import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RoPE

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2.
    Uses low-rank joint latent compression for K and V to drastically reduce memory
    and limit the parameter surface area needed by the genotypic DNA encoder.
    """
    def __init__(self, d_model, num_heads, d_kv_latent, rope_base=10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_kv_latent = d_kv_latent
        
        # Query projection (standard)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        
        # KV compression down-projection
        # This is the primary target for DNA encoding since it's the bottleneck
        self.w_dkv = nn.Linear(d_model, d_kv_latent, bias=False)
        
        # Up-projections for K and V from the latent
        self.w_uk = nn.Linear(d_kv_latent, d_model, bias=False)
        self.w_uv = nn.Linear(d_kv_latent, d_model, bias=False)
        
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_head, rope_base)
        
    def forward(self, x, mask=None, kv_cache=None):
        # x: [B, S, D]
        B, S, _ = x.shape
        
        # 1. Project Query
        q = self.w_q(x) # [B, S, D]
        q = q.view(B, S, self.num_heads, self.d_head).transpose(1, 2) # [B, H, S, D_h]
        
        # 2. Compress KV to latent
        c_kv = self.w_dkv(x) # [B, S, d_kv]
        
        # 3. Up-project to K and V
        k = self.w_uk(c_kv) # [B, S, D]
        v = self.w_uv(c_kv) # [B, S, D]
        
        k = k.view(B, S, self.num_heads, self.d_head).transpose(1, 2) # [B, H, S, D_h]
        v = v.view(B, S, self.num_heads, self.d_head).transpose(1, 2) # [B, H, S, D_h]
        
        # 4. Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Optional: KV cache concatenation would happen here in autoregressive mode
        
        # 5. FlashAttention (IO-aware tiling via F.scaled_dot_product_attention)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            is_causal=(mask is None) # If no explicit mask, assume causal
        )
        
        # [B, H, S, D_h] -> [B, S, H, D_h] -> [B, S, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        
        # 6. Output projection
        out = self.o_proj(attn_out)
        return out
