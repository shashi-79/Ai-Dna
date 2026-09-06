"""
High-Performance Recurrent Qwen Causal Language Model Runner.
Supports both:
  - Option A: Step-Modulated LoRA (W_base + A_t * B_t per loop step)
  - Option B: Pure Recurrent Consolidation (W_base + step embeddings e_step(t))
Provides Hugging Face-compatible generate() API with large batch inference support.
"""

import os
import math
import json
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class RecurrentRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        normed = (x.float() * rms).to(self.weight.dtype) * self.weight
        return normed.to(x.dtype)


class RecurrentRoPE:
    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, H, S, D)
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        cos_t = cos.to(dtype=x.dtype)
        sin_t = sin.to(dtype=x.dtype)
        return (x * cos_t) + (rotated * sin_t)

    @staticmethod
    def precompute_cos_sin(seq_len: int, dim: int, device: torch.device, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, dim)
        sin = emb.sin().unsqueeze(0).unsqueeze(1)
        return cos, sin


class RecurrentQwenForCausalLM(nn.Module):
    """
    Recurrent Qwen Causal Language Model.
    Executes a single recurrent transformer layer looped T times.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.vocab_size = config.get("vocab_size", 151936)
        self.d_model = config.get("hidden_size", 896)
        self.recurrent_depth = config.get("recurrent_depth", config.get("num_hidden_layers", 24))
        self.strategy = config.get("recurrent_strategy", "step_lora")
        self.num_heads = config.get("num_attention_heads", 14)
        self.num_kv_heads = config.get("num_key_value_heads", 2)
        self.head_dim = config.get("head_dim", self.d_model // self.num_heads)
        self.intermediate_size = config.get("intermediate_size", 4864)
        self.rope_theta = config.get("rope_theta", 1000000.0)
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)

        # Base recurrent weights dictionary (Layer 0 anchor)
        self.base_weights: Dict[str, torch.Tensor] = {}
        # Step adapters: (t, subkey, 'lora_A'|'lora_B'|'delta_1d')
        self.step_adapters: Dict[str, torch.Tensor] = {}
        # Step embeddings for temporal differentiation
        self.step_embeddings: Optional[torch.Tensor] = None

        # Norm modules
        self.input_layernorm = RecurrentRMSNorm(self.d_model, eps=self.rms_norm_eps)
        self.post_attention_layernorm = RecurrentRMSNorm(self.d_model, eps=self.rms_norm_eps)
        self.final_norm = RecurrentRMSNorm(self.d_model, eps=self.rms_norm_eps)

        # Non-layer tensors
        self.embed_tokens: Optional[torch.Tensor] = None
        self.lm_head: Optional[torch.Tensor] = None

    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = "cuda", dtype: torch.dtype = torch.float32) -> "RecurrentQwenForCausalLM":
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(config)
        dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        st_path = os.path.join(model_dir, "model.safetensors")

        print(f"[Recurrent Model Loader] Loading Type 7 model from {st_path} to {dev} ...")
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k).to(dtype=dtype, device=dev)
                if k == "model.embed_tokens.weight":
                    model.embed_tokens = t
                elif k == "model.norm.weight":
                    model.final_norm.weight.data.copy_(t)
                elif k == "lm_head.weight":
                    model.lm_head = t
                elif k == "model.step_embeddings.weight":
                    model.step_embeddings = t
                elif k.startswith("model.layers.0."):
                    sub = k[len("model.layers.0."):]
                    model.base_weights[sub] = t
                    if sub == "input_layernorm.weight":
                        model.input_layernorm.weight = nn.Parameter(t.clone())
                    elif sub == "post_attention_layernorm.weight":
                        model.post_attention_layernorm.weight = nn.Parameter(t.clone())
                elif k.startswith("model.step_adapters."):
                    sub = k[len("model.step_adapters."):]
                    model.step_adapters[sub] = t

        if model.lm_head is None and model.embed_tokens is not None:
            model.lm_head = model.embed_tokens

        # Precompute RoPE tables
        model.cos_cached, model.sin_cached = RecurrentRoPE.precompute_cos_sin(
            seq_len=4096,
            dim=model.head_dim,
            device=dev,
            base=model.rope_theta,
        )

        model.to(dev)
        model.to(dtype=dtype)
        model.eval()
        return model

    def _linear_step(self, x: torch.Tensor, subkey: str, t: int) -> torch.Tensor:
        """
        Executes linear projection with Type 7 full-rank step residual:
        y = x @ W_base^T + (x @ B_t^T) @ A_t^T
        """
        W_base = self.base_weights[f"{subkey}.weight"]
        y = F.linear(x, W_base)

        bias_key = f"{subkey}.bias"
        if bias_key in self.base_weights:
            y = y + self.base_weights[bias_key]

        a_key = f"{t}.{subkey}.weight.lora_A"
        b_key = f"{t}.{subkey}.weight.lora_B"
        if a_key in self.step_adapters and b_key in self.step_adapters:
            A = self.step_adapters[a_key]  # (out_dim, r)
            B = self.step_adapters[b_key]  # (r, in_dim)
            # Low-rank forward: (x @ B^T) @ A^T
            y_lora = F.linear(F.linear(x, B), A)
            y = y + y_lora

        d_bias_key = f"{t}.{subkey}.bias.delta_1d"
        if d_bias_key in self.step_adapters:
            y = y + self.step_adapters[d_bias_key]

        return y

    def _forward_recurrent_layer(self, h: torch.Tensor, t: int, position_ids: torch.Tensor) -> torch.Tensor:
        B, S, D = h.shape

        # Step embedding modulation (temporal differentiation)
        if self.step_embeddings is not None and t < self.step_embeddings.shape[0]:
            emb_t = self.step_embeddings[t:t+1, :].unsqueeze(0)
            if emb_t.abs().max() > 0:
                h = h + emb_t

        # 1. Attention Block
        norm_h = self.input_layernorm(h)
        d_norm_key = f"{t}.input_layernorm.weight.delta_1d"
        if d_norm_key in self.step_adapters:
            variance = h.pow(2).mean(-1, keepdim=True)
            norm_raw = h * torch.rsqrt(variance + self.input_layernorm.eps)
            norm_h = norm_h + norm_raw * self.step_adapters[d_norm_key]

        # Self-Attention Projections
        q = self._linear_step(norm_h, "self_attn.q_proj", t)
        k = self._linear_step(norm_h, "self_attn.k_proj", t)
        v = self._linear_step(norm_h, "self_attn.v_proj", t)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos = self.cos_cached[:, :, position_ids[0], :]
        sin = self.sin_cached[:, :, position_ids[0], :]
        q = RecurrentRoPE.apply_rotary_emb(q, cos, sin)
        k = RecurrentRoPE.apply_rotary_emb(k, cos, sin)

        # Grouped Query Attention (repeat KV heads)
        if self.num_kv_heads < self.num_heads:
            num_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_rep, dim=1)
            v = v.repeat_interleave(num_rep, dim=1)

        # Scaled Dot-Product Attention with causal masking
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_proj = self._linear_step(attn_out, "self_attn.o_proj", t)

        # Residual connection
        h = h + attn_proj

        # 2. MLP Block (SwiGLU)
        norm_mlp = self.post_attention_layernorm(h)
        d_mlp_norm_key = f"{t}.post_attention_layernorm.weight.delta_1d"
        if d_mlp_norm_key in self.step_adapters:
            variance = h.pow(2).mean(-1, keepdim=True)
            norm_raw = h * torch.rsqrt(variance + self.post_attention_layernorm.eps)
            norm_mlp = norm_mlp + norm_raw * self.step_adapters[d_mlp_norm_key]

        gate = self._linear_step(norm_mlp, "mlp.gate_proj", t)
        up = self._linear_step(norm_mlp, "mlp.up_proj", t)
        swiglu = F.silu(gate) * up
        down = self._linear_step(swiglu, "mlp.down_proj", t)

        # Residual connection
        h = h + down
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        h = F.embedding(input_ids, self.embed_tokens)
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # Execute Recurrent Loop across T iterations
        for t in range(self.recurrent_depth):
            h = self._forward_recurrent_layer(h, t=t, position_ids=position_ids)

        if last_token_only:
            h = h[:, -1:, :]

        h = self.final_norm(h)
        logits = F.linear(h, self.lm_head)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 36,
        do_sample: bool = False,
        temperature: float = 0.0,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Batched Autoregressive Generation using Recurrent Depth forward passes.
        """
        curr_ids = input_ids.clone()
        B = curr_ids.shape[0]
        eos_id = pad_token_id or getattr(self, "eos_token_id", None)
        unfinished = torch.ones(B, dtype=torch.long, device=input_ids.device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(curr_ids, last_token_only=True)
                next_logits = logits.squeeze(1)

                if do_sample and temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                if eos_id is not None:
                    # Keep finished sequences as eos/pad
                    next_token = next_token * unfinished.unsqueeze(1) + eos_id * (1 - unfinished.unsqueeze(1))
                    unfinished = unfinished * (next_token.squeeze(1) != eos_id).long()
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)
                    if unfinished.max() == 0:
                        break
                else:
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)

        return curr_ids

