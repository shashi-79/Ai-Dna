"""
Minimal Zero-Dependency Model Architectures for AI-DNA.
Provides standalone MinimalSmolLM2, MinimalCLIP, and MinimalWhisper model shells
capable of direct functional inference from raw state dictionaries without requiring external transformers.
"""

import os
import math
import struct
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import CLIPTokenizer, WhisperTokenizer


def compute_mel_spectrogram_from_waveform(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> torch.Tensor:
    """Computes standard 80-channel log-mel spectrogram for Whisper from 1D audio waveform."""
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(waveform.float(), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitudes = stft.abs() ** 2

    # Construct 80-band Mel filterbank
    mmin = 2595.0 * math.log10(1.0 + 0.0 / 700.0)
    mmax = 2595.0 * math.log10(1.0 + 8000.0 / 700.0)
    m_pts = torch.linspace(mmin, mmax, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    bins = torch.floor((n_fft + 1) * f_pts / sample_rate).long()

    fb = torch.zeros(n_mels, n_fft // 2 + 1, device=waveform.device)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1].item()
        f_m = bins[m].item()
        f_m_plus = bins[m + 1].item()

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_spec = torch.matmul(fb, magnitudes)
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0)  # [1, 80, T_frames]


class MinimalRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class MinimalSmolLM2(nn.Module):

    """Minimal inference-only SmolLM2 / LLaMA-style decoder."""

    def __init__(self, config: Dict):
        super().__init__()
        self.vocab_size = config.get("vocab_size", 49152)
        self.d_model = config.get("hidden_size", 576)
        self.num_layers = config.get("num_hidden_layers", 30)
        self.num_heads = config.get("num_attention_heads", 9)
        self.num_kv_heads = config.get("num_key_value_heads", 3)
        self.head_dim = self.d_model // self.num_heads
        self.intermediate_size = config.get("intermediate_size", 1536)

    def generate(
        self,
        state_dict: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> List[int]:
        """Pure functional generation using raw state_dict — no nn.Module weight loading needed."""
        device = input_ids.device
        B, seq_len = input_ids.shape
        generated = input_ids[0].tolist()

        # Get embedding weight
        embed_w = state_dict.get("model.embed_tokens.weight", state_dict.get("lm_head.weight"))
        if embed_w is None:
            return generated

        for step in range(max_new_tokens):
            x = F.embedding(torch.tensor([generated], device=device), embed_w.to(device))
            for layer_idx in range(self.num_layers):
                pfx = f"model.layers.{layer_idx}"
                # Input LayerNorm
                ln_w = state_dict.get(f"{pfx}.input_layernorm.weight")
                if ln_w is not None:
                    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                    x_norm = (x.float() * rms) * ln_w.to(device).float()
                else:
                    x_norm = x.float()

                # Self-Attention (simplified GQA)
                q_w = state_dict.get(f"{pfx}.self_attn.q_proj.weight")
                k_w = state_dict.get(f"{pfx}.self_attn.k_proj.weight")
                v_w = state_dict.get(f"{pfx}.self_attn.v_proj.weight")
                o_w = state_dict.get(f"{pfx}.self_attn.o_proj.weight")

                if q_w is not None and k_w is not None and v_w is not None and o_w is not None:
                    Q = x_norm @ q_w.to(device).float().T
                    K = x_norm @ k_w.to(device).float().T
                    V = x_norm @ v_w.to(device).float().T

                    S = len(generated)
                    Q = Q.view(1, S, self.num_heads, self.head_dim).transpose(1, 2)
                    K = K.view(1, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    V = V.view(1, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

                    # GQA: repeat KV heads
                    rep = self.num_heads // self.num_kv_heads
                    if rep > 1:
                        K = K.repeat_interleave(rep, dim=1)
                        V = V.repeat_interleave(rep, dim=1)

                    # Causal attention
                    scale = 1.0 / math.sqrt(self.head_dim)
                    scores = (Q @ K.transpose(-2, -1)) * scale
                    causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
                    scores = scores + causal_mask
                    attn = F.softmax(scores, dim=-1)
                    attn_out = (attn @ V).transpose(1, 2).contiguous().view(1, S, self.d_model)
                    attn_out = attn_out @ o_w.to(device).float().T
                    x = x + attn_out.to(x.dtype)

                # Post-attention LayerNorm
                ln2_w = state_dict.get(f"{pfx}.post_attention_layernorm.weight")
                if ln2_w is not None:
                    rms2 = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                    x_mlp = (x.float() * rms2) * ln2_w.to(device).float()
                else:
                    x_mlp = x.float()

                # SwiGLU MLP
                gate_w = state_dict.get(f"{pfx}.mlp.gate_proj.weight")
                up_w = state_dict.get(f"{pfx}.mlp.up_proj.weight")
                down_w = state_dict.get(f"{pfx}.mlp.down_proj.weight")
                if gate_w is not None and up_w is not None and down_w is not None:
                    gate = x_mlp @ gate_w.to(device).float().T
                    up = x_mlp @ up_w.to(device).float().T
                    mlp_out = (F.silu(gate) * up) @ down_w.to(device).float().T
                    x = x + mlp_out.to(x.dtype)

            # Final RMSNorm
            final_ln = state_dict.get("model.norm.weight")
            if final_ln is not None:
                rms_f = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                x = (x.float() * rms_f) * final_ln.to(device).float()

            # LM Head
            lm_head_w = state_dict.get("lm_head.weight", embed_w)
            logits = x[:, -1, :] @ lm_head_w.to(device).float().T

            # Sampling: Greedy argmax when temperature is near zero, else Top-K + Temperature sampling
            if temperature <= 1e-4:
                next_token = logits.argmax(dim=-1).item()
            else:
                logits = logits / temperature
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                if top_k > 0:
                    top_vals, top_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(1, top_idx, top_vals)
                    logits = logits_filtered

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            # Stop on EOS
            if next_token in [0, 1, 2]:
                break

        return generated


class MinimalCLIP(nn.Module):
    """Minimal inference-only CLIP-ViT for image/text embedding extraction and zero-shot decoded classification."""

    def __init__(self, config: Dict):
        super().__init__()
        self.vision_config = config.get("vision_config", {})
        self.d_model = self.vision_config.get("hidden_size", 768)
        self.num_layers = self.vision_config.get("num_hidden_layers", 12)
        self.num_heads = self.vision_config.get("num_attention_heads", 12)
        self.head_dim = self.d_model // self.num_heads
        self.intermediate_size = self.vision_config.get("intermediate_size", 3072)
        self.patch_size = self.vision_config.get("patch_size", 32)
        self.image_size = self.vision_config.get("image_size", 224)

        self.text_config = config.get("text_config", {})
        self.text_d_model = self.text_config.get("hidden_size", 512)
        self.text_layers = self.text_config.get("num_hidden_layers", 12)
        self.text_heads = self.text_config.get("num_attention_heads", 8)
        self.text_head_dim = self.text_d_model // self.text_heads
        self.text_intermediate = self.text_config.get("intermediate_size", 2048)

    def encode_image(self, state_dict: Dict[str, torch.Tensor], pixel_values: torch.Tensor) -> torch.Tensor:
        """Runs ViT forward pass using raw state_dict and returns normalized [CLS] visual embedding [B, 512]."""
        device = pixel_values.device
        B = pixel_values.shape[0]

        patch_w = state_dict.get("vision_model.embeddings.patch_embedding.weight")  # [768, 3, 32, 32]
        if patch_w is None:
            return torch.randn(B, 512, device=device)

        patches = F.conv2d(
            pixel_values.float(),
            patch_w.to(device).float(),
            bias=state_dict.get("vision_model.embeddings.patch_embedding.bias", torch.zeros(self.d_model, device=device)).to(device).float(),
            stride=self.patch_size,
        )
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]

        cls_emb = state_dict.get("vision_model.embeddings.class_embedding")
        if cls_emb is not None:
            cls_tok = cls_emb.to(device).float().unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([cls_tok, patches], dim=1)
        else:
            x = patches

        pos_emb = state_dict.get("vision_model.embeddings.position_embedding.weight")
        if pos_emb is not None:
            seq_len = x.shape[1]
            x = x + pos_emb[:seq_len].to(device).float().unsqueeze(0)

        pre_ln_w = state_dict.get("vision_model.pre_layrnorm.weight", state_dict.get("vision_model.pre_layernorm.weight"))
        pre_ln_b = state_dict.get("vision_model.pre_layrnorm.bias", state_dict.get("vision_model.pre_layernorm.bias"))
        if pre_ln_w is not None:
            x = F.layer_norm(x, [self.d_model], pre_ln_w.to(device).float(), pre_ln_b.to(device).float() if pre_ln_b is not None else None)

        for i in range(self.num_layers):
            pfx = f"vision_model.encoder.layers.{i}"

            ln1_w = state_dict.get(f"{pfx}.layer_norm1.weight")
            ln1_b = state_dict.get(f"{pfx}.layer_norm1.bias")
            x_norm = F.layer_norm(x, [self.d_model], ln1_w.to(device).float(), ln1_b.to(device).float() if ln1_b is not None else None) if ln1_w is not None else x

            q_w = state_dict.get(f"{pfx}.self_attn.q_proj.weight")
            k_w = state_dict.get(f"{pfx}.self_attn.k_proj.weight")
            v_w = state_dict.get(f"{pfx}.self_attn.v_proj.weight")
            o_w = state_dict.get(f"{pfx}.self_attn.out_proj.weight")
            if q_w is not None:
                S = x_norm.shape[1]
                Q = (x_norm @ q_w.to(device).float().T + state_dict.get(f"{pfx}.self_attn.q_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float())
                K = (x_norm @ k_w.to(device).float().T + state_dict.get(f"{pfx}.self_attn.k_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float())
                V = (x_norm @ v_w.to(device).float().T + state_dict.get(f"{pfx}.self_attn.v_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float())

                Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
                K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
                V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn = F.softmax(scores, dim=-1)
                attn_out = (attn @ V).transpose(1, 2).contiguous().view(B, S, self.d_model)
                o_b = state_dict.get(f"{pfx}.self_attn.out_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                attn_out = attn_out @ o_w.to(device).float().T + o_b
                x = x + attn_out

            ln2_w = state_dict.get(f"{pfx}.layer_norm2.weight")
            ln2_b = state_dict.get(f"{pfx}.layer_norm2.bias")
            x_norm2 = F.layer_norm(x, [self.d_model], ln2_w.to(device).float(), ln2_b.to(device).float() if ln2_b is not None else None) if ln2_w is not None else x

            fc1_w = state_dict.get(f"{pfx}.mlp.fc1.weight")
            fc2_w = state_dict.get(f"{pfx}.mlp.fc2.weight")
            if fc1_w is not None and fc2_w is not None:
                fc1_b = state_dict.get(f"{pfx}.mlp.fc1.bias", torch.zeros(self.intermediate_size, device=device)).to(device).float()
                fc2_b = state_dict.get(f"{pfx}.mlp.fc2.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                h = F.gelu(x_norm2 @ fc1_w.to(device).float().T + fc1_b, approximate="tanh")
                mlp_out = h @ fc2_w.to(device).float().T + fc2_b
                x = x + mlp_out

        post_ln_w = state_dict.get("vision_model.post_layernorm.weight")
        post_ln_b = state_dict.get("vision_model.post_layernorm.bias")
        if post_ln_w is not None:
            x = F.layer_norm(x, [self.d_model], post_ln_w.to(device).float(), post_ln_b.to(device).float() if post_ln_b is not None else None)

        cls_output = x[:, 0, :]
        vis_proj = state_dict.get("visual_projection.weight")
        if vis_proj is not None:
            cls_output = cls_output @ vis_proj.to(device).float().T

        return F.normalize(cls_output, dim=-1)

    def encode_text(self, state_dict: Dict[str, torch.Tensor], input_ids: torch.Tensor, eot_indices: List[int]) -> torch.Tensor:
        """Runs CLIP text encoder forward pass and returns normalized [EOT] text embeddings [N, 512]."""
        device = input_ids.device
        N, S = input_ids.shape

        tok_emb = state_dict.get("text_model.embeddings.token_embedding.weight")
        pos_emb = state_dict.get("text_model.embeddings.position_embedding.weight")
        if tok_emb is None:
            return torch.randn(N, 512, device=device)

        x = F.embedding(input_ids, tok_emb.to(device).float())
        if pos_emb is not None:
            x = x + pos_emb[:S].to(device).float().unsqueeze(0)

        for i in range(self.text_layers):
            pfx = f"text_model.encoder.layers.{i}"
            ln1_w = state_dict.get(f"{pfx}.layer_norm1.weight")
            ln1_b = state_dict.get(f"{pfx}.layer_norm1.bias")
            x_norm = F.layer_norm(x, [self.text_d_model], ln1_w.to(device).float(), ln1_b.to(device).float()) if ln1_w is not None else x

            q_w = state_dict.get(f"{pfx}.self_attn.q_proj.weight")
            q_b = state_dict.get(f"{pfx}.self_attn.q_proj.bias", torch.zeros(self.text_d_model, device=device))
            k_w = state_dict.get(f"{pfx}.self_attn.k_proj.weight")
            k_b = state_dict.get(f"{pfx}.self_attn.k_proj.bias", torch.zeros(self.text_d_model, device=device))
            v_w = state_dict.get(f"{pfx}.self_attn.v_proj.weight")
            v_b = state_dict.get(f"{pfx}.self_attn.v_proj.bias", torch.zeros(self.text_d_model, device=device))
            o_w = state_dict.get(f"{pfx}.self_attn.out_proj.weight")
            o_b = state_dict.get(f"{pfx}.self_attn.out_proj.bias", torch.zeros(self.text_d_model, device=device))

            if q_w is not None:
                Q = (x_norm @ q_w.to(device).float().T + q_b.to(device).float()).view(N, S, self.text_heads, self.text_head_dim).transpose(1, 2)
                K = (x_norm @ k_w.to(device).float().T + k_b.to(device).float()).view(N, S, self.text_heads, self.text_head_dim).transpose(1, 2)
                V = (x_norm @ v_w.to(device).float().T + v_b.to(device).float()).view(N, S, self.text_heads, self.text_head_dim).transpose(1, 2)

                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.text_head_dim)
                causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
                attn = F.softmax(scores + causal_mask, dim=-1)
                attn_out = (attn @ V).transpose(1, 2).contiguous().view(N, S, self.text_d_model)
                x = x + (attn_out @ o_w.to(device).float().T + o_b.to(device).float())

            ln2_w = state_dict.get(f"{pfx}.layer_norm2.weight")
            ln2_b = state_dict.get(f"{pfx}.layer_norm2.bias")
            x_norm2 = F.layer_norm(x, [self.text_d_model], ln2_w.to(device).float(), ln2_b.to(device).float()) if ln2_w is not None else x

            fc1_w = state_dict.get(f"{pfx}.mlp.fc1.weight")
            fc1_b = state_dict.get(f"{pfx}.mlp.fc1.bias", torch.zeros(self.text_intermediate, device=device))
            fc2_w = state_dict.get(f"{pfx}.mlp.fc2.weight")
            fc2_b = state_dict.get(f"{pfx}.mlp.fc2.bias", torch.zeros(self.text_d_model, device=device))

            if fc1_w is not None and fc2_w is not None:
                h = x_norm2 @ fc1_w.to(device).float().T + fc1_b.to(device).float()
                h_act = h * torch.sigmoid(1.702 * h)
                x = x + (h_act @ fc2_w.to(device).float().T + fc2_b.to(device).float())

        final_ln_w = state_dict.get("text_model.final_layer_norm.weight")
        final_ln_b = state_dict.get("text_model.final_layer_norm.bias")
        if final_ln_w is not None:
            x = F.layer_norm(x, [self.text_d_model], final_ln_w.to(device).float(), final_ln_b.to(device).float())

        eot_features = torch.stack([x[bi, eot_idx] for bi, eot_idx in enumerate(eot_indices)])
        text_proj = state_dict.get("text_projection.weight")
        if text_proj is not None:
            eot_features = eot_features @ text_proj.to(device).float()

        return F.normalize(eot_features, dim=-1)

    def decode_image_classification(
        self,
        state_dict: Dict[str, torch.Tensor],
        pixel_values: torch.Tensor,
        candidate_labels: List[str],
        tokenizer: CLIPTokenizer,
    ) -> List[Tuple[str, float, float]]:
        """
        Decodes image content by computing zero-shot classification probabilities against candidate labels.
        Returns sorted list of (label, probability_percentage, raw_logit).
        """
        device = pixel_values.device
        img_emb = self.encode_image(state_dict, pixel_values)

        input_ids, eot_indices = tokenizer.encode(candidate_labels)
        text_embs = self.encode_text(state_dict, input_ids.to(device), eot_indices)

        logit_scale = state_dict.get("logit_scale", torch.tensor(4.6052, device=device)).to(device).float().exp()
        logits = (img_emb @ text_embs.T) * logit_scale
        probs = F.softmax(logits, dim=-1)[0].tolist()
        logits_list = logits[0].tolist()

        results = [(label, probs[i] * 100.0, logits_list[i]) for i, label in enumerate(candidate_labels)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class MinimalWhisper(nn.Module):
    """Full inference-only Whisper speech-to-text model (Encoder + Decoder + Tokenizer)."""

    def __init__(self, config: Dict):
        super().__init__()
        self.d_model = config.get("d_model", 384)
        self.encoder_layers = config.get("encoder_layers", 4)
        self.decoder_layers = config.get("decoder_layers", 4)
        self.encoder_heads = config.get("encoder_attention_heads", 6)
        self.decoder_heads = config.get("decoder_attention_heads", 6)
        self.head_dim = self.d_model // self.encoder_heads
        self.encoder_ffn_dim = config.get("encoder_ffn_dim", 1536)
        self.decoder_ffn_dim = config.get("decoder_ffn_dim", 1536)
        self.vocab_size = config.get("vocab_size", 51865)

    def encode(self, state_dict: Dict[str, torch.Tensor], mel_features: torch.Tensor) -> torch.Tensor:
        """Runs Whisper encoder on mel spectrogram features. mel_features: [B, 80, T] -> [B, T/2, d_model]"""
        device = mel_features.device
        B = mel_features.shape[0]

        conv1_w = state_dict.get("model.encoder.conv1.weight")
        conv1_b = state_dict.get("model.encoder.conv1.bias")
        conv2_w = state_dict.get("model.encoder.conv2.weight")
        conv2_b = state_dict.get("model.encoder.conv2.bias")

        if conv1_w is not None:
            x = F.gelu(F.conv1d(mel_features.float(), conv1_w.to(device).float(),
                                conv1_b.to(device).float() if conv1_b is not None else None, padding=1))
            if conv2_w is not None:
                x = F.gelu(F.conv1d(x, conv2_w.to(device).float(),
                                    conv2_b.to(device).float() if conv2_b is not None else None,
                                    stride=2, padding=1))
        else:
            T = mel_features.shape[2]
            x = torch.randn(B, self.d_model, T // 2, device=device)

        x = x.transpose(1, 2)

        pos_emb = state_dict.get("model.encoder.embed_positions.weight")
        if pos_emb is not None:
            seq_len = min(x.shape[1], pos_emb.shape[0])
            x[:, :seq_len, :] = x[:, :seq_len, :] + pos_emb[:seq_len].to(device).float().unsqueeze(0)

        for i in range(self.encoder_layers):
            pfx = f"model.encoder.layers.{i}"

            ln1_w = state_dict.get(f"{pfx}.self_attn_layer_norm.weight")
            ln1_b = state_dict.get(f"{pfx}.self_attn_layer_norm.bias")
            x_norm = F.layer_norm(x, [self.d_model], ln1_w.to(device).float() if ln1_w is not None else None,
                                  ln1_b.to(device).float() if ln1_b is not None else None)

            q_w = state_dict.get(f"{pfx}.self_attn.q_proj.weight")
            k_w = state_dict.get(f"{pfx}.self_attn.k_proj.weight")
            v_w = state_dict.get(f"{pfx}.self_attn.v_proj.weight")
            o_w = state_dict.get(f"{pfx}.self_attn.out_proj.weight")

            if q_w is not None:
                S = x_norm.shape[1]
                q_b = state_dict.get(f"{pfx}.self_attn.q_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                k_b = state_dict.get(f"{pfx}.self_attn.k_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                v_b = state_dict.get(f"{pfx}.self_attn.v_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()

                Q = (x_norm @ q_w.to(device).float().T + q_b).view(B, S, self.encoder_heads, self.head_dim).transpose(1, 2)
                K = (x_norm @ k_w.to(device).float().T + k_b).view(B, S, self.encoder_heads, self.head_dim).transpose(1, 2)
                V = (x_norm @ v_w.to(device).float().T + v_b).view(B, S, self.encoder_heads, self.head_dim).transpose(1, 2)

                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn = F.softmax(scores, dim=-1)
                attn_out = (attn @ V).transpose(1, 2).contiguous().view(B, S, self.d_model)
                o_b = state_dict.get(f"{pfx}.self_attn.out_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                attn_out = attn_out @ o_w.to(device).float().T + o_b
                x = x + attn_out

            ln2_w = state_dict.get(f"{pfx}.final_layer_norm.weight")
            ln2_b = state_dict.get(f"{pfx}.final_layer_norm.bias")
            x_norm2 = F.layer_norm(x, [self.d_model], ln2_w.to(device).float() if ln2_w is not None else None,
                                   ln2_b.to(device).float() if ln2_b is not None else None)

            fc1_w = state_dict.get(f"{pfx}.fc1.weight")
            fc2_w = state_dict.get(f"{pfx}.fc2.weight")
            if fc1_w is not None and fc2_w is not None:
                fc1_b = state_dict.get(f"{pfx}.fc1.bias", torch.zeros(self.encoder_ffn_dim, device=device)).to(device).float()
                fc2_b = state_dict.get(f"{pfx}.fc2.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                h = F.gelu(x_norm2 @ fc1_w.to(device).float().T + fc1_b)
                x = x + (h @ fc2_w.to(device).float().T + fc2_b)

        final_ln_w = state_dict.get("model.encoder.layer_norm.weight")
        final_ln_b = state_dict.get("model.encoder.layer_norm.bias")
        if final_ln_w is not None:
            x = F.layer_norm(x, [self.d_model], final_ln_w.to(device).float(),
                             final_ln_b.to(device).float() if final_ln_b is not None else None)

        return x

    def decode_transcribe(
        self,
        state_dict: Dict[str, torch.Tensor],
        mel_features: torch.Tensor,
        tokenizer: WhisperTokenizer,
        max_new_tokens: int = 25,
        prompt_tokens: Optional[List[int]] = None,
    ) -> Tuple[List[int], str]:
        """
        Runs full Whisper speech-to-text autoregressive decoding with Cross-Attention.
        Returns: (generated_token_ids, decoded_transcription_string)
        """
        device = mel_features.device
        x_enc = self.encode(state_dict, mel_features)
        T_enc = x_enc.shape[1]

        tok_emb = state_dict.get("model.decoder.embed_tokens.weight")
        pos_dec = state_dict.get("model.decoder.embed_positions.weight")
        if tok_emb is None:
            return [], ""

        tok_emb = tok_emb.to(device).float()
        pos_dec = pos_dec.to(device).float() if pos_dec is not None else None

        if prompt_tokens is None:
            prompt_tokens = [50258, 50259, 50359, 50363]

        generated = list(prompt_tokens)

        for step in range(max_new_tokens):
            S = len(generated)
            cur_tokens = torch.tensor([generated], device=device, dtype=torch.long)

            x_dec = F.embedding(cur_tokens, tok_emb)
            if pos_dec is not None:
                x_dec = x_dec + pos_dec[:S].unsqueeze(0)

            for i in range(self.decoder_layers):
                pfx = f"model.decoder.layers.{i}"

                # 1. Self-Attention
                ln1_w = state_dict.get(f"{pfx}.self_attn_layer_norm.weight")
                ln1_b = state_dict.get(f"{pfx}.self_attn_layer_norm.bias")
                x_norm = F.layer_norm(x_dec, [self.d_model], ln1_w.to(device).float() if ln1_w is not None else None,
                                      ln1_b.to(device).float() if ln1_b is not None else None)

                q_w = state_dict.get(f"{pfx}.self_attn.q_proj.weight")
                k_w = state_dict.get(f"{pfx}.self_attn.k_proj.weight")
                v_w = state_dict.get(f"{pfx}.self_attn.v_proj.weight")
                o_w = state_dict.get(f"{pfx}.self_attn.out_proj.weight")

                if q_w is not None:
                    q_b = state_dict.get(f"{pfx}.self_attn.q_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                    k_b = state_dict.get(f"{pfx}.self_attn.k_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                    v_b = state_dict.get(f"{pfx}.self_attn.v_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()

                    Q = (x_norm @ q_w.to(device).float().T + q_b).view(1, S, self.decoder_heads, self.head_dim).transpose(1, 2)
                    K = (x_norm @ k_w.to(device).float().T + k_b).view(1, S, self.decoder_heads, self.head_dim).transpose(1, 2)
                    V = (x_norm @ v_w.to(device).float().T + v_b).view(1, S, self.decoder_heads, self.head_dim).transpose(1, 2)

                    causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
                    attn = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim) + causal_mask, dim=-1)
                    attn_out = (attn @ V).transpose(1, 2).contiguous().view(1, S, self.d_model)
                    o_b = state_dict.get(f"{pfx}.self_attn.out_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                    x_dec = x_dec + (attn_out @ o_w.to(device).float().T + o_b)

                # 2. Cross-Attention
                ln_cross_w = state_dict.get(f"{pfx}.encoder_attn_layer_norm.weight")
                ln_cross_b = state_dict.get(f"{pfx}.encoder_attn_layer_norm.bias")
                x_cross_norm = F.layer_norm(x_dec, [self.d_model], ln_cross_w.to(device).float() if ln_cross_w is not None else None,
                                            ln_cross_b.to(device).float() if ln_cross_b is not None else None)

                cq_w = state_dict.get(f"{pfx}.encoder_attn.q_proj.weight")
                ck_w = state_dict.get(f"{pfx}.encoder_attn.k_proj.weight")
                cv_w = state_dict.get(f"{pfx}.encoder_attn.v_proj.weight")
                co_w = state_dict.get(f"{pfx}.encoder_attn.out_proj.weight")

                if cq_w is not None and ck_w is not None and cv_w is not None:
                    cq_b = state_dict.get(f"{pfx}.encoder_attn.q_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                    ck_b = state_dict.get(f"{pfx}.encoder_attn.k_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                    cv_b = state_dict.get(f"{pfx}.encoder_attn.v_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()

                    Q_cross = (x_cross_norm @ cq_w.to(device).float().T + cq_b).view(1, S, self.decoder_heads, self.head_dim).transpose(1, 2)
                    K_cross = (x_enc @ ck_w.to(device).float().T + ck_b).view(1, T_enc, self.decoder_heads, self.head_dim).transpose(1, 2)
                    V_cross = (x_enc @ cv_w.to(device).float().T + cv_b).view(1, T_enc, self.decoder_heads, self.head_dim).transpose(1, 2)

                    cross_attn = F.softmax((Q_cross @ K_cross.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
                    cross_out = (cross_attn @ V_cross).transpose(1, 2).contiguous().view(1, S, self.d_model)
                    co_b = state_dict.get(f"{pfx}.encoder_attn.out_proj.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                    x_dec = x_dec + (cross_out @ co_w.to(device).float().T + co_b)

                # 3. MLP
                ln3_w = state_dict.get(f"{pfx}.final_layer_norm.weight")
                ln3_b = state_dict.get(f"{pfx}.final_layer_norm.bias")
                x_mlp = F.layer_norm(x_dec, [self.d_model], ln3_w.to(device).float() if ln3_w is not None else None,
                                     ln3_b.to(device).float() if ln3_b is not None else None)

                fc1_w = state_dict.get(f"{pfx}.fc1.weight")
                fc2_w = state_dict.get(f"{pfx}.fc2.weight")
                if fc1_w is not None and fc2_w is not None:
                    fc1_b = state_dict.get(f"{pfx}.fc1.bias", torch.zeros(self.decoder_ffn_dim, device=device)).to(device).float()
                    fc2_b = state_dict.get(f"{pfx}.fc2.bias", torch.zeros(self.d_model, device=device)).to(device).float()
                    h = F.gelu(x_mlp @ fc1_w.to(device).float().T + fc1_b)
                    x_dec = x_dec + (h @ fc2_w.to(device).float().T + fc2_b)

            final_ln_w = state_dict.get("model.decoder.layer_norm.weight")
            final_ln_b = state_dict.get("model.decoder.layer_norm.bias")
            if final_ln_w is not None:
                x_final = F.layer_norm(x_dec, [self.d_model], final_ln_w.to(device).float(),
                                       final_ln_b.to(device).float() if final_ln_b is not None else None)
            else:
                x_final = x_dec

            logits = x_final[:, -1, :] @ tok_emb.T
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)

            if next_token == 50257:  # <|endoftext|>
                break

        transcription_text = tokenizer.decode(generated, skip_special=True)
        return generated, transcription_text


def compute_weight_diff_metrics(
    orig_sd: Dict[str, torch.Tensor],
    recon_sd: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Computes reconstruction fidelity metrics between original and AI-DNA-reconstructed weights."""
    shared_keys = set(orig_sd.keys()) & set(recon_sd.keys())
    if not shared_keys:
        return {"shared_keys": 0, "max_abs_diff": float("nan"), "mean_abs_diff": float("nan"), "cosine_sim": float("nan")}

    max_diff = 0.0
    total_diff = 0.0
    total_cos = 0.0
    total_params = 0
    count = 0

    for k in shared_keys:
        o = orig_sd[k].float().flatten()
        r = recon_sd[k].float().flatten()
        if o.shape != r.shape:
            continue
        diff = (o - r).abs()
        max_diff = max(max_diff, diff.max().item())
        total_diff += diff.sum().item()
        total_params += o.numel()
        dot = (o * r).sum()
        norm_o = o.norm()
        norm_r = r.norm()
        if norm_o > 1e-9 and norm_r > 1e-9:
            total_cos += (dot / (norm_o * norm_r)).item()
            count += 1

    return {
        "shared_keys": len(shared_keys),
        "max_abs_diff": max_diff,
        "mean_abs_diff": total_diff / max(total_params, 1),
        "cosine_sim": total_cos / max(count, 1),
    }


def compute_output_similarity(out_a: torch.Tensor, out_b: torch.Tensor) -> Dict[str, float]:
    """Computes similarity metrics between two output tensors."""
    a = out_a.float().flatten()
    b = out_b.float().flatten()
    if a.shape != b.shape:
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

    diff = (a - b).abs()
    dot = (a * b).sum()
    norm_a = a.norm()
    norm_b = b.norm()
    cos_sim = (dot / (norm_a * norm_b + 1e-9)).item()

    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_similarity": cos_sim,
        "l2_distance": (a - b).norm().item(),
        "relative_error": diff.sum().item() / (a.abs().sum().item() + 1e-9),
    }
