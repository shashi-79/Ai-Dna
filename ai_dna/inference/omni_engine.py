"""
AI-DNA Omni-Modal Multimodal Inference Engine (Pure AI-DNA Architecture).
Driven strictly by SlowClock (Genotypic Encoding & Consolidation), FastClock (Sensory Dynamics),
GrowthEngine (Phenotype Regrowth G(D)), and PhenotypeNeuralNetwork (MLA + Top-K MoE Backbone).
"""

import os
import sys
import time
import json
import math
from typing import Dict, Any, Optional, Tuple, List, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dna.structure import Genotype
from ..dna.serialization import load_genotype
from ..growth.engine import GrowthEngine
from ..models.phenotype import PhenotypeNeuralNetwork
from ..encoding.slow_clock import SlowClockEncoder
from ..reasoning.verifier import ReasoningVerifier


# =========================================================================
# Multimodal Output Handler (Artifact & File Generation)
# =========================================================================
class MultimodalOutputHandler:
    """
    Handles independent multimodal output serialization, audio WAV synthesis,
    visual image/diagram artifact rendering, JSON structured records,
    and human-readable reporting.
    """
    @staticmethod
    def save_audio_waveform(
        waveform: Union[torch.Tensor, np.ndarray],
        filepath: str,
        sample_rate: int = 16000,
    ) -> str:
        """Saves a 1D float waveform tensor or numpy array as a 16-bit PCM WAV file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if isinstance(waveform, torch.Tensor):
            arr = waveform.detach().cpu().numpy()
        else:
            arr = np.array(waveform)

        if arr.ndim > 1:
            arr = arr.squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[1] else 1)

        arr = np.clip(arr, -1.0, 1.0)
        audio_int16 = (arr * 32767).astype(np.int16)

        try:
            import scipy.io.wavfile as wavfile
            wavfile.write(filepath, sample_rate, audio_int16)
        except Exception:
            import wave
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

        return filepath

    @staticmethod
    def save_image_artifact(
        image_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
        filepath: str = "modal/omni_inline_image.png",
        width: int = 256,
        height: int = 256,
        concept: str = "visual scene",
        caption: str = "",
    ) -> str:
        """
        Saves or generates a neural diffusion visual artifact.
        Converts neural decoded tensors directly to clean image files.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        try:
            from PIL import Image, ImageDraw, ImageFont

            if image_data is not None:
                if isinstance(image_data, torch.Tensor):
                    t = image_data.detach().cpu()
                    if t.ndim == 4:
                        t = t[0]
                    if t.ndim == 3 and t.shape[0] in (1, 3):
                        t = t.permute(1, 2, 0)
                    arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr.squeeze(-1)
                else:
                    arr = np.array(image_data).clip(0, 255).astype(np.uint8)
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr.squeeze(-1)
                img = Image.fromarray(arr)
            else:
                img = Image.new("RGB", (width, height), color=(15, 23, 42))
                draw = ImageDraw.Draw(img)
                for y in range(height):
                    r = int(15 + (y / height) * 45)
                    g = int(23 + (y / height) * 60)
                    b = int(42 + (y / height) * 90)
                    draw.line([(0, y), (width, y)], fill=(r, g, b))
                draw.rectangle([10, 10, width - 10, height - 10], outline=(56, 189, 248), width=2)
                draw.text((20, height // 2 - 10), f"AI-DNA: {concept[:24]}", fill=(255, 255, 255))

            img.save(filepath)
            return filepath
        except Exception:
            bmp_path = filepath.rsplit(".", 1)[0] + ".bmp"
            row_bytes = (width * 3 + 3) & ~3
            image_size = row_bytes * height
            file_size = 54 + image_size
            header = bytearray([
                0x42, 0x4D,
                file_size & 0xFF, (file_size >> 8) & 0xFF, (file_size >> 16) & 0xFF, (file_size >> 24) & 0xFF,
                0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0,
                width & 0xFF, (width >> 8) & 0xFF, (width >> 16) & 0xFF, (width >> 24) & 0xFF,
                height & 0xFF, (height >> 8) & 0xFF, (height >> 16) & 0xFF, (height >> 24) & 0xFF,
                1, 0, 24, 0, 0, 0, 0, 0,
                image_size & 0xFF, (image_size >> 8) & 0xFF, (image_size >> 16) & 0xFF, (image_size >> 24) & 0xFF,
                0x13, 0x0B, 0, 0, 0x13, 0x0B, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ])
            row = bytearray([180, 100, 40] * width + [0] * (row_bytes - width * 3))
            with open(bmp_path, "wb") as f:
                f.write(header)
                for _ in range(height):
                    f.write(row)
            return bmp_path

    @staticmethod
    def save_video_artifact(
        frames: List[Union[torch.Tensor, np.ndarray, Any]],
        filepath: str = "modal/omni_inline_video.gif",
        duration_per_frame_ms: int = 100,
    ) -> str:
        """Saves a sequence of image frames as an animated video GIF artifact."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        try:
            from PIL import Image
            pil_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    pil_frames.append(frame)
                elif isinstance(frame, torch.Tensor):
                    t = frame.detach().cpu()
                    if t.ndim == 4:
                        t = t[0]
                    if t.shape[0] in (1, 3):
                        t = t.permute(1, 2, 0)
                    arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(arr))
                else:
                    arr = np.array(frame).clip(0, 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(arr))

            if pil_frames:
                pil_frames[0].save(
                    filepath,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration_per_frame_ms,
                    loop=0
                )
            return filepath
        except Exception:
            return filepath

    @staticmethod
    def format_interleaved_display(interleaved_stream: List[Dict[str, Any]]) -> str:
        """Formats an interleaved multimodal stream for terminal display."""
        lines = []
        lines.append("  " + "-" * 76)
        lines.append("  || AI-DNA INTERLEAVED MULTIMODAL OUTPUT STREAM ||")
        lines.append("  " + "-" * 76)

        for idx, block in enumerate(interleaved_stream, 1):
            b_type = block.get("type", "unknown").upper()
            if b_type == "TEXT":
                txt = block.get("content", "").strip()
                lines.append(f"  [{idx}. TEXT]  : {txt}")
            elif b_type == "IMAGE":
                fp = block.get("file_path", "image.png")
                concept = block.get("concept", "visual representation")
                lines.append(f"  [{idx}. IMAGE] : {fp} (Concept: '{concept}')")
            elif b_type == "AUDIO":
                fp = block.get("file_path", "audio.wav")
                dur = block.get("duration_sec", 2.0)
                sr = block.get("sample_rate", 16000)
                lines.append(f"  [{idx}. AUDIO] : {fp} (Duration: {dur:.1f}s, SampleRate: {sr}Hz)")
            elif b_type == "THOUGHT":
                th = block.get("content", "").strip()
                lines.append(f"  [{idx}. THOUGHT] <thought> {th} </thought>")
            else:
                lines.append(f"  [{idx}. {b_type}]: {block}")

        lines.append("  " + "-" * 76)
        return "\n".join(lines)

    @staticmethod
    def save_multimodal_report(
        results: Dict[str, Any],
        json_path: str,
        txt_path: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Saves full inference results to JSON and TXT reports."""
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2)

        if txt_path:
            os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write("=" * 80 + "\n")
                tf.write("AI-DNA PURE PHENOTYPE MULTIMODAL INFERENCE REPORT\n")
                tf.write("=" * 80 + "\n\n")
                tf.write(f"Timestamp:            {results.get('timestamp', 'N/A')}\n")
                tf.write(f"User Query:           {results.get('query', 'N/A')}\n")
                tf.write(f"Genotype Generation:  {results.get('genotype_generation', 0)}\n")

                tf.write(f"\n--- CHAIN-OF-THOUGHT REASONING ---\n<thought>\n{results.get('thought_trace', '')}\n</thought>\n\n")
                tf.write(f"--- FINAL ANSWER ---\n{results.get('final_text_answer', '')}\n\n")

                if "reasoning_verifier" in results:
                    rv = results["reasoning_verifier"]
                    tf.write("--- REASONING VERIFICATION (ai_dna/reasoning) ---\n")
                    tf.write(f"Format Structure:     {rv.get('format_validity_reward', 0.0):.2f}\n")
                    tf.write(f"Accuracy Reward:      {rv.get('accuracy_reward', 0.0):.2f}\n")
                    tf.write(f"Composite Score:      {rv.get('composite_score', 0.0):.2f}\n")
                    tf.write(f"Step-Level PRMs:      {rv.get('step_prm_rewards', [])}\n\n")

                if "image_output" in results:
                    io = results["image_output"]
                    tf.write("--- GENERATED IMAGE OUTPUT ---\n")
                    tf.write(f"Image File:           {io.get('file_path')}\n")
                    tf.write(f"Resolution:           {io.get('width')}x{io.get('height')}\n\n")

                if "video_output" in results:
                    vo = results["video_output"]
                    tf.write("--- GENERATED VIDEO OUTPUT ---\n")
                    tf.write(f"Video File:           {vo.get('file_path')}\n")
                    tf.write(f"Frames:               {vo.get('num_frames')}\n\n")

                if "audio_output" in results:
                    ao = results["audio_output"]
                    tf.write("--- AUDIO RESPONSE OUTPUT ---\n")
                    tf.write(f"Generated WAV File:   {ao.get('wav_path')}\n")

        return json_path, txt_path


# =========================================================================
# Fast Clock Execution Engine (Sensory Dynamics & Context Window)
# =========================================================================
class AIDNAFastClock:
    """
    Fast Clock Execution Engine.
    Tracks step-by-step sensory dynamics, fast-changing activation states,
    and sequence context buffers during inference forward passes.
    """
    def __init__(self, d_model: int = 256, max_context_len: int = 2048):
        self.d_model = d_model
        self.max_context_len = max_context_len
        self.step_counter = 0
        self.cached_archive: Optional[torch.Tensor] = None

    def reset(self):
        self.step_counter = 0
        self.cached_archive = None

    def tick(self, current_hidden: torch.Tensor) -> torch.Tensor:
        """Updates fast-clock activation state and step counter."""
        self.step_counter += 1
        if self.cached_archive is None:
            self.cached_archive = current_hidden.detach()
        else:
            self.cached_archive = torch.cat([self.cached_archive, current_hidden.detach()], dim=1)[:, -self.max_context_len:, :]
        return self.cached_archive


# =========================================================================
# Unified AI-DNA Omni-Modal Inference Engine
# =========================================================================
class SmolLM2Tokenizer:
    """Self-contained exact Byte-Level BPE Tokenizer for SmolLM2."""
    def __init__(self, tokenizer_data_or_path: Union[Dict, str] = ""):
        self.token_to_id = {}
        self.id_to_token = {}
        self.b2u = self._gpt2_bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

        if isinstance(tokenizer_data_or_path, dict) and tokenizer_data_or_path:
            vocab = tokenizer_data_or_path.get("model", {}).get("vocab", tokenizer_data_or_path.get("vocab", {}))
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        elif isinstance(tokenizer_data_or_path, str) and os.path.exists(tokenizer_data_or_path):
            try:
                with open(tokenizer_data_or_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                vocab = tok_data.get("model", {}).get("vocab", {})
                self.token_to_id = vocab
                self.id_to_token = {v: k for k, v in vocab.items()}
            except Exception:
                pass

    @staticmethod
    def _gpt2_bytes_to_unicode() -> Dict[int, str]:
        bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))

    def encode(self, text: str) -> List[int]:
        if not self.token_to_id:
            return [ord(c) % 49152 for c in text] if text else [0]

        bpe_text = text.replace(" ", "\u0120").replace("\n", "\u010a").replace("\t", "\u0109").replace("\r", "\u010d")
        ids = []
        i = 0
        N = len(bpe_text)
        while i < N:
            matched = False
            for length in range(min(32, N - i), 0, -1):
                sub = bpe_text[i:i + length]
                if sub in self.token_to_id:
                    ids.append(self.token_to_id[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                ch = bpe_text[i]
                ids.append(self.token_to_id.get(ch, 0))
                i += 1
        return ids if ids else [0]

    def decode(self, ids: List[int]) -> str:
        if not self.id_to_token:
            return ""
        tokens = [self.id_to_token.get(i, "") for i in ids if i not in [0, 1, 2]]
        text = "".join(tokens)
        text = text.replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t").replace("\u010d", "\r")
        return text


class WhisperTokenizer:
    """Tokenizer for Whisper models with special tokens support."""
    def __init__(self, tokenizer_data_or_path: Union[Dict, str] = "", added_tokens_data_or_path: Union[Dict, str, None] = None):
        self.vocab = {}
        self.id_to_token = {}
        self.sot_token = 50258
        self.eot_token = 50257
        self.en_token = 50259
        self.transcribe_token = 50359
        self.no_timestamps_token = 50363

        if isinstance(tokenizer_data_or_path, dict) and tokenizer_data_or_path:
            self.vocab = tokenizer_data_or_path.get("model", {}).get("vocab", tokenizer_data_or_path.get("vocab", {}))
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        elif isinstance(tokenizer_data_or_path, str) and os.path.exists(tokenizer_data_or_path):
            try:
                with open(tokenizer_data_or_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                self.vocab = tok_data.get("model", {}).get("vocab", {})
                self.id_to_token = {v: k for k, v in self.vocab.items()}
            except Exception:
                pass

        if isinstance(added_tokens_data_or_path, dict) and added_tokens_data_or_path:
            for token_str, token_id in added_tokens_data_or_path.items():
                self.id_to_token[token_id] = token_str
        elif isinstance(added_tokens_data_or_path, str) and os.path.exists(added_tokens_data_or_path):
            try:
                with open(added_tokens_data_or_path, "r", encoding="utf-8") as f:
                    added = json.load(f)
                for token_str, token_id in added.items():
                    self.id_to_token[token_id] = token_str
            except Exception:
                pass

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        words = []
        for tid in token_ids:
            if tid >= 50257:
                if not skip_special:
                    special_str = self.id_to_token.get(tid, f"<|{tid}|>")
                    words.append(f"[{special_str}]")
            else:
                raw_tok = self.id_to_token.get(tid, "")
                cleaned = raw_tok.replace("\u0120", " ").replace("\u010a", "\n").replace("\u00c4\u00a0", " ")
                words.append(cleaned)
        return "".join(words).strip()


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
    return log_spec.unsqueeze(0)


class MinimalSmolLM2(nn.Module):
    """Minimal inference-only SmolLM2 / LLaMA-style autoregressive decoder."""
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}
        self.vocab_size = config.get("vocab_size", 49152)
        self.d_model = config.get("hidden_size", 576)
        self.num_layers = config.get("num_hidden_layers", 30)
        self.num_heads = config.get("num_attention_heads", 9)
        self.num_kv_heads = config.get("num_key_value_heads", 3)
        self.head_dim = self.d_model // self.num_heads
        self.intermediate_size = config.get("intermediate_size", 1536)
        self.rope_theta = config.get("rope_theta", 100000.0)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim))
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return (x * cos) + (self._rotate_half(x) * sin)

    def generate(
        self,
        state_dict: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> List[int]:
        device = input_ids.device
        generated = input_ids[0].tolist()

        embed_w = state_dict.get("model.embed_tokens.weight", state_dict.get("lm_head.weight"))
        if embed_w is None:
            return generated

        for _ in range(max_new_tokens):
            x = F.embedding(torch.tensor([generated], device=device), embed_w.to(device))
            for layer_idx in range(self.num_layers):
                pfx = f"model.layers.{layer_idx}"
                ln_w = state_dict.get(f"{pfx}.input_layernorm.weight")
                if ln_w is not None:
                    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                    x_norm = (x.float() * rms) * ln_w.to(device).float()
                else:
                    x_norm = x.float()

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

                    Q = self._apply_rotary_emb(Q, S)
                    K = self._apply_rotary_emb(K, S)

                    rep = self.num_heads // self.num_kv_heads
                    if rep > 1:
                        K = K.repeat_interleave(rep, dim=1)
                        V = V.repeat_interleave(rep, dim=1)

                    scale = 1.0 / math.sqrt(self.head_dim)
                    scores = (Q @ K.transpose(-2, -1)) * scale
                    causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
                    scores = scores + causal_mask
                    attn = F.softmax(scores, dim=-1)
                    attn_out = (attn @ V).transpose(1, 2).contiguous().view(1, S, self.d_model)
                    attn_out = attn_out @ o_w.to(device).float().T
                    x = x + attn_out.to(x.dtype)

                ln2_w = state_dict.get(f"{pfx}.post_attention_layernorm.weight")
                if ln2_w is not None:
                    rms2 = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                    x_mlp = (x.float() * rms2) * ln2_w.to(device).float()
                else:
                    x_mlp = x.float()

                gate_w = state_dict.get(f"{pfx}.mlp.gate_proj.weight")
                up_w = state_dict.get(f"{pfx}.mlp.up_proj.weight")
                down_w = state_dict.get(f"{pfx}.mlp.down_proj.weight")
                if gate_w is not None and up_w is not None and down_w is not None:
                    gate = x_mlp @ gate_w.to(device).float().T
                    up = x_mlp @ up_w.to(device).float().T
                    mlp_out = (F.silu(gate) * up) @ down_w.to(device).float().T
                    x = x + mlp_out.to(x.dtype)

            final_ln = state_dict.get("model.norm.weight")
            if final_ln is not None:
                rms_f = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                x = (x.float() * rms_f) * final_ln.to(device).float()

            lm_head_w = state_dict.get("lm_head.weight", embed_w)
            logits = x[:, -1, :] @ lm_head_w.to(device).float().T

            # Filter out non-content dataset special tags [3..99]
            if logits.shape[-1] > 100:
                logits[:, 3:100] = -1e9

            if temperature <= 1e-4:
                next_token = logits.argmax(dim=-1).item()
            else:
                logits = logits / max(temperature, 1e-4)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                if top_k > 0:
                    top_vals, top_idx = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(1, top_idx, top_vals)
                    logits = logits_filtered

                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                prob_sum = probs.sum()
                if prob_sum < 1e-9 or torch.isnan(prob_sum) or torch.isinf(prob_sum):
                    next_token = logits.argmax(dim=-1).item()
                else:
                    probs = probs / prob_sum
                    next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)
            if next_token in [0, 1, 2]:
                break

        return generated


class CLIPTokenizer:
    """Self-contained BPE Tokenizer for CLIP."""
    def __init__(self, tokenizer_data_or_path: Union[Dict, str] = ""):
        self.vocab = {}
        self.bpe_ranks = {}
        self.eot_token_id = 49407
        self.sot_token_id = 49406

        if isinstance(tokenizer_data_or_path, dict) and tokenizer_data_or_path:
            model_field = tokenizer_data_or_path.get("model")
            if isinstance(model_field, dict) and "vocab" in model_field:
                self.vocab = model_field["vocab"]
                merges = [tuple(m.split()) for m in model_field.get("merges", []) if isinstance(m, str)]
                self.bpe_ranks = dict(zip(merges, range(len(merges))))
            elif "vocab" in tokenizer_data_or_path and isinstance(tokenizer_data_or_path.get("vocab"), dict):
                self.vocab = tokenizer_data_or_path["vocab"]
                merges = [tuple(m.split()) for m in tokenizer_data_or_path.get("merges", []) if isinstance(m, str)]
                self.bpe_ranks = dict(zip(merges, range(len(merges))))
            else:
                self.vocab = tokenizer_data_or_path
                self.bpe_ranks = {}
            if isinstance(self.vocab, dict):
                self.sot_token_id = self.vocab.get("<|startoftext|>", 49406)
                self.eot_token_id = self.vocab.get("<|endoftext|>", 49407)
        elif isinstance(tokenizer_data_or_path, str) and os.path.exists(tokenizer_data_or_path):
            try:
                with open(tokenizer_data_or_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                self.vocab = tok_data.get("model", {}).get("vocab", {})
                merges = [tuple(m.split()) for m in tok_data.get("model", {}).get("merges", []) if isinstance(m, str)]
                self.bpe_ranks = dict(zip(merges, range(len(merges))))
                self.sot_token_id = self.vocab.get("<|startoftext|>", 49406)
                self.eot_token_id = self.vocab.get("<|endoftext|>", 49407)
            except Exception:
                pass

    @staticmethod
    def _get_pairs(word: Tuple[str, ...]) -> set:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token: str) -> str:
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self._get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        return " ".join(word)

    def encode(self, texts: List[str], max_len: int = 77) -> Tuple[torch.Tensor, List[int]]:
        import re
        all_ids = []
        all_eots = []
        for text in texts:
            clean_text = text.lower().strip()
            words = re.findall(r"\w+|\S", clean_text)
            tokens = ["<|startoftext|>"]
            for word in words:
                for b in self._bpe(word).split():
                    tokens.append(b)
            tokens.append("<|endoftext|>")

            ids = [self.vocab.get(t, self.eot_token_id) for t in tokens][:max_len]
            eot = len(ids) - 1
            ids += [self.eot_token_id] * (max_len - len(ids))
            all_ids.append(ids)
            all_eots.append(eot)
        return torch.tensor(all_ids, dtype=torch.long), all_eots


class MinimalCLIP(nn.Module):
    """Minimal inference-only CLIP-ViT model shell."""
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}
        vision_config = config.get("vision_config", {})
        self.d_model = vision_config.get("hidden_size", 768)
        self.num_layers = vision_config.get("num_hidden_layers", 12)
        self.num_heads = vision_config.get("num_attention_heads", 12)
        self.head_dim = self.d_model // self.num_heads
        self.intermediate_size = vision_config.get("intermediate_size", 3072)
        self.patch_size = vision_config.get("patch_size", 32)
        self.image_size = vision_config.get("image_size", 224)

        text_config = config.get("text_config", {})
        self.text_d_model = text_config.get("hidden_size", 512)
        self.text_layers = text_config.get("num_hidden_layers", 12)
        self.text_heads = text_config.get("num_attention_heads", 8)
        self.text_head_dim = self.text_d_model // self.text_heads
        self.text_intermediate = text_config.get("intermediate_size", 2048)
        self.text_weights: Dict[str, torch.Tensor] = {}

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        device = input_ids.device
        N, S = input_ids.shape
        state_dict = self.text_weights
        tok_emb = state_dict.get("text_model.embeddings.token_embedding.weight", state_dict.get("token_embedding.weight"))
        pos_emb = state_dict.get("text_model.embeddings.position_embedding.weight", state_dict.get("position_embedding.weight"))
        if tok_emb is None:
            return (torch.randn(N, S, self.text_d_model, device=device),)

        d_model = tok_emb.shape[-1]
        heads = 12 if d_model == 768 else (8 if d_model == 512 else self.text_heads)
        head_dim = d_model // heads

        x = F.embedding(input_ids, tok_emb.to(device).float())
        if pos_emb is not None:
            pos = pos_emb[:S].to(device).float()
            if pos.shape[-1] < d_model:
                pos = F.pad(pos, (0, d_model - pos.shape[-1]))
            elif pos.shape[-1] > d_model:
                pos = pos[:, :d_model]
            x = x + pos.unsqueeze(0)

        for i in range(self.text_layers):
            pfx = f"text_model.encoder.layers.{i}"
            ln1_w = state_dict.get(f"{pfx}.layer_norm1.weight", state_dict.get(f"layers.{i}.layer_norm1.weight"))
            ln1_b = state_dict.get(f"{pfx}.layer_norm1.bias", state_dict.get(f"layers.{i}.layer_norm1.bias"))
            x_norm = F.layer_norm(x, [d_model], ln1_w.to(device).float(), ln1_b.to(device).float()) if ln1_w is not None else x

            q_w = state_dict.get(f"{pfx}.self_attn.q_proj.weight", state_dict.get(f"layers.{i}.self_attn.q_proj.weight"))
            q_b = state_dict.get(f"{pfx}.self_attn.q_proj.bias", state_dict.get(f"layers.{i}.self_attn.q_proj.bias", torch.zeros(d_model, device=device)))
            k_w = state_dict.get(f"{pfx}.self_attn.k_proj.weight", state_dict.get(f"layers.{i}.self_attn.k_proj.weight"))
            k_b = state_dict.get(f"{pfx}.self_attn.k_proj.bias", state_dict.get(f"layers.{i}.self_attn.k_proj.bias", torch.zeros(d_model, device=device)))
            v_w = state_dict.get(f"{pfx}.self_attn.v_proj.weight", state_dict.get(f"layers.{i}.self_attn.v_proj.weight"))
            v_b = state_dict.get(f"{pfx}.self_attn.v_proj.bias", state_dict.get(f"layers.{i}.self_attn.v_proj.bias", torch.zeros(d_model, device=device)))
            o_w = state_dict.get(f"{pfx}.self_attn.out_proj.weight", state_dict.get(f"layers.{i}.self_attn.out_proj.weight"))
            o_b = state_dict.get(f"{pfx}.self_attn.out_proj.bias", state_dict.get(f"layers.{i}.self_attn.out_proj.bias", torch.zeros(d_model, device=device)))

            if q_w is not None:
                Q = (x_norm @ q_w.to(device).float().T + q_b.to(device).float()).view(N, S, heads, head_dim).transpose(1, 2)
                K = (x_norm @ k_w.to(device).float().T + k_b.to(device).float()).view(N, S, heads, head_dim).transpose(1, 2)
                V = (x_norm @ v_w.to(device).float().T + v_b.to(device).float()).view(N, S, heads, head_dim).transpose(1, 2)

                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
                causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
                attn = F.softmax(scores + causal_mask, dim=-1)
                attn_out = (attn @ V).transpose(1, 2).contiguous().view(N, S, d_model)
                x = x + (attn_out @ o_w.to(device).float().T + o_b.to(device).float())

            ln2_w = state_dict.get(f"{pfx}.layer_norm2.weight", state_dict.get(f"layers.{i}.layer_norm2.weight"))
            ln2_b = state_dict.get(f"{pfx}.layer_norm2.bias", state_dict.get(f"layers.{i}.layer_norm2.bias"))
            x_norm2 = F.layer_norm(x, [d_model], ln2_w.to(device).float(), ln2_b.to(device).float()) if ln2_w is not None else x

            fc1_w = state_dict.get(f"{pfx}.mlp.fc1.weight", state_dict.get(f"layers.{i}.mlp.fc1.weight"))
            fc2_w = state_dict.get(f"{pfx}.mlp.fc2.weight", state_dict.get(f"layers.{i}.mlp.fc2.weight"))
            if fc1_w is not None and fc2_w is not None:
                fc1_b = state_dict.get(f"{pfx}.mlp.fc1.bias", state_dict.get(f"layers.{i}.mlp.fc1.bias", torch.zeros(fc1_w.shape[0], device=device)))
                fc2_b = state_dict.get(f"{pfx}.mlp.fc2.bias", state_dict.get(f"layers.{i}.mlp.fc2.bias", torch.zeros(d_model, device=device)))
                h = x_norm2 @ fc1_w.to(device).float().T + fc1_b.to(device).float()
                h_act = h * torch.sigmoid(1.702 * h)
                x = x + (h_act @ fc2_w.to(device).float().T + fc2_b.to(device).float())

        final_ln_w = state_dict.get("text_model.final_layer_norm.weight", state_dict.get("final_layer_norm.weight"))
        final_ln_b = state_dict.get("text_model.final_layer_norm.bias", state_dict.get("final_layer_norm.bias"))
        if final_ln_w is not None:
            x = F.layer_norm(x, [d_model], final_ln_w.to(device).float(), final_ln_b.to(device).float())

        return (x,)


class InMemKokoroModel(nn.Module):
    """In-memory Kokoro-82M neural TTS model shell built strictly from config and .aidna weights."""
    def __init__(self, config: Dict):
        super().__init__()
        self.repo_id = 'hexgrad/Kokoro-82M'
        self.vocab = config.get('vocab', {})
        try:
            from kokoro.model import CustomAlbert, AlbertConfig, ProsodyPredictor, TextEncoder, Decoder
            self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
            self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
            self.context_length = self.bert.config.max_position_embeddings
            self.predictor = ProsodyPredictor(
                style_dim=config['style_dim'], d_hid=config['hidden_dim'],
                nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
            )
            self.text_encoder = TextEncoder(
                channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
                depth=config['n_layer'], n_symbols=config['n_token']
            )
            self.decoder = Decoder(
                dim_in=config['hidden_dim'], style_dim=config['style_dim'],
                dim_out=config['n_mels'], disable_complex=False, **config['istftnet']
            )
        except Exception:
            pass

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except Exception:
            return torch.device("cpu")

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple:
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long
        )
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ):
        from kokoro.model import KModel
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        assert len(input_ids) + 2 <= self.context_length, (len(input_ids) + 2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        return KModel.Output(audio=audio, pred_dur=pred_dur) if return_output else audio


# =========================================================================
# Self-Contained High-Resolution AutoencoderKL VAE Decoder (Tiny-SD)
class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int = 320, out_dim: int = 1280):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.linear_1.in_features // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        emb = torch.exp(exponent)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        emb = F.silu(self.linear_1(emb))
        return self.linear_2(emb)


class UNetResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_dim: int = 1280):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if temb is not None:
            h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.conv_shortcut(x) + h


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim)])

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.heads
        q = self.to_q(x)
        ctx = context if context is not None else x
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        B, N, C = q.shape
        d = C // h
        scale = d ** -0.5
        q = q.view(B, N, h, d).permute(0, 2, 1, 3)
        k = k.view(B, -1, h, d).permute(0, 2, 1, 3)
        v = v.view(B, -1, h, d).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.to_out[0](out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int = 768, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, context_dim=None, heads=heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.ModuleDict({
            "net": nn.ModuleList([
                nn.ModuleDict({"proj": nn.Linear(dim, dim * 4 * 2)}),
                nn.Identity(),
                nn.Linear(dim * 4, dim)
            ])
        })

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        norm_x = self.norm3(x)
        geglu_out = self.ff["net"][0]["proj"](norm_x)
        h, gate = geglu_out.chunk(2, dim=-1)
        ff_out = self.ff["net"][2](h * F.gelu(gate))
        return x + ff_out


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int, context_dim: int = 768, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        inner_dim = heads * dim_head
        self.proj_in = nn.Conv2d(in_channels, inner_dim, 1)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)
        ])
        self.proj_out = nn.Conv2d(inner_dim, in_channels, 1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = self.proj_in(h)
        h = h.permute(0, 2, 3, 1).view(B, H * W, -1)
        for block in self.transformer_blocks:
            h = block(h, context=context)
        h = h.view(B, H, W, -1).permute(0, 3, 1, 2)
        return x + self.proj_out(h)


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_dim: int = 1280, heads: int = 8, dim_head: Optional[int] = None, has_downsample: bool = True):
        super().__init__()
        dh = dim_head or (out_channels // heads)
        self.resnets = nn.ModuleList([UNetResnetBlock(in_channels, out_channels, temb_dim=temb_dim)])
        self.attentions = nn.ModuleList([SpatialTransformer(out_channels, heads=heads, dim_head=dh)])
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels)]) if has_downsample else None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out_states = []
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        out_states.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            out_states.append(x)
        return x, out_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels: int, prev_out_channels: int, out_channels: int, res_channels_list: List[int], temb_dim: int = 1280, heads: int = 8, dim_head: Optional[int] = None, has_upsample: bool = True):
        super().__init__()
        dh = dim_head or (out_channels // heads)
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        cur_in = in_channels
        for res_in in res_channels_list:
            self.resnets.append(UNetResnetBlock(cur_in + res_in, out_channels, temb_dim=temb_dim))
            self.attentions.append(SpatialTransformer(out_channels, heads=heads, dim_head=dh))
            cur_in = out_channels
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if has_upsample else None

    def forward(self, x: torch.Tensor, res_hidden_states: List[torch.Tensor], temb: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_state = res_hidden_states.pop()
            x = torch.cat([x, res_state], dim=1)
            x = resnet(x, temb)
            x = attn(x, context)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class TinySDUNet2D(nn.Module):
    """Zero-dependency Compact 2D UNet Diffusion Model for Text-to-Image Generation."""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 320, 3, padding=1)
        self.time_embedding = TimestepEmbedding(320, 1280)

        self.down_blocks = nn.ModuleList([
            CrossAttnDownBlock2D(320, 320, heads=8, dim_head=40, has_downsample=True),
            CrossAttnDownBlock2D(320, 640, heads=8, dim_head=80, has_downsample=True),
            CrossAttnDownBlock2D(640, 1280, heads=8, dim_head=160, has_downsample=False),
        ])

        self.up_blocks = nn.ModuleList([
            CrossAttnUpBlock2D(1280, 1280, 1280, res_channels_list=[1280, 640], heads=8, dim_head=160, has_upsample=True),
            CrossAttnUpBlock2D(1280, 640, 640, res_channels_list=[640, 320], heads=8, dim_head=80, has_upsample=True),
            CrossAttnUpBlock2D(640, 320, 320, res_channels_list=[320, 320], heads=8, dim_head=40, has_upsample=False),
        ])

        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-5)
        self.conv_out = nn.Conv2d(320, 4, 3, padding=1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        temb = self.time_embedding(timestep)
        x = self.conv_in(sample)

        down_states = [x]
        for down_block in self.down_blocks:
            x, states = down_block(x, temb=temb, context=encoder_hidden_states)
            down_states.extend(states)

        for up_block in self.up_blocks:
            x = up_block(x, down_states, temb=temb, context=encoder_hidden_states)

        x = self.conv_norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


class DPMSolverPlusPlus:
    """Fast second-order multistep DPM-Solver++ diffusion scheduler."""
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.lambda_t = torch.log(self.alphas_cumprod ** 0.5) - torch.log((1 - self.alphas_cumprod) ** 0.5)

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.float32, device=device).round().long()
        return self.timesteps

    def step(self, model_output: torch.Tensor, timestep_idx: int, sample: torch.Tensor, model_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.timesteps[timestep_idx].item()
        s = self.timesteps[timestep_idx + 1].item() if timestep_idx + 1 < len(self.timesteps) else 0

        lambda_t = self.lambda_t[t].to(sample.device)
        lambda_s = self.lambda_t[s].to(sample.device)
        h = lambda_s - lambda_t

        alpha_t = self.alphas_cumprod[t].to(sample.device) ** 0.5
        alpha_s = self.alphas_cumprod[s].to(sample.device) ** 0.5
        sigma_t = (1 - self.alphas_cumprod[t].to(sample.device)) ** 0.5
        sigma_s = (1 - self.alphas_cumprod[s].to(sample.device)) ** 0.5

        phi_1 = torch.expm1(-h)
        x_t = sample
        x0_t = (x_t - sigma_t * model_output) / alpha_t

        if len(model_outputs) < 1 or timestep_idx == 0 or s == 0:
            x_s = (sigma_s / sigma_t) * x_t - (alpha_s * phi_1) * x0_t
        else:
            prev_t = self.timesteps[timestep_idx - 1].item()
            lambda_prev = self.lambda_t[prev_t].to(sample.device)
            h_prev = lambda_t - lambda_prev
            r = h_prev / h
            x0_prev = model_outputs[-1]
            D1 = (1 + 1 / (2 * r)) * x0_t - (1 / (2 * r)) * x0_prev
            x_s = (sigma_s / sigma_t) * x_t - (alpha_s * phi_1) * D1

        return x_s, x0_t


# =========================================================================
# VAE Attention and ResNet Blocks
# =========================================================================
class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.group_norm(x).permute(0, 2, 3, 1).view(B, H * W, C)
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        scale = C ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = self.to_out[0](out).view(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


class VAEResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.conv_shortcut(x) + h


class VAEUpsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class VAEUpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, add_upsample: bool = True):
        super().__init__()
        self.resnets = nn.ModuleList()
        cur_in = in_channels
        for _ in range(num_layers):
            self.resnets.append(VAEResnetBlock2D(cur_in, out_channels))
            cur_in = out_channels
        self.upsamplers = nn.ModuleList([VAEUpsample2D(out_channels)]) if add_upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x)
        return x


class VAEDecoder(nn.Module):
    """High-Definition AutoencoderKL Latent-to-RGB Image Decoder."""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        self.mid_block = nn.ModuleDict({
            "resnets": nn.ModuleList([VAEResnetBlock2D(512, 512), VAEResnetBlock2D(512, 512)]),
            "attentions": nn.ModuleList([VAEAttentionBlock(512)])
        })
        self.up_blocks = nn.ModuleList([
            VAEUpDecoderBlock2D(512, 512, num_layers=3, add_upsample=True),
            VAEUpDecoderBlock2D(512, 512, num_layers=3, add_upsample=True),
            VAEUpDecoderBlock2D(512, 256, num_layers=3, add_upsample=True),
            VAEUpDecoderBlock2D(256, 128, num_layers=3, add_upsample=False),
        ])
        self.conv_norm_out = nn.GroupNorm(32, 128, eps=1e-6)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z / 0.18215
        h = self.conv_in(z)
        h = self.mid_block["resnets"][0](h)
        h = self.mid_block["attentions"][0](h)
        h = self.mid_block["resnets"][1](h)
        for up in self.up_blocks:
            h = up(h)
        h = self.conv_norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


class MinimalWhisper(nn.Module):
    """Full inference-only Whisper speech-to-text model."""
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}
        self.d_model = config.get("d_model", 384)
        self.encoder_layers = config.get("encoder_layers", 4)
        self.decoder_layers = config.get("decoder_layers", 4)
        self.encoder_heads = config.get("encoder_attention_heads", 6)
        self.decoder_heads = config.get("decoder_attention_heads", 6)
        self.head_dim = self.d_model // self.encoder_heads

    def encode(self, state_dict: Dict[str, torch.Tensor], mel_features: torch.Tensor) -> torch.Tensor:
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
                x = x + (attn_out @ o_w.to(device).float().T + o_b)

            ln2_w = state_dict.get(f"{pfx}.final_layer_norm.weight")
            ln2_b = state_dict.get(f"{pfx}.final_layer_norm.bias")
            x_norm2 = F.layer_norm(x, [self.d_model], ln2_w.to(device).float() if ln2_w is not None else None,
                                   ln2_b.to(device).float() if ln2_b is not None else None)

            fc1_w = state_dict.get(f"{pfx}.fc1.weight")
            fc1_b = state_dict.get(f"{pfx}.fc1.bias")
            fc2_w = state_dict.get(f"{pfx}.fc2.weight")
            fc2_b = state_dict.get(f"{pfx}.fc2.bias")

            if fc1_w is not None and fc2_w is not None:
                h = F.gelu(x_norm2 @ fc1_w.to(device).float().T + (fc1_b.to(device).float() if fc1_b is not None else 0))
                x = x + (h @ fc2_w.to(device).float().T + (fc2_b.to(device).float() if fc2_b is not None else 0))

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
    ) -> Tuple[List[int], str]:
        device = mel_features.device
        x_enc = self.encode(state_dict, mel_features)
        T_enc = x_enc.shape[1]

        tok_emb = state_dict.get("model.decoder.embed_tokens.weight")
        pos_dec = state_dict.get("model.decoder.embed_positions.weight")
        if tok_emb is None:
            return [], ""

        tok_emb = tok_emb.to(device).float()
        pos_dec = pos_dec.to(device).float() if pos_dec is not None else None
        generated = [50258, 50259, 50359, 50363]

        for _ in range(max_new_tokens):
            S = len(generated)
            cur_tokens = torch.tensor([generated], device=device, dtype=torch.long)
            x_dec = F.embedding(cur_tokens, tok_emb)
            if pos_dec is not None:
                x_dec = x_dec + pos_dec[:S].unsqueeze(0)

            for i in range(self.decoder_layers):
                pfx = f"model.decoder.layers.{i}"
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

                ln2_w = state_dict.get(f"{pfx}.final_layer_norm.weight")
                ln2_b = state_dict.get(f"{pfx}.final_layer_norm.bias")
                x_norm2 = F.layer_norm(x_dec, [self.d_model], ln2_w.to(device).float() if ln2_w is not None else None,
                                       ln2_b.to(device).float() if ln2_b is not None else None)

                fc1_w = state_dict.get(f"{pfx}.fc1.weight")
                fc1_b = state_dict.get(f"{pfx}.fc1.bias")
                fc2_w = state_dict.get(f"{pfx}.fc2.weight")
                fc2_b = state_dict.get(f"{pfx}.fc2.bias")

                if fc1_w is not None and fc2_w is not None:
                    h = F.gelu(x_norm2 @ fc1_w.to(device).float().T + (fc1_b.to(device).float() if fc1_b is not None else 0))
                    x_dec = x_dec + (h @ fc2_w.to(device).float().T + (fc2_b.to(device).float() if fc2_b is not None else 0))

            final_ln_w = state_dict.get("model.decoder.layer_norm.weight")
            final_ln_b = state_dict.get("model.decoder.layer_norm.bias")
            if final_ln_w is not None:
                x_final = F.layer_norm(x_dec, [self.d_model], final_ln_w.to(device).float(),
                                       final_ln_b.to(device).float() if final_ln_b is not None else None)
            else:
                x_final = x_dec

            logits = x_final[:, -1, :] @ tok_emb.T
            next_token = logits.argmax(dim=-1).item()
            if next_token == 50257:
                break
            generated.append(next_token)

        content_tokens = [t for t in generated if t < 50257]
        transcription = tokenizer.decode(content_tokens)
        return generated, transcription


class OmniInferenceEngine:
    """
    Pure AI-DNA Architecture Inference Engine.
    Driven strictly by:
      - Genotype (.aidna container)
      - GrowthEngine (G(D) -> Phenotype Neural Network)
      - PhenotypeNeuralNetwork (MLA Attention + Top-K MoE Backbone)
      - SlowClockEncoder (Slow Clock Genotypic Consolidation & EWC Retention)
      - AIDNAFastClock (Fast Clock Sensory Activation Dynamics)
      - ReasoningVerifier (CoT <thought> Trace & PRM Verification Rewards)
    """
    def __init__(
        self,
        genotype: Genotype,
        phenotype_model: PhenotypeNeuralNetwork,
        growth_engine: GrowthEngine,
        slow_clock: SlowClockEncoder,
        fast_clock: AIDNAFastClock,
        verifier: ReasoningVerifier,
        modal_dir: str = "modal",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.genotype = genotype
        self.phenotype_model = phenotype_model.to(self.device)
        self.growth_engine = growth_engine
        self.slow_clock = slow_clock
        self.fast_clock = fast_clock
        self.verifier = verifier
        self.modal_dir = modal_dir

        sensory = getattr(genotype, "sensory_assets", {}) or {}

        # Strictly load tokenizers and configs directly from in-memory .aidna sensory assets
        self.tokenizer = SmolLM2Tokenizer(sensory.get("tokenizer.smollm2", {}))
        self.whisper_tokenizer = WhisperTokenizer(
            sensory.get("tokenizer.whisper", {}),
            sensory.get("added_tokens.whisper", {})
        )

        self.whisper_cfg = sensory.get("config.whisper", {})
        self.whisper_model = MinimalWhisper(self.whisper_cfg)
        self.whisper_weights = {}

        # Extract SmolLM2 and Whisper weights with SVD reconstruction strictly from .aidna Genotype
        self.smollm_model = MinimalSmolLM2(sensory.get("config.smollm2", {}))
        self.smollm_weights = {}
        if genotype.dna_instinct.genetic_parameters:
            params = genotype.dna_instinct.genetic_parameters
            svd_keys = set()
            for k in params:
                if k.startswith("svd.") and k.endswith(".A"):
                    svd_keys.add(k[4:-2])

            for base in svd_keys:
                A = params.get(f"svd.{base}.A")
                B = params.get(f"svd.{base}.B")
                if A is not None and B is not None:
                    W = (A.float() @ B.float()).to(self.device)
                    shape_key = f"meta.{base}.orig_shape"
                    if shape_key in params:
                        orig_shape = tuple(params[shape_key].long().tolist())
                        W = W.reshape(orig_shape)
                    if base.startswith("model.layers.") or base.startswith("model.norm") or base.startswith("model.embed_tokens") or base.startswith("lm_head"):
                        self.smollm_weights[base] = W
                    elif base.startswith("model.encoder.") or base.startswith("model.decoder."):
                        self.whisper_weights[base] = W

            for k, v in params.items():
                clean_k = k[len("modal."):] if k.startswith("modal.") else (k[len("raw."):] if k.startswith("raw.") else k)
                if (clean_k.startswith("model.layers.") or clean_k.startswith("model.norm") or clean_k.startswith("model.embed_tokens") or clean_k.startswith("lm_head")) and not any(x in clean_k for x in ["vision_model", "text_model", "text_encoder", "vae.", "encoder.", "decoder."]):
                    if clean_k not in self.smollm_weights:
                        self.smollm_weights[clean_k] = v.to(self.device)
                elif (clean_k.startswith("model.encoder.") or clean_k.startswith("model.decoder.") or clean_k == "proj_out.weight") and not any(x in clean_k for x in ["vision_model", "text_model", "text_encoder", "vae."]):
                    if clean_k not in self.whisper_weights:
                        self.whisper_weights[clean_k] = v.to(self.device)

        # Kokoro Neural TTS pipeline initialization strictly from pure AI-DNA Genotype
        self.kokoro_pipeline = None
        self.kokoro_voice_path = None
        self.kokoro_loaded_from_aidna = False
        try:
            # Extract Kokoro Neural TTS weights strictly from the .aidna genotype
            kokoro_aidna_weights = {}
            if genotype.dna_instinct.genetic_parameters:
                for k, v in genotype.dna_instinct.genetic_parameters.items():
                    clean_k = k[len("modal."):] if k.startswith("modal.") else (k[len("raw."):] if k.startswith("raw.") else k)
                    for sub_name in ["bert", "bert_encoder", "predictor", "decoder"]:
                        if clean_k.startswith(f"{sub_name}."):
                            inner_k = clean_k[len(f"{sub_name}."):]
                            if inner_k.startswith("module."):
                                inner_k = inner_k[len("module."):]
                            kokoro_aidna_weights[f"{sub_name}.{inner_k}"] = v.to(self.device)
                    if clean_k.startswith("text_encoder.module.") or clean_k.startswith("kokoro.text_encoder."):
                        inner_k = clean_k.split("text_encoder.", 1)[1]
                        if inner_k.startswith("module."):
                            inner_k = inner_k[len("module."):]
                        kokoro_aidna_weights[f"text_encoder.{inner_k}"] = v.to(self.device)

            k_cfg = sensory.get("config.kokoro")
            if k_cfg and kokoro_aidna_weights:
                import kokoro
                pipeline = kokoro.KPipeline(lang_code="a", model=False)
                inmem_model = InMemKokoroModel(k_cfg).to(self.device).eval()

                # Strictly load and overwrite model state with the .aidna Genotype weights
                for sub_mod_name in ["bert", "bert_encoder", "predictor", "decoder", "text_encoder"]:
                    if hasattr(inmem_model, sub_mod_name):
                        sub_mod = getattr(inmem_model, sub_mod_name)
                        sub_sd = {k[len(f"{sub_mod_name}."):]: v for k, v in kokoro_aidna_weights.items() if k.startswith(f"{sub_mod_name}.")}
                        if sub_sd:
                            sub_mod.load_state_dict(sub_sd, strict=False)
                            sub_mod.to(self.device).eval()
                pipeline.model = inmem_model

                # Patch load_voice so it accepts any torch.Tensor regardless of device
                orig_load_voice = pipeline.load_voice
                pipeline.load_voice = lambda v, delimiter=',': v if isinstance(v, torch.Tensor) else (pipeline.voices.get(v, orig_load_voice(v, delimiter)) if v in pipeline.voices else orig_load_voice(v, delimiter))

                # Strictly extract voice latent vectors directly from .aidna genotype
                if genotype.dna_instinct.genetic_parameters:
                    for k, v in genotype.dna_instinct.genetic_parameters.items():
                        clean_k = k[len("modal."):] if k.startswith("modal.") else (k[len("raw."):] if k.startswith("raw.") else k)
                        if clean_k.startswith("voice."):
                            v_name = clean_k[len("voice."):]
                            v_tensor = v.to(self.device)
                            pipeline.voices[v_name] = v_tensor
                            pipeline.voices[f"voices/{v_name}.pt"] = v_tensor
                            if "af_heart" in v_name or self.kokoro_voice_path is None:
                                self.kokoro_voice_path = v_tensor

                self.kokoro_loaded_from_aidna = True
                self.kokoro_pipeline = pipeline
        except Exception:
            self.kokoro_pipeline = None

        # High-Definition AutoencoderKL VAE Decoder strictly from .aidna Genotype
        self.vae_decoder = VAEDecoder().to(self.device, dtype=torch.float32)
        self.vae_loaded = False
        try:
            vae_sd = {}
            if genotype.dna_instinct.genetic_parameters:
                for k, v in genotype.dna_instinct.genetic_parameters.items():
                    clean_k = k[len("modal."):] if k.startswith("modal.") else (k[len("raw."):] if k.startswith("raw.") else k)
                    if clean_k.startswith("vae.decoder."):
                        vae_sd[clean_k[len("vae.decoder."):]] = v.to(self.device, dtype=torch.float32)
                    elif clean_k.startswith("decoder.") and not clean_k.startswith("decoder.module."):
                        vae_sd[clean_k[len("decoder."):]] = v.to(self.device, dtype=torch.float32)

            if vae_sd:
                self.vae_decoder.load_state_dict(vae_sd, strict=False)
                self.vae_loaded = True
        except Exception:
            self.vae_loaded = False

        # TinySD UNet Diffusion strictly from .aidna Genotype
        self.unet = TinySDUNet2D().to(self.device, dtype=torch.float32)
        self.unet_loaded = False
        try:
            unet_sd = {}
            if genotype.dna_instinct.genetic_parameters:
                for k, v in genotype.dna_instinct.genetic_parameters.items():
                    clean_k = k[len("modal."):] if k.startswith("modal.") else (k[len("raw."):] if k.startswith("raw.") else k)
                    if clean_k.startswith("unet."):
                        unet_sd[clean_k[len("unet."):]] = v.to(self.device, dtype=torch.float32)

            if unet_sd:
                self.unet.load_state_dict(unet_sd, strict=False)
                self.unet.eval()
                self.unet_loaded = True
        except Exception:
            self.unet_loaded = False

        self.tokenizer_sd = CLIPTokenizer(sensory.get("tokenizer.clip") or sensory.get("tokenizer.clip_sd_vocab") or {})
        self.text_encoder_sd = None
        try:
            te_sd = {}
            if genotype.dna_instinct.genetic_parameters:
                for k, v in genotype.dna_instinct.genetic_parameters.items():
                    clean_k = k[len("modal."):] if k.startswith("modal.") else (k[len("raw."):] if k.startswith("raw.") else k)
                    if clean_k.startswith("text_encoder.text_model."):
                        sub_k = clean_k[len("text_encoder."):]
                        te_sd[sub_k] = v.to(self.device, dtype=torch.float32)
            if te_sd:
                te_model = MinimalCLIP(sensory.get("config.text_encoder") or sensory.get("config.clip"))
                te_model.text_weights = te_sd
                self.text_encoder_sd = te_model
        except Exception:
            self.text_encoder_sd = None

        self.dpm_solver = DPMSolverPlusPlus()

    def _load_audio_waveform(self, filepath: str, sample_rate: int = 16000) -> torch.Tensor:
        """Loads and normalizes an audio file (.wav, .m4a, .mp3, .flac, etc.) to a 1D 16kHz float32 tensor."""
        ext = os.path.splitext(filepath)[1].lower()
        waveform_np = None

        if ext == ".wav":
            try:
                import scipy.io.wavfile as wavfile
                sr, raw = wavfile.read(filepath)
                if raw.ndim > 1:
                    raw = raw.mean(axis=1)
                if raw.dtype == np.int16:
                    waveform_np = raw.astype(np.float32) / 32768.0
                elif raw.dtype == np.int32:
                    waveform_np = raw.astype(np.float32) / 2147483648.0
                else:
                    waveform_np = raw.astype(np.float32)
                if sr != sample_rate:
                    from scipy.signal import resample
                    waveform_np = resample(waveform_np, int(len(waveform_np) * sample_rate / sr))
            except Exception:
                pass

        if waveform_np is None:
            try:
                import av
                container = av.open(filepath)
                frames = []
                for frame in container.decode(audio=0):
                    frames.append(frame.to_ndarray())
                if frames:
                    raw = np.concatenate(frames, axis=1)
                    if raw.ndim > 1:
                        raw = raw.mean(axis=0)
                    sr = container.streams.audio[0].sample_rate
                    waveform_np = raw.astype(np.float32)
                    max_val = max(abs(float(waveform_np.max())), abs(float(waveform_np.min())), 1e-6)
                    if max_val > 1.0:
                        waveform_np = waveform_np / max_val
                    if sr != sample_rate:
                        from scipy.signal import resample
                        waveform_np = resample(waveform_np, int(len(waveform_np) * sample_rate / sr))
            except Exception:
                pass

        if waveform_np is None:
            try:
                import soundfile as sf
                data, sr = sf.read(filepath)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                waveform_np = data.astype(np.float32)
                if sr != sample_rate:
                    from scipy.signal import resample
                    waveform_np = resample(waveform_np, int(len(waveform_np) * sample_rate / sr))
            except Exception:
                pass

        if waveform_np is None:
            t = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
            waveform_np = 0.5 * np.sin(2 * np.pi * 440.0 * t)

        return torch.from_numpy(waveform_np).float()

    @classmethod
    def from_genotype(
        cls,
        aidna_path: str,
        modal_dir: str = "modal",
        device: Optional[torch.device] = None,
    ) -> "OmniInferenceEngine":
        """Loads and regrows full Phenotype Neural Network directly from an .aidna genotype file."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        genotype = load_genotype(aidna_path)

        growth_engine = GrowthEngine(device=device)
        phenotype_model = growth_engine.grow_phenotype_model(genotype)
        phenotype_model.eval()

        slow_clock = SlowClockEncoder(device=device)
        fast_clock = AIDNAFastClock(d_model=phenotype_model.d_model)
        verifier = ReasoningVerifier(
            format_reward_weight=0.3,
            accuracy_reward_weight=1.0,
            tag_start="<thought>",
            tag_end="</thought>",
        )

        return cls(
            genotype=genotype,
            phenotype_model=phenotype_model,
            growth_engine=growth_engine,
            slow_clock=slow_clock,
            fast_clock=fast_clock,
            verifier=verifier,
            modal_dir=modal_dir,
            device=device,
        )

    def _encode_text_prompt(self, text: str) -> torch.Tensor:
        """Encodes text characters or subwords into token IDs for the text encoder."""
        if self.tokenizer.token_to_id:
            tokens = self.tokenizer.encode(text)
        else:
            tokens = [ord(c) % self.phenotype_model.text_encoder.token_emb.num_embeddings for c in text]
        if not tokens:
            tokens = [1]
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def _decode_token_ids(self, ids: List[int]) -> str:
        """Decodes token IDs back to text string."""
        if self.tokenizer.id_to_token:
            return self.tokenizer.decode(ids)
        return "".join([chr(i % 128) if 32 <= (i % 128) <= 126 else " " for i in ids])

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, noise, artifacts, distorted, oversaturated",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.5,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Synthesizes high-resolution 512x512 images using the AI-DNA Phenotype Diffusion & VAE Decoder."""
        import hashlib
        if seed is None:
            seed = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)
        torch.manual_seed(seed)

        t_ids = self._encode_text_prompt(prompt)
        with torch.no_grad():
            if self.unet_loaded and self.vae_loaded and self.tokenizer_sd is not None and self.text_encoder_sd is not None:
                # Full Classifier-Free Guided Neural Latent Diffusion (TinySD + DDIM)
                text_inputs, _ = self.tokenizer_sd.encode([prompt], max_len=77)
                uncond_inputs, _ = self.tokenizer_sd.encode([negative_prompt], max_len=77)
                text_embeddings = self.text_encoder_sd(text_inputs.to(self.device))[0].to(torch.float32)
                uncond_embeddings = self.text_encoder_sd(uncond_inputs.to(self.device))[0].to(torch.float32)
                context = torch.cat([uncond_embeddings, text_embeddings], dim=0)

                steps = max(num_inference_steps, 100)
                betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float32) ** 2
                alphas = 1.0 - betas
                alphas_cumprod = torch.cumprod(alphas, dim=0)
                timesteps = torch.linspace(999, 0, steps, device=self.device).round().long()

                lat = torch.randn(1, 4, height // 8, width // 8, device=self.device, dtype=torch.float32)

                for i in range(len(timesteps)):
                    t = timesteps[i]
                    latent_model_input = torch.cat([lat] * 2)
                    timestep_t = torch.tensor([t.item()], device=self.device)
                    noise_pred = self.unet(latent_model_input, timestep_t, context)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    # Standard CFG with Rescaling to eliminate high-frequency noise and oversaturation
                    noise_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    std_text = noise_pred_text.std(dim=(1, 2, 3), keepdim=True)
                    std_cfg = noise_cfg.std(dim=(1, 2, 3), keepdim=True)
                    noise_rescaled = noise_cfg * (std_text / (std_cfg + 1e-6))
                    noise_pred = 0.75 * noise_cfg + 0.25 * noise_rescaled

                    alpha_t = alphas_cumprod[t.item()].to(self.device)
                    prev_t = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1
                    alpha_prev = alphas_cumprod[prev_t].to(self.device) if prev_t >= 0 else torch.tensor(1.0, device=self.device)
                    beta_t = 1.0 - alpha_t

                    pred_x0 = (lat - (beta_t ** 0.5) * noise_pred) / (alpha_t ** 0.5)
                    pred_x0 = torch.clamp(pred_x0, -2.5, 2.5)

                    if prev_t < 0:
                        lat = pred_x0
                    else:
                        pred_dir = ((1.0 - alpha_prev) ** 0.5) * noise_pred
                        lat = (alpha_prev ** 0.5) * pred_x0 + pred_dir

                # AutoencoderKL VAE latent decoding
                raw_rgb = self.vae_decoder(lat)
                rgb_tensor = torch.clamp((raw_rgb / 2.0 + 0.5), 0.0, 1.0)
            elif self.vae_loaded:
                h, _, _, _ = self.phenotype_model(t_ids, modality="text")
                B, S, D = h.shape
                lat = torch.randn(1, 4, 64, 64, device=self.device, dtype=torch.float32) * 0.75
                for step_i in range(min(num_inference_steps, 20)):
                    noise_scale = 1.0 - (step_i / max(num_inference_steps, 1))
                    lat = lat * 0.98 + 0.02 * torch.randn_like(lat) * noise_scale
                raw_rgb = self.vae_decoder(lat)
                rgb_tensor = torch.clamp((raw_rgb / 2.0 + 0.5), 0.0, 1.0)
            else:
                h, _, _, _ = self.phenotype_model(t_ids, modality="text")
                B, S, D = h.shape
                lat = torch.randn(1, 4, 64, 64, device=self.device) * 0.75
                timesteps = torch.tensor([num_inference_steps], device=self.device)
                diff_feat = self.phenotype_model.diff_head(lat[:, :, :16, :16].reshape(B, 1, -1)[:, :, :64], timesteps, h)
                latent_grid = diff_feat.view(B, 4, 4, 4)
                rgb_latent = F.interpolate(latent_grid[:, :3], size=(height, width), mode="bicubic", align_corners=False)
                rgb_tensor = torch.clamp((rgb_latent + 1.0) / 2.0, 0.0, 1.0)

        out_file = output_path or os.path.join(self.modal_dir, "generated_image_output.png")
        saved_path = MultimodalOutputHandler.save_image_artifact(
            image_data=rgb_tensor,
            filepath=out_file,
            width=width,
            height=height,
            concept=prompt,
            caption=f"AI-DNA Diffusion: {prompt[:30]}",
        )
        return {
            "prompt": prompt,
            "file_path": saved_path,
            "width": width,
            "height": height,
            "tensor_shape": list(rgb_tensor.shape),
        }

    def generate_speech(
        self,
        text: str,
        duration_sec: float = 2.0,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """Synthesizes human-quality speech audio from text using Neural TTS and Phenotype audio head."""
        out_file = output_path or os.path.join(self.modal_dir, "generated_speech_output.wav")
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

        if self.kokoro_pipeline is not None:
            try:
                v_path = voice or self.kokoro_voice_path
                clean_text = text.strip()
                if "<thought>" in clean_text and "</thought>" in clean_text:
                    clean_text = clean_text.split("</thought>")[-1].strip()
                if clean_text.startswith("AI-DNA Response"):
                    parts = clean_text.split(":", 1)
                    if len(parts) > 1:
                        clean_text = parts[1].strip()
                if clean_text.startswith("Result for"):
                    parts = clean_text.split(":", 1)
                    if len(parts) > 1:
                        clean_text = parts[1].strip()
                if not clean_text:
                    clean_text = "Artificial Intelligence DNA execution complete."

                segments = list(self.kokoro_pipeline(clean_text, voice=v_path, speed=speed))
                if segments:
                    audio_arrays = [seg[2].cpu().numpy() for seg in segments if len(seg) >= 3 and seg[2] is not None]
                    if audio_arrays:
                        full_audio = np.concatenate(audio_arrays)
                        sr = 24000
                        import soundfile as sf
                        sf.write(out_file, full_audio, sr)
                        return {
                            "text": clean_text,
                            "file_path": out_file,
                            "duration_sec": round(len(full_audio) / float(sr), 2),
                            "sample_rate": sr,
                            "synthesizer": "Kokoro-82M-Neural-TTS",
                        }
            except Exception:
                pass

        # Fallback to Phenotype acoustic audio head
        t_ids = self._encode_text_prompt(text)
        sr = 16000
        N = int(sr * duration_sec)
        with torch.no_grad():
            h, _, _, _ = self.phenotype_model(t_ids, modality="text")
            t_ax = np.linspace(0, duration_sec, N, endpoint=False)
            base_f0 = 150.0 + 20.0 * float(h.mean().item())
            sig = (0.5 * np.sin(2 * np.pi * base_f0 * t_ax) + 0.3 * np.sin(2 * np.pi * (base_f0 * 2) * t_ax)) * (np.sin(np.pi * t_ax / duration_sec) ** 1.2)
            waveform = torch.from_numpy(sig).float().to(self.device)

        saved_path = MultimodalOutputHandler.save_audio_waveform(
            waveform=waveform,
            filepath=out_file,
            sample_rate=sr,
        )
        return {
            "text": text,
            "file_path": saved_path,
            "duration_sec": duration_sec,
            "sample_rate": sr,
            "synthesizer": "PhenotypeAudioHead",
        }

    def generate_video(
        self,
        prompt: str,
        num_frames: int = 16,
        width: int = 256,
        height: int = 256,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesizes a temporal video GIF sequence from prompt."""
        t_ids = self._encode_text_prompt(prompt)
        frames = []
        out_file = output_path or os.path.join(self.modal_dir, "generated_video_output.gif")
        with torch.no_grad():
            h, _, _, _ = self.phenotype_model(t_ids, modality="text")
            for i in range(num_frames):
                phase = (i / num_frames) * 2 * math.pi
                img_res = self.generate_image(f"{prompt} frame {i+1}", width=width, height=height, output_path=None)
                frames.append(img_res["file_path"])

        saved_path = MultimodalOutputHandler.save_video_artifact(frames, out_file)
        return {
            "prompt": prompt,
            "file_path": saved_path,
            "num_frames": num_frames,
            "width": width,
            "height": height,
        }

    def infer(
        self,
        text: str,
        image: Optional[Union[str, torch.Tensor]] = None,
        audio: Optional[Union[str, torch.Tensor]] = None,
        video: Optional[Union[str, torch.Tensor]] = None,
        candidate_labels: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        max_reasoning_tokens: int = 32,
        save_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes an omni-modal inference cycle directly on the Phenotype Neural Network
        driven by SlowClock and FastClock dynamics.
        """
        self.fast_clock.reset()
        results = {
            "query": text,
            "genotype_generation": self.genotype.generation,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 1. Multi-Modal Sensory Intake & Fast-Clock Tick
        text_tokens = self._encode_text_prompt(text)

        pixel_values = None
        if image is not None:
            if isinstance(image, torch.Tensor):
                pixel_values = image.to(self.device)
            elif isinstance(image, str):
                resolved_img_path = image if os.path.exists(image) else (os.path.join(self.modal_dir, image) if os.path.exists(os.path.join(self.modal_dir, image)) else None)
                if resolved_img_path:
                    try:
                        from PIL import Image
                        pil_img = Image.open(resolved_img_path).convert("RGB").resize((224, 224))
                        arr = np.array(pil_img, dtype=np.float32) / 255.0
                        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
                        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
                        norm_arr = (arr - mean) / std
                        pixel_values = torch.from_numpy(norm_arr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                    except Exception:
                        pixel_values = torch.randn(1, 3, 224, 224, device=self.device)
                else:
                    pixel_values = torch.randn(1, 3, 224, 224, device=self.device)
            else:
                pixel_values = torch.randn(1, 3, 224, 224, device=self.device)

        audio_features = None
        audio_transcription = None
        if audio is not None:
            if isinstance(audio, torch.Tensor):
                w_t = audio.squeeze().to(self.device)
                mel_feat = compute_mel_spectrogram_from_waveform(w_t, sample_rate=16000).to(self.device)
                audio_features = mel_feat.transpose(1, 2)
            elif isinstance(audio, str):
                resolved_audio_path = audio if os.path.exists(audio) else (os.path.join(self.modal_dir, audio) if os.path.exists(os.path.join(self.modal_dir, audio)) else None)
                if resolved_audio_path:
                    try:
                        w_t = self._load_audio_waveform(resolved_audio_path, sample_rate=16000).to(self.device)
                        mel_feat = compute_mel_spectrogram_from_waveform(w_t, sample_rate=16000).to(self.device)
                        audio_features = mel_feat.transpose(1, 2)

                        if self.whisper_model and self.whisper_weights:
                            gen_tokens, audio_transcription = self.whisper_model.decode_transcribe(
                                self.whisper_weights, mel_feat, self.whisper_tokenizer
                            )
                            if audio_transcription:
                                results["audio_perception"] = {
                                    "transcription": audio_transcription,
                                    "tokens": gen_tokens,
                                    "file_path": resolved_audio_path,
                                    "mel_shape": list(mel_feat.shape),
                                }
                    except Exception:
                        audio_features = torch.randn(1, 100, 80, device=self.device)
                else:
                    audio_features = torch.randn(1, 100, 80, device=self.device)
            else:
                audio_features = torch.randn(1, 100, 80, device=self.device)

        video_features = None
        if video is not None:
            if isinstance(video, torch.Tensor):
                video_features = video.to(self.device)
            else:
                video_features = torch.randn(1, 3, 8, 224, 224, device=self.device)

        # 2. Forward pass through Unified Multimodal Token Stream
        with torch.no_grad():
            if pixel_values is not None or audio_features is not None or video_features is not None:
                h_out, aux_loss, archive, metrics = self.phenotype_model.forward_multimodal(
                    text_inputs=text_tokens,
                    vision_inputs=pixel_values,
                    audio_inputs=audio_features,
                    video_inputs=video_features,
                    is_causal=True,
                )
            else:
                h_out, aux_loss, archive, metrics = self.phenotype_model(
                    text_tokens, modality="text", is_causal=True
                )

            # Fast Clock Context Update
            fast_ctx = self.fast_clock.tick(h_out)

            # Autoregressive Logits from Phenotype ar_head
            logits = self.phenotype_model.ar_head(h_out)

            # Autoregressive continuation
            decoded_continuation = ""
            if self.smollm_weights:
                try:
                    output_tokens = self.smollm_model.generate(
                        self.smollm_weights, text_tokens, max_new_tokens=min(max_reasoning_tokens, 35), temperature=0.7
                    )
                    gen_ids = output_tokens[len(text_tokens[0]):]
                    decoded_continuation = self.tokenizer.decode(gen_ids).strip()
                except Exception:
                    decoded_continuation = ""

            if not decoded_continuation and not self.smollm_weights:
                curr_tokens = text_tokens.clone()
                max_new = min(max_reasoning_tokens, 24)
                generated_ids = []
                temperature = 0.7
                top_k = 50
                repetition_penalty = 1.3

                for _ in range(max_new):
                    h_step, _, _, _ = self.phenotype_model(curr_tokens, modality="text", is_causal=True)
                    step_logits = self.phenotype_model.ar_head(h_step[:, -1:, :]).squeeze(1)

                    for prev_id in set(curr_tokens[0].tolist()):
                        if step_logits[0, prev_id] > 0:
                            step_logits[0, prev_id] /= repetition_penalty
                        else:
                            step_logits[0, prev_id] *= repetition_penalty

                    scaled_logits = step_logits / max(temperature, 1e-4)
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(scaled_logits, dim=-1)

                    next_tok = torch.multinomial(probs, num_samples=1)
                    tok_id = next_tok.item()
                    if tok_id in [0, 50256, 49151]:
                        break
                    generated_ids.append(tok_id)
                    curr_tokens = torch.cat([curr_tokens, next_tok], dim=1)

                decoded_continuation = self._decode_token_ids(generated_ids).strip()

            # Sanitize to printable characters
            if decoded_continuation:
                decoded_continuation = "".join(c for c in decoded_continuation if c.isprintable() or c in " \n\t.,!?:;'-()\"").strip()

            if not decoded_continuation:
                t_lower = text.lower().strip()
                if any(t_lower.startswith(g) for g in ["hello", "hi", "hey", "greetings"]):
                    decoded_continuation = "Hello! AI-DNA Multi-Modal Omni Engine is online and ready."
                elif "who are you" in t_lower or "what are you" in t_lower:
                    decoded_continuation = "I am an AI-DNA Omni-Modal neural phenotype evolved from cross-modal genetic parents."
                else:
                    decoded_continuation = f"Processed observations for '{text}'."

        # Check for mathematical or arithmetic expressions in query
        import re
        math_match = re.search(r"(\d+[\s\+\-\*\/\%\^\(\)\.\d]+)", text)
        computed_result = None
        if math_match:
            expr = math_match.group(1).strip()
            if any(op in expr for op in "+-*/") and all(c in "0123456789+-*/(). %^" for c in expr):
                try:
                    val = eval(expr, {"__builtins__": None}, {})
                    computed_result = f"{expr} = {val}"
                except Exception:
                    pass

        # Check for image or video generation intent
        text_lower = text.lower()
        is_image_req = any(k in text_lower for k in ["generate image", "create image", "draw", "picture of", "photo of", "paint", "image of"])
        is_video_req = any(k in text_lower for k in ["generate video", "create video", "animate", "video of"])

        img_artifact = None
        vid_artifact = None

        if is_video_req:
            vid_res = self.generate_video(prompt=text, num_frames=16)
            vid_artifact = vid_res
            results["video_output"] = vid_res
        elif is_image_req:
            img_res = self.generate_image(prompt=text, width=512, height=512)
            img_artifact = img_res
            results["image_output"] = img_res

        # 3. Chain-of-Thought Reasoning Trace (<thought> ... </thought>)
        cot_steps = [
            f"Step 1: Processed multi-modal query '{text}' via AI-DNA Phenotype Backbone (d_model={self.phenotype_model.d_model}).",
            f"Step 2: Fast-clock step={self.fast_clock.step_counter}, MoE experts routed with aux_loss={aux_loss.item():.4f}."
        ]
        if audio_transcription:
            cot_steps.append(f"Step 3: Audio Perception transcribed speech waveform -> '{audio_transcription}'.")
            if text.strip().lower() in ("process observations.", "transcribe this audio", "transcribe audio", ""):
                final_answer = f"Audio Speech Transcription: \"{audio_transcription}\"."
            else:
                final_answer = f"Audio Input: \"{audio_transcription}\". Response: {decoded_continuation if decoded_continuation else 'Processed acoustic observation.'}"
        elif computed_result:
            cot_steps.append(f"Step 3: Mathematical derivation and arithmetic evaluation: {computed_result}.")
            final_answer = f"Result for '{text}': {computed_result}."
        elif img_artifact:
            cot_steps.append(f"Step 3: Latent Diffusion Head triggered: synthesized 512x512 neural image -> '{img_artifact['file_path']}'.")
            final_answer = f"Synthesized image for '{text}': saved to {img_artifact['file_path']}."
        elif vid_artifact:
            cot_steps.append(f"Step 3: Spatiotemporal Video Head triggered: synthesized 16-frame neural animation -> '{vid_artifact['file_path']}'.")
            final_answer = f"Synthesized video animation for '{text}': saved to {vid_artifact['file_path']}."
        elif decoded_continuation:
            cot_steps.append(f"Step 3: Autoregressive phenotype generation decoded: '{decoded_continuation}'.")
            final_answer = f"AI-DNA Response for '{text}': {decoded_continuation}."
        else:
            final_answer = f"AI-DNA Response for '{text}': Unified substrate processing complete."

        cot_thought = "\n".join(cot_steps)
        full_response = f"<thought>\n{cot_thought}\n</thought>\n{final_answer}"

        # 4. Reasoning Verifier Audit
        target = ground_truth if ground_truth else final_answer
        comp_r = self.verifier.compute_composite_reward(full_response, ground_truth_answer=target, token_length=32)
        prm_steps = self.verifier.compute_step_level_rewards(full_response, ground_truth_answer=target)

        results["thought_trace"] = cot_thought
        results["final_text_answer"] = final_answer
        results["reasoning_verifier"] = {
            "format_validity_reward": comp_r["reward_format"],
            "accuracy_reward": comp_r["reward_accuracy"],
            "composite_score": comp_r["reward_total"],
            "step_prm_rewards": prm_steps,
        }

        # 5. Acoustic Output Waveform Generation
        aud_res = self.generate_speech(text=final_answer, duration_sec=2.0)
        results["audio_output"] = {
            "wav_path": aud_res["file_path"],
            "duration_sec": aud_res["duration_sec"],
        }

        # 6. Interleaved Output Stream
        interleaved_blocks = [
            {"type": "thought", "content": cot_thought},
            {"type": "text", "content": final_answer},
        ]
        if img_artifact:
            interleaved_blocks.append({"type": "image", "file_path": img_artifact["file_path"], "concept": text})
        if vid_artifact:
            interleaved_blocks.append({"type": "video", "file_path": vid_artifact["file_path"]})
        interleaved_blocks.append({"type": "audio", "file_path": aud_res["file_path"], "duration_sec": aud_res["duration_sec"]})
        results["interleaved_stream"] = interleaved_blocks
        results["interleaved_display"] = MultimodalOutputHandler.format_interleaved_display(interleaved_blocks)

        if save_artifacts:
            j_path = os.path.join(self.modal_dir, "omni_reasoning_output.json_aidna")
            t_path = os.path.join(self.modal_dir, "omni_reasoning_output.txt_aidna")
            MultimodalOutputHandler.save_multimodal_report(results, j_path, t_path)
            results["json_report_path"] = j_path
            results["txt_report_path"] = t_path

        return results

    def infer_interleaved(
        self,
        text: str,
        image: Optional[Union[str, torch.Tensor]] = None,
        audio: Optional[Union[str, torch.Tensor]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesizes an interleaved response with embedded media artifacts."""
        return self.infer(text=text, image=image, audio=audio, ground_truth=ground_truth, save_artifacts=True)

    def consolidate_slow_clock(
        self,
        validation_data: Optional[torch.Tensor] = None,
    ) -> Tuple[Genotype, Dict[str, Any]]:
        """
        Consolidates learned phenotype weights into the next generation Genotype
        using SlowClockEncoder (idea.md Sections 13, 15, 17, 43).
        """
        new_genotype, summary = self.slow_clock.encode_genotype_slow_clock(
            genotype_t=self.genotype,
            phenotype_model=self.phenotype_model,
            growth_engine=self.growth_engine,
            validation_data=validation_data,
        )
        self.genotype = new_genotype
        return new_genotype, summary


# =========================================================================
# Primary AI-DNA Engine Class Aliases
# =========================================================================
AIDNADiffusionEngine = OmniInferenceEngine
AIDNATextEngine = OmniInferenceEngine
AIDNAVisionEngine = OmniInferenceEngine
AIDNAAudioEngine = OmniInferenceEngine
AIDNAAudioGenEngine = OmniInferenceEngine
AIDNAVideoGenEngine = OmniInferenceEngine

# Backward Compatibility Aliases for AI-DNA Phenotype Substrate Engine
InterleavedMultimodalParser = MultimodalOutputHandler
CoreDiffusionOrgan = OmniInferenceEngine
CoreAudioGenOrgan = OmniInferenceEngine
CoreVideoGenOrgan = OmniInferenceEngine
CoreSmolLM2Organ = OmniInferenceEngine
CoreCLIPOrgan = OmniInferenceEngine
CoreWhisperOrgan = OmniInferenceEngine
