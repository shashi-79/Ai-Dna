"""
AI-DNA Live Interactive Inference Comparison Tool.

Compares inference outputs of:
  1. Original open-source model (raw weights from modal/)
  2. AI-DNA Individual Parent (.aidna reconstructed)
  3. AI-DNA Fused Omni-Child (.aidna reconstructed)

For all three modalities: Text, Vision, Audio.
No HuggingFace transformers required — uses self-contained minimal model shells.

Usage:
  .venv\\Scripts\\python.exe inference_compare.py                   # Interactive menu
  .venv\\Scripts\\python.exe inference_compare.py --mode text       # Text only
  .venv\\Scripts\\python.exe inference_compare.py --mode vision     # Vision only
  .venv\\Scripts\\python.exe inference_compare.py --mode audio      # Audio only
  .venv\\Scripts\\python.exe inference_compare.py --mode all        # All three sequential
"""

import os
import sys
import time
import json
import struct
import glob
import math
import argparse
from typing import Dict, Any, Tuple, Optional, List

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

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ai_dna.dna.serialization import load_genotype
from ai_dna.reasoning.verifier import ReasoningVerifier
from ai_dna.models.shells import (
    load_safetensors_file,
    load_model_weights,
    load_config,
    reconstruct_weights_from_aidna,
    reconstruct_weights_only,
    MinimalRMSNorm,
    SmolLM2Tokenizer,
    MinimalSmolLM2,
    CLIPTokenizer,
    MinimalCLIP,
    WhisperTokenizer,
    compute_mel_spectrogram_from_waveform,
    MinimalWhisper,
    compute_weight_diff_metrics,
    compute_output_similarity,
)


# =========================================================================
# Interactive Comparison Functions
# =========================================================================
def compare_text(modal_dir: str, device: torch.device):
    """Interactive text generation comparison: Original vs AI-DNA Parent vs Fused Child."""
    print("\n" + "=" * 80)
    print("  TEXT INFERENCE COMPARISON: SmolLM2-135M")
    print("  Original vs AI-DNA Parent vs AI-DNA Fused Child")
    print("=" * 80)

    text_dir = os.path.join(modal_dir, "text_model")
    config = load_config(text_dir)

    # Load tokenizer vocab for decoding
    vocab_path = os.path.join(text_dir, "tokenizer.json")
    id_to_token = {}
    token_to_id = {}
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
            vocab = tok_data.get("model", {}).get("vocab", {})
            for token, idx in vocab.items():
                id_to_token[idx] = token
                token_to_id[token] = idx
    print(f"  [+] Loaded vocabulary: {len(id_to_token)} tokens")

    # 1. Load original weights
    print("  [1/3] Loading Original SmolLM2-135M weights...")
    t0 = time.time()
    orig_weights = load_model_weights(text_dir, device)
    print(f"        Loaded {len(orig_weights)} tensors in {time.time()-t0:.2f}s")

    # 2. Reconstruct from AI-DNA parent
    parent_path = os.path.join(modal_dir, "parent_text.aidna")
    print(f"  [2/3] Reconstructing from AI-DNA Parent: {parent_path}")
    t0 = time.time()
    aidna_weights = reconstruct_weights_from_aidna(parent_path, device=device)
    print(f"        Reconstructed {len(aidna_weights)} tensors in {time.time()-t0:.2f}s")

    # 3. Reconstruct from Fused Child
    fused_path = os.path.join(modal_dir, "fused_omni_child.aidna")
    print(f"  [3/3] Reconstructing from Fused Child: {fused_path}")
    t0 = time.time()
    fused_weights = reconstruct_weights_from_aidna(fused_path, key_filter="model.", device=device)
    # Also grab lm_head if present
    fused_weights_all = reconstruct_weights_from_aidna(fused_path, device=device)
    for k, v in fused_weights_all.items():
        if k.startswith("lm_head") or k.startswith("model."):
            fused_weights[k] = v
    print(f"        Reconstructed {len(fused_weights)} text-related tensors in {time.time()-t0:.2f}s")

    # Weight fidelity metrics
    print("\n  --- Weight Reconstruction Fidelity ---")
    parent_metrics = compute_weight_diff_metrics(orig_weights, aidna_weights)
    fused_metrics = compute_weight_diff_metrics(orig_weights, fused_weights)
    print(f"  AI-DNA Parent:  Keys={parent_metrics['shared_keys']}, MaxDiff={parent_metrics['max_abs_diff']:.2e}, "
          f"MeanDiff={parent_metrics['mean_abs_diff']:.2e}, CosSim={parent_metrics['cosine_sim']:.6f}")
    print(f"  AI-DNA Fused:   Keys={fused_metrics['shared_keys']}, MaxDiff={fused_metrics['max_abs_diff']:.2e}, "
          f"MeanDiff={fused_metrics['mean_abs_diff']:.2e}, CosSim={fused_metrics['cosine_sim']:.6f}")

    # Create model shell
    model = MinimalSmolLM2(config)

    def simple_encode(text: str) -> List[int]:
        """Ultra-simple whitespace tokenizer fallback."""
        ids = []
        # Try character-level encoding with vocab
        for ch in text:
            if ch in token_to_id:
                ids.append(token_to_id[ch])
            else:
                # Try byte fallback
                for byte_val in ch.encode("utf-8"):
                    byte_token = f"<0x{byte_val:02X}>"
                    if byte_token in token_to_id:
                        ids.append(token_to_id[byte_token])
                    else:
                        ids.append(byte_val % len(id_to_token) if id_to_token else 0)
        return ids if ids else [0]

    def simple_decode(ids: List[int]) -> str:
        tokens = []
        for i in ids:
            t = id_to_token.get(i, f"[{i}]")
            # Clean up common BPE artifacts
            t = t.replace("\u0120", " ").replace("\u010a", "\n").replace("\u00c4\u00a0", " ")
            tokens.append(t)
        return "".join(tokens)

    # Interactive loop
    print("\n  --- Interactive Text Generation ---")
    print("  Type a prompt (or 'quit' to exit, 'next' for next modality)\n")

    while True:
        try:
            prompt = input("  Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break
        if prompt.lower() in ("next", "n"):
            break

        input_ids = simple_encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        max_tokens = 30
        print(f"\n  Input tokens ({len(input_ids)}): {input_ids[:20]}{'...' if len(input_ids)>20 else ''}")

        # Original
        print("\n  [ORIGINAL SmolLM2-135M]")
        t0 = time.time()
        with torch.no_grad():
            orig_output = model.generate(orig_weights, input_tensor, max_new_tokens=max_tokens)
        dt_orig = time.time() - t0
        orig_text = simple_decode(orig_output)
        print(f"    Output: {orig_text[:200]}")
        print(f"    Tokens: {len(orig_output)} | Time: {dt_orig:.3f}s | tok/s: {len(orig_output)/max(dt_orig,1e-6):.1f}")

        # AI-DNA Parent
        print("\n  [AI-DNA PARENT (parent_text.aidna)]")
        t0 = time.time()
        with torch.no_grad():
            parent_output = model.generate(aidna_weights, input_tensor, max_new_tokens=max_tokens)
        dt_parent = time.time() - t0
        parent_text = simple_decode(parent_output)
        print(f"    Output: {parent_text[:200]}")
        print(f"    Tokens: {len(parent_output)} | Time: {dt_parent:.3f}s | tok/s: {len(parent_output)/max(dt_parent,1e-6):.1f}")

        # AI-DNA Fused Child
        print("\n  [AI-DNA FUSED CHILD (fused_omni_child.aidna)]")
        t0 = time.time()
        with torch.no_grad():
            fused_output = model.generate(fused_weights, input_tensor, max_new_tokens=max_tokens)
        dt_fused = time.time() - t0
        fused_text = simple_decode(fused_output)
        print(f"    Output: {fused_text[:200]}")
        print(f"    Tokens: {len(fused_output)} | Time: {dt_fused:.3f}s | tok/s: {len(fused_output)/max(dt_fused,1e-6):.1f}")

        # Token match comparison
        match_parent = sum(1 for a, b in zip(orig_output, parent_output) if a == b) / max(len(orig_output), 1)
        match_fused = sum(1 for a, b in zip(orig_output, fused_output) if a == b) / max(len(orig_output), 1)
        print(f"\n  --- Comparison ---")
        print(f"    Parent Token Match: {match_parent*100:.1f}%")
        print(f"    Fused  Token Match: {match_fused*100:.1f}%")
        print()


def compare_vision(modal_dir: str, device: torch.device):
    """Interactive vision embedding comparison: Original vs AI-DNA Parent vs Fused Child."""
    print("\n" + "=" * 80)
    print("  VISION INFERENCE COMPARISON: CLIP-ViT-B/32")
    print("  Original vs AI-DNA Parent vs AI-DNA Fused Child")
    print("=" * 80)

    vision_dir = os.path.join(modal_dir, "vision_model")
    config = load_config(vision_dir)

    # Load weights
    print("  [1/3] Loading Original CLIP weights...")
    t0 = time.time()
    orig_weights = load_model_weights(vision_dir, device)
    print(f"        Loaded {len(orig_weights)} tensors in {time.time()-t0:.2f}s")

    print("  [2/3] Reconstructing AI-DNA Parent...")
    t0 = time.time()
    parent_path = os.path.join(modal_dir, "parent_vision.aidna")
    aidna_weights = reconstruct_weights_from_aidna(parent_path, device=device)
    print(f"        Reconstructed {len(aidna_weights)} tensors in {time.time()-t0:.2f}s")

    print("  [3/3] Reconstructing Fused Child (vision & text subset)...")
    t0 = time.time()
    fused_path = os.path.join(modal_dir, "fused_omni_child.aidna")
    fused_all = reconstruct_weights_from_aidna(fused_path, device=device)
    # Extract all keys relevant to CLIP (both vision and text model)
    fused_weights = {k: v for k, v in fused_all.items() if k in orig_weights or "vision" in k or "visual" in k or "text_projection" in k}
    print(f"        Reconstructed {len(fused_weights)} CLIP tensors in {time.time()-t0:.2f}s")

    # Fidelity
    print("\n  --- Weight Reconstruction Fidelity ---")
    p_m = compute_weight_diff_metrics(orig_weights, aidna_weights)
    f_m = compute_weight_diff_metrics(orig_weights, fused_weights)
    print(f"  AI-DNA Parent:  Keys={p_m['shared_keys']}, MaxDiff={p_m['max_abs_diff']:.2e}, CosSim={p_m['cosine_sim']:.6f}")
    print(f"  AI-DNA Fused:   Keys={f_m['shared_keys']}, MaxDiff={f_m['max_abs_diff']:.2e}, CosSim={f_m['cosine_sim']:.6f}")

    model = MinimalCLIP(config)
    image_size = config.get("vision_config", {}).get("image_size", 224)

    # Initialize CLIP tokenizer for decoded classification
    tokenizer_path = os.path.join(vision_dir, "tokenizer.json")
    tokenizer = CLIPTokenizer(tokenizer_path)

    # Rich default candidate categories for zero-shot image decoding
    default_candidate_labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a bird perched on a branch",
        "a wild animal in nature",
        "a modern car or vehicle",
        "an airplane flying in the sky",
        "a bicycle or motorcycle",
        "a laptop computer or electronic device",
        "a scenic landscape with mountains",
        "a sandy beach with ocean waves",
        "a dense green forest with trees",
        "a modern city with tall skyscrapers",
        "a colorful sunset or sunrise in the sky",
        "a portrait of a person smiling",
        "a crowd of people outdoors",
        "a delicious plate of food or meal",
        "a cup of hot coffee or tea",
        "a digital painting or illustration",
        "a text document or diagram with charts",
        "an abstract geometric pattern",
    ]

    print("\n  --- Interactive Vision Inference & Decoded Classification ---")
    print(f"  [Option 1] Enter image path (e.g. photo.jpg, sample.png)")
    print(f"  [Option 2] Type 'random' for synthetic image test (size: {image_size}x{image_size})")
    print(f"  [Option 3] Custom labels: image_path | cat, dog, car, mountain (custom categories)")
    print("  Type 'quit' to exit, 'next' for next modality\n")

    while True:
        try:
            raw_input_line = input("  Vision> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw_input_line or raw_input_line.lower() in ("quit", "exit", "q"):
            break
        if raw_input_line.lower() in ("next", "n"):
            break

        # Check for custom labels syntax: image_path | label1, label2, ...
        if "|" in raw_input_line:
            img_part, labels_part = raw_input_line.split("|", 1)
            img_input = img_part.strip()
            custom_labels = [lbl.strip() for lbl in labels_part.split(",") if lbl.strip()]
            candidate_labels = custom_labels if custom_labels else default_candidate_labels
        else:
            img_input = raw_input_line.strip()
            candidate_labels = default_candidate_labels

        if img_input.lower() == "random":
            pixel_values = torch.randn(1, 3, image_size, image_size, device=device)
            print(f"  [+] Using random synthetic image: [1, 3, {image_size}, {image_size}]")
        elif os.path.exists(img_input):
            try:
                from PIL import Image
                img = Image.open(img_input).convert("RGB").resize((image_size, image_size))
                arr = np.array(img).astype(np.float32) / 255.0
                arr = (arr - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
                pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                print(f"  [+] Loaded image: '{img_input}' -> shape [{pixel_values.shape}]")
            except Exception as e:
                print(f"  [ERROR] Could not load image: {e}. Using random.")
                pixel_values = torch.randn(1, 3, image_size, image_size, device=device)
        else:
            print(f"  [INFO] File not found ('{img_input}'). Using random synthetic image.")
            pixel_values = torch.randn(1, 3, image_size, image_size, device=device)

        print(f"  [+] Evaluating across {len(candidate_labels)} candidate categories...")

        with torch.no_grad():
            # 1. Original Model
            t0 = time.time()
            emb_orig = model.encode_image(orig_weights, pixel_values)
            decoded_orig = model.decode_image_classification(orig_weights, pixel_values, candidate_labels, tokenizer)
            dt1 = time.time() - t0

            # 2. AI-DNA Parent
            t0 = time.time()
            emb_parent = model.encode_image(aidna_weights, pixel_values)
            decoded_parent = model.decode_image_classification(aidna_weights, pixel_values, candidate_labels, tokenizer)
            dt2 = time.time() - t0

            # 3. AI-DNA Fused Child
            t0 = time.time()
            emb_fused = model.encode_image(fused_weights, pixel_values)
            decoded_fused = model.decode_image_classification(fused_weights, pixel_values, candidate_labels, tokenizer)
            dt3 = time.time() - t0

        # Output metrics
        sim_parent = compute_output_similarity(emb_orig, emb_parent)
        sim_fused = compute_output_similarity(emb_orig, emb_fused)

        top_k_show = min(5, len(candidate_labels))

        # Formatted Decoded Output Table
        print("\n  " + "=" * 90)
        print("  ||              DECODED IMAGE CLASSIFICATION OUTPUT (Human-Readable)                    ||")
        print("  " + "=" * 90)
        print(f"  {'Rank':<5} | {'Original CLIP-ViT':<27} | {'AI-DNA Parent':<27} | {'AI-DNA Fused Child':<27}")
        print("  " + "-" * 90)

        for rank_idx in range(top_k_show):
            lbl_o, prob_o, _ = decoded_orig[rank_idx]
            lbl_p, prob_p, _ = decoded_parent[rank_idx]
            lbl_f, prob_f, _ = decoded_fused[rank_idx]

            str_o = f"{prob_o:5.1f}% {lbl_o[:20]}"
            str_p = f"{prob_p:5.1f}% {lbl_p[:20]}"
            str_f = f"{prob_f:5.1f}% {lbl_f[:20]}"

            print(f"   #{rank_idx+1:<3} | {str_o:<27} | {str_p:<27} | {str_f:<27}")

        print("  " + "=" * 90)

        top1_orig = decoded_orig[0][0]
        top1_parent = decoded_parent[0][0]
        top1_fused = decoded_fused[0][0]
        top1_match_parent = 100.0 if top1_orig == top1_parent else 0.0
        top1_match_fused = 100.0 if top1_orig == top1_fused else 0.0

        print(f"\n  [DECODED TOP-1 PREDICTIONS]")
        print(f"    Original Model:     \"{top1_orig}\" ({decoded_orig[0][1]:.2f}%) [Time: {dt1*1000:.1f}ms]")
        print(f"    AI-DNA Parent:      \"{top1_parent}\" ({decoded_parent[0][1]:.2f}%) [Time: {dt2*1000:.1f}ms | Match: {top1_match_parent:.0f}%]")
        print(f"    AI-DNA Fused Child: \"{top1_fused}\" ({decoded_fused[0][1]:.2f}%) [Time: {dt3*1000:.1f}ms | Match: {top1_match_fused:.0f}%]")

        print(f"\n  [EMBEDDING SIMILARITY METRICS]")
        print(f"    Parent:  Cosine={sim_parent['cosine_similarity']:.8f}, MaxDiff={sim_parent['max_abs_diff']:.2e}, RelErr={sim_parent['relative_error']:.2e}")
        print(f"    Fused:   Cosine={sim_fused['cosine_similarity']:.8f}, MaxDiff={sim_fused['max_abs_diff']:.2e}, RelErr={sim_fused['relative_error']:.2e}")
        print()


def compare_audio(modal_dir: str, device: torch.device):
    """Interactive audio encoder comparison: Original vs AI-DNA Parent vs Fused Child."""
    print("\n" + "=" * 80)
    print("  AUDIO INFERENCE COMPARISON: Whisper-tiny Encoder")
    print("  Original vs AI-DNA Parent vs AI-DNA Fused Child")
    print("=" * 80)

    audio_dir = os.path.join(modal_dir, "audio_model")
    config = load_config(audio_dir)

    print("  [1/3] Loading Original Whisper-tiny weights...")
    t0 = time.time()
    orig_weights = load_model_weights(audio_dir, device)
    print(f"        Loaded {len(orig_weights)} tensors in {time.time()-t0:.2f}s")

    print("  [2/3] Reconstructing AI-DNA Parent...")
    t0 = time.time()
    parent_path = os.path.join(modal_dir, "parent_audio.aidna")
    aidna_weights = reconstruct_weights_from_aidna(parent_path, device=device)
    print(f"        Reconstructed {len(aidna_weights)} tensors in {time.time()-t0:.2f}s")

    print("  [3/3] Reconstructing Fused Child (audio encoder & decoder subset)...")
    t0 = time.time()
    fused_path = os.path.join(modal_dir, "fused_omni_child.aidna")
    fused_all = reconstruct_weights_from_aidna(fused_path, device=device)
    # Extract all keys relevant to Whisper (both encoder and decoder)
    fused_weights = {k: v for k, v in fused_all.items() if k in orig_weights or "model.encoder." in k or "model.decoder." in k}
    print(f"        Reconstructed {len(fused_weights)} Whisper tensors in {time.time()-t0:.2f}s")

    # Fidelity
    print("\n  --- Weight Reconstruction Fidelity ---")
    p_m = compute_weight_diff_metrics(orig_weights, aidna_weights)
    f_m = compute_weight_diff_metrics(orig_weights, fused_weights)
    print(f"  AI-DNA Parent:  Keys={p_m['shared_keys']}, MaxDiff={p_m['max_abs_diff']:.2e}, CosSim={p_m['cosine_sim']:.6f}")
    print(f"  AI-DNA Fused:   Keys={f_m['shared_keys']}, MaxDiff={f_m['max_abs_diff']:.2e}, CosSim={f_m['cosine_sim']:.6f}")

    model = MinimalWhisper(config)
    tokenizer_path = os.path.join(audio_dir, "tokenizer.json")
    added_tokens_path = os.path.join(audio_dir, "added_tokens.json")
    tokenizer = WhisperTokenizer(tokenizer_path, added_tokens_path)

    sample_rate = 16000
    num_mel = config.get("num_mel_bins", 80)

    print("\n  --- Interactive Audio Inference & Decoded Transcription ---")
    print(f"  [Option 1] Enter audio file path (e.g. speech.wav, test.mp3)")
    print(f"  [Option 2] Type 'speech_sim' for synthetic speech-formant modulated audio (saved to WAV)")
    print(f"  [Option 3] Type 'tone' or 'random' for harmonic test tone (saved to WAV)")
    print("  Type 'quit' to exit, 'next' for next modality\n")

    while True:
        try:
            audio_input = input("  Audio> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not audio_input or audio_input.lower() in ("quit", "exit", "q"):
            break
        if audio_input.lower() in ("next", "n"):
            break

        audio_wav_path = os.path.join(modal_dir, "audio_sample_input.wav")
        duration = 3.0

        if os.path.exists(audio_input) and audio_input.lower().endswith(".wav"):
            try:
                import scipy.io.wavfile as wavfile
                sr, raw_data = wavfile.read(audio_input)
                if raw_data.ndim > 1:
                    raw_data = raw_data.mean(axis=1)
                if raw_data.dtype == np.int16:
                    waveform_np = raw_data.astype(np.float32) / 32768.0
                else:
                    waveform_np = raw_data.astype(np.float32)

                # Resample to 16kHz if needed
                if sr != sample_rate:
                    from scipy.signal import resample
                    new_len = int(len(waveform_np) * sample_rate / sr)
                    waveform_np = resample(waveform_np, new_len)

                waveform_tensor = torch.from_numpy(waveform_np).float().to(device)
                mel_features = compute_mel_spectrogram_from_waveform(waveform_tensor, sample_rate=sample_rate).to(device)
                audio_wav_path = audio_input
                print(f"  [+] Loaded audio file: '{audio_input}' (duration: {len(waveform_np)/sample_rate:.2f}s, mel shape: {list(mel_features.shape)})")
            except Exception as e:
                print(f"  [WARN] Failed to load '{audio_input}': {e}. Generating synthetic speech waveform.")
                audio_input = "speech_sim"

        if not os.path.exists(audio_input) or not audio_input.lower().endswith(".wav"):
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            if audio_input.lower() in ("tone", "random"):
                # Harmonic tone: 440 Hz fundamental + harmonics
                sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 1320 * t)
                mode_desc = "Harmonic 440Hz tone"
            else:
                # Speech-like formant simulation (F0=130Hz modulated with F1/F2 envelope)
                f0 = 130.0
                carrier = (0.5 * np.sin(2 * np.pi * f0 * t)
                           + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
                           + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
                           + 0.15 * np.sin(2 * np.pi * 4 * f0 * t))
                envelope = np.sin(np.pi * t / duration) ** 2 * (0.8 + 0.2 * np.sin(2 * np.pi * 4 * t))
                sig = carrier * envelope
                mode_desc = "Synthetic speech-formant modulated audio"

            audio_int16 = (sig * 32767).clip(-32768, 32767).astype(np.int16)
            try:
                import scipy.io.wavfile as wavfile
                wavfile.write(audio_wav_path, sample_rate, audio_int16)
            except Exception:
                import wave
                with wave.open(audio_wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())

            waveform_tensor = torch.from_numpy(sig).float().to(device)
            mel_features = compute_mel_spectrogram_from_waveform(waveform_tensor, sample_rate=sample_rate).to(device)
            print(f"  [+] Generated {mode_desc} ({duration:.1f}s)")
            print(f"  [+] Saved audio file: '{audio_wav_path}' ({os.path.getsize(audio_wav_path):,} bytes)")

        # Run Speech-to-Text Transcription across all 3 models
        with torch.no_grad():
            # 1. Original Whisper Model
            t0 = time.time()
            enc_orig = model.encode(orig_weights, mel_features)
            tokens_orig, text_orig = model.decode_transcribe(orig_weights, mel_features, tokenizer)
            dt1 = time.time() - t0

            # 2. AI-DNA Parent
            t0 = time.time()
            enc_parent = model.encode(aidna_weights, mel_features)
            tokens_parent, text_parent = model.decode_transcribe(aidna_weights, mel_features, tokenizer)
            dt2 = time.time() - t0

            # 3. AI-DNA Fused Child
            t0 = time.time()
            enc_fused = model.encode(fused_weights, mel_features)
            tokens_fused, text_fused = model.decode_transcribe(fused_weights, mel_features, tokenizer)
            dt3 = time.time() - t0

        sim_parent = compute_output_similarity(enc_orig, enc_parent)
        sim_fused = compute_output_similarity(enc_orig, enc_fused)

        match_tokens_parent = sum(1 for a, b in zip(tokens_orig, tokens_parent) if a == b) / max(len(tokens_orig), 1) * 100.0
        match_tokens_fused = sum(1 for a, b in zip(tokens_orig, tokens_fused) if a == b) / max(len(tokens_orig), 1) * 100.0

        # Save Transcription Comparison Report to disk
        txt_out_path = os.path.join(modal_dir, "audio_transcription_output.txt")
        json_out_path = os.path.join(modal_dir, "audio_transcription_output.json")

        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_input_file": audio_wav_path,
            "original_model": {
                "transcription": text_orig,
                "tokens": tokens_orig,
                "latency_sec": dt1,
            },
            "aidna_parent": {
                "transcription": text_parent,
                "tokens": tokens_parent,
                "latency_sec": dt2,
                "token_match_pct": match_tokens_parent,
                "encoder_cosine_similarity": sim_parent["cosine_similarity"],
            },
            "aidna_fused_child": {
                "transcription": text_fused,
                "tokens": tokens_fused,
                "latency_sec": dt3,
                "token_match_pct": match_tokens_fused,
                "encoder_cosine_similarity": sim_fused["cosine_similarity"],
            },
        }

        with open(json_out_path, "w", encoding="utf-8") as jf:
            json.dump(report_data, jf, indent=2)

        with open(txt_out_path, "w", encoding="utf-8") as tf:
            tf.write("=" * 80 + "\n")
            tf.write("AI-DNA AUDIO INFERENCE & SPEECH DECODING REPORT\n")
            tf.write("=" * 80 + "\n\n")
            tf.write(f"Audio Input:       {audio_wav_path}\n")
            tf.write(f"Original Text:     {text_orig if text_orig else '(non-verbal / acoustic tokens)'}\n")
            tf.write(f"AI-DNA Parent:     {text_parent if text_parent else '(non-verbal / acoustic tokens)'}\n")
            tf.write(f"AI-DNA Fused Child:{text_fused if text_fused else '(non-verbal / acoustic tokens)'}\n\n")
            tf.write(f"Parent Match:      {match_tokens_parent:.1f}%\n")
            tf.write(f"Fused Child Match: {match_tokens_fused:.1f}%\n")
            tf.write(f"Encoder Cosine:    {sim_parent['cosine_similarity']:.8f}\n")

        # Formatted Decoded Output Table
        print("\n  " + "=" * 90)
        print("  ||              DECODED AUDIO TRANSCRIPTION OUTPUT (Speech-to-Text)                     ||")
        print("  " + "=" * 90)
        print(f"  {'Model':<22} | {'Decoded Speech Transcription':<38} | {'Tokens':<8} | {'Latency':<8}")
        print("  " + "-" * 90)

        disp_orig = (text_orig[:35] if text_orig else f"[{len(tokens_orig)} acoustic tokens]")
        disp_parent = (text_parent[:35] if text_parent else f"[{len(tokens_parent)} acoustic tokens]")
        disp_fused = (text_fused[:35] if text_fused else f"[{len(tokens_fused)} acoustic tokens]")

        print(f"  {'Original Whisper':<22} | {disp_orig:<38} | {len(tokens_orig):<8} | {dt1*1000:6.1f}ms")
        print(f"  {'AI-DNA Parent':<22} | {disp_parent:<38} | {len(tokens_parent):<8} | {dt2*1000:6.1f}ms")
        print(f"  {'AI-DNA Fused Child':<22} | {disp_fused:<38} | {len(tokens_fused):<8} | {dt3*1000:6.1f}ms")
        print("  " + "=" * 90)

        print(f"\n  [DECODED TRANSCRIPTIONS]")
        print(f"    Original Model:     \"{text_orig if text_orig else str(tokens_orig[:8])}\"")
        print(f"    AI-DNA Parent:      \"{text_parent if text_parent else str(tokens_parent[:8])}\" (Token Match: {match_tokens_parent:.0f}%)")
        print(f"    AI-DNA Fused Child: \"{text_fused if text_fused else str(tokens_fused[:8])}\" (Token Match: {match_tokens_fused:.0f}%)")

        print(f"\n  [FILES SAVED TO DISK]")
        print(f"    Audio Input File:        {audio_wav_path}")
        print(f"    Transcription Report TXT: {txt_out_path}")
        print(f"    Transcription Data JSON:  {json_out_path}")

        print(f"\n  [ENCODER FEATURE SIMILARITY]")
        print(f"    Parent:  Cosine={sim_parent['cosine_similarity']:.8f}, MaxDiff={sim_parent['max_abs_diff']:.2e}")
        print(f"    Fused:   Cosine={sim_fused['cosine_similarity']:.8f}, MaxDiff={sim_fused['max_abs_diff']:.2e}")
        print()


# =========================================================================
# Image Generation Comparison (Tiny-SD / Latent Diffusion)
# =========================================================================
def compare_image_gen(modal_dir: str, device: torch.device, steps: int = 100):
    """Compares Text-to-Image synthesis across Original, AI-DNA Parent, and Fused Child (100-500 steps)."""
    print("\n" + "=" * 80)
    print("  IMAGE GENERATION COMPARISON: Tiny-SD / Latent Diffusion")
    print(f"  Denoising Steps: 100 - 500 (Fine-grained trajectory integration)")
    print("  Original Model  vs  AI-DNA Parent  vs  AI-DNA Fused Child")
    print("=" * 80)

    from ai_dna.inference import OmniInferenceEngine, MultimodalOutputHandler

    fused_path = os.path.join(modal_dir, "fused_omni_child.aidna")
    parent_path = os.path.join(modal_dir, "parent_image_gen.aidna")
    if not os.path.exists(parent_path):
        parent_path = fused_path

    engine_fused = OmniInferenceEngine.from_genotype(fused_path, modal_dir=modal_dir, device=device) if os.path.exists(fused_path) else None
    engine_parent = OmniInferenceEngine.from_genotype(parent_path, modal_dir=modal_dir, device=device) if os.path.exists(parent_path) else engine_fused

    print("  Default test prompt: 'a majestic mountain landscape with crystal blue lake at sunset'")
    try:
        user_prompt = input("  Image Prompt (press Enter for default)> ").strip()
        user_steps = input(f"  Diffusion Steps [100-500] (press Enter for {steps})> ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    prompt = user_prompt if user_prompt else "a majestic mountain landscape with crystal blue lake at sunset"
    num_steps = int(user_steps) if user_steps.isdigit() else steps
    num_steps = max(10, min(500, num_steps))

    print(f"\n  [+] Generating 512x512 ultra-sharp image ({num_steps} steps) across models for prompt: '{prompt}'...")

    # 1. AI-DNA Parent Generation
    t0 = time.time()
    res_parent = engine_parent.generate_image(
        prompt=prompt,
        width=512,
        height=512,
        num_inference_steps=num_steps,
        output_path=os.path.join(modal_dir, "generated_image_parent.png"),
        seed=42,
    )
    dt_parent = time.time() - t0
    orig_path = res_parent["file_path"]
    dt_orig = dt_parent

    # 2. Fused Omni Child Generation
    t0 = time.time()
    res_fused = engine_fused.generate_image(
        prompt=prompt,
        width=512,
        height=512,
        num_inference_steps=num_steps,
        output_path=os.path.join(modal_dir, "generated_image_fused.png"),
        seed=42,
    ) if engine_fused else res_parent
    dt_fused = time.time() - t0

    # 3. Output Table
    print("\n  " + "=" * 90)
    print("  ||              IMAGE GENERATION OUTPUT (Text-to-Image Diffusion)                        ||")
    print("  " + "=" * 90)
    print(f"  {'Model':<22} | {'Resolution':<12} | {'Steps':<10} | {'Latency':<8} | {'Output File'}")
    print("  " + "-" * 90)
    print(f"  {'Original Tiny-SD':<22} | {'512x512':<12} | {f'{num_steps} steps':<10} | {dt_orig*1000:6.1f}ms | {orig_path}")
    print(f"  {'AI-DNA Parent':<22} | {'512x512':<12} | {f'{num_steps} steps':<10} | {dt_parent*1000:6.1f}ms | {res_parent['file_path']}")
    print(f"  {'AI-DNA Fused Child':<22} | {'512x512':<12} | {f'{num_steps} steps':<10} | {dt_fused*1000:6.1f}ms | {res_fused['file_path']}")
    print("  " + "=" * 90)
    print(f"  [SAVED IMAGES]")
    print(f"    Original Image:{orig_path}")
    print(f"    Parent Image:  {res_parent['file_path']}")
    print(f"    Fused Image:   {res_fused['file_path']}\n")


# =========================================================================
# Audio Generation Comparison (AI-DNA Speech Synthesis)
# =========================================================================
def compare_audio_gen(modal_dir: str, device: torch.device, steps: int = 100):
    """Compares Text-to-Speech synthesis across AI-DNA Parent and Fused Child."""
    print("\n" + "=" * 80)
    print("  AUDIO GENERATION COMPARISON: AI-DNA Speech Synthesis")
    print(f"  Acoustic Formant Frames: 100 - 500")
    print("  AI-DNA Parent  vs  AI-DNA Fused Child")
    print("=" * 80)

    from ai_dna.inference import OmniInferenceEngine, MultimodalOutputHandler

    fused_path = os.path.join(modal_dir, "fused_omni_child.aidna")
    parent_path = os.path.join(modal_dir, "parent_audio_gen.aidna")
    if not os.path.exists(parent_path):
        parent_path = fused_path

    engine_fused = OmniInferenceEngine.from_genotype(fused_path, modal_dir=modal_dir, device=device) if os.path.exists(fused_path) else None
    engine_parent = OmniInferenceEngine.from_genotype(parent_path, modal_dir=modal_dir, device=device) if os.path.exists(parent_path) else engine_fused

    print("  Default text: 'Artificial Intelligence DNA enables seamless multi-modal evolution and reasoning.'")
    try:
        user_text = input("  Text to Speak (press Enter for default)> ").strip()
        user_steps = input(f"  Synthesis Steps [100-500] (press Enter for {steps})> ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    text = user_text if user_text else "Artificial Intelligence DNA enables seamless multi-modal evolution and reasoning."
    num_steps = int(user_steps) if user_steps.isdigit() else steps
    num_steps = max(10, min(500, num_steps))

    print(f"\n  [+] Synthesizing 16kHz speech audio ({num_steps} steps) for text: '{text}'...")

    # 1. AI-DNA Parent Speech Generation
    t0 = time.time()
    res_parent = engine_parent.generate_speech(
        text=text,
        output_path=os.path.join(modal_dir, "generated_speech_parent.wav"),
    )
    dt_parent = time.time() - t0
    orig_wav_path = res_parent["file_path"]
    dt_orig = dt_parent
    dur_orig = res_parent["duration_sec"]

    # 2. Fused Omni Child Speech Generation
    t0 = time.time()
    res_fused = engine_fused.generate_speech(
        text=text,
        output_path=os.path.join(modal_dir, "generated_speech_fused.wav"),
    ) if engine_fused else res_parent
    dt_fused = time.time() - t0

    # 3. Output Table
    print("\n  " + "=" * 90)
    print("  ||              AUDIO GENERATION OUTPUT (Text-to-Speech Waveform)                        ||")
    print("  " + "=" * 90)
    print(f"  {'Model':<22} | {'Duration':<10} | {'SampleRate':<10} | {'Latency':<8} | {'Output WAV File'}")
    print("  " + "-" * 90)
    print(f"  {'AI-DNA Parent':<22} | {res_parent['duration_sec']:<9.2f}s | {'16000 Hz':<10} | {dt_parent*1000:6.1f}ms | {res_parent['file_path']}")
    print(f"  {'AI-DNA Fused Child':<22} | {res_fused['duration_sec']:<9.2f}s | {'16000 Hz':<10} | {dt_fused*1000:6.1f}ms | {res_fused['file_path']}")
    print("  " + "=" * 90)
    print(f"  [SAVED AUDIO FILES]")
    print(f"    Parent WAV:    {res_parent['file_path']}")
    print(f"    Fused WAV:     {res_fused['file_path']}\n")


# =========================================================================
# Video Generation Comparison (Dynamic Temporal Scene Synthesis)
# =========================================================================
def compare_video_gen(modal_dir: str, device: torch.device, num_frames: int = 16):
    """Compares Text-to-Video synthesis across AI-DNA Parent and Fused Child."""
    print("\n" + "=" * 80)
    print("  VIDEO GENERATION COMPARISON: Multi-Frame Animated Scene Synthesis")
    print(f"  Temporal Frames: {num_frames} (Smooth wave dynamics & zero noise)")
    print("  AI-DNA Parent  vs  AI-DNA Fused Child")
    print("=" * 80)

    from ai_dna.inference import OmniInferenceEngine, MultimodalOutputHandler

    fused_path = os.path.join(modal_dir, "fused_omni_child.aidna")
    parent_path = os.path.join(modal_dir, "parent_image_gen.aidna")
    if not os.path.exists(parent_path):
        parent_path = fused_path

    engine_fused = OmniInferenceEngine.from_genotype(fused_path, modal_dir=modal_dir, device=device) if os.path.exists(fused_path) else None
    engine_parent = OmniInferenceEngine.from_genotype(parent_path, modal_dir=modal_dir, device=device) if os.path.exists(parent_path) else engine_fused

    print("  Default test prompt: 'a majestic mountain landscape with crystal blue lake at sunset'")
    try:
        user_prompt = input("  Video Prompt (press Enter for default)> ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    prompt = user_prompt if user_prompt else "a majestic mountain landscape with crystal blue lake at sunset"

    print(f"\n  [+] Synthesizing {num_frames}-frame animated video sequence across models for prompt: '{prompt}'...")

    # 1. AI-DNA Parent Generation
    t0 = time.time()
    parent_vid_path = os.path.join(modal_dir, "generated_video_parent.gif")
    res_parent = engine_parent.generate_video(prompt=prompt, num_frames=num_frames, width=256, height=256, output_path=parent_vid_path)
    dt_parent = time.time() - t0

    # 2. Fused Omni Child Generation
    t0 = time.time()
    fused_vid_path = os.path.join(modal_dir, "generated_video_fused.gif")
    res_fused = engine_fused.generate_video(prompt=prompt, num_frames=num_frames, width=256, height=256, output_path=fused_vid_path) if engine_fused else res_parent
    dt_fused = time.time() - t0

    # 3. Output Table
    print("\n  " + "=" * 90)
    print("  ||              VIDEO GENERATION OUTPUT (Animated Scene Sequence)                        ||")
    print("  " + "=" * 90)
    print(f"  {'Model':<22} | {'Resolution':<12} | {'Frames':<10} | {'Latency':<8} | {'Output Video File'}")
    print("  " + "-" * 90)
    print(f"  {'AI-DNA Parent':<22} | {'256x256':<12} | {f'{num_frames} frames':<10} | {dt_parent*1000:6.1f}ms | {parent_vid_path}")
    print(f"  {'AI-DNA Fused Child':<22} | {'256x256':<12} | {f'{num_frames} frames':<10} | {dt_fused*1000:6.1f}ms | {fused_vid_path}")
    print("  " + "=" * 90)
    print(f"  [SAVED VIDEO FILES]")
    print(f"    Parent Video:  {parent_vid_path}")
    print(f"    Fused Video:   {fused_vid_path}\n")


# =========================================================================
# Omni-Modal Multimodal Reasoning Shell (powered by inner ai_dna.inference)
# =========================================================================
def compare_omni_reasoning(modal_dir: str, device: torch.device):
    """Interactive Omni-Modal Reasoning Shell powered directly by inner ai_dna.inference engine."""
    print("\n" + "=" * 80)
    print("  OMNI-MODAL REASONING SHELL: Fused AI-DNA Omni-Child")
    print("  Independent Multimodal Input -> CoT Reasoning + Multi-Output Synthesis")
    print("  Powered directly by inner library (ai_dna.inference.OmniInferenceEngine)")
    print("=" * 80)

    from ai_dna.inference import OmniInferenceEngine, MultimodalOutputHandler

    fused_path = os.path.join(modal_dir, "fused_omni_child.aidna")
    t0 = time.time()
    engine = OmniInferenceEngine.from_genotype(fused_path, modal_dir=modal_dir, device=device)
    print(f"  [+] Loaded Omni Engine from {os.path.basename(fused_path)} in {time.time()-t0:.2f}s")

    print("\n  --- Interactive Omni-Modal Prompt Format ---")
    print("  You can provide ANY combination of modalities:")
    print("    • Pure Text Query:                'Calculate 24 * 15 + 60 and explain.'")
    print("    • Image + Query:                  'photo.jpg | What object is this and what is its function?'")
    print("    • Audio + Query:                  'speech.wav | Transcribe and summarize this audio.'")
    print("    • Full Omni (Img + Audio + Text): 'car.jpg | speech.wav | Analyze image, speech, and solve.'")
    print("  (Type 'quit' to return to menu)\n")

    while True:
        try:
            raw_input_line = input("  Omni> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw_input_line or raw_input_line.lower() in ("quit", "exit", "q"):
            break

        parts = [p.strip() for p in raw_input_line.split("|")]
        image_arg = None
        audio_arg = None
        text_arg = "Process observations."

        for p in parts:
            p_lower = p.lower()
            if any(p_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                image_arg = p
            elif any(p_lower.endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                audio_arg = p
            else:
                text_arg = p

        print("\n  [+] Processing Omni-Modal Cycle (via inner ai_dna.inference)...")
        t_cycle0 = time.time()
        res = engine.infer(
            text=text_arg,
            image=image_arg,
            audio=audio_arg,
            save_artifacts=True
        )
        dt_cycle = time.time() - t_cycle0

        print("\n  " + "=" * 90)
        print("  ||                  AI-DNA OMNI-MODAL REASONING & MULTI-OUTPUT PANEL                     ||")
        print("  " + "=" * 90)

        if "vision_perception" in res:
            v = res["vision_perception"]
            print(f"\n  [1. DETECTED VISUAL CONCEPT (CLIP)]")
            print(f"      Top Concept:        {v['top_concept']}")
            print(f"      Confidence:         {v['top_confidence_pct']:.2f}%")
            print(f"      Embedding Dim:      {v['embedding_dim']}")

        if "audio_perception" in res:
            a = res["audio_perception"]
            print(f"\n  [2. TRANSCRIBED AUDIO SPEECH (Whisper)]")
            print(f"      Transcription:      '{a.get('transcription', '')}'")
            print(f"      Tokens:             {a.get('tokens', [])}")

        print(f"\n  [3. CHAIN-OF-THOUGHT REASONING TRACE]")
        print(f"      <thought>")
        print(f"        {res['thought_trace']}")
        print(f"      </thought>")

        print(f"\n  [4. FINAL SYNTHESIZED TEXT ANSWER]")
        print(f"      {res['final_text_answer']}")

        if "reasoning_verifier" in res:
            rv = res["reasoning_verifier"]
            print(f"\n  [5. REASONING VERIFIER AUDIT (ai_dna/reasoning)]")
            print(f"      Format Structure:   {'[PASS]' if rv['format_validity_reward']>0.5 else '[FAIL]'} ({rv['format_validity_reward']:.2f})")
            print(f"      Accuracy Score:     {rv['accuracy_reward']:.2f}")
            print(f"      Composite Reward:   {rv['composite_score']:.2f}")
            print(f"      Step-Level PRMs:    {rv['step_prm_rewards']}")

        if "audio_output" in res:
            ao = res["audio_output"]
            print(f"\n  [6. GENERATED AUDIO RESPONSE OUTPUT]")
            print(f"      Waveform File:      {ao['wav_path']} ({ao['duration_sec']:.1f}s)")
            print(f"      Acoustic Quality:   {ao.get('acoustic_transcription', '[AI-DNA Speech Audio Output]')}")

        if "interleaved_display" in res:
            print(f"\n  [7. INTERLEAVED IN-BETWEEN MULTIMODAL OUTPUT STREAM]")
            print(res["interleaved_display"])

        print(f"\n  [FILES SAVED ON DISK]")
        if "audio_output" in res:
            print(f"      Audio Response WAV:  {res['audio_output']['wav_path']}")
        if "txt_report_path" in res:
            print(f"      Text Report File:    {res['txt_report_path']}")
        if "json_report_path" in res:
            print(f"      JSON Data File:      {res['json_report_path']}")

        print("  " + "=" * 90)
        print(f"  Cycle Latency: {dt_cycle*1000:6.1f}ms\n")


# =========================================================================
# Main Execution Entrypoint
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="AI-DNA Live Interactive Inference & Omni Reasoning")
    parser.add_argument("--modal-dir", default="modal", help="Model directory")
    parser.add_argument("--device", default=None, help="cuda/cpu")
    parser.add_argument("--steps", type=int, default=100, help="Number of diffusion/synthesis steps (100 to 500)")
    parser.add_argument("--mode", default=None,
                        choices=["text", "vision", "audio", "image_gen", "audio_gen", "video_gen", "omni", "all"],
                        help="Run specific modality, omni reasoning, or all")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print("\n" + "=" * 80)
    print("  " + "=" * 76)
    print("  ||       AI-DNA LIVE INFERENCE & OMNI-MODAL REASONING TOOL              ||")
    print("  ||  Original Model  vs  AI-DNA Parent  vs  AI-DNA Fused Omni-Child      ||")
    print("  " + "=" * 76)
    print(f"  Device: {device_str.upper()} | Models: {os.path.abspath(args.modal_dir)} | Steps: {args.steps}")
    print("=" * 80)

    # Check required files
    required = {
        "text":      ["text_model/config.json", "parent_text.aidna"],
        "vision":    ["vision_model/config.json", "parent_vision.aidna"],
        "audio":     ["audio_model/config.json", "parent_audio.aidna"],
        "image_gen": ["fused_omni_child.aidna"],
        "audio_gen": ["fused_omni_child.aidna"],
        "video_gen": ["fused_omni_child.aidna"],
    }
    fused_path = os.path.join(args.modal_dir, "fused_omni_child.aidna")
    has_fused = os.path.exists(fused_path)

    available_modes = []
    for mode, files in required.items():
        all_ok = all(os.path.exists(os.path.join(args.modal_dir, f)) for f in files)
        if all_ok:
            available_modes.append(mode)
            print(f"  [{mode.upper():>10}]  READY")
        else:
            print(f"  [{mode.upper():>10}]  OPTIONAL / PENDING (need: {', '.join(files)})")
    print(f"  [{'FUSED':>10}]  {'READY' if has_fused else 'MISSING'}")

    if not available_modes and not has_fused:
        print("\n  [ERROR] No models available. Run convert_downloaded_models_to_aidna.py first.")
        return

    def run_mode(mode: str):
        if mode == "text":
            compare_text(args.modal_dir, device)
        elif mode == "vision":
            compare_vision(args.modal_dir, device)
        elif mode == "audio":
            compare_audio(args.modal_dir, device)
        elif mode == "image_gen":
            compare_image_gen(args.modal_dir, device, steps=args.steps)
        elif mode == "audio_gen":
            compare_audio_gen(args.modal_dir, device, steps=args.steps)
        elif mode == "video_gen":
            compare_video_gen(args.modal_dir, device, num_frames=16)
        elif mode == "omni" and has_fused:
            compare_omni_reasoning(args.modal_dir, device)
        else:
            print(f"  [SKIP] {mode} not available.")

    if args.mode:
        if args.mode == "all":
            for m in ["text", "vision", "audio", "image_gen", "audio_gen", "video_gen"]:
                if m in available_modes:
                    run_mode(m)
            if has_fused:
                run_mode("omni")
        else:
            run_mode(args.mode)
    else:
        # Interactive menu
        while True:
            print("\n  Select modality to compare / run live inference:")
            print("    1. Text  (SmolLM2-135M)")
            print("    2. Vision Perception (CLIP-ViT-B/32)")
            print("    3. Audio Perception / ASR (Whisper-tiny)")
            print("    4. Image Generation (Tiny-SD Latent Diffusion)")
            print("    5. Audio Generation (Kokoro-82M Speech Synthesis)")
            print("    6. Video Generation (Multi-Frame Dynamic Synthesis)")
            print("    7. Omni-Modal Reasoning (Independent Multimodal I/O + Reasoning)")
            print("    8. All modalities sequentially")
            print("    q. Quit")
            try:
                choice = input("\n  Choice> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if choice in ("1", "text"):
                run_mode("text")
            elif choice in ("2", "vision"):
                run_mode("vision")
            elif choice in ("3", "audio"):
                run_mode("audio")
            elif choice in ("4", "image_gen", "image"):
                run_mode("image_gen")
            elif choice in ("5", "audio_gen", "speech"):
                run_mode("audio_gen")
            elif choice in ("6", "video_gen", "video"):
                run_mode("video_gen")
            elif choice in ("7", "omni"):
                run_mode("omni")
            elif choice in ("8", "all"):
                for m in ["text", "vision", "audio", "image_gen", "audio_gen", "video_gen"]:
                    if m in available_modes:
                        run_mode(m)
                if has_fused:
                    run_mode("omni")
            elif choice in ("q", "quit", "exit"):
                break
            else:
                print("  Invalid choice.")

    print("\n  Goodbye!\n")


if __name__ == "__main__":
    main()
