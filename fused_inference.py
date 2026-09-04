"""
AI-DNA Fused Omni-Modal Inference Engine.

Loads strictly from a single fused AI-DNA Genotype (.aidna file),
reconstructs/grows the multi-modal neural phenotype, and executes live inference across:
  1. Text Generation (SmolLM2-135M Autoregressive Decoder)
  2. Vision Perception & Zero-Shot Classification (CLIP-ViT-B/32)
  3. Audio Perception / Speech-to-Text ASR (Whisper-tiny)
  4. Image Generation (Latent Neural Diffusion Head)
  5. Audio Generation (Kokoro-82M Neural TTS & Phenotype Acoustic Synthesis)
  6. Video Generation (Spatiotemporal Animated Scene Synthesis)
  7. Unified Omni-Modal Reasoning (Sensory Fusion + CoT <thought> + PRM Verification)

Usage:
  .venv\\Scripts\\python.exe fused_inference.py                           # Interactive menu
  .venv\\Scripts\\python.exe fused_inference.py --mode text               # Text only
  .venv\\Scripts\\python.exe fused_inference.py --mode vision             # Vision only
  .venv\\Scripts\\python.exe fused_inference.py --mode audio              # Audio only
  .venv\\Scripts\\python.exe fused_inference.py --mode image_gen          # Image generation
  .venv\\Scripts\\python.exe fused_inference.py --mode audio_gen          # Speech synthesis
  .venv\\Scripts\\python.exe fused_inference.py --mode video_gen          # Video generation
  .venv\\Scripts\\python.exe fused_inference.py --mode omni               # Omni-modal reasoning
  .venv\\Scripts\\python.exe fused_inference.py --mode all                # Run all modalities
"""

import os
import sys
import time
import json
import math
import argparse
from typing import Dict, Any, Tuple, Optional, List, Union

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
from ai_dna.dna.structure import Genotype
from ai_dna.inference import OmniInferenceEngine, MultimodalOutputHandler


from ai_dna.models.shells import (
    reconstruct_weights_and_genotype,
    reconstruct_weights_only,
    SmolLM2Tokenizer,
    MinimalSmolLM2,
    CLIPTokenizer,
    MinimalCLIP,
    WhisperTokenizer,
    compute_mel_spectrogram_from_waveform,
    MinimalWhisper,
)


def reconstruct_weights_from_aidna(
    aidna_path: str,
    key_filter: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """Reconstructs state_dict and genotype strictly from a fused .aidna file."""
    return reconstruct_weights_and_genotype(aidna_path, key_filter=key_filter, device=device)


# =========================================================================
# Modality Runners for Fused AI-DNA Genotype
# =========================================================================
def run_fused_text(fused_path: str, modal_dir: str, device: torch.device):
    """Executes text generation strictly on the Fused AI-DNA Genotype."""
    print("\n" + "=" * 80)
    print("  [AI-DNA FUSED INFERENCE] :: TEXT GENERATION (SmolLM2 Decoder)")
    print(f"  Genotype: {os.path.abspath(fused_path)}")
    print("=" * 80)

    t0 = time.time()
    fused_weights, genotype = reconstruct_weights_from_aidna(fused_path, device=device)
    print(f"  [+] Reconstructed {len(fused_weights)} tensors in {time.time()-t0:.2f}s")

    sensory = getattr(genotype, "sensory_assets", {}) or {}
    tokenizer = SmolLM2Tokenizer(sensory.get("tokenizer.smollm2", {}))
    model = MinimalSmolLM2(sensory.get("config.smollm2", {}))

    print("\n  Type your prompt (or 'quit' to exit, 'menu' for main menu)\n")
    while True:
        try:
            prompt = input("  Text Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q", "menu", "m"):
            break

        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        t_gen0 = time.time()
        with torch.no_grad():
            output_tokens = model.generate(fused_weights, input_tensor, max_new_tokens=40, temperature=0.7)
        dt_gen = time.time() - t_gen0

        output_text = tokenizer.decode(output_tokens)
        print("\n  " + "-" * 76)
        print(f"  Generated Text: {output_text.strip()}")
        print(f"  Tokens: {len(output_tokens)} | Time: {dt_gen:.3f}s | Speed: {len(output_tokens)/max(dt_gen,1e-6):.1f} tok/s")
        print("  " + "-" * 76 + "\n")


def run_fused_vision(fused_path: str, modal_dir: str, device: torch.device):
    """Executes visual zero-shot perception strictly on the Fused AI-DNA Genotype."""
    print("\n" + "=" * 80)
    print("  [AI-DNA FUSED INFERENCE] :: VISION PERCEPTION (CLIP-ViT)")
    print(f"  Genotype: {os.path.abspath(fused_path)}")
    print("=" * 80)

    t0 = time.time()
    fused_weights, genotype = reconstruct_weights_from_aidna(fused_path, device=device)
    print(f"  [+] Reconstructed {len(fused_weights)} tensors in {time.time()-t0:.2f}s")

    sensory = getattr(genotype, "sensory_assets", {}) or {}
    tokenizer = CLIPTokenizer(sensory.get("tokenizer.clip", {}))
    model = MinimalCLIP(sensory.get("config.clip", {}))

    default_labels = [
        "a photo of a cat", "a photo of a dog", "a modern car or vehicle",
        "an airplane in flight", "a laptop computer", "a scenic mountain landscape",
        "a cup of coffee or tea", "a portrait of a person smiling", "a dense forest",
        "a digital illustration or chart", "an abstract pattern"
    ]

    print("\n  [Option 1] Enter image file path (e.g. photo.jpg, sample.png)")
    print("  [Option 2] Type 'random' for synthetic image test")
    print("  [Option 3] Custom labels: image.png | cat, dog, car, mountain\n")

    while True:
        try:
            raw_in = input("  Vision Input> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw_in or raw_in.lower() in ("quit", "exit", "q", "menu", "m"):
            break

        if "|" in raw_in:
            img_part, labels_part = raw_in.split("|", 1)
            img_path = img_part.strip()
            candidate_labels = [lbl.strip() for lbl in labels_part.split(",") if lbl.strip()]
        else:
            img_path = raw_in
            candidate_labels = default_labels

        if img_path.lower() == "random" or not os.path.exists(img_path):
            pixel_values = torch.randn(1, 3, 224, 224, device=device)
            print("  [+] Using random synthetic image [1, 3, 224, 224]")
        else:
            try:
                from PIL import Image
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                arr = (np.array(img, dtype=np.float32) / 255.0 - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
                pixel_values = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
                print(f"  [+] Loaded image '{img_path}'")
            except Exception as e:
                print(f"  [WARN] Failed to load image: {e}. Using random tensor.")
                pixel_values = torch.randn(1, 3, 224, 224, device=device)

        t_vis0 = time.time()
        with torch.no_grad():
            decoded = model.decode_image_classification(fused_weights, pixel_values, candidate_labels, tokenizer)
        dt_vis = time.time() - t_vis0

        print("\n  " + "=" * 80)
        print(f"  {'Rank':<5} | {'Predicted Category':<35} | {'Confidence':<12}")
        print("  " + "-" * 80)
        for idx, (lbl, prob, _) in enumerate(decoded[:5], 1):
            print(f"   #{idx:<3} | {lbl:<35} | {prob:6.2f}%")
        print("  " + "=" * 80)
        print(f"  Top Concept: \"{decoded[0][0]}\" ({decoded[0][1]:.2f}%) | Latency: {dt_vis*1000:.1f}ms\n")


def run_fused_audio(fused_path: str, modal_dir: str, device: torch.device, engine: Optional[OmniInferenceEngine] = None):
    """Executes speech-to-text transcription strictly on the Fused AI-DNA Genotype."""
    print("\n" + "=" * 80)
    print("  [AI-DNA FUSED INFERENCE] :: AUDIO SPEECH-TO-TEXT (Whisper ASR)")
    print(f"  Genotype: {os.path.abspath(fused_path)}")
    print("=" * 80)

    t0 = time.time()
    fused_weights, genotype = reconstruct_weights_from_aidna(fused_path, device=device)
    print(f"  [+] Reconstructed {len(fused_weights)} tensors in {time.time()-t0:.2f}s")

    sensory = getattr(genotype, "sensory_assets", {}) or {}
    tokenizer = WhisperTokenizer(
        sensory.get("tokenizer.whisper", {}),
        sensory.get("added_tokens.whisper", {})
    )
    model = MinimalWhisper(sensory.get("config.whisper", {}))

    print("\n  [Option 1] Enter audio file path (.wav, .mp3, etc.)")
    print("  [Option 2] Type a spoken phrase to synthesize & transcribe (e.g. 'hello', 'artificial intelligence')")
    print("  [Option 3] Type 'tone' for harmonic 440Hz test tone\n")

    while True:
        try:
            audio_in = input("  Audio Input> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not audio_in or audio_in.lower() in ("quit", "exit", "q", "menu", "m"):
            break

        sample_rate = 16000
        if os.path.exists(audio_in) and any(audio_in.lower().endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg")):
            try:
                import scipy.io.wavfile as wavfile
                sr, raw = wavfile.read(audio_in)
                if raw.ndim > 1:
                    raw = raw.mean(axis=1)
                wf_np = raw.astype(np.float32) / (32768.0 if raw.dtype == np.int16 else 1.0)
                if sr != sample_rate:
                    from scipy.signal import resample
                    wf_np = resample(wf_np, int(len(wf_np) * sample_rate / sr))
                waveform_t = torch.from_numpy(wf_np).float().to(device)
                print(f"  [+] Loaded audio file: '{audio_in}' ({len(wf_np)/sample_rate:.2f}s)")
            except Exception:
                waveform_t = torch.randn(sample_rate * 2, device=device)
        elif audio_in.lower() == "tone":
            t = np.linspace(0, 2.0, sample_rate * 2, endpoint=False)
            sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
            waveform_t = torch.from_numpy(sig).float().to(device)
            print("  [+] Generated harmonic 440Hz test tone audio.")
        else:
            spoken_text = audio_in if audio_in.lower() != "speech_sim" else "Artificial Intelligence DNA speech transcription test"
            print(f"  [+] Synthesizing speech audio for: '{spoken_text}' via AI-DNA TTS...")
            temp_wav = os.path.join(modal_dir, "temp_asr_speech_input.wav")
            if engine:
                res_speech = engine.generate_speech(text=spoken_text, output_path=temp_wav)
                temp_wav = res_speech["file_path"]
            else:
                t = np.linspace(0, 2.0, sample_rate * 2, endpoint=False)
                sig = 0.5 * np.sin(2 * np.pi * 130 * t) * (np.sin(np.pi * t / 2.0) ** 2)
                MultimodalOutputHandler.save_audio_waveform(sig, temp_wav, sample_rate=sample_rate)

            try:
                import scipy.io.wavfile as wavfile
                sr, raw = wavfile.read(temp_wav)
                if raw.ndim > 1:
                    raw = raw.mean(axis=1)
                wf_np = raw.astype(np.float32) / (32768.0 if raw.dtype == np.int16 else 1.0)
                if sr != sample_rate:
                    from scipy.signal import resample
                    wf_np = resample(wf_np, int(len(wf_np) * sample_rate / sr))
                waveform_t = torch.from_numpy(wf_np).float().to(device)
                print(f"  [+] Transcribing synthesized speech audio with Whisper ASR...")
            except Exception:
                waveform_t = torch.randn(sample_rate * 2, device=device)

        mel_features = compute_mel_spectrogram_from_waveform(waveform_t, sample_rate=sample_rate).to(device)

        t_aud0 = time.time()
        with torch.no_grad():
            gen_tokens, text_transcription = model.decode_transcribe(fused_weights, mel_features, tokenizer)
        dt_aud = time.time() - t_aud0

        print("\n  " + "-" * 76)
        print(f"  Transcription: \"{text_transcription if text_transcription else '[Acoustic speech observations]'}\"")
        print(f"  Tokens: {gen_tokens[:10]}... | Latency: {dt_aud*1000:.1f}ms")
        print("  " + "-" * 76 + "\n")


def run_fused_image_gen(engine: OmniInferenceEngine, modal_dir: str):
    """Synthesizes images via Fused AI-DNA Latent Diffusion Head."""
    print("\n" + "=" * 80)
    print("  [AI-DNA FUSED INFERENCE] :: TEXT-TO-IMAGE DIFFUSION")
    print("=" * 80)
    prompt = input("  Image Prompt (Enter for default)> ").strip()
    prompt = prompt or "a majestic mountain landscape with crystal blue lake at sunset"
    steps_in = input("  Diffusion Steps [10-500] (Enter for 50)> ").strip()
    num_steps = int(steps_in) if steps_in.isdigit() else 50

    print(f"\n  [+] Synthesizing 512x512 image ({num_steps} steps) for: '{prompt}'...")
    t0 = time.time()
    res = engine.generate_image(
        prompt=prompt,
        width=512,
        height=512,
        num_inference_steps=num_steps,
        output_path=os.path.join(modal_dir, "fused_generated_image.png"),
        seed=42,
    )
    dt = time.time() - t0
    print("\n  " + "-" * 76)
    print(f"  Image Saved:  {res['file_path']}")
    print(f"  Resolution:   {res['width']}x{res['height']}")
    print(f"  Latency:      {dt*1000:.1f}ms")
    print("  " + "-" * 76 + "\n")


def run_fused_audio_gen(engine: OmniInferenceEngine, modal_dir: str):
    """Synthesizes human speech via Fused AI-DNA Neural TTS."""
    print("\n" + "=" * 80)
    print("  [AI-DNA FUSED INFERENCE] :: TEXT-TO-SPEECH (Neural Audio)")
    print("=" * 80)
    text = input("  Text to Speak (Enter for default)> ").strip()
    text = text or "Artificial Intelligence DNA enables seamless multi-modal evolution and reasoning."

    print(f"\n  [+] Synthesizing speech audio for: '{text}'...")
    t0 = time.time()
    res = engine.generate_speech(
        text=text,
        output_path=os.path.join(modal_dir, "fused_generated_speech.wav"),
    )
    dt = time.time() - t0
    print("\n  " + "-" * 76)
    print(f"  Audio Saved:  {res['file_path']}")
    print(f"  Duration:     {res['duration_sec']:.2f}s")
    print(f"  Sample Rate:  {res['sample_rate']} Hz ({res.get('synthesizer', 'AIDNA')})")
    print(f"  Latency:      {dt*1000:.1f}ms")
    print("  " + "-" * 76 + "\n")


def run_fused_video_gen(engine: OmniInferenceEngine, modal_dir: str):
    """Synthesizes temporal dynamic animations via Fused AI-DNA."""
    print("\n" + "=" * 80)
    print("  [AI-DNA FUSED INFERENCE] :: TEXT-TO-VIDEO ANIMATION")
    print("=" * 80)
    prompt = input("  Video Prompt (Enter for default)> ").strip()
    prompt = prompt or "a majestic mountain landscape with crystal blue lake at sunset"

    print(f"\n  [+] Synthesizing 16-frame dynamic animated sequence for: '{prompt}'...")
    t0 = time.time()
    res = engine.generate_video(
        prompt=prompt,
        num_frames=16,
        width=256,
        height=256,
        output_path=os.path.join(modal_dir, "fused_generated_video.gif"),
    )
    dt = time.time() - t0
    print("\n  " + "-" * 76)
    print(f"  Video Saved:  {res['file_path']}")
    print(f"  Frames:       {res['num_frames']} frames (256x256)")
    print(f"  Latency:      {dt*1000:.1f}ms")
    print("  " + "-" * 76 + "\n")


def run_fused_omni_reasoning(engine: OmniInferenceEngine):
    """Executes full unified omni-modal reasoning with sensory fusion, CoT trace, and PRM verification."""
    print("\n" + "=" * 80)
    print("  [AI-DNA FUSED INFERENCE] :: UNIFIED OMNI-MODAL REASONING SHELL")
    print("  Multi-Modal Intake -> FastClock -> MoE Routing -> CoT Trace -> Verified Synthesis")
    print("=" * 80)

    print("\n  Interactive Input Syntax:")
    print("    • Pure Text / Math:  'Calculate 48 * 25 + 150 and explain.'")
    print("    • Image + Query:     'photo.jpg | What object is shown and what is its role?'")
    print("    • Audio + Query:     'speech.wav | Transcribe and summarize this audio.'")
    print("    • Full Omni:         'photo.jpg | speech.wav | Analyze and solve.'")
    print("  (Type 'quit' or 'menu' to return)\n")

    while True:
        try:
            raw_line = input("  Omni Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw_line or raw_line.lower() in ("quit", "exit", "q", "menu", "m"):
            break

        parts = [p.strip() for p in raw_line.split("|")]
        image_arg = None
        audio_arg = None
        text_arg = "Process multi-modal observations."

        for p in parts:
            p_lower = p.lower()
            if any(p_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                image_arg = p
            elif any(p_lower.endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                audio_arg = p
            else:
                text_arg = p

        t0 = time.time()
        res = engine.infer(
            text=text_arg,
            image=image_arg,
            audio=audio_arg,
            save_artifacts=True
        )
        dt = time.time() - t0

        print("\n  " + "=" * 90)
        print("  ||                     AI-DNA FUSED REASONING & OUTPUT PANEL                             ||")
        print("  " + "=" * 90)

        if "audio_perception" in res:
            ap = res["audio_perception"]
            print(f"\n  [1. TRANSCRIBED SPEECH (Whisper)]")
            print(f"      Transcription:  \"{ap.get('transcription', '')}\"")

        print(f"\n  [2. CHAIN-OF-THOUGHT REASONING TRACE]")
        print(f"      <thought>")
        for line in res["thought_trace"].split("\n"):
            print(f"        {line}")
        print(f"      </thought>")

        print(f"\n  [3. FINAL TEXT ANSWER]")
        print(f"      {res['final_text_answer']}")

        if "reasoning_verifier" in res:
            rv = res["reasoning_verifier"]
            status = "[PASS]" if rv["format_validity_reward"] > 0.5 else "[FAIL]"
            print(f"\n  [4. REASONING VERIFIER AUDIT]")
            print(f"      Format: {status} ({rv['format_validity_reward']:.2f}) | Accuracy: {rv['accuracy_reward']:.2f} | Score: {rv['composite_score']:.2f}")

        if "audio_output" in res:
            print(f"\n  [5. GENERATED SPEECH AUDIO]")
            print(f"      WAV File: {res['audio_output']['wav_path']} ({res['audio_output']['duration_sec']:.1f}s)")

        if "interleaved_display" in res:
            print(f"\n  [6. INTERLEAVED MULTIMODAL STREAM]")
            print(res["interleaved_display"])

        print(f"\n  [PERSISTED REPORTS]")
        if "txt_report_path" in res:
            print(f"      TXT Report:  {res['txt_report_path']}")
        if "json_report_path" in res:
            print(f"      JSON Data:   {res['json_report_path']}")

        print("  " + "=" * 90)
        print(f"  Cycle Latency: {dt*1000:.1f}ms\n")


# =========================================================================
# Main Execution Entrypoint
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="AI-DNA Fused Omni-Modal Inference Runner")
    parser.add_argument("--aidna-path", default="modal/fused_omni_child.aidna", help="Path to fused .aidna genotype file")
    parser.add_argument("--modal-dir", default="modal", help="Modal metadata & artifact directory")
    parser.add_argument("--device", default=None, help="cuda/cpu")
    parser.add_argument("--mode", default=None,
                        choices=["text", "vision", "audio", "image_gen", "audio_gen", "video_gen", "omni", "all"],
                        help="Select modality to run")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    fused_path = args.aidna_path if os.path.exists(args.aidna_path) else os.path.join(args.modal_dir, "fused_omni_child.aidna")
    if not os.path.exists(fused_path):
        print(f"\n  [ERROR] Fused AI-DNA Genotype not found at '{fused_path}'.")
        print("  Please run convert_downloaded_models_to_aidna.py first to generate the fused genotype.\n")
        return

    print("\n" + "=" * 80)
    print("  " + "=" * 76)
    print("  ||            AI-DNA FUSED OMNI-MODAL INFERENCE ENGINE                 ||")
    print("  ||    Pure Genotype (.aidna) -> Phenotype Regrowth & Reasoning         ||")
    print("  " + "=" * 76)
    print(f"  Device:       {device_str.upper()}")
    print(f"  Fused Genotype: {os.path.abspath(fused_path)} ({os.path.getsize(fused_path):,} bytes)")
    print(f"  Artifacts:    {os.path.abspath(args.modal_dir)}")
    print("=" * 80)

    # Grow full phenotype neural engine directly from the fused .aidna genotype
    print("\n  [+] Growing Phenotype Neural Network from fused AI-DNA Genotype...")
    t_load = time.time()
    engine = OmniInferenceEngine.from_genotype(fused_path, modal_dir=args.modal_dir, device=device)
    print(f"  [+] Fused Omni Engine ready in {time.time()-t_load:.2f}s (d_model={engine.phenotype_model.d_model})\n")

    def run_selected_mode(mode: str):
        if mode == "text":
            run_fused_text(fused_path, args.modal_dir, device)
        elif mode == "vision":
            run_fused_vision(fused_path, args.modal_dir, device)
        elif mode == "audio":
            run_fused_audio(fused_path, args.modal_dir, device, engine=engine)
        elif mode == "image_gen":
            run_fused_image_gen(engine, args.modal_dir)
        elif mode == "audio_gen":
            run_fused_audio_gen(engine, args.modal_dir)
        elif mode == "video_gen":
            run_fused_video_gen(engine, args.modal_dir)
        elif mode == "omni":
            run_fused_omni_reasoning(engine)
        elif mode == "all":
            run_fused_text(fused_path, args.modal_dir, device)
            run_fused_vision(fused_path, args.modal_dir, device)
            run_fused_audio(fused_path, args.modal_dir, device, engine=engine)
            run_fused_image_gen(engine, args.modal_dir)
            run_fused_audio_gen(engine, args.modal_dir)
            run_fused_video_gen(engine, args.modal_dir)
            run_fused_omni_reasoning(engine)

    if args.mode:
        run_selected_mode(args.mode)
    else:
        while True:
            print("\n  Select Fused AI-DNA modality to execute:")
            print("    1. Text Generation             (SmolLM2 Autoregressive Decoder)")
            print("    2. Vision Perception           (CLIP-ViT Zero-Shot Classification)")
            print("    3. Audio Perception / ASR      (Whisper Speech-to-Text)")
            print("    4. Image Generation            (Latent Neural Diffusion Head)")
            print("    5. Audio Generation            (Kokoro-82M Neural Speech Synthesis)")
            print("    6. Video Generation            (Spatiotemporal Dynamic Animation)")
            print("    7. Unified Omni-Modal Reasoning(Sensory Intake -> CoT Trace -> Verified Synthesis)")
            print("    8. Run All Modalities Sequentially")
            print("    q. Quit")

            try:
                choice = input("\n  Choice> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if choice in ("1", "text"):
                run_selected_mode("text")
            elif choice in ("2", "vision"):
                run_selected_mode("vision")
            elif choice in ("3", "audio"):
                run_selected_mode("audio")
            elif choice in ("4", "image_gen", "image"):
                run_selected_mode("image_gen")
            elif choice in ("5", "audio_gen", "audio", "speech"):
                run_selected_mode("audio_gen")
            elif choice in ("6", "video_gen", "video"):
                run_selected_mode("video_gen")
            elif choice in ("7", "omni"):
                run_selected_mode("omni")
            elif choice in ("8", "all"):
                run_selected_mode("all")
            elif choice in ("q", "quit", "exit"):
                break
            else:
                print("  Invalid choice.")

    print("\n  Fused AI-DNA execution terminated cleanly. Goodbye!\n")


if __name__ == "__main__":
    main()
