"""
Download Script for Open-Source Modality Models into ./modal
Supports downloading multiple models per modality category, including a curated suite of 5 open-source LLMs:

1. Text-to-Text (LLMs):
   - SmolLM2-135M-Instruct  (HuggingFaceTB/SmolLM2-135M-Instruct, ~270MB)
   - Qwen2.5-0.5B-Instruct  (Qwen/Qwen2.5-0.5B-Instruct, ~990MB)
   - SmolLM2-360M-Instruct  (HuggingFaceTB/SmolLM2-360M-Instruct, ~720MB)
   - TinyLlama-1.1B-Chat    (TinyLlama/TinyLlama-1.1B-Chat-v1.0, ~1.2GB)
   - OPT-125M               (facebook/opt-125m, ~250MB)
2. Vision Perception / Zero-Shot: CLIP-ViT-B/32, SmolVLM-256M
3. Speech-to-Text / Audio Perception: Whisper-tiny, Whisper-base
4. Text-to-Image Generation: Segmind Tiny-SD (~246MB U-Net)
5. Text-to-Speech / Audio Generation: Kokoro-82M (~320MB)
"""

import os
import sys
import argparse
from typing import Dict, Any, List

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed. Please run:")
    print("        pip install huggingface_hub")
    sys.exit(1)


# Comprehensive Multi-Model Catalog
MODEL_CATALOG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "text": {
        "smollm2-135m": {
            "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "folder": "modal/text_model",
            "params": "135M",
            "size": "~270 MB",
            "architecture": "Llama-based",
            "description": "Ultra-fast compact instruct LLM (SmolLM2-135M-Instruct)",
            "is_default": True,
        },
        "qwen2.5-0.5b": {
            "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "folder": "modal/text_models/qwen2.5-0.5b",
            "params": "490M",
            "size": "~990 MB",
            "architecture": "Qwen2.5 (RoPE + GQA)",
            "description": "SOTA small reasoning & multilingual instruct LLM (Qwen2.5-0.5B)",
            "is_default": False,
        },
        "smollm2-360m": {
            "repo_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "folder": "modal/text_models/smollm2-360m",
            "params": "360M",
            "size": "~720 MB",
            "architecture": "Llama-based",
            "description": "Balanced high-performance instruct LLM (SmolLM2-360M)",
            "is_default": False,
        },
        "tinyllama-1.1b": {
            "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "folder": "modal/text_models/tinyllama-1.1b",
            "params": "1.1B",
            "size": "~1.2 GB",
            "architecture": "Llama-based",
            "description": "Standard 1.1B open conversational chat benchmark (TinyLlama-1.1B)",
            "is_default": False,
        },
        "opt-125m": {
            "repo_id": "facebook/opt-125m",
            "folder": "modal/text_models/opt-125m",
            "params": "125M",
            "size": "~250 MB",
            "architecture": "OPT-Decoder",
            "description": "Foundational autoregressive decoder LLM (Meta OPT-125M)",
            "is_default": False,
        },
    },
    "vision": {
        "clip-vit-b32": {
            "repo_id": "openai/clip-vit-base-patch32",
            "folder": "modal/vision_model",
            "params": "151M",
            "size": "~330 MB",
            "architecture": "ViT-B/32",
            "description": "Image-to-Text / Vision Encoder (CLIP-ViT-B/32)",
            "is_default": True,
        },
        "smolvlm-256m": {
            "repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "folder": "modal/vision_models/smolvlm-256m",
            "params": "256M",
            "size": "~500 MB",
            "architecture": "SmolVLM",
            "description": "Compact Multimodal Vision-Language Model (SmolVLM-256M)",
            "is_default": False,
        },
    },
    "audio": {
        "whisper-tiny": {
            "repo_id": "openai/whisper-tiny",
            "folder": "modal/audio_model",
            "params": "39M",
            "size": "~150 MB",
            "architecture": "Encoder-Decoder ASR",
            "description": "Speech Perception / ASR Model (Whisper-tiny)",
            "is_default": True,
        },
        "whisper-base": {
            "repo_id": "openai/whisper-base",
            "folder": "modal/audio_models/whisper-base",
            "params": "74M",
            "size": "~290 MB",
            "architecture": "Encoder-Decoder ASR",
            "description": "Higher-accuracy Speech Perception ASR (Whisper-base)",
            "is_default": False,
        },
    },
    "image_gen": {
        "tiny-sd": {
            "repo_id": "segmind/tiny-sd",
            "folder": "modal/image_gen_model",
            "params": "323M",
            "size": "~246 MB U-Net",
            "architecture": "Latent Diffusion",
            "description": "Text-to-Image Latent Diffusion Model (Tiny-SD)",
            "is_default": True,
        },
    },
    "audio_gen": {
        "kokoro-82m": {
            "repo_id": "hexgrad/Kokoro-82M",
            "folder": "modal/audio_gen_model",
            "params": "82M",
            "size": "~320 MB",
            "architecture": "StyleTTS2-derived",
            "description": "Text-to-Speech / Audio Synthesis Model (Kokoro-82M)",
            "is_default": True,
        },
    },
}

# Backward compatibility alias
DEFAULT_MODELS = {
    mod: next(cfg for cfg in models.values() if cfg.get("is_default", False))
    for mod, models in MODEL_CATALOG.items()
}


def is_model_downloaded(folder_path: str) -> bool:
    """Checks if a model directory exists and has configuration and weight files."""
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return False
    files = os.listdir(folder_path)
    has_config = "config.json" in files or any(f.endswith("config.json") for f in files)
    has_weights = any(
        f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt")
        for f in files
    )
    return has_config and has_weights


def list_catalog():
    """Prints a formatted table of all models in the catalog and their download status."""
    print("=" * 95)
    print("  AI-DNA OPEN-SOURCE MODALITY MODEL CATALOG")
    print("=" * 95)
    print(f"  {'MODALITY':<12} {'MODEL KEY':<16} {'PARAMS':<8} {'SIZE':<14} {'STATUS':<12} {'REPO ID'}")
    print("  " + "-" * 91)

    for mod, models in MODEL_CATALOG.items():
        for key, info in models.items():
            downloaded = is_model_downloaded(info["folder"])
            status_str = "[INSTALLED]" if downloaded else "[NOT FOUND]"
            default_tag = " (def)" if info.get("is_default") else ""
            print(f"  {mod.upper():<12} {key + default_tag:<16} {info['params']:<8} {info['size']:<14} {status_str:<12} {info['repo_id']}")
    print("=" * 95)


def download_modality(modality: str, repo_id: str, dest_dir: str):
    """Downloads snapshot of a model repository into dest_dir."""
    print(f"\n[DOWNLOAD] Starting download for {modality.upper()} model...")
    print(f"           Repository:  {repo_id}")
    print(f"           Destination: {os.path.abspath(dest_dir)}")
    os.makedirs(dest_dir, exist_ok=True)

    if is_model_downloaded(dest_dir):
        print(f"[INFO] Model already downloaded in {dest_dir}. Checking for updates/completeness...")

    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite", "*.mlmodel"],
        )
        print(f"[SUCCESS] {modality.upper()} model downloaded successfully to: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {modality.upper()} model ({repo_id}): {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download open-source modality test models into ./modal folder.")
    parser.add_argument(
        "--modality",
        choices=["all", "text", "llm", "vision", "audio", "image_gen", "audio_gen", "generative"],
        default="generative",
        help="Which modality model to download: text/llm, vision, audio, image_gen, audio_gen, generative, or all",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model key to download within modality, or 'all' to download all models in the modality category",
    )
    parser.add_argument(
        "--all-llms",
        action="store_true",
        help="Convenience shortcut: download all 5 curated open-source LLM models in the text category",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models in the catalog and check local installation status",
    )
    # Legacy specific repo overrides
    parser.add_argument("--text-repo", default=None, help="Custom HuggingFace repo ID for default text model")
    parser.add_argument("--vision-repo", default=None, help="Custom HuggingFace repo ID for default vision model")
    parser.add_argument("--audio-repo", default=None, help="Custom HuggingFace repo ID for default audio model")
    parser.add_argument("--image-gen-repo", default=None, help="Custom HuggingFace repo ID for default image gen model")
    parser.add_argument("--audio-gen-repo", default=None, help="Custom HuggingFace repo ID for default audio gen model")
    args = parser.parse_args()

    if args.list:
        list_catalog()
        return

    print("=" * 75)
    print("  AI-DNA OPEN-SOURCE MODALITY DOWNLOADER")
    print("=" * 75)

    download_queue: List[Dict[str, Any]] = []

    if args.all_llms:
        print("[MODE] Downloading all 5 curated LLM models in the text category...")
        for key, info in MODEL_CATALOG["text"].items():
            download_queue.append({
                "modality": "text",
                "key": key,
                "repo_id": info["repo_id"],
                "folder": info["folder"],
                "description": info["description"],
            })
    else:
        modality = "text" if args.modality == "llm" else args.modality

        if modality == "all":
            targets = ["text", "vision", "audio", "image_gen", "audio_gen"]
        elif modality == "generative":
            targets = ["image_gen", "audio_gen"]
        else:
            targets = [modality]

        for mod in targets:
            mod_models = MODEL_CATALOG[mod]
            if args.model == "all":
                for key, info in mod_models.items():
                    download_queue.append({
                        "modality": mod,
                        "key": key,
                        "repo_id": info["repo_id"],
                        "folder": info["folder"],
                        "description": info["description"],
                    })
            elif args.model and args.model in mod_models:
                info = mod_models[args.model]
                download_queue.append({
                    "modality": mod,
                    "key": args.model,
                    "repo_id": info["repo_id"],
                    "folder": info["folder"],
                    "description": info["description"],
                })
            else:
                # Default model for this modality, with legacy repo override support
                default_cfg = DEFAULT_MODELS[mod]
                attr_name = f"{mod.replace('-', '_')}_repo"
                custom_repo = getattr(args, attr_name, None)
                repo = custom_repo if custom_repo else default_cfg["repo_id"]
                download_queue.append({
                    "modality": mod,
                    "key": "default",
                    "repo_id": repo,
                    "folder": default_cfg["folder"],
                    "description": default_cfg["description"],
                })

    results = {}
    for item in download_queue:
        label = f"{item['modality'].upper()} ({item['key']})"
        success = download_modality(item["modality"], item["repo_id"], item["folder"])
        results[label] = {
            "success": success,
            "folder": item["folder"],
            "repo_id": item["repo_id"],
            "description": item["description"],
        }

    print("\n" + "=" * 80)
    print("  DOWNLOAD SUMMARY")
    print("=" * 80)
    for label, res in results.items():
        status = "COMPLETED" if res["success"] else "FAILED"
        print(f"  - {label:<24} : [{status}] -> {res['folder']} ({res['repo_id']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
