"""
Hugging Face Hub Model Uploader & Open LLM Leaderboard Submitter.

Programmatically creates a repository and uploads the entire model folder
(config.json, generation_config.json, model.safetensors, tokenizer, and README.md)
to the Hugging Face Model Hub, with optional 1-click submission to the Open LLM Leaderboard.

Usage:
    # Set your HF token via environment variable or --token
    export HF_TOKEN="hf_..."   (or in PowerShell: $env:HF_TOKEN="hf_...")

    # Upload local folder to your HF account
    python upload_to_huggingface.py --repo-id "your-username/my-custom-llm" --folder "./my_llm_folder"

    # Upload and submit to Open LLM Leaderboard
    python upload_to_huggingface.py --repo-id "your-username/my-custom-llm" --submit-leaderboard

    # Private repository
    python upload_to_huggingface.py --repo-id "your-username/my-custom-llm" --private
"""

import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ai_dna.hub import upload_model_folder, submit_to_leaderboard, validate_model_folder


def main():
    parser = argparse.ArgumentParser(
        description="Upload a converted SafeTensors model directory to Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r", "--repo-id",
        required=True,
        help="Hugging Face repository ID (format: 'username/model-name').",
    )
    parser.add_argument(
        "-f", "--folder",
        default="./my_llm_folder",
        help="Local folder path containing model.safetensors, config.json, etc.",
    )
    parser.add_argument(
        "-t", "--token",
        default=None,
        help="Hugging Face access token (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository instead of public.",
    )
    parser.add_argument(
        "-m", "--commit-message",
        default="Upload AI-DNA SafeTensors model and leaderboard model card",
        help="Git commit message for the upload.",
    )
    parser.add_argument(
        "--submit-leaderboard",
        action="store_true",
        help="Also submit the model to the official Open LLM Leaderboard evaluation queue.",
    )
    parser.add_argument(
        "--precision",
        default="bfloat16",
        choices=["bfloat16", "float16", "8bit", "4bit"],
        help="Model precision for leaderboard submission.",
    )
    parser.add_argument(
        "--model-type",
        default="fine-tuned",
        choices=["base", "fine-tuned", "chat"],
        help="Model type for leaderboard submission.",
    )

    args = parser.parse_args()

    # Step 1: Upload Model
    url = upload_model_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
        verbose=True,
    )

    # Step 2: Submit to Leaderboard if requested
    if args.submit_leaderboard:
        if args.private:
            print("\n[!] Warning: Models submitted to the Open LLM Leaderboard must be public.")
        submit_to_leaderboard(
            model_id=args.repo_id,
            precision=args.precision,
            model_type=args.model_type,
            token=args.token,
            verbose=True,
        )


if __name__ == "__main__":
    main()
