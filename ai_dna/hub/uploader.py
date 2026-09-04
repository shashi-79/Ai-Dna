"""
Hugging Face Model Hub Uploader Module for AI-DNA.
Validates required SafeTensors folder structures, authenticates securely,
and pushes models to the Hugging Face Hub.
"""

import os
import sys
from typing import List, Optional
from huggingface_hub import HfApi

MANDATORY_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
]


def validate_model_folder(folder_path: str) -> List[str]:
    """Validates that standard auto-class files exist in the model folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Model folder not found: {folder_path}")

    missing = []
    for f in MANDATORY_FILES:
        full_p = os.path.join(folder_path, f)
        if not os.path.exists(full_p):
            missing.append(f)
    return missing


def get_hf_token(token: Optional[str] = None) -> Optional[str]:
    """Retrieves Hugging Face token securely from parameter or environment variables."""
    resolved = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if resolved:
        resolved = resolved.strip().strip('"').strip("'")
    return resolved


def upload_model_folder(
    folder_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload AI-DNA SafeTensors LLM checkpoint and leaderboard card",
    verbose: bool = True,
) -> str:
    """
    Creates or updates a Hugging Face repository and uploads the entire model folder.
    Returns the repository URL.
    """
    token = get_hf_token(token)
    if not token:
        raise ValueError(
            "Hugging Face token not found! Set the HF_TOKEN environment variable "
            "or pass --token 'your_token'."
        )

    api = HfApi(token=token)

    # 1. Validate structure
    missing = validate_model_folder(folder_path)
    if missing and verbose:
        print(f"[!] Warning: Missing standard files in {folder_path}: {missing}")
        print("    Proceeding, but Hugging Face auto-classes or leaderboards may require them.")

    # 2. Verify identity
    user_info = api.whoami(token=token)
    username = user_info.get("name", user_info.get("username", "User"))
    if verbose:
        print(f"[+] Authenticated successfully as: @{username}")
        print(f"[+] Initializing repository '{repo_id}' (private={private})...")

    # 3. Create repository if needed
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=private,
        token=token,
    )

    # 4. Upload folder
    if verbose:
        print(f"[+] Uploading files from '{folder_path}' to '{repo_id}'...")

    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        token=token,
    )

    full_url = f"https://huggingface.co/{repo_id}"
    if verbose:
        print(f"[SUCCESS] Model pushed successfully: {full_url}")
    return full_url
