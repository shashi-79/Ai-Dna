"""
Open LLM Leaderboard Evaluator Submission Module for AI-DNA.
Submits evaluation requests to the Hugging Face Open LLM Leaderboard queue.
"""

import json
from typing import Dict, Any, Optional
from huggingface_hub import HfApi

from .uploader import get_hf_token

REQUESTS_REPO = "open-llm-leaderboard/requests"


def submit_to_leaderboard(
    model_id: str,
    precision: str = "bfloat16",
    model_type: str = "fine-tuned",
    base_model: str = "",
    weight_type: str = "Original",
    token: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Submits a model evaluation request to the Open LLM Leaderboard.
    
    Args:
        model_id: Full repository ID, e.g. 'username/model-name'.
        precision: Model precision: 'bfloat16', 'float16', '8bit', '4bit'.
        model_type: Model type: 'base', 'fine-tuned', 'chat'.
        base_model: Base model ID if LoRA or adapter.
        weight_type: 'Original' or 'Adapter'.
        token: Hugging Face WRITE token (defaults to HF_TOKEN env var).
    """
    token = get_hf_token(token)
    if not token:
        raise ValueError(
            "Hugging Face token not found! Set the HF_TOKEN environment variable "
            "or pass token='your_token'."
        )

    try:
        org_or_user, model_name = model_id.split("/")
    except ValueError:
        raise ValueError("model_id must be in 'username/model-name' format.")

    api = HfApi(token=token)

    submission_data = {
        "model": model_id,
        "base_model": base_model,
        "revision": "main",
        "precision": precision,
        "weight_type": weight_type,
        "status": "PENDING",
        "submitted_time": "",
        "model_type": model_type,
        "likes": 0,
        "params": 0,
    }

    json_bytes = json.dumps(submission_data, indent=4).encode("utf-8")
    filename = f"{model_name}_eval_request_False_{precision}_{weight_type}.json"
    path_in_repo = f"{org_or_user}/{filename}"

    if verbose:
        print(f"[+] Creating Open LLM Leaderboard evaluation request: {path_in_repo}...")

    api.upload_file(
        path_or_fileobj=json_bytes,
        path_in_repo=path_in_repo,
        repo_id=REQUESTS_REPO,
        repo_type="dataset",
        commit_message=f"Add {model_id} to Open LLM Leaderboard evaluation queue",
        token=token,
    )

    request_url = f"https://huggingface.co/datasets/{REQUESTS_REPO}/blob/main/{path_in_repo}"
    if verbose:
        print(f"[SUCCESS] Submitted successfully to Open LLM Leaderboard!")
        print(f"          Request URL: {request_url}")

    return request_url
