"""
Unified Batched Evaluation Engine for AI-DNA.
Supports resource auto-tuning, batched parallel inference, and standardized task evaluation
across Hugging Face models and direct .aidna genetic containers.
"""

import os
import sys
import time
import math
import json
import ctypes
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import TASK_LOADERS, TASK_SCORERS


def auto_detect_resources(model_dir: Optional[str] = None) -> Dict[str, Any]:
    """Auto-detects GPU VRAM and system RAM, computing optimal batch size for throughput."""
    has_cuda = torch.cuda.is_available()
    free_vram = 0
    total_vram = 0
    gpu_name = "CPU Only"

    if has_cuda:
        try:
            free_vram, total_vram = torch.cuda.mem_get_info(0)
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            free_vram = 4 * 1024**3
            total_vram = 8 * 1024**3

    # System RAM detection
    free_ram = 8 * 1024**3
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            free_ram = stat.ullAvailPhys
    except Exception:
        pass

    # Model size estimation
    model_bytes = 270 * 1024 * 1024
    if model_dir and os.path.exists(model_dir):
        total = 0
        for root, _, files in os.walk(model_dir):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".pt")):
                    total += os.path.getsize(os.path.join(root, f))
        if total > 0:
            model_bytes = total

    activation_per_item = 32 * 1024 * 1024
    if has_cuda:
        usable = max(0, free_vram - model_bytes - 512 * 1024 * 1024)
    else:
        usable = free_ram * 0.40

    raw_batch = max(1, int(usable / max(activation_per_item, 1)))
    batch_size = min(raw_batch, 64)

    device = "cuda" if has_cuda else "cpu"
    cpu_count = os.cpu_count() or 4
    dataloader_workers = max(0, min(cpu_count // 2, 4))

    return {
        "device": device,
        "gpu_name": gpu_name,
        "free_vram_gb": round(free_vram / 1e9, 2),
        "total_vram_gb": round(total_vram / 1e9, 2),
        "free_ram_gb": round(free_ram / 1e9, 2),
        "cpu_count": cpu_count,
        "model_size_mb": round(model_bytes / 1e6, 1),
        "batch_size": batch_size,
        "dataloader_workers": dataloader_workers,
    }


def auto_batch(model_dir: Optional[str] = None, device: Optional[str] = None) -> int:
    """Convenience helper returning the recommended batch size for a given model path and device."""
    return auto_detect_resources(model_dir=model_dir)["batch_size"]



class BatchedInferenceEngine:
    """Wraps a Hugging Face AutoModel / AutoTokenizer for parallel batched evaluation."""

    def __init__(self, model, tokenizer, device: str = "cpu", batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 64) -> List[str]:
        """Runs batched generation on prompts."""
        if not prompts:
            return []
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.pad_id,
            )

        responses = []
        input_len = enc["input_ids"].shape[1]
        for seq in out:
            gen_tokens = seq[input_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            responses.append(text)
        return responses


def evaluate_hf_model(
    model_dir: str,
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates a Hugging Face model folder across specified benchmark tasks.
    Returns dictionary with task accuracies, average score, and elapsed times.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    res = auto_detect_resources(model_dir)
    dev = device or res["device"]
    bs = batch_size or res["batch_size"]

    if verbose:
        print(f"\n[+] Loading model from: {model_dir} on {dev.upper()} (Batch Size: {bs})...")

    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if dev == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, trust_remote_code=True).to(dev)
    model.eval()

    engine = BatchedInferenceEngine(model, tokenizer, device=dev, batch_size=bs)
    target_tasks = tasks or ["mmlu", "gsm8k", "arc", "ifeval"]

    results = {"model": os.path.basename(model_dir), "tasks": {}, "summary": {}}
    total_correct = 0
    total_questions = 0

    for task_name in target_tasks:
        if task_name not in TASK_LOADERS:
            continue
        loader = TASK_LOADERS[task_name]
        scorer = TASK_SCORERS[task_name]
        questions = loader(limit=limit)

        if not questions:
            continue

        t0 = time.time()
        task_correct = 0
        max_tokens = 16 if task_name in ["mmlu", "arc"] else (64 if task_name == "gsm8k" else 128)

        # Batch processing
        for i in range(0, len(questions), bs):
            chunk = questions[i: i + bs]
            prompts = [q["prompt"] for q in chunk]
            responses = engine.generate_batch(prompts, max_new_tokens=max_tokens)

            for q, resp in zip(chunk, responses):
                is_correct = scorer(resp, q)
                task_correct += int(is_correct)

        elapsed = time.time() - t0
        accuracy = (task_correct / len(questions)) * 100.0 if questions else 0.0
        q_per_sec = len(questions) / max(elapsed, 1e-4)

        results["tasks"][task_name] = {
            "accuracy": round(accuracy, 2),
            "correct": task_correct,
            "total": len(questions),
            "time_sec": round(elapsed, 2),
            "qps": round(q_per_sec, 2),
        }
        total_correct += task_correct
        total_questions += len(questions)

        if verbose:
            print(f"  [{task_name.upper():<6}] Accuracy: {accuracy:6.2f}% ({task_correct}/{len(questions)}) | {q_per_sec:.1f} q/s ({elapsed:.1f}s)")

    avg_accuracy = (total_correct / total_questions * 100.0) if total_questions > 0 else 0.0
    results["summary"] = {
        "average_accuracy": round(avg_accuracy, 2),
        "total_correct": total_correct,
        "total_questions": total_questions,
    }

    return results
