"""
Zero-Dependency Weight & Configuration Loaders.
Reads .safetensors and .bin / .pt files using standard library struct, numpy, and torch,
without requiring external safetensors or transformers packages.
"""

import os
import sys
import glob
import json
import struct
from typing import Dict, Any, Optional
import numpy as np
import torch


def load_safetensors_file(
    filepath: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Reads a .safetensors file using Python's standard struct + numpy / torch.
    Eliminates external safetensors pip package requirement.
    """
    tensors = {}
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))

        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "I64": np.int64,
            "I32": np.int32,
            "I16": np.int16,
            "I8": np.int8,
            "U8": np.uint8,
        }
        for k, v in header.items():
            if k == "__metadata__":
                continue
            raw_dtype = v.get("dtype", "F32")
            shape = v["shape"]
            start, end = v["data_offsets"]
            f.seek(8 + header_len + start)
            raw = f.read(end - start)
            if raw_dtype == "BF16":
                t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).to(device)
            else:
                dtype = dtype_map.get(raw_dtype, np.float32)
                arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
                t = torch.from_numpy(arr.copy()).to(device)
            tensors[k] = t
    return tensors


def load_model_weights(
    folder_path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Scans folder_path for model weights: prefers .safetensors, falls back to .bin / .pt.
    Returns mapping of weight name -> torch.Tensor on device.
    """
    if not os.path.exists(folder_path):
        return {}
    st_files = sorted(glob.glob(os.path.join(folder_path, "*.safetensors")))
    if st_files:
        w = {}
        for sf in st_files:
            w.update(load_safetensors_file(sf, device=device))
        return w
    bin_files = glob.glob(os.path.join(folder_path, "*.bin")) + glob.glob(os.path.join(folder_path, "*.pt"))
    for bf in bin_files:
        try:
            raw_w = torch.load(bf, map_location=device, weights_only=False)
            if isinstance(raw_w, dict):
                return {k: v.to(device) for k, v in raw_w.items() if isinstance(v, torch.Tensor)}
        except Exception:
            pass
    return {}


def load_config(folder_path: str) -> Dict[str, Any]:
    """Loads config.json from model folder, returns empty dict if missing."""
    config_path = os.path.join(folder_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def reconstruct_weights_and_genotype(
    aidna_path: str,
    key_filter: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """
    Reconstructs original-format weight tensors strictly from a fused .aidna genotype file.
    Returns (reconstructed_state_dict, genotype).
    Handles SVD low-rank factor decomposition (W = A @ B), tensor reshaping via meta.<key>.orig_shape,
    and direct raw and modal parameter copying.
    """
    from ai_dna.dna.serialization import load_genotype

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genotype = load_genotype(aidna_path)
    params = genotype.dna_instinct.genetic_parameters
    reconstructed = {}

    # 1. Collect SVD pairs
    svd_keys = set()
    for k in params:
        if k.startswith("svd.") and k.endswith(".A"):
            base = k[4:-2]
            svd_keys.add(base)

    # 2. Reconstruct SVD pairs: W = A @ B
    for base in svd_keys:
        if key_filter and not base.startswith(key_filter):
            continue
        A = params.get(f"svd.{base}.A")
        B = params.get(f"svd.{base}.B")
        if A is not None and B is not None:
            W = (A.float() @ B.float()).to(device)
            shape_key = f"meta.{base}.orig_shape"
            if shape_key in params:
                orig_shape = tuple(params[shape_key].long().tolist())
                W = W.reshape(orig_shape)
            reconstructed[base] = W

    # 3. Copy raw parameters directly
    for k, v in params.items():
        if k.startswith("raw."):
            real_key = k[4:]
            if key_filter and not real_key.startswith(key_filter):
                continue
            reconstructed[real_key] = v.to(device)

    # 4. Copy modal parameters directly
    for k, v in params.items():
        if k.startswith("modal."):
            real_key = k[6:]
            if key_filter and not real_key.startswith(key_filter):
                continue
            reconstructed[real_key] = v.to(device)

    # 5. Copy unprefixed parameters
    for k, v in params.items():
        if not (k.startswith("svd.") or k.startswith("modal.") or k.startswith("raw.") or k.startswith("meta.")):
            if key_filter and not k.startswith(key_filter):
                continue
            if k not in reconstructed:
                reconstructed[k] = v.to(device)

    return reconstructed, genotype


def reconstruct_weights_from_aidna(
    aidna_path: str,
    key_filter: Optional[str] = None,
    device: Optional[torch.device] = None,
    return_genotype: bool = False,
):
    """Reconstructs state_dict from .aidna file. If return_genotype is True, returns (dict, genotype)."""
    weights, genotype = reconstruct_weights_and_genotype(aidna_path, key_filter=key_filter, device=device)
    if return_genotype:
        return weights, genotype
    return weights


def reconstruct_weights_only(
    aidna_path: str,
    key_filter: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Reconstructs and returns only the state_dict from .aidna file."""
    weights, _ = reconstruct_weights_and_genotype(aidna_path, key_filter=key_filter, device=device)
    return weights

