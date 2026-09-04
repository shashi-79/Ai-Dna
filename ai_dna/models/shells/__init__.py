"""
Zero-Dependency Standalone Model Shells & Weight Loaders Package.
Exports MinimalSmolLM2, MinimalCLIP, MinimalWhisper, CLIPTokenizer, WhisperTokenizer,
and zero-dependency safetensors/bin loaders.
"""

from .io import (
    load_safetensors_file,
    load_model_weights,
    load_config,
    reconstruct_weights_from_aidna,
    reconstruct_weights_and_genotype,
    reconstruct_weights_only,
)
from .tokenizers import (
    SmolLM2Tokenizer,
    CLIPTokenizer,
    WhisperTokenizer,
)
from .architectures import (
    MinimalRMSNorm,
    MinimalSmolLM2,
    MinimalCLIP,
    MinimalWhisper,
    compute_mel_spectrogram_from_waveform,
    compute_weight_diff_metrics,
    compute_output_similarity,
)

__all__ = [
    "load_safetensors_file",
    "load_model_weights",
    "load_config",
    "reconstruct_weights_from_aidna",
    "reconstruct_weights_and_genotype",
    "reconstruct_weights_only",
    "SmolLM2Tokenizer",
    "CLIPTokenizer",
    "WhisperTokenizer",
    "MinimalRMSNorm",
    "MinimalSmolLM2",
    "MinimalCLIP",
    "MinimalWhisper",
    "compute_mel_spectrogram_from_waveform",
    "compute_weight_diff_metrics",
    "compute_output_similarity",
]

