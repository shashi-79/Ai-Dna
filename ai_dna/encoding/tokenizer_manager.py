import os
from typing import Dict, Any, Optional, List
from .tokenizers import (
    EvolvingTokenizer,
    TextBPETokenizer,
    VisionPatchTokenizer,
    AudioSpectralTokenizer,
    VideoSpatiotemporalTokenizer
)

class TokenizerManager:
    """
    Manages per-modality evolving tokenizers co-evolving with DNA.
    Loads and saves individual tokenizer states next to the genotype.
    """
    def __init__(self, text_vocab_size: int = 256 + 4, vision_vocab_size: int = 100, audio_vocab_size: int = 100, video_vocab_size: int = 100):
        self.tokenizers: Dict[str, EvolvingTokenizer] = {
            "text": TextBPETokenizer(vocab_size=text_vocab_size),
            "vision": VisionPatchTokenizer(vocab_size=vision_vocab_size),
            "audio": AudioSpectralTokenizer(vocab_size=audio_vocab_size),
            "video": VideoSpatiotemporalTokenizer(vocab_size=video_vocab_size),
        }

    def get_vocab_size(self, modality: str) -> int:
        """Get vocab size for a specific modality."""
        if modality in self.tokenizers:
            return self.tokenizers[modality].vocab_size
        return 0

    def train_tokenizer(self, modality: str, data: List[Any], target_vocab_size: int) -> None:
        """Train a specific modality tokenizer from scratch."""
        if modality in self.tokenizers:
            self.tokenizers[modality].train(data, target_vocab_size)

    def evolve_tokenizer(self, modality: str, data: List[Any], target_vocab_size: int) -> int:
        """Evolve a specific modality tokenizer with new data."""
        if modality in self.tokenizers:
            return self.tokenizers[modality].evolve(data, target_vocab_size)
        return 0

    def save_all(self, directory: str, prefix: str = "tokenizer_") -> None:
        """Save all tokenizers to the specified checkpoint directory."""
        os.makedirs(directory, exist_ok=True)
        for modality, tokenizer in self.tokenizers.items():
            filename = f"{prefix}{modality}.json"
            path = os.path.join(directory, filename)
            tokenizer.save(path)
            print(f"      [+] Saved Evolving Tokenizer [{modality.upper()}] state to: {path}")

    def load_all(self, directory: str, prefix: str = "tokenizer_") -> None:
        """Load all tokenizers from the specified checkpoint directory if files exist."""
        for modality, tokenizer in self.tokenizers.items():
            filename = f"{prefix}{modality}.json"
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                print(f"[*] Loading Evolving Tokenizer [{modality.upper()}] state from: {path}")
                tokenizer.load(path)
            else:
                print(f"[*] Using initial/empty state for Evolving Tokenizer [{modality.upper()}]")
