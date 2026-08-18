"""
Dynamic Text Tokenizer for AI DNA Inference wrapping BPE evolving tokenization.
Delegates to the evolving tokenizer framework in ai_dna/encoding/tokenizers.
"""

import os
import torch
from typing import List, Union, Dict, Optional, Tuple
from ..encoding.tokenizers import TextBPETokenizer


class TextTokenizer:
    """
    Wrapper around TextBPETokenizer for inference compatibility.
    Loads co-evolved BPE tokenizer states if checkpoint_dir is provided.
    """
    def __init__(self, vocab_size: int = 128, mode: str = "word", checkpoint_dir: Optional[str] = None):
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        self.bpe = TextBPETokenizer(vocab_size=vocab_size)
        
        # Load from checkpoint if available
        if checkpoint_dir:
            path = os.path.join(checkpoint_dir, "tokenizer_text.json")
            if os.path.exists(path):
                print(f"[*] Loading Text Tokenizer from: {path}")
                self.bpe.load(path)
            else:
                # If path doesn't exist, we can pre-train BPE with basic corpus of seed words
                # to initialize it, ensuring it is not totally blank
                self._initialize_fallback_bpe()
        else:
            self._initialize_fallback_bpe()

        # Expose properties for external compatibility
        self.vocab_size = self.bpe.vocab_size
        self.pad_token_id = self.bpe.pad_token_id
        self.unk_token_id = self.bpe.unk_token_id
        self.bos_token_id = self.bpe.bos_token_id
        self.eos_token_id = self.bpe.eos_token_id
        self.token2idx = {tok.decode("utf-8", errors="replace"): idx for idx, tok in enumerate(self.bpe.vocab)}
        self.idx2token = {idx: tok.decode("utf-8", errors="replace") for idx, tok in enumerate(self.bpe.vocab)}
        
        # Compatibility aliases
        self.char2idx = self.token2idx
        self.idx2char = self.idx2token

    def _initialize_fallback_bpe(self):
        # Text corpus of AI DNA terms to initialize BPE vocabulary
        seed_corpus = [
            "ai dna architecture learns and evolves dynamically across generations",
            "fast clock trains phenotype neural weights while genotype stays frozen",
            "slow clock distills learned parameter instincts into structural dna",
            "sparse mixture of experts routes active tokens through dynamic pathways",
            "hierarchical memory controller archives long context facts and episodes",
            "multi parent fusion combines distinct specialist parents into omni modal dna",
            "developmental growth engine maps compact genotype into full neural phenotype",
            "neural networks adapt to unseen tasks with high sample efficiency",
            "intelligence emerges from structural instinct and evolutionary adaptation",
            "causal language modeling predicts the next token in sequence",
            "hello how are you doing today welcome to the system",
            "good morning friend can i help you achieved successfully accuracy ratio"
        ]
        # Train BPE using fallback corpus to make it functional
        self.bpe.train(seed_corpus, self.bpe.vocab_size)

    def encode(self, text: str) -> torch.Tensor:
        """Encodes text string into a 2D LongTensor (1, L)."""
        return self.bpe.encode(text)

    def decode(self, tokens: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into human-readable string."""
        return self.bpe.decode(tokens, skip_special_tokens=skip_special_tokens)

    def decode_verbose(self, tokens: Union[List[int], torch.Tensor]) -> List[Tuple[int, str, bool]]:
        """Returns list of (token_id, token_string, is_special)."""
        return self.bpe.decode_verbose(tokens)

