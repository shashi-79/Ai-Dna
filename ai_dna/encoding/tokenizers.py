import os
import json
import torch
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Tuple, Any, Optional

class EvolvingTokenizer(ABC):
    """Base class for auto-evolving tokenizers that co-evolve with DNA."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    @abstractmethod
    def train(self, data: List[Any], target_vocab_size: int) -> None:
        """Learn/extend vocabulary from training data."""
        pass
    
    @abstractmethod
    def encode(self, raw_input: Any) -> torch.Tensor:
        """Tokenize raw input into token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> Any:
        """Convert token IDs back to human-readable output."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save tokenizer state to JSON."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer state from JSON."""
        pass


class TextBPETokenizer(EvolvingTokenizer):
    """
    Byte-Pair Encoding (BPE) tokenizer that co-evolves with DNA.
    Starts with 256 byte tokens + 4 special tokens.
    Learns merge rules from data and expands vocabulary dynamically.
    """
    def __init__(self, vocab_size: int = 256 + 4):
        super().__init__(vocab_size)
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Initialize base vocab: 4 special tokens + 256 byte values
        self.vocab: List[bytes] = [tok.encode("utf-8") for tok in self.special_tokens]
        for b in range(256):
            self.vocab.append(bytes([b]))
            
        self.vocab_size = len(self.vocab)
        self.token2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2token = {idx: tok for idx, tok in enumerate(self.vocab)}
        
        # Merges: Tuple[int, int] -> int
        self.merges: Dict[Tuple[int, int], int] = {}

    def _get_stats(self, ids_list: List[List[int]]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for ids in ids_list:
            for pair in zip(ids[:-1], ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids_list: List[List[int]], pair: Tuple[int, int], idx: int) -> List[List[int]]:
        new_ids_list = []
        for ids in ids_list:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_ids_list.append(new_ids)
        return new_ids_list

    def train(self, texts: List[str], target_vocab_size: int) -> None:
        """
        Train BPE from scratch on a corpus of text.
        """
        # Re-initialize vocabulary to base state
        self.vocab = [tok.encode("utf-8") for tok in self.special_tokens]
        for b in range(256):
            self.vocab.append(bytes([b]))
        self.merges = {}
        self.vocab_size = len(self.vocab)
        self.token2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2token = {idx: tok for idx, tok in enumerate(self.vocab)}

        self.evolve(texts, target_vocab_size)

    def evolve(self, texts: List[str], target_vocab_size: int) -> int:
        """
        Evolve vocabulary by learning new merge rules from new text.
        Preserves existing merges. Returns number of new tokens added.
        """
        if target_vocab_size <= self.vocab_size:
            return 0

        # Encode texts into base tokens / currently learned BPE tokens
        ids_list = []
        for text in texts:
            # First map UTF-8 bytes to base token IDs
            raw_bytes = text.encode("utf-8")
            ids = [self.token2idx[bytes([b])] for b in raw_bytes]
            
            # Iteratively apply existing merges in order
            for pair, merge_idx in self.merges.items():
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                        new_ids.append(merge_idx)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            ids_list.append(ids)

        num_merges = target_vocab_size - self.vocab_size
        added = 0

        for _ in range(num_merges):
            stats = self._get_stats(ids_list)
            if not stats:
                break
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break

            # Assign new token ID
            new_idx = self.vocab_size
            self.merges[best_pair] = new_idx
            
            # Compute new vocab representation as concatenation of merged byte sequences
            part1 = self.idx2token[best_pair[0]]
            part2 = self.idx2token[best_pair[1]]
            new_token_val = part1 + part2
            
            self.vocab.append(new_token_val)
            self.token2idx[new_token_val] = new_idx
            self.idx2token[new_idx] = new_token_val
            self.vocab_size += 1
            added += 1

            # Update working lists
            ids_list = self._merge(ids_list, best_pair, new_idx)

        return added

    def encode(self, text: str) -> torch.Tensor:
        """Tokenize text into a 2D tensor (1, L)."""
        raw_bytes = text.encode("utf-8")
        ids = [self.token2idx.get(bytes([b]), self.unk_token_id) for b in raw_bytes]
        
        # Apply BPE merge rules
        for pair, merge_idx in self.merges.items():
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    new_ids.append(merge_idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        if not ids:
            ids = [self.unk_token_id]
            
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to string."""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.flatten().tolist()
        else:
            ids = list(token_ids)
            
        byte_list = []
        for t in ids:
            if skip_special_tokens and t < len(self.special_tokens):
                continue
            if t in self.idx2token:
                byte_list.append(self.idx2token[t])
            else:
                # If unknown index, fallback
                byte_list.append(bytes(f"<T_{t}>", "utf-8"))
                
        # Join bytes and decode
        all_bytes = b"".join(byte_list)
        return all_bytes.decode("utf-8", errors="replace")

    def decode_verbose(self, token_ids: Union[List[int], torch.Tensor]) -> List[Tuple[int, str, bool]]:
        """Returns detailed per-token breakdown: (token_id, token_string, is_special)."""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.flatten().tolist()
        else:
            ids = list(token_ids)
            
        result = []
        for t in ids:
            is_special = t < len(self.special_tokens)
            if t in self.idx2token:
                tok_str = self.idx2token[t].decode("utf-8", errors="replace")
                result.append((t, tok_str, is_special))
            else:
                result.append((t, f"<T_{t}>", False))
        return result

    def save(self, path: str) -> None:
        """Save BPE merges and vocab list to JSON."""
        # Convert merges tuples keys to string "p1,p2" for JSON
        merges_json = {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()}
        # Vocab contains bytes, convert to hex or string representation for serialization
        vocab_hex = [v.hex() for v in self.vocab]
        
        state = {
            "vocab_size": self.vocab_size,
            "merges": merges_json,
            "vocab": vocab_hex
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load BPE tokenizer from JSON."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No tokenizer state file found at: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        self.vocab_size = state["vocab_size"]
        self.vocab = [bytes.fromhex(h) for h in state["vocab"]]
        
        self.token2idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx2token = {idx: tok for idx, tok in enumerate(self.vocab)}
        
        self.merges = {}
        for pair_str, idx in state["merges"].items():
            p1, p2 = map(int, pair_str.split(","))
            self.merges[(p1, p2)] = idx


class VisionPatchTokenizer(EvolvingTokenizer):
    """
    Placeholder/stub for evolving vision patch tokenizer.
    Translates image inputs to grid patch tokens.
    """
    def __init__(self, vocab_size: int = 100):
        super().__init__(vocab_size)
        self.vocab_size = vocab_size

    def train(self, data: List[Any], target_vocab_size: int) -> None:
        self.vocab_size = target_vocab_size

    def evolve(self, data: List[Any], target_vocab_size: int) -> int:
        added = target_vocab_size - self.vocab_size
        self.vocab_size = target_vocab_size
        return max(0, added)

    def encode(self, raw_input: Any) -> torch.Tensor:
        # Standard dummy grid-patch encoding
        if isinstance(raw_input, torch.Tensor):
            # Flat sequence representation placeholder
            return torch.randint(0, self.vocab_size, (1, 16), dtype=torch.long)
        return torch.zeros((1, 16), dtype=torch.long)

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> Any:
        return "<Vision Patches>"

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "modality": "vision"}, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            state = json.load(f)
            self.vocab_size = state["vocab_size"]


class AudioSpectralTokenizer(EvolvingTokenizer):
    """
    Placeholder/stub for evolving audio spectral tokenizer.
    """
    def __init__(self, vocab_size: int = 100):
        super().__init__(vocab_size)
        self.vocab_size = vocab_size

    def train(self, data: List[Any], target_vocab_size: int) -> None:
        self.vocab_size = target_vocab_size

    def evolve(self, data: List[Any], target_vocab_size: int) -> int:
        added = target_vocab_size - self.vocab_size
        self.vocab_size = target_vocab_size
        return max(0, added)

    def encode(self, raw_input: Any) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (1, 16), dtype=torch.long)

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> Any:
        return "<Audio Spectral Frames>"

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "modality": "audio"}, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            state = json.load(f)
            self.vocab_size = state["vocab_size"]


class VideoSpatiotemporalTokenizer(EvolvingTokenizer):
    """
    Placeholder/stub for evolving video tokenizer.
    """
    def __init__(self, vocab_size: int = 100):
        super().__init__(vocab_size)
        self.vocab_size = vocab_size

    def train(self, data: List[Any], target_vocab_size: int) -> None:
        self.vocab_size = target_vocab_size

    def evolve(self, data: List[Any], target_vocab_size: int) -> int:
        added = target_vocab_size - self.vocab_size
        self.vocab_size = target_vocab_size
        return max(0, added)

    def encode(self, raw_input: Any) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (1, 16), dtype=torch.long)

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> Any:
        return "<Video Spatiotemporal Cuboids>"

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "modality": "video"}, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            state = json.load(f)
            self.vocab_size = state["vocab_size"]
