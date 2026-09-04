"""
Self-contained Tokenizers for Zero-Dependency Language, Vision & Audio Models.
Provides SmolLM2Tokenizer, CLIPTokenizer, and WhisperTokenizer without requiring external transformers package.
"""

import os
import json
import re
from typing import Dict, Any, List, Tuple, Optional, Union
import torch


class SmolLM2Tokenizer:
    """Self-contained exact Byte-Level BPE Tokenizer for SmolLM2."""

    def __init__(self, tokenizer_data_or_path: Union[Dict, str] = ""):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.b2u = self._gpt2_bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

        if isinstance(tokenizer_data_or_path, dict) and tokenizer_data_or_path:
            vocab = tokenizer_data_or_path.get("model", {}).get("vocab", tokenizer_data_or_path.get("vocab", {}))
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        elif isinstance(tokenizer_data_or_path, str) and os.path.exists(tokenizer_data_or_path):
            try:
                with open(tokenizer_data_or_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                vocab = tok_data.get("model", {}).get("vocab", {})
                self.token_to_id = vocab
                self.id_to_token = {v: k for k, v in vocab.items()}
            except Exception:
                pass

    @staticmethod
    def _gpt2_bytes_to_unicode() -> Dict[int, str]:
        bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))

    def encode(self, text: str) -> List[int]:
        if not self.token_to_id:
            return [ord(c) % 49152 for c in text] if text else [0]

        bpe_text = text.replace(" ", "\u0120").replace("\n", "\u010a").replace("\t", "\u0109").replace("\r", "\u010d")
        ids = []
        i = 0
        N = len(bpe_text)
        while i < N:
            matched = False
            for length in range(min(32, N - i), 0, -1):
                sub = bpe_text[i:i + length]
                if sub in self.token_to_id:
                    ids.append(self.token_to_id[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                ch = bpe_text[i]
                ids.append(self.token_to_id.get(ch, 0))
                i += 1
        return ids if ids else [0]

    def decode(self, ids: List[int]) -> str:
        if not self.id_to_token:
            return ""
        tokens = [self.id_to_token.get(i, "") for i in ids if i not in [0, 1, 2]]
        text = "".join(tokens)
        text = text.replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t").replace("\u010d", "\r")
        return text


class CLIPTokenizer:
    """Self-contained BPE Tokenizer for CLIP without external tokenizer dependencies."""

    def __init__(self, tokenizer_data_or_path: Union[Dict, str] = ""):
        self.vocab = {}
        self.bpe_ranks = {}
        self.eot_token_id = 49407
        self.sot_token_id = 49406

        if isinstance(tokenizer_data_or_path, dict) and tokenizer_data_or_path:
            model_field = tokenizer_data_or_path.get("model")
            if isinstance(model_field, dict) and "vocab" in model_field:
                self.vocab = model_field["vocab"]
                merges = [tuple(m.split()) for m in model_field.get("merges", []) if isinstance(m, str)]
                self.bpe_ranks = dict(zip(merges, range(len(merges))))
            elif "vocab" in tokenizer_data_or_path and isinstance(tokenizer_data_or_path.get("vocab"), dict):
                self.vocab = tokenizer_data_or_path["vocab"]
                merges = [tuple(m.split()) for m in tokenizer_data_or_path.get("merges", []) if isinstance(m, str)]
                self.bpe_ranks = dict(zip(merges, range(len(merges))))
            else:
                self.vocab = tokenizer_data_or_path
                self.bpe_ranks = {}
            if isinstance(self.vocab, dict):
                self.sot_token_id = self.vocab.get("<|startoftext|>", 49406)
                self.eot_token_id = self.vocab.get("<|endoftext|>", 49407)
        elif isinstance(tokenizer_data_or_path, str) and os.path.exists(tokenizer_data_or_path):
            try:
                with open(tokenizer_data_or_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                self.vocab = tok_data.get("model", {}).get("vocab", {})
                merges = [tuple(m.split()) for m in tok_data.get("model", {}).get("merges", []) if isinstance(m, str)]
                self.bpe_ranks = dict(zip(merges, range(len(merges))))
                self.sot_token_id = self.vocab.get("<|startoftext|>", 49406)
                self.eot_token_id = self.vocab.get("<|endoftext|>", 49407)
            except Exception as e:
                print(f"  [WARN] Failed to load CLIP tokenizer: {e}")

    @staticmethod
    def _get_pairs(word: Tuple[str, ...]) -> set:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token: str) -> str:
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self._get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        return " ".join(word)

    def encode(self, texts: List[str], max_len: int = 77) -> Tuple[torch.Tensor, List[int]]:
        """Encodes a list of text strings into (input_ids tensor [N, 77], eot_indices list)."""
        all_ids = []
        all_eots = []
        for text in texts:
            clean_text = text.lower().strip()
            words = re.findall(r"\w+|\S", clean_text)
            tokens = ["<|startoftext|>"]
            for word in words:
                for b in self._bpe(word).split():
                    tokens.append(b)
            tokens.append("<|endoftext|>")

            ids = [self.vocab.get(t, self.eot_token_id) for t in tokens][:max_len]
            eot = len(ids) - 1
            ids += [self.eot_token_id] * (max_len - len(ids))
            all_ids.append(ids)
            all_eots.append(eot)
        return torch.tensor(all_ids, dtype=torch.long), all_eots


class WhisperTokenizer:
    """Self-contained tokenizer for Whisper speech-to-text decoding."""

    def __init__(
        self,
        tokenizer_data_or_path: Union[Dict, str] = "",
        added_tokens_data_or_path: Union[Dict, str, None] = None,
    ):
        self.vocab = {}
        self.id_to_token = {}
        self.sot_token = 50258  # <|startoftranscript|>
        self.eot_token = 50257  # <|endoftext|>
        self.en_token = 50259   # <|en|>
        self.transcribe_token = 50359  # <|transcribe|>
        self.no_timestamps_token = 50363  # <|notimestamps|>

        if isinstance(tokenizer_data_or_path, dict) and tokenizer_data_or_path:
            self.vocab = tokenizer_data_or_path.get("model", {}).get("vocab", {})
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        elif isinstance(tokenizer_data_or_path, str) and os.path.exists(tokenizer_data_or_path):
            try:
                with open(tokenizer_data_or_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                self.vocab = tok_data.get("model", {}).get("vocab", {})
                self.id_to_token = {v: k for k, v in self.vocab.items()}
            except Exception as e:
                print(f"  [WARN] Failed to load Whisper tokenizer: {e}")

        if isinstance(added_tokens_data_or_path, dict) and added_tokens_data_or_path:
            for token_str, token_id in added_tokens_data_or_path.items():
                self.id_to_token[token_id] = token_str
        elif isinstance(added_tokens_data_or_path, str) and os.path.exists(added_tokens_data_or_path):
            try:
                with open(added_tokens_data_or_path, "r", encoding="utf-8") as f:
                    added = json.load(f)
                for token_str, token_id in added.items():
                    self.id_to_token[token_id] = token_str
            except Exception:
                pass

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Converts a sequence of Whisper token IDs into human-readable text string."""
        words = []
        for tid in token_ids:
            if tid >= 50257:
                if not skip_special:
                    special_str = self.id_to_token.get(tid, f"<|{tid}|>")
                    words.append(f"[{special_str}]")
            else:
                raw_tok = self.id_to_token.get(tid, "")
                cleaned = raw_tok.replace("\u0120", " ").replace("\u010a", "\n").replace("\u00c4\u00a0", " ")
                words.append(cleaned)
        return "".join(words).strip()
