"""Simple word-level encoding for text attacks (AG News / Transformer)."""

from __future__ import annotations

import re

import torch

PAD_ID = 0
CLS_ID = 1
UNK_ID = 2


class SimpleWordEncoder:
    """
    Word-level encoder aligned with TextClassifier ([CLS] + tokens + pad).

    Vocabulary grows on the fly for unseen words (suitable for attack demos).
    """

    def __init__(self, vocab_size: int = 30522, max_seq_len: int = 128):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.word2id: dict[str, int] = {}
        self._next_id = 3

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _word_id(self, word: str) -> int:
        if word not in self.word2id:
            if self._next_id >= self.vocab_size:
                return UNK_ID
            self.word2id[word] = self._next_id
            self._next_id += 1
        return self.word2id[word]

    def encode(self, text: str) -> tuple[torch.LongTensor, list[str]]:
        """Return (1, seq_len) input_ids and the word list (no special tokens)."""
        words = self._tokenize(text)[: self.max_seq_len - 2]
        ids = [CLS_ID] + [self._word_id(w) for w in words]
        while len(ids) < self.max_seq_len:
            ids.append(PAD_ID)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0), words

    def decode_words(self, words: list[str]) -> str:
        return " ".join(words)

    def ids_from_words(self, words: list[str]) -> torch.LongTensor:
        ids = [CLS_ID] + [self._word_id(w) for w in words]
        while len(ids) < self.max_seq_len:
            ids.append(PAD_ID)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
