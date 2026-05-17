"""
TextFooler attack (Jin et al., 2020).

Targets discrete text inputs for the AG News Transformer classifier:
1. Rank words by leave-one-out importance on true-class probability.
2. Substitute important words with WordNet synonyms (POS-matched).
3. Filter candidates by sentence-embedding cosine similarity.
4. Greedily substitute until the prediction flips or candidates are exhausted.

Vision attacks use Attack.perturb(x, y); text is discrete, so this class
exposes attack(text, label) and attack_ids(input_ids, label, words) instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.text_encoding import SimpleWordEncoder, UNK_ID


class SynonymProvider(Protocol):
    def synonyms(self, word: str, pos: str | None) -> list[str]: ...


class _WordNetSynonyms:
    """WordNet synonym lookup with NLTK POS tagging."""

    def __init__(self) -> None:
        self._ready = False

    def _ensure_nltk(self) -> None:
        if self._ready:
            return
        import nltk

        for resource in ("wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"):
            nltk.download(resource, quiet=True)
        self._ready = True

    @staticmethod
    def _wn_pos(tag: str) -> str | None:
        if tag.startswith("J"):
            return "a"
        if tag.startswith("N"):
            return "n"
        if tag.startswith("R"):
            return "r"
        if tag.startswith("V"):
            return "v"
        return None

    def synonyms(self, word: str, pos: str | None) -> list[str]:
        self._ensure_nltk()
        from nltk.corpus import wordnet as wn

        kwargs = {}
        if pos:
            kwargs["pos"] = pos
        syns: list[str] = []
        for syn in wn.synsets(word, **kwargs):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ").lower()
                if candidate != word and candidate.isalpha():
                    syns.append(candidate)
        return list(dict.fromkeys(syns))[:20]


class TextFooler:
    """
    TextFooler word-substitution attack.

    Args:
        model:                 TextClassifier (or compatible) in eval mode.
        encoder:               maps raw text → input_ids + word list.
        similarity_threshold:  minimum cosine sim (original, perturbed).
        max_substitutions:     cap word changes (0 = no-op attack).
        synonym_provider:      injectable for tests (defaults to WordNet).
        similarity_fn:         injectable (text_a, text_b) → cosine sim.
        sentence_model_name:   model for default similarity_fn.
    """

    def __init__(
        self,
        model: nn.Module,
        encoder: SimpleWordEncoder | None = None,
        similarity_threshold: float = 0.8,
        max_substitutions: int | None = None,
        synonym_provider: SynonymProvider | None = None,
        similarity_fn: Callable[[str, str], float] | None = None,
        sentence_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = model
        self.encoder = encoder or SimpleWordEncoder()
        self.similarity_threshold = similarity_threshold
        self.max_substitutions = max_substitutions
        self.synonym_provider = synonym_provider or _WordNetSynonyms()
        self._similarity_fn = similarity_fn
        self.sentence_model_name = sentence_model_name
        self._sentence_model = None
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Public API (text is discrete — not Attack.perturb)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def attack(self, text: str, label: int) -> tuple[str, list[tuple[str, str]]]:
        """
        Run TextFooler on a single string.

        Returns:
            adversarial_text, list of (original_word, substitute) applied.
        """
        input_ids, words = self.encoder.encode(text)
        adv_ids, substitutions = self.attack_ids(input_ids, label, words)
        if not substitutions:
            return text, []
        adv_words = words[:]
        for orig, sub in substitutions:
            adv_words = [sub if w == orig else w for w in adv_words]
        return self.encoder.decode_words(adv_words), substitutions

    @torch.no_grad()
    def attack_ids(
        self,
        input_ids: torch.LongTensor,
        label: int,
        words: list[str],
    ) -> tuple[torch.LongTensor, list[tuple[str, str]]]:
        """Attack using pre-tokenized words (length matches content tokens)."""
        if self.max_substitutions == 0 or not words:
            return input_ids.clone(), []

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        label_t = torch.tensor([label], device=device)

        if self._predict(input_ids) != label:
            return input_ids.clone(), []

        baseline_prob = self._true_class_prob(input_ids, label_t)
        ranked = self._rank_word_importance(input_ids, words, label_t, baseline_prob)
        substitutions: list[tuple[str, str]] = []
        current_words = words[:]
        original_text = self.encoder.decode_words(words)

        for word_idx, _ in ranked:
            if self.max_substitutions is not None and len(substitutions) >= self.max_substitutions:
                break

            word = current_words[word_idx]
            pos = self._pos_tag(word, current_words)
            candidates = self.synonym_provider.synonyms(word, pos)
            if not candidates:
                continue

            for candidate in candidates:
                trial_words = current_words[:]
                trial_words[word_idx] = candidate
                trial_text = self.encoder.decode_words(trial_words)
                if self._similarity(original_text, trial_text) < self.similarity_threshold:
                    continue

                trial_ids = self.encoder.ids_from_words(trial_words).to(device)
                if self._predict(trial_ids) != label:
                    substitutions.append((word, candidate))
                    return trial_ids, substitutions

            # Greedy: keep best synonym even if prediction unchanged (for progress)
            for candidate in candidates:
                trial_words = current_words[:]
                trial_words[word_idx] = candidate
                trial_text = self.encoder.decode_words(trial_words)
                if self._similarity(original_text, trial_text) >= self.similarity_threshold:
                    current_words = trial_words
                    substitutions.append((word, candidate))
                    input_ids = self.encoder.ids_from_words(current_words).to(device)
                    if self._predict(input_ids) != label:
                        return input_ids, substitutions
                    break

        return self.encoder.ids_from_words(current_words).to(device), substitutions

    # ------------------------------------------------------------------ #
    # Scoring helpers
    # ------------------------------------------------------------------ #
    def _predict(self, input_ids: torch.LongTensor) -> int:
        return int(self.model(input_ids).argmax(dim=-1).item())

    def _true_class_prob(self, input_ids: torch.LongTensor, label: torch.Tensor) -> float:
        logits = self.model(input_ids)
        probs = F.softmax(logits, dim=-1)
        return float(probs[0, label.item()].item())

    def _rank_word_importance(
        self,
        input_ids: torch.LongTensor,
        words: list[str],
        label: torch.Tensor,
        baseline_prob: float,
    ) -> list[tuple[int, float]]:
        scores: list[tuple[int, float]] = []
        for i in range(len(words)):
            masked = input_ids.clone()
            masked[0, i + 1] = UNK_ID  # +1 for [CLS]
            masked_prob = self._true_class_prob(masked, label)
            scores.append((i, baseline_prob - masked_prob))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _pos_tag(self, word: str, sentence_words: list[str]) -> str | None:
        if isinstance(self.synonym_provider, _WordNetSynonyms):
            import nltk

            self.synonym_provider._ensure_nltk()
            tag = nltk.pos_tag([word])[0][1]
            return self.synonym_provider._wn_pos(tag)
        return None

    def _similarity(self, text_a: str, text_b: str) -> float:
        if self._similarity_fn is not None:
            return self._similarity_fn(text_a, text_b)
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer(self.sentence_model_name)
        emb = self._sentence_model.encode([text_a, text_b], convert_to_tensor=True)
        return float(F.cosine_similarity(emb[0].unsqueeze(0), emb[1].unsqueeze(0)).item())
