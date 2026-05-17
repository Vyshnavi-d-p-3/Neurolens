"""Tests for TextFooler word-substitution attack."""

import torch
import pytest

from models.transformer import TextClassifier
from attacks.textfooler import TextFooler
from utils.text_encoding import SimpleWordEncoder


class _MockSynonyms:
    def synonyms(self, word: str, pos: str | None) -> list[str]:
        table = {
            "government": ["administration", "authorities"],
            "local": ["regional", "municipal"],
            "sports": ["athletics", "games"],
        }
        return table.get(word, [])


@pytest.fixture
def encoder():
    return SimpleWordEncoder(vocab_size=512, max_seq_len=64)


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = TextClassifier(vocab_size=512, num_classes=4)
    m.eval()
    return m


@pytest.fixture
def attack(model, encoder):
    return TextFooler(
        model,
        encoder=encoder,
        similarity_threshold=0.8,
        synonym_provider=_MockSynonyms(),
        similarity_fn=lambda a, b: 0.95 if a != b else 1.0,
    )


class TestTextFooler:
    def test_zero_substitutions_unchanged_prediction(self, attack, encoder, model):
        """max_substitutions=0 is the ε→0 analogue: no edits, same prediction."""
        text = "local government announces sports funding"
        input_ids, words = encoder.encode(text)
        label = model(input_ids).argmax(dim=-1).item()

        no_op = TextFooler(
            attack.model,
            encoder=encoder,
            max_substitutions=0,
            synonym_provider=_MockSynonyms(),
            similarity_fn=lambda a, b: 1.0,
        )
        adv_ids, subs = no_op.attack_ids(input_ids, label, words)
        assert subs == []
        assert torch.equal(adv_ids, input_ids)
        assert no_op._predict(adv_ids) == label

    def test_successful_attack_respects_similarity(self, model, encoder):
        """Applied substitutions must meet the similarity threshold (Table-style filter)."""
        text = "local government announces sports funding"
        input_ids, words = encoder.encode(text)
        label = model(input_ids).argmax(dim=-1).item()

        attacker = TextFooler(
            model,
            encoder=encoder,
            similarity_threshold=0.8,
            max_substitutions=2,
            synonym_provider=_MockSynonyms(),
            similarity_fn=lambda a, b: 0.85 if a != b else 1.0,
        )
        adv_text, subs = attacker.attack(text, label)

        assert subs, "expected at least one greedy substitution in smoke test"
        assert attacker._similarity(text, adv_text) >= attacker.similarity_threshold

    def test_importance_ranks_words(self, attack, encoder, model):
        """Importance scores are produced for every content word."""
        text = "government local sports"
        input_ids, words = encoder.encode(text)
        label = model(input_ids).argmax(dim=-1).item()
        baseline = attack._true_class_prob(input_ids, torch.tensor([label]))
        ranked = attack._rank_word_importance(
            input_ids, words, torch.tensor([label]), baseline,
        )
        assert len(ranked) == len(words)
