"""Tests for EmbeddingSynergyScorer — uses a fake model, no actual ML."""

import pytest
import numpy as np

from mtg_deck_builder.embedding_scorer import (
    EmbeddingSynergyScorer, EmbeddingConfig, is_embeddings_available,
)
from mtg_deck_builder.models import Card, CommanderAnalysis


class FakeModel:
    """
    Fake sentence transformer that returns deterministic embeddings.

    Design: we want texts with synergy keywords to embed in nearly the same
    direction as the commander query, and texts without to embed elsewhere.
    We use a fixed "synergy axis" and vary a small identity-differentiating
    component so different texts don't produce identical vectors.
    """

    def __init__(self, synergy_keywords=None):
        self.synergy_keywords = synergy_keywords or ["gain life", "lifelink", "counter"]

    def encode(self, texts, convert_to_numpy=True):
        """Encode text(s) to deterministic 32-d vectors."""
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        vectors = []
        for text in texts:
            vec = self._encode_one(text)
            vectors.append(vec)

        arr = np.array(vectors, dtype="float32")
        if single:
            return arr[0]
        return arr

    def _encode_one(self, text: str):
        """
        Deterministic embedding.

        Structure:
          - First 16 dims: "synergy axis", set to 1.0 if text has any synergy
            keyword, 0.0 otherwise.
          - Last 16 dims: small identity noise (seeded from text hash) so
            different texts don't produce identical vectors. Magnitude is
            small enough that it doesn't overwhelm the synergy signal.
        """
        text_lower = text.lower()
        has_synergy = any(kw.lower() in text_lower for kw in self.synergy_keywords)

        vec = np.zeros(32, dtype="float32")
        if has_synergy:
            vec[:16] = 1.0  # strong synergy direction

        # Small identity component (magnitude 0.1, much smaller than synergy signal)
        h = hash(text) & 0xFFFFFF
        rng = np.random.default_rng(h)
        vec[16:] = rng.standard_normal(16).astype("float32") * 0.1

        return vec


@pytest.fixture
def analysis():
    return CommanderAnalysis(
        name="Lathiel",
        color_identity="WG",
        key_mechanics=["lifegain", "+1/+1 counters"],
        build_around_text="Gain life and distribute counters.",
        evaluation_notes="",
        category_queries={},
        synergy_keywords=["gain life", "lifelink", "+1/+1 counter"],
    )


class TestAvailabilityCheck:
    def test_returns_bool(self):
        """is_embeddings_available returns a bool."""
        result = is_embeddings_available()
        assert isinstance(result, bool)


class TestScoring:
    def test_synergy_cards_score_higher(self, analysis):
        """Cards with synergy keywords in their text should score higher."""
        fake = FakeModel(synergy_keywords=analysis.synergy_keywords)
        scorer = EmbeddingSynergyScorer(analysis, model=fake)

        # This card's text contains a literal synergy keyword ("lifelink")
        synergy_card = Card(
            name="Ajani's Pridemate", mana_cost="{1}{W}", mana_value=2,
            card_type="Creature",
            text="Lifelink. Whenever you gain life, put a +1/+1 counter on this.",
            color_identity="W", colors="W",
        )
        unrelated_card = Card(
            name="Grizzly Bears", mana_cost="{1}{G}", mana_value=2,
            card_type="Creature", text="",
            color_identity="G", colors="G", power="2", toughness="2",
        )

        scores = scorer.score_cards([synergy_card, unrelated_card])
        assert scores["Ajani's Pridemate"] > scores["Grizzly Bears"]

    def test_scores_in_valid_range(self, analysis):
        """All scores should be in [0, 100]."""
        fake = FakeModel()
        scorer = EmbeddingSynergyScorer(analysis, model=fake)
        card = Card(
            name="Test", mana_cost="{1}", mana_value=1,
            card_type="Creature", text="some text",
            color_identity="", colors="",
        )
        score = scorer.score_card(card)
        assert 0 <= score <= 100

    def test_no_model_returns_neutral(self, analysis):
        """Scorer without a model returns neutral 50."""
        scorer = EmbeddingSynergyScorer(analysis, model=None)
        card = Card(
            name="Test", mana_cost="{1}", mana_value=1,
            card_type="Creature", text="",
            color_identity="", colors="",
        )
        score = scorer.score_card(card)
        assert score == 50.0

    def test_batch_scoring(self, analysis):
        fake = FakeModel()
        scorer = EmbeddingSynergyScorer(analysis, model=fake)
        cards = [
            Card(name=f"Card {i}", mana_cost="", mana_value=i, card_type="Creature",
                 text=f"Card {i}", color_identity="", colors="")
            for i in range(10)
        ]
        scores = scorer.score_cards(cards)
        assert len(scores) == 10
        for name in scores:
            assert name.startswith("Card ")


class TestConfigCustomization:
    def test_score_range_respected(self, analysis):
        """Custom floor/ceiling should bound the output range."""
        fake = FakeModel()
        config = EmbeddingConfig(score_floor=40.0, score_ceiling=80.0)
        scorer = EmbeddingSynergyScorer(analysis, config=config, model=fake)
        # Test internal similarity-to-score mapping with known similarity values
        assert scorer._similarity_to_score(0.0) == 40.0
        assert scorer._similarity_to_score(1.0) == 80.0
        assert scorer._similarity_to_score(0.5) == 60.0


class TestTextConstruction:
    def test_commander_query_includes_strategy(self, analysis):
        fake = FakeModel()
        scorer = EmbeddingSynergyScorer(analysis, model=fake)
        query = scorer._build_commander_query()
        # The NAME is deliberately excluded: name tokens produce spurious
        # cosine matches against cards sharing a word ("Jasmine Dragon Tea
        # Shop" bypassed into the GA pool for "Jasmine Boreal of the Seven")
        # without carrying any strategy semantics.
        assert analysis.name not in query
        assert "gain life" in query.lower() or "lifegain" in query.lower()

    def test_card_text_includes_text(self):
        card = Card(
            name="Soul Warden", mana_cost="{W}", mana_value=1,
            card_type="Creature", text="Whenever a creature enters",
            color_identity="W", colors="W", keywords="Flying",
        )
        text = EmbeddingSynergyScorer._build_card_text(card)
        # Name excluded (v0.9.13): name tokens create spurious cosine
        # matches against queries sharing a word, without strategy meaning.
        assert "Soul Warden" not in text
        assert "Creature" in text
        assert "Flying" in text

    def test_empty_fields_handled(self):
        # With the name excluded (v0.9.13), a card with no type/text/keywords
        # yields an empty embed text — must not crash, and stays empty rather
        # than leaking the name back in.
        card = Card(
            name="Test", mana_cost="", mana_value=0, card_type="",
            text="", color_identity="", colors="",
        )
        text = EmbeddingSynergyScorer._build_card_text(card)
        assert text == ""


class TestFactoryMethod:
    def test_factory_returns_none_without_library(self, analysis, monkeypatch):
        """create_if_available returns None when sentence-transformers missing."""
        # Simulate the library being missing
        import mtg_deck_builder.embedding_scorer as es_module
        monkeypatch.setattr(es_module, "is_embeddings_available", lambda: False)

        result = EmbeddingSynergyScorer.create_if_available(analysis)
        assert result is None


class TestSimilarityMath:
    def test_cosine_identical_vectors(self, analysis):
        fake = FakeModel()
        scorer = EmbeddingSynergyScorer(analysis, model=fake)
        v = np.array([1.0, 2.0, 3.0])
        assert abs(scorer._cosine_similarity(v, v) - 1.0) < 1e-5

    def test_cosine_orthogonal(self, analysis):
        fake = FakeModel()
        scorer = EmbeddingSynergyScorer(analysis, model=fake)
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(scorer._cosine_similarity(a, b)) < 1e-5

    def test_cosine_opposite(self, analysis):
        fake = FakeModel()
        scorer = EmbeddingSynergyScorer(analysis, model=fake)
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(scorer._cosine_similarity(a, b) - (-1.0)) < 1e-5
