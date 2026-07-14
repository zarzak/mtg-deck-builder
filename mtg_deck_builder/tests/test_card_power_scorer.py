"""
Tests for v0.9.7 LLM intrinsic card-power scoring.

Covers:
  - CardPowerScorer: JSON parse, clamping, batching, global disk cache,
    stale-text re-scoring, malformed-response tolerance.
  - The synergy-led pre-rank blend: power nudges a card up within a tier but
    cosine still dominates, tiers still gate, and an empty power map reproduces
    the v0.9.6 (cosine-only) ordering exactly.

No real API: a fake LLM echoes a power for every card name in the prompt.
"""

import json
import re

import pytest

from mtg_deck_builder.card_power_scorer import (
    CardPowerScorer, _extract_json_array,
)
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import BuildConfig, CommanderAnalysis, Card
from mtg_deck_builder.llm_engine import LLMConfig


def _card(name: str, text: str = "") -> Card:
    return Card(
        name=name, mana_cost="{1}{G}", mana_value=2,
        card_type="Creature", text=text or f"text {name}",
        color_identity="G", colors="G",
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


class _EchoLLM:
    """Fake LLM: returns a fixed power for every card name in the prompt.

    Records call count and per-call batch sizes so tests can assert caching
    and batching behavior without touching the network.
    """

    def __init__(self, power=50):
        self.power = power
        self.call_count = 0
        self.batch_sizes: list[int] = []

    def _call_api(self, system, user, temperature=None, max_tokens=None,
                  model=None):
        names = re.findall(r"\*\*([^*]+)\*\*", user)
        self.call_count += 1
        self.batch_sizes.append(len(names))
        return json.dumps([{"name": n.strip(), "power": self.power}
                           for n in names])


# ----------------------------------------------------------------------
# JSON extraction
# ----------------------------------------------------------------------

class TestExtractJsonArray:
    def test_plain_array(self):
        assert _extract_json_array('[{"name":"A","power":50}]') == \
            [{"name": "A", "power": 50}]

    def test_strips_markdown_fences(self):
        text = '```json\n[{"name":"A","power":50}]\n```'
        assert _extract_json_array(text) == [{"name": "A", "power": 50}]

    def test_ignores_surrounding_prose(self):
        text = 'Here you go:\n[{"name":"A","power":1}]\nThanks!'
        assert _extract_json_array(text) == [{"name": "A", "power": 1}]

    def test_raises_without_array(self):
        with pytest.raises(ValueError):
            _extract_json_array("no array here")


# ----------------------------------------------------------------------
# Scorer
# ----------------------------------------------------------------------

class TestCardPowerScorer:
    def test_scores_all_requested(self, tmp_path):
        llm = _EchoLLM(power=72)
        scorer = CardPowerScorer(llm, cache_dir=str(tmp_path))
        out = scorer.score_cards([_card("A"), _card("B")])
        assert out == {"A": 72.0, "B": 72.0}

    def test_clamps_out_of_range(self, tmp_path):
        class _Over:
            def _call_api(self, *a, **k):
                return json.dumps([{"name": "A", "power": 150},
                                   {"name": "B", "power": -5}])
        scorer = CardPowerScorer(_Over(), cache_dir=str(tmp_path))
        out = scorer.score_cards([_card("A"), _card("B")])
        assert out == {"A": 100.0, "B": 0.0}

    def test_cache_hit_skips_llm(self, tmp_path):
        llm = _EchoLLM()
        scorer = CardPowerScorer(llm, cache_dir=str(tmp_path))
        cards = [_card("A"), _card("B")]
        scorer.score_cards(cards)
        assert llm.call_count == 1
        # Second call: everything cached → no new API call.
        scorer.score_cards(cards)
        assert llm.call_count == 1

    def test_only_new_cards_scored(self, tmp_path):
        llm = _EchoLLM()
        scorer = CardPowerScorer(llm, cache_dir=str(tmp_path))
        scorer.score_cards([_card("A")])
        scorer.score_cards([_card("A"), _card("B")])  # A cached, B new
        assert llm.batch_sizes == [1, 1]  # second call scored only B

    def test_stale_text_rescored(self, tmp_path):
        llm = _EchoLLM()
        scorer = CardPowerScorer(llm, cache_dir=str(tmp_path))
        scorer.score_cards([_card("A", text="old text")])
        assert llm.call_count == 1
        # Same name, different oracle text → hash mismatch → re-score.
        scorer.score_cards([_card("A", text="brand new text")])
        assert llm.call_count == 2

    def test_batching_splits_calls(self, tmp_path):
        llm = _EchoLLM()
        scorer = CardPowerScorer(llm, cache_dir=str(tmp_path), batch_size=2)
        cards = [_card(f"C{i}") for i in range(5)]
        scorer.score_cards(cards)
        assert llm.batch_sizes == [2, 2, 1]
        assert llm.call_count == 3

    def test_malformed_batch_tolerated(self, tmp_path):
        class _Junk:
            def _call_api(self, *a, **k):
                return "not json at all"
        scorer = CardPowerScorer(_Junk(), cache_dir=str(tmp_path))
        out = scorer.score_cards([_card("A")])
        # Unparseable → card simply absent (caller falls back to heuristic).
        assert out == {}

    def test_disk_cache_persists_across_instances(self, tmp_path):
        llm1 = _EchoLLM(power=88)
        CardPowerScorer(llm1, cache_dir=str(tmp_path)).score_cards([_card("A")])
        assert llm1.call_count == 1
        # A fresh scorer over the same dir reads the pickle — no new call.
        llm2 = _EchoLLM(power=11)
        out = CardPowerScorer(llm2, cache_dir=str(tmp_path)).score_cards(
            [_card("A")])
        assert llm2.call_count == 0
        assert out == {"A": 88.0}  # cached value, not the new LLM's 11


# ----------------------------------------------------------------------
# Synergy-led pre-rank blend
# ----------------------------------------------------------------------

def _make_builder(test_csv_path, **overrides) -> DeckBuilder:
    analysis = CommanderAnalysis(
        name="Lathiel, the Bounteous Dawn", color_identity="G,W",
        key_mechanics=["lifegain"], build_around_text="gain life",
        evaluation_notes="...", category_queries={},
        synergy_keywords=["gain life"], synergy_patterns=["gain life"],
    )
    config = BuildConfig(commander_name=analysis.name, random_seed=42,
                         **overrides)
    b = DeckBuilder(card_database_path=test_csv_path, config=config,
                    llm_config=LLMConfig(mock_mode=True))
    b._commander = b.db.get_by_name(analysis.name)
    b._analysis = analysis
    return b


class TestPreRankPowerBlend:
    def test_power_nudges_card_up_within_tier(self, test_csv_path):
        # Same tier (untagged). B has lower cosine but high power; with the
        # default 0.15 weight it should edge ahead: A=0.50, B=0.40+0.15=0.55.
        b = _make_builder(test_csv_path)
        b._embedding_recall_scores = {"A": 0.50, "B": 0.40}
        b._card_power_scores = {"A": 0.0, "B": 100.0}
        ranked = b._rank_synergy_engine_pool([_card("A"), _card("B")], hints={})
        assert [c.name for c in ranked] == ["B", "A"]

    def test_cosine_still_dominates_large_gap(self, test_csv_path):
        # Synergy-led: a big cosine lead is NOT overcome by max power.
        # A=0.80, B=0.40+0.15=0.55 → A stays ahead.
        b = _make_builder(test_csv_path)
        b._embedding_recall_scores = {"A": 0.80, "B": 0.40}
        b._card_power_scores = {"A": 0.0, "B": 100.0}
        ranked = b._rank_synergy_engine_pool([_card("A"), _card("B")], hints={})
        assert [c.name for c in ranked] == ["A", "B"]

    def test_tier_still_gates(self, test_csv_path):
        # A tagged +++ card with zero power/cosine still beats an untagged
        # high-power high-cosine card — tier is the primary key.
        b = _make_builder(test_csv_path)
        b._embedding_recall_scores = {"Tagged": 0.0, "Untagged": 0.99}
        b._card_power_scores = {"Tagged": 0.0, "Untagged": 100.0}
        ranked = b._rank_synergy_engine_pool(
            [_card("Untagged"), _card("Tagged")],
            hints={"Tagged": "[SYN+++]"},
        )
        assert [c.name for c in ranked] == ["Tagged", "Untagged"]

    def test_empty_power_reproduces_cosine_only(self, test_csv_path):
        # No power scores → composite == cosine → identical to v0.9.6.
        b = _make_builder(test_csv_path)
        b._embedding_recall_scores = {"Hi": 0.9, "Mid": 0.5, "Lo": 0.1}
        b._card_power_scores = {}
        ranked = b._rank_synergy_engine_pool(
            [_card("Lo"), _card("Hi"), _card("Mid")], hints={})
        assert [c.name for c in ranked] == ["Hi", "Mid", "Lo"]

    def test_weight_zero_disables_blend(self, test_csv_path):
        b = _make_builder(test_csv_path, card_power_recall_weight=0.0)
        b._embedding_recall_scores = {"A": 0.50, "B": 0.40}
        b._card_power_scores = {"A": 0.0, "B": 100.0}
        ranked = b._rank_synergy_engine_pool([_card("A"), _card("B")], hints={})
        # Power ignored → pure cosine ordering.
        assert [c.name for c in ranked] == ["A", "B"]
