"""
Tests for the v0.9.6 synergy_engine pre-rank + top-tier bypass.

Headline claims under test:
  - _rank_synergy_engine_pool orders by (adaptive hint tier, cosine, name),
    with tier taking precedence over cosine.
  - The top `synergy_engine_bypass` cards reach filtered.synergy_engine
    WITHOUT the LLM (guaranteed into the GA pool).
  - The LLM only ever sees the top `synergy_engine_shortlist` cards (after
    the bypass slice), not the whole recall pool.
  - Ranking degrades gracefully when embedding scores / hints are absent.
  - _comparable_score_history hides the incomparable fast-eval segment.

Mock-mode LLM keeps these deterministic and API-free.
"""

import pytest

from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import BuildConfig, CommanderAnalysis, Card
from mtg_deck_builder.llm_engine import LLMConfig
from mtg_deck_builder.html_report import _comparable_score_history


def _card(name: str, text: str = "") -> Card:
    return Card(
        name=name, mana_cost="{1}{G}", mana_value=2,
        card_type="Creature", text=text or f"text {name}",
        color_identity="G", colors="G",
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _make_builder(test_csv_path, **overrides) -> DeckBuilder:
    analysis = CommanderAnalysis(
        name="Lathiel, the Bounteous Dawn",
        color_identity="G,W",
        key_mechanics=["lifegain"],
        build_around_text="gain life",
        evaluation_notes="...",
        category_queries={},
        synergy_keywords=["gain life"],
        synergy_patterns=["gain life"],
    )
    config = BuildConfig(
        commander_name=analysis.name,
        random_seed=42,
        candidates_per_category=10,
        **overrides,
    )
    builder = DeckBuilder(
        card_database_path=test_csv_path,
        config=config,
        llm_config=LLMConfig(mock_mode=True),
    )
    builder._commander = builder.db.get_by_name(analysis.name)
    builder._analysis = analysis
    return builder


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------

class TestRankSynergyEnginePool:
    def test_cosine_orders_within_same_tier(self, test_csv_path):
        b = _make_builder(test_csv_path)
        b._embedding_recall_scores = {"Hi": 0.9, "Mid": 0.5, "Lo": 0.1}
        pool = [_card("Lo"), _card("Hi"), _card("Mid")]
        ranked = b._rank_synergy_engine_pool(pool, hints={})
        assert [c.name for c in ranked] == ["Hi", "Mid", "Lo"]

    def test_tier_beats_cosine(self, test_csv_path):
        # A top-tier card with a LOW cosine must outrank an untagged card
        # with a HIGH cosine — tier dominates the sort key.
        b = _make_builder(test_csv_path)
        b._embedding_recall_scores = {"Tagged": 0.01, "Untagged": 0.99}
        hints = {"Tagged": "[SYN+++]"}
        pool = [_card("Untagged"), _card("Tagged")]
        ranked = b._rank_synergy_engine_pool(pool, hints)
        assert [c.name for c in ranked] == ["Tagged", "Untagged"]

    def test_missing_cosine_defaults_to_zero(self, test_csv_path):
        # No embedding scores at all → falls back to (tier, name) and never
        # raises. Pure alphabetical when untagged.
        b = _make_builder(test_csv_path)
        b._embedding_recall_scores = {}
        pool = [_card("Charlie"), _card("Alpha"), _card("Bravo")]
        ranked = b._rank_synergy_engine_pool(pool, hints={})
        assert [c.name for c in ranked] == ["Alpha", "Bravo", "Charlie"]


# ----------------------------------------------------------------------
# Bypass + shortlist integration
# ----------------------------------------------------------------------

class TestBypassAndShortlist:
    def _setup(self, test_csv_path):
        b = _make_builder(
            test_csv_path,
            synergy_engine_target=6,
            synergy_engine_bypass=3,
            synergy_engine_shortlist=2,
        )
        b._phase_generate_pools()

        # Deterministic synergy pool of 8 distinct cards not in role buckets.
        pool = [_card(f"Eng{i:02d}") for i in range(8)]
        b._candidates.synergy = pool
        # Descending cosine: Eng00 highest … Eng07 lowest.
        b._embedding_recall_scores = {
            f"Eng{i:02d}": (8 - i) / 8.0 for i in range(8)
        }
        return b, pool

    def test_top_cards_bypass_into_synergy_engine(self, test_csv_path):
        b, pool = self._setup(test_csv_path)
        b._phase_llm_filtering()
        engine = {c.name for c in b._candidates.synergy_engine}
        # Top-3 by cosine are guaranteed present (the bypass).
        assert {"Eng00", "Eng01", "Eng02"}.issubset(engine)

    def test_llm_only_sees_shortlist(self, test_csv_path):
        b, pool = self._setup(test_csv_path)

        captured = {}
        original = b.llm.select_synergy_engine_cards

        def spy(analysis, candidates, count, already_selected=None,
                synergy_hints=None):
            captured["names"] = [c.name for c in candidates]
            captured["count"] = count
            return original(
                analysis, candidates, count,
                already_selected=already_selected,
                synergy_hints=synergy_hints,
            )

        b.llm.select_synergy_engine_cards = spy
        b._phase_llm_filtering()

        # Shortlist = the 2 cards right after the 3 bypassed → Eng03, Eng04.
        assert captured["names"] == ["Eng03", "Eng04"]
        # remaining = target(6) - bypass(3) = 3.
        assert captured["count"] == 3

    def test_bypass_zero_falls_back_to_pure_llm(self, test_csv_path):
        b = _make_builder(
            test_csv_path,
            synergy_engine_target=5,
            synergy_engine_bypass=0,
            synergy_engine_shortlist=4,
        )
        b._phase_generate_pools()
        pool = [_card(f"Eng{i:02d}") for i in range(8)]
        b._candidates.synergy = pool
        b._embedding_recall_scores = {
            f"Eng{i:02d}": (8 - i) / 8.0 for i in range(8)
        }

        captured = {}
        original = b.llm.select_synergy_engine_cards

        def spy(analysis, candidates, count, already_selected=None,
                synergy_hints=None):
            captured["names"] = [c.name for c in candidates]
            return original(
                analysis, candidates, count,
                already_selected=already_selected, synergy_hints=synergy_hints,
            )

        b.llm.select_synergy_engine_cards = spy
        b._phase_llm_filtering()
        # No bypass → LLM sees the top-4 shortlist (Eng00..Eng03).
        assert captured["names"] == ["Eng00", "Eng01", "Eng02", "Eng03"]


# ----------------------------------------------------------------------
# Sparkline: hide the incomparable fast-eval phase
# ----------------------------------------------------------------------

class TestComparableScoreHistory:
    def test_keeps_only_full_eval_points(self):
        history = [10.0, 20.0, 30.0, 5.0, 6.0, 7.0]
        modes = ["fast", "fast", "fast", "full", "full", "full"]
        assert _comparable_score_history(history, modes) == [5.0, 6.0, 7.0]

    def test_no_modes_returns_full_history(self):
        history = [1.0, 2.0, 3.0]
        assert _comparable_score_history(history, None) == history

    def test_mismatched_length_returns_full_history(self):
        history = [1.0, 2.0, 3.0]
        assert _comparable_score_history(history, ["full"]) == history

    def test_no_full_points_falls_back_to_all(self):
        history = [1.0, 2.0]
        modes = ["fast", "fast"]
        assert _comparable_score_history(history, modes) == history
