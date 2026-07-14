"""
Integration test for the v0.8 layered candidate recall.

The headline claim: with `recall_use_patterns=True`, the synergy candidate
pool for a lifegain commander now contains Soul Warden — which the legacy
substring matcher silently dropped because its text says "gain 1 life",
not "gain life".

These tests build the synergy pool only (not the full deck) so they're
fast and don't depend on the GA or LLM filter.
"""

import pytest

from mtg_deck_builder.card_database import CardDatabase
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import BuildConfig, CommanderAnalysis
from mtg_deck_builder.llm_engine import LLMEngine, LLMConfig


def _build_with_analysis(test_csv_path, recall_flags: dict, analysis: CommanderAnalysis):
    """
    Construct a DeckBuilder, inject a fixed analysis, and just run pool
    generation — no GA, no LLM filter.
    """
    config = BuildConfig(
        commander_name=analysis.name,
        population_size=4, generations=2,
        patience_generations=50, random_seed=42,
        candidates_per_category=15,
        **recall_flags,
    )
    builder = DeckBuilder(
        card_database_path=test_csv_path,
        config=config,
        llm_config=LLMConfig(mock_mode=True),
    )
    # Force the lazy-loaded DB to materialize, then inject the commander
    # and analysis directly to skip the full commander-analysis phase.
    db = builder.db
    builder._commander = db.get_by_name(analysis.name)
    assert builder._commander is not None, f"{analysis.name} missing from test CSV"
    builder._analysis = analysis
    builder._phase_generate_pools()
    return builder


# ----------------------------------------------------------------------
# Smoking-gun test
# ----------------------------------------------------------------------

class TestPatternRecallFixesSoulWarden:
    """The bug: legacy substring 'gain life' doesn't match 'gain 1 life'."""

    def test_legacy_path_silently_drops_soul_warden(self, test_csv_path):
        """Confirm the bug exists in the legacy path so we know what we're fixing."""
        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W",
            key_mechanics=["lifegain"],
            build_around_text="gain life and distribute counters.",
            evaluation_notes="Lifegain triggers matter.",
            category_queries={},
            synergy_keywords=["gain life", "lifelink", "+1/+1 counter"],
            synergy_patterns=[],  # legacy path doesn't read this
        )
        builder = _build_with_analysis(test_csv_path, {}, analysis)
        names = {c.name for c in builder._candidates.synergy}
        # Soul Warden's text is "gain 1 life" — substring match for
        # "gain life" silently drops it. THIS IS THE BUG.
        assert "Soul Warden" not in names, (
            "Test fixture changed: Soul Warden is now caught by legacy "
            "matcher. Update or remove this regression-prevention test."
        )

    def test_pattern_recall_catches_soul_warden(self, test_csv_path):
        """With recall_use_patterns=True and digit-normalization, Soul Warden appears."""
        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W",
            key_mechanics=["lifegain"],
            build_around_text="gain life and distribute counters.",
            evaluation_notes="Lifegain triggers matter.",
            category_queries={},
            synergy_keywords=["gain life"],  # Same legacy keyword
            synergy_patterns=["gain life", "lifelink", "+1/+1 counter"],
        )
        builder = _build_with_analysis(
            test_csv_path,
            {"recall_use_patterns": True},
            analysis,
        )
        names = {c.name for c in builder._candidates.synergy}
        assert "Soul Warden" in names, (
            f"Soul Warden missing from synergy pool. Pool size: "
            f"{len(names)}. Sample: {sorted(names)[:10]}"
        )

    def test_legacy_path_runs_when_all_recall_flags_off(self, test_csv_path):
        """Backward-compatible default: with no recall flags, legacy path runs."""
        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W",
            key_mechanics=["lifegain"],
            build_around_text="gain life",
            evaluation_notes="...",
            category_queries={},
            # Use a keyword that matches with literal substring so we get a non-empty pool
            synergy_keywords=["lifelink"],
            synergy_patterns=["gain life"],  # would catch more if recall on
        )
        builder = _build_with_analysis(test_csv_path, {}, analysis)
        # Legacy path: only literal "lifelink" substring matches
        names = {c.name for c in builder._candidates.synergy}
        # We expect SOME results — actual count depends on test_cards.csv
        # contents. The contract is: legacy behavior preserved.
        for name in names:
            card = builder.db.get_by_name(name)
            assert card is not None
            assert "lifelink" in (card.text or "").lower()


# ----------------------------------------------------------------------
# Pattern fallback when LLM didn't produce synergy_patterns
# ----------------------------------------------------------------------

class TestPatternFallbackToKeywords:
    def test_falls_back_to_keywords_when_patterns_empty(self, test_csv_path):
        """If LLM didn't return patterns, fall back to keywords (so flag still does something)."""
        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W",
            key_mechanics=["lifegain"],
            build_around_text="gain life",
            evaluation_notes="...",
            category_queries={},
            synergy_keywords=["lifelink"],
            synergy_patterns=[],  # missing — fall back to keywords
        )
        builder = _build_with_analysis(
            test_csv_path,
            {"recall_use_patterns": True},
            analysis,
        )
        names = {c.name for c in builder._candidates.synergy}
        # We should still get cards matching the keywords; fallback path
        # ensures the flag isn't a silent no-op.
        for name in names:
            card = builder.db.get_by_name(name)
            assert card is not None
            assert "lifelink" in (card.text or "").lower()


# ----------------------------------------------------------------------
# Cap respected on union
# ----------------------------------------------------------------------

class TestPoolCap:
    def test_pool_cap_respected(self, test_csv_path):
        """recall_pool_cap is the upper bound on the synergy pool size."""
        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W",
            key_mechanics=["broad"],
            build_around_text="...",
            evaluation_notes="...",
            category_queries={},
            synergy_keywords=["creature"],  # broad — would match many
            synergy_patterns=["creature"],
        )
        builder = _build_with_analysis(
            test_csv_path,
            {"recall_use_patterns": True, "recall_pool_cap": 25},
            analysis,
        )
        assert len(builder._candidates.synergy) <= 25
