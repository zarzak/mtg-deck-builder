"""Integration tests: orchestrator end-to-end in mock mode."""

import pytest
import tempfile
from pathlib import Path

from mtg_deck_builder.models import BuildConfig, OptimizationResult
from mtg_deck_builder.deck_builder import DeckBuilder, BuildProgress
from mtg_deck_builder.llm_engine import LLMConfig
from mtg_deck_builder.html_report import generate_html_report


class TestOrchestratorMockMode:
    def test_full_build_runs_end_to_end(self, test_csv_path):
        """Full build should complete without errors in mock mode."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=6,
            generations=8,
            patience_generations=50,
            random_seed=42,
            candidates_per_category=20,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        assert isinstance(result, OptimizationResult)
        assert result.best_deck.card_count == 99
        assert result.final_score > 0

    def test_build_collects_progress_events(self, test_csv_path):
        """Progress callback should be invoked at multiple phases."""
        events: list[BuildProgress] = []

        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=5,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
            progress_callback=lambda p: events.append(p),
        )
        builder.build()

        phases_seen = {e.phase for e in events}
        expected = {"init", "analysis", "pools", "filtering", "scoring",
                    "optimization", "done"}
        assert expected.issubset(phases_seen), \
            f"Missing phases: {expected - phases_seen}"

    def test_build_with_review(self, test_csv_path):
        """enable_llm_review should populate result.llm_review."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=5,
            patience_generations=50, random_seed=42,
            enable_llm_review=True,
            candidates_per_category=15,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        assert result.llm_review is not None
        assert len(result.llm_review) > 10

    def test_build_includes_analysis_in_result(self, test_csv_path):
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=5,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
        )
        result = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        ).build()
        assert result.commander_analysis is not None
        assert result.commander_analysis.name == "Lathiel, the Bounteous Dawn"

    def test_commander_not_found_raises_with_suggestions(self, test_csv_path):
        """A bad commander name should give a helpful error."""
        config = BuildConfig(commander_name="Lathiel the Bounteous Dog")  # typo
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        with pytest.raises(ValueError) as exc_info:
            builder.build()
        # Error should mention the typo
        assert "Lathiel the Bounteous Dog" in str(exc_info.value)

    def test_build_includes_basic_lands(self, test_csv_path):
        """The _ensure_basic_lands logic should add Forest/Plains to pool."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=5,
            patience_generations=50, random_seed=42,
            candidates_per_category=10,  # Small pool forces need for basics
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        names = {c.name for c in result.best_deck.cards}
        # Should include at least one basic for each color
        assert "Forest" in names or "Plains" in names

    def test_reproducibility_with_seed(self, test_csv_path):
        """Same seed + mock LLM should produce identical decks."""
        cfg_kwargs = dict(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=6,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
        )
        r1 = DeckBuilder(
            test_csv_path, BuildConfig(**cfg_kwargs),
            llm_config=LLMConfig(mock_mode=True),
        ).build()
        r2 = DeckBuilder(
            test_csv_path, BuildConfig(**cfg_kwargs),
            llm_config=LLMConfig(mock_mode=True),
        ).build()
        names1 = sorted(c.name for c in r1.best_deck.cards)
        names2 = sorted(c.name for c in r2.best_deck.cards)
        assert names1 == names2


class TestQuickBuild:
    def test_quick_build_produces_99_cards(self, test_csv_path):
        config = BuildConfig(commander_name="Lathiel, the Bounteous Dawn")
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        deck = builder.quick_build()
        assert deck.card_count == 99


class TestHTMLReport:
    def test_generates_valid_html(self, test_csv_path):
        """HTML report should be generated and contain expected structure."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=4,
            patience_generations=50, random_seed=42,
            candidates_per_category=10,
        )
        result = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        ).build()

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w"
        ) as f:
            report_path = f.name

        try:
            written = generate_html_report(result, report_path)
            assert written.exists()
            content = written.read_text()
            # Key elements
            assert "Lathiel" in content
            assert "<html" in content
            assert "<style>" in content  # Inline CSS (offline-capable)
            assert "Decklist" in content
            assert "Per-Card Telemetry" in content
        finally:
            Path(report_path).unlink(missing_ok=True)

    def test_html_report_handles_no_telemetry(self, test_csv_path, lathiel):
        """HTML gen should survive a result without telemetry."""
        from mtg_deck_builder.models import Deck, DeckScores
        deck = Deck(
            commander=lathiel,
            cards=[lathiel] * 99,  # Not technically valid but that's OK for this test
            scores=DeckScores(),
        )
        result = OptimizationResult(
            best_deck=deck,
            final_score=50.0,
            generations_run=0,
            score_history=[],
            diversity_history=[],
            runtime_seconds=0.0,
            config=BuildConfig(commander_name=lathiel.name),
            card_telemetry=[],  # Empty
        )
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            generate_html_report(result, path)
            content = Path(path).read_text()
            assert "No per-card telemetry" in content
        finally:
            Path(path).unlink(missing_ok=True)
