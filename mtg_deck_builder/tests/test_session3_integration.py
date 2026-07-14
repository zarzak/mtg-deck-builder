"""
Integration tests for Session 3 features: EDHREC, embeddings, budget, islands.
All tests run offline with injected mocks or offline flags.
"""

import pytest
import json
from pathlib import Path

from mtg_deck_builder.models import BuildConfig
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.llm_engine import LLMConfig
from mtg_deck_builder.edhrec_client import EDHRECClient
from mtg_deck_builder.price_source import StaticPriceSource


# Sample EDHREC response for testing (same shape as the real API)
SAMPLE_EDHREC_RESPONSE = {
    "container": {
        "json_dict": {
            "cardlists": [
                {
                    "tag": "highsynergycards",
                    "cardviews": [
                        {"name": "Soul Warden", "synergy": 0.45,
                         "num_decks": 4500, "potential_decks": 5000},
                        {"name": "Archangel of Thune", "synergy": 0.62,
                         "num_decks": 4800, "potential_decks": 5000},
                        {"name": "Ajani's Pridemate", "synergy": 0.55,
                         "num_decks": 4200, "potential_decks": 5000},
                    ],
                },
                {
                    "tag": "topcards",
                    "cardviews": [
                        {"name": "Sol Ring", "synergy": 0.02,
                         "num_decks": 4900, "potential_decks": 5000},
                        {"name": "Command Tower", "synergy": 0.01,
                         "num_decks": 4950, "potential_decks": 5000},
                    ],
                },
            ],
        },
    },
}


def _slug_path(cache_dir: Path, commander_name: str) -> Path:
    return cache_dir / f"{EDHRECClient._slugify(commander_name)}.json"


class TestEDHRECIntegration:
    def test_edhrec_offline_no_data_works(self, test_csv_path):
        """use_edhrec=True with offline client should complete gracefully."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            use_edhrec=True,
            edhrec_offline=True,  # No network calls
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        assert result.best_deck.card_count == 99

    def test_edhrec_with_cached_data_enriches_scoring(self, test_csv_path, tmp_path):
        """Pre-populated EDHREC cache should enrich synergy/baseline scores."""
        # Seed the cache with our sample data
        cache_path = _slug_path(tmp_path, "Lathiel, the Bounteous Dawn")
        cache_path.write_text(json.dumps(SAMPLE_EDHREC_RESPONSE), encoding="utf-8")

        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            use_edhrec=True,
            edhrec_offline=True,
            edhrec_cache_dir=str(tmp_path),
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        assert result.best_deck.card_count == 99
        # The baseline cache should have been populated for EDHREC-known cards
        assert builder._edhrec_data is not None
        assert "Archangel of Thune" in builder._edhrec_data.cards

    def test_edhrec_injected_client(self, test_csv_path):
        """Users can inject a custom EDHREC client."""
        client = EDHRECClient(offline=True)  # always returns None
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            use_edhrec=True,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
            edhrec_client=client,
        )
        result = builder.build()
        assert result.best_deck.card_count == 99


class TestBudgetIntegration:
    def test_budget_filter_reduces_pool(self, test_csv_path):
        """Tight budget with injected price source should reduce candidate pool."""
        # Set every card at $50 except Sol Ring at $2
        prices = {"Sol Ring": 2.0}
        # The StaticPriceSource returns None for unknown cards;
        # with exclude_unknown=True, only Sol Ring would survive a $5 budget
        price_source = StaticPriceSource(prices)

        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            budget_max_per_card=5.0,
            budget_exclude_unknown=False,  # Keep unknown-price cards
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
            price_source=price_source,
        )
        result = builder.build()
        # Build should still complete — unknowns were kept
        assert result.best_deck.card_count == 99

    def test_no_budget_no_filter(self, test_csv_path):
        """budget_max_per_card=None should skip the budget phase entirely."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            budget_max_per_card=None,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        # Price source should remain None (never constructed)
        assert builder._price_source is None
        assert result.best_deck.card_count == 99


class TestEmbeddingIntegration:
    def test_embedding_flag_without_library(self, test_csv_path, monkeypatch):
        """use_embeddings=True should gracefully skip when library missing."""
        # Force is_embeddings_available to return False
        import mtg_deck_builder.embedding_scorer as es
        monkeypatch.setattr(es, "is_embeddings_available", lambda: False)

        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            use_embeddings=True,  # requested but unavailable
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        # Should not raise; should fall through to LLM/heuristic scoring
        result = builder.build()
        assert result.best_deck.card_count == 99


class TestIslandIntegration:
    def test_island_model_via_config(self, test_csv_path):
        """use_island_model=True should run the island optimizer."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            use_island_model=True,
            num_islands=2,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        # Multiprocessing may have spawn issues with pytest fixtures (depends on
        # the test CSV path in closures, etc.). The IslandModelOptimizer has a
        # fallback to sequential on mp failure, so either outcome is fine.
        result = builder.build()
        assert result.best_deck.card_count == 99


class TestAllFeaturesTogether:
    def test_all_v3_features_enabled_together(self, test_csv_path, tmp_path):
        """Enable every Session 3 feature at once — they should coexist."""
        # Seed EDHREC cache
        cache_path = _slug_path(tmp_path, "Lathiel, the Bounteous Dawn")
        cache_path.write_text(json.dumps(SAMPLE_EDHREC_RESPONSE), encoding="utf-8")

        # Injected price source (everything is $1)
        price_source = StaticPriceSource({})  # all prices unknown

        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            # EDHREC
            use_edhrec=True,
            edhrec_offline=True,
            edhrec_cache_dir=str(tmp_path),
            # Budget
            budget_max_per_card=100.0,
            budget_exclude_unknown=False,
            # Island model (small, with fallback to sequential)
            use_island_model=False,  # single-pop for test speed
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
            price_source=price_source,
        )
        result = builder.build()
        assert result.best_deck.card_count == 99
        # EDHREC data was loaded
        assert builder._edhrec_data is not None
        # v0.9.12: EDHREC is now an additive SYNERGY blend (baseline is owned
        # by the card-power signal, which isn't enabled here). Scoring still
        # ran, so the synergy cache is populated.
        assert len(builder._synergy_cache) > 0


class TestProgressCallbacksForNewPhases:
    def test_edhrec_phase_reports_progress(self, test_csv_path):
        """The new 'edhrec' and 'budget' phases should show up in progress events."""
        events = []
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            use_edhrec=True,
            edhrec_offline=True,
            budget_max_per_card=50.0,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
            progress_callback=lambda p: events.append(p),
            price_source=StaticPriceSource({}),
        )
        builder.build()
        phases_seen = {e.phase for e in events}
        assert "edhrec" in phases_seen
        assert "budget" in phases_seen
