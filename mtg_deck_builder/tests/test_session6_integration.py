"""
Session 6 integration tests: bulk source + role validation through DeckBuilder.
All use --mock LLM and offline tag/bulk modes — no network calls.
"""

import json
import time
import pytest
from pathlib import Path

from mtg_deck_builder.models import BuildConfig
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.llm_engine import LLMConfig
from mtg_deck_builder.scryfall_bulk import BulkCardSource
from mtg_deck_builder.scryfall_tags import ScryfallTagClient, TagCacheEntry


SAMPLE_BULK = [
    {
        "object": "card",
        "name": "Sol Ring",
        "type_line": "Artifact",
        "image_uris": {"small": "https://example.com/solring_small.jpg"},
        "artist": "Some Artist",
    },
    {
        "object": "card",
        "name": "Forest",
        "type_line": "Basic Land — Forest",
        "image_uris": {"small": "https://example.com/forest_small.jpg"},
        "artist": "John Avon",
    },
]


class TestBulkSourceIntegration:
    def test_build_with_bulk_source_offline(self, test_csv_path, tmp_path):
        """Full build with use_bulk_source=True and a seeded cache."""
        # Pre-seed the bulk cache
        data_path = tmp_path / "oracle_cards.json"
        data_path.write_text(json.dumps(SAMPLE_BULK))
        meta_path = tmp_path / "oracle_cards.meta.json"
        meta_path.write_text(json.dumps({"updated_at": "2026-04-20T00:00:00Z"}))

        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            use_bulk_source=True,
            bulk_cache_dir=str(tmp_path),
            bulk_offline=True,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        # card_source should be a BulkCardSource
        src = builder.card_source
        assert isinstance(src, BulkCardSource)
        # Build should complete
        result = builder.build()
        assert result.best_deck.card_count == 99

    def test_bulk_source_missing_cache_falls_back_gracefully(
        self, test_csv_path, tmp_path
    ):
        """Offline + no cache = card_source is None (not a crash)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            use_bulk_source=True,
            bulk_cache_dir=str(empty_dir),
            bulk_offline=True,
            # Note: use_images=False, so there's nothing to fall back to
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        # card_source should be None (bulk failed, no per-card fallback)
        assert builder.card_source is None
        # But the build should still succeed
        result = builder.build()
        assert result.best_deck.card_count == 99

    def test_bulk_source_falls_back_to_images_when_bulk_fails(
        self, test_csv_path, tmp_path
    ):
        """If bulk offline has no cache but use_images=True, use per-card source."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            use_bulk_source=True,
            bulk_cache_dir=str(empty_dir),
            bulk_offline=True,
            use_images=True,
            images_offline=True,  # will also be empty
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        # Should get a ScryfallCardSource as the fallback
        from mtg_deck_builder.scryfall_cards import ScryfallCardSource
        assert isinstance(builder.card_source, ScryfallCardSource)


class TestRoleValidationIntegration:
    def test_validate_roles_offline_skips_all(self, test_csv_path, tmp_path):
        """Offline tag client with no cache: all roles skipped, no disagreements."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            validate_roles_after_build=True,
            tags_cache_dir=str(tmp_path),
            tags_offline=True,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        assert result.role_validation_report is not None
        report = result.role_validation_report
        # Offline + no cache -> every role is skipped
        assert len(report.skipped_roles) == len(report.roles_checked)
        assert report.total_disagreements == 0

    def test_validate_roles_with_seeded_tag_cache(
        self, test_csv_path, tmp_path
    ):
        """When tag cache has data, validation reports meaningful results."""
        # Build a tag cache on disk first
        tag_client_seed = ScryfallTagClient(
            cache_dir=str(tmp_path), offline=True,
        )
        # Seed entries for ramp — say only Sol Ring is community-tagged
        ramp_entry = TagCacheEntry(
            tag="mana-ramp", kind="oracle",
            card_names=["Sol Ring"], fetched_at=time.time(),
        )
        key = ScryfallTagClient._cache_key(
            "oracle", "mana-ramp", "GW",  # color-filtered key
        )
        tag_client_seed._memory_cache[key] = ramp_entry
        tag_client_seed._write_disk_cache(key, ramp_entry)

        # Also seed the unfiltered cache in case validation uses no color filter
        key_any = ScryfallTagClient._cache_key(
            "oracle", "mana-ramp", None,
        )
        tag_client_seed._memory_cache[key_any] = ramp_entry
        tag_client_seed._write_disk_cache(key_any, ramp_entry)

        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            validate_roles_after_build=True,
            tags_cache_dir=str(tmp_path),
            tags_offline=True,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        assert result.role_validation_report is not None
        # At minimum we should have validated ramp (since we seeded its cache)
        # The "unavailable" roles (draw/removal/wipe) will be skipped
        report = result.role_validation_report
        # ramp should NOT be in skipped_roles (we seeded it)
        assert "ramp" not in report.skipped_roles

    def test_validate_roles_off_by_default(self, test_csv_path):
        """Without the flag, no validation runs and no report is attached."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        assert result.role_validation_report is None
