"""
Integration tests for v0.4 HTML reports with images.
Uses offline mode with seeded cache — no network calls.
"""

import json
import pytest
import time
import tempfile
from pathlib import Path

from mtg_deck_builder.models import (
    Card, Deck, DeckScores, BuildConfig, OptimizationResult, CommanderAnalysis,
)
from mtg_deck_builder.html_report import generate_html_report
from mtg_deck_builder.scryfall_cards import ScryfallCardSource, CardCacheEntry


SAMPLE_IMAGE_URIS = {
    "small": "https://cards.scryfall.io/small/x.jpg",
    "normal": "https://cards.scryfall.io/normal/x.jpg",
    "large": "https://cards.scryfall.io/large/x.jpg",
    "png": "https://cards.scryfall.io/png/x.png",
    "art_crop": "https://cards.scryfall.io/art_crop/x.jpg",
    "border_crop": "https://cards.scryfall.io/border_crop/x.jpg",
}


def _make_card(name: str, is_creature: bool = True) -> Card:
    if is_creature:
        return Card(
            name=name, mana_cost="{G}", mana_value=1,
            card_type="Creature — Elf", text="",
            color_identity="G", colors="G",
            power="1", toughness="1",
            types="Creature", subtypes="Elf",
        )
    return Card(
        name=name, mana_cost="", mana_value=0,
        card_type="Land", text="{T}: Add {G}.",
        color_identity="G", colors="",
        types="Land", subtypes="Forest",
    )


def _seed_source_with_cards(card_names: list[str]) -> ScryfallCardSource:
    """Create a card source and seed its memory cache so offline mode can serve."""
    source = ScryfallCardSource(offline=True)
    for name in card_names:
        source._memory_cache[name] = CardCacheEntry(
            data={
                "name": name,
                "image_uris": SAMPLE_IMAGE_URIS,
                "artist": "Test Artist",
                "scryfall_uri": f"https://scryfall.com/card/x/{name}",
            },
            fetched_at=time.time(),
        )
    return source


@pytest.fixture
def sample_result(lathiel, wg_pool):
    """Build a minimal OptimizationResult for report generation."""
    cards = list(wg_pool[:99])
    while len(cards) < 99:
        cards.append(wg_pool[0])
    deck = Deck(
        commander=lathiel,
        cards=cards[:99],
        scores=DeckScores(
            mana_curve=80, role_coverage=75, synergy=60,
            power_level=55, creativity=70, flavor=0,
        ),
    )
    from mtg_deck_builder.models import CardTelemetry
    telemetry = [
        CardTelemetry(
            name=c.name, baseline_power=50, synergy_score=60,
            effective_score=55, role="synergy/other",
        )
        for c in deck.cards[:10]
    ]
    return OptimizationResult(
        best_deck=deck,
        final_score=65.0,
        generations_run=10,
        score_history=[50, 55, 60, 65],
        diversity_history=[1.0, 0.9, 0.85, 0.8],
        runtime_seconds=2.5,
        config=BuildConfig(commander_name=lathiel.name),
        card_telemetry=telemetry,
        commander_analysis=CommanderAnalysis(
            name=lathiel.name,
            color_identity=lathiel.color_identity,
            key_mechanics=["lifegain"],
            build_around_text="Gain life, distribute counters.",
            evaluation_notes="",
            category_queries={},
            synergy_keywords=["gain life"],
        ),
    )


class TestHTMLReportWithoutImages:
    def test_renders_without_card_source(self, sample_result, tmp_path):
        """Without card_source, report should render cleanly — no <img> tags."""
        out = tmp_path / "report.html"
        generate_html_report(sample_result, out)
        content = out.read_text()
        assert "<html" in content
        assert sample_result.best_deck.commander.name in content
        # No <img> tags when no card_source
        assert "<img" not in content

    def test_backwards_compat_no_card_source_param(self, sample_result, tmp_path):
        """Explicitly omitting card_source kwarg should work (default None)."""
        out = tmp_path / "report.html"
        # No card_source kwarg at all
        path = generate_html_report(sample_result, out)
        assert path.exists()


class TestHTMLReportWithImages:
    def test_renders_with_card_source(self, sample_result, tmp_path):
        """With card_source, report should contain <img> tags."""
        # Seed all telemetry card names AND the commander
        names = [t.name for t in sample_result.card_telemetry]
        names.append(sample_result.best_deck.commander.name)
        source = _seed_source_with_cards(names)

        out = tmp_path / "report.html"
        generate_html_report(sample_result, out, card_source=source)
        content = out.read_text()

        # Telemetry thumbnails appear
        assert "<img" in content
        # lazy-loading attribute present for mobile performance
        assert 'loading="lazy"' in content

    def test_commander_art_included(self, sample_result, tmp_path):
        source = _seed_source_with_cards([sample_result.best_deck.commander.name])
        out = tmp_path / "report.html"
        generate_html_report(sample_result, out, card_source=source)
        content = out.read_text()
        # Commander art_crop URL appears
        assert "art_crop" in content

    def test_artist_credit_shown(self, sample_result, tmp_path):
        source = _seed_source_with_cards([sample_result.best_deck.commander.name])
        out = tmp_path / "report.html"
        generate_html_report(sample_result, out, card_source=source)
        content = out.read_text()
        # Our seed data sets artist = "Test Artist"
        assert "Test Artist" in content

    def test_missing_images_degrade_gracefully(self, sample_result, tmp_path):
        """If card_source can't find a card, report still generates (just no image for it)."""
        # Seed only ONE card; the rest will return None from get_image_url
        source = _seed_source_with_cards(["Sol Ring"])  # not in our telemetry
        out = tmp_path / "report.html"
        # Should not crash
        generate_html_report(sample_result, out, card_source=source)
        content = out.read_text()
        # Report still valid HTML
        assert "<html" in content
        assert sample_result.best_deck.commander.name in content

    def test_card_source_exceptions_dont_break_report(self, sample_result, tmp_path):
        """v0.5.5 regression: a card_source that raises must not crash the
        whole HTML render. Cards just render without images."""

        class ExplodingCardSource:
            """Simulates a buggy custom card_source implementation."""
            def get_image_url(self, name, size="small"):
                raise RuntimeError(f"oops on {name}")

            def get_artist(self, name):
                raise RuntimeError(f"oops artist on {name}")

        out = tmp_path / "report.html"
        # Should not raise
        generate_html_report(
            sample_result, out,
            card_source=ExplodingCardSource(),
        )
        content = out.read_text()
        # Report still valid HTML
        assert "<html" in content
        assert sample_result.best_deck.commander.name in content
        # No <img> tags because every call raised
        assert "<img" not in content


class TestImagesInBuildConfig:
    def test_use_images_default_false(self):
        cfg = BuildConfig(commander_name="Test")
        assert cfg.use_images is False

    def test_images_fields_configurable(self):
        cfg = BuildConfig(
            commander_name="Test",
            use_images=True,
            images_cache_dir="/tmp/my_cache",
            images_offline=True,
        )
        assert cfg.use_images is True
        assert cfg.images_cache_dir == "/tmp/my_cache"
        assert cfg.images_offline is True


class TestDeckBuilderCardSource:
    def test_card_source_property_none_without_flag(self, test_csv_path):
        """card_source should stay None when use_images=False."""
        from mtg_deck_builder.deck_builder import DeckBuilder
        from mtg_deck_builder.llm_engine import LLMConfig
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn")
        builder = DeckBuilder(
            test_csv_path, cfg,
            llm_config=LLMConfig(mock_mode=True),
        )
        assert builder.card_source is None

    def test_card_source_property_constructs_when_enabled(self, test_csv_path, tmp_path):
        """When use_images=True, card_source lazy-constructs a ScryfallCardSource."""
        from mtg_deck_builder.deck_builder import DeckBuilder
        from mtg_deck_builder.llm_engine import LLMConfig
        cfg = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            use_images=True,
            images_cache_dir=str(tmp_path),
            images_offline=True,
        )
        builder = DeckBuilder(
            test_csv_path, cfg,
            llm_config=LLMConfig(mock_mode=True),
        )
        source = builder.card_source
        assert source is not None
        assert isinstance(source, ScryfallCardSource)
        assert source.offline is True

    def test_injected_card_source_respected(self, test_csv_path):
        """User can inject their own card_source, bypassing config."""
        from mtg_deck_builder.deck_builder import DeckBuilder
        from mtg_deck_builder.llm_engine import LLMConfig
        my_source = _seed_source_with_cards(["Sol Ring"])
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn")
        builder = DeckBuilder(
            test_csv_path, cfg,
            llm_config=LLMConfig(mock_mode=True),
            card_source=my_source,
        )
        assert builder.card_source is my_source
