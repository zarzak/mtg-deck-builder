"""Tests for price sources and budget filtering. All offline."""

import json
import time
import pytest
from pathlib import Path

from mtg_deck_builder.models import Card
from mtg_deck_builder.price_source import (
    NullPriceSource, StaticPriceSource, ScryfallPriceSource,
    PriceCacheEntry, filter_cards_by_budget, deck_total_price,
    _safe_filename,
)


def _make_card(name: str) -> Card:
    return Card(
        name=name, mana_cost="{1}", mana_value=1, card_type="Creature",
        text="", color_identity="", colors="",
    )


class TestNullPriceSource:
    def test_always_returns_none(self):
        src = NullPriceSource()
        assert src.get_price("Any Card") is None
        assert src.get_price("") is None


class TestStaticPriceSource:
    def test_known_price(self):
        src = StaticPriceSource({"Sol Ring": 1.50, "Lightning Bolt": 0.25})
        assert src.get_price("Sol Ring") == 1.50
        assert src.get_price("Lightning Bolt") == 0.25

    def test_unknown_card(self):
        src = StaticPriceSource({"Sol Ring": 1.50})
        assert src.get_price("Missing Card") is None


class TestScryfallOfflineMode:
    def test_offline_returns_none(self):
        """Offline mode should never fetch."""
        src = ScryfallPriceSource(offline=True)
        assert src.get_price("Sol Ring") is None

    def test_uses_memory_cache_even_offline(self):
        """If cache is populated, offline mode should use it."""
        src = ScryfallPriceSource(offline=True)
        src._memory_cache["Sol Ring"] = PriceCacheEntry(
            price=1.25, fetched_at=time.time(),
        )
        assert src.get_price("Sol Ring") == 1.25

    def test_stale_memory_cache_invalid(self):
        """Cache entries older than TTL shouldn't be returned."""
        src = ScryfallPriceSource(offline=True, ttl_seconds=10)
        src._memory_cache["Sol Ring"] = PriceCacheEntry(
            price=1.25,
            fetched_at=time.time() - 100,  # 100s ago, TTL 10
        )
        # Offline + stale cache => None
        assert src.get_price("Sol Ring") is None


class TestScryfallDiskCache:
    def test_disk_cache_roundtrip(self, tmp_path):
        src = ScryfallPriceSource(cache_dir=tmp_path, offline=True)
        entry = PriceCacheEntry(price=2.50, fetched_at=time.time())
        src._write_disk_cache("Sol Ring", entry)
        # Fresh instance, populate via cache
        src2 = ScryfallPriceSource(cache_dir=tmp_path, offline=True)
        assert src2.get_price("Sol Ring") == 2.50

    def test_stale_disk_cache_ignored(self, tmp_path):
        src = ScryfallPriceSource(cache_dir=tmp_path, offline=True, ttl_seconds=1)
        # Write an old entry
        old_entry = PriceCacheEntry(price=1.00, fetched_at=time.time() - 10)
        src._write_disk_cache("Sol Ring", old_entry)
        # Fresh instance, offline: stale disk should yield None
        src2 = ScryfallPriceSource(cache_dir=tmp_path, offline=True, ttl_seconds=1)
        assert src2.get_price("Sol Ring") is None

    def test_corrupted_cache_file(self, tmp_path):
        """A malformed cache file shouldn't crash reads."""
        path = tmp_path / f"{_safe_filename('Sol Ring')}.json"
        path.write_text("this is not json", encoding="utf-8")
        src = ScryfallPriceSource(cache_dir=tmp_path, offline=True)
        # Should gracefully return None, not raise
        assert src.get_price("Sol Ring") is None


class TestBudgetFilter:
    def test_no_budget_returns_all(self):
        cards = [_make_card("A"), _make_card("B")]
        src = NullPriceSource()
        result = filter_cards_by_budget(cards, src, max_price_per_card=None)
        assert result == cards

    def test_filters_expensive_cards(self):
        cards = [_make_card("Cheap"), _make_card("Expensive"), _make_card("MidPrice")]
        src = StaticPriceSource({"Cheap": 0.50, "Expensive": 100.00, "MidPrice": 5.00})
        result = filter_cards_by_budget(cards, src, max_price_per_card=10.0)
        names = {c.name for c in result}
        assert "Cheap" in names
        assert "MidPrice" in names
        assert "Expensive" not in names

    def test_exclude_unknown_false_keeps_unknown(self):
        """By default, unknown-price cards are kept."""
        cards = [_make_card("Known"), _make_card("Unknown")]
        src = StaticPriceSource({"Known": 1.0})  # "Unknown" has no price
        result = filter_cards_by_budget(
            cards, src, max_price_per_card=5.0, exclude_unknown=False,
        )
        names = {c.name for c in result}
        assert "Known" in names
        assert "Unknown" in names

    def test_exclude_unknown_true_drops_unknown(self):
        cards = [_make_card("Known"), _make_card("Unknown")]
        src = StaticPriceSource({"Known": 1.0})
        result = filter_cards_by_budget(
            cards, src, max_price_per_card=5.0, exclude_unknown=True,
        )
        names = {c.name for c in result}
        assert "Known" in names
        assert "Unknown" not in names

    def test_exactly_at_budget_included(self):
        """Cards at exactly the budget should be included (<=, not <)."""
        cards = [_make_card("Exactly10")]
        src = StaticPriceSource({"Exactly10": 10.0})
        result = filter_cards_by_budget(cards, src, max_price_per_card=10.0)
        assert len(result) == 1


class TestDeckTotal:
    def test_sum_of_known_prices(self):
        cards = [_make_card("A"), _make_card("B"), _make_card("C")]
        src = StaticPriceSource({"A": 1.50, "B": 2.00, "C": 0.50})
        assert deck_total_price(cards, src) == 4.0

    def test_unknown_treated_as_zero(self):
        cards = [_make_card("Known"), _make_card("Unknown")]
        src = StaticPriceSource({"Known": 5.0})
        assert deck_total_price(cards, src) == 5.0

    def test_empty_deck(self):
        assert deck_total_price([], NullPriceSource()) == 0.0


class TestFilenameSafety:
    def test_strip_apostrophe(self):
        assert "Kodama" in _safe_filename("Kodama's Reach")
        assert "'" not in _safe_filename("Kodama's Reach")

    def test_strip_special_chars(self):
        safe = _safe_filename("Ach! Hans, Run!")
        assert "!" not in safe
        assert "," not in safe

    def test_truncation(self):
        long_name = "A" * 200
        assert len(_safe_filename(long_name)) <= 100
