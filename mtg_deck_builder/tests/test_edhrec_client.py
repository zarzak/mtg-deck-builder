"""Tests for EDHRECClient — all offline, no real HTTP calls."""

import json
import pytest
import tempfile
import time
from pathlib import Path

from mtg_deck_builder.edhrec_client import (
    EDHRECClient, EDHRECCardData, EDHRECCommanderData,
)


# Sample EDHREC JSON response shape (based on the public format)
SAMPLE_RESPONSE = {
    "container": {
        "json_dict": {
            "cardlists": [
                {
                    "tag": "highsynergycards",
                    "header": "High Synergy Cards",
                    "cardviews": [
                        {
                            "name": "Soul Warden",
                            "synergy": 0.45,
                            "num_decks": 4500,
                            "potential_decks": 5000,
                        },
                        {
                            "name": "Archangel of Thune",
                            "synergy": 0.62,
                            "num_decks": 4800,
                            "potential_decks": 5000,
                        },
                    ],
                },
                {
                    "tag": "topcards",
                    "header": "Top Cards",
                    "cardviews": [
                        {
                            "name": "Sol Ring",
                            "synergy": 0.02,  # low commander-specific synergy
                            "num_decks": 4900,
                            "potential_decks": 5000,  # ubiquitous (precon bias!)
                        },
                    ],
                },
            ],
        },
    },
}


class TestSlugify:
    def test_basic_name(self):
        assert EDHRECClient._slugify("Lathiel, the Bounteous Dawn") == \
            "lathiel-the-bounteous-dawn"

    def test_apostrophe_removed(self):
        assert EDHRECClient._slugify("Kodama's Reach") == "kodamas-reach"

    def test_special_chars(self):
        assert EDHRECClient._slugify("Ach! Hans, Run!") == "ach-hans-run"

    def test_punctuation_stripped(self):
        assert EDHRECClient._slugify("Jasmine Boreal") == "jasmine-boreal"


class TestCardDataConversion:
    def test_positive_synergy_maps_above_50(self):
        c = EDHRECCardData(name="X", synergy=0.5)
        assert c.to_synergy_score() > 50

    def test_negative_synergy_maps_below_50(self):
        c = EDHRECCardData(name="X", synergy=-0.5)
        assert c.to_synergy_score() < 50

    def test_neutral_synergy_is_50(self):
        c = EDHRECCardData(name="X", synergy=0.0)
        assert c.to_synergy_score() == 50.0

    def test_extreme_synergy_clamped(self):
        c = EDHRECCardData(name="X", synergy=5.0)  # absurdly high
        assert c.to_synergy_score() == 100.0

    def test_extreme_negative_clamped(self):
        c = EDHRECCardData(name="X", synergy=-5.0)
        assert c.to_synergy_score() == 0.0

    def test_missing_synergy_uses_inclusion_rate(self):
        c = EDHRECCardData(name="X", synergy=None, inclusion_rate=0.9)
        # 50 + 0.9 * 30 = 77
        assert 75 <= c.to_synergy_score() <= 80

    def test_missing_both_returns_neutral(self):
        c = EDHRECCardData(name="X")
        assert c.to_synergy_score() == 50.0

    def test_baseline_power_from_inclusion(self):
        # High inclusion = high baseline
        high = EDHRECCardData(name="Sol Ring", inclusion_rate=0.95)
        assert high.to_baseline_power() > 80

        # Low inclusion = modest baseline
        niche = EDHRECCardData(name="Weird Card", inclusion_rate=0.05)
        assert niche.to_baseline_power() < 50

    def test_baseline_floor(self):
        c = EDHRECCardData(name="X", inclusion_rate=0.01)
        assert c.to_baseline_power() >= 30


class TestParseData:
    def test_parses_sample_response(self):
        client = EDHRECClient(offline=True)
        data = client._parse_data("Lathiel", "lathiel", SAMPLE_RESPONSE)
        assert data is not None
        assert "Soul Warden" in data.cards
        assert "Archangel of Thune" in data.cards
        assert "Sol Ring" in data.cards

    def test_synergy_score_from_parsed(self):
        client = EDHRECClient(offline=True)
        data = client._parse_data("Lathiel", "lathiel", SAMPLE_RESPONSE)
        # Archangel has higher synergy than Soul Warden
        archangel_score = data.get_synergy_score("Archangel of Thune")
        soul_warden_score = data.get_synergy_score("Soul Warden")
        assert archangel_score > soul_warden_score

    def test_inclusion_rate_computed(self):
        client = EDHRECClient(offline=True)
        data = client._parse_data("Lathiel", "lathiel", SAMPLE_RESPONSE)
        sol_ring = data.cards["Sol Ring"]
        # 4900 / 5000 = 0.98
        assert sol_ring.inclusion_rate is not None
        assert 0.97 < sol_ring.inclusion_rate < 0.99

    def test_missing_card_returns_none(self):
        client = EDHRECClient(offline=True)
        data = client._parse_data("Lathiel", "lathiel", SAMPLE_RESPONSE)
        assert data.get_synergy_score("Nonexistent Card") is None

    def test_precon_bias_mitigation(self):
        """Sol Ring has low synergy (+0.02) despite high inclusion (0.98).

        Our synergy score should track synergy, not inclusion."""
        client = EDHRECClient(offline=True)
        data = client._parse_data("Lathiel", "lathiel", SAMPLE_RESPONSE)
        sol_ring_score = data.get_synergy_score("Sol Ring")
        archangel_score = data.get_synergy_score("Archangel of Thune")
        # Archangel (synergy 0.62) should score much higher than Sol Ring (synergy 0.02)
        # despite Sol Ring being in 98% of decks
        assert archangel_score > sol_ring_score

    def test_high_synergy_sort(self):
        client = EDHRECClient(offline=True)
        data = client._parse_data("Lathiel", "lathiel", SAMPLE_RESPONSE)
        high = data.get_high_synergy_cards(min_synergy=0.1)
        assert len(high) >= 2
        # Sorted descending
        assert high[0].synergy >= high[1].synergy

    def test_malformed_response_returns_empty_data(self):
        client = EDHRECClient(offline=True)
        bad = {"foo": "bar"}  # no cardlists
        data = client._parse_data("X", "x", bad)
        assert data is not None
        assert len(data.cards) == 0


class TestCache:
    def test_cache_roundtrip(self, tmp_path):
        client = EDHRECClient(cache_dir=tmp_path, offline=True)
        client._write_cache("test-slug", SAMPLE_RESPONSE)
        cached = client._read_cache("test-slug")
        assert cached == SAMPLE_RESPONSE

    def test_stale_cache_ignored(self, tmp_path):
        client = EDHRECClient(cache_dir=tmp_path, offline=True, ttl_seconds=1)
        client._write_cache("test-slug", SAMPLE_RESPONSE)

        # Move the file's mtime into the past
        cache_file = tmp_path / "test-slug.json"
        old_time = time.time() - 10  # 10 seconds ago, TTL is 1
        import os
        os.utime(cache_file, (old_time, old_time))

        cached = client._read_cache("test-slug")
        assert cached is None

    def test_missing_cache_returns_none(self, tmp_path):
        client = EDHRECClient(cache_dir=tmp_path, offline=True)
        assert client._read_cache("never-written") is None

    def test_no_cache_dir(self):
        """With no cache_dir, reads return None, writes no-op."""
        client = EDHRECClient(cache_dir=None, offline=True)
        assert client._read_cache("x") is None
        client._write_cache("x", {})  # no-op, should not raise

    def test_offline_mode_never_fetches(self, tmp_path):
        """In offline mode, fetch_commander returns None without HTTP."""
        client = EDHRECClient(cache_dir=tmp_path, offline=True)
        result = client.fetch_commander("Totally Unknown Commander")
        assert result is None

    def test_uses_cached_even_in_offline(self, tmp_path):
        """Cached data should be returned even in offline mode."""
        client = EDHRECClient(cache_dir=tmp_path, offline=True)
        client._write_cache("lathiel-the-bounteous-dawn", SAMPLE_RESPONSE)
        result = client.fetch_commander("Lathiel, the Bounteous Dawn")
        assert result is not None
        assert "Soul Warden" in result.cards


class TestFetchCombos:
    """v0.9.30: the combos-page fetch (human-verified combo database)."""

    COMBOS_PAGE = {
        "container": {"json_dict": {"cardlists": [
            {"header": "Doomsday + Bolas's Citadel (24,072 decks)",
             "cardviews": [{"name": "Doomsday"},
                           {"name": "Bolas's Citadel"}]},
            {"header": "A + B + C (99 decks)",
             "cardviews": [{"name": "A"}, {"name": "B"}, {"name": "C"}]},
            {"header": "Malformed single",
             "cardviews": [{"name": "Lonely Card"}]},
        ]}}
    }

    def test_parses_cached_page(self, tmp_path):
        client = EDHRECClient(cache_dir=tmp_path, offline=True)
        client._write_cache("combos_doctor-doom-unrivaled", self.COMBOS_PAGE)
        combos = client.fetch_combos("Doctor Doom, Unrivaled")
        assert {"cards": ["Doomsday", "Bolas's Citadel"], "decks": 24072} \
            in combos
        assert {"cards": ["A", "B", "C"], "decks": 99} in combos
        # Single-card sections are malformed, not combos.
        assert len(combos) == 2

    def test_offline_no_cache_returns_empty(self, tmp_path):
        client = EDHRECClient(cache_dir=tmp_path, offline=True)
        assert client.fetch_combos("Unknown Commander") == []
