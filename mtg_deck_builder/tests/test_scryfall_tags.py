"""
Tests for ScryfallTagClient. All offline mode with seeded caches —
no real network calls happen in tests.
"""

import json
import pytest
import time
from pathlib import Path

from mtg_deck_builder.scryfall_tags import (
    ScryfallTagClient, TagCacheEntry, TAG_OPERATORS, _safe_filename,
)


class TestOfflineMode:
    def test_offline_no_cache_returns_empty(self):
        """Offline without any cache = empty list, never raises."""
        client = ScryfallTagClient(offline=True)
        assert client.get_cards_with_art_tag("mammoth") == []
        assert client.get_cards_with_oracle_tag("removal") == []

    def test_offline_uses_memory_cache(self):
        client = ScryfallTagClient(offline=True)
        client._memory_cache["art__mammoth__any"] = TagCacheEntry(
            tag="mammoth", kind="art",
            card_names=["Phyrexian Rager", "Spike Feeder", "Mammoth Spider"],
            fetched_at=time.time(),
        )
        names = client.get_cards_with_art_tag("mammoth")
        assert len(names) == 3
        assert "Mammoth Spider" in names

    def test_offline_color_identity_cached_separately(self):
        """Same tag with different color filters should cache separately."""
        client = ScryfallTagClient(offline=True)
        client._memory_cache["art__forest__any"] = TagCacheEntry(
            tag="forest", kind="art",
            card_names=["A", "B", "C"],
            fetched_at=time.time(),
        )
        # v0.6: color identity normalizes to sorted lowercase, so WG -> "gw"
        client._memory_cache["art__forest__gw"] = TagCacheEntry(
            tag="forest", kind="art",
            card_names=["A"],  # subset
            fetched_at=time.time(),
        )
        all_forest = client.get_cards_with_art_tag("forest")
        wg_forest = client.get_cards_with_art_tag("forest", color_identity="WG")
        assert len(all_forest) == 3
        assert len(wg_forest) == 1

    def test_color_identity_format_normalization(self):
        """v0.6: 'WG', 'W,G', 'gw', 'G W' should all hit the same cache entry."""
        client = ScryfallTagClient(offline=True)
        # Seed once with the canonical form
        canonical_key = ScryfallTagClient._cache_key("art", "forest", "WG")
        client._memory_cache[canonical_key] = TagCacheEntry(
            tag="forest", kind="art",
            card_names=["Test Card"],
            fetched_at=time.time(),
        )
        # Every input form should find it
        for form in ("WG", "wg", "GW", "gw", "W,G", "g,w", "W G", " w g "):
            result = client.get_cards_with_art_tag("forest", color_identity=form)
            assert result == ["Test Card"], (
                f"form={form!r} didn't hit cache; got {result}"
            )


class TestDiskCache:
    def test_disk_cache_roundtrip(self, tmp_path):
        """Write to disk, read from a fresh instance."""
        client = ScryfallTagClient(cache_dir=tmp_path, offline=True)
        entry = TagCacheEntry(
            tag="mammoth", kind="art",
            card_names=["Card A", "Card B"],
            fetched_at=time.time(),
        )
        client._write_disk_cache("art__mammoth__any", entry)

        # Fresh instance
        client2 = ScryfallTagClient(cache_dir=tmp_path, offline=True)
        names = client2.get_cards_with_art_tag("mammoth")
        assert names == ["Card A", "Card B"]

    def test_stale_cache_ignored(self, tmp_path):
        """Entries past TTL shouldn't be returned (in offline mode = empty)."""
        client = ScryfallTagClient(
            cache_dir=tmp_path, offline=True, ttl_seconds=1,
        )
        old_entry = TagCacheEntry(
            tag="mammoth", kind="art",
            card_names=["X"], fetched_at=time.time() - 100,
        )
        client._write_disk_cache("art__mammoth__any", old_entry)

        # Fresh instance, cache is stale, offline -> empty
        client2 = ScryfallTagClient(
            cache_dir=tmp_path, offline=True, ttl_seconds=1,
        )
        assert client2.get_cards_with_art_tag("mammoth") == []

    def test_corrupt_cache_file(self, tmp_path):
        """Malformed JSON should be gracefully ignored."""
        path = tmp_path / f"{_safe_filename('art__mammoth__any')}.json"
        path.write_text("garbage not json", encoding="utf-8")
        client = ScryfallTagClient(cache_dir=tmp_path, offline=True)
        # Should not raise
        assert client.get_cards_with_art_tag("mammoth") == []


class TestQueryBuilding:
    def test_unknown_kind_returns_empty(self):
        """Internal _query_tag should handle unknown kinds gracefully."""
        client = ScryfallTagClient(offline=True)
        assert client._query_tag("not_a_kind", "mammoth", None) == []

    def test_cache_key_includes_all_dimensions(self):
        """Cache keys must distinguish kind, tag, and color identity."""
        key_a = ScryfallTagClient._cache_key("art", "forest", None)
        key_b = ScryfallTagClient._cache_key("oracle", "forest", None)
        key_c = ScryfallTagClient._cache_key("art", "forest", "WG")
        key_d = ScryfallTagClient._cache_key("art", "mammoth", None)
        assert len({key_a, key_b, key_c, key_d}) == 4

    def test_cache_key_normalizes_case(self):
        """Same tag in different cases should hit the same cache entry."""
        k1 = ScryfallTagClient._cache_key("art", "Mammoth", "wg")
        k2 = ScryfallTagClient._cache_key("art", "mammoth", "WG")
        assert k1 == k2


class TestPaginationHandling:
    """
    Test that _fetch_paginated stops at has_more=False and respects max_pages.
    We stub _http_get to return canned JSON responses.
    """

    def test_single_page_no_more(self):
        client = ScryfallTagClient(offline=False)
        # Prevent real network access
        page1 = json.dumps({
            "object": "list",
            "has_more": False,
            "data": [{"name": "Alpha"}, {"name": "Beta"}],
        })
        calls = []
        def fake_get(url):
            calls.append(url)
            return page1
        client._http_get = fake_get

        names = client._fetch_paginated("art:mammoth")
        assert names == ["Alpha", "Beta"]
        assert len(calls) == 1

    def test_follows_next_page(self):
        client = ScryfallTagClient(offline=False)
        page1 = json.dumps({
            "object": "list",
            "has_more": True,
            "next_page": "https://api.scryfall.com/cards/search?q=art:mammoth&page=2",
            "data": [{"name": "Alpha"}],
        })
        page2 = json.dumps({
            "object": "list",
            "has_more": False,
            "data": [{"name": "Beta"}],
        })
        responses = [page1, page2]
        def fake_get(url):
            return responses.pop(0)
        client._http_get = fake_get

        names = client._fetch_paginated("art:mammoth")
        assert names == ["Alpha", "Beta"]

    def test_respects_max_pages(self):
        client = ScryfallTagClient(offline=False, max_pages=2)
        page = json.dumps({
            "object": "list",
            "has_more": True,
            "next_page": "https://api.scryfall.com/cards/search?page=next",
            "data": [{"name": "Card"}],
        })
        calls = []
        def fake_get(url):
            calls.append(url)
            return page
        client._http_get = fake_get

        client._fetch_paginated("art:popular")
        assert len(calls) == 2  # max_pages cap enforced

    def test_handles_404_error_body(self):
        """A 404 comes back as object=='error'; should return empty, not crash."""
        client = ScryfallTagClient(offline=False)
        error_body = json.dumps({
            "object": "error",
            "code": "not_found",
            "details": "No cards found.",
        })
        client._http_get = lambda url: error_body
        assert client._fetch_paginated("art:nonexistent") == []

    def test_handles_malformed_json(self):
        """Garbage response -> empty list, not crash."""
        client = ScryfallTagClient(offline=False)
        client._http_get = lambda url: "<html>error</html>"
        assert client._fetch_paginated("art:mammoth") == []

    def test_handles_none_response(self):
        """Network failure -> empty list."""
        client = ScryfallTagClient(offline=False)
        client._http_get = lambda url: None
        assert client._fetch_paginated("art:mammoth") == []


class TestFullQueryFlow:
    def test_full_query_with_stubbed_http(self):
        """Complete end-to-end: query builds, http called, result cached."""
        client = ScryfallTagClient(offline=False)
        page = json.dumps({
            "object": "list",
            "has_more": False,
            "data": [{"name": "Llanowar Elves"}, {"name": "Elvish Mystic"}],
        })
        captured_urls = []
        def fake_get(url):
            captured_urls.append(url)
            return page
        client._http_get = fake_get

        names = client.get_cards_with_oracle_tag("ramp", color_identity="G")
        assert "Llanowar Elves" in names

        # URL should contain our query operator and color filter
        assert "otag%3Aramp" in captured_urls[0] or "otag:ramp" in captured_urls[0]
        # ColorId filter
        assert "id%3C%3Dg" in captured_urls[0] or "id<=g" in captured_urls[0]

        # Second call should hit memory cache (no new HTTP)
        captured_urls.clear()
        names2 = client.get_cards_with_oracle_tag("ramp", color_identity="G")
        assert names2 == names
        assert captured_urls == []

    def test_art_tag_operator_used(self):
        """get_cards_with_art_tag should use `art:` operator, not `otag:`."""
        client = ScryfallTagClient(offline=False)
        page = json.dumps({
            "object": "list", "has_more": False,
            "data": [{"name": "Mammoth Card"}],
        })
        captured = []
        client._http_get = lambda url: (captured.append(url), page)[1]

        client.get_cards_with_art_tag("mammoth")
        joined = " ".join(captured)
        assert "art%3Amammoth" in joined or "art:mammoth" in joined
        # And NOT otag
        assert "otag" not in joined


class TestOperatorMapping:
    def test_both_operators_present(self):
        assert "art" in TAG_OPERATORS
        assert "oracle" in TAG_OPERATORS
        assert TAG_OPERATORS["art"] == "art"
        assert TAG_OPERATORS["oracle"] == "otag"
