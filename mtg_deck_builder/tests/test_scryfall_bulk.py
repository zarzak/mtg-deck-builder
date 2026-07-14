"""
Tests for scryfall_bulk: bulk downloader + BulkCardSource adapter.
All offline — any HTTP calls are stubbed at the _http.http_get_text level.
"""

import json
import pytest
from pathlib import Path

from mtg_deck_builder.scryfall_bulk import (
    ScryfallBulkFetcher, BulkCardSource, VALID_BULK_TYPES,
)


# ---- Sample card fixtures ----

SAMPLE_CARDS = [
    {
        "object": "card",
        "name": "Sol Ring",
        "type_line": "Artifact",
        "oracle_text": "{T}: Add {C}{C}.",
        "artist": "Volkan Baga",
        "scryfall_uri": "https://scryfall.com/card/c17/237/sol-ring",
        "image_uris": {
            "small": "https://cards.scryfall.io/small/solring.jpg",
            "normal": "https://cards.scryfall.io/normal/solring.jpg",
            "large": "https://cards.scryfall.io/large/solring.jpg",
            "png": "https://cards.scryfall.io/png/solring.png",
            "art_crop": "https://cards.scryfall.io/art_crop/solring.jpg",
            "border_crop": "https://cards.scryfall.io/border_crop/solring.jpg",
        },
    },
    {
        "object": "card",
        "name": "Forest",
        "type_line": "Basic Land — Forest",
        "oracle_text": "{T}: Add {G}.",
        "artist": "John Avon",
        "scryfall_uri": "https://scryfall.com/card/m21/forest",
        "image_uris": {
            "small": "https://cards.scryfall.io/small/forest.jpg",
            "normal": "https://cards.scryfall.io/normal/forest.jpg",
            "large": "https://cards.scryfall.io/large/forest.jpg",
            "png": "https://cards.scryfall.io/png/forest.png",
            "art_crop": "https://cards.scryfall.io/art_crop/forest.jpg",
            "border_crop": "https://cards.scryfall.io/border_crop/forest.jpg",
        },
    },
    {
        "object": "card",
        "name": "Delver of Secrets // Insectile Aberration",
        "layout": "transform",
        "card_faces": [
            {
                "name": "Delver of Secrets",
                "artist": "Nils Hamm",
                "image_uris": {
                    "small": "https://example.com/delver_front_small.jpg",
                    "normal": "https://example.com/delver_front_normal.jpg",
                    "large": "https://example.com/delver_front_large.jpg",
                    "png": "https://example.com/delver_front.png",
                    "art_crop": "https://example.com/delver_front_art.jpg",
                    "border_crop": "https://example.com/delver_front_border.jpg",
                },
            },
            {
                "name": "Insectile Aberration",
                "artist": "Nils Hamm",
                "image_uris": {
                    "small": "https://example.com/delver_back_small.jpg",
                    "normal": "https://example.com/delver_back_normal.jpg",
                    "large": "https://example.com/delver_back_large.jpg",
                    "png": "https://example.com/delver_back.png",
                    "art_crop": "https://example.com/delver_back_art.jpg",
                    "border_crop": "https://example.com/delver_back_border.jpg",
                },
            },
        ],
        "scryfall_uri": "https://scryfall.com/card/delver",
    },
]


# ---- BulkCardSource ----

class TestBulkCardSource:
    def test_construct_from_list(self):
        source = BulkCardSource(SAMPLE_CARDS)
        assert source.card_count == 3

    def test_get_card_data(self):
        source = BulkCardSource(SAMPLE_CARDS)
        data = source.get_card_data("Sol Ring")
        assert data is not None
        assert data["name"] == "Sol Ring"

    def test_missing_card_returns_none(self):
        source = BulkCardSource(SAMPLE_CARDS)
        assert source.get_card_data("Not A Real Card") is None
        assert source.get_image_url("Not A Real Card") is None
        assert source.get_artist("Not A Real Card") is None
        assert source.get_scryfall_uri("Not A Real Card") is None

    def test_image_url_all_sizes(self):
        source = BulkCardSource(SAMPLE_CARDS)
        for size in ("small", "normal", "large", "png", "art_crop", "border_crop"):
            url = source.get_image_url("Sol Ring", size=size)
            assert url is not None
            assert size in url

    def test_invalid_size_returns_none(self):
        source = BulkCardSource(SAMPLE_CARDS)
        assert source.get_image_url("Sol Ring", size="huge") is None

    def test_artist(self):
        source = BulkCardSource(SAMPLE_CARDS)
        assert source.get_artist("Sol Ring") == "Volkan Baga"
        assert source.get_artist("Forest") == "John Avon"

    def test_scryfall_uri(self):
        source = BulkCardSource(SAMPLE_CARDS)
        uri = source.get_scryfall_uri("Sol Ring")
        assert uri is not None
        assert "scryfall.com" in uri

    def test_card_missing_fields_no_crash(self):
        """Card with no image_uris / artist should return None, not crash."""
        sparse = [{"name": "Sparse Card"}]
        source = BulkCardSource(sparse)
        assert source.get_card_data("Sparse Card") is not None
        assert source.get_image_url("Sparse Card") is None
        assert source.get_artist("Sparse Card") is None

    def test_card_without_name_skipped(self):
        """Entries without a name field should be silently skipped."""
        malformed = [{"oracle_text": "no name"}, {"name": "Good One"}]
        source = BulkCardSource(malformed)
        assert source.card_count == 1
        assert source.get_card_data("Good One") is not None


class TestDoubleFacedCards:
    def test_full_name_lookup(self):
        source = BulkCardSource(SAMPLE_CARDS)
        url = source.get_image_url(
            "Delver of Secrets // Insectile Aberration",
            size="small",
        )
        assert url is not None
        assert "delver_front_small" in url

    def test_front_face_alias(self):
        """Looking up by front face name alone should also work."""
        source = BulkCardSource(SAMPLE_CARDS)
        url = source.get_image_url("Delver of Secrets", size="small")
        assert url is not None
        assert "delver_front_small" in url

    def test_dfc_artist(self):
        source = BulkCardSource(SAMPLE_CARDS)
        assert source.get_artist("Delver of Secrets") == "Nils Hamm"
        assert (
            source.get_artist("Delver of Secrets // Insectile Aberration")
            == "Nils Hamm"
        )

    def test_back_face_name_does_not_alias(self):
        """Intentional: back-face name shouldn't be indexed (avoid ambiguity).

        Scryfall's tagger and search system treats the front face name as
        the canonical single-name reference. Users rarely search by back
        face in a deck-building context."""
        source = BulkCardSource(SAMPLE_CARDS)
        # The back face "Insectile Aberration" should NOT be in the index
        # as a standalone entry
        assert source.get_card_data("Insectile Aberration") is None


class TestLoadFromFile:
    def test_load_valid_file(self, tmp_path):
        f = tmp_path / "cards.json"
        f.write_text(json.dumps(SAMPLE_CARDS))
        source = BulkCardSource.load_from_file(f)
        assert source is not None
        assert source.card_count == 3

    def test_load_missing_file(self, tmp_path):
        assert BulkCardSource.load_from_file(tmp_path / "missing.json") is None

    def test_load_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json at all")
        assert BulkCardSource.load_from_file(f) is None

    def test_load_non_list_json(self, tmp_path):
        """A JSON object (not list) isn't bulk data; should gracefully fail."""
        f = tmp_path / "obj.json"
        f.write_text(json.dumps({"object": "card", "name": "Single"}))
        assert BulkCardSource.load_from_file(f) is None


# ---- ScryfallBulkFetcher ----

class TestFetcher:
    def test_valid_bulk_types(self):
        # Sanity: all types we ship for are actually valid
        assert "oracle_cards" in VALID_BULK_TYPES
        assert "default_cards" in VALID_BULK_TYPES

    def test_unknown_type_returns_none(self, tmp_path):
        f = ScryfallBulkFetcher(cache_dir=tmp_path)
        assert f.ensure_bulk("not_a_real_type") is None

    def test_offline_no_cache(self, tmp_path):
        f = ScryfallBulkFetcher(cache_dir=tmp_path, offline=True)
        assert f.ensure_bulk("oracle_cards") is None

    def test_offline_with_cache(self, tmp_path):
        """Offline + existing cache file = return path without any HTTP."""
        data_path = tmp_path / "oracle_cards.json"
        data_path.write_text(json.dumps(SAMPLE_CARDS))
        meta_path = tmp_path / "oracle_cards.meta.json"
        meta_path.write_text(json.dumps({"updated_at": "2024-01-01T00:00:00Z"}))
        f = ScryfallBulkFetcher(cache_dir=tmp_path, offline=True)
        result = f.ensure_bulk("oracle_cards")
        assert result == data_path


class TestFetcherWithStubbedHTTP:
    """Exercise the downloader path without hitting the network."""

    def _stub_http(self, monkeypatch, responses):
        """Stub http_get_text to return canned responses by URL match."""
        from mtg_deck_builder import scryfall_bulk as sb

        def fake_get(url, timeout=10.0, log_label=""):
            for substring, response in responses.items():
                if substring in url:
                    return response
            return None

        monkeypatch.setattr(sb, "http_get_text", fake_get)

    def test_full_download_cycle(self, tmp_path, monkeypatch):
        """Metadata fetch + bulk download + cache write."""
        # Simulate Scryfall's two-step flow:
        # 1. GET /bulk-data/oracle_cards -> returns bulk_data object
        # 2. GET <download_uri> -> returns the card list JSON
        metadata_response = json.dumps({
            "object": "bulk_data",
            "type": "oracle_cards",
            "updated_at": "2026-04-20T00:00:00Z",
            "size": 12345,
            "download_uri": "https://data.scryfall.io/oracle-cards/test.json",
        })
        bulk_response = json.dumps(SAMPLE_CARDS)
        self._stub_http(monkeypatch, {
            "/bulk-data/oracle_cards": metadata_response,
            "data.scryfall.io": bulk_response,
        })

        f = ScryfallBulkFetcher(cache_dir=tmp_path)
        path = f.ensure_bulk("oracle_cards")
        assert path is not None
        assert path.exists()
        # Meta file should also be written
        assert f.meta_path("oracle_cards").exists()

        # Load it and verify
        source = BulkCardSource.load_from_file(path)
        assert source is not None
        assert source.get_card_data("Sol Ring") is not None

    def test_cache_fresh_no_redownload(self, tmp_path, monkeypatch):
        """If cached metadata matches remote, we don't re-download."""
        # Pre-seed the cache
        data_path = tmp_path / "oracle_cards.json"
        data_path.write_text(json.dumps(SAMPLE_CARDS))
        meta_path = tmp_path / "oracle_cards.meta.json"
        meta_path.write_text(json.dumps({
            "updated_at": "2026-04-20T00:00:00Z",
        }))

        # Remote returns the same updated_at — cache is fresh
        metadata_response = json.dumps({
            "object": "bulk_data",
            "updated_at": "2026-04-20T00:00:00Z",
            "download_uri": "https://data.scryfall.io/should-not-be-called.json",
        })
        download_calls = []
        def fake_get(url, timeout=10.0, log_label=""):
            if "/bulk-data/" in url:
                return metadata_response
            download_calls.append(url)
            return json.dumps(SAMPLE_CARDS)

        from mtg_deck_builder import scryfall_bulk as sb
        monkeypatch.setattr(sb, "http_get_text", fake_get)

        f = ScryfallBulkFetcher(cache_dir=tmp_path)
        path = f.ensure_bulk("oracle_cards")
        assert path == data_path
        # The download URI should NOT have been fetched
        assert download_calls == []

    def test_stale_cache_triggers_redownload(self, tmp_path, monkeypatch):
        """If remote is newer than cache, we re-download."""
        data_path = tmp_path / "oracle_cards.json"
        data_path.write_text("old content")
        meta_path = tmp_path / "oracle_cards.meta.json"
        meta_path.write_text(json.dumps({
            "updated_at": "2020-01-01T00:00:00Z",  # ancient
        }))

        metadata_response = json.dumps({
            "updated_at": "2026-04-20T00:00:00Z",  # much newer
            "download_uri": "https://data.scryfall.io/new.json",
        })
        bulk_response = json.dumps(SAMPLE_CARDS)
        self._stub_http(monkeypatch, {
            "/bulk-data/": metadata_response,
            "data.scryfall.io": bulk_response,
        })

        f = ScryfallBulkFetcher(cache_dir=tmp_path)
        f.ensure_bulk("oracle_cards")
        # Cache file should have been rewritten with the new content
        content = data_path.read_text()
        assert "Sol Ring" in content

    def test_metadata_unreachable_uses_stale_cache(self, tmp_path, monkeypatch):
        """If metadata HTTP fails but a cache exists, use the stale cache."""
        data_path = tmp_path / "oracle_cards.json"
        data_path.write_text(json.dumps(SAMPLE_CARDS))
        # Note: no meta file; cache is technically not fresh but is available
        self._stub_http(monkeypatch, {})  # returns None for everything

        f = ScryfallBulkFetcher(cache_dir=tmp_path)
        path = f.ensure_bulk("oracle_cards")
        assert path == data_path  # fell back to existing cache

    def test_metadata_unreachable_no_cache(self, tmp_path, monkeypatch):
        """If metadata fails AND no cache, return None."""
        self._stub_http(monkeypatch, {})
        f = ScryfallBulkFetcher(cache_dir=tmp_path)
        assert f.ensure_bulk("oracle_cards") is None

    def test_malformed_bulk_response_ignored(self, tmp_path, monkeypatch):
        """Non-list response to the download URI should not clobber cache."""
        data_path = tmp_path / "oracle_cards.json"
        data_path.write_text(json.dumps(SAMPLE_CARDS))  # valid existing cache
        meta_path = tmp_path / "oracle_cards.meta.json"
        meta_path.write_text(json.dumps({"updated_at": "2020-01-01T00:00:00Z"}))

        metadata_response = json.dumps({
            "updated_at": "2026-04-20T00:00:00Z",
            "download_uri": "https://data.scryfall.io/bad.json",
        })
        # Download returns a JSON object, not a list
        bad_response = json.dumps({"object": "error", "code": "bad"})
        self._stub_http(monkeypatch, {
            "/bulk-data/": metadata_response,
            "data.scryfall.io": bad_response,
        })

        f = ScryfallBulkFetcher(cache_dir=tmp_path)
        path = f.ensure_bulk("oracle_cards")
        # Should return the existing cache, not overwrite it with garbage
        assert path == data_path
        # Verify original contents preserved
        content = json.loads(data_path.read_text())
        assert isinstance(content, list)
        assert any(c.get("name") == "Sol Ring" for c in content)
