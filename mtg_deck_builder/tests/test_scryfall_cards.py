"""
Tests for ScryfallCardSource. All offline — we seed the cache directly
rather than hitting the real API.
"""

import json
import pytest
import time
from pathlib import Path

from mtg_deck_builder.scryfall_cards import (
    ScryfallCardSource, CardCacheEntry, VALID_IMAGE_SIZES,
    _safe_filename,
)


# Sample Scryfall card JSON (trimmed from a real response)
SAMPLE_SOL_RING = {
    "object": "card",
    "name": "Sol Ring",
    "mana_cost": "{1}",
    "type_line": "Artifact",
    "oracle_text": "{T}: Add {C}{C}.",
    "artist": "Volkan Baga",
    "scryfall_uri": "https://scryfall.com/card/c17/237/sol-ring",
    "image_uris": {
        "small": "https://cards.scryfall.io/small/front/x/y/solring.jpg",
        "normal": "https://cards.scryfall.io/normal/front/x/y/solring.jpg",
        "large": "https://cards.scryfall.io/large/front/x/y/solring.jpg",
        "png": "https://cards.scryfall.io/png/front/x/y/solring.png",
        "art_crop": "https://cards.scryfall.io/art_crop/front/x/y/solring.jpg",
        "border_crop": "https://cards.scryfall.io/border_crop/front/x/y/solring.jpg",
    },
}

# Sample transform/double-faced card — image_uris live on card_faces entries
SAMPLE_DFC = {
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
    "scryfall_uri": "https://scryfall.com/card/x/delver",
}


class TestOfflineMode:
    def test_offline_no_cache_returns_none(self):
        """Without cache, offline mode returns None."""
        src = ScryfallCardSource(offline=True)
        assert src.get_card_data("Sol Ring") is None
        assert src.get_image_url("Sol Ring") is None

    def test_offline_uses_memory_cache(self):
        """Memory-seeded cache entries are returned offline."""
        src = ScryfallCardSource(offline=True)
        src._memory_cache["Sol Ring"] = CardCacheEntry(
            data=SAMPLE_SOL_RING, fetched_at=time.time(),
        )
        data = src.get_card_data("Sol Ring")
        assert data is not None
        assert data["name"] == "Sol Ring"

    def test_offline_uses_disk_cache(self, tmp_path):
        """Disk-seeded cache entries are loaded without network."""
        src = ScryfallCardSource(cache_dir=tmp_path, offline=True)
        entry = CardCacheEntry(data=SAMPLE_SOL_RING, fetched_at=time.time())
        src._write_disk_cache("Sol Ring", entry)

        # Fresh instance; should load from disk
        src2 = ScryfallCardSource(cache_dir=tmp_path, offline=True)
        data = src2.get_card_data("Sol Ring")
        assert data is not None
        assert data["name"] == "Sol Ring"


class TestImageURLs:
    def _seeded(self):
        src = ScryfallCardSource(offline=True)
        src._memory_cache["Sol Ring"] = CardCacheEntry(
            data=SAMPLE_SOL_RING, fetched_at=time.time(),
        )
        return src

    def test_small_image(self):
        src = self._seeded()
        url = src.get_image_url("Sol Ring", size="small")
        assert url is not None
        assert "small" in url

    def test_art_crop(self):
        src = self._seeded()
        url = src.get_image_url("Sol Ring", size="art_crop")
        assert url is not None
        assert "art_crop" in url

    def test_all_valid_sizes_work(self):
        src = self._seeded()
        for size in VALID_IMAGE_SIZES:
            url = src.get_image_url("Sol Ring", size=size)
            assert url is not None, f"Size {size} returned None"

    def test_invalid_size_returns_none(self):
        src = self._seeded()
        assert src.get_image_url("Sol Ring", size="huge") is None

    def test_missing_card_returns_none(self):
        src = ScryfallCardSource(offline=True)
        assert src.get_image_url("Nonexistent Card") is None


class TestDoubleFacedCards:
    def _seeded_dfc(self):
        src = ScryfallCardSource(offline=True)
        src._memory_cache["Delver of Secrets"] = CardCacheEntry(
            data=SAMPLE_DFC, fetched_at=time.time(),
        )
        return src

    def test_dfc_uses_front_face(self):
        """Double-faced cards: get_image_url should return front-face URL."""
        src = self._seeded_dfc()
        url = src.get_image_url("Delver of Secrets", size="small")
        assert url is not None
        assert "delver_front_small" in url
        assert "delver_back" not in url

    def test_dfc_art_crop(self):
        src = self._seeded_dfc()
        url = src.get_image_url("Delver of Secrets", size="art_crop")
        assert url is not None
        assert "delver_front_art" in url

    def test_dfc_artist_from_front(self):
        src = self._seeded_dfc()
        artist = src.get_artist("Delver of Secrets")
        assert artist == "Nils Hamm"


class TestMetadata:
    def test_artist(self):
        src = ScryfallCardSource(offline=True)
        src._memory_cache["Sol Ring"] = CardCacheEntry(
            data=SAMPLE_SOL_RING, fetched_at=time.time(),
        )
        assert src.get_artist("Sol Ring") == "Volkan Baga"

    def test_scryfall_uri(self):
        src = ScryfallCardSource(offline=True)
        src._memory_cache["Sol Ring"] = CardCacheEntry(
            data=SAMPLE_SOL_RING, fetched_at=time.time(),
        )
        uri = src.get_scryfall_uri("Sol Ring")
        assert uri is not None
        assert "scryfall.com" in uri

    def test_missing_artist_returns_none(self):
        """Card with no artist field should return None, not crash."""
        src = ScryfallCardSource(offline=True)
        src._memory_cache["NoArt"] = CardCacheEntry(
            data={"name": "NoArt"},  # no artist field
            fetched_at=time.time(),
        )
        assert src.get_artist("NoArt") is None


class TestDiskCacheTTL:
    def test_stale_cache_ignored(self, tmp_path):
        """Entries older than TTL shouldn't be returned."""
        src = ScryfallCardSource(
            cache_dir=tmp_path, offline=True, ttl_seconds=1,
        )
        # Write a 10-second-old entry
        old_entry = CardCacheEntry(
            data=SAMPLE_SOL_RING, fetched_at=time.time() - 10,
        )
        src._write_disk_cache("Sol Ring", old_entry)

        # Fresh instance: cache is stale + offline -> None
        src2 = ScryfallCardSource(
            cache_dir=tmp_path, offline=True, ttl_seconds=1,
        )
        assert src2.get_card_data("Sol Ring") is None

    def test_fresh_cache_used(self, tmp_path):
        """Entries within TTL are returned from disk."""
        src = ScryfallCardSource(
            cache_dir=tmp_path, offline=True, ttl_seconds=3600,
        )
        src._write_disk_cache(
            "Sol Ring",
            CardCacheEntry(data=SAMPLE_SOL_RING, fetched_at=time.time()),
        )
        src2 = ScryfallCardSource(
            cache_dir=tmp_path, offline=True, ttl_seconds=3600,
        )
        data = src2.get_card_data("Sol Ring")
        assert data is not None


class TestCacheCorruption:
    def test_corrupt_cache_file_doesnt_crash(self, tmp_path):
        """Malformed JSON on disk should be gracefully ignored."""
        path = tmp_path / f"{_safe_filename('Sol Ring')}.json"
        path.write_text("not valid json at all", encoding="utf-8")
        src = ScryfallCardSource(cache_dir=tmp_path, offline=True)
        # Shouldn't raise
        assert src.get_card_data("Sol Ring") is None


class TestNotFoundHandling:
    def test_404_style_none_data(self):
        """A cached 'not found' entry (data=None) returns None."""
        src = ScryfallCardSource(offline=True)
        src._memory_cache["Fake Card"] = CardCacheEntry(
            data=None, fetched_at=time.time(),
        )
        assert src.get_card_data("Fake Card") is None
        assert src.get_image_url("Fake Card") is None
