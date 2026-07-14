"""
Scryfall bulk data downloader and local card lookup (v0.6).

Scryfall provides daily-updated JSON dumps of their card database as "bulk
data". Downloading once gives us every card's full JSON locally — images,
prices, artists, everything — eliminating per-card HTTP lookups for a
single ~130MB download.

Two classes:

- `ScryfallBulkFetcher` — handles downloading, caching, and metadata
  checks. Knows about the `/bulk-data` API endpoint and the
  content-encoding quirks.

- `BulkCardSource` — drop-in replacement for `ScryfallCardSource`. Same
  interface (`get_card_data`, `get_image_url`, `get_artist`,
  `get_scryfall_uri`) but backed by a local name→JSON index built once
  at construction time.

Usage:

    from mtg_deck_builder.scryfall_bulk import (
        ScryfallBulkFetcher, BulkCardSource,
    )

    fetcher = ScryfallBulkFetcher(cache_dir="./scryfall_bulk")
    path = fetcher.ensure_bulk("oracle_cards")  # ~130MB, one HTTP call
    source = BulkCardSource.load_from_file(path)
    url = source.get_image_url("Sol Ring", size="small")  # O(1) lookup

Design notes:
- Two bulk types are most useful: "oracle_cards" (~130MB, one entry per
  unique Oracle name) and "default_cards" (~300MB, one entry per
  printing). For our use case — image lookup by name — "oracle_cards" is
  the right choice.
- Scryfall serves the bulk file with Content-Encoding: gzip when the
  client accepts it. urllib handles this transparently; `requests` does
  too if we don't set Accept-Encoding explicitly.
- We decompress on write so the on-disk cache is plain JSON. Simpler to
  inspect and test, and the disk savings from gzip aren't worth the
  load-time cost of decompressing every build.
- The metadata endpoint tells us `updated_at` and we compare against our
  cached copy's mtime. If Scryfall's is newer, we re-download.
- Double-faced cards (transforms, adventures, split cards) are indexed
  by BOTH the full name ("Wear // Tear") AND the front face name ("Wear")
  so lookups work either way.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ._http import http_get_text

logger = logging.getLogger(__name__)


BULK_METADATA_URL = "https://api.scryfall.com/bulk-data"

# Valid bulk types per Scryfall docs
VALID_BULK_TYPES = frozenset({
    "oracle_cards",     # ~130MB, one per unique Oracle name — preferred
    "unique_artwork",   # ~200MB, one per unique illustration
    "default_cards",    # ~300MB, one per printing (English only)
    "all_cards",        # ~2GB, everything — probably too big
    "rulings",          # rulings, not cards
})


class ScryfallBulkFetcher:
    """
    Downloads and caches Scryfall's bulk data files.

    The cache stores two files per bulk type:
    - `<type>.json` — the actual card data
    - `<type>.meta.json` — the bulk_data API object (has updated_at,
      size, checksum info) captured at download time

    On each `ensure_bulk()` call we fetch the latest metadata from
    /bulk-data/:type and compare its `updated_at` to our stored metadata.
    If Scryfall has a newer version, we download; otherwise we reuse the
    cached file.
    """

    DEFAULT_TIMEOUT = 60.0  # bulk downloads can be slow

    def __init__(
        self,
        cache_dir: str | Path,
        offline: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.timeout = timeout

    def data_path(self, bulk_type: str) -> Path:
        return self.cache_dir / f"{bulk_type}.json"

    def meta_path(self, bulk_type: str) -> Path:
        return self.cache_dir / f"{bulk_type}.meta.json"

    def ensure_bulk(self, bulk_type: str = "oracle_cards") -> Optional[Path]:
        """
        Ensure a fresh copy of the requested bulk type is on disk.

        Returns the path to the JSON file on success, or None on failure
        (e.g. offline mode with no cache, network error). Never raises.

        If the cached file is still current (per Scryfall's updated_at),
        this is a single cheap metadata HTTP call.
        """
        if bulk_type not in VALID_BULK_TYPES:
            logger.warning(f"Unknown bulk type {bulk_type!r}")
            return None

        data_file = self.data_path(bulk_type)

        # Offline mode: return path iff cache exists
        if self.offline:
            return data_file if data_file.exists() else None

        # Fetch latest metadata to see if re-download is needed
        metadata_url = f"{BULK_METADATA_URL}/{bulk_type}"
        raw = http_get_text(
            metadata_url, timeout=self.timeout,
            log_label="Scryfall bulk meta",
        )
        remote_meta: Optional[dict] = None
        if raw is not None:
            try:
                remote_meta = json.loads(raw)
            except json.JSONDecodeError:
                remote_meta = None

        # Decide: do we need to download?
        if self._is_cache_fresh(bulk_type, remote_meta):
            logger.info(f"Bulk cache fresh for {bulk_type}")
            return data_file

        if remote_meta is None:
            # Can't reach metadata; fall back to any cached copy
            if data_file.exists():
                logger.warning(
                    f"Couldn't check for updates; using cached {bulk_type}"
                )
                return data_file
            return None

        # Download the new bulk file
        download_uri = remote_meta.get("download_uri")
        if not download_uri:
            logger.warning(f"Bulk metadata missing download_uri for {bulk_type}")
            return data_file if data_file.exists() else None

        logger.info(
            f"Downloading {bulk_type} "
            f"(~{(remote_meta.get('size') or 0) / 1024 / 1024:.0f}MB) "
            f"from {download_uri}"
        )
        body = http_get_text(
            download_uri, timeout=self.timeout,
            log_label="Scryfall bulk download",
        )
        if body is None:
            logger.warning(f"Bulk download failed for {bulk_type}")
            return data_file if data_file.exists() else None

        # Quick sanity check: should parse as a list of cards
        try:
            parsed = json.loads(body)
            if not isinstance(parsed, list):
                logger.warning(
                    f"Bulk {bulk_type} response wasn't a list "
                    f"(got {type(parsed).__name__}); ignoring"
                )
                return data_file if data_file.exists() else None
        except json.JSONDecodeError as e:
            logger.warning(f"Bulk {bulk_type} response wasn't JSON: {e}")
            return data_file if data_file.exists() else None

        # Write atomically: write to temp then rename
        tmp = data_file.with_suffix(data_file.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(data_file)

        self.meta_path(bulk_type).write_text(
            json.dumps(remote_meta, indent=2), encoding="utf-8",
        )
        logger.info(f"Bulk {bulk_type} saved: {len(parsed)} cards")
        return data_file

    def _is_cache_fresh(
        self,
        bulk_type: str,
        remote_meta: Optional[dict],
    ) -> bool:
        """
        Is our cached bulk file at least as recent as Scryfall's current one?
        """
        data_file = self.data_path(bulk_type)
        meta_file = self.meta_path(bulk_type)

        if not data_file.exists() or not meta_file.exists():
            return False
        if remote_meta is None:
            # Can't compare; treat as fresh-enough if any cache exists so we
            # don't re-download on every metadata failure.
            return True

        try:
            local_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        local_updated = local_meta.get("updated_at")
        remote_updated = remote_meta.get("updated_at")
        if not local_updated or not remote_updated:
            return False
        # Lexicographic compare works on ISO 8601 timestamps
        return local_updated >= remote_updated


class BulkCardSource:
    """
    Drop-in replacement for `ScryfallCardSource` backed by bulk data.

    Builds an in-memory name→card dict at construction. Lookup is O(1).
    After construction, no network calls are made.

    For large bulk files this uses ~300-500MB of RAM (all card JSON held
    in a dict). That's a one-time cost — if you're running the deck
    builder as a long-lived process or iterating on the same commander,
    the memory is paid once and every lookup is free.

    Interface matches ScryfallCardSource exactly so existing callers
    (HTML report, DeckBuilder) work without modification.
    """

    def __init__(self, cards: list[dict]):
        """
        Args:
            cards: List of card JSON dicts (from a parsed bulk file).
        """
        self._by_name: dict[str, dict] = {}
        for card in cards:
            name = card.get("name")
            if not name:
                continue
            # Primary index: full name
            self._by_name[name] = card
            # For double-faced / split cards, also index under the front
            # face name so single-name lookups work.
            card_faces = card.get("card_faces")
            if isinstance(card_faces, list) and card_faces:
                front_name = card_faces[0].get("name")
                if front_name and front_name not in self._by_name:
                    self._by_name[front_name] = card

    @classmethod
    def load_from_file(cls, path: str | Path) -> Optional["BulkCardSource"]:
        """
        Load a BulkCardSource from a bulk JSON file on disk.

        Returns None if the file doesn't exist, can't be parsed, or
        isn't a list.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            cards = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Couldn't load bulk file {path}: {e}")
            return None
        if not isinstance(cards, list):
            logger.warning(f"Bulk file {path} wasn't a list")
            return None
        logger.info(f"Loaded {len(cards)} cards from {path}")
        return cls(cards)

    @property
    def card_count(self) -> int:
        # Unique names in the index (may include duplicates for DFC front-face
        # aliases, but those point to the same dict so we count dicts)
        return len(set(id(v) for v in self._by_name.values()))

    # --- ScryfallCardSource interface ---

    def get_card_data(self, card_name: str) -> Optional[dict]:
        return self._by_name.get(card_name)

    def get_image_url(
        self,
        card_name: str,
        size: str = "small",
    ) -> Optional[str]:
        # Accept the same size values as ScryfallCardSource
        VALID_SIZES = {
            "small", "normal", "large", "png", "art_crop", "border_crop",
        }
        if size not in VALID_SIZES:
            return None

        data = self.get_card_data(card_name)
        if data is None:
            return None

        # Top-level image_uris (single-face cards)
        image_uris = data.get("image_uris")
        if image_uris and size in image_uris:
            return image_uris[size]

        # Double-faced fallback: use front face
        card_faces = data.get("card_faces")
        if isinstance(card_faces, list) and card_faces:
            front = card_faces[0]
            face_images = front.get("image_uris")
            if face_images and size in face_images:
                return face_images[size]

        return None

    def get_artist(self, card_name: str) -> Optional[str]:
        data = self.get_card_data(card_name)
        if data is None:
            return None
        artist = data.get("artist")
        if artist:
            return artist
        card_faces = data.get("card_faces")
        if isinstance(card_faces, list) and card_faces:
            return card_faces[0].get("artist")
        return None

    def get_scryfall_uri(self, card_name: str) -> Optional[str]:
        data = self.get_card_data(card_name)
        if data is None:
            return None
        return data.get("scryfall_uri")
