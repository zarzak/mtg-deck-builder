"""
Scryfall card source — fetches and caches full card JSON.

Distinct from `ScryfallPriceSource` (which only caches prices) because:
1. Full card JSON is ~2KB per card, not worth duplicating in the price cache
2. Image URLs are the primary motivation here (for HTML reports)
3. Keeping the two separate means users who only need prices don't pay the
   cache-bloat cost

Usage:
    source = ScryfallCardSource(cache_dir="./scryfall_cache")
    img_url = source.get_image_url("Sol Ring", size="small")
    art_crop = source.get_image_url("Sol Ring", size="art_crop")
    full_data = source.get_card_data("Sol Ring")  # raw dict if you need other fields

Available image sizes (per Scryfall docs):
    small       - 146 x 204 PNG (~10KB, fast to load)
    normal      - 488 x 680 JPG (standard card image)
    large       - 672 x 936 JPG (high-res card)
    png         - 745 x 1040 PNG (highest quality)
    art_crop    - just the illustration, no frame
    border_crop - slightly cropped card (removes outer border)

For HTML reports, `small` is usually right (fast thumbnails); `art_crop` is
great when you want a mood-board feel without the card frame taking up space.

Scryfall usage guidelines (paraphrased from their docs):
- User-Agent header required (we set one)
- Rate limit ~10 req/s (we enforce 100ms min interval)
- Don't distort or alter images
- Keep artist credit visible if showing art_crop
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Valid image sizes per Scryfall /cards image_uris
VALID_IMAGE_SIZES = frozenset({
    "small", "normal", "large", "png", "art_crop", "border_crop",
})


@dataclass
class CardCacheEntry:
    """A cached Scryfall card JSON response."""
    data: Optional[dict]  # the full JSON response; None if the card wasn't found
    fetched_at: float


class ScryfallCardSource:
    """
    Fetches and caches full Scryfall card JSON.

    Focus is on providing image URLs for the HTML report, but the raw data
    is also available (card_data) if you want other fields like artist,
    set, rarity, etc.

    Returns None from get_image_url and get_card_data on any error
    (network failure, card not found, etc.) — never raises.
    """

    BASE_URL = "https://api.scryfall.com/cards/named"
    DEFAULT_TTL_SECONDS = 30 * 24 * 3600  # 30 days — images don't change
    MIN_REQUEST_INTERVAL = 0.1  # Scryfall asks for <10 req/s
    DEFAULT_TIMEOUT = 8.0

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        ttl_seconds: Optional[int] = None,
        offline: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = ttl_seconds or self.DEFAULT_TTL_SECONDS
        self.offline = offline
        self.timeout = timeout
        self._last_request_time = 0.0
        self._memory_cache: dict[str, CardCacheEntry] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_card_data(self, card_name: str) -> Optional[dict]:
        """
        Return the full Scryfall card JSON for this card, or None on failure.

        Cached in memory and on disk. The cache is used first; a miss
        triggers a network fetch unless we're in offline mode.
        """
        # Memory cache
        entry = self._memory_cache.get(card_name)
        if entry is not None and self._is_fresh(entry):
            return entry.data

        # Disk cache
        disk_entry = self._read_disk_cache(card_name)
        if disk_entry is not None and self._is_fresh(disk_entry):
            self._memory_cache[card_name] = disk_entry
            return disk_entry.data

        # Offline: can't fetch
        if self.offline:
            return None

        data = self._fetch_card(card_name)
        entry = CardCacheEntry(data=data, fetched_at=time.time())
        self._memory_cache[card_name] = entry
        self._write_disk_cache(card_name, entry)
        return data

    def get_image_url(
        self,
        card_name: str,
        size: str = "small",
    ) -> Optional[str]:
        """
        Return the URL for the requested image size, or None if unavailable.

        Handles both single-face cards (image_uris at the top level) and
        double-faced cards (image_uris on each card_faces entry; we use the
        front face).
        """
        if size not in VALID_IMAGE_SIZES:
            logger.warning(f"Unknown image size {size!r}; valid: {sorted(VALID_IMAGE_SIZES)}")
            return None

        data = self.get_card_data(card_name)
        if data is None:
            return None

        # Single-face case
        image_uris = data.get("image_uris")
        if image_uris and size in image_uris:
            return image_uris[size]

        # Double-faced / transform cards: try the first face
        card_faces = data.get("card_faces")
        if isinstance(card_faces, list) and card_faces:
            front = card_faces[0]
            face_images = front.get("image_uris")
            if face_images and size in face_images:
                return face_images[size]

        # No image available at that size for this card
        return None

    def get_artist(self, card_name: str) -> Optional[str]:
        """Return the artist name for the card, or None if unknown."""
        data = self.get_card_data(card_name)
        if data is None:
            return None
        artist = data.get("artist")
        if artist:
            return artist
        # Double-faced fallback
        card_faces = data.get("card_faces")
        if isinstance(card_faces, list) and card_faces:
            return card_faces[0].get("artist")
        return None

    def get_scryfall_uri(self, card_name: str) -> Optional[str]:
        """Return the Scryfall page URL for this card (nice for linking)."""
        data = self.get_card_data(card_name)
        if data is None:
            return None
        return data.get("scryfall_uri")

    # ------------------------------------------------------------------
    # Internal: fetch + cache
    # ------------------------------------------------------------------

    def _is_fresh(self, entry: CardCacheEntry) -> bool:
        return (time.time() - entry.fetched_at) < self.ttl_seconds

    def _fetch_card(self, card_name: str) -> Optional[dict]:
        """Fetch card JSON from Scryfall, returning None on any failure."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)

        url = f"{self.BASE_URL}?exact={_url_quote(card_name)}"
        raw = self._http_get(url)
        self._last_request_time = time.time()

        if raw is None:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _http_get(self, url: str) -> Optional[str]:
        """Thin wrapper around shared http util (kept as a method so tests
        can monkey-patch it on instances)."""
        from ._http import http_get_text
        return http_get_text(url, timeout=self.timeout, log_label="Scryfall card")

    # ------------------------------------------------------------------
    # Disk cache (full JSON, not just images)
    # ------------------------------------------------------------------

    def _disk_path(self, card_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{_safe_filename(card_name)}.json"

    def _read_disk_cache(self, card_name: str) -> Optional[CardCacheEntry]:
        path = self._disk_path(card_name)
        if path is None or not path.exists():
            return None
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            return CardCacheEntry(
                data=wrapper.get("data"),
                fetched_at=float(wrapper.get("fetched_at", 0)),
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.debug(f"Card cache read failed for {card_name}: {e}")
            return None

    def _write_disk_cache(self, card_name: str, entry: CardCacheEntry):
        path = self._disk_path(card_name)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps({"data": entry.data, "fetched_at": entry.fetched_at}),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug(f"Card cache write failed for {card_name}: {e}")


# ----------------------------------------------------------------------
# Helpers — now thin shims over mtg_deck_builder._http for backwards
# compatibility with any tests/users that imported these directly.
# ----------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    from ._http import safe_filename
    return safe_filename(name)


def _url_quote(s: str) -> str:
    from ._http import url_quote
    return url_quote(s)
