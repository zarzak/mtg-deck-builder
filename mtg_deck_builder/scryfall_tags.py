"""
Scryfall Tag Client — queries for cards matching art/oracle tags.

Scryfall imports community-sourced tags from their Tagger project daily. Two
types exist:

- Art tags (`art:`, `atag:`, `arttag:`): what's depicted in the artwork.
  Example: `art:mammoth` finds every card with a mammoth drawn on it, even
  if the card is an instant or land.
- Oracle tags (`function:`, `otag:`, `oracletag:`): what the card DOES.
  Example: `function:removal` finds cards that remove threats, beyond our
  regex role matching.

Key design decisions:
- Tag queries return LISTS of cards, so the caching model is different from
  `ScryfallCardSource` (which is per-card). We cache each (tag, operator)
  pair as a list of card names.
- Results are just card *names* — we don't need full JSON here. If callers
  need images for these cards, they go through `ScryfallCardSource`.
- Rate limiting + disk cache + offline mode identical to other Scryfall
  modules.
- Gracefully returns empty list on any failure.

Scryfall pagination: the search endpoint returns up to 175 results per page.
We follow `next_page` links up to a configurable maximum (default 3 pages =
up to ~525 cards per tag — plenty for most tags).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Valid tag types (maps to Scryfall search operator)
TAG_OPERATORS = {
    "art": "art",        # art:mammoth
    "oracle": "otag",    # otag:removal
}


@dataclass
class TagCacheEntry:
    """Cached result of a tag query."""
    tag: str
    kind: str            # 'art' or 'oracle'
    card_names: list[str]
    fetched_at: float


class ScryfallTagClient:
    """
    Fetches cards matching Scryfall art or oracle tags.

    Usage:
        client = ScryfallTagClient(cache_dir="./tag_cache")

        # Art tag — cards depicting mammoths
        names = client.get_cards_with_art_tag("mammoth")

        # Oracle tag — cards that function as removal
        names = client.get_cards_with_oracle_tag("removal")

        # Color-identity filtered
        names = client.get_cards_with_oracle_tag("ramp", color_identity="WG")

    Returns [] on any failure — never raises.
    """

    BASE_URL = "https://api.scryfall.com/cards/search"
    DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days (tags update occasionally)
    MIN_REQUEST_INTERVAL = 0.1
    DEFAULT_TIMEOUT = 10.0
    MAX_PAGES = 3  # cap to avoid runaway pagination

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        ttl_seconds: Optional[int] = None,
        offline: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        max_pages: int = MAX_PAGES,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = ttl_seconds or self.DEFAULT_TTL_SECONDS
        self.offline = offline
        self.timeout = timeout
        self.max_pages = max_pages
        self._last_request_time = 0.0
        self._memory_cache: dict[str, TagCacheEntry] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cards_with_art_tag(
        self,
        tag: str,
        color_identity: Optional[str] = None,
    ) -> list[str]:
        """
        Return the names of cards whose artwork has this tag.

        tag: a Scryfall art tag slug (e.g. "mammoth", "forest", "fire").
            See https://scryfall.com/docs/tagger-tags for the full list.
        color_identity: optional filter, e.g. "WG" for Lathiel-legal cards.
            Uses Scryfall's `id<=` operator for color-identity subset match.
        """
        return self._query_tag("art", tag, color_identity)

    def get_cards_with_oracle_tag(
        self,
        tag: str,
        color_identity: Optional[str] = None,
    ) -> list[str]:
        """
        Return the names of cards whose oracle (functional) tag matches.

        tag: a Scryfall oracle tag slug (e.g. "removal", "ramp",
            "counterspell", "card-draw"). Community-curated via the Tagger.
        """
        return self._query_tag("oracle", tag, color_identity)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _query_tag(
        self,
        kind: str,
        tag: str,
        color_identity: Optional[str],
    ) -> list[str]:
        """
        Execute a tag query. Returns list of card names, [] on failure.
        """
        if kind not in TAG_OPERATORS:
            logger.warning(f"Unknown tag kind {kind!r}")
            return []

        cache_key = self._cache_key(kind, tag, color_identity)

        # Memory cache
        entry = self._memory_cache.get(cache_key)
        if entry is not None and self._is_fresh(entry):
            return list(entry.card_names)

        # Disk cache
        disk_entry = self._read_disk_cache(cache_key)
        if disk_entry is not None and self._is_fresh(disk_entry):
            self._memory_cache[cache_key] = disk_entry
            return list(disk_entry.card_names)

        if self.offline:
            return []

        # Build query string
        operator = TAG_OPERATORS[kind]
        query_parts = [f"{operator}:{tag}"]
        if color_identity:
            # Normalize: accept either "WG", "W,G", "w g", etc.
            # Scryfall's id<= operator wants concatenated letters like "wg".
            normalized = _normalize_color_identity(color_identity)
            if normalized:
                query_parts.append(f"id<={normalized}")
        query = " ".join(query_parts)

        names = self._fetch_paginated(query)

        entry = TagCacheEntry(
            tag=tag, kind=kind,
            card_names=names,
            fetched_at=time.time(),
        )
        self._memory_cache[cache_key] = entry
        self._write_disk_cache(cache_key, entry)

        return list(names)

    def _fetch_paginated(self, query: str) -> list[str]:
        """Fetch all pages of a search query, up to max_pages."""
        from ._http import url_quote
        url = f"{self.BASE_URL}?q={url_quote(query)}"
        all_names: list[str] = []

        for page in range(self.max_pages):
            self._rate_limit()
            raw = self._http_get(url)
            if raw is None:
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                break

            # Scryfall 404 returns object_type 'error' — no matches
            if data.get("object") == "error":
                logger.debug(f"No matches for query {query!r}")
                return []

            cards = data.get("data") or []
            for card in cards:
                name = card.get("name")
                if name:
                    all_names.append(name)

            if not data.get("has_more"):
                break
            next_url = data.get("next_page")
            if not next_url:
                break
            url = next_url

        return all_names

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _http_get(self, url: str) -> Optional[str]:
        """Thin wrapper around shared http util (kept as a method so tests
        can monkey-patch it on instances)."""
        from ._http import http_get_text
        return http_get_text(url, timeout=self.timeout, log_label="Scryfall tag")

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _is_fresh(self, entry: TagCacheEntry) -> bool:
        return (time.time() - entry.fetched_at) < self.ttl_seconds

    @staticmethod
    def _cache_key(kind: str, tag: str, color_identity: Optional[str]) -> str:
        # Normalize color identity to a canonical sorted lowercase form
        # so "WG", "W,G", "gw", "G,W" all hit the same cache entry.
        if color_identity:
            normalized = _normalize_color_identity(color_identity)
            ci = normalized or "any"
        else:
            ci = "any"
        return f"{kind}__{tag.lower()}__{ci}"

    def _disk_path(self, cache_key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = _safe_filename(cache_key)
        return self.cache_dir / f"{safe}.json"

    def _read_disk_cache(self, cache_key: str) -> Optional[TagCacheEntry]:
        path = self._disk_path(cache_key)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TagCacheEntry(
                tag=data["tag"],
                kind=data["kind"],
                card_names=list(data.get("card_names", [])),
                fetched_at=float(data.get("fetched_at", 0)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.debug(f"Tag cache read failed for {cache_key}: {e}")
            return None

    def _write_disk_cache(self, cache_key: str, entry: TagCacheEntry):
        path = self._disk_path(cache_key)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps({
                    "tag": entry.tag,
                    "kind": entry.kind,
                    "card_names": entry.card_names,
                    "fetched_at": entry.fetched_at,
                }),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug(f"Tag cache write failed: {e}")


def _safe_filename(s: str) -> str:
    # Kept as a thin shim so existing test imports keep working.
    # New code should call mtg_deck_builder._http.safe_filename directly.
    from ._http import safe_filename
    return safe_filename(s, max_len=120)


def _normalize_color_identity(s: Optional[str]) -> str:
    """
    Normalize a color identity string to Scryfall's `id<=` format.

    Accepts:
      "WG", "wg", "W,G", "w g", "g w", "G,W", " W, G "

    Returns a sorted lowercase string of WUBRG letters:
      "WG" -> "gw"      (sorted alphabetically)
      "W,G" -> "gw"
      "WUBRG" -> "bgruw"
      "" or None -> ""

    Sorting gives us a canonical form so different input representations
    map to the same cache key AND the same Scryfall query. Note: Scryfall
    accepts any order in `id<=`, so "gw" and "wg" produce identical
    results — we pick sorted just for canonicalization.
    """
    if not s:
        return ""
    # Extract only WUBRG letters, ignore commas/spaces/case
    letters = [c.lower() for c in s if c.upper() in "WUBRG"]
    if not letters:
        return ""
    return "".join(sorted(set(letters)))
