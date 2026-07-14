"""
Price source abstraction and Scryfall implementation.

Price sources answer "how much does card X cost?" — used for budget
constraints. This module defines the protocol and ships one implementation
(Scryfall) but any source implementing `PriceSource` works.

Design:
- `PriceSource` is a protocol (duck type): any object with `get_price(name)`
  returning Optional[float] is a valid price source.
- Disk cache because Scryfall data is rate-limited and prices don't change
  fast; a 24h cache is plenty.
- `NullPriceSource` for tests and offline use.
- Graceful degradation: if price lookups fail, the card is allowed (treated
  as if price unknown) — better to run without full budget enforcement than
  to stall the whole build.

Scryfall JSON format (relevant bits):
    /cards/named?exact=X  returns:
        { "prices": {
            "usd": "1.23",           // null if no paper data
            "usd_foil": "5.00",
            "usd_etched": null,
            "eur": "1.10",
            "tix": "0.05"             // MTGO tickets
          },
          ...
        }

We treat null/missing as "unknown" (returns None), not 0.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class PriceSource(Protocol):
    """Protocol for any price data source."""

    def get_price(self, card_name: str) -> Optional[float]:
        """Return USD price or None if unknown/unavailable."""
        ...


class NullPriceSource:
    """Price source that always returns None. Used when no source configured."""

    def get_price(self, card_name: str) -> Optional[float]:
        return None


@dataclass
class PriceCacheEntry:
    price: Optional[float]
    fetched_at: float


class ScryfallPriceSource:
    """
    Scryfall-based price source.

    Uses the public /cards/named endpoint with USD prices.
    - 24-hour disk cache (prices don't change often)
    - 100ms minimum request interval (Scryfall asks for <10req/s)
    - Graceful degradation: on errors, returns None

    Note: Not a hard dependency on `requests`. Falls back to stdlib urllib.
    """

    BASE_URL = "https://api.scryfall.com/cards/named"
    DEFAULT_TTL_SECONDS = 24 * 3600
    MIN_REQUEST_INTERVAL = 0.1
    DEFAULT_TIMEOUT = 8.0

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        ttl_seconds: Optional[int] = None,
        offline: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Args:
            cache_dir: Disk cache directory. If None, uses in-memory only.
            ttl_seconds: Cache entry TTL (default 24h).
            offline: If True, never make HTTP calls (testing).
            timeout: Request timeout seconds.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = ttl_seconds or self.DEFAULT_TTL_SECONDS
        self.offline = offline
        self.timeout = timeout
        self._last_request_time = 0.0
        self._memory_cache: dict[str, PriceCacheEntry] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_price(self, card_name: str) -> Optional[float]:
        """Return USD price for card, or None if unknown."""
        # Check memory cache first
        entry = self._memory_cache.get(card_name)
        if entry is not None and self._is_fresh(entry):
            return entry.price

        # Check disk cache
        disk_entry = self._read_disk_cache(card_name)
        if disk_entry is not None and self._is_fresh(disk_entry):
            self._memory_cache[card_name] = disk_entry
            return disk_entry.price

        # Offline mode: no fetch
        if self.offline:
            return None

        # Fetch from Scryfall
        price = self._fetch_price(card_name)
        entry = PriceCacheEntry(price=price, fetched_at=time.time())
        self._memory_cache[card_name] = entry
        self._write_disk_cache(card_name, entry)
        return price

    def _is_fresh(self, entry: PriceCacheEntry) -> bool:
        return (time.time() - entry.fetched_at) < self.ttl_seconds

    def _fetch_price(self, card_name: str) -> Optional[float]:
        """Fetch the price from Scryfall, returning None on any failure."""
        # Rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)

        url = f"{self.BASE_URL}?exact={_url_quote(card_name)}"
        raw = self._http_get(url)
        self._last_request_time = time.time()

        if raw is None:
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        prices = data.get("prices") or {}
        usd = prices.get("usd")
        if usd is None:
            return None
        try:
            return float(usd)
        except (TypeError, ValueError):
            return None

    def _http_get(self, url: str) -> Optional[str]:
        """Thin wrapper around shared http util."""
        from ._http import http_get_text
        return http_get_text(url, timeout=self.timeout, log_label="Scryfall price")

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _disk_path(self, card_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = _safe_filename(card_name)
        return self.cache_dir / f"{safe}.json"

    def _read_disk_cache(self, card_name: str) -> Optional[PriceCacheEntry]:
        path = self._disk_path(card_name)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PriceCacheEntry(
                price=data.get("price"),
                fetched_at=float(data.get("fetched_at", 0)),
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.debug(f"Disk cache read failed for {card_name}: {e}")
            return None

    def _write_disk_cache(self, card_name: str, entry: PriceCacheEntry):
        path = self._disk_path(card_name)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps({"price": entry.price, "fetched_at": entry.fetched_at}),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug(f"Disk cache write failed for {card_name}: {e}")


class StaticPriceSource:
    """Price source backed by a user-provided dict. Good for tests."""

    def __init__(self, prices: dict[str, float]):
        self._prices = dict(prices)

    def get_price(self, card_name: str) -> Optional[float]:
        return self._prices.get(card_name)


# ----------------------------------------------------------------------
# Budget filter: apply a PriceSource to a candidate pool
# ----------------------------------------------------------------------

def filter_cards_by_budget(
    cards: list,
    price_source: PriceSource,
    max_price_per_card: Optional[float],
    exclude_unknown: bool = False,
) -> list:
    """
    Return cards that fit a per-card budget.

    Args:
        cards: list of Card objects
        price_source: any PriceSource implementation
        max_price_per_card: ceiling in USD; None means no filter
        exclude_unknown: if True, cards with no price data are dropped.
            if False (default), we keep them (better to allow than stall).

    Returns:
        Filtered list (same objects, not copies)
    """
    if max_price_per_card is None:
        return cards

    out = []
    for c in cards:
        price = price_source.get_price(c.name)
        if price is None:
            if not exclude_unknown:
                out.append(c)
            continue
        if price <= max_price_per_card:
            out.append(c)
    return out


def deck_total_price(deck_cards, price_source: PriceSource) -> float:
    """Sum prices of all cards in a deck, treating unknown as 0."""
    total = 0.0
    for c in deck_cards:
        p = price_source.get_price(c.name)
        if p is not None:
            total += p
    return total


# ----------------------------------------------------------------------
# Helpers — thin shims over mtg_deck_builder._http for back-compat
# ----------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    from ._http import safe_filename
    return safe_filename(name)


def _url_quote(s: str) -> str:
    from ._http import url_quote
    return url_quote(s)
