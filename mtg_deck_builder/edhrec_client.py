"""
EDHREC integration client.

EDHREC publishes community-sourced deck-building data. We use it to get:
1. High-synergy cards per commander (for candidate pool augmentation)
2. Synergy scores per card (commander-specific deltas over baseline)
3. Popularity data (inclusion rates, for baseline power estimates)

Design goals:
- Opt-in: no EDHREC calls unless explicitly enabled
- Precon-bias mitigation: prefer the "high-synergy" endpoint over raw
  inclusion rates, because precon cards show up in 70%+ of decks even
  when they aren't great
- Graceful degradation: if EDHREC is unavailable, deck building proceeds
  with heuristic-only scoring
- Disk caching: EDHREC data changes slowly; cache on disk with a TTL so we
  don't hammer their servers during development
- Rate limiting: respect the service, cap at ~5 requests/sec
- No hard dependency on `requests`: falls back to urllib if requests isn't installed

EDHREC's public URL layout (as of 2026, subject to change):
  https://json.edhrec.com/pages/commanders/<commander-slug>.json

The JSON includes:
  - cardlists: sections like "highsynergycards", "topcards", "utility",
    "cards_by_type", etc.
  - Each card entry has: name, sanitized_name, num_decks, potential_decks,
    synergy (commander-specific), salt (controversy score), etc.

IMPORTANT: We use `synergy` (commander-specific signal) as the primary metric,
NOT `num_decks/potential_decks` (which is inclusion rate — biased by precons).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EDHRECCardData:
    """Scraped data about a single card from EDHREC."""
    name: str
    # Commander-specific synergy (EDHREC's own metric, typically -1 to +1).
    # Positive = appears more in this commander than baseline. This is what
    # we want for commander-specific signal without precon bias.
    synergy: Optional[float] = None
    # Global popularity: fraction of decks of this commander that run this card.
    # Biased by precons.
    inclusion_rate: Optional[float] = None
    # Raw deck counts (for diagnostics)
    num_decks: Optional[int] = None
    potential_decks: Optional[int] = None
    # Which section of the EDHREC page this card appeared in
    section: str = ""

    def to_synergy_score(self) -> float:
        """
        Convert EDHREC data to a 0-100 synergy score.

        Uses `synergy` (the commander-specific delta) as the primary signal.
        Falls back to derived synergy from inclusion rate if missing.
        """
        if self.synergy is not None:
            # EDHREC synergy is typically in [-1, +1], occasionally larger.
            # Map: -1 -> 20, 0 -> 50, +1 -> 80. Clamp to [0, 100].
            return max(0.0, min(100.0, 50.0 + self.synergy * 30.0))

        if self.inclusion_rate is not None:
            # Fallback: high inclusion rate = assumed synergy (imperfect)
            return 50.0 + self.inclusion_rate * 30.0

        return 50.0  # neutral when no data

    def to_baseline_power(self) -> float:
        """
        Convert to a baseline power estimate (0-100).

        Baseline power is 'how good is this card in any deck'. EDHREC's num_decks
        across ALL commanders would be better but we only have per-commander data
        here. Use inclusion_rate * scaling as a rough proxy.
        """
        if self.inclusion_rate is None:
            return 50.0
        # Cards run in 90%+ of decks are staples (80+); 10% = niche (40).
        return max(30.0, min(95.0, 30.0 + self.inclusion_rate * 70.0))


@dataclass
class EDHRECCommanderData:
    """All EDHREC data for a single commander."""
    commander_name: str
    commander_slug: str
    cards: dict[str, EDHRECCardData] = field(default_factory=dict)
    fetched_at: float = 0.0  # Unix timestamp

    def get_synergy_score(self, card_name: str) -> Optional[float]:
        """Return 0-100 synergy score for a card, or None if not in data."""
        entry = self.cards.get(card_name)
        if entry is None:
            return None
        return entry.to_synergy_score()

    def get_high_synergy_cards(self, min_synergy: float = 0.1) -> list[EDHRECCardData]:
        """Return cards sorted by synergy (highest first)."""
        candidates = [
            c for c in self.cards.values()
            if c.synergy is not None and c.synergy >= min_synergy
        ]
        return sorted(candidates, key=lambda c: c.synergy or 0, reverse=True)


class EDHRECClient:
    """
    HTTP client for fetching EDHREC data with disk caching and rate limiting.

    Usage:
        client = EDHRECClient(cache_dir="./edhrec_cache")
        data = client.fetch_commander("Lathiel, the Bounteous Dawn")
        if data:
            synergy = data.get_synergy_score("Soul Warden")

    Usage (testing, no network):
        client = EDHRECClient(offline=True)  # Always returns None
    """

    # EDHREC's public JSON endpoint. Subject to change; wrap in try/except.
    BASE_URL = "https://json.edhrec.com/pages/commanders"

    # Cache TTL: EDHREC data updates slowly, weekly refresh is plenty
    DEFAULT_TTL_SECONDS = 7 * 24 * 3600

    # Rate limiting: don't hammer the service during development
    MIN_REQUEST_INTERVAL = 0.2  # seconds between requests

    # Cards to look for by section name — these are the sections we care about
    RELEVANT_SECTIONS = (
        "highsynergycards",
        "topcards",
        "newcards",
        "creature",
        "instant",
        "sorcery",
        "artifact",
        "enchantment",
        "planeswalker",
        "land",
        "utility",
        "gamechangers",
    )

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        ttl_seconds: Optional[int] = None,
        offline: bool = False,
        timeout: float = 10.0,
    ):
        """
        Args:
            cache_dir: Where to cache JSON responses. If None, caching is off.
            ttl_seconds: How long cached responses are valid. Default: 1 week.
            offline: If True, never make HTTP calls. Used for testing.
            timeout: Per-request timeout in seconds.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = ttl_seconds or self.DEFAULT_TTL_SECONDS
        self.offline = offline
        self.timeout = timeout
        self._last_request_time = 0.0

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_commander(
        self, commander_name: str
    ) -> Optional[EDHRECCommanderData]:
        """
        Fetch EDHREC data for a commander.

        Returns None if:
        - Offline mode
        - Network error
        - Commander not found on EDHREC
        - JSON parse failure

        Never raises — always returns None on any failure.
        """
        slug = self._slugify(commander_name)

        # Try cache first
        cached = self._read_cache(slug)
        if cached is not None:
            logger.debug(f"EDHREC cache hit for {commander_name}")
            return self._parse_data(commander_name, slug, cached)

        # Offline mode: no network attempt
        if self.offline:
            logger.debug(f"EDHREC offline mode; no data for {commander_name}")
            return None

        # Rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)

        # Fetch
        url = f"{self.BASE_URL}/{slug}.json"
        raw = self._fetch_url(url)
        self._last_request_time = time.time()

        if raw is None:
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"EDHREC returned non-JSON for {commander_name}: {e}")
            return None

        # Cache successful response
        self._write_cache(slug, data)

        return self._parse_data(commander_name, slug, data)

    COMBOS_BASE_URL = "https://json.edhrec.com/pages/combos"

    def fetch_combos(self, commander_name: str) -> list[dict]:
        """v0.9.30: the commander's combos page — human-verified combo data
        (EDHREC surfaces Commander Spellbook's database).

        Returns [{"cards": [names...], "decks": int}]. Combo membership is
        FACTUAL rules data (philosophy-safe as a detection source); the deck
        count is carried for debugging only and must never become a score.
        Empty list on any failure (offline, network, not found, parse) —
        never raises; the LLM passes remain the fallback source.
        """
        slug = self._slugify(commander_name)
        cache_key = f"combos_{slug}"
        data = self._read_cache(cache_key)
        if data is None:
            if self.offline:
                return []
            elapsed = time.time() - self._last_request_time
            if elapsed < self.MIN_REQUEST_INTERVAL:
                time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
            raw = self._fetch_url(f"{self.COMBOS_BASE_URL}/{slug}.json")
            self._last_request_time = time.time()
            if raw is None:
                return []
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"EDHREC combos non-JSON for {commander_name}: {e}")
                return []
            self._write_cache(cache_key, data)

        out: list[dict] = []
        try:
            cardlists = (data.get("container", {})
                         .get("json_dict", {})
                         .get("cardlists", []))
            for section in cardlists:
                names = [cv.get("name") for cv in section.get("cardviews", [])
                         if isinstance(cv, dict) and cv.get("name")]
                if len(names) < 2:
                    continue  # malformed section, not a combo
                m = re.search(r"\((\d[\d,]*) decks?\)",
                              section.get("header") or "")
                decks = int(m.group(1).replace(",", "")) if m else 0
                out.append({"cards": names, "decks": decks})
        except Exception as e:
            logger.warning(
                f"EDHREC combos parse failed for {commander_name}: {e}")
            return []
        logger.info(
            f"EDHREC combos: {len(out)} human-verified combo(s) for "
            f"{commander_name}"
        )
        return out

    def _fetch_url(self, url: str) -> Optional[str]:
        """Thin wrapper around shared http util."""
        from ._http import http_get_text
        return http_get_text(url, timeout=self.timeout, log_label="EDHREC")

    def _parse_data(
        self,
        commander_name: str,
        slug: str,
        raw: dict,
    ) -> Optional[EDHRECCommanderData]:
        """Parse a raw EDHREC JSON response into structured data."""
        try:
            # EDHREC JSON structure has varied over time; handle a few shapes
            container = raw.get("container") or raw
            cardlists = container.get("json_dict", {}).get("cardlists") or container.get("cardlists")

            if not cardlists:
                logger.debug(f"No cardlists in EDHREC response for {commander_name}")
                return EDHRECCommanderData(
                    commander_name=commander_name,
                    commander_slug=slug,
                    fetched_at=time.time(),
                )

            result = EDHRECCommanderData(
                commander_name=commander_name,
                commander_slug=slug,
                fetched_at=time.time(),
            )

            for section in cardlists:
                tag = (section.get("tag") or section.get("header") or "").lower()
                # Keep all sections — downstream code filters by tag if desired
                cardviews = section.get("cardviews") or section.get("cards") or []
                for cv in cardviews:
                    name = cv.get("name") or cv.get("sanitized") or ""
                    if not name:
                        continue
                    # Normalize the name (EDHREC sometimes has sanitized forms)
                    name = name.strip()
                    # Skip if we already have a better entry (first-seen wins)
                    if name in result.cards:
                        continue

                    entry = EDHRECCardData(
                        name=name,
                        synergy=_safe_float(cv.get("synergy")),
                        num_decks=_safe_int(cv.get("num_decks")),
                        potential_decks=_safe_int(cv.get("potential_decks")),
                        section=tag,
                    )
                    # Derive inclusion rate
                    if entry.num_decks and entry.potential_decks:
                        entry.inclusion_rate = entry.num_decks / entry.potential_decks
                    result.cards[name] = entry

            logger.info(
                f"EDHREC: {len(result.cards)} card entries for {commander_name}"
            )
            return result

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"EDHREC parse error for {commander_name}: {e}")
            return None

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a commander name to an EDHREC URL slug."""
        # EDHREC uses lowercased, hyphenated slugs without punctuation.
        # Example: "Lathiel, the Bounteous Dawn" -> "lathiel-the-bounteous-dawn"
        slug = name.lower()
        slug = re.sub(r"[',.!?]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"[^a-z0-9-]", "", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug

    def _cache_path(self, slug: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{slug}.json"

    def _read_cache(self, slug: str) -> Optional[dict]:
        path = self._cache_path(slug)
        if path is None or not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.ttl_seconds:
            logger.debug(f"EDHREC cache stale for {slug} (age {age:.0f}s)")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"EDHREC cache read failed for {slug}: {e}")
            return None

    def _write_cache(self, slug: str, data: dict):
        path = self._cache_path(slug)
        if path is None:
            return
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to write EDHREC cache for {slug}: {e}")


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
