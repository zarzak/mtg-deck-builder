"""
Shared utilities for HTTP fetching and cache filename safety.

Extracted in v0.5.5 cleanup pass — previously these were duplicated across
price_source.py, scryfall_cards.py, scryfall_tags.py, and edhrec_client.py
with subtle differences in:
- Filename length limits (100 vs 120 chars)
- Whether User-Agent was set on the requests path (it wasn't, in some)
- Whether Accept header was set
- Hardcoded version strings (frozen at the version each file was written)

Single source of truth fixes all five problems at once.

Design:
- `http_get_text(url, timeout, ...)` — single entry point, tries `requests`
  then falls back to `urllib`. Always sets User-Agent (using the package's
  current `__version__`) and Accept headers on both paths.
- `safe_filename(name, max_len)` — sanitize a string for use as a cache
  filename. Default 100 char limit (matches the previous more-common value).
- HTTP behavior: returns body string on 2xx, returns body string on 404
  (callers may want to inspect Scryfall's JSON error body), returns None
  on any other failure. Never raises.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Default headers sent on every Scryfall/EDHREC request.
# User-Agent reads from the package version dynamically so we never have
# stale version strings hardcoded across modules.
def _default_headers(extra_ua_suffix: str = "") -> dict:
    """Build the standard headers dict, with current package version."""
    try:
        from . import __version__ as _v
    except Exception:
        _v = "unknown"
    ua = f"mtg-deck-builder/{_v}"
    if extra_ua_suffix:
        ua = f"{ua} {extra_ua_suffix}".strip()
    return {
        "User-Agent": ua,
        "Accept": "application/json;q=0.9,*/*;q=0.8",
    }


def http_get_text(
    url: str,
    timeout: float = 10.0,
    *,
    log_label: str = "HTTP",
) -> Optional[str]:
    """
    Fetch a URL and return the response body as text.

    Returns:
    - body text on 2xx success
    - body text on 404 (Scryfall returns a JSON error body that callers
      may want to inspect, e.g. to detect "no matches" vs "not found")
    - None on any other failure (network error, non-2xx-non-404 status,
      timeout, etc.)

    Never raises. Tries `requests` first (nicer behavior), falls back to
    `urllib` from stdlib if `requests` isn't installed.

    log_label is used in debug log messages to identify which subsystem
    made the call (e.g. "Scryfall card", "EDHREC", "Scryfall tag").
    """
    headers = _default_headers()

    # Path 1: requests (preferred)
    try:
        import requests  # type: ignore
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if 200 <= r.status_code < 300:
                return r.text
            if r.status_code == 404:
                return r.text
            logger.debug(f"{log_label} HTTP {r.status_code} for {url}")
            return None
        except requests.RequestException as e:
            logger.debug(f"{log_label} requests fetch failed: {e}")
            return None
    except ImportError:
        pass  # fall through to urllib

    # Path 2: urllib fallback (stdlib, always available)
    try:
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except HTTPError as he:
            if he.code == 404:
                # Return error body so caller can inspect it
                try:
                    return he.read().decode("utf-8")
                except Exception:
                    return None
            logger.debug(f"{log_label} urllib HTTP {he.code} for {url}")
            return None
        except URLError as e:
            logger.debug(f"{log_label} urllib URL error: {e}")
            return None
    except Exception as e:
        # Last-resort catch — should be rare. Log louder so we notice.
        logger.warning(f"{log_label} unexpected fetch error: {e}")
        return None


# Pre-compiled regex for filename sanitization (faster than recompiling per call)
_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9_\-]")


def safe_filename(name: str, max_len: int = 100) -> str:
    """
    Sanitize a string for use as a cache filename.

    Replaces any character that isn't alphanumeric or `_`/`-` with `_`,
    then truncates to `max_len`. Default 100 chars is comfortably under
    every filesystem's per-component limit (typically 255 bytes on
    ext4/xfs, also 255 on NTFS), with margin for our cache-key suffixes.

    Note: this is NOT a one-way hash — different inputs can collide if
    they only differ in unsafe chars. For Scryfall card names that's
    extremely unlikely (no two cards have names that differ only in
    punctuation), but if you ever cache user-supplied data, hash it.
    """
    return _UNSAFE_CHARS.sub("_", name)[:max_len]


def url_quote(s: str) -> str:
    """URL-encode a string for use as a query parameter value."""
    from urllib.parse import quote
    return quote(s, safe="")
