"""
MTGJSON card-database refresh (v0.9.18).

A faithful Python port of the user's mtg-deck-extract.js, plus:
  - downloads AtomicCards.json (or .json.gz) straight from MTGJSON;
  - extracts the SAME pipe-delimited columns the .js produced, so the output
    is a drop-in replacement for the existing cards.csv;
  - ADDS an `isGameChanger` column (the .js didn't), which makes the Game
    Changer list self-refreshing from the source data instead of a hardcoded
    constant.

Transformation parity with the .js (verified against its logic):
  - one row per card, taking cardVersions[0];
  - layouts token / emblem / art_series are skipped;
  - array fields (colorIdentity, colors, types, subtypes, supertypes,
    keywords) are comma-joined;
  - legalities → comma-joined lowercased format names where status == "Legal";
  - CSV escaping, in this exact order: backslash → "\\\\", "|" → "\\|",
    newline → "\\n".

The .js's >30MB "chunking for Claude context" step is intentionally dropped —
we always want a single cards.csv, and our loader reads one file.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# MTGJSON v5 direct download (no auth). The .gz is ~30MB vs ~140MB plain.
DEFAULT_SOURCE_URL = "https://mtgjson.com/api/v5/AtomicCards.json.gz"

# Same 17 fields, same order, as the .js — so the output is a drop-in
# replacement. `isGameChanger` is appended as the 18th column (new).
_FIELDS = [
    "name", "manaCost", "manaValue", "type", "text", "colorIdentity",
    "colors", "power", "toughness", "loyalty", "defense", "types",
    "subtypes", "supertypes", "keywords", "layout", "legalities",
]
_GC_FIELD = "isGameChanger"

_ARRAY_FIELDS = frozenset({
    "colorIdentity", "colors", "types", "subtypes", "supertypes", "keywords",
})
_SKIP_LAYOUTS = frozenset({"token", "emblem", "art_series"})


def _escape(value) -> str:
    """CSV cell escaping, byte-for-byte matching the .js escapeCsvValue."""
    if value is None:
        return ""
    s = str(value)
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")


def _num(value) -> str:
    """Stringify a numeric field like JS String() — drop a trailing .0 so
    manaValue reads "2" not "2.0" (matches the existing cards.csv)."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _legalities(legalities: Optional[dict]) -> str:
    if not legalities:
        return ""
    return ",".join(
        fmt.lower() for fmt, status in legalities.items() if status == "Legal"
    )


def extract_rows(atomic_data: dict) -> tuple[list[list[str]], int]:
    """Port of processCards(). `atomic_data` is the MTGJSON `data` dict
    (cardName -> list of printings). Returns (rows, count) where each row is
    the ordered cell list including the trailing isGameChanger cell."""
    rows: list[list[str]] = []
    for name, versions in atomic_data.items():
        if not versions:
            continue
        card = versions[0]
        if card.get("layout") in _SKIP_LAYOUTS:
            continue
        row: list[str] = []
        for field in _FIELDS:
            # The AtomicCards dict KEY is the canonical name (always present
            # and identical to the printing's name); use it directly.
            val = name if field == "name" else card.get(field)
            if field in _ARRAY_FIELDS:
                row.append(",".join(val) if val else "")
            elif field == "legalities":
                row.append(_legalities(val))
            elif field == "manaValue":
                row.append(_num(val))
            else:
                row.append(_escape(val))
        # New column: MTGJSON exposes isGameChanger as a bool on the printing.
        row.append("true" if card.get(_GC_FIELD) else "false")
        rows.append(row)
    return rows, len(rows)


def _header(total: int) -> str:
    from datetime import date
    cols = _FIELDS + [_GC_FIELD]
    lines = [
        "# MTG COMPREHENSIVE CARD DATABASE - OPTIMIZED FOR THE DECK BUILDER",
        "# Format: CSV with vertical bar (|) delimiters",
        f"# Date extracted: {date.today().isoformat()}",
        f"# Total cards: {total}",
        "#",
        "# FIELD DESCRIPTIONS:",
        "# name - Card name",
        "# manaCost - Mana cost with symbols in {}",
        "# manaValue - Converted mana cost (numeric)",
        "# type - Full type line",
        "# text - Full rules text (\\n = newline, \\| = pipe, \\\\ = backslash)",
        "# colorIdentity / colors - comma-separated WUBRG",
        "# power / toughness / loyalty / defense - stats",
        "# types / subtypes / supertypes / keywords - comma-separated",
        "# layout - normal / split / modal_dfc / ...",
        "# legalities - comma-separated formats where legal",
        "# isGameChanger - true/false, official Commander bracket Game Changer",
        "#",
    ]
    return "\n".join(lines) + "\n" + "|".join(cols) + "\n"


def write_csv(atomic_data: dict, output_path: str | Path) -> int:
    """Extract and write the pipe-delimited CSV. Returns the card count."""
    rows, total = extract_rows(atomic_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write(_header(total))
        for row in rows:
            f.write("|".join(row))
            f.write("\n")
    logger.info(f"Wrote {total} cards to {output_path}")
    return total


def download_atomic(
    url: str = DEFAULT_SOURCE_URL,
    timeout: float = 120.0,
) -> dict:
    """Download + parse AtomicCards.json[.gz] from MTGJSON. Returns the
    top-level object (with 'meta' and 'data'). Raises on network/parse error
    (the CLI wraps this)."""
    import urllib.request
    logger.info(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "mtg-deck-builder"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if url.endswith(".gz"):
        raw = gzip.decompress(raw)
    logger.info(f"Downloaded {len(raw) / 1024 / 1024:.0f}MB (decompressed); parsing...")
    return json.loads(raw)


def load_atomic_file(path: str | Path) -> dict:
    """Load a local AtomicCards.json or .json.gz (skips the download)."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data)


def refresh(
    output_path: str | Path,
    source_url: str = DEFAULT_SOURCE_URL,
    atomic_json_path: Optional[str] = None,
    timeout: float = 120.0,
) -> int:
    """Full refresh: (download or load AtomicCards) -> extract -> write CSV.
    Returns the card count. Raises on failure (the CLI reports it)."""
    if atomic_json_path:
        logger.info(f"Using local AtomicCards file: {atomic_json_path}")
        obj = load_atomic_file(atomic_json_path)
    else:
        obj = download_atomic(source_url, timeout=timeout)
    data = obj.get("data")
    if not isinstance(data, dict):
        raise ValueError("AtomicCards JSON has no 'data' object")
    return write_csv(data, output_path)
