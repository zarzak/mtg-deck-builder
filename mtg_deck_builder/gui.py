"""
Local web GUI for the deck builder (v0.9.20).

Design:
  - stdlib only (http.server), bound to 127.0.0.1 — no new dependencies,
    nothing exposed off-machine.
  - Every action shells out to the SAME CLI used manually (subprocess of
    `python -m mtg_deck_builder.cli`), so a GUI build is byte-identical to a
    terminal build and the log/report/output artifacts land in the same
    places.
  - ONE job at a time: builds cost real API credits, so a second submission
    while one runs is rejected instead of queued.
  - Simple mode = the usual defaults (the flags the user always runs with);
    the Advanced panel exposes each knob across its allowed range plus an
    escape-hatch free-text field for anything new.

Launch: `python -m mtg_deck_builder.cli --csv cards.csv gui`
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PAGE_PATH = Path(__file__).parent / "gui_page.html"

# v0.9.21: GUI settings persisted next to the project outputs. The UI only
# ever gets back a masked tail of the key.
SETTINGS_FILE = Path("gui_settings.json")


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# v0.9.22: API-key encryption at rest.
# On Windows the key is wrapped with DPAPI (CryptProtectData) — decryptable
# only by the same Windows user on the same machine, no password to manage.
# Elsewhere it falls back to plaintext with an honest scheme marker (the
# threat model is "don't leave a greppable sk-ant-... on disk"; DPAPI
# covers the actual platform this project runs on).
# ----------------------------------------------------------------------

def _dpapi(data: bytes, protect: bool) -> Optional[bytes]:
    """CryptProtectData / CryptUnprotectData via ctypes. None on failure."""
    if sys.platform != "win32":
        return None
    import ctypes
    import ctypes.wintypes as wt

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = DATA_BLOB(len(data),
                        ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    fn = (ctypes.windll.crypt32.CryptProtectData if protect
          else ctypes.windll.crypt32.CryptUnprotectData)
    ok = fn(ctypes.byref(blob_in), None, None, None, None, 0,
            ctypes.byref(blob_out))
    if not ok:
        return None
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def set_api_key(settings: dict, key: str) -> None:
    """Store (or clear, when key is empty) the API key IN PLACE, encrypted
    with DPAPI when available."""
    import base64
    settings.pop("api_key", None)  # legacy plaintext field
    settings.pop("api_key_enc", None)
    settings.pop("api_key_scheme", None)
    key = (key or "").strip()
    if not key:
        return
    blob = _dpapi(key.encode("utf-8"), protect=True)
    if blob is not None:
        settings["api_key_enc"] = base64.b64encode(blob).decode("ascii")
        settings["api_key_scheme"] = "dpapi"
    else:
        settings["api_key_enc"] = base64.b64encode(
            key.encode("utf-8")).decode("ascii")
        settings["api_key_scheme"] = "plain"


def get_api_key(settings: dict) -> str:
    """Resolve the stored key ('' if none). Reads the encrypted field and,
    for pre-v0.9.22 files, the legacy plaintext one."""
    import base64
    enc = settings.get("api_key_enc")
    if enc:
        try:
            raw = base64.b64decode(enc)
            if settings.get("api_key_scheme") == "dpapi":
                out = _dpapi(raw, protect=False)
                return out.decode("utf-8") if out else ""
            return raw.decode("utf-8")
        except Exception:
            return ""
    return (settings.get("api_key") or "").strip()  # legacy


def migrate_legacy_key() -> None:
    """One-shot at server start: if a pre-v0.9.22 plaintext key exists,
    rewrite it encrypted."""
    settings = load_settings()
    legacy = (settings.get("api_key") or "").strip()
    if legacy:
        set_api_key(settings, legacy)
        save_settings(settings)
        logger.info("GUI: migrated stored API key to encrypted form")


def api_key_status(settings: dict) -> dict:
    """Where the key will come from, with a masked tail for display.
    A key saved in the GUI wins over the environment (explicit user action
    beats inherited shell state)."""
    saved = get_api_key(settings)
    env = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if saved:
        scheme = settings.get("api_key_scheme",
                              "plain" if settings.get("api_key") else "dpapi")
        return {"source": "saved", "masked": f"…{saved[-4:]}",
                "encrypted": scheme == "dpapi"}
    if env:
        return {"source": "environment", "masked": f"…{env[-4:]}",
                "encrypted": None}
    return {"source": "missing", "masked": "", "encrypted": None}

# The GUI's "usual defaults" — the simple-mode build command. These mirror
# the user's standard invocation (NOT always the CLI defaults: e.g. the CLI
# defaults combos/card-power to off, the GUI turns them on because that's
# how real builds are run).
BUILD_DEFAULTS = {
    "bracket": 4,
    "seed": 42,
    "generations": 300,
    "population": 100,
    "refine": 3,
    "refine_max_swaps": 8,
    "recall_edhrec": True,
    "recall_embeddings": True,
    "recall_patterns": True,
    "recall_edhrec_limit": 300,
    "recall_embedding_limit": 1500,
    "recall_pool_cap": 2500,
    "synergy_scoring_mode": "llm",       # choices: auto | llm | embedding
    "tournament_model": "claude-haiku-4-5",
    "synergy_engine_target": 25,
    "synergy_engine_shortlist": 300,
    "synergy_engine_bypass": 12,
    "card_power_mode": "llm",            # choices: off | llm
    "card_power_model": "claude-sonnet-4-6",
    "power_staples": 60,
    "role_power_bypass": 15,
    "consistency_weight": 0.12,
    "quality_roles": True,
    "combos": "llm",                     # choices: off | llm
    "combo_model": "claude-sonnet-4-6",
    "combo_weight": 0.12,
    "signature_pass": True,
    "synergy_cache": True,
    "analysis_cache": True,
    "engine_boost": "power",             # choices: off | floor | power
    "engine_floor": 80.0,
    "edhrec_floor": 0.75,
    "structural_synergy": "on",          # choices: on | off
    "structural_floor": 95.0,
    "model": "claude-sonnet-4-6",
    "budget": None,
    "budget_exclude_unknown": False,
    "locks": "",
    "bans": "",
    "game_changers_file": "",
    "extra_flags": "",
}


def slugify(name: str) -> str:
    """Commander name -> filesystem-safe output prefix ("Jodah, the Unifier"
    -> "jodah_the_unifier")."""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s or "deck"


# ----------------------------------------------------------------------
# v0.9.23: on-disk layout. GUI builds write into decks/<slug>/<timestamp>_*
# so successive builds of the same commander never overwrite each other and
# artifacts group naturally. Legacy root-level files (<slug>_deck.txt from
# CLI runs / earlier versions) stay readable — the scanner picks up both.
# ----------------------------------------------------------------------
DECKS_DIR = Path("decks")


def build_prefix(commander: str, now: Optional[time.struct_time] = None) -> str:
    """Relative output prefix for a new GUI build:
    decks/<slug>/<YYYY-MM-DD_HHMMSS>."""
    ts = time.strftime("%Y-%m-%d_%H%M%S", now or time.localtime())
    return f"{DECKS_DIR.as_posix()}/{slugify(commander)}/{ts}"


def safe_relpath(raw: str) -> Optional[Path]:
    """Resolve a client-supplied relative path and confirm it stays inside
    the working directory. Returns the resolved Path or None.

    A Windows drive-letter path (e.g. "C:/...") is rejected on EVERY OS: a
    legitimate file request is always relative, and on POSIX such a path
    would otherwise be treated as a literal "C:" subfolder inside the
    sandbox rather than flagged — so rejecting it keeps behavior identical
    across platforms.
    """
    raw = urllib.parse.unquote(raw or "").replace("\\", "/")
    if re.match(r"[A-Za-z]:", raw):
        return None
    raw = raw.lstrip("/")
    if not raw:
        return None
    p = (Path(".") / raw).resolve()
    try:
        p.relative_to(Path(".").resolve())
    except ValueError:
        return None
    return p


def build_argv(payload: dict, csv_path: str,
               out_prefix: Optional[str] = None) -> list[str]:
    """Payload from the GUI form -> full CLI argv. Pure function (tested).

    Only emits flags that differ from the CLI's own defaults where the CLI
    default matches the GUI default; always emits the ones the GUI considers
    part of the usual run so the command is explicit and reproducible from
    the log line.

    out_prefix (e.g. "decks/jodah_the_unifier/2026-07-10_141530") anchors
    the three artifacts; default is the legacy "<slug>" in the cwd.
    """
    commander = (payload.get("commander") or "").strip()
    if not commander:
        raise ValueError("commander is required")
    p = {**BUILD_DEFAULTS, **{k: v for k, v in payload.items() if v is not None}}
    prefix = out_prefix or slugify(commander)

    argv = [sys.executable, "-m", "mtg_deck_builder.cli",
            "--csv", csv_path,
            "--log-file", f"{prefix}_build.log",
            "build", commander,
            "--bracket", str(int(p["bracket"])),
            "--generations", str(int(p["generations"])),
            "--population", str(int(p["population"])),
            "--refine", str(int(p["refine"])),
            "--refine-max-swaps", str(int(p["refine_max_swaps"])),
            "--synergy-scoring-mode", str(p["synergy_scoring_mode"]),
            "--tournament-model", str(p["tournament_model"]),
            "--synergy-engine-target", str(int(p["synergy_engine_target"])),
            "--synergy-engine-shortlist", str(int(p["synergy_engine_shortlist"])),
            "--synergy-engine-bypass", str(int(p["synergy_engine_bypass"])),
            "--card-power-mode", str(p["card_power_mode"]),
            "--card-power-model", str(p["card_power_model"]),
            "--power-staples", str(int(p["power_staples"])),
            "--role-power-bypass", str(int(p["role_power_bypass"])),
            "--consistency-weight", str(float(p["consistency_weight"])),
            "--combos", str(p["combos"]),
            "--combo-model", str(p["combo_model"]),
            "--combo-weight", str(float(p["combo_weight"])),
            "--engine-boost", str(p["engine_boost"]),
            "--engine-floor", str(float(p["engine_floor"])),
            "--edhrec-floor", str(float(p["edhrec_floor"])),
            "--structural-synergy", str(p["structural_synergy"]),
            "--structural-floor", str(float(p["structural_floor"])),
            "--model", str(p["model"]),
            "--report", f"{prefix}_deck_report.html",
            "--output", f"{prefix}_deck.txt",
            ]

    if p.get("seed") not in (None, ""):
        argv += ["--seed", str(int(p["seed"]))]
    if p.get("recall_edhrec"):
        argv += ["--recall-edhrec",
                 "--recall-edhrec-limit", str(int(p["recall_edhrec_limit"]))]
    if p.get("recall_embeddings"):
        argv += ["--recall-embeddings",
                 "--recall-embedding-limit",
                 str(int(p["recall_embedding_limit"]))]
    if p.get("recall_patterns"):
        argv += ["--recall-patterns"]
    argv += ["--recall-pool-cap", str(int(p["recall_pool_cap"]))]
    if not p.get("quality_roles", True):
        argv += ["--no-quality-roles"]
    if not p.get("signature_pass", True):
        argv += ["--no-signature-pass"]
    if not p.get("synergy_cache", True):
        argv += ["--no-synergy-cache"]
    if not p.get("analysis_cache", True):
        argv += ["--no-analysis-cache"]
    if p.get("budget") not in (None, ""):
        argv += ["--budget", str(float(p["budget"]))]
    if p.get("budget_exclude_unknown"):
        argv += ["--budget-exclude-unknown"]
    for lock in _split_names(p.get("locks", "")):
        argv += ["--lock", lock]
    for ban in _split_names(p.get("bans", "")):
        argv += ["--ban", ban]
    if (p.get("game_changers_file") or "").strip():
        argv += ["--game-changers", p["game_changers_file"].strip()]
    extra = (p.get("extra_flags") or "").strip()
    if extra:
        argv += shlex.split(extra)
    return argv


def refresh_argv(csv_path: str) -> list[str]:
    return [sys.executable, "-m", "mtg_deck_builder.cli", "--csv", csv_path,
            "refresh-cards", "--force"]


def power_scan_argv(payload: dict, csv_path: str) -> list[str]:
    argv = [sys.executable, "-m", "mtg_deck_builder.cli", "--csv", csv_path,
            "power-scan"]
    colors = (payload.get("colors") or "").strip().upper()
    if colors:  # empty = whole DB (includes colorless via subset semantics)
        argv += ["--colors", colors]
    if payload.get("dry_run", True):
        argv += ["--dry-run"]
    if payload.get("batch_size"):
        argv += ["--batch-size", str(int(payload["batch_size"]))]
    return argv


def _split_names(raw: str) -> list[str]:
    return [n.strip() for n in (raw or "").split(",") if n.strip()]


def argv_output(argv: Optional[list[str]]) -> Optional[str]:
    """The --output path from a build argv (None for non-build jobs) —
    lets the UI link 'Open deck' straight from the completion banner."""
    if not argv:
        return None
    try:
        return argv[argv.index("--output") + 1]
    except (ValueError, IndexError):
        return None


class JobManager:
    """Runs ONE subprocess at a time, streaming its output into a ring
    buffer the UI polls. Rejects concurrent submissions (builds cost money)."""

    def __init__(self, max_lines: int = 400):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._log: deque[str] = deque(maxlen=max_lines)
        self.kind: Optional[str] = None
        self.label: Optional[str] = None
        self.argv: Optional[list[str]] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.returncode: Optional[int] = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, argv: list[str], kind: str, label: str,
              extra_env: Optional[dict] = None) -> bool:
        with self._lock:
            if self.running:
                return False
            self._log.clear()
            self.kind, self.label, self.argv = kind, label, argv
            self.started_at, self.finished_at = time.time(), None
            self.returncode = None
            self._log.append(f"$ {' '.join(argv)}")
            env = os.environ.copy()
            if extra_env:
                env.update(extra_env)
            self._proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=env,
            )
        threading.Thread(target=self._pump, daemon=True).start()
        return True

    def _pump(self):
        proc = self._proc
        try:
            for line in proc.stdout:
                self._log.append(line.rstrip("\n"))
        except Exception as e:  # reader died; the poll() below still works
            self._log.append(f"[gui] output reader error: {e}")
        proc.wait()
        self.returncode = proc.returncode
        self.finished_at = time.time()
        self._log.append(f"[exit code {proc.returncode}]")

    def cancel(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self._proc.terminate()
            self._log.append("[cancelled by user]")
            return True

    def state(self) -> dict:
        return {
            "running": self.running,
            "kind": self.kind,
            "label": self.label,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output": argv_output(self.argv),
            "log": list(self._log),
        }


def parse_decklist(path: Path) -> dict:
    """Parse a `<slug>_deck.txt` produced by the builder into
    {commander, sections: [{name, cards: [{count, name}]}], total}.

    Format: "Commander: X" header, then "// Section (N)" groups of
    "<count> <name>" lines. Unknown lines are skipped defensively so a
    hand-edited file still mostly renders.
    """
    commander = None
    sections: list[dict] = []
    current: Optional[dict] = None
    total = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("commander:"):
            commander = line.split(":", 1)[1].strip()
        elif line.startswith("//"):
            name = re.sub(r"\s*\(\d+\)\s*$", "", line[2:].strip())
            current = {"name": name, "cards": []}
            sections.append(current)
        else:
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m and current is not None:
                count = int(m.group(1))
                current["cards"].append(
                    {"count": count, "name": m.group(2).strip()})
                total += count
    return {"commander": commander, "sections": sections, "total": total}


# ----------------------------------------------------------------------
# v0.9.22: card database access for the editor (autocomplete, validation,
# type-derived sections, curve data). Lazy + preloaded in a background
# thread at server start so it's warm by the time anyone edits.
# ----------------------------------------------------------------------
_db = None
_db_lock = threading.Lock()


def _get_db(csv_path: str):
    """Lazy shared CardDatabase; None if the CSV can't be loaded (the
    viewer/editor then degrades: no validation, no curve, sections kept)."""
    global _db
    with _db_lock:
        if _db is None:
            try:
                from .card_database import CardDatabase
                db = CardDatabase(csv_path)
                db.load()
                _db = db
            except Exception as e:
                logger.warning(f"GUI: card DB unavailable ({e})")
                return None
        return _db


def search_cards(csv_path: str, query: str, commanders_only: bool = False,
                 limit: int = 15) -> list[dict]:
    """Name autocomplete. Prefix matches rank above substring matches;
    commanders_only restricts to legendary creatures."""
    db = _get_db(csv_path)
    q = (query or "").strip().lower()
    if db is None or len(q) < 2:
        return []
    prefix, contains = [], []
    for card in db.all_cards:
        if commanders_only and not (
            "legendary" in (card.supertypes or "").lower()
            and "creature" in (card.types or "").lower()
        ):
            continue
        name_l = card.name.lower()
        if name_l.startswith(q):
            prefix.append(card)
        elif q in name_l:
            contains.append(card)
        if len(prefix) >= limit:
            break
    out = (prefix + contains)[:limit]
    return [{"name": c.name, "type": c.card_type,
             "mana_cost": c.mana_cost or ""} for c in out]


_SECTION_ORDER = ["Creatures", "Instants", "Sorcerys", "Artifacts",
                  "Enchantments", "Planeswalkers", "Battles", "Other",
                  "Lands"]


def _section_for(card) -> str:
    """Primary type -> section name (matching the builder's output naming,
    including its 'Sorcerys' spelling so files stay consistent)."""
    types = (card.types or "").lower()
    if "land" in types:
        return "Lands"
    for key, sec in (("creature", "Creatures"), ("instant", "Instants"),
                     ("sorcery", "Sorcerys"), ("artifact", "Artifacts"),
                     ("enchantment", "Enchantments"),
                     ("planeswalker", "Planeswalkers"),
                     ("battle", "Battles")):
        if key in types:
            return sec
    return "Other"


def save_deck(csv_path: str, file_name: str, commander: str,
              cards: list[dict]) -> dict:
    """Write an edited decklist back to `<slug>_deck.txt`.

    - Card names are validated against the DB; unknown names raise
      ValueError (nothing is written).
    - Sections are re-derived from card types, so hand-moved cards always
      land in the right group.
    - The previous file (if any) is kept as `<name>.bak` — one level of
      undo for a hand-edit gone wrong.
    Returns {"warnings": [...]} — count/color-identity issues that are
    worth flagging but shouldn't block saving a work in progress.
    """
    target = safe_relpath(file_name)
    if target is None or not target.name.endswith("_deck.txt"):
        raise ValueError("deck files must end in _deck.txt")
    db = _get_db(csv_path)
    if db is None:
        raise ValueError("card database unavailable; cannot validate/save")

    commander = (commander or "").strip()
    warnings: list[str] = []
    resolved: list[tuple[int, object]] = []
    unknown: list[str] = []
    for entry in cards:
        cname = (entry.get("name") or "").strip()
        count = max(1, int(entry.get("count") or 1))
        card = db.get_by_name(cname)
        if card is None:
            unknown.append(cname)
        else:
            resolved.append((count, card))
    if unknown:
        raise ValueError("unknown card(s): " + ", ".join(unknown[:8]))

    cmd_card = db.get_by_name(commander) if commander else None
    if commander and cmd_card is None:
        warnings.append(f"commander {commander!r} not found in DB")

    total = sum(c for c, _ in resolved)
    if total != 99:
        warnings.append(f"deck has {total} cards (Commander wants 99)")

    if cmd_card is not None:
        cmd_ci = set(ch for ch in (cmd_card.color_identity or "")
                     if ch in "WUBRG")
        off = [card.name for _, card in resolved
               if not set(ch for ch in (card.color_identity or "")
                          if ch in "WUBRG") <= cmd_ci]
        if off:
            warnings.append("outside commander color identity: "
                            + ", ".join(off[:6])
                            + (" …" if len(off) > 6 else ""))

    dupes = {}
    for count, card in resolved:
        dupes[card.name] = dupes.get(card.name, 0) + count
    basics = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
              "Snow-Covered Plains", "Snow-Covered Island",
              "Snow-Covered Swamp", "Snow-Covered Mountain",
              "Snow-Covered Forest"}
    multi = [n for n, c in dupes.items() if c > 1 and n not in basics]
    if multi:
        warnings.append("more than one copy of: " + ", ".join(multi[:6]))

    sections: dict[str, list[tuple[int, object]]] = {}
    for count, card in resolved:
        sections.setdefault(_section_for(card), []).append((count, card))

    lines = [f"Commander: {commander}", ""]
    for sec in _SECTION_ORDER:
        entries = sections.get(sec)
        if not entries:
            continue
        n = sum(c for c, _ in entries)
        lines.append(f"// {sec} ({n})")
        for count, card in sorted(entries, key=lambda t: t[1].name):
            lines.append(f"{count} {card.name}")
        lines.append("")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.replace(target.with_suffix(".bak"))
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"warnings": warnings}


_card_source = None
_card_source_lock = threading.Lock()


def _get_card_source():
    """Lazy shared ScryfallCardSource (disk-cached; reused across requests).
    Returns None if construction fails — the viewer then renders text-only."""
    global _card_source
    with _card_source_lock:
        if _card_source is None:
            try:
                from .scryfall_cards import ScryfallCardSource
                _card_source = ScryfallCardSource(
                    cache_dir="./scryfall_cache")
            except Exception as e:
                logger.warning(f"GUI: no card image source ({e})")
                return None
        return _card_source


def deck_with_images(path: Path, want_images: bool = True,
                     csv_path: Optional[str] = None) -> dict:
    """parse_decklist + per-card Scryfall image URLs (normal size for the
    grid, small for hover previews). Image failures degrade to None — the
    UI shows a text placeholder. First view of a new deck does ~100 cached
    Scryfall fetches (a few seconds); later views are instant.

    v0.9.22: when the card DB is available, each card is enriched with
    mana value and type line (feeds the curve chart and editor display)."""
    deck = parse_decklist(path)
    db = _get_db(csv_path) if csv_path else None
    if db is not None:
        for sec in deck["sections"]:
            for c in sec["cards"]:
                card = db.get_by_name(c["name"])
                if card is not None:
                    c["mv"] = card.mana_value
                    c["type"] = card.card_type
                    c["is_land"] = "land" in (card.types or "").lower()
    source = _get_card_source() if want_images else None
    names = [c["name"] for s in deck["sections"] for c in s["cards"]]
    if deck.get("commander"):
        names.append(deck["commander"])
    images: dict[str, dict] = {}
    if source is not None:
        def _one(name: str) -> tuple[str, dict]:
            try:
                return name, {
                    "normal": source.get_image_url(name, size="normal"),
                    "small": source.get_image_url(name, size="small"),
                }
            except Exception:
                return name, {"normal": None, "small": None}

        # 4 workers ≈ 3-6 req/s against Scryfall's 10/s guidance — cache
        # misses on a fresh deck resolve in ~15-30s instead of minutes;
        # cached decks are instant either way.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            images = dict(pool.map(_one, names))
    deck["images"] = images
    return deck


_KIND_SUFFIXES = (("_deck.txt", "deck"), ("_deck_report.html", "report"),
                  ("_build.log", "log"))


def _artifact_kind(name: str) -> Optional[str]:
    for suffix, kind in _KIND_SUFFIXES:
        if name.endswith(suffix):
            return kind
    return None


def _artifact_slug(path: Path) -> str:
    """Grouping key: the commander slug. decks/<slug>/... uses the folder;
    legacy root files fall back to the filename prefix."""
    if len(path.parts) >= 2 and path.parts[0] == DECKS_DIR.name:
        return path.parts[1]
    name = path.name
    for suffix, _ in _KIND_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] or "deck"
    return "deck"


def _list_outputs() -> list[dict]:
    """Deck/report/log artifacts — decks/<slug>/* plus legacy root files.
    Flat entries {path, kind, slug, mtime}; the UI groups by slug."""
    out = []
    seen = set()
    candidates = list(Path(".").glob("*"))
    if DECKS_DIR.is_dir():
        candidates += list(DECKS_DIR.glob("*/*"))
    for f in candidates:
        if not f.is_file():
            continue
        kind = _artifact_kind(f.name)
        if kind is None:
            continue
        rel = f.as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        out.append({"path": rel, "kind": kind, "slug": _artifact_slug(f),
                    "mtime": f.stat().st_mtime, "size": f.stat().st_size})
    return sorted(out, key=lambda d: -d["mtime"])[:240]


def read_deck_meta(deck_path: Path) -> Optional[dict]:
    """Build metadata for a deck file. Prefers the sidecar
    <prefix>_meta.json (written by GUI builds); for legacy decks, sniffs the
    matching build log's recorded command line for the bracket."""
    prefix = str(deck_path)[: -len("_deck.txt")]
    meta_file = Path(prefix + "_meta.json")
    if meta_file.is_file():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    log_file = Path(prefix + "_build.log")
    if log_file.is_file():
        # The refinement role-status lines near the END of the log carry
        # "[Bracket N — ...]"; read the tail only (logs run to ~14MB).
        try:
            size = log_file.stat().st_size
            with open(log_file, "rb") as f:
                f.seek(max(0, size - 300_000))
                tail = f.read().decode("utf-8", errors="replace")
            m = re.search(r"\[Bracket (\d) ", tail)
            if m:
                return {"bracket": int(m.group(1)), "source": "log"}
        except Exception:
            pass
    return None


def make_handler(csv_path: str, jobs: JobManager):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet the default stderr spam
            logger.debug("gui http: " + fmt % args)

        # -- helpers -------------------------------------------------
        def _json(self, obj, status: int = 200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        # -- routes --------------------------------------------------
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = _PAGE_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json({
                    "csv": csv_path,
                    "defaults": BUILD_DEFAULTS,
                    "job": jobs.state(),
                    "outputs": _list_outputs(),
                    "api_key": api_key_status(load_settings()),
                })
            elif self.path.startswith("/api/cards/search"):
                qs = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query)
                self._json({"results": search_cards(
                    csv_path,
                    (qs.get("q") or [""])[0],
                    commanders_only=(qs.get("commanders") or ["0"])[0] == "1",
                )})
            elif self.path.startswith("/api/deck"):
                self._deck_view()
            elif self.path.startswith("/files/"):
                self._serve_file(self.path[len("/files/"):])
            else:
                self._json({"error": "not found"}, 404)

        def _deck_view(self):
            qs = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query)
            raw = (qs.get("file") or [""])[0]
            f = safe_relpath(raw)
            if f is None or not f.name.endswith("_deck.txt") or not f.is_file():
                self._json({"error": "deck file not found"}, 404)
                return
            want_images = (qs.get("images") or ["1"])[0] != "0"
            try:
                deck = deck_with_images(f, want_images=want_images,
                                        csv_path=csv_path)
                deck["meta"] = read_deck_meta(
                    Path(raw.replace("\\", "/").lstrip("/")))
                self._json(deck)
            except Exception as e:
                self._json({"error": f"could not parse deck: {e}"}, 500)

        def _serve_file(self, raw_name: str):
            # Sandboxed relative path (decks/<slug>/... or legacy root);
            # whitelist extensions.
            f = safe_relpath(raw_name)
            if (f is None or not f.is_file()
                    or f.suffix not in (".html", ".txt", ".log", ".json")):
                self._json({"error": "not found"}, 404)
                return
            ctype = ("text/html; charset=utf-8" if f.suffix == ".html"
                     else "text/plain; charset=utf-8")
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                payload = self._read_json()
            except Exception as e:
                self._json({"error": f"bad JSON: {e}"}, 400)
                return
            try:
                if self.path == "/api/build":
                    commander = (payload.get("commander") or "").strip()
                    prefix = build_prefix(commander)
                    argv = build_argv(payload, csv_path, out_prefix=prefix)
                    # Sidecar metadata: what the viewer shows (bracket etc.)
                    # and a reproducibility record of the exact request.
                    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
                    Path(prefix + "_meta.json").write_text(json.dumps({
                        "commander": commander,
                        "bracket": int(payload.get("bracket")
                                       or BUILD_DEFAULTS["bracket"]),
                        "seed": payload.get("seed"),
                        "budget": payload.get("budget"),
                        "started_at": time.time(),
                        "argv": argv[1:],  # sans interpreter path
                    }, indent=2), encoding="utf-8")
                    label = f"Build: {commander}"
                    self._start(argv, "build", label)
                elif self.path == "/api/refresh-cards":
                    self._start(refresh_argv(csv_path), "refresh",
                                "Refresh card DB from MTGJSON")
                elif self.path == "/api/power-scan":
                    argv = power_scan_argv(payload, csv_path)
                    scope = payload.get("colors") or "whole DB"
                    dry = " (dry run)" if payload.get("dry_run", True) else ""
                    self._start(argv, "power-scan",
                                f"Power scan: {scope}{dry}")
                elif self.path == "/api/settings":
                    settings = load_settings()
                    if "api_key" in payload:
                        set_api_key(settings, payload.get("api_key") or "")
                        save_settings(settings)
                    self._json({"api_key": api_key_status(settings)})
                elif self.path == "/api/deck/save":
                    result = save_deck(
                        csv_path,
                        payload.get("file") or "",
                        payload.get("commander") or "",
                        payload.get("cards") or [],
                    )
                    self._json({"saved": True, **result})
                elif self.path == "/api/cancel":
                    self._json({"cancelled": jobs.cancel()})
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)

        def _start(self, argv: list[str], kind: str, label: str):
            # A key saved in the GUI is injected into the job's environment
            # (wins over the inherited shell env — explicit beats implicit).
            extra_env = {}
            saved = get_api_key(load_settings())
            if saved:
                extra_env["ANTHROPIC_API_KEY"] = saved
            if jobs.start(argv, kind, label, extra_env=extra_env):
                self._json({"started": True, "label": label})
            else:
                self._json({"error": "a job is already running — one at a "
                                     "time (builds cost API credits)"}, 409)

    return Handler


def serve(csv_path: str, port: int = 8765, open_browser: bool = True) -> None:
    """Start the GUI server (blocks until Ctrl+C)."""
    migrate_legacy_key()
    # Warm the card DB in the background so autocomplete/editor/curve are
    # ready by the time anyone clicks (34K-card CSV takes a few seconds).
    threading.Thread(target=_get_db, args=(csv_path,), daemon=True).start()
    jobs = JobManager()
    httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                make_handler(csv_path, jobs))
    url = f"http://127.0.0.1:{port}/"
    print(f"Deck-builder GUI: {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nGUI stopped.")
    finally:
        jobs.cancel()
        httpd.server_close()
