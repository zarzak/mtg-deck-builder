"""
Commander Brackets (v0.9.15) — the official 1-5 bracket system.

Replaces the old 1-10 power_level knob. Brackets encode two distinct things
at once, which is why they fit this engine better than a scalar:

  1. A RULES REGIME (official deck restrictions per bracket):
       B1 Exhibition — no Game Changers, no mass land denial, no extra
          turns, no two-card combos.
       B2 Core       — no Game Changers, no MLD, no CHAINING extra turns
          (we allow 1 extra-turn card), no two-card combos.
       B3 Upgraded   — up to 3 Game Changers, no MLD, no chaining extra
          turns, no EARLY two-card combos ("before turn six").
       B4 Optimized  — no restrictions beyond the ban list.
       B5 cEDH       — no restrictions; the build POSTURE changes
          (structural templates: curve, ramp, interaction — see
          deck_evaluator.CEDH_CURVE and models.CEDH_ROLE_TARGETS).

  2. A BUILD POSTURE (weight scaling, structural templates at B5).

Everything deterministic here is sourced to the OFFICIAL bracket rules and
Game Changer list (WotC, as supplied July 2026) — per the project rule:
deterministic constants only for format structure. The one judgment proxy
is EARLY_COMBO_MV (see below), which approximates the official "before
turn six" language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

# ----------------------------------------------------------------------
# Game Changers — sourced entirely from the card data (v0.9.18).
# ----------------------------------------------------------------------
# There is NO hardcoded list. The authoritative source is the `isGameChanger`
# column in the card CSV, populated by `refresh-cards` straight from MTGJSON
# (self-refreshing — MTGJSON tracks the official WotC list). A card is a Game
# Changer iff:
#   - its per-card `is_game_changer` attribute is True (the CSV column), OR
#   - an external override list (--game-changers FILE) names it.
# If the CSV has no isGameChanger column and no override is given, NO cards
# are flagged (the builder warns at brackets 1-3). Refresh the CSV; don't
# reintroduce a constant to maintain.
_OVERRIDE_NAMES: Optional[frozenset] = None


def set_game_changer_names(names) -> None:
    """Set an external Game Changer name list (from --game-changers FILE).
    When set, name matching uses it instead of the per-card CSV attribute.
    Pass None to clear (fall back to the CSV attribute)."""
    global _OVERRIDE_NAMES
    _OVERRIDE_NAMES = frozenset(names) if names is not None else None


def reset_game_changer_source() -> None:
    """Clear any external override (test hygiene / between builds)."""
    global _OVERRIDE_NAMES
    _OVERRIDE_NAMES = None


def active_game_changer_names() -> Optional[frozenset]:
    """The external override name set, or None when the CSV attribute is the
    source (there is no embedded list)."""
    return _OVERRIDE_NAMES


def is_game_changer(card) -> bool:
    """True if the card is a Game Changer.

    An external override list (--game-changers FILE), when set, is the sole
    source and is matched by name (INCLUDING double-faced names —
    "Tergrid, God of Fright // Tergrid's Lantern"). Otherwise the per-card
    `is_game_changer` attribute (from the CSV isGameChanger column) decides.
    """
    if _OVERRIDE_NAMES is not None:
        name = getattr(card, "name", "") or ""
        if name in _OVERRIDE_NAMES:
            return True
        if " // " in name:
            return any(f.strip() in _OVERRIDE_NAMES
                       for f in name.split(" // "))
        return False
    return bool(getattr(card, "is_game_changer", None))


def load_game_changer_names(path: str) -> Optional[set]:
    """Load a Game Changer name list from an external file for --game-changers.

    Accepts (auto-detected):
      - a JSON array of names: ["Sol Ring", ...];
      - a JSON object — either {"names": [...]} or an MTGJSON-atomic-style
        {"data": {"Card Name": {... "isGameChanger": true ...}}} (or a list
        of printings per name), from which names flagged isGameChanger are
        extracted;
      - a plain newline-delimited text file (one name per line, # comments ok).
    Returns a set of names, or None on failure (caller keeps the embedded list).
    """
    import json
    try:
        raw = open(path, encoding="utf-8").read()
    except OSError:
        return None
    raw = raw.strip()
    if not raw:
        return None
    # JSON first.
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Newline text.
        return {
            ln.strip() for ln in raw.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }
    if isinstance(obj, list):
        return {str(x).strip() for x in obj if str(x).strip()}
    if isinstance(obj, dict):
        if isinstance(obj.get("names"), list):
            return {str(x).strip() for x in obj["names"] if str(x).strip()}
        data = obj.get("data", obj)
        if isinstance(data, dict):
            out: set = set()
            for name, entry in data.items():
                printings = entry if isinstance(entry, list) else [entry]
                for p in printings:
                    if isinstance(p, dict) and p.get("isGameChanger"):
                        out.add(str(name).strip())
                        break
            return out or None
    return None


# ----------------------------------------------------------------------
# Mass land denial + extra turns (regex classification, conservative).
# ----------------------------------------------------------------------
# Official definition (bracket rules): cards that destroy/exile/bounce
# lands en masse or prevent them from untapping/producing mana. Patterns
# are deliberately conservative — false negatives are safer than tagging
# fair cards as MLD.
_MLD_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in (
        r"\bdestroy all lands\b",
        r"\bdestroy each land\b",
        r"\bdestroy all \w+ lands\b",           # "destroy all snow lands" etc.
        r"\bexile all lands\b",
        r"\bdestroy all (?:artifacts, creatures, and lands|nonland permanents and lands)\b",
        r"\beach player sacrifices [^.]{0,20}lands\b",
        r"\bsacrifices? all lands\b",
        r"\breturn all lands\b",
        r"lands don't untap during their controllers?' untap steps",
        r"players can't play lands",
    )
]

# "take(s) an extra turn" covers Time Warp-class cards. Cards granting an
# OPPONENT an extra turn are rare enough to ignore the false positive.
_EXTRA_TURN_RE = re.compile(r"takes? an extra turn", re.IGNORECASE)


@lru_cache(maxsize=None)
def _text_flags(name: str, text: str) -> tuple[bool, bool]:
    """(is_mld, grants_extra_turn) for a card's rules text, memoized —
    these run inside the GA fitness loop."""
    is_mld = any(p.search(text) for p in _MLD_PATTERNS)
    extra = bool(_EXTRA_TURN_RE.search(text))
    return is_mld, extra


def is_mass_land_denial(card) -> bool:
    return _text_flags(card.name, card.text or "")[0]


def grants_extra_turn(card) -> bool:
    return _text_flags(card.name, card.text or "")[1]


# ----------------------------------------------------------------------
# Two-card combo policy.
# ----------------------------------------------------------------------
# The official B3 rule is "no two-card combos BEFORE TURN SIX". Deck
# construction can't see turns, so we use a deterministic proxy: a
# two-card combo whose pieces' combined mana value is below this threshold
# is assumed assemblable by ~turn 6 with normal ramp ("early"). E.g.
# Heliod (3) + Walking Ballista (0) = 3 -> early; Selenia (5) + Mirror
# Universe (6) = 11 -> late (B3-legal). This is the module's one judgment
# constant — it approximates official rule language, and the compliance
# report always lists the combos so a human can overrule.
from .tuning import EARLY_COMBO_MV  # single source of truth (re-exported)


def two_card_combo_banned(combo, bracket: int, mv_of) -> bool:
    """Is this detected combo banned AT DECK CONSTRUCTION for `bracket`?

    Only 2-card combos are restricted by the rules (the commander counts
    as one of the two — it is always available, which is exactly why
    commander-inclusive pairs are the FIRST thing the rules care about).
    `mv_of` maps a card name -> mana value (commander included); unknown
    names count as 0 (conservative: unknown = assumed cheap).
    """
    cards = getattr(combo, "cards", None) or []
    if len(cards) != 2:
        return False
    if bracket >= 4:
        return False
    if bracket <= 2:
        return True
    # Bracket 3: early combos only.
    total_mv = sum(mv_of(n) or 0 for n in cards)
    return total_mv < EARLY_COMBO_MV


# ----------------------------------------------------------------------
# Bracket rules table + compliance audit.
# ----------------------------------------------------------------------

# gc_limit / extra_turn_limit: None = unrestricted.
BRACKET_RULES: dict[int, dict] = {
    1: {"name": "Exhibition", "gc_limit": 0, "mld_allowed": False,
        "extra_turn_limit": 0, "combos": "none"},
    2: {"name": "Core", "gc_limit": 0, "mld_allowed": False,
        "extra_turn_limit": 1, "combos": "none"},
    3: {"name": "Upgraded", "gc_limit": 3, "mld_allowed": False,
        "extra_turn_limit": 1, "combos": "late"},
    4: {"name": "Optimized", "gc_limit": None, "mld_allowed": True,
        "extra_turn_limit": None, "combos": "any"},
    5: {"name": "cEDH", "gc_limit": None, "mld_allowed": True,
        "extra_turn_limit": None, "combos": "any"},
}


def bracket_name(bracket: int) -> str:
    return BRACKET_RULES.get(bracket, {}).get("name", "?")


@dataclass
class BracketAudit:
    """Compliance audit of a FINISHED deck against its target bracket."""
    bracket: int
    game_changers: list[str] = field(default_factory=list)
    gc_limit: Optional[int] = None
    mld_cards: list[str] = field(default_factory=list)
    extra_turn_cards: list[str] = field(default_factory=list)
    # Two-card combos fully present in the deck (commander counts), with
    # combined MV, banned-at-this-bracket flag, and description.
    two_card_combos: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return not self.violations

    @property
    def effective_bracket(self) -> int:
        """Lowest bracket this deck's CONTENTS actually conform to."""
        for b in (1, 2, 3, 4):
            if not _violations_for(self, b):
                return b
        return 4  # contents-wise, 4 and 5 have identical rules


def _violations_for(audit: "BracketAudit", bracket: int) -> list[str]:
    rules = BRACKET_RULES[bracket]
    v: list[str] = []
    gc_limit = rules["gc_limit"]
    if gc_limit is not None and len(audit.game_changers) > gc_limit:
        v.append(
            f"{len(audit.game_changers)} Game Changers (bracket {bracket} "
            f"allows {gc_limit}): {', '.join(audit.game_changers)}"
        )
    if not rules["mld_allowed"] and audit.mld_cards:
        v.append(f"mass land denial: {', '.join(audit.mld_cards)}")
    et_limit = rules["extra_turn_limit"]
    if et_limit is not None and len(audit.extra_turn_cards) > et_limit:
        v.append(
            f"{len(audit.extra_turn_cards)} extra-turn cards (bracket "
            f"{bracket} allows {et_limit}): "
            f"{', '.join(audit.extra_turn_cards)}"
        )
    if rules["combos"] == "none":
        banned = [c for c in audit.two_card_combos]
        if banned:
            v.append(
                "two-card combos present: "
                + "; ".join(c["desc"] for c in banned)
            )
    elif rules["combos"] == "late":
        early = [c for c in audit.two_card_combos if c["early"]]
        if early:
            v.append(
                "early two-card combos present: "
                + "; ".join(c["desc"] for c in early)
            )
    return v


def audit_deck(deck, combos, bracket: int) -> BracketAudit:
    """Audit the finished deck's CONTENTS against its target bracket.

    Report-only — enforcement lives in the pool filter, the GA penalty,
    and the refinement guard. `combos` is the detected-combo list (reward
    + banned alike); presence is checked against the deck's 99 plus the
    commander (always available).
    """
    audit = BracketAudit(bracket=bracket,
                         gc_limit=BRACKET_RULES[bracket]["gc_limit"])
    if deck is None or not deck.cards:
        return audit

    seen: set[str] = set()
    for card in deck.cards:
        if card.name in seen:
            continue
        seen.add(card.name)
        if is_game_changer(card):
            audit.game_changers.append(card.name)
        if is_mass_land_denial(card):
            audit.mld_cards.append(card.name)
        if grants_extra_turn(card):
            audit.extra_turn_cards.append(card.name)

    names = {c.name for c in deck.cards}
    mv_by_name = {c.name: c.mana_value for c in deck.cards}
    if deck.commander is not None:
        names.add(deck.commander.name)
        mv_by_name[deck.commander.name] = deck.commander.mana_value

    seen_pairs: set[frozenset] = set()
    for combo in combos or []:
        cards = getattr(combo, "cards", None) or []
        if len(cards) != 2 or not all(n in names for n in cards):
            continue
        key = frozenset(cards)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        total_mv = sum(mv_by_name.get(n, 0) for n in cards)
        audit.two_card_combos.append({
            "cards": list(cards),
            "mv": total_mv,
            "early": total_mv < EARLY_COMBO_MV,
            "desc": f"{' + '.join(cards)} (MV {total_mv})",
        })

    audit.violations = _violations_for(audit, bracket)
    return audit


def power_level_to_bracket(power_level: int) -> int:
    """Map the DEPRECATED 1-10 power_level onto brackets: 1-2 -> B1,
    3-4 -> B2, 5-7 -> B3, 8-9 -> B4, 10 -> B5."""
    p = max(1, min(10, int(power_level)))
    if p <= 2:
        return 1
    if p <= 4:
        return 2
    if p <= 7:
        return 3
    if p <= 9:
        return 4
    return 5
