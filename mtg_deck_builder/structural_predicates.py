"""
Structural / attribute synergy predicates (v0.9.9).

The synergy pipeline (embeddings, patterns, LLM-on-text) keys entirely on card
TEXT. That breaks for "attribute" archetypes whose payoff is a structural
property of the card rather than its text — most starkly "vanilla creatures
matter" (Jasmine Boreal): a vanilla creature has NO text, so it's invisible to
every text-based signal even though it's the whole gameplan.

This module lets a commander reward card ATTRIBUTES directly. The commander
analysis emits a small, bounded vocabulary of predicates; this evaluator maps
each to a check over existing `Card` fields (no new data needed). Downstream,
the builder pulls matching cards into recall and floors their synergy so they
compete — independent of text.

Predicate grammar (case-insensitive):
  - bare flags:   vanilla | no_abilities | colorless | creature | land
  - key:value:    subtype:Bear | type:Creature | supertype:Legendary |
                  keyword:Trample
  - numeric:      mv<=2 | cmc>=6 | power>=4 | toughness<=1  (ops: <= >= == < > =)
Unknown predicates simply never match (safe no-op).
"""

from __future__ import annotations

import operator
import re
from typing import Optional

from .models import Card

_OPS = {
    "<=": operator.le, ">=": operator.ge, "==": operator.eq,
    "=": operator.eq, "<": operator.lt, ">": operator.gt,
}
_NUMERIC_RE = re.compile(r"^(mv|cmc|power|toughness)\s*(<=|>=|==|=|<|>)\s*(-?\d+)$")

# Cues in the commander-analysis prose that imply a structural predicate —
# insurance so the theme is caught even if the LLM doesn't emit a predicate.
# Cues must be SPECIFIC: a false positive here flips the whole structural
# machinery (LLM synergy routing, attribute recall, on-ramp, synergy floor).
# The bare word "colorless" is NOT specific — nearly any analysis can mention
# "colorless mana" or "colorless mana rocks" in passing — so the colorless cue
# requires a colorless-matters phrase.
_TEXT_CUES = [
    (("vanilla", "no abilities", "no ability", "without abilities",
      "creatures with no"), "vanilla"),
    (("colorless creature", "colorless spell", "colorless permanent",
      "colorless card", "colorless matter", "devoid"), "colorless"),
]


def _numeric_attr(card: Card, attr: str) -> Optional[int]:
    if attr in ("mv", "cmc"):
        return card.mana_value
    raw = card.power if attr == "power" else card.toughness
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None  # '*', 'X', empty → not comparable


def _tokens(field: str) -> set[str]:
    """Tokenize a comma-separated card field for key:value matching.

    Yields BOTH the full comma-separated phrases and their individual words,
    so multi-word values match either way: keywords "First strike, Lifelink"
    match "keyword:first strike" (phrase) as well as "keyword:lifelink";
    subtypes "Time Lord, Doctor" match "subtype:time lord"."""
    out: set[str] = set()
    for part in (field or "").split(","):
        phrase = re.sub(r"\s+", " ", part.strip().lower())
        if not phrase:
            continue
        out.add(phrase)
        out.update(w for w in phrase.split(" ") if w)
    return out


def card_matches_predicate(card: Card, predicate: str) -> bool:
    """True if `card` satisfies the single structural `predicate`."""
    p = (predicate or "").strip().lower()
    if not p:
        return False

    if p in ("vanilla", "no_abilities", "no abilities"):
        return card.is_vanilla
    if p == "colorless":
        # No colored mana identity (Eldrazi, artifact creatures, etc.).
        return not (card.colors or "").strip()
    if p == "creature":
        return card.is_creature
    if p == "land":
        return card.is_land

    if ":" in p:
        key, _, val = p.partition(":")
        key, val = key.strip(), val.strip()
        if not val:
            return False
        field = {
            "subtype": card.subtypes, "type": card.types,
            "supertype": card.supertypes, "keyword": card.keywords,
        }.get(key)
        if field is None:
            return False
        return val in _tokens(field)

    m = _NUMERIC_RE.match(p)
    if m:
        attr, op, num = m.group(1), m.group(2), int(m.group(3))
        val = _numeric_attr(card, attr)
        return val is not None and _OPS[op](val, num)

    return False


def card_matches_any(card: Card, predicates) -> bool:
    return any(card_matches_predicate(card, p) for p in (predicates or []))


def derive_structural_predicates(analysis) -> list[str]:
    """Union the analysis's explicit `structural_predicates` with any implied
    by cues in its prose (insurance for when the LLM names the theme but omits
    the predicate). Returns a sorted, de-duplicated list.

    Cues are scanned in `build_around_text` ONLY — the 2-3 sentence core
    strategy, where "vanilla" appears only if it IS the plan. They are NOT
    scanned in `evaluation_notes`: that field is full of asides like
    "creatures that are otherwise vanilla are better here" (a real Lathiel
    analysis), which falsely flipped the entire structural machinery
    (attribute recall, on-ramp, LLM routing) for a lifegain commander."""
    preds = {str(p).strip() for p in getattr(analysis, "structural_predicates", []) if str(p).strip()}
    blob = (getattr(analysis, "build_around_text", "") or "").lower()
    for cues, pred in _TEXT_CUES:
        if any(c in blob for c in cues):
            preds.add(pred)
    return sorted(preds)
