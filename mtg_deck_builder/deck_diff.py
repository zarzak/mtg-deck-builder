"""
Deck diff mode (v0.5).

Compare two decks — typically a warm-start snapshot and a new result —
and report what was kept, added, and removed. Groups changes by the role
each card fills (ramp, removal, draw, etc.) so you can see at a glance
whether your iteration moved the deck in the direction you intended.

Typical usage:
    from mtg_deck_builder.deck_diff import diff_decks, format_diff

    from_deck = WarmStartDeck.from_json_file("lathiel_v1.json")
    to_deck   = WarmStartDeck.from_json_file("lathiel_v2.json")

    result = diff_decks(from_deck, to_deck)
    print(format_diff(result))

Output example:
    Commander: Lathiel, the Bounteous Dawn
    Kept: 87 cards
    Added: 12 cards
    Removed: 12 cards

    === Added ===
      [removal]    Swords to Plowshares, Path to Exile, Generous Gift
      [ramp]       Sol Ring, Arcane Signet
      [synergy]    Soul Warden, Archangel of Thune
      [other]      ...

    === Removed ===
      [creature]   ...
      [other]      ...

Role classification for diffing uses the same card_fills_role helper the
rest of the system uses, with a best-effort "other" bucket for cards that
don't fit any role cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .models import WarmStartDeck, Deck
    from .card_database import CardDatabase


# A deck-like input: WarmStartDeck (names only) or Deck (full Card objects).
# We accept both so callers can diff saved snapshots or live results.
DeckLike = Union["WarmStartDeck", "Deck", dict, list]


@dataclass
class DiffResult:
    """Structured result of diffing two decks."""
    commander_from: Optional[str]
    commander_to: Optional[str]
    kept: list[str]
    added: list[str]
    removed: list[str]
    # Role grouping (populated if a CardDatabase is provided)
    added_by_role: dict[str, list[str]] = field(default_factory=dict)
    removed_by_role: dict[str, list[str]] = field(default_factory=dict)

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def added_count(self) -> int:
        return len(self.added)

    @property
    def removed_count(self) -> int:
        return len(self.removed)

    @property
    def commander_changed(self) -> bool:
        return (
            self.commander_from is not None
            and self.commander_to is not None
            and self.commander_from != self.commander_to
        )


def _extract_names_and_commander(
    deck_like: DeckLike,
) -> tuple[list[str], Optional[str]]:
    """
    Normalize a deck-like input to (card_names, commander_name).

    Supports:
    - WarmStartDeck (has .card_names, .commander_name)
    - Deck (has .cards list of Card, .commander Card)
    - dict with 'card_names'/'commander_name' keys
    - raw list of strings (no commander info)
    """
    # Import inside the function to avoid circular imports at module load
    from .models import WarmStartDeck, Deck

    if isinstance(deck_like, WarmStartDeck):
        return list(deck_like.card_names), deck_like.commander_name

    if isinstance(deck_like, Deck):
        names = [c.name for c in deck_like.cards]
        commander = deck_like.commander.name if deck_like.commander else None
        return names, commander

    if isinstance(deck_like, dict):
        # Canonical WarmStartDeck format: commander_name + card_names
        # Plus a convenience fallback for humans typing `commander` +
        # `cards` by hand.
        card_names = (
            deck_like.get("card_names")
            or deck_like.get("cards")
            or []
        )
        commander_name = (
            deck_like.get("commander_name")
            or deck_like.get("commander")
        )
        return list(card_names), commander_name

    if isinstance(deck_like, list):
        return list(deck_like), None

    raise TypeError(
        f"Cannot diff object of type {type(deck_like).__name__}; "
        "expected WarmStartDeck, Deck, dict, or list of card names."
    )


def diff_decks(
    from_deck: DeckLike,
    to_deck: DeckLike,
    card_db: Optional["CardDatabase"] = None,
) -> DiffResult:
    """
    Compute the diff between two decks.

    Args:
        from_deck: The "before" deck.
        to_deck: The "after" deck.
        card_db: Optional CardDatabase. When provided, added and removed
            cards are grouped by role (ramp, removal, etc.) for easier
            reading. Without it, they land in a single "other" bucket.

    Returns:
        DiffResult with structured diff info.

    Multiset semantics: basic lands can appear multiple times. We use a
    multiset (Counter) so "went from 15 Forests to 17 Forests" shows as
    2 added Forests, not "Forest is in both so it's kept."
    """
    from collections import Counter

    from_names, from_commander = _extract_names_and_commander(from_deck)
    to_names, to_commander = _extract_names_and_commander(to_deck)

    from_count = Counter(from_names)
    to_count = Counter(to_names)

    # Per-name: kept = min(from, to); added = max(0, to-from); removed = max(0, from-to)
    kept: list[str] = []
    added: list[str] = []
    removed: list[str] = []

    all_names = set(from_count) | set(to_count)
    for name in sorted(all_names):
        f = from_count.get(name, 0)
        t = to_count.get(name, 0)
        keep_n = min(f, t)
        add_n = max(0, t - f)
        rem_n = max(0, f - t)
        kept.extend([name] * keep_n)
        added.extend([name] * add_n)
        removed.extend([name] * rem_n)

    # Role grouping (optional)
    added_by_role: dict[str, list[str]] = {}
    removed_by_role: dict[str, list[str]] = {}
    if card_db is not None:
        added_by_role = _group_by_role(added, card_db)
        removed_by_role = _group_by_role(removed, card_db)

    return DiffResult(
        commander_from=from_commander,
        commander_to=to_commander,
        kept=kept,
        added=added,
        removed=removed,
        added_by_role=added_by_role,
        removed_by_role=removed_by_role,
    )


def _group_by_role(
    card_names: list[str],
    card_db: "CardDatabase",
) -> dict[str, list[str]]:
    """
    Bucket card names by the first role they match.

    Checks roles in a priority order so a ramp spell isn't listed under
    "draw" just because the same role pattern matches loosely. Cards that
    fit no role land in "other".
    """
    from .card_database import card_fills_role

    # Priority order: most distinctive first. A ramp spell is usually more
    # interesting for the diff than its incidental text properties.
    priority_roles = ("ramp", "draw", "removal", "wipe", "land")

    buckets: dict[str, list[str]] = {}
    for name in card_names:
        card = card_db.get_by_name(name)
        if card is None:
            buckets.setdefault("other", []).append(name)
            continue
        role = "other"
        for candidate_role in priority_roles:
            try:
                if card_fills_role(card, candidate_role):
                    role = candidate_role
                    break
            except Exception:
                continue
        buckets.setdefault(role, []).append(name)

    return buckets


def format_diff(
    diff: DiffResult,
    show_kept: bool = False,
    max_per_group: int = 20,
) -> str:
    """
    Format a DiffResult as a human-readable string.

    Args:
        diff: The DiffResult to format.
        show_kept: If True, also list the kept cards (usually too noisy).
        max_per_group: Cap per role bucket; extras are summarized as
            "(... and N more)". Keeps output readable on mobile.
    """
    lines: list[str] = []

    # Header
    if diff.commander_changed:
        lines.append(
            f"Commander changed: {diff.commander_from} -> {diff.commander_to}"
        )
    elif diff.commander_to:
        lines.append(f"Commander: {diff.commander_to}")

    lines.append(
        f"Kept: {diff.kept_count}   Added: {diff.added_count}   "
        f"Removed: {diff.removed_count}"
    )

    # Added
    if diff.added:
        lines.append("")
        lines.append("=== Added ===")
        lines.extend(
            _format_bucket(diff.added, diff.added_by_role, max_per_group)
        )

    # Removed
    if diff.removed:
        lines.append("")
        lines.append("=== Removed ===")
        lines.extend(
            _format_bucket(diff.removed, diff.removed_by_role, max_per_group)
        )

    if show_kept and diff.kept:
        lines.append("")
        lines.append("=== Kept ===")
        lines.append("  " + ", ".join(diff.kept[:max_per_group]))
        if len(diff.kept) > max_per_group:
            lines.append(f"  (... and {len(diff.kept) - max_per_group} more)")

    if not diff.added and not diff.removed:
        lines.append("")
        lines.append("(No changes.)")

    return "\n".join(lines)


def _format_bucket(
    flat_names: list[str],
    grouped: dict[str, list[str]],
    max_per_group: int,
) -> list[str]:
    """Helper: render added/removed either grouped (if available) or flat."""
    out: list[str] = []
    if grouped:
        # Stable ordering: ramp, draw, removal, wipe, land, other, then alpha
        preferred_order = ["ramp", "draw", "removal", "wipe", "land", "other"]
        extras = sorted(k for k in grouped if k not in preferred_order)
        ordered_roles = [r for r in preferred_order if r in grouped] + extras
        for role in ordered_roles:
            names = grouped[role]
            shown = names[:max_per_group]
            line = f"  [{role:<10}] " + ", ".join(shown)
            if len(names) > max_per_group:
                line += f"  (... and {len(names) - max_per_group} more)"
            out.append(line)
    else:
        shown = flat_names[:max_per_group]
        out.append("  " + ", ".join(shown))
        if len(flat_names) > max_per_group:
            out.append(f"  (... and {len(flat_names) - max_per_group} more)")
    return out
