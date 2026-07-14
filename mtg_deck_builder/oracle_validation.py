"""
Oracle-tag role validation (v0.6).

Our role classification (ramp/draw/removal/wipe/land) comes from regex
patterns in card_database.py. Those patterns are reasonable but not
perfect — they'll miss cards with unusual phrasing, or match things that
happen to include removal-like words in flavor text.

The Scryfall Tagger project crowdsources functional tags on every card.
`function:removal` finds every card the community has classified as
removal, using their own hand-curated criteria. Comparing our regex
output against community tags gives us a useful second opinion.

This module doesn't change scoring or deck contents. It's purely
diagnostic — it produces a report of disagreements so you (the user) can
decide whether our regex is missing something or the community tag is
too loose.

Two kinds of disagreement:

- **Missed**: our regex didn't flag the card as role X, but oracle tags
  did. Examples: a creature with an activated ability that's technically
  removal but our regex keyword list didn't catch.
- **Extra**: our regex flagged the card as role X, but community tags
  didn't. Examples: a card whose text mentions "destroy" as part of a
  cost or condition, not an effect.

Usage:

    from mtg_deck_builder.oracle_validation import validate_roles

    report = validate_roles(
        deck=result.best_deck,
        tag_client=tag_client,
        roles_to_check=["ramp", "removal", "draw"],
    )
    print(format_role_report(report))

Returns a ValidationReport with per-role details. Graceful when tags
aren't available — an unreachable tag_client produces an empty report
rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import Deck, Card
    from .scryfall_tags import ScryfallTagClient


# Mapping from our internal role names to Scryfall oracle tags.
# Conservative: only tags that clearly match the concept. A card can fit
# multiple community tags; we union them.
#
# Community tags sourced from:
#   https://scryfall.com/docs/tagger-tags
# If we're unsure, we leave the mapping out — false negatives on
# validation (not catching real disagreements) are better than false
# positives (flagging non-disagreements).
ROLE_TO_ORACLE_TAGS: dict[str, list[str]] = {
    "ramp": [
        "mana-ramp",
        "ramp-artifact",
        "ramp-creature",
    ],
    "draw": [
        "card-draw",
        "draw",
    ],
    "removal": [
        "removal",
        "creature-removal",
        "spot-removal",
    ],
    "wipe": [
        "boardwipe",
        "mass-removal",
    ],
    "land": [
        "land",
    ],
}


@dataclass
class RoleDisagreement:
    """
    A single card where our regex and the community oracle tag disagree.

    Kind is either:
    - "missed": regex didn't flag the card, but oracle tag did
    - "extra": regex flagged it, but oracle tag didn't
    """
    card_name: str
    role: str
    kind: str  # "missed" or "extra"


@dataclass
class ValidationReport:
    """Diagnostic report comparing regex role classification vs oracle tags."""
    roles_checked: list[str]
    cards_checked: int
    disagreements: list[RoleDisagreement] = field(default_factory=list)
    # Per-role summary: role -> (missed_count, extra_count)
    per_role_summary: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Roles we couldn't validate (no oracle tag mapping, or fetch failed)
    skipped_roles: list[str] = field(default_factory=list)

    def count_by_kind(self, kind: str) -> int:
        return sum(1 for d in self.disagreements if d.kind == kind)

    @property
    def total_disagreements(self) -> int:
        return len(self.disagreements)


def validate_roles(
    deck: "Deck",
    tag_client: "ScryfallTagClient",
    roles_to_check: Optional[list[str]] = None,
    color_identity: Optional[str] = None,
) -> ValidationReport:
    """
    Compare regex role classification against community oracle tags.

    Args:
        deck: The deck to audit. Uses deck.cards (not deck.commander).
        tag_client: ScryfallTagClient instance for querying oracle tags.
        roles_to_check: List of role names to validate. Defaults to all
            roles that have a ROLE_TO_ORACLE_TAGS mapping. Typically you
            want ["ramp", "draw", "removal", "wipe"]; "land" is very
            obvious from type line so skipping it saves a query.
        color_identity: Optional color filter. If provided, oracle tag
            queries use `id<=<colors>` to restrict results to cards
            compatible with the commander. Useful when the deck's own
            commander has a narrow identity; no point flagging missed
            cards that couldn't legally be in this deck anyway.

    Returns:
        ValidationReport. Always returns — never raises. Unavailable
        tags become skipped_roles; partial failures still validate the
        roles that could be fetched.
    """
    # Default: every role we have a mapping for, minus "land"
    if roles_to_check is None:
        roles_to_check = [r for r in ROLE_TO_ORACLE_TAGS if r != "land"]

    report = ValidationReport(
        roles_checked=list(roles_to_check),
        cards_checked=len(deck.cards),
    )

    for role in roles_to_check:
        oracle_tags = ROLE_TO_ORACLE_TAGS.get(role)
        if not oracle_tags:
            report.skipped_roles.append(role)
            continue

        # Union-of-tags: any community tag counts as "flagged for this role"
        tag_matches: set[str] = set()
        fetch_ok = False
        for ot in oracle_tags:
            try:
                names = tag_client.get_cards_with_oracle_tag(
                    ot, color_identity=color_identity,
                )
            except Exception:
                # Never let tag fetch break validation — mark role as skipped
                # and continue. A single raising query doesn't prevent others
                # from succeeding.
                continue
            if names:
                fetch_ok = True
                tag_matches.update(names)

        if not fetch_ok:
            # No tag data at all for this role — skip rather than report
            # every card as "missed"
            report.skipped_roles.append(role)
            continue

        # Now compare for each deck card
        missed = 0
        extra = 0
        for card in deck.cards:
            regex_says = _regex_flags(card, role)
            tags_say = card.name in tag_matches

            if regex_says and not tags_say:
                report.disagreements.append(RoleDisagreement(
                    card_name=card.name, role=role, kind="extra",
                ))
                extra += 1
            elif tags_say and not regex_says:
                report.disagreements.append(RoleDisagreement(
                    card_name=card.name, role=role, kind="missed",
                ))
                missed += 1
        report.per_role_summary[role] = (missed, extra)

    return report


def _regex_flags(card: "Card", role: str) -> bool:
    """Safe wrapper around card_fills_role that never raises."""
    try:
        from .card_database import card_fills_role
        return bool(card_fills_role(card, role))
    except Exception:
        return False


def format_role_report(
    report: ValidationReport,
    max_per_kind: int = 10,
) -> str:
    """
    Format a ValidationReport as readable text.

    Args:
        report: The report to render.
        max_per_kind: Cap per (role, kind) bucket; extras shown as
            "... and N more". Keeps output readable on mobile.
    """
    lines: list[str] = []
    lines.append(
        f"Role validation: {report.cards_checked} cards, "
        f"{len(report.roles_checked)} roles"
    )

    if report.skipped_roles:
        lines.append(
            f"Skipped (no tag data): {', '.join(report.skipped_roles)}"
        )

    if not report.disagreements:
        lines.append("No disagreements found.")
        return "\n".join(lines)

    lines.append(f"Total disagreements: {report.total_disagreements}")
    lines.append("")

    # Group disagreements by (role, kind) for readable output
    from collections import defaultdict
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for d in report.disagreements:
        buckets[(d.role, d.kind)].append(d.card_name)

    # Stable ordering: follow roles_checked, then "missed" before "extra"
    # since missed cards are usually more interesting than extras
    for role in report.roles_checked:
        for kind in ("missed", "extra"):
            key = (role, kind)
            if key not in buckets:
                continue
            names = buckets[key]
            shown = names[:max_per_kind]
            header = (
                f"=== {role} / {kind} "
                f"({'tags flagged, regex did not' if kind == 'missed' else 'regex flagged, tags did not'}) ==="
            )
            lines.append(header)
            lines.append("  " + ", ".join(shown))
            if len(names) > max_per_kind:
                lines.append(f"  (... and {len(names) - max_per_kind} more)")
            lines.append("")

    return "\n".join(lines).rstrip()
