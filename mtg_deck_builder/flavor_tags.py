"""
Art-tag-based flavor scoring (v0.5).

Takes a list of Scryfall art tags ("forest", "mammoth", "woodland", ...)
and builds a set of card names whose artwork matches any of those tags.
During deck evaluation, we count how many deck cards are in this union
set — more matches = higher flavor score.

Design:
- Tag-matching happens at evaluator-construction time (one bulk pre-fetch),
  then scoring is O(1) per card during deck evaluation.
- Union-of-tags model, not intersection: ANY matching tag counts. Someone
  building a "wilderness" deck who specifies `["forest", "mammoth",
  "deer"]` gets credit for cards matching any of those.
- Combines additively with tribal-subtype flavor from v0.4 (evaluator
  handles the combination).
- Returns 50 (neutral) when no tags configured — zero signal, not a penalty.
- Missing/misspelled tags return empty sets and log a warning.

Typical usage:
    scorer = FlavorTagScorer(
        art_tags=["mammoth", "woodland"],
        tag_client=client,
        color_identity="WG",
    )
    score_0_to_100 = scorer.score_deck(deck)
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Deck
    from .scryfall_tags import ScryfallTagClient

logger = logging.getLogger(__name__)


class FlavorTagScorer:
    """
    Scores a deck by how many of its cards match user-specified art tags.

    Pre-fetches all card sets at construction time for O(1) lookups during
    scoring. Use `create_if_configured` to get None when no tags are
    configured (saves the trouble of network calls for no-op scoring).
    """

    def __init__(
        self,
        art_tags: list[str],
        tag_client: "ScryfallTagClient",
        color_identity: Optional[str] = None,
    ):
        self.art_tags = list(art_tags or [])
        self.tag_client = tag_client
        self.color_identity = color_identity
        # Pre-fetch and union all matching card names
        self._matching_names: set[str] = self._fetch_matching_names()

    @classmethod
    def create_if_configured(
        cls,
        art_tags: list[str],
        tag_client: Optional["ScryfallTagClient"],
        color_identity: Optional[str] = None,
    ) -> Optional["FlavorTagScorer"]:
        """
        Returns a scorer only if art_tags is non-empty and tag_client is
        provided. Otherwise returns None — use this to avoid paying setup
        costs when flavor tags aren't configured.
        """
        if not art_tags or tag_client is None:
            return None
        return cls(art_tags, tag_client, color_identity)

    def _fetch_matching_names(self) -> set[str]:
        """Pre-fetch the union of all card names across configured tags."""
        names: set[str] = set()
        for tag in self.art_tags:
            if not tag or not isinstance(tag, str):
                continue
            matches = self.tag_client.get_cards_with_art_tag(
                tag, color_identity=self.color_identity,
            )
            if not matches:
                logger.info(
                    f"Art tag {tag!r} returned no matches "
                    f"(check spelling at https://scryfall.com/docs/tagger-tags)"
                )
                continue
            names.update(matches)
            logger.debug(f"Art tag {tag!r}: {len(matches)} matching cards")

        logger.info(
            f"Flavor-tag matching: {len(names)} unique cards across "
            f"{len(self.art_tags)} tags"
        )
        return names

    @property
    def matching_count(self) -> int:
        """How many cards in the universe match any configured tag."""
        return len(self._matching_names)

    def score_deck(self, deck: "Deck") -> float:
        """
        Score a deck 0-100 based on art-tag match ratio.

        Mapping:
        - 0% matches: 40 (low — you specified flavor tags but nothing matches)
        - 10% matches: 60
        - 25% matches: 80
        - 50%+ matches: 95 (strong thematic coherence)

        Returns 50 (neutral) if no matching names are in the universe at all,
        because in that case the user's tags don't filter anything — no
        meaningful signal can be extracted.
        """
        if not self._matching_names:
            return 50.0

        total = len(deck.cards)
        if total == 0:
            return 50.0

        match_count = sum(
            1 for card in deck.cards
            if card.name in self._matching_names
        )
        ratio = match_count / total

        # Piecewise linear mapping chosen to reward moderate theme-adherence
        # heavily, since 100% art-tag-match is usually impossible (lands,
        # utility artifacts, etc. rarely have specific art tags).
        if ratio <= 0.10:
            score = 40 + ratio * 200  # 40 → 60 over 0-10%
        elif ratio <= 0.25:
            score = 60 + (ratio - 0.10) * 133  # 60 → 80 over 10-25%
        elif ratio <= 0.50:
            score = 80 + (ratio - 0.25) * 60  # 80 → 95 over 25-50%
        else:
            score = 95 + (ratio - 0.50) * 10  # 95 → 100 over 50-100%

        return max(0.0, min(100.0, score))

    def card_matches(self, card_name: str) -> bool:
        """Check if a single card is in the matching set (utility)."""
        return card_name in self._matching_names
