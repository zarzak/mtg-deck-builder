"""
Tests for the v0.9.7 snow-covered basic-land normalization.

Snow-covered basics are functionally identical to regular basics unless the
deck runs a snow payoff (there is no snow-walk; they keep their normal land
types). The decision is made on the FINAL deck — not the candidate pool, which
almost always contains some snow-matters card. So _normalize_snow_basics:
  - SWAPS snow basics -> regular basics when the finished deck has no snow
    payoff,
  - LEAVES them when a snow payoff is in the deck,
  - and NEVER touches regular basics or non-basic snow lands.
"""

import pytest

from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import BuildConfig, Card, Deck
from mtg_deck_builder.llm_engine import LLMConfig


def _snow_basic(name="Snow-Covered Plains", subtype="Plains", ci="W") -> Card:
    return Card(
        name=name, mana_cost="", mana_value=0,
        card_type=f"Basic Snow Land — {subtype}", text="",
        color_identity=ci, colors="",
        power="", toughness="", loyalty="", defense="",
        types="Land", subtypes=subtype, supertypes="Basic Snow",
        keywords="", layout="normal", legalities="commander:legal",
    )


def _regular_basic(name="Plains", subtype="Plains", ci="W") -> Card:
    return Card(
        name=name, mana_cost="", mana_value=0,
        card_type=f"Basic Land — {subtype}", text="",
        color_identity=ci, colors="",
        power="", toughness="", loyalty="", defense="",
        types="Land", subtypes=subtype, supertypes="Basic",
        keywords="", layout="normal", legalities="commander:legal",
    )


def _snow_payoff() -> Card:
    return Card(
        name="Marit Lage's Slumber", mana_cost="{3}{U}", mana_value=4,
        card_type="Enchantment",
        text=("At the beginning of your upkeep, if you control ten or more "
              "snow permanents, sacrifice this and create Marit Lage."),
        color_identity="U", colors="U",
        power="", toughness="", loyalty="", defense="",
        types="Enchantment", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _plain_card(name="Vanilla Thing") -> Card:
    """A non-land card that does not care about snow."""
    return Card(
        name=name, mana_cost="{1}{G}", mana_value=2,
        card_type="Creature", text="Trample.",
        color_identity="G", colors="G",
        power="3", toughness="3", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="Trample",
        layout="normal", legalities="commander:legal",
    )


def _snow_mana_card() -> Card:
    return Card(
        name="Coldsteel Heart", mana_cost="{2}", mana_value=2,
        card_type="Snow Artifact", text="{T}: Add {S}.",
        color_identity="", colors="",
        power="", toughness="", loyalty="", defense="",
        types="Artifact", subtypes="", supertypes="Snow", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _make_builder(test_csv_path) -> DeckBuilder:
    config = BuildConfig(
        commander_name="Lathiel, the Bounteous Dawn", random_seed=42,
    )
    return DeckBuilder(
        card_database_path=test_csv_path,
        config=config,
        llm_config=LLMConfig(mock_mode=True),
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class TestSnowBasicHelpers:
    def test_is_snow_basic_true_for_snow_basic(self):
        assert DeckBuilder._is_snow_basic(_snow_basic()) is True

    def test_is_snow_basic_false_for_regular_basic(self):
        assert DeckBuilder._is_snow_basic(_regular_basic()) is False

    def test_cares_about_snow_via_text(self):
        assert DeckBuilder._cares_about_snow(_snow_payoff()) is True

    def test_cares_about_snow_via_snow_mana_symbol(self):
        # {S} in a cost or ability text counts as a snow payoff.
        assert DeckBuilder._cares_about_snow(_snow_mana_card()) is True

    def test_snow_basic_is_not_its_own_payoff(self):
        # A snow basic has empty text and no {S} cost — it must not count as
        # the payoff that justifies keeping snow basics.
        assert DeckBuilder._cares_about_snow(_snow_basic()) is False

    def test_plain_card_does_not_care_about_snow(self):
        assert DeckBuilder._cares_about_snow(_regular_basic()) is False


# ----------------------------------------------------------------------
# Pool filtering
# ----------------------------------------------------------------------

class TestNormalizeSnowBasics:
    def _deck(self, b, cards) -> Deck:
        return Deck(commander=b._commander, cards=list(cards))

    def test_swaps_snow_basics_without_payoff(self, test_csv_path):
        b = _make_builder(test_csv_path)
        deck = self._deck(b, [
            _snow_basic("Snow-Covered Plains", "Plains", "W"),
            _snow_basic("Snow-Covered Forest", "Forest", "G"),
            _plain_card(),
        ])
        before = len(deck.cards)
        b._normalize_snow_basics(deck)
        assert len(deck.cards) == before  # count preserved
        # No snow basics survive; they became regular basics from the DB.
        assert not any(DeckBuilder._is_snow_basic(c) for c in deck.cards)
        names = {c.name for c in deck.cards}
        assert {"Plains", "Forest"}.issubset(names)

    def test_keeps_snow_basics_when_deck_has_payoff(self, test_csv_path):
        b = _make_builder(test_csv_path)
        deck = self._deck(b, [
            _snow_basic("Snow-Covered Plains", "Plains", "W"),
            _snow_payoff(),  # a real snow payoff IS in the deck
        ])
        b._normalize_snow_basics(deck)
        names = [c.name for c in deck.cards]
        assert "Snow-Covered Plains" in names  # payoff present → keep

    def test_keeps_snow_basics_when_payoff_is_snow_mana(self, test_csv_path):
        b = _make_builder(test_csv_path)
        deck = self._deck(b, [_snow_basic(), _snow_mana_card()])
        b._normalize_snow_basics(deck)
        assert "Snow-Covered Plains" in {c.name for c in deck.cards}

    def test_regular_basics_untouched(self, test_csv_path):
        b = _make_builder(test_csv_path)
        deck = self._deck(b, [
            _regular_basic("Plains", "Plains", "W"),
            _regular_basic("Forest", "Forest", "G"),
            _plain_card(),
        ])
        b._normalize_snow_basics(deck)
        assert [c.name for c in deck.cards] == ["Plains", "Forest", "Vanilla Thing"]

    def test_snow_basic_is_not_its_own_payoff(self, test_csv_path):
        # A deck of ONLY snow basics (no real payoff) must still normalize —
        # the snow basics themselves don't count as the payoff.
        b = _make_builder(test_csv_path)
        deck = self._deck(b, [
            _snow_basic("Snow-Covered Plains", "Plains", "W"),
            _snow_basic("Snow-Covered Forest", "Forest", "G"),
        ])
        b._normalize_snow_basics(deck)
        assert not any(DeckBuilder._is_snow_basic(c) for c in deck.cards)
