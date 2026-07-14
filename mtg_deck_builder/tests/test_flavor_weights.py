"""
Tests for v0.4 flavor dimension and weight-tuning CLI helpers.
"""

import pytest

from mtg_deck_builder.models import (
    Card, Deck, BuildConfig, CommanderAnalysis, DeckScores,
)
from mtg_deck_builder.deck_evaluator import DeckEvaluator


def _creature(name: str, subtypes: str = "", text: str = "") -> Card:
    return Card(
        name=name, mana_cost="{G}", mana_value=1,
        card_type="Creature",
        text=text, color_identity="G", colors="G",
        power="2", toughness="2",
        types="Creature", subtypes=subtypes,
    )


def _non_creature(name: str = "Sol Ring") -> Card:
    return Card(
        name=name, mana_cost="{1}", mana_value=1,
        card_type="Artifact", text="",
        color_identity="", colors="",
        types="Artifact",
    )


class TestDeckScoresFlavor:
    def test_flavor_in_total_when_weighted(self):
        """If score_weights includes 'flavor', it contributes to total."""
        scores = DeckScores(
            mana_curve=50, role_coverage=50, synergy=50,
            power_level=50, creativity=50, flavor=90,
        )
        weights = {
            "mana_curve": 0.1, "role_coverage": 0.1, "synergy": 0.1,
            "power_level": 0.1, "creativity": 0.1, "flavor": 0.5,
        }
        total = scores.total(weights)
        # flavor=90 * 0.5 = 45; mana/role/synergy/power each 50 * 0.1 = 5 (×4 = 20).
        # creativity is excluded (v0.9.7); strategy_density defaults to 0.20
        # weight but scores 0 here. Weighted sum = 65; v0.9.25 normalizes by
        # the active weight sum (0.4 + 0.5 + 0.2 = 1.10) → 59.09.
        assert abs(total - 65.0 / 1.10) < 0.1

    def test_flavor_ignored_without_weight(self):
        """If score_weights has no 'flavor' key, it shouldn't contribute."""
        scores = DeckScores(
            mana_curve=50, role_coverage=50, synergy=50,
            power_level=50, creativity=50, flavor=100,  # high flavor
        )
        # Classic v0.2 weights dict (no flavor key)
        weights = {
            "mana_curve": 0.15, "role_coverage": 0.20, "synergy": 0.35,
            "power_level": 0.20, "creativity": 0.10,
        }
        total = scores.total(weights)
        # flavor (no weight) contributes 0. creativity is excluded (v0.9.7);
        # strategy_density defaults to 0.20 weight but scores 0. Weighted
        # sum = 50 * 0.90 = 45; v0.9.25 normalizes by the active weight sum
        # (0.90 + 0.20 density = 1.10) → 40.91.
        assert abs(total - 45.0 / 1.10) < 0.1


class TestFlavorScoring:
    def _make_commander_with_subtype(self, subtypes: str) -> Card:
        return Card(
            name="Test Commander",
            mana_cost="{2}{G}{W}", mana_value=4,
            card_type="Legendary Creature — Unicorn",
            text="...",
            color_identity="GW", colors="GW",
            power="2", toughness="2",
            types="Creature", subtypes=subtypes, supertypes="Legendary",
        )

    def _make_eval(self) -> DeckEvaluator:
        analysis = CommanderAnalysis(
            name="Test", color_identity="GW",
            key_mechanics=[], build_around_text="",
            evaluation_notes="", category_queries={},
            synergy_keywords=[],
        )
        cfg = BuildConfig(commander_name="Test")
        return DeckEvaluator(cfg, analysis)

    def test_flavor_neutral_no_commander_subtypes(self):
        """Commanders without creature subtypes (e.g. PW commanders) return 50."""
        commander = Card(
            name="Planeswalker Commander",
            mana_cost="{2}{G}{W}", mana_value=4,
            card_type="Legendary Planeswalker",
            text="", color_identity="GW", colors="GW",
            types="Planeswalker", subtypes="",
        )
        deck = Deck(commander=commander, cards=[_creature("C", "Elf")] * 99)
        evaluator = self._make_eval()
        assert evaluator._score_flavor(deck) == 50.0

    def test_flavor_high_for_tribal_match(self):
        """Deck full of Unicorns under a Unicorn commander should score high."""
        commander = self._make_commander_with_subtype("Unicorn")
        # 50 unicorns + 49 non-unicorn creatures
        unicorns = [_creature(f"Unicorn{i}", "Unicorn") for i in range(50)]
        others = [_creature(f"Other{i}", "Elf") for i in range(49)]
        deck = Deck(commander=commander, cards=unicorns + others)
        evaluator = self._make_eval()
        # 50/99 = ~50% shared; should be around 85 per our mapping
        score = evaluator._score_flavor(deck)
        assert 75 <= score <= 95, f"Expected strong tribal score, got {score}"

    def test_flavor_low_for_no_tribal_match(self):
        """Deck with zero matching subtypes should score low."""
        commander = self._make_commander_with_subtype("Unicorn")
        deck = Deck(commander=commander, cards=[_creature(f"E{i}", "Elf") for i in range(99)])
        evaluator = self._make_eval()
        score = evaluator._score_flavor(deck)
        # 0/99 shared -> score ~40
        assert 35 <= score <= 50, f"Expected low tribal score, got {score}"

    def test_flavor_midrange_for_partial_tribal(self):
        """~20% matching should be midrange."""
        commander = self._make_commander_with_subtype("Unicorn")
        cards = (
            [_creature(f"U{i}", "Unicorn") for i in range(20)]
            + [_creature(f"E{i}", "Elf") for i in range(79)]
        )
        deck = Deck(commander=commander, cards=cards)
        evaluator = self._make_eval()
        score = evaluator._score_flavor(deck)
        # ~20% tribal -> score ~60
        assert 55 <= score <= 70, f"Expected mid score, got {score}"

    def test_flavor_ignores_non_creatures(self):
        """Non-creatures shouldn't drag down the tribal ratio."""
        commander = self._make_commander_with_subtype("Unicorn")
        # All creatures are unicorns; non-creatures don't count
        cards = (
            [_creature(f"U{i}", "Unicorn") for i in range(30)]
            + [_non_creature(f"Artifact{i}") for i in range(69)]
        )
        deck = Deck(commander=commander, cards=cards)
        evaluator = self._make_eval()
        score = evaluator._score_flavor(deck)
        # 30/30 creatures match = 100% tribal; non-creatures excluded
        assert score >= 90

    def test_flavor_handles_empty_deck(self):
        commander = self._make_commander_with_subtype("Unicorn")
        deck = Deck(commander=commander, cards=[])
        evaluator = self._make_eval()
        # No creatures to measure; returns 50 (neutral)
        assert evaluator._score_flavor(deck) == 50.0

    def test_flavor_multiple_commander_subtypes(self):
        """Commander with multiple subtypes: any match counts."""
        commander = self._make_commander_with_subtype("Human,Knight")
        cards = [_creature(f"K{i}", "Knight") for i in range(50)] + \
                [_creature(f"W{i}", "Warrior") for i in range(49)]
        deck = Deck(commander=commander, cards=cards)
        evaluator = self._make_eval()
        # 50/99 knights should match
        score = evaluator._score_flavor(deck)
        assert score > 70


class TestWeightCLIHelpers:
    def test_parse_role_target_valid(self):
        from mtg_deck_builder.cli import _parse_role_target
        role, target = _parse_role_target("removal=10,14")
        assert role == "removal"
        assert target == (10, 14)

    def test_parse_role_target_reversed_bounds_fixed(self):
        from mtg_deck_builder.cli import _parse_role_target
        role, target = _parse_role_target("removal=14,10")
        assert target == (10, 14)

    def test_parse_role_target_rejects_bad_format(self):
        import argparse
        from mtg_deck_builder.cli import _parse_role_target
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_role_target("no-equals")
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_role_target("removal=10")  # no comma
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_role_target("removal=abc,def")

    def test_parse_weight_valid(self):
        from mtg_deck_builder.cli import _parse_weight
        dim, val = _parse_weight("synergy=0.6")
        assert dim == "synergy"
        assert val == 0.6

    def test_parse_weight_negative_rejected(self):
        import argparse
        from mtg_deck_builder.cli import _parse_weight
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_weight("synergy=-0.1")

    def test_parse_weight_non_numeric_rejected(self):
        import argparse
        from mtg_deck_builder.cli import _parse_weight
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_weight("synergy=abc")

    def test_build_weight_dict_normalizes_to_one(self):
        from mtg_deck_builder.cli import _build_weight_dict
        weights = _build_weight_dict(
            preset_name=None,
            weight_overrides=["synergy=0.6", "power_level=0.4"],
        )
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_build_weight_dict_preset_plus_override(self):
        from mtg_deck_builder.cli import _build_weight_dict
        # Start from "power" preset, override synergy
        weights = _build_weight_dict(
            preset_name="power",
            weight_overrides=["synergy=0.5"],
        )
        # After normalization synergy should be significant
        assert weights["synergy"] > 0.1

    def test_build_weight_dict_returns_none_when_no_flags(self):
        from mtg_deck_builder.cli import _build_weight_dict
        assert _build_weight_dict(None, []) is None

    def test_build_weight_dict_flavor_preset_has_flavor(self):
        from mtg_deck_builder.cli import _build_weight_dict
        weights = _build_weight_dict(preset_name="flavor", weight_overrides=[])
        # flavor preset should set a non-zero flavor weight
        assert weights["flavor"] > 0

    def test_unknown_weight_dim_ignored_not_crashed(self, caplog):
        from mtg_deck_builder.cli import _build_weight_dict
        weights = _build_weight_dict(
            preset_name=None,
            weight_overrides=["nonexistent_dim=0.5", "synergy=0.5"],
        )
        # Weights should still be built (unknown dim gets a warning and is ignored)
        assert "synergy" in weights
