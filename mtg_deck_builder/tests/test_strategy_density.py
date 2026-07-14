"""
Tests for v0.9.3 metric improvements:
  - is_mana_only_land classifier (basics, duals, tris, fetches vs utility lands)
  - _score_synergy excludes mana-only lands from the average
  - _score_strategy_density counts non-mana-land cards scoring >= 60
  - DeckScores.total() includes the new strategy_density dimension
"""

import pytest

from mtg_deck_builder.deck_evaluator import (
    DeckEvaluator, is_mana_only_land,
)
from mtg_deck_builder.models import (
    Card, Deck, DeckScores, CommanderAnalysis, BuildConfig,
)


def _land(name: str, text: str = "", types: str = "Land") -> Card:
    return Card(
        name=name, mana_cost="", mana_value=0,
        card_type=types, text=text,
        color_identity="", colors="",
        power="", toughness="", loyalty="", defense="",
        types=types, subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _creature(name: str, text: str = "") -> Card:
    return Card(
        name=name, mana_cost="{1}{W}", mana_value=2,
        card_type="Creature", text=text or "vanilla bear",
        color_identity="W", colors="W",
        power="2", toughness="2", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _analysis() -> CommanderAnalysis:
    return CommanderAnalysis(
        name="Test Commander", color_identity="W",
        key_mechanics=["lifegain"], build_around_text="...",
        evaluation_notes="...", category_queries={},
        synergy_keywords=["gain life"],
    )


# ----------------------------------------------------------------------
# is_mana_only_land — the classifier itself
# ----------------------------------------------------------------------

class TestIsManaOnlyLand:
    def test_basic_plains_is_mana_only(self):
        assert is_mana_only_land(_land("Plains",
                                       "({T}: Add {W}.)",
                                       types="Basic Land — Plains"))

    def test_basic_forest_with_no_text_is_mana_only(self):
        # Some basic land entries in the DB have empty text
        assert is_mana_only_land(_land("Forest", "",
                                       types="Basic Land — Forest"))

    def test_dual_land_is_mana_only(self):
        assert is_mana_only_land(_land(
            "Tundra",
            "{T}: Add {W} or {U}.",
            types="Land — Plains Island",
        ))

    def test_tap_dual_is_mana_only(self):
        assert is_mana_only_land(_land(
            "Selesnya Sanctuary",
            "This land enters tapped. {T}: Add {G}{W}.",
        ))

    def test_fetch_land_is_mana_only(self):
        assert is_mana_only_land(_land(
            "Polluted Delta",
            "{T}, Pay 1 life, Sacrifice this land: Search your library "
            "for an Island or Swamp card, put it onto the battlefield, "
            "then shuffle.",
        ))

    def test_mana_confluence_is_mana_only(self):
        assert is_mana_only_land(_land(
            "Mana Confluence",
            "{T}, Pay 1 life: Add one mana of any color.",
        ))

    def test_strip_mine_is_utility(self):
        # Has a non-mana ability (destroy target land)
        assert not is_mana_only_land(_land(
            "Strip Mine",
            "{T}: Add {C}. {T}, Sacrifice this land: Destroy target land.",
        ))

    def test_karns_bastion_is_utility(self):
        assert not is_mana_only_land(_land(
            "Karn's Bastion",
            "{T}: Add {C}. {4}, {T}: Proliferate.",
        ))

    def test_hall_of_heliods_is_utility(self):
        assert not is_mana_only_land(_land(
            "Hall of Heliod's Generosity",
            "{T}: Add {W}. {2}{W}, {T}: Return target enchantment from "
            "your graveyard to your hand.",
        ))

    def test_gavony_township_is_utility(self):
        assert not is_mana_only_land(_land(
            "Gavony Township",
            "{T}: Add {C}. {4}{G}{W}, {T}: Put a +1/+1 counter on each "
            "creature you control.",
        ))

    def test_maze_of_ith_is_utility(self):
        # No mana ability at all — but still a land
        assert not is_mana_only_land(_land(
            "Maze of Ith",
            "{T}: Untap target attacking creature. Prevent all combat "
            "damage that creature would deal this turn.",
        ))

    def test_creature_not_classified_as_land(self):
        # Non-land cards always return False
        assert not is_mana_only_land(_creature("Grizzly Bears"))


# ----------------------------------------------------------------------
# _score_synergy excludes mana-only lands
# ----------------------------------------------------------------------

class TestSynergyExcludesManaLands:
    def test_synergy_average_ignores_basics(self):
        """Adding 30 basic lands to a deck should not change the
        synergy average."""
        config = BuildConfig(commander_name="Test")
        analysis = _analysis()
        synergy_cache = {
            "Soul Warden": 90,
            "Heliod, Sun-Crowned": 95,
            "Trelasarra": 85,
        }
        evaluator = DeckEvaluator(config, analysis, synergy_cache=synergy_cache)

        # Deck with 3 high-synergy creatures + 30 basics
        creatures = [_creature("Soul Warden"), _creature("Heliod, Sun-Crowned"),
                     _creature("Trelasarra")]
        basics = [_land(f"Plains_{i}", "({T}: Add {W}.)",
                        types="Basic Land — Plains") for i in range(30)]

        deck_with_basics = Deck(
            commander=_creature("Commander"),
            cards=creatures + basics,
        )
        deck_no_basics = Deck(
            commander=_creature("Commander"),
            cards=creatures,
        )

        syn_with = evaluator._score_synergy(deck_with_basics)
        syn_without = evaluator._score_synergy(deck_no_basics)

        # Both should be the same — basics are excluded
        assert abs(syn_with - syn_without) < 0.5

    def test_utility_lands_DO_count(self):
        """Utility lands (Karn's Bastion etc.) should still contribute
        to the synergy average."""
        config = BuildConfig(commander_name="Test")
        analysis = _analysis()
        synergy_cache = {
            "Soul Warden": 90,
            "Karn's Bastion": 70,
        }
        evaluator = DeckEvaluator(config, analysis, synergy_cache=synergy_cache)

        deck = Deck(
            commander=_creature("Commander"),
            cards=[
                _creature("Soul Warden"),
                _land("Karn's Bastion",
                      "{T}: Add {C}. {4}, {T}: Proliferate."),
            ],
        )
        # Average of 90 and 70 = 80
        assert abs(evaluator._score_synergy(deck) - 80) < 0.5


# ----------------------------------------------------------------------
# _score_strategy_density
# ----------------------------------------------------------------------

class TestStrategyDensity:
    def test_density_100_when_all_non_mana_cards_strong(self):
        config = BuildConfig(commander_name="Test")
        analysis = _analysis()
        synergy_cache = {
            "Soul Warden": 90,
            "Heliod, Sun-Crowned": 95,
            "Trelasarra": 85,
        }
        evaluator = DeckEvaluator(config, analysis, synergy_cache=synergy_cache)

        deck = Deck(
            commander=_creature("Commander"),
            cards=[
                _creature("Soul Warden"),
                _creature("Heliod, Sun-Crowned"),
                _creature("Trelasarra"),
                # Basic lands — excluded
                _land("Plains", "({T}: Add {W}.)",
                      types="Basic Land — Plains"),
                _land("Forest", "({T}: Add {G}.)",
                      types="Basic Land — Forest"),
            ],
        )
        density = evaluator._score_strategy_density(deck)
        # 3/3 non-mana-land cards scoring >= 60 → 100
        assert density == 100.0

    def test_density_zero_at_or_below_ramp_low(self):
        # v0.9.25: the binary >=60 cliff is a linear ramp — zero credit at
        # or below DENSITY_RAMP_LOW (30), proportional credit up to HIGH (80).
        config = BuildConfig(commander_name="Test")
        analysis = _analysis()
        synergy_cache = {
            "Bear A": 30,   # exactly LOW -> 0.0
            "Bear B": 25,   # below LOW  -> 0.0
            "Bear C": 40,   # (40-30)/50 -> 0.2
        }
        evaluator = DeckEvaluator(config, analysis, synergy_cache=synergy_cache)
        deck = Deck(
            commander=_creature("Commander"),
            cards=[_creature("Bear A"), _creature("Bear B"), _creature("Bear C")],
        )
        # (0 + 0 + 0.2) / 3 * 100
        assert evaluator._score_strategy_density(deck) == pytest.approx(
            20.0 / 3, abs=0.01)

    def test_density_ramp_is_proportional(self):
        # v0.9.25: a synergy-35 card earns 10% credit — no longer identical
        # to a blank, no longer a coin-flip around the old 60 cliff.
        config = BuildConfig(commander_name="Test")
        analysis = _analysis()
        synergy_cache = {
            "Soul Warden": 90,    # >= HIGH -> 1.0
            "Heliod": 85,         # >= HIGH -> 1.0
            "Generic Bear": 35,   # (35-30)/50 -> 0.1
            "Random Card": 25,    # below LOW -> 0.0
        }
        evaluator = DeckEvaluator(config, analysis, synergy_cache=synergy_cache)
        deck = Deck(
            commander=_creature("Commander"),
            cards=[
                _creature("Soul Warden"),
                _creature("Heliod"),
                _creature("Generic Bear"),
                _creature("Random Card"),
            ],
        )
        # (1.0 + 1.0 + 0.1 + 0.0) / 4 * 100 = 52.5
        assert evaluator._score_strategy_density(deck) == pytest.approx(52.5)

    def test_density_midband_half_credit(self):
        # A synergy-55 card is worth half a synergy-80+ card.
        config = BuildConfig(commander_name="Test")
        analysis = _analysis()
        synergy_cache = {"Mid": 55, "Strong": 90}
        evaluator = DeckEvaluator(config, analysis, synergy_cache=synergy_cache)
        deck = Deck(commander=_creature("Commander"),
                    cards=[_creature("Mid"), _creature("Strong")])
        # (0.5 + 1.0) / 2 * 100 = 75
        assert evaluator._score_strategy_density(deck) == pytest.approx(75.0)

    def test_density_ignores_basic_lands(self):
        """Basic lands should be excluded from both numerator and
        denominator — they don't drag the metric down."""
        config = BuildConfig(commander_name="Test")
        analysis = _analysis()
        synergy_cache = {"Soul Warden": 90}
        evaluator = DeckEvaluator(config, analysis, synergy_cache=synergy_cache)
        deck = Deck(
            commander=_creature("Commander"),
            cards=[
                _creature("Soul Warden"),
            ] + [_land(f"Plains_{i}", "", types="Basic Land — Plains")
                 for i in range(36)],
        )
        # 1/1 non-mana-land cards is 100%
        assert evaluator._score_strategy_density(deck) == 100.0


# ----------------------------------------------------------------------
# DeckScores.total() includes strategy_density
# ----------------------------------------------------------------------

def _artifact(name: str, text: str) -> Card:
    return Card(
        name=name, mana_cost="{1}", mana_value=1,
        card_type="Artifact", text=text,
        color_identity="", colors="",
        power="", toughness="", loyalty="", defense="",
        types="Artifact", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


class TestRampDensityNeutrality:
    """v0.9.25: ramp-role cards below clear-support leave the DENSITY set
    (mana infrastructure owes the theme nothing) while staying in the
    synergy average (rock slots compete with spell slots, so mediocre rocks
    still pay a cost). Regression: Sol Ring [syn 40, pow 98] was delivered
    to the pool and to refinement in two consecutive real runs (Jodah B4,
    Doom B5) and declined both times on density math alone."""

    def _evaluator(self, synergy_cache):
        return DeckEvaluator(BuildConfig(commander_name="Test"), _analysis(),
                             synergy_cache=synergy_cache)

    def test_low_synergy_rock_is_density_neutral(self):
        # Same synergy scores; the ramp-role card vanishes from density,
        # the non-ramp card counts (and drags).
        rock_deck = Deck(commander=_creature("Cmd"), cards=[
            _creature("Payoff"), _artifact("Rock", "{T}: Add {C}{C}."),
        ])
        blank_deck = Deck(commander=_creature("Cmd"), cards=[
            _creature("Payoff"), _creature("Blank"),
        ])
        ev = self._evaluator({"Payoff": 90, "Rock": 40, "Blank": 40})
        assert ev._score_strategy_density(rock_deck) == pytest.approx(100.0)
        assert ev._score_strategy_density(blank_deck) == pytest.approx(60.0)

    def test_on_theme_ramp_still_counts(self):
        # A ramp card AT/above clear-support stays in the density set and
        # earns its smoothed credit (Great Henge-class must keep its reward).
        deck = Deck(commander=_creature("Cmd"), cards=[
            _creature("Payoff"),
            _artifact("Henge", "{T}: Add {G}. Whenever a creature enters, "
                               "draw a card."),
        ])
        ev = self._evaluator({"Payoff": 90, "Henge": 75})
        # (1.0 + 0.9) / 2 * 100
        assert ev._score_strategy_density(deck) == pytest.approx(95.0)

    def test_rock_still_drags_synergy_average(self):
        # The anti-flooding half of the design: unlike lands, rocks stay in
        # the synergy AVERAGE.
        deck = Deck(commander=_creature("Cmd"), cards=[
            _creature("Payoff"), _artifact("Rock", "{T}: Add {C}{C}."),
        ])
        ev = self._evaluator({"Payoff": 90, "Rock": 40})
        assert ev._score_synergy(deck) == pytest.approx(65.0)  # (90+40)/2


class TestStapleSlotDecision:
    """v0.9.25 acceptance test for the whole package: with roles already
    saturated, the marginal slot goes to Sol Ring-class rate (syn 40,
    pow 98) but NOT to a mediocre rock (syn 30, pow 70) — the selectivity
    that prevents rock flooding. These deltas were verified against the
    real Doom-run numbers before the fix (Sol Ring lost by ~0.29)."""

    def _setup(self):
        core = []
        for i in range(12):
            core.append(_creature(f"ThemeRamp{i}", "on-theme. {T}: Add {B}."))
        for i in range(10):
            core.append(_creature(f"ThemeDraw{i}", "on-theme. Draw a card."))
        for i in range(8):
            core.append(_creature(f"ThemeKill{i}",
                                  "on-theme. Destroy target creature."))
        core += [_creature(f"Theme{i}", "on-theme text") for i in range(29)]
        syn = {c.name: 60 + (i % 30) for i, c in enumerate(core)}
        power = {c.name: 55.0 for c in core}
        ev = DeckEvaluator(BuildConfig(commander_name="Cmd"), _analysis(),
                           synergy_cache=syn, baseline_power_cache=power)
        weights = {"mana_curve": 0.10, "role_coverage": 0.15, "synergy": 0.35,
                   "strategy_density": 0.20, "power_level": 0.20,
                   "combo": 0.12, "consistency": 0.12}
        ev.synergy_cache = syn  # keep reference for updates
        return core, syn, power, ev, weights

    def _slot_delta(self, challenger, challenger_syn, challenger_pow):
        core, syn, power, ev, weights = self._setup()
        filler = _creature("On-Theme Filler", "on-theme text")
        syn[filler.name] = 65.0
        power[filler.name] = 55.0
        syn[challenger.name] = challenger_syn
        power[challenger.name] = challenger_pow

        def total(extra):
            s = ev.evaluate(Deck(commander=_creature("Cmd"),
                                 cards=core + [extra]))
            s.constraint_penalty = 0.0  # toy decks aren't 99 cards
            return s.total(weights)

        return total(challenger) - total(filler)

    def test_sol_ring_class_wins_marginal_slot(self):
        sol = _artifact("Sol Ring", "{T}: Add {C}{C}.")
        assert self._slot_delta(sol, 40.0, 98.0) > 0

    def test_mediocre_rock_still_loses(self):
        rock = _artifact("Mind Stone", "{T}: Add {C}.")
        assert self._slot_delta(rock, 30.0, 70.0) < 0


class TestTotalIncludesDensity:
    def test_total_uses_strategy_density_weight(self):
        # Two decks identical except for strategy_density
        scores_high = DeckScores(
            mana_curve=80, role_coverage=80, synergy=50,
            strategy_density=80, power_level=50, creativity=50,
        )
        scores_low = DeckScores(
            mana_curve=80, role_coverage=80, synergy=50,
            strategy_density=0, power_level=50, creativity=50,
        )

        weights = {
            "mana_curve": 0.10, "role_coverage": 0.15, "synergy": 0.25,
            "strategy_density": 0.20, "power_level": 0.20, "creativity": 0.10,
        }
        # v0.9.25: total is normalized by the active weight sum (0.90 here;
        # creativity is ignored by design), so the density delta is
        # 80 * 0.20 / 0.90 = 17.78.
        assert (
            abs(scores_high.total(weights) - scores_low.total(weights)
                - 80 * 0.20 / 0.90)
            < 0.1
        )

    def test_default_total_weight_for_density(self):
        # If weights dict omits strategy_density, the default (0.20)
        # should still apply.
        scores = DeckScores(strategy_density=100)
        # With ALL other dimensions zero, total = 100 * 0.20 = 20
        assert abs(scores.total({}) - 20) < 0.1
