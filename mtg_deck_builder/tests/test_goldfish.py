"""
Tests for v0.9.34 (#35): the goldfish / playtest simulator.

Deterministic (seeded) Monte Carlo over constructed decks with known
composition — assertions are either exact (mulligan logic, determinism)
or statistical with wide tolerances (land curves vs hypergeometric
expectations over 400+ trials).
"""

from mtg_deck_builder.goldfish import (
    GoldfishConfig, rock_production, simulate,
)
from mtg_deck_builder.models import Card, Combo


def _card(name, text="", types="Creature", mv=2):
    is_land = "land" in types.lower()
    return Card(
        name=name, mana_cost="" if is_land else f"{{{mv}}}",
        mana_value=0 if is_land else mv,
        card_type=types, text=text, color_identity="G", colors="G",
        power="2" if types == "Creature" else "",
        toughness="2" if types == "Creature" else "",
        loyalty="", defense="", types=types, subtypes="", supertypes="",
        keywords="", layout="normal", legalities="commander:legal",
    )


def _land(i):
    return _card(f"Forest{i}", "({T}: Add {G}.)", types="Land", mv=0)


def _sol_ring(i=0):
    return _card(f"Sol Ring{i}", "{T}: Add {C}{C}.", types="Artifact", mv=1)


def _deck(n_lands, n_spells, spell_mv=3):
    return ([_land(i) for i in range(n_lands)]
            + [_card(f"Spell{i}", "some text", mv=spell_mv)
               for i in range(n_spells)])


CMD = _card("Cmd", "", types="Legendary Creature", mv=4)


class TestRockProduction:
    def test_sol_ring_produces_two(self):
        assert rock_production(_sol_ring()) == 2

    def test_word_form_any_color(self):
        signet = _card("Signet", "{T}: Add one mana of any color.",
                       types="Artifact", mv=2)
        assert rock_production(signet) == 1

    def test_activation_cost_not_counted(self):
        # The {G} before "Add" is a cost, not production.
        rock = _card("Odd Rock", "{G}, {T}: Add {C}.", types="Artifact", mv=2)
        assert rock_production(rock) == 1

    def test_lands_creatures_nonmana_artifacts_zero(self):
        assert rock_production(_land(0)) == 0
        dork = _card("Elf", "{T}: Add {G}.", types="Creature", mv=1)
        assert rock_production(dork) == 0  # artifacts only (v1)
        sword = _card("Sword", "Equip {2}", types="Artifact", mv=3)
        assert rock_production(sword) == 0


class TestMulligans:
    def test_all_land_hands_are_mulled(self):
        report = simulate(_deck(99, 0), CMD,
                          config=GoldfishConfig(trials=50, seed=1))
        assert report.keep7_rate == 0.0      # 7 lands is never keepable
        assert report.avg_mulligans >= 2.0   # mulls to 5 every game

    def test_no_land_hands_are_mulled(self):
        report = simulate(_deck(0, 99), CMD,
                          config=GoldfishConfig(trials=50, seed=1))
        assert report.keep7_rate == 0.0
        assert report.commander_uncast_rate == 1.0  # never any mana

    def test_normal_deck_mostly_keeps(self):
        report = simulate(_deck(40, 59), CMD,
                          config=GoldfishConfig(trials=400, seed=1))
        assert report.keep7_rate > 0.75


class TestLandCurveAndCommander:
    def test_land_curve_sane_and_monotone(self):
        report = simulate(_deck(40, 59), CMD,
                          config=GoldfishConfig(trials=400, seed=7))
        lands = report.avg_lands_by_turn
        # ~40% lands: expect roughly on-curve through turn 3.
        assert 2.2 <= lands[3] <= 3.2
        assert lands[1] <= lands[2] <= lands[3] <= lands[4]
        assert 0.0 <= report.missed_drop_by_t3_rate <= 0.5

    def test_commander_cast_turn_tracks_mv(self):
        report = simulate(_deck(45, 54), CMD,  # MV 4, land-rich
                          config=GoldfishConfig(trials=400, seed=7))
        assert 4.0 <= report.avg_commander_turn <= 5.2
        dist = report.commander_cast_by_turn
        assert dist[4] <= dist[5] <= dist[6]      # cumulative
        assert report.commander_uncast_rate < 0.2

    def test_rocks_accelerate_commander(self):
        big_cmd = _card("BigCmd", types="Legendary Creature", mv=6)
        plain = simulate(_deck(40, 59), big_cmd,
                         config=GoldfishConfig(trials=400, seed=3))
        rocky_deck = ([_land(i) for i in range(40)]
                      + [_sol_ring(i) for i in range(20)]
                      + [_card(f"S{i}", mv=3) for i in range(39)])
        rocky = simulate(rocky_deck, big_cmd,
                         config=GoldfishConfig(trials=400, seed=3))
        assert rocky.avg_commander_turn < plain.avg_commander_turn - 0.4
        assert rocky.rock_count == 20 and plain.rock_count == 0


class TestCombos:
    def test_two_card_combo_probability_in_open_interval(self):
        deck = _deck(40, 57) + [_card("Piece A"), _card("Piece B")]
        combos = [Combo(cards=["Piece A", "Piece B"], payoff=95)]
        report = simulate(deck, CMD, combos=combos,
                          config=GoldfishConfig(trials=500, seed=11))
        p = report.combo_details[0]["drawn_by_final"]
        assert 0.0 < p < 0.2   # two specific cards of 99 by ~13 draws
        # cumulative monotone
        by_turn = report.any_combo_drawn_by_turn
        assert by_turn[4] <= by_turn[6]

    def test_commander_counts_as_available(self):
        # Combo = commander + a card that fills the whole deck -> always
        # complete from the opening hand.
        deck = [_card("Everywhere") for _ in range(99)]
        combos = [Combo(cards=["Cmd", "Everywhere"], payoff=90)]
        report = simulate(deck, CMD, combos=combos,
                          config=GoldfishConfig(trials=30, seed=2))
        assert report.any_combo_drawn_by_turn[1] == 1.0
        assert report.combo_details[0]["drawn_by_final"] == 1.0


class TestReportPlumbing:
    def test_deterministic_given_seed(self):
        a = simulate(_deck(38, 61), CMD,
                     config=GoldfishConfig(trials=100, seed=42))
        b = simulate(_deck(38, 61), CMD,
                     config=GoldfishConfig(trials=100, seed=42))
        assert a.to_dict() == b.to_dict()

    def test_to_text_renders(self):
        report = simulate(_deck(40, 59), CMD,
                          config=GoldfishConfig(trials=50, seed=5))
        text = report.to_text()
        assert "Keepable opening 7" in text
        assert "Commander (Cmd, MV 4)" in text

    def test_empty_deck_is_safe(self):
        report = simulate([], CMD, config=GoldfishConfig(trials=10))
        assert report.trials == 0
