"""
Tests for the v0.9.15 Commander Bracket system (official 1-5 brackets).

Phase 1: bracket knob + Game Changer list + compliance audit.
Phase 2: enforcement — pool filters, GA penalties, refinement guards.
Phase 3: bracket-5 (cEDH) structural templates.
"""

import pytest

from mtg_deck_builder.bracket import (
    is_game_changer, is_mass_land_denial, grants_extra_turn,
    two_card_combo_banned, audit_deck, power_level_to_bracket,
    EARLY_COMBO_MV,
)
from mtg_deck_builder.deck_evaluator import (
    DeckEvaluator, FastEvaluator, IDEAL_CURVE, CEDH_CURVE,
)
from mtg_deck_builder.models import (
    Card, Deck, BuildConfig, CommanderAnalysis, Combo, CEDH_ROLE_TARGETS,
)


def _card(name, text="", types="Creature", mv=2, is_land=False,
          gc=False) -> Card:
    """v0.9.18: `gc=True` marks the card as a Game Changer via the per-card
    attribute (the CSV isGameChanger column path) — there is no embedded
    name list anymore."""
    return Card(
        name=name, mana_cost=f"{{{mv}}}", mana_value=mv,
        card_type="Land" if is_land else "Creature", text=text,
        color_identity="G", colors="G",
        power=None if is_land else "2", toughness=None if is_land else "2",
        loyalty="", defense="",
        types="Land" if is_land else types, subtypes="", supertypes="",
        keywords="", layout="normal", legalities="commander:legal",
        is_game_changer=gc,
    )


def _analysis() -> CommanderAnalysis:
    return CommanderAnalysis(
        name="Cmd", color_identity="G", key_mechanics=[],
        build_around_text="", evaluation_notes="", category_queries={},
        synergy_keywords=[],
    )


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------

class TestClassification:
    def test_game_changer_from_csv_attribute(self):
        # v0.9.18: the per-card is_game_changer attribute (CSV isGameChanger
        # column) is the source — there is no embedded name list.
        assert is_game_changer(_card("Rhystic Study", gc=True))
        assert not is_game_changer(_card("Sol Ring"))  # not flagged
        assert not is_game_changer(_card("Rhystic Study"))  # unflagged -> no

    def test_game_changer_explicit_attribute_wins(self):
        c = _card("Some Future Card")
        object.__setattr__(c, "is_game_changer", True)
        assert is_game_changer(c)


class TestGameChangerRefresh:
    """v0.9.18: external-file override + CSV-attribute source (no embedded)."""

    def teardown_method(self):
        from mtg_deck_builder.bracket import reset_game_changer_source
        reset_game_changer_source()

    def test_external_override_is_sole_source_when_set(self):
        from mtg_deck_builder.bracket import set_game_changer_names
        set_game_changer_names({"Sol Ring"})
        # Override matches by name and IGNORES the per-card attribute.
        assert is_game_changer(_card("Sol Ring"))
        assert not is_game_changer(_card("Rhystic Study", gc=True))
        # Clearing reverts to the per-card attribute.
        set_game_changer_names(None)
        assert is_game_changer(_card("Rhystic Study", gc=True))
        assert not is_game_changer(_card("Sol Ring"))

    def test_override_matches_dfc_faces(self):
        from mtg_deck_builder.bracket import set_game_changer_names
        set_game_changer_names({"Tergrid, God of Fright"})
        assert is_game_changer(
            _card("Tergrid, God of Fright // Tergrid's Lantern"))

    def test_no_source_flags_nothing(self):
        from mtg_deck_builder.bracket import reset_game_changer_source
        reset_game_changer_source()
        # No override, no per-card attr -> not a GC (no embedded list).
        assert not is_game_changer(_card("Rhystic Study"))

    def test_load_json_array(self, tmp_path):
        from mtg_deck_builder.bracket import load_game_changer_names
        p = tmp_path / "gc.json"
        p.write_text('["Sol Ring", "Mana Crypt"]', encoding="utf-8")
        assert load_game_changer_names(str(p)) == {"Sol Ring", "Mana Crypt"}

    def test_load_mtgjson_atomic_shape(self, tmp_path):
        from mtg_deck_builder.bracket import load_game_changer_names
        import json
        p = tmp_path / "atomic.json"
        p.write_text(json.dumps({"data": {
            "Sol Ring": [{"isGameChanger": True}],
            "Grizzly Bears": [{"isGameChanger": False}],
            "Mana Crypt": {"isGameChanger": True},
        }}), encoding="utf-8")
        assert load_game_changer_names(str(p)) == {"Sol Ring", "Mana Crypt"}

    def test_load_newline_text_and_bad_path(self, tmp_path):
        from mtg_deck_builder.bracket import load_game_changer_names
        p = tmp_path / "gc.txt"
        p.write_text("# comment\nSol Ring\nMana Crypt\n", encoding="utf-8")
        assert load_game_changer_names(str(p)) == {"Sol Ring", "Mana Crypt"}
        assert load_game_changer_names(str(tmp_path / "nope.txt")) is None

    def test_mld_detection(self):
        assert is_mass_land_denial(_card("Armageddon-ish",
                                         text="Destroy all lands."))
        assert is_mass_land_denial(_card("Winter-ish",
                                         text="Lands don't untap during their "
                                              "controllers' untap steps."))
        assert not is_mass_land_denial(_card("Strip-ish",
                                             text="Destroy target land."))

    def test_extra_turn_detection(self):
        assert grants_extra_turn(_card("Warp-ish",
                                       text="Take an extra turn after this one."))
        assert not grants_extra_turn(_card("Plain", text="Draw a card."))


# ----------------------------------------------------------------------
# Two-card combo policy
# ----------------------------------------------------------------------

class TestComboPolicy:
    MV = {"Cheap A": 1, "Cheap B": 2, "Big A": 5, "Big B": 6}

    def _mv(self, name):
        return self.MV.get(name, 0)

    def test_any_two_card_banned_at_b1_b2(self):
        combo = Combo(cards=["Big A", "Big B"], payoff=95)
        assert two_card_combo_banned(combo, 1, self._mv)
        assert two_card_combo_banned(combo, 2, self._mv)

    def test_only_early_banned_at_b3(self):
        cheap = Combo(cards=["Cheap A", "Cheap B"], payoff=95)  # MV 3
        big = Combo(cards=["Big A", "Big B"], payoff=95)        # MV 11
        assert two_card_combo_banned(cheap, 3, self._mv)
        assert not two_card_combo_banned(big, 3, self._mv)
        assert EARLY_COMBO_MV == 10

    def test_nothing_banned_at_b4_b5(self):
        cheap = Combo(cards=["Cheap A", "Cheap B"], payoff=95)
        assert not two_card_combo_banned(cheap, 4, self._mv)
        assert not two_card_combo_banned(cheap, 5, self._mv)

    def test_three_card_combos_never_banned(self):
        combo = Combo(cards=["Cheap A", "Cheap B", "Big A"], payoff=95)
        assert not two_card_combo_banned(combo, 1, self._mv)


# ----------------------------------------------------------------------
# Audit
# ----------------------------------------------------------------------

class TestAudit:
    def _deck(self, cards):
        return Deck(commander=_card("Cmd", mv=4), cards=cards)

    def test_compliant_deck(self):
        deck = self._deck([_card("Grizzly"), _card("Forest", is_land=True)])
        a = audit_deck(deck, [], bracket=2)
        assert a.compliant
        assert a.effective_bracket == 1

    def test_violations_reported(self):
        deck = self._deck([
            _card("Rhystic Study", gc=True),
            _card("Geddon", text="Destroy all lands."),
            _card("Combo A", mv=1), _card("Combo B", mv=2),
        ])
        combos = [Combo(cards=["Combo A", "Combo B"], payoff=95)]
        a = audit_deck(deck, combos, bracket=2)
        assert not a.compliant
        assert a.game_changers == ["Rhystic Study"]
        assert a.mld_cards == ["Geddon"]
        assert len(a.two_card_combos) == 1 and a.two_card_combos[0]["early"]
        assert len(a.violations) == 3
        # Contents conform only to bracket 4 (MLD bans it from 1-3).
        assert a.effective_bracket == 4

    def test_commander_counts_toward_combo_presence(self):
        deck = self._deck([_card("Partner Piece", mv=1)])
        combos = [Combo(cards=["Cmd", "Partner Piece"], payoff=90)]
        a = audit_deck(deck, combos, bracket=2)
        assert len(a.two_card_combos) == 1

    def test_b3_gc_budget(self):
        deck = self._deck([_card("Rhystic Study", gc=True),
                           _card("Demonic Tutor", gc=True),
                           _card("Mystical Tutor", gc=True),
                           _card("Vampiric Tutor", gc=True)])
        a = audit_deck(deck, [], bracket=3)
        assert not a.compliant  # 4 GCs > 3
        a3 = audit_deck(self._deck([_card("Rhystic Study", gc=True)]), [],
                        bracket=3)
        assert a3.compliant

    def test_power_level_mapping(self):
        assert power_level_to_bracket(1) == 1
        assert power_level_to_bracket(4) == 2
        assert power_level_to_bracket(7) == 3
        assert power_level_to_bracket(9) == 4
        assert power_level_to_bracket(10) == 5


# ----------------------------------------------------------------------
# Config resolution + weights
# ----------------------------------------------------------------------

class TestConfigResolution:
    def test_default_bracket_is_4(self):
        assert BuildConfig(commander_name="X").bracket == 4

    def test_explicit_bracket_wins(self):
        cfg = BuildConfig(commander_name="X", bracket=2, power_level=10)
        assert cfg.bracket == 2

    def test_deprecated_power_level_maps(self):
        assert BuildConfig(commander_name="X", power_level=7).bracket == 3
        assert BuildConfig(commander_name="X", power_level=10).bracket == 5

    def test_bracket_clamped(self):
        assert BuildConfig(commander_name="X", bracket=9).bracket == 5
        assert BuildConfig(commander_name="X", bracket=0).bracket == 1


# ----------------------------------------------------------------------
# Evaluator enforcement + templates
# ----------------------------------------------------------------------

class TestEvaluatorEnforcement:
    def _evaluator(self, bracket, banned=None) -> DeckEvaluator:
        cfg = BuildConfig(commander_name="X", bracket=bracket)
        return DeckEvaluator(cfg, _analysis(), banned_combos=banned)

    def test_no_penalty_at_b4(self):
        ev = self._evaluator(4)
        deck = Deck(commander=_card("Cmd"),
                    cards=[_card("Rhystic Study"),
                           _card("Geddon", text="Destroy all lands.")])
        assert ev._bracket_penalty(deck) == 0.0

    def test_b3_gc_excess_penalized(self):
        ev = self._evaluator(3)
        deck = Deck(commander=_card("Cmd"), cards=[
            _card("Rhystic Study", gc=True), _card("Demonic Tutor", gc=True),
            _card("Mystical Tutor", gc=True), _card("Vampiric Tutor", gc=True),
            _card("Worldly Tutor", gc=True),
        ])
        # 5 GCs, limit 3 -> 2 excess x 4.0
        assert ev._bracket_penalty(deck) == pytest.approx(8.0)

    def test_banned_combo_assembly_penalized(self):
        banned = [Combo(cards=["A", "B"], payoff=95)]
        ev = self._evaluator(2, banned=banned)
        assembled = Deck(commander=_card("Cmd"),
                         cards=[_card("A"), _card("B")])
        near = Deck(commander=_card("Cmd"), cards=[_card("A")])
        assert ev._bracket_penalty(assembled) == pytest.approx(8.0)
        assert ev._bracket_penalty(near) == 0.0

    def test_extra_turn_limits(self):
        warp = _card("Warp", text="Take an extra turn after this one.")
        warp2 = _card("Walk", text="Take an extra turn after this one.")
        deck = Deck(commander=_card("Cmd"), cards=[warp, warp2])
        # B2 allows 1 -> one excess x 4.0; B1 allows 0 -> two excess.
        assert self._evaluator(2)._bracket_penalty(deck) == pytest.approx(4.0)
        assert self._evaluator(1)._bracket_penalty(deck) == pytest.approx(8.0)

    def test_cedh_curve_selected_at_b5(self):
        # A low-to-the-ground deck scores BETTER at bracket 5 than at 4,
        # and vice versa for a midrange curve.
        low = [_card(f"c{i}", mv=1) for i in range(30)] + \
              [_card(f"d{i}", mv=2) for i in range(25)]
        ev5 = self._evaluator(5)
        ev4 = self._evaluator(4)
        deck = Deck(commander=_card("Cmd"), cards=low)
        assert ev5._score_mana_curve(deck) > ev4._score_mana_curve(deck)
        assert CEDH_CURVE != IDEAL_CURVE

    def test_cedh_role_targets_at_b5(self):
        cfg5 = BuildConfig(commander_name="X", bracket=5)
        t = cfg5.get_effective_role_targets()
        assert t["land"] == CEDH_ROLE_TARGETS["land"]
        assert "protection" in t
        # User overrides still win.
        cfg5b = BuildConfig(commander_name="X", bracket=5,
                            role_target_overrides={"land": (30, 34)})
        assert cfg5b.get_effective_role_targets()["land"] == (30, 34)
        # Bracket 4 keeps the casual template.
        cfg4 = BuildConfig(commander_name="X", bracket=4)
        assert cfg4.get_effective_role_targets()["land"] == (35, 38)

    def test_fast_evaluator_cedh_shape(self):
        # 30 lands + avg MV ~1.5 is rewarded at B5, penalized at B4.
        cards = ([_card(f"Land{i}", is_land=True) for i in range(30)]
                 + [_card(f"x{i}", mv=1) for i in range(35)]
                 + [_card(f"y{i}", mv=2) for i in range(34)])
        deck = Deck(commander=_card("Cmd"), cards=cards[:99])
        f5 = FastEvaluator(BuildConfig(commander_name="X", bracket=5), _analysis())
        f4 = FastEvaluator(BuildConfig(commander_name="X", bracket=4), _analysis())
        assert f5.evaluate(deck) > f4.evaluate(deck)


class TestCedhManaBase:
    """v0.9.33 (#32): the coupled 'lands + fast mana >= 38' cEDH floor. A
    rock-heavy build may run leaner lands (>= 26) as long as total mana
    sources clear 38; the old model forced 28 lands regardless of ramp."""

    def _ev(self, bracket):
        return DeckEvaluator(BuildConfig(commander_name="X", bracket=bracket),
                             _analysis())

    def _rock(self, name):
        return _card(name, text="{T}: Add {C}.", types="Artifact", mv=2)

    def _deck(self, n_lands, n_rocks):
        cards = [_card(f"Land{i}", is_land=True) for i in range(n_lands)]
        cards += [self._rock(f"Rock{i}") for i in range(n_rocks)]
        return Deck(commander=_card("Cmd"), cards=cards)

    def test_penalty_only_at_bracket_5(self):
        deck = self._deck(20, 5)  # 25 sources, well under 38
        assert self._ev(4)._cedh_mana_base_penalty(deck) == 0.0
        assert self._ev(5)._cedh_mana_base_penalty(deck) > 0.0

    def test_lean_lands_ok_when_fast_mana_compensates(self):
        # 26 lands + 14 rocks = 40 sources: clears the coupled floor even
        # though lands are below the old 28-land requirement.
        deck = self._deck(26, 14)
        assert self._ev(5)._cedh_mana_base_penalty(deck) == 0.0

    def test_land_and_source_deficit_penalized(self):
        # 24 lands + 10 rocks = 34 sources: 4 short of 38.
        deck = self._deck(24, 10)
        # Default role_shortfall_penalty rate is 2.0 -> 4 * 2.0 = 8.0.
        assert self._ev(5)._cedh_mana_base_penalty(deck) == pytest.approx(8.0)

    def test_land_floor_lowered_to_26(self):
        from mtg_deck_builder.models import CEDH_ROLE_TARGETS
        assert CEDH_ROLE_TARGETS["land"][0] == 26


# ----------------------------------------------------------------------
# v0.9.15c: land synergy threshold, inclusion recall, flex role
# ----------------------------------------------------------------------

class TestLandSynergyThreshold:
    def _evaluator(self, synergy_cache) -> DeckEvaluator:
        cfg = BuildConfig(commander_name="X")
        return DeckEvaluator(cfg, _analysis(), synergy_cache=synergy_cache)

    def test_offtheme_utility_land_excluded_from_synergy_avg(self):
        # Regression (real cEDH run): Boseiju-class lands (honest synergy
        # ~35) DRAGGED the synergy average while textless filler was
        # neutral — the fitness repelled the format's best utility lands.
        boseiju = _card("Boseiju-ish", text="Channel — destroy target.",
                        is_land=True)
        cradle = _card("Cradle-ish", text="Add G for each creature.",
                       is_land=True)
        payoff = _card("Payoff")
        ev = self._evaluator({"Boseiju-ish": 35.0, "Cradle-ish": 90.0,
                              "Payoff": 80.0})
        deck = Deck(commander=_card("Cmd"), cards=[boseiju, cradle, payoff])
        scoreable = {c.name for c in ev._synergy_scoreable(deck)}
        assert "Boseiju-ish" not in scoreable   # land doing a land's job
        assert "Cradle-ish" in scoreable        # genuinely on-theme land
        assert "Payoff" in scoreable            # nonlands always count
        # Average = (90 + 80) / 2, not dragged to (35+90+80)/3.
        assert ev._score_synergy(deck) == pytest.approx(85.0)

    def test_density_uses_same_scoreable_set(self):
        boseiju = _card("Boseiju-ish", text="Channel.", is_land=True)
        payoff = _card("Payoff")
        ev = self._evaluator({"Boseiju-ish": 35.0, "Payoff": 80.0})
        deck = Deck(commander=_card("Cmd"), cards=[boseiju, payoff])
        # Density: 1 of 1 scoreable on-strategy (Boseiju no longer deflates).
        assert ev._score_strategy_density(deck) == pytest.approx(100.0)


class TestInclusionRecall:
    def test_bracket5_only_and_ranked_by_inclusion(self, db):
        from mtg_deck_builder.candidate_recall import recall_from_edhrec_inclusion

        class _E:
            def __init__(self, name, incl):
                self.name = name
                self.inclusion_rate = incl
                self.synergy = 0.0  # generically good = zero distinctive

        class _D:
            cards = {
                "Sol Ring": _E("Sol Ring", 0.95),
                "Soul Warden": _E("Soul Warden", 0.5),
                "Phyrexian Arena": _E("Phyrexian Arena", 0.9),  # off-color (B)
                "Fringe": _E("Fringe", 0.1),  # below min_inclusion
            }
        out = recall_from_edhrec_inclusion(_D(), db, "G,W", limit=10)
        names = [c.name for c in out]
        assert names[0] == "Sol Ring"          # highest inclusion first
        assert "Soul Warden" in names
        assert "Phyrexian Arena" not in names  # off-color filtered
        assert "Fringe" not in names           # below floor
        assert recall_from_edhrec_inclusion(None, db, "G,W") == []


class TestProtectionGrantFix:
    """v0.9.15d regression: equipment/aura grants say "has hexproof", not
    "gains" — Lightning Greaves and Swiftfoot Boots were role-orphans.
    (The flex/tutor buckets that came out of the same orphan review were
    later REMOVED in favor of the general power-staples channel — cards
    that help a plan enter via recall/power, not via invented sub-types.)"""

    def test_equipment_protection_grants(self):
        from mtg_deck_builder.card_database import card_fills_role
        greaves = _card("Greaves-ish", types="Artifact",
                        text="Equipped creature has hexproof and haste. "
                             "Equip {0}")
        anthem = _card("Anthem-ish", types="Enchantment",
                       text="Creatures you control have indestructible.")
        assert card_fills_role(greaves, "protection")
        assert card_fills_role(anthem, "protection")

    def test_removed_roles_are_gone(self):
        # flex/tutor were deliberately removed: unknown roles never match.
        from mtg_deck_builder.card_database import ROLE_PATTERNS, card_fills_role
        assert "flex" not in ROLE_PATTERNS
        assert "tutor" not in ROLE_PATTERNS
        clone = _card("Clone-ish", text="becomes a copy of target creature.")
        assert not card_fills_role(clone, "flex")


# ----------------------------------------------------------------------
# Builder integration: pool filter + combo partition
# ----------------------------------------------------------------------

class TestBuilderIntegration:
    def _builder(self, test_csv_path, bracket):
        from mtg_deck_builder.deck_builder import DeckBuilder, CandidatePool
        from mtg_deck_builder.llm_engine import LLMConfig
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42, bracket=bracket, generations=2,
                          population_size=8, combo_mode="llm")
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        b._analysis = _analysis()
        b._candidates = CandidatePool(
            synergy=[b.db.get_by_name("Soul Warden")],
        )
        return b

    def test_combo_partition_by_bracket(self, test_csv_path):
        from mtg_deck_builder.models import ComboReport
        b = self._builder(test_csv_path, bracket=2)

        combos = [
            Combo(cards=["Heliod, Sun-Crowned", "Soul Warden"], payoff=90),
            Combo(cards=["Heliod, Sun-Crowned", "Soul Warden",
                         "Soul's Attendant"], payoff=80),
        ]

        class _Stub:
            def detect(self, analysis, pool, edhrec_fallback=None, **kw):
                return ComboReport(combos=list(combos))
        b._get_combo_detector = lambda: _Stub()
        b._phase_detect_combos()
        assert len(b._banned_combos) == 1   # the 2-card pair
        assert len(b._reward_combos) == 1   # the 3-card combo stays rewarded

    def test_pool_filter_removes_gc_at_b2(self, test_csv_path):
        b = self._builder(test_csv_path, bracket=2)
        # Teferi's Protection and Smothering Tithe are GCs in the test CSV.
        b._phase_generate_pools()
        b._phase_llm_filtering()
        b._phase_synergy_scoring()
        result = b._phase_optimization()
        pool_names = {c.name for c in b._ga_candidate_pool}
        assert "Teferi's Protection" not in pool_names
        assert "Smothering Tithe" not in pool_names
        assert "Sol Ring" in pool_names  # not a GC
        deck_names = {c.name for c in result.best_deck.cards}
        assert "Teferi's Protection" not in deck_names

    def test_locked_gc_exempt_from_filter(self, test_csv_path):
        b = self._builder(test_csv_path, bracket=2)
        b.config.locked_cards = ["Teferi's Protection"]
        b._phase_generate_pools()
        b._phase_llm_filtering()
        b._phase_synergy_scoring()
        b._phase_optimization()
        assert "Teferi's Protection" in {c.name for c in b._ga_candidate_pool}
