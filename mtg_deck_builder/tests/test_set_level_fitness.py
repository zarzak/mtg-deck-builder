"""
Tests for v0.9.14 set-level fitness + LLM refinement.

The GA's per-card averages can't express set-level deck properties. Four
additions close the gap:
  1. Post-GA LLM refinement loop (holistic swaps on the assembled 99).
  2. Consistency dimension: core-effect-class redundancy with diminishing
     returns per copy.
  3. Removal sub-type roles (creature vs artifact/enchantment spread).
  4. Quality-weighted role coverage (weak fillers count < 1 toward targets).
"""

import pytest

from mtg_deck_builder.card_database import card_fills_role
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.deck_evaluator import DeckEvaluator
from mtg_deck_builder.llm_engine import LLMEngine, LLMConfig
from mtg_deck_builder.models import (
    Card, Deck, BuildConfig, CommanderAnalysis, OptimizationResult,
)


def _card(name, text="", types="Creature", mv=2, ci="W", is_land=False,
          supertypes="") -> Card:
    return Card(
        name=name, mana_cost=f"{{{mv}}}", mana_value=mv,
        card_type=("Land" if is_land else "Creature"), text=text,
        color_identity=ci, colors=ci,
        power="2" if not is_land else None,
        toughness="2" if not is_land else None,
        loyalty="", defense="",
        types="Land" if is_land else types, subtypes="",
        supertypes=supertypes, keywords="",
        layout="normal", legalities="commander:legal",
    )


def _analysis(classes=None) -> CommanderAnalysis:
    return CommanderAnalysis(
        name="Test Commander", color_identity="W", key_mechanics=[],
        build_around_text="gain life", evaluation_notes="",
        category_queries={}, synergy_keywords=["gain life"],
        core_effect_classes=classes or [],
    )


# ----------------------------------------------------------------------
# Item 2: consistency dimension
# ----------------------------------------------------------------------

class TestConsistency:
    CLASSES = [
        {"name": "lifegain trigger", "min_count": 4},
        {"name": "lifegain payoff", "min_count": 2},
    ]

    def _evaluator(self, tags, classes=None) -> DeckEvaluator:
        cfg = BuildConfig(commander_name="X")
        cls = self.CLASSES if classes is None else classes
        return DeckEvaluator(cfg, _analysis(cls), card_effect_classes=tags)

    def _deck(self, names) -> Deck:
        return Deck(commander=_card("Cmd"), cards=[_card(n) for n in names])

    def test_full_coverage_scores_100(self):
        tags = {f"T{i}": "lifegain trigger" for i in range(4)}
        tags.update({f"P{i}": "lifegain payoff" for i in range(2)})
        ev = self._evaluator(tags)
        deck = self._deck(list(tags.keys()) + ["Filler"])
        assert ev._score_consistency(deck) == pytest.approx(100.0)

    def test_empty_class_scores_zero_for_that_class(self):
        # All triggers, zero payoffs -> payoff class contributes 0.
        tags = {f"T{i}": "lifegain trigger" for i in range(4)}
        ev = self._evaluator(tags)
        deck = self._deck(list(tags.keys()))
        # trigger class full (1.0), payoff class 0 -> mean 0.5
        assert ev._score_consistency(deck) == pytest.approx(50.0)

    def test_diminishing_returns_reward_early_copies_most(self):
        # 1 copy of a min-4 class earns MORE than a quarter of the class
        # value — the first copy is the most valuable (consistency!).
        ev1 = self._evaluator({"T0": "lifegain trigger"},
                              classes=[{"name": "lifegain trigger",
                                        "min_count": 4}])
        one = ev1._score_consistency(self._deck(["T0"]))
        assert one > 25.0
        # And the 2nd copy adds more than the 4th.
        def score(n):
            tags = {f"T{i}": "lifegain trigger" for i in range(n)}
            ev = self._evaluator(tags, classes=[{"name": "lifegain trigger",
                                                 "min_count": 4}])
            return ev._score_consistency(self._deck(list(tags.keys())))
        assert (score(2) - score(1)) > (score(4) - score(3))

    def test_copies_beyond_min_add_nothing(self):
        classes = [{"name": "lifegain trigger", "min_count": 2}]
        tags4 = {f"T{i}": "lifegain trigger" for i in range(4)}
        ev = self._evaluator(tags4, classes=classes)
        assert ev._score_consistency(self._deck(list(tags4.keys()))) == \
            pytest.approx(100.0)

    def test_no_data_scores_zero(self):
        ev = self._evaluator({}, classes=self.CLASSES)
        assert ev._score_consistency(self._deck(["A"])) == 0.0
        ev2 = self._evaluator({"A": "lifegain trigger"}, classes=[])
        assert ev2._score_consistency(self._deck(["A"])) == 0.0

    def test_weight_injected_only_with_classes(self):
        cfg = BuildConfig(commander_name="X", consistency_weight=0.12)
        with_classes = cfg.get_effective_weights(_analysis(self.CLASSES))
        without = cfg.get_effective_weights(_analysis([]))
        assert with_classes.get("consistency") == pytest.approx(0.12)
        assert "consistency" not in without

    def test_parse_effect_classes_defensive(self):
        raw = [
            {"name": "lifegain trigger", "min_count": 4},
            {"name": "  ", "min_count": 3},            # blank -> dropped
            {"min_count": 3},                            # no name -> dropped
            {"name": "payoff", "min_count": "lots"},     # bad count -> 3
            {"name": "Payoff", "min_count": 99},         # dupe (case) -> dropped
            {"name": "huge", "min_count": 99},           # clamped to 8
            "not a dict",
        ]
        out = LLMEngine._parse_effect_classes(raw)
        assert [c["name"] for c in out] == ["lifegain trigger", "payoff", "huge"]
        assert out[1]["min_count"] == 3
        assert out[2]["min_count"] == 8

    def test_scoring_pass_fills_class_sink(self, lathiel_analysis, db):
        lathiel_analysis.core_effect_classes = [
            {"name": "lifegain trigger", "min_count": 4},
        ]
        eng = LLMEngine(LLMConfig(mock_mode=True))
        eng.config.mock_mode = False
        eng._call_api = lambda *a, **k: (
            '{"scores": [{"name": "Soul Warden", "score": 88, '
            '"class": "Lifegain Trigger"}, '
            '{"name": "Sun Titan", "score": 40, "class": "made-up class"}]}'
        )
        cards = [db.get_by_name("Soul Warden"), db.get_by_name("Sun Titan")]
        sink: dict = {}
        eng._score_synergy_single(lathiel_analysis, cards, class_sink=sink)
        # Valid class canonicalized (case-insensitive); unknown class dropped.
        assert sink == {"Soul Warden": "lifegain trigger"}


# ----------------------------------------------------------------------
# Item 3: removal sub-type roles
# ----------------------------------------------------------------------

class TestRemovalSpread:
    def test_creature_removal_classified(self):
        c = _card("Swords-ish", text="Exile target creature.",
                  types="Instant")
        assert card_fills_role(c, "removal_creature") is True
        assert card_fills_role(c, "removal_artifact") is False

    def test_artifact_removal_classified(self):
        c = _card("Molder-ish",
                  text="Destroy target artifact or enchantment.",
                  types="Instant")
        assert card_fills_role(c, "removal_artifact") is True
        assert card_fills_role(c, "removal_creature") is False

    def test_flexible_removal_fills_both(self):
        c = _card("Gift-ish", text="Destroy target permanent.",
                  types="Instant")
        assert card_fills_role(c, "removal_creature") is True
        assert card_fills_role(c, "removal_artifact") is True

    def test_wipes_excluded_from_subs(self):
        c = _card("Wrath-ish", text="Destroy all creatures.",
                  types="Sorcery")
        assert card_fills_role(c, "removal_creature") is False

    def test_own_creature_targeting_excluded(self):
        c = _card("Sac-ish",
                  text="Destroy target creature you control.",
                  types="Instant")
        assert card_fills_role(c, "removal_creature") is False

    def test_subs_tracked_but_no_default_targets(self):
        # Sub-types are CLASSIFIED and TRACKED (diagnostics, refinement
        # visibility) but deliberately have no default minimums — the right
        # interaction spread is strategy/meta judgment, not a hardcoded
        # constant. Users can opt into a hard floor via role_target_overrides.
        cfg = BuildConfig(commander_name="X")
        assert "removal_creature" not in cfg.role_targets
        assert "removal_artifact" not in cfg.role_targets
        assert "removal_creature" in DeckEvaluator.TRACKED_ROLES
        assert "removal_artifact" in DeckEvaluator.TRACKED_ROLES

    def test_sub_targets_opt_in_via_override(self):
        cfg = BuildConfig(
            commander_name="X",
            role_target_overrides={"removal_creature": (4, 12)},
        )
        assert cfg.get_effective_role_targets()["removal_creature"] == (4, 12)


# ----------------------------------------------------------------------
# Item 4: quality-weighted role coverage
# ----------------------------------------------------------------------

class TestQualityWeightedRoles:
    def _evaluator(self, baseline, quality=True) -> DeckEvaluator:
        cfg = BuildConfig(commander_name="X", quality_weighted_roles=quality)
        return DeckEvaluator(cfg, _analysis(),
                             baseline_power_cache=baseline)

    def test_weak_filler_counts_less_than_one(self):
        ramp = _card("Weak Ramp", text="{T}: Add {G}.", types="Creature")
        ev = self._evaluator({"Weak Ramp": 20.0})
        deck = Deck(commander=_card("Cmd"), cards=[ramp])
        assert ev._quality_weighted_role_count(deck, "ramp") == \
            pytest.approx(0.5 + 20.0 / 120.0)

    def test_power_60_counts_fully(self):
        ramp = _card("Good Ramp", text="{T}: Add {G}.", types="Creature")
        ev = self._evaluator({"Good Ramp": 60.0})
        deck = Deck(commander=_card("Cmd"), cards=[ramp])
        assert ev._quality_weighted_role_count(deck, "ramp") == \
            pytest.approx(1.0)

    def test_flag_off_uses_integer_counts(self):
        ramp = _card("Weak Ramp", text="{T}: Add {G}.", types="Creature")
        ev = self._evaluator({"Weak Ramp": 20.0}, quality=False)
        deck = Deck(commander=_card("Cmd"), cards=[ramp] * 10)
        # With quality weighting off, coverage uses plain counts.
        cov_off = ev._score_role_coverage(deck)
        ev_on = self._evaluator({"Weak Ramp": 20.0}, quality=True)
        cov_on = ev_on._score_role_coverage(deck)
        assert cov_on < cov_off  # weak fillers hurt when weighting is on


# ----------------------------------------------------------------------
# v0.9.15b: role power bypass
# ----------------------------------------------------------------------

class TestRolePowerBypass:
    def _builder(self, test_csv_path, bypass=15):
        from mtg_deck_builder.deck_builder import DeckBuilder
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42, candidates_per_category=5,
                          synergy_engine_target=0, role_power_bypass=bypass)
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        b._analysis = _analysis()
        return b

    def test_top_power_card_rescued_from_tournament_cut(self, test_csv_path):
        # Regression (real cEDH run): the selection funnel cut Llanowar
        # Elves (power 78) and Force of Negation (88) before the GA ever
        # saw them. The bucket's top-N by cached power must join the pool
        # ADDITIVELY even when the LLM selection skips them.
        b = self._builder(test_csv_path)
        b._phase_generate_pools()
        ramp_pool = b._candidates.ramp
        assert len(ramp_pool) > 5  # tournament will cut
        # Give one card an overwhelming power score and force the mock
        # selector to skip it by making its heuristic synergy zero.
        target = ramp_pool[0]
        b._card_power_scores = {target.name: 99.0}
        b.llm.select_cards = lambda *a, **k: [
            c.name for c in ramp_pool[1:6]
        ]
        b._phase_llm_filtering()
        names = {c.name for c in b._candidates.ramp}
        assert target.name in names            # rescued
        assert len(b._candidates.ramp) == 6    # 5 picked + 1 bypass (additive)

    def test_bypass_zero_disables(self, test_csv_path):
        b = self._builder(test_csv_path, bypass=0)
        b._phase_generate_pools()
        ramp_pool = b._candidates.ramp
        target = ramp_pool[0]
        b._card_power_scores = {target.name: 99.0}
        b.llm.select_cards = lambda *a, **k: [c.name for c in ramp_pool[1:6]]
        b._phase_llm_filtering()
        assert target.name not in {c.name for c in b._candidates.ramp}

    def test_no_power_scores_noop(self, test_csv_path):
        b = self._builder(test_csv_path)
        b._phase_generate_pools()
        ramp_pool = b._candidates.ramp
        b._card_power_scores = {}
        b.llm.select_cards = lambda *a, **k: [c.name for c in ramp_pool[1:6]]
        b._phase_llm_filtering()
        assert len(b._candidates.ramp) == 5  # nothing rescued

    def test_global_cache_rescues_outside_recall_scope(self, test_csv_path):
        # v0.9.19 regression (Jodah run): Sol Ring (global-cache power 98)
        # was funnel-cut from the ramp bucket but invisible to the bypass,
        # because _card_power_scores only covers the recall union while role
        # buckets draw from the whole DB. The bypass must rank by the GLOBAL
        # cache so bucket cards outside recall still get their safety net.
        b = self._builder(test_csv_path)
        b._phase_generate_pools()
        ramp_pool = b._candidates.ramp
        target = ramp_pool[0]
        b._card_power_scores = {}  # not in the recall union

        class _FakeScorer:
            @staticmethod
            def cached_scores():
                return {target.name: 99.0}

        b._get_card_power_scorer = lambda: _FakeScorer()
        b.llm.select_cards = lambda *a, **k: [c.name for c in ramp_pool[1:6]]
        b._phase_llm_filtering()
        assert target.name in {c.name for c in b._candidates.ramp}  # rescued


# ----------------------------------------------------------------------
# Item 1: LLM refinement loop
# ----------------------------------------------------------------------

class TestRefinement:
    def _builder(self, test_csv_path, rounds=1, locked=None) -> DeckBuilder:
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42, refine_iterations=rounds,
                          locked_cards=locked or [])
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        b._analysis = _analysis()
        # Refinement requires a real (non-mock) engine flag; the LLM call
        # itself is stubbed per-test.
        b.llm.config.mock_mode = False
        return b

    def _result(self, b, deck_names) -> OptimizationResult:
        forest = b.db.get_by_name("Forest")
        cards = [b.db.get_by_name(n) for n in deck_names]
        assert all(c is not None for c in cards), "fixture card missing"
        while len(cards) < 99:
            cards.append(forest)
        deck = Deck(commander=b._commander, cards=cards[:99])
        return OptimizationResult(
            best_deck=deck, final_score=50.0, generations_run=1,
            score_history=[], diversity_history=[], runtime_seconds=0.0,
            config=b.config,
        )

    def test_valid_swap_applied_and_logged(self, test_csv_path):
        b = self._builder(test_csv_path)
        result = self._result(b, ["Soul Warden", "Sun Titan"])
        heliod = b.db.get_by_name("Heliod, Sun-Crowned")
        b._ga_candidate_pool = list(result.best_deck.cards) + [heliod]
        b.llm.refine_deck_swaps = lambda *a, **k: [
            {"out": "Sun Titan", "in": "Heliod, Sun-Crowned", "reason": "payoff"},
        ]
        b._phase_llm_refinement(result)
        names = {c.name for c in result.best_deck.cards}
        assert "Heliod, Sun-Crowned" in names
        assert "Sun Titan" not in names
        assert len(result.refinement_log) == 1
        assert result.refinement_log[0]["out"] == "Sun Titan"
        # Final score was honestly re-evaluated (not the GA's 50.0 stub).
        assert result.final_score != 50.0

    def test_land_parity_enforced(self, test_csv_path):
        b = self._builder(test_csv_path)
        result = self._result(b, ["Soul Warden"])
        heliod = b.db.get_by_name("Heliod, Sun-Crowned")
        b._ga_candidate_pool = list(result.best_deck.cards) + [heliod]
        # Illegal: swap a land (Forest) for a nonland.
        b.llm.refine_deck_swaps = lambda *a, **k: [
            {"out": "Forest", "in": "Heliod, Sun-Crowned", "reason": "x"},
        ]
        b._phase_llm_refinement(result)
        assert "Heliod, Sun-Crowned" not in {c.name for c in result.best_deck.cards}
        assert result.refinement_log == []

    def test_duplicate_nonbasic_rejected(self, test_csv_path):
        b = self._builder(test_csv_path)
        result = self._result(b, ["Soul Warden", "Sun Titan"])
        b._ga_candidate_pool = list(result.best_deck.cards)
        # Illegal: Soul Warden is already in the deck.
        b.llm.refine_deck_swaps = lambda *a, **k: [
            {"out": "Sun Titan", "in": "Soul Warden", "reason": "x"},
        ]
        b._phase_llm_refinement(result)
        names = [c.name for c in result.best_deck.cards]
        assert names.count("Soul Warden") == 1
        assert result.refinement_log == []

    def test_role_floor_guard_blocks_starving_swaps(self, test_csv_path):
        # v0.9.14b regression (real run): refinement isn't fitness-gated, so
        # it traded ramp below its minimum (10 -> 6) for engine pieces. The
        # mechanical guard must reject swaps that drop a role below its
        # floor, while allowing like-for-like swaps within the role.
        b = self._builder(test_csv_path)
        b.config.role_target_overrides = {"ramp": (1, 14)}
        result = self._result(b, ["Sol Ring", "Soul Warden"])  # exactly 1 ramp
        heliod = b.db.get_by_name("Heliod, Sun-Crowned")
        rampant = b.db.get_by_name("Rampant Growth")
        b._ga_candidate_pool = list(result.best_deck.cards) + [heliod, rampant]
        # Swap 1 (ramp -> non-ramp) must be BLOCKED: ramp would fall to 0 < 1.
        # Swap 2 (ramp -> ramp) must be ALLOWED.
        b.llm.refine_deck_swaps = lambda *a, **k: [
            {"out": "Sol Ring", "in": "Heliod, Sun-Crowned", "reason": "x"},
            {"out": "Sol Ring", "in": "Rampant Growth", "reason": "upgrade"},
        ]
        b._phase_llm_refinement(result)
        names = {c.name for c in result.best_deck.cards}
        assert "Heliod, Sun-Crowned" not in names   # floor-guarded
        assert "Rampant Growth" in names            # like-for-like allowed
        assert "Sol Ring" not in names
        assert len(result.refinement_log) == 1

    def test_role_status_passed_to_llm(self, test_csv_path):
        b = self._builder(test_csv_path)
        result = self._result(b, ["Sol Ring"])
        # Pool must contain at least one unused card or the loop exits
        # before the LLM is consulted.
        b._ga_candidate_pool = list(result.best_deck.cards) + [
            b.db.get_by_name("Heliod, Sun-Crowned"),
        ]
        captured = {}
        def spy(*a, **k):
            captured.update(k)
            return []
        b.llm.refine_deck_swaps = spy
        b._phase_llm_refinement(result)
        assert "ramp:" in captured.get("role_status", "")
        assert "(min 10)" in captured["role_status"]
        # v0.9.26: a role at/below its floor carries an explicit AT FLOOR
        # marker so the LLM stops proposing swaps the guard will reject.
        assert "AT FLOOR" in captured["role_status"]
        # v0.9.28: per-card role tags — the LLM must see WHICH cards hold a
        # floor (deck cards AND alternatives).
        roles = captured.get("card_roles", {})
        assert "ramp" in roles.get("Sol Ring", [])

    def test_disabled_and_mock_are_noops(self, test_csv_path):
        b = self._builder(test_csv_path, rounds=0)
        result = self._result(b, ["Soul Warden"])
        b._ga_candidate_pool = list(result.best_deck.cards)
        b._phase_llm_refinement(result)
        assert result.refinement_log == []
        b2 = self._builder(test_csv_path, rounds=2)
        b2.llm.config.mock_mode = True
        result2 = self._result(b2, ["Soul Warden"])
        b2._ga_candidate_pool = list(result2.best_deck.cards)
        b2._phase_llm_refinement(result2)
        assert result2.refinement_log == []

    def test_refine_deck_swaps_parses_and_filters(self, lathiel_analysis, db):
        eng = LLMEngine(LLMConfig(mock_mode=True))
        eng.config.mock_mode = False
        eng._call_api = lambda *a, **k: (
            '{"swaps": ['
            '{"out": "sun titan", "in": "heliod, sun-crowned", "reason": "ok"},'
            '{"out": "Locked Card", "in": "Heliod, Sun-Crowned", "reason": "no"},'
            '{"out": "Not In Deck", "in": "Heliod, Sun-Crowned", "reason": "no"},'
            '{"out": "Sun Titan", "in": "Fake Card", "reason": "no"}'
            ']}'
        )
        deck = Deck(
            commander=db.get_by_name("Lathiel, the Bounteous Dawn"),
            cards=[db.get_by_name("Sun Titan"), db.get_by_name("Soul Warden")],
        )
        alts = [db.get_by_name("Heliod, Sun-Crowned")]
        swaps = eng.refine_deck_swaps(
            lathiel_analysis, deck, alts, locked={"Locked Card"},
        )
        # Case drift canonicalized; locked / unknown out / unknown in dropped.
        assert swaps == [{"out": "Sun Titan", "in": "Heliod, Sun-Crowned",
                          "reason": "ok"}]
