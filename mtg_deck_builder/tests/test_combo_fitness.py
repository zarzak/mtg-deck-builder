"""
Tests for v0.9.8 interaction-aware combo fitness + the Leak A on-ramp.

Covers:
  - _score_combos: full credit on assembly, the graded near-complete gradient
    (incl. the extra step for 4+ card combos), redundancy for multiple
    wincons, partial cap, and the 0-100 cap;
  - combo weight injection in get_effective_weights (power-level scaled);
  - DeckScores.total includes the combo term;
  - _apply_onramp guarantees engine/combo cards into the GA pool.
"""

import pytest

from mtg_deck_builder.deck_evaluator import DeckEvaluator
from mtg_deck_builder.deck_builder import DeckBuilder, CandidatePool
from mtg_deck_builder.html_report import _render_combos_section
from mtg_deck_builder.models import (
    Card, Deck, DeckScores, Combo, ComboReport, BuildConfig, CommanderAnalysis,
)
from mtg_deck_builder.llm_engine import LLMConfig


def _card(name: str, ci="W") -> Card:
    return Card(
        name=name, mana_cost="{1}", mana_value=1,
        card_type="Creature", text=f"text {name}",
        color_identity=ci, colors=ci,
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _evaluator(combos) -> DeckEvaluator:
    cfg = BuildConfig(commander_name="X")
    an = CommanderAnalysis(
        name="X", color_identity="W", key_mechanics=[], build_around_text="",
        evaluation_notes="", category_queries={}, synergy_keywords=[],
    )
    return DeckEvaluator(cfg, an, combos=combos)


def _deck(card_names) -> Deck:
    return Deck(commander=_card("Cmd"), cards=[_card(n) for n in card_names])


# ----------------------------------------------------------------------
# _score_combos
# ----------------------------------------------------------------------

class TestScoreCombos:
    def test_no_combos_zero(self):
        assert _evaluator([])._score_combos(_deck(["A", "B"])) == 0.0

    def test_assembled_two_card_full_payoff(self):
        ev = _evaluator([Combo(cards=["A", "B"], payoff=90)])
        assert ev._score_combos(_deck(["A", "B", "C"])) == 90.0

    def test_two_card_one_away_is_small_gradient(self):
        ev = _evaluator([Combo(cards=["A", "B"], payoff=90)])
        # k=1, n=2 -> 90 * 0.15 = 13.5
        assert ev._score_combos(_deck(["A", "Z"])) == pytest.approx(13.5)

    def test_three_card_gradient(self):
        combo = Combo(cards=["A", "B", "C"], payoff=80)
        ev = _evaluator([combo])
        assert ev._score_combos(_deck(["A", "B", "C"])) == 80.0          # full
        assert ev._score_combos(_deck(["A", "B"])) == pytest.approx(12.0)  # 2/3 -> *0.15
        assert ev._score_combos(_deck(["A"])) == 0.0                       # 1/3 -> nothing

    def test_four_card_has_two_step_ramp(self):
        combo = Combo(cards=["A", "B", "C", "D"], payoff=100)
        ev = _evaluator([combo])
        assert ev._score_combos(_deck(["A", "B", "C", "D"])) == 100.0           # 4/4
        assert ev._score_combos(_deck(["A", "B", "C"])) == pytest.approx(15.0)  # 3/4 -> *0.15
        assert ev._score_combos(_deck(["A", "B"])) == pytest.approx(5.0)        # 2/4 -> *0.05
        assert ev._score_combos(_deck(["A"])) == 0.0                            # 1/4 -> nothing

    def test_redundancy_for_multiple_completed(self):
        ev = _evaluator([
            Combo(cards=["A", "B"], payoff=60),
            Combo(cards=["C", "D"], payoff=40),
        ])
        # best 60 + 0.25*40 = 70
        assert ev._score_combos(_deck(["A", "B", "C", "D"])) == pytest.approx(70.0)

    def test_near_uses_best_single_not_hub_sum(self):
        # Five near-complete combos sharing nothing: the near term is the BEST
        # single (100*0.15 = 15), NOT the sum — so a hub card in many combos
        # can't saturate the dimension from partials alone.
        combos = [Combo(cards=[f"P{i}", f"Q{i}"], payoff=100) for i in range(5)]
        ev = _evaluator(combos)
        deck = _deck([f"P{i}" for i in range(5)])  # one piece of each
        assert ev._score_combos(deck) == pytest.approx(15.0)

    def test_commander_counts_as_present(self):
        # "X + Commander" completes when X is in the deck — the commander is a
        # permanent combo piece every game.
        ev = _evaluator([Combo(cards=["Branching Evolution", "Cmd"], payoff=90)])
        full = Deck(commander=_card("Cmd"), cards=[_card("Branching Evolution")])
        assert ev._score_combos(full) == 90.0
        # Without the partner it's only the commander -> 1/2 -> near gradient.
        empty = Deck(commander=_card("Cmd"), cards=[_card("Other")])
        assert ev._score_combos(empty) == pytest.approx(13.5)

    def test_capped_at_100(self):
        ev = _evaluator([
            Combo(cards=["A", "B"], payoff=100),
            Combo(cards=["C", "D"], payoff=100),
            Combo(cards=["E", "F"], payoff=100),
        ])
        assert ev._score_combos(_deck(["A", "B", "C", "D", "E", "F"])) == 100.0

    def test_completing_best_combo_gains_even_at_saturation(self):
        # v0.9.13 regression (real Lathiel run): a rich synergy web (many
        # assembled pairs) used to pin the score at 100, so completing the
        # marquee 95-payoff combo added ZERO fitness — the GA had no reason
        # to pick up the last piece (and the near-credit for that very combo
        # helped saturate the score). The headroom squeeze must keep a
        # strict gradient: assembling the better combo scores strictly
        # higher than sitting one piece away from it.
        web = [Combo(cards=[f"W{i}", f"V{i}"], payoff=80) for i in range(10)]
        marquee = Combo(cards=["Spike", "Heliod"], payoff=95)
        ev = _evaluator(web + [marquee])
        web_cards = [c for i in range(10) for c in (f"W{i}", f"V{i}")]
        near = ev._score_combos(_deck(web_cards + ["Spike"]))
        done = ev._score_combos(_deck(web_cards + ["Spike", "Heliod"]))
        assert done > near
        assert done <= 100.0

    def test_combo_score_flows_into_evaluate(self):
        ev = _evaluator([Combo(cards=["A", "B"], payoff=90)])
        scores = ev.evaluate(_deck(["A", "B"] + [f"x{i}" for i in range(97)]))
        assert scores.combo == 90.0


# ----------------------------------------------------------------------
# Weight injection + total
# ----------------------------------------------------------------------

class TestComboWeight:
    def test_weight_absent_when_off(self):
        w = BuildConfig(commander_name="X", combo_mode="off").get_effective_weights()
        assert "combo" not in w

    def test_weight_present_when_on(self):
        w = BuildConfig(commander_name="X", combo_mode="llm",
                        power_level=7).get_effective_weights()
        assert w["combo"] == pytest.approx(0.12)

    def test_weight_scales_with_bracket(self):
        # v0.9.15: brackets 4-5 chase combos harder (x1.5); bracket 3 is
        # neutral; at brackets 1-2 the combo dimension is NOT injected at
        # all — the official rules ban two-card combos there and the casual
        # posture means combo assembly must not be chased (detection still
        # runs for the compliance audit + penalty).
        hi = BuildConfig(commander_name="X", combo_mode="llm",
                         bracket=4).get_effective_weights()["combo"]
        mid = BuildConfig(commander_name="X", combo_mode="llm",
                          bracket=3).get_effective_weights()["combo"]
        lo = BuildConfig(commander_name="X", combo_mode="llm",
                         bracket=2).get_effective_weights()
        assert hi == pytest.approx(0.18)   # 0.12 * 1.5
        assert mid == pytest.approx(0.12)
        assert "combo" not in lo

    def test_total_includes_combo(self):
        s = DeckScores(synergy=0, combo=50)
        # v0.9.25: total is normalized by the active weight sum — defaults
        # (1.00) + combo 0.12 = 1.12; weighted sum = 50 * 0.12 = 6.0.
        assert s.total({"combo": 0.12}) == pytest.approx(6.0 / 1.12)


# ----------------------------------------------------------------------
# Leak A on-ramp
# ----------------------------------------------------------------------

class TestOnramp:
    def _builder(self, test_csv_path) -> DeckBuilder:
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42)
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        return b

    def test_onramp_adds_missing_card(self, test_csv_path):
        b = self._builder(test_csv_path)
        # Find a real DB card not yet in an empty pool, guarantee it.
        target = "Plains"  # exists in the test DB
        assert b.db.get_by_name(target) is not None
        b._onramp_names = {target}
        filtered = CandidatePool()
        b._apply_onramp(filtered)
        assert target in {c.name for c in filtered.all_cards()}

    def test_onramp_noop_when_empty(self, test_csv_path):
        b = self._builder(test_csv_path)
        b._onramp_names = set()
        filtered = CandidatePool()
        filtered.ramp = [_card("Existing")]
        b._apply_onramp(filtered)
        assert {c.name for c in filtered.all_cards()} == {"Existing"}

    def test_onramp_skips_already_present(self, test_csv_path):
        b = self._builder(test_csv_path)
        # "Plains" already in a role bucket -> not duplicated by the on-ramp.
        plains = b.db.get_by_name("Plains")
        b._onramp_names = {"Plains"}
        filtered = CandidatePool()
        filtered.lands = [plains]
        b._apply_onramp(filtered)
        names = [c.name for c in filtered.all_cards()]
        assert names.count("Plains") == 1


# ----------------------------------------------------------------------
# Combo pruning + budget enforcement on late pool additions (v0.9.13)
# ----------------------------------------------------------------------

class TestComboPruning:
    def _builder(self, test_csv_path) -> DeckBuilder:
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42, combo_mode="llm")
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        b._analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn", color_identity="G,W",
            key_mechanics=[], build_around_text="lifegain",
            evaluation_notes="", category_queries={}, synergy_keywords=[],
        )
        # Non-empty synergy pool so the detect path actually runs.
        b._candidates = CandidatePool(
            synergy=[b.db.get_by_name("Soul Warden")],
        )
        return b

    def _run_with_combos(self, b, combos, missing=None):
        from mtg_deck_builder.models import ComboReport

        class _Stub:
            def detect(self, analysis, pool, edhrec_fallback=None, **kw):
                return ComboReport(combos=list(combos),
                                   missing_pieces=list(missing or []))
        b._get_combo_detector = lambda: _Stub()
        b._phase_detect_combos()
        return b._combo_report

    def test_unbuildable_combos_pruned(self, test_csv_path):
        # A combo naming a card that isn't in the DB (hallucinated) or is
        # off-color can never be assembled; leaving it in pays the GA
        # near-complete partial credit for hoarding pieces of an impossible
        # combo. It must be pruned. The commander itself always counts.
        b = self._builder(test_csv_path)
        report = self._run_with_combos(b, [
            Combo(cards=["Heliod, Sun-Crowned", "Soul Warden"], payoff=80),
            Combo(cards=["Heliod, Sun-Crowned", "Totally Fake Card"], payoff=95),
            Combo(cards=["Heliod, Sun-Crowned", "Phyrexian Arena"], payoff=90),  # B, off-color
            Combo(cards=["Lathiel, the Bounteous Dawn", "Soul's Attendant"], payoff=70),
        ])
        kept = [frozenset(c.cards) for c in report.combos]
        assert frozenset(["Heliod, Sun-Crowned", "Soul Warden"]) in kept
        assert frozenset(["Lathiel, the Bounteous Dawn", "Soul's Attendant"]) in kept
        assert frozenset(["Heliod, Sun-Crowned", "Totally Fake Card"]) not in kept
        assert frozenset(["Heliod, Sun-Crowned", "Phyrexian Arena"]) not in kept

    def test_missing_pieces_respect_budget(self, test_csv_path):
        # Missing-piece recall runs AFTER the budget phase; with a budget cap
        # active it must not smuggle in expensive cards.
        b = self._builder(test_csv_path)
        b.config.budget_max_per_card = 5.0

        class _FakePrices:
            def get_price(self, name):
                return {"Heliod, Sun-Crowned": 30.0,
                        "Soul's Attendant": 1.0}.get(name)
        b._price_source = _FakePrices()

        self._run_with_combos(
            b,
            [Combo(cards=["Soul Warden", "Heliod, Sun-Crowned"], payoff=80)],
            missing=["Heliod, Sun-Crowned", "Soul's Attendant"],
        )
        pool = {c.name for c in b._candidates.synergy}
        assert "Soul's Attendant" in pool        # within budget -> added
        assert "Heliod, Sun-Crowned" not in pool  # over budget -> kept out


class TestOnrampBudget:
    def test_onramp_respects_budget(self, test_csv_path):
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42, budget_max_per_card=5.0)
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")

        class _FakePrices:
            def get_price(self, name):
                return {"Heliod, Sun-Crowned": 30.0, "Soul Warden": 1.0}.get(name)
        b._price_source = _FakePrices()

        b._onramp_names = {"Heliod, Sun-Crowned", "Soul Warden"}
        filtered = CandidatePool()
        b._apply_onramp(filtered)
        names = {c.name for c in filtered.all_cards()}
        assert "Soul Warden" in names
        assert "Heliod, Sun-Crowned" not in names

    def test_onramp_unfiltered_without_budget(self, test_csv_path):
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42)  # no budget cap
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        b._onramp_names = {"Heliod, Sun-Crowned", "Soul Warden"}
        filtered = CandidatePool()
        b._apply_onramp(filtered)
        names = {c.name for c in filtered.all_cards()}
        assert {"Heliod, Sun-Crowned", "Soul Warden"} <= names


# ----------------------------------------------------------------------
# Report section
# ----------------------------------------------------------------------

class TestComboReportSection:
    def test_empty_when_no_combos(self):
        assert _render_combos_section(None, {"A"}) == ""
        assert _render_combos_section([], {"A"}) == ""

    def test_lists_assembled_combo(self):
        combos = [Combo(cards=["Spike Feeder", "Heliod"], payoff=95,
                        result="infinite life")]
        html = _render_combos_section(combos, {"Spike Feeder", "Heliod", "X"})
        assert "Assembled" in html
        assert "Spike Feeder + Heliod" in html
        assert "infinite life" in html

    def test_lists_one_piece_away(self):
        combos = [Combo(cards=["Spike Feeder", "Heliod"], payoff=95)]
        html = _render_combos_section(combos, {"Spike Feeder"})
        assert "One piece away" in html
        assert "Heliod" in html  # the missing piece is shown

    def test_no_assembly_message(self):
        combos = [Combo(cards=["A", "B", "C"], payoff=80)]
        html = _render_combos_section(combos, {"A"})  # only 1/3
        assert "No detected combos were assembled" in html


# ----------------------------------------------------------------------
# Engine boost
# ----------------------------------------------------------------------

class TestEngineBoost:
    def _builder(self, test_csv_path, mode="floor", floor=80.0) -> DeckBuilder:
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42, engine_boost_mode=mode,
                          engine_boost_floor=floor)
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._combo_report = ComboReport(engines={"Soul Warden": "n", "Heliod": "n"})
        return b

    def test_floor_lifts_low_engine(self, test_csv_path):
        b = self._builder(test_csv_path, mode="floor", floor=80.0)
        syn = {"Soul Warden": 50.0, "Heliod": 90.0, "Other": 50.0}
        b._apply_engine_boost(syn, baseline={})
        assert syn["Soul Warden"] == 80.0   # lifted to floor
        assert syn["Heliod"] == 90.0        # already above floor -> unchanged
        assert syn["Other"] == 50.0         # not an engine -> untouched

    def test_power_mode_uses_card_power(self, test_csv_path):
        b = self._builder(test_csv_path, mode="power")
        syn = {"Soul Warden": 50.0, "Heliod": 90.0}
        base = {"Soul Warden": 62.0, "Heliod": 78.0}
        b._apply_engine_boost(syn, base)
        assert syn["Soul Warden"] == 62.0   # floored at its own power
        assert syn["Heliod"] == 90.0        # power 78 < current 90 -> unchanged

    def test_power_mode_falls_back_to_floor_without_power_scores(self, test_csv_path):
        # Regression: with card_power_mode off the baseline dict is empty, and
        # "power" mode used to silently boost nothing (target 0.0). It must
        # fall back to the flat floor so the default config still lifts engines.
        b = self._builder(test_csv_path, mode="power", floor=80.0)
        syn = {"Soul Warden": 50.0}
        b._apply_engine_boost(syn, baseline={})
        assert syn["Soul Warden"] == 80.0

    def test_power_mode_genuine_low_power_stands(self, test_csv_path):
        # A card WITH a (low) power score keeps quality scaling — the fallback
        # only applies when the score is missing entirely.
        b = self._builder(test_csv_path, mode="power", floor=80.0)
        syn = {"Soul Warden": 50.0}
        b._apply_engine_boost(syn, baseline={"Soul Warden": 55.0})
        assert syn["Soul Warden"] == 55.0

    def test_off_mode_no_change(self, test_csv_path):
        b = self._builder(test_csv_path, mode="off")
        syn = {"Soul Warden": 50.0}
        b._apply_engine_boost(syn, baseline={})
        assert syn["Soul Warden"] == 50.0

    def test_only_raises_never_lowers(self, test_csv_path):
        b = self._builder(test_csv_path, mode="floor", floor=80.0)
        syn = {"Soul Warden": 95.0}  # already strong
        b._apply_engine_boost(syn, baseline={})
        assert syn["Soul Warden"] == 95.0

    def test_noop_without_combo_report(self, test_csv_path):
        b = self._builder(test_csv_path, mode="floor")
        b._combo_report = None
        syn = {"Soul Warden": 50.0}
        b._apply_engine_boost(syn, baseline={})
        assert syn["Soul Warden"] == 50.0
