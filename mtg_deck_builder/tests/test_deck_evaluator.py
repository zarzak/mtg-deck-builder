"""Tests for DeckEvaluator."""

import pytest
from mtg_deck_builder.models import Deck, BuildConfig
from mtg_deck_builder.deck_evaluator import DeckEvaluator, FastEvaluator, is_staple


def _make_deck(commander, cards):
    """Helper to build a deck, padding with Forest if short."""
    return Deck(commander=commander, cards=cards)


def _make_99_card_deck(commander, cards, db):
    """Build a 99-card deck padded with Forest."""
    forest = db.get_by_name("Forest")
    padded = list(cards)
    while len(padded) < 99:
        padded.append(forest)
    return Deck(commander=commander, cards=padded[:99])


class TestDeckEvaluatorSynergy:
    def test_uses_synergy_cache(self, lathiel, lathiel_analysis, db):
        """If a card is in synergy_cache, that value is used (not heuristic)."""
        sol_ring = db.get_by_name("Sol Ring")
        cache = {"Sol Ring": 95.0}  # Unreasonably high for testing
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis, synergy_cache=cache)
        assert evaluator._get_card_synergy(sol_ring) == 95.0

    def test_heuristic_fallback(self, lathiel, lathiel_analysis, db):
        """Cards not in cache get heuristic score."""
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis, synergy_cache={})
        # Soul Warden triggers "gain 1 life" which is a synergy keyword
        soul_warden = db.get_by_name("Soul Warden")
        score = evaluator._get_card_synergy(soul_warden)
        assert score > 45  # Baseline is 35, should be higher due to synergy match

    def test_heuristic_scaled_to_0_100(self, lathiel, lathiel_analysis, db):
        """Heuristic scores are clamped to 0-100."""
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)

        # Score many cards; all should be in valid range
        for card in db.all_cards:
            score = evaluator._heuristic_synergy(card)
            assert 0 <= score <= 100, f"{card.name} got score {score}"


class TestDeckEvaluatorBaseline:
    def test_staples_get_high_baseline(self, lathiel, lathiel_analysis, db):
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        sol_ring = db.get_by_name("Sol Ring")
        assert evaluator._heuristic_baseline(sol_ring) >= 75

    def test_cheap_cards_valued(self, lathiel, lathiel_analysis, db):
        """1-2 mana cards should score better than 7+ mana cards (heuristic)."""
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        llanowar = db.get_by_name("Llanowar Elves")  # 1 mana
        elesh = db.get_by_name("Elesh Norn, Grand Cenobite")  # 8 mana
        assert (
            evaluator._heuristic_baseline(llanowar)
            > evaluator._heuristic_baseline(elesh)
        )

    def test_etb_tapped_land_penalized(self, lathiel, lathiel_analysis, db):
        """Lands that ETB tapped are worth less than ones that don't."""
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        scattered = db.get_by_name("Scattered Groves")  # ETB tapped
        temple_garden = db.get_by_name("Temple Garden")  # optional 2 life
        assert (
            evaluator._heuristic_baseline(scattered)
            < evaluator._heuristic_baseline(temple_garden)
        )


class TestDeckEvaluatorScoring:
    def test_mana_curve_empty_deck(self, lathiel, lathiel_analysis):
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        deck = Deck(commander=lathiel, cards=[])
        assert evaluator._score_mana_curve(deck) == 0.0

    def test_role_coverage_empty_deck(self, lathiel, lathiel_analysis):
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        deck = Deck(commander=lathiel, cards=[])
        # No roles filled = very low score
        score = evaluator._score_role_coverage(deck)
        assert score < 50

    def test_perfect_role_count(self):
        """_score_role_count should return 100 when in range."""
        # Counts within [min, max] get 100
        assert DeckEvaluator._score_role_count(10, 10, 15) == 100.0
        assert DeckEvaluator._score_role_count(12, 10, 15) == 100.0
        assert DeckEvaluator._score_role_count(15, 10, 15) == 100.0

    def test_under_target_role(self):
        """Under-target ramp scales linearly from 0 to 80."""
        assert DeckEvaluator._score_role_count(0, 10, 15) == 0.0
        assert DeckEvaluator._score_role_count(5, 10, 15) == 40.0  # 5/10 * 80
        assert abs(DeckEvaluator._score_role_count(9, 10, 15) - 72.0) < 0.1

    def test_over_target_role_penalty(self):
        """Over-target gets small penalty per excess card."""
        # 16 cards when max is 15 = 1 over = 95 points
        assert DeckEvaluator._score_role_count(16, 10, 15) == 95.0

    def test_full_evaluate(self, lathiel, lathiel_analysis, db, wg_pool):
        """Full evaluate returns valid DeckScores."""
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        deck = _make_99_card_deck(lathiel, wg_pool[:50], db)
        scores = evaluator.evaluate(deck)
        # All dimensions populated
        assert 0 <= scores.mana_curve <= 100
        assert 0 <= scores.role_coverage <= 100
        assert 0 <= scores.synergy <= 100
        assert 0 <= scores.power_level <= 100
        assert 0 <= scores.creativity <= 100
        # Role counts recorded
        assert "ramp" in scores.role_counts
        assert "land" in scores.role_counts


class TestRoleShortfallPenalty:
    """v0.9.13: penalty per card below a role's minimum target — stops the GA
    starving ramp/removal in favor of on-theme cards (a real Jasmine run
    ended at 4 ramp / 3 removal because the coverage average under-priced
    the shortfall)."""

    def _evaluator(self, lathiel, lathiel_analysis, rate) -> DeckEvaluator:
        cfg = BuildConfig(commander_name=lathiel.name,
                          role_shortfall_penalty=rate)
        return DeckEvaluator(cfg, lathiel_analysis)

    def test_penalty_counts_missing_cards(self, lathiel, lathiel_analysis):
        ev = self._evaluator(lathiel, lathiel_analysis, rate=2.0)
        # Defaults: ramp min 10, draw 10, removal 8, wipe 2, land 35.
        counts = {"ramp": 4, "draw": 10, "removal": 3, "wipe": 2, "land": 36}
        # Shortfall: ramp 6 + removal 5 = 11 -> 22.0
        assert ev._role_shortfall_penalty(counts) == pytest.approx(22.0)

    def test_no_penalty_when_targets_met(self, lathiel, lathiel_analysis):
        ev = self._evaluator(lathiel, lathiel_analysis, rate=2.0)
        counts = {"ramp": 10, "draw": 12, "removal": 8, "wipe": 2, "land": 36}
        assert ev._role_shortfall_penalty(counts) == 0.0

    def test_rate_zero_disables(self, lathiel, lathiel_analysis):
        ev = self._evaluator(lathiel, lathiel_analysis, rate=0.0)
        counts = {"ramp": 0, "draw": 0, "removal": 0, "wipe": 0, "land": 0}
        assert ev._role_shortfall_penalty(counts) == 0.0

    def test_flows_into_constraint_penalty(self, lathiel, lathiel_analysis, db):
        # A 99-card deck of mostly Forests meets the land minimum but has ~0
        # ramp/draw/removal/wipe -> shortfall (10+10+8+2)=30 -> 60.0 at rate 2.
        ev = self._evaluator(lathiel, lathiel_analysis, rate=2.0)
        deck = _make_99_card_deck(lathiel, [], db)
        scores = ev.evaluate(deck)
        assert scores.constraint_penalty >= 60.0

    def test_respects_role_target_overrides(self, lathiel, lathiel_analysis):
        cfg = BuildConfig(commander_name=lathiel.name,
                          role_shortfall_penalty=2.0,
                          role_target_overrides={"removal": (12, 15)})
        ev = DeckEvaluator(cfg, lathiel_analysis)
        counts = {"ramp": 10, "draw": 10, "removal": 8, "wipe": 2, "land": 36}
        # removal min raised to 12 -> shortfall 4 -> 8.0
        assert ev._role_shortfall_penalty(counts) == pytest.approx(8.0)


class TestEffectiveSynergy:
    def test_effective_synergy_formula(self, lathiel, lathiel_analysis, db, wg_pool):
        """Effective = baseline * base_weight + synergy * synergy_weight."""
        cfg = BuildConfig(
            commander_name=lathiel.name, synergy_weight=0.7, base_weight=0.3,
        )
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        deck = _make_99_card_deck(lathiel, wg_pool[:20], db)
        scores = evaluator.evaluate(deck)

        # Manually compute what we expect
        total_baseline = sum(evaluator._get_card_baseline(c) for c in deck.cards)
        total_synergy = sum(evaluator._get_card_synergy(c) for c in deck.cards)
        expected_eff = (
            total_baseline * 0.3 / len(deck.cards)
            + total_synergy * 0.7 / len(deck.cards)
        )
        assert abs(scores.effective_synergy - expected_eff) < 0.01


class TestBuildTelemetry:
    def test_telemetry_one_entry_per_card(self, lathiel, lathiel_analysis, db, wg_pool):
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        deck = _make_99_card_deck(lathiel, wg_pool[:50], db)
        tele = evaluator.build_telemetry(deck)
        assert len(tele) == 99

    def test_telemetry_roles_classified(self, lathiel, lathiel_analysis, db, wg_pool):
        cfg = BuildConfig(commander_name=lathiel.name)
        evaluator = DeckEvaluator(cfg, lathiel_analysis)
        deck = _make_99_card_deck(lathiel, wg_pool[:50], db)
        tele = evaluator.build_telemetry(deck)
        roles = {t.role for t in tele}
        # Should have at least lands and some other roles
        assert "land" in roles


class TestFastEvaluator:
    def test_fast_eval_invalid_deck(self, lathiel, lathiel_analysis):
        """Fast evaluator should return 0 for wrong card count."""
        cfg = BuildConfig(commander_name=lathiel.name)
        fast = FastEvaluator(cfg, lathiel_analysis)
        deck = Deck(commander=lathiel, cards=[])
        assert fast.evaluate(deck) == 0.0

    def test_fast_eval_duplicate_non_basic(self, lathiel, lathiel_analysis, db):
        """Duplicate non-basic = 0."""
        cfg = BuildConfig(commander_name=lathiel.name)
        fast = FastEvaluator(cfg, lathiel_analysis)
        sol_ring = db.get_by_name("Sol Ring")
        forest = db.get_by_name("Forest")
        deck = Deck(commander=lathiel, cards=[sol_ring, sol_ring] + [forest] * 97)
        assert fast.evaluate(deck) == 0.0

    def test_fast_eval_valid_deck(self, lathiel, lathiel_analysis, db, wg_pool):
        """Valid deck gets nonzero score."""
        cfg = BuildConfig(commander_name=lathiel.name)
        fast = FastEvaluator(cfg, lathiel_analysis)
        deck = _make_99_card_deck(lathiel, wg_pool[:50], db)
        score = fast.evaluate(deck)
        assert score > 0

    def test_fast_eval_uses_synergy_cache(self, lathiel, lathiel_analysis, db,
                                          wg_pool):
        """v0.9.12: when a synergy cache is provided, the fast evaluator scores
        a high-synergy deck above a low-synergy one (so early GA gens pursue the
        real strategy, not just keyword hits)."""
        cfg = BuildConfig(commander_name=lathiel.name)
        deck = _make_99_card_deck(lathiel, wg_pool[:50], db)
        names = [c.name for c in deck.cards]
        hi = FastEvaluator(cfg, lathiel_analysis,
                           synergy_cache={n: 95.0 for n in names})
        lo = FastEvaluator(cfg, lathiel_analysis,
                           synergy_cache={n: 5.0 for n in names})
        assert hi.evaluate(deck) > lo.evaluate(deck)


class TestIsStaple:
    def test_sol_ring_is_staple(self, db):
        assert is_staple(db.get_by_name("Sol Ring"))

    def test_grizzly_bears_not_staple(self, db):
        assert not is_staple(db.get_by_name("Grizzly Bears"))
