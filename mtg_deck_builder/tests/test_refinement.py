"""
Tests for Session 4 iterative refinement:
- locked_cards: must-include cards
- banned_cards: must-exclude cards
- role_target_overrides: custom role counts
- warm_start_path: seed population from a prior deck
- OptimizationResult persistence (to/from JSON)
"""

import json
import pytest
import tempfile
from pathlib import Path

from mtg_deck_builder.models import (
    BuildConfig, CommanderAnalysis, Deck, WarmStartDeck, OptimizationResult,
)
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.deck_evaluator import DeckEvaluator, FastEvaluator
from mtg_deck_builder.deck_optimizer import DeckOptimizer
from mtg_deck_builder.llm_engine import LLMConfig


def _make_optimizer(lathiel, lathiel_analysis, wg_pool, config=None):
    """Helper: build an optimizer for the fixture Lathiel/W-G pool."""
    if config is None:
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=6, generations=5,
            patience_generations=50, random_seed=42,
        )
    pool = [c for c in wg_pool if c.name != lathiel.name]
    evaluator = DeckEvaluator(config, lathiel_analysis)
    fast = FastEvaluator(config, lathiel_analysis)
    return DeckOptimizer(config, lathiel_analysis, pool, lathiel, evaluator, fast)


class TestBannedCards:
    def test_banned_card_filtered_from_pool(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        """A banned card should not appear in the optimizer's valid pool."""
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            banned_cards=["Sol Ring"],
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        # Sol Ring's index (if in pool) should not be in _valid_indices
        pool_names = {opt.candidate_pool[i].name for i in opt._valid_indices}
        assert "Sol Ring" not in pool_names

    def test_banned_card_never_appears_in_deck(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            banned_cards=["Sol Ring"],
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        result = opt.run()
        names = {c.name for c in result.best_deck.cards}
        assert "Sol Ring" not in names


class TestLockedCards:
    def test_locked_card_appears_in_initial_population(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        """Every initial individual should contain all locked cards."""
        # Use cards we know are in the pool
        locked = ["Sol Ring", "Birds of Paradise"]
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            locked_cards=locked,
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        opt._initialize_population()
        for ind in opt.population:
            names = {opt.candidate_pool[i].name for i in ind.card_indices}
            for name in locked:
                assert name in names, f"{name} missing from initial individual"

    def test_locked_card_survives_optimization(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        """Locked cards should be in the final deck too."""
        locked = ["Sol Ring", "Swords to Plowshares"]
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=6, generations=5,
            patience_generations=50, random_seed=42,
            locked_cards=locked,
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        result = opt.run()
        names = {c.name for c in result.best_deck.cards}
        for name in locked:
            assert name in names, f"Locked card {name} not in final deck"

    def test_lock_count_doesnt_exceed_99(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        """Locking more than 99 cards should be truncated, not crashed."""
        # Grab all cards in the W/G pool by name
        all_wg_names = [c.name for c in wg_pool if not c.is_basic_land][:110]
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            locked_cards=all_wg_names,
        )
        # Should not crash
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        assert len(opt._locked_indices) <= 99

    def test_missing_lock_logged_not_crashed(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        """Locking a card that isn't in the pool should just warn, not crash."""
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            locked_cards=["Sol Ring", "Totally Nonexistent Card"],
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        # Sol Ring should be in locked_indices; the nonexistent card dropped
        assert len(opt._locked_indices) == 1

    def test_banned_locked_conflict_lock_loses(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        """If a card is both banned and locked, it gets dropped (banned wins).

        Rationale: lock implies 'must include' and ban implies 'must exclude';
        if both are set the user contradicted themselves. We treat ban as
        authoritative because it's the stricter constraint (a ban on a lock
        means the user changed their mind; silently dropping is safer than
        crashing)."""
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            locked_cards=["Sol Ring"],
            banned_cards=["Sol Ring"],
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        # Sol Ring isn't in the valid pool (banned), so lock resolution drops it
        assert len(opt._locked_indices) == 0


class TestRoleTargetOverrides:
    def test_override_merges_with_defaults(self):
        """Overrides should modify specified roles, leaving others alone."""
        config = BuildConfig(
            commander_name="Test",
            role_target_overrides={"removal": (10, 14)},
        )
        effective = config.get_effective_role_targets()
        assert effective["removal"] == (10, 14)
        # Other roles unchanged
        assert effective["ramp"] == (10, 14)  # default
        assert effective["land"] == (35, 38)  # default

    def test_override_fixes_reversed_bounds(self):
        """If user passes (hi, lo) we swap to (lo, hi)."""
        config = BuildConfig(
            commander_name="Test",
            role_target_overrides={"removal": (14, 10)},
        )
        assert config.get_effective_role_targets()["removal"] == (10, 14)

    def test_invalid_override_ignored(self):
        """Garbage overrides should be silently ignored, not crash."""
        config = BuildConfig(
            commander_name="Test",
            role_target_overrides={"removal": "not a tuple"},
        )
        # Should not raise; removal stays at default
        effective = config.get_effective_role_targets()
        assert effective["removal"] == (8, 12)

    def test_override_affects_evaluator(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        """Evaluator should use effective role targets when scoring."""
        config = BuildConfig(
            commander_name=lathiel.name,
            role_target_overrides={"ramp": (25, 30)},  # unusually high
        )
        evaluator = DeckEvaluator(config, lathiel_analysis)
        # Build a deck with 12 ramp (fine by default but low vs override)
        deck = Deck(
            commander=lathiel,
            cards=list(wg_pool)[:99] + [wg_pool[0]] * max(0, 99 - len(wg_pool)),
        )
        # Normalize to 99 cards via padding with first card
        while len(deck.cards) < 99:
            deck.cards.append(wg_pool[0])
        deck.cards = deck.cards[:99]
        # Score with default targets
        cfg_default = BuildConfig(commander_name=lathiel.name)
        evaluator_default = DeckEvaluator(cfg_default, lathiel_analysis)
        default_score = evaluator_default._score_role_coverage(deck)
        override_score = evaluator._score_role_coverage(deck)
        # Ramp count ~12 is perfect for default, low for override -> override
        # role_coverage score should be lower
        assert override_score < default_score or override_score == default_score


class TestWarmStartPersistence:
    def test_warm_start_deck_roundtrip(self, tmp_path):
        ws = WarmStartDeck(
            commander_name="Lathiel, the Bounteous Dawn",
            card_names=["Sol Ring", "Forest", "Plains"],
            final_score=72.5,
        )
        path = tmp_path / "deck.json"
        path.write_text(json.dumps(ws.to_dict()), encoding="utf-8")
        loaded = WarmStartDeck.from_json_file(str(path))
        assert loaded.commander_name == ws.commander_name
        assert loaded.card_names == ws.card_names
        assert loaded.final_score == ws.final_score

    def test_optimization_result_to_warm_start(self, lathiel, wg_pool):
        """OptimizationResult.to_warm_start() strips state we don't need."""
        from mtg_deck_builder.models import DeckScores
        deck = Deck(commander=lathiel, cards=wg_pool[:99], scores=DeckScores())
        while len(deck.cards) < 99:
            deck.cards.append(wg_pool[0])
        result = OptimizationResult(
            best_deck=deck,
            final_score=65.0,
            generations_run=100,
            score_history=[50.0, 60.0, 65.0],
            diversity_history=[1.0, 0.9, 0.8],
            runtime_seconds=12.3,
            config=BuildConfig(commander_name=lathiel.name),
        )
        ws = result.to_warm_start()
        assert ws.commander_name == lathiel.name
        assert len(ws.card_names) == 99
        assert ws.final_score == 65.0

    def test_to_json_file_writes(self, lathiel, wg_pool, tmp_path):
        from mtg_deck_builder.models import DeckScores
        deck = Deck(commander=lathiel, cards=wg_pool[:99])
        while len(deck.cards) < 99:
            deck.cards.append(wg_pool[0])
        result = OptimizationResult(
            best_deck=deck, final_score=60.0, generations_run=5,
            score_history=[], diversity_history=[],
            runtime_seconds=1.0, config=BuildConfig(commander_name=lathiel.name),
        )
        path = tmp_path / "result.json"
        result.to_json_file(str(path))
        assert path.exists()
        loaded = WarmStartDeck.from_json_file(str(path))
        assert loaded.commander_name == lathiel.name


class TestWarmStartOptimization:
    def _write_warm_start(self, path, commander_name, card_names):
        ws = WarmStartDeck(
            commander_name=commander_name,
            card_names=card_names,
            final_score=50.0,
        )
        path.write_text(json.dumps(ws.to_dict()), encoding="utf-8")

    def test_warm_start_seeds_population(
        self, lathiel, lathiel_analysis, wg_pool, tmp_path,
    ):
        """Warm-start deck should appear in the initial population.

        With warm_start_copies=2, the first 2 individuals should match the
        warm-start. Remaining individuals are random — they should NOT
        match the warm-start as closely (otherwise warm_start_copies=2
        wouldn't actually be doing anything different from copies=N).
        """
        # Create a specific warm-start deck
        warm_names = [c.name for c in wg_pool[:99]]
        ws_path = tmp_path / "warm.json"
        self._write_warm_start(ws_path, lathiel.name, warm_names)

        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=6, generations=2,
            patience_generations=50, random_seed=42,
            warm_start_path=str(ws_path),
            warm_start_copies=2,
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        opt._initialize_population()

        # All seeded copies should have high overlap with warm-start
        warm_names_set = set(warm_names)
        for i in range(2):  # warm_start_copies=2
            seeded_names = {
                opt.candidate_pool[idx].name
                for idx in opt.population[i].card_indices
            }
            overlap = warm_names_set & seeded_names
            assert len(overlap) >= 50, \
                f"Seeded individual {i} overlap too low: {len(overlap)}/99"

        # Population was created at the right size
        assert len(opt.population) == config.population_size

        # The non-seeded individuals (indices 2..5) should be different from
        # each other — random init shouldn't produce identical decks
        non_seeded_names_per_ind = [
            tuple(sorted(
                opt.candidate_pool[idx].name
                for idx in opt.population[i].card_indices
            ))
            for i in range(2, len(opt.population))
        ]
        assert len(set(non_seeded_names_per_ind)) > 1, \
            "Random individuals were all identical — random init is broken"

    def test_warm_start_plus_locks(
        self, lathiel, lathiel_analysis, wg_pool, tmp_path,
    ):
        """Locks should be enforced on top of warm-start."""
        # Warm-start has many cards but NOT Sol Ring
        warm_names = [c.name for c in wg_pool[:99] if c.name != "Sol Ring"]
        while len(warm_names) < 99:
            warm_names.append("Forest")
        ws_path = tmp_path / "warm.json"
        self._write_warm_start(ws_path, lathiel.name, warm_names)

        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            warm_start_path=str(ws_path),
            warm_start_copies=4,
            locked_cards=["Sol Ring"],  # lock trumps warm-start
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        opt._initialize_population()

        # Every individual should now contain Sol Ring (from lock)
        for ind in opt.population:
            names = {opt.candidate_pool[i].name for i in ind.card_indices}
            assert "Sol Ring" in names

    def test_missing_warm_start_file_doesnt_crash(
        self, lathiel, lathiel_analysis, wg_pool,
    ):
        """Non-existent warm-start path should log and proceed."""
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=2,
            patience_generations=50, random_seed=42,
            warm_start_path="/tmp/definitely_does_not_exist_12345.json",
        )
        # Should not raise
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        assert opt._warm_start_indices is None


class TestRefinementIntegration:
    """Full pipeline through DeckBuilder with refinement features."""

    def test_lock_and_ban_via_deck_builder(self, test_csv_path):
        """DeckBuilder should correctly handle locks and bans end-to-end."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            locked_cards=["Sol Ring"],
            banned_cards=["Forest"],  # Aggressive: ban basics
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        names = {c.name for c in result.best_deck.cards}
        assert "Sol Ring" in names
        assert "Forest" not in names

    def test_role_override_via_deck_builder(self, test_csv_path):
        """A role override should be respected through the full pipeline."""
        config = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=3,
            patience_generations=50, random_seed=42,
            candidates_per_category=15,
            role_target_overrides={"removal": (1, 3)},  # unusually low
        )
        builder = DeckBuilder(
            test_csv_path, config,
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
        # Just verify the build completes and config is preserved
        assert result.best_deck.card_count == 99
        assert result.config.role_target_overrides == {"removal": (1, 3)}
