"""Tests for DeckOptimizer."""

import pytest
from mtg_deck_builder.models import BuildConfig, CommanderAnalysis
from mtg_deck_builder.deck_evaluator import DeckEvaluator, FastEvaluator
from mtg_deck_builder.deck_optimizer import DeckOptimizer, EvalMode, Individual


def _make_optimizer(lathiel, lathiel_analysis, wg_pool, config=None):
    """Helper to construct an optimizer."""
    if config is None:
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=6,
            generations=5,
            patience_generations=20,
            random_seed=42,
        )
    pool = [c for c in wg_pool if c.name != lathiel.name]
    evaluator = DeckEvaluator(config, lathiel_analysis)
    fast = FastEvaluator(config, lathiel_analysis)
    return DeckOptimizer(config, lathiel_analysis, pool, lathiel, evaluator, fast)


class TestValueWeightedMutation:
    """v0.9.26: mutation replacement draws are value-weighted (∝ eff²) half
    the time, uniform otherwise — so the pool's best cards actually get
    PROPOSED. Regression: Sol Ring (strictly dominant over a same-category
    deck card) went unproposed across 300 generations under uniform draws."""

    def _opt_with_scores(self, lathiel, lathiel_analysis, wg_pool, scores):
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt._effective_scores = [
            scores.get(opt.candidate_pool[i].name, 50.0)
            for i in range(len(opt.candidate_pool))
        ]
        return opt

    def test_high_value_card_drawn_far_more_than_uniform(
            self, lathiel, lathiel_analysis, wg_pool):
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        candidates = list(range(min(20, len(opt.candidate_pool))))
        star = candidates[7]
        # One standout (95) among mediocrity (40).
        opt._effective_scores = [40.0] * len(opt.candidate_pool)
        opt._effective_scores[star] = 95.0
        draws = [opt._choose_replacement(candidates) for _ in range(4000)]
        star_rate = draws.count(star) / len(draws)
        uniform_rate = 1 / len(candidates)  # 5%
        # Expected ≈ 0.5*uniform + 0.5*weighted(95²/Σ) ≈ 0.5*5% + 0.5*22.9%
        # ≈ 14%. Assert well above uniform, below total dominance.
        assert star_rate > 2 * uniform_rate
        assert star_rate < 0.5

    def test_uniform_share_preserves_exploration(
            self, lathiel, lathiel_analysis, wg_pool):
        # Even the worst card must still be drawable (the uniform half).
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        candidates = list(range(min(10, len(opt.candidate_pool))))
        dud = candidates[3]
        opt._effective_scores = [90.0] * len(opt.candidate_pool)
        opt._effective_scores[dud] = 1.0
        draws = [opt._choose_replacement(candidates) for _ in range(4000)]
        assert draws.count(dud) > 0

    def test_single_candidate_short_circuits(
            self, lathiel, lathiel_analysis, wg_pool):
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        assert opt._choose_replacement([5]) == 5

    def test_seeded_reproducibility_with_weights(
            self, lathiel, lathiel_analysis, wg_pool):
        # Same seed -> identical draw sequence (rng.choices uses self.rng).
        o1 = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        o2 = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        cands = list(range(min(15, len(o1.candidate_pool))))
        s1 = [o1._choose_replacement(cands) for _ in range(50)]
        s2 = [o2._choose_replacement(cands) for _ in range(50)]
        assert s1 == s2

    def test_weighting_disabled_during_fast_phase(
            self, lathiel, lathiel_analysis, wg_pool):
        # v0.9.27: fast phase explores uniformly (the heuristic can't see
        # combos/consistency, so value bias just homogenizes early).
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        candidates = list(range(min(20, len(opt.candidate_pool))))
        star = candidates[7]
        opt._effective_scores = [40.0] * len(opt.candidate_pool)
        opt._effective_scores[star] = 95.0
        opt._fast_phase_end = 100
        opt.generation = 50  # inside the fast phase
        draws = [opt._choose_replacement(candidates) for _ in range(4000)]
        fast_rate = draws.count(star) / len(draws)
        assert abs(fast_rate - 1 / len(candidates)) < 0.02  # ≈ uniform
        opt.generation = 150  # full phase
        draws = [opt._choose_replacement(candidates) for _ in range(4000)]
        assert draws.count(star) / len(draws) > 2 / len(candidates)


class TestFastPhaseStagnation:
    """v0.9.27 regression: fast-phase stagnation must SWITCH to full
    evaluation, never terminate the run. Observed (Doom B5): early stop at
    gen 148 of 150 fast gens skipped the full evaluator entirely — the
    shipped deck lost its combo core and consistency collapsed 98 -> 73."""

    def test_stagnation_switches_to_full_not_stop(
            self, lathiel, lathiel_analysis, wg_pool):
        from mtg_deck_builder.models import BuildConfig
        config = BuildConfig(
            commander_name=lathiel.name, population_size=6, generations=30,
            patience_generations=2, random_seed=42,
        )
        pool = [c for c in wg_pool if c.name != lathiel.name]
        evaluator = DeckEvaluator(config, lathiel_analysis)
        fast = FastEvaluator(config, lathiel_analysis)
        opt = DeckOptimizer(config, lathiel_analysis, pool, lathiel,
                            evaluator, fast)
        result = opt.run()
        # With patience=2 the fast phase (gens 1-15) stagnates almost
        # immediately. The OLD bug ended the run there: the full evaluator's
        # only call was the final re-eval (eval_count ~1). Fixed behavior:
        # the run switches to full mode and evolves real generations.
        assert evaluator.eval_count > config.population_size
        assert result.best_deck is not None


class TestOptimizerInitialization:
    def test_seeded_reproducibility(self, lathiel, lathiel_analysis, wg_pool):
        """Same seed should produce same initial population."""
        opt1 = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt1._initialize_population()
        indices1 = [ind.card_indices for ind in opt1.population]

        opt2 = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt2._initialize_population()
        indices2 = [ind.card_indices for ind in opt2.population]

        # Populations should be identical with same seed
        # (Compare as sets since order may vary slightly but content should match)
        for i1, i2 in zip(indices1, indices2):
            assert set(i1) == set(i2)

    def test_populates_basic_land_indices(self, lathiel, lathiel_analysis, wg_pool):
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        # Test pool includes Forest and Plains as basics
        assert len(opt._basic_land_indices) >= 1

    def test_color_identity_filtering(self, lathiel, lathiel_analysis, db):
        """Cards outside commander's color identity are filtered out."""
        # Use FULL DB, not just wg_pool, to ensure some get filtered
        config = BuildConfig(
            commander_name=lathiel.name, population_size=4, generations=2,
            random_seed=42,
        )
        pool = list(db.all_cards)
        evaluator = DeckEvaluator(config, lathiel_analysis)
        opt = DeckOptimizer(config, lathiel_analysis, pool, lathiel, evaluator)
        # Should have filtered some cards (Exquisite Blood, Sanguine Bond, etc.)
        assert len(opt._valid_indices) < len(pool)


class TestOptimizerEvolution:
    def test_creates_valid_individuals(self, lathiel, lathiel_analysis, wg_pool):
        """Initial population should have 99-card decks."""
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt._initialize_population()
        for ind in opt.population:
            assert len(ind.card_indices) == 99

    def test_crossover_preserves_size(self, lathiel, lathiel_analysis, wg_pool):
        """Children should have 99 cards."""
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt._initialize_population()
        p1, p2 = opt.population[0], opt.population[1]
        c1, c2 = opt._crossover_smart(p1, p2)
        assert len(c1.card_indices) == 99
        assert len(c2.card_indices) == 99

    def test_crossover_no_non_basic_dupes(self, lathiel, lathiel_analysis, wg_pool):
        """Children should have no duplicate non-basic cards."""
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt._initialize_population()
        p1, p2 = opt.population[0], opt.population[1]
        c1, _ = opt._crossover_smart(p1, p2)

        non_basics = [i for i in c1.card_indices if i not in opt._basic_land_set]
        assert len(non_basics) == len(set(non_basics)), "Found duplicate non-basic"

    def test_mutation_preserves_size(self, lathiel, lathiel_analysis, wg_pool):
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt._initialize_population()
        ind = opt.population[0]
        original_size = len(ind.card_indices)
        opt._mutate(ind)
        assert len(ind.card_indices) == original_size

    def test_mutation_marks_for_re_eval(self, lathiel, lathiel_analysis, wg_pool):
        """After mutation, fitness should be reset so it's re-evaluated."""
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt._initialize_population()
        ind = opt.population[0]
        ind.fitness = 50.0
        ind.fitness_mode = EvalMode.FULL
        opt._mutate(ind)
        assert ind.fitness == 0.0
        assert ind.fitness_mode == EvalMode.NONE


class TestTwoPhaseTransition:
    """Test the bug-fix for two-phase fast/full evaluator transition."""

    def test_refreshes_on_mode_change(self, lathiel, lathiel_analysis, wg_pool):
        """When switching from fast to full, individuals should be re-evaluated."""
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool)
        opt._initialize_population()

        # First eval in fast mode
        opt._evaluate_population(mode=EvalMode.FAST)
        for ind in opt.population:
            assert ind.fitness_mode == EvalMode.FAST

        # Switch to full mode — all should be re-evaluated
        opt._evaluate_population(mode=EvalMode.FULL)
        for ind in opt.population:
            assert ind.fitness_mode == EvalMode.FULL


class TestEndToEnd:
    def test_full_run(self, lathiel, lathiel_analysis, wg_pool):
        """A small full run should produce a valid result."""
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=8, generations=10,
            patience_generations=50, random_seed=42,
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        result = opt.run()

        assert result.best_deck.card_count == 99
        assert result.best_deck.is_valid
        assert result.final_score > 0
        assert result.generations_run <= 10
        assert len(result.card_telemetry) == 99
        assert len(result.score_history) > 0

    def test_reproducibility(self, lathiel, lathiel_analysis, wg_pool):
        """Same seed should produce same final deck."""
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=6, generations=8,
            patience_generations=50, random_seed=42,
        )
        pool = [c for c in wg_pool if c.name != lathiel.name]

        # Run 1
        ev1 = DeckEvaluator(config, lathiel_analysis)
        opt1 = DeckOptimizer(config, lathiel_analysis, pool, lathiel, ev1,
                             FastEvaluator(config, lathiel_analysis))
        r1 = opt1.run()

        # Run 2 — same seed
        ev2 = DeckEvaluator(config, lathiel_analysis)
        opt2 = DeckOptimizer(config, lathiel_analysis, pool, lathiel, ev2,
                             FastEvaluator(config, lathiel_analysis))
        r2 = opt2.run()

        # Should get same result
        names1 = sorted(c.name for c in r1.best_deck.cards)
        names2 = sorted(c.name for c in r2.best_deck.cards)
        assert names1 == names2

    def test_early_stopping(self, lathiel, lathiel_analysis, wg_pool):
        """Early stop triggers if patience exceeded."""
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=4, generations=100,
            patience_generations=3,  # Very short patience
            random_seed=42,
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        result = opt.run()
        # Should stop well before gen 100
        assert result.generations_run < 100


class TestOptimizerElitism:
    def test_elite_preserved(self, lathiel, lathiel_analysis, wg_pool):
        """After evolution, the best from old pop should still be in new."""
        config = BuildConfig(
            commander_name=lathiel.name,
            population_size=6, generations=3,
            elitism_count=2, random_seed=42, patience_generations=50,
        )
        opt = _make_optimizer(lathiel, lathiel_analysis, wg_pool, config)
        opt._initialize_population()
        opt._evaluate_population(mode=EvalMode.FULL)

        # Record best fitness
        best_before = max(ind.fitness for ind in opt.population)
        best_deck_before = max(opt.population, key=lambda x: x.fitness)
        best_indices_before = frozenset(best_deck_before.card_indices)

        # Evolve one generation
        new_pop = opt._evolve()
        new_pop = opt._apply_elitism(new_pop)
        opt.population = new_pop
        opt._evaluate_population(mode=EvalMode.FULL)

        # Elite set should still be in new pop (same indices as a set)
        best_after_indices = {
            frozenset(ind.card_indices) for ind in opt.population
        }
        assert best_indices_before in best_after_indices

        # Best fitness should not decrease
        best_after = max(ind.fitness for ind in opt.population)
        assert best_after >= best_before - 0.01  # allow tiny float drift
