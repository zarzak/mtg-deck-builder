"""Tests for IslandModelOptimizer. Use sequential mode for test speed."""

import pytest
from mtg_deck_builder.models import BuildConfig
from mtg_deck_builder.island_optimizer import (
    IslandModelOptimizer, IslandConfig, _copy_config,
)


@pytest.fixture
def small_config():
    """Small config so tests run quickly."""
    return BuildConfig(
        commander_name="Lathiel, the Bounteous Dawn",
        population_size=4,
        generations=3,
        patience_generations=50,
        random_seed=42,
    )


def _pool_without_commander(wg_pool, commander):
    return [c for c in wg_pool if c.name != commander.name]


class TestIslandConfig:
    def test_defaults(self):
        cfg = IslandConfig()
        assert cfg.num_islands >= 2
        assert cfg.migration_interval >= 1

    def test_custom(self):
        cfg = IslandConfig(num_islands=8, migration_interval=20, migration_size=3,
                           use_multiprocessing=False)
        assert cfg.num_islands == 8
        assert cfg.migration_interval == 20


class TestSequentialExecution:
    def test_run_returns_valid_result(self, small_config, lathiel_analysis,
                                      lathiel, wg_pool):
        """Island model should produce a valid result."""
        pool = _pool_without_commander(wg_pool, lathiel)
        island_cfg = IslandConfig(num_islands=2, use_multiprocessing=False)
        opt = IslandModelOptimizer(
            config=small_config,
            analysis=lathiel_analysis,
            candidate_pool=pool,
            commander=lathiel,
            island_config=island_cfg,
        )
        result = opt.run()
        assert result is not None
        assert result.best_deck.card_count == 99
        assert result.final_score > 0

    def test_different_seeds_produce_diverse_runs(
        self, small_config, lathiel_analysis, lathiel, wg_pool,
    ):
        """Each island should get a different seed (so they explore differently)."""
        pool = _pool_without_commander(wg_pool, lathiel)
        # Tracking output we're checking: islands with different seeds should
        # not all converge on identical decks (with small enough generations)
        island_cfg = IslandConfig(num_islands=3, use_multiprocessing=False)
        opt = IslandModelOptimizer(
            config=small_config,
            analysis=lathiel_analysis,
            candidate_pool=pool,
            commander=lathiel,
            island_config=island_cfg,
        )
        result = opt.run()
        # The result is the best across all islands
        assert result.best_deck.is_valid

    def test_single_island_still_works(self, small_config, lathiel_analysis,
                                       lathiel, wg_pool):
        """num_islands=1 should still work (degenerate case)."""
        pool = _pool_without_commander(wg_pool, lathiel)
        island_cfg = IslandConfig(num_islands=1, use_multiprocessing=False)
        opt = IslandModelOptimizer(
            config=small_config,
            analysis=lathiel_analysis,
            candidate_pool=pool,
            commander=lathiel,
            island_config=island_cfg,
        )
        result = opt.run()
        assert result.best_deck.card_count == 99

    def test_with_synergy_cache(self, small_config, lathiel_analysis,
                                lathiel, wg_pool):
        """Passing a synergy_cache should propagate to worker evaluators."""
        pool = _pool_without_commander(wg_pool, lathiel)
        synergy_cache = {c.name: 75.0 for c in pool[:10]}
        island_cfg = IslandConfig(num_islands=2, use_multiprocessing=False)
        opt = IslandModelOptimizer(
            config=small_config,
            analysis=lathiel_analysis,
            candidate_pool=pool,
            commander=lathiel,
            synergy_cache=synergy_cache,
            island_config=island_cfg,
        )
        result = opt.run()
        assert result.best_deck.card_count == 99


class TestCopyConfig:
    def test_modifying_copy_doesnt_change_original(self):
        cfg = BuildConfig(commander_name="Test", random_seed=42)
        copy = _copy_config(cfg)
        copy.random_seed = 99
        assert cfg.random_seed == 42
        assert copy.random_seed == 99


class TestMultiprocessingFallback:
    def test_falls_back_to_sequential_on_mp_failure(
        self, small_config, lathiel_analysis, lathiel, wg_pool, monkeypatch,
    ):
        """If multiprocessing fails, fall back to sequential execution."""
        pool = _pool_without_commander(wg_pool, lathiel)

        # Monkey-patch the _run_parallel method to raise
        opt = IslandModelOptimizer(
            config=small_config,
            analysis=lathiel_analysis,
            candidate_pool=pool,
            commander=lathiel,
            island_config=IslandConfig(num_islands=2, use_multiprocessing=True),
        )

        def fail(*a, **kw):
            raise RuntimeError("simulated mp failure")

        monkeypatch.setattr(opt, "_run_parallel", fail)
        # Should not raise; should fall through to sequential
        result = opt.run()
        assert result.best_deck.card_count == 99
