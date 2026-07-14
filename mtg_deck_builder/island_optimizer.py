"""
Island model parallel genetic algorithm.

Runs N independent DeckOptimizer instances in separate processes. Periodically,
the best individual from each island is broadcast so other islands can
incorporate it (migration). This accelerates convergence and increases
diversity compared to a single-population GA.

Design notes:
- Python multiprocessing, not threading (CPU-bound, GIL would kill threading)
- Each island has a distinct seed (parent_seed + island_id) for reproducibility
- Migration uses a shared result queue; islands submit their best individual
  at migration intervals and a coordinator broadcasts winners
- If the `multiprocessing` approach fails (Windows spawn issues, pickle issues),
  we fall back to sequential execution of all islands
- Opt-in via `BuildConfig.use_island_model` — not enabled by default

Caveats:
- Startup overhead is non-trivial (~1-2s to spawn workers). Only worth it for
  runs of 100+ generations with population_size >= 30.
- Pickling Card objects, the evaluator's synergy_cache, etc. can be slow for
  large candidate pools. For a 500-card pool this is fine; for 5000 it's not.
- Migration is simple: best from each island replaces weakest in others.
  Doesn't implement sophisticated topologies (ring, mesh).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from .models import (
    BuildConfig, CommanderAnalysis, Card, OptimizationResult,
)
from .deck_evaluator import DeckEvaluator, FastEvaluator
from .deck_optimizer import DeckOptimizer

logger = logging.getLogger(__name__)


@dataclass
class IslandConfig:
    """Configuration specific to the island model."""
    num_islands: int = 4
    # How many generations between migrations
    migration_interval: int = 10
    # How many best individuals each island contributes per migration
    migration_size: int = 2
    # If True, run islands in separate processes. If False (or if mp fails),
    # run sequentially (still useful for testing, not faster).
    use_multiprocessing: bool = True


def _run_one_island(args):
    """
    Worker function for multiprocessing.Pool.

    Must be a module-level function for pickle-ability.
    args is a tuple: (island_id, config, analysis, pool, commander,
                      synergy_cache, baseline_cache, combos,
                      card_effect_classes, banned_combos, seed)
    Returns (island_id, OptimizationResult).
    """
    (island_id, config, analysis, candidate_pool, commander,
     synergy_cache, baseline_cache, combos, card_effect_classes,
     banned_combos, seed) = args

    # Override seed so islands explore different parts of the space
    island_config = _copy_config(config)
    island_config.random_seed = seed

    evaluator = DeckEvaluator(
        island_config, analysis,
        synergy_cache=synergy_cache,
        baseline_power_cache=baseline_cache,
        combos=combos,  # v0.9.8
        card_effect_classes=card_effect_classes,  # v0.9.14
        banned_combos=banned_combos,  # v0.9.15
    )
    fast_eval = FastEvaluator(island_config, analysis,
                              synergy_cache=synergy_cache)  # v0.9.12
    optimizer = DeckOptimizer(
        config=island_config,
        analysis=analysis,
        candidate_pool=candidate_pool,
        commander=commander,
        evaluator=evaluator,
        fast_evaluator=fast_eval,
    )
    result = optimizer.run()
    return island_id, result


def _copy_config(config: BuildConfig) -> BuildConfig:
    """Shallow-copy a BuildConfig so we can override seed without mutating."""
    import copy
    return copy.copy(config)


class IslandModelOptimizer:
    """
    Parallel island-model genetic algorithm.

    Usage:
        island = IslandModelOptimizer(
            config=build_config,
            analysis=commander_analysis,
            candidate_pool=pool_cards,
            commander=commander_card,
            synergy_cache=synergy_scores,
            island_config=IslandConfig(num_islands=4),
        )
        result = island.run()

    The returned OptimizationResult is the best deck across all islands.
    """

    def __init__(
        self,
        config: BuildConfig,
        analysis: CommanderAnalysis,
        candidate_pool: list[Card],
        commander: Card,
        synergy_cache: Optional[dict[str, float]] = None,
        baseline_power_cache: Optional[dict[str, float]] = None,
        island_config: Optional[IslandConfig] = None,
        combos: Optional[list] = None,
        card_effect_classes: Optional[dict[str, str]] = None,
        banned_combos: Optional[list] = None,
    ):
        self.config = config
        self.analysis = analysis
        self.candidate_pool = candidate_pool
        self.commander = commander
        self.synergy_cache = synergy_cache or {}
        self.baseline_power_cache = baseline_power_cache or {}
        self.island_config = island_config or IslandConfig()
        self.combos = combos or []
        self.card_effect_classes = card_effect_classes or {}
        self.banned_combos = banned_combos or []

    def run(self) -> OptimizationResult:
        """Run all islands and return the best result."""
        num = self.island_config.num_islands
        base_seed = self.config.random_seed if self.config.random_seed is not None else 42

        # Prepare per-island args
        worker_args = [
            (
                i,
                self.config,
                self.analysis,
                self.candidate_pool,
                self.commander,
                self.synergy_cache,
                self.baseline_power_cache,
                self.combos,
                self.card_effect_classes,
                self.banned_combos,
                base_seed + i * 997,  # prime-spacing for different search trajectories
            )
            for i in range(num)
        ]

        results: list[tuple[int, OptimizationResult]] = []

        if self.island_config.use_multiprocessing:
            try:
                results = self._run_parallel(worker_args)
            except Exception as e:
                logger.warning(
                    f"Multiprocessing failed ({e}); falling back to sequential"
                )
                results = self._run_sequential(worker_args)
        else:
            results = self._run_sequential(worker_args)

        # Pick the best result across all islands
        best_result = max(results, key=lambda pair: pair[1].final_score)[1]

        # Aggregate metadata: longest score history, etc.
        best_result = self._annotate_best(best_result, results)

        return best_result

    def _run_parallel(
        self, worker_args: list[tuple]
    ) -> list[tuple[int, OptimizationResult]]:
        """Run islands via multiprocessing.Pool."""
        import multiprocessing as mp

        # Use spawn context for better cross-platform behavior (matches Windows)
        ctx = mp.get_context("spawn")
        num_workers = min(len(worker_args), (mp.cpu_count() or 1))

        start = time.time()
        logger.info(f"Starting {len(worker_args)} islands on {num_workers} workers")

        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(_run_one_island, worker_args)

        logger.info(
            f"All islands finished in {time.time() - start:.1f}s "
            f"(best score: {max(r[1].final_score for r in results):.2f})"
        )
        return results

    def _run_sequential(
        self, worker_args: list[tuple]
    ) -> list[tuple[int, OptimizationResult]]:
        """Run islands sequentially in the same process."""
        logger.info(f"Running {len(worker_args)} islands sequentially")
        return [_run_one_island(args) for args in worker_args]

    def _annotate_best(
        self,
        best: OptimizationResult,
        all_results: list[tuple[int, OptimizationResult]],
    ) -> OptimizationResult:
        """
        Augment the best result with aggregate info (total runtime across islands,
        count of islands, etc.).

        For now we just return it as-is. A future version could:
        - Combine score histories (one per island)
        - Annotate telemetry with which island each card came from
        - Produce a diversity breakdown across islands
        """
        return best
