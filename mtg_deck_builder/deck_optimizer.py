"""
Deck Optimizer - Genetic Algorithm for EDH deck optimization.

Key v0.2 fixes:
- Invalid decks (duplicates, wrong color identity) get fitness=0 (reject, don't penalize)
- Two-phase fast/full evaluator transition properly re-evaluates carried individuals
- O(n²) bug in initial population creation fixed (set lookup)
- Crossover tracks taken indices to avoid creating decks that are 50% duplicates
- Configurable early stopping via config.patience_generations
- Generation field on Individual now actually used (tracks when each was created)
- Better mutation: respects role categories, configurable mutation strength
"""

import random
import logging
import time
from typing import Optional, Callable
from dataclasses import dataclass

from .models import Card, Deck, DeckScores, BuildConfig, CommanderAnalysis, OptimizationResult
from .deck_evaluator import DeckEvaluator, FastEvaluator
from . import tuning

logger = logging.getLogger(__name__)


# Evaluator "mode" tag so we know if a fitness score is from fast or full evaluation.
# Used to force re-eval when we switch modes.
class EvalMode:
    FAST = 'fast'
    FULL = 'full'
    NONE = 'none'


@dataclass
class Individual:
    """A single deck in the GA population. Stores card indices for efficiency."""
    card_indices: list[int]
    fitness: float = 0.0
    fitness_mode: str = EvalMode.NONE  # which evaluator produced fitness
    scores: Optional[DeckScores] = None
    generation_born: int = 0  # when this individual was first created
    is_valid: bool = True  # basic validity (set during eval)


@dataclass
class PopulationStats:
    """Statistics about the current population."""
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    diversity: float  # unique cards across population / total pool size
    invalid_count: int
    improvements_since_best: int
    # v0.9.29: which evaluator scored this generation. Fast-phase and
    # full-phase fitnesses are DIFFERENT SCALES — surfacing the mode lets
    # the progress UI explain the apparent score cliff at the switch.
    mode: str = ""


class DeckOptimizer:
    """
    Genetic algorithm optimizer for EDH decks.

    Usage:
        optimizer = DeckOptimizer(config, analysis, pool, commander, evaluator)
        result = optimizer.run()
    """

    # Mutation intensity — fraction of deck cards swapped per mutation event
    MUTATION_STRENGTH = tuning.GA_MUTATION_STRENGTH

    # When to switch from fast to full evaluator (fraction of total generations)
    FAST_PHASE_FRACTION = tuning.GA_FAST_PHASE_FRACTION

    def __init__(
        self,
        config: BuildConfig,
        analysis: CommanderAnalysis,
        candidate_pool: list[Card],
        commander: Card,
        evaluator: DeckEvaluator,
        fast_evaluator: Optional[FastEvaluator] = None,
    ):
        self.config = config
        self.analysis = analysis
        self.candidate_pool = candidate_pool
        self.commander = commander
        self.evaluator = evaluator
        self.fast_evaluator = fast_evaluator

        # Index lookup
        self._card_to_index = {c.name: i for i, c in enumerate(candidate_pool)}

        # Pre-compute validity info about the pool (for faster GA)
        self._commander_colors = set(
            ch for ch in (commander.color_identity or '') if ch in 'WUBRG'
        )
        self._valid_indices = self._compute_valid_pool_indices()
        if len(self._valid_indices) != len(candidate_pool):
            logger.warning(
                f"Pool contained {len(candidate_pool) - len(self._valid_indices)} "
                f"cards with color identity violating {self._commander_colors}; "
                f"these will not be used."
            )

        # Basic lands are special: can appear multiple times in a deck
        self._basic_land_indices = self._get_basic_land_indices()
        self._basic_land_set = set(self._basic_land_indices)

        # Categorize cards by broad type for smart mutation
        self._cards_by_category = self._categorize_cards()

        # v0.9.26: per-index effective scores for value-weighted mutation
        # draws (see _choose_replacement). Falls back to a flat 50 when the
        # evaluator can't score a card — weighted and uniform draws then
        # coincide for it.
        self._effective_scores: list[float] = []
        for c in candidate_pool:
            try:
                eff = (evaluator._get_card_baseline(c) * evaluator.base_weight
                       + evaluator._get_card_synergy(c)
                       * evaluator.synergy_weight)
            except Exception:
                eff = 50.0
            self._effective_scores.append(max(1.0, float(eff)))

        # Pre-compute effective scoring weights (for this commander)
        self._weights = config.get_effective_weights(analysis)

        # v0.4: Locked cards (must appear in every individual)
        # Resolve card names to pool indices. Any locked card not in the pool
        # is logged and ignored (DeckBuilder should have added them).
        self._locked_indices = self._resolve_locked_indices(
            config.locked_cards or []
        )
        if self._locked_indices:
            logger.info(
                f"Locking {len(self._locked_indices)} cards in every individual"
            )

        # v0.4: Warm-start deck (seed the initial population with it).
        # Resolved lazily because it depends on the pool.
        self._warm_start_indices: Optional[list[int]] = (
            self._load_warm_start(config.warm_start_path)
            if config.warm_start_path else None
        )

        # RNG (seeded for reproducibility)
        self.rng = random.Random(config.random_seed)

        # State
        self.population: list[Individual] = []
        self.generation = 0
        self.best_ever: Optional[Individual] = None
        self.score_history: list[float] = []
        self.diversity_history: list[float] = []
        # v0.9.6: which evaluator produced each score_history point. Fast and
        # full evaluators score on different scales, so the report needs this
        # to avoid plotting the two incomparable phases on one axis.
        self.eval_mode_history: list[str] = []
        self.generations_since_improvement = 0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _compute_valid_pool_indices(self) -> set[int]:
        """
        Indices in pool that pass color identity check AND aren't banned.
        Banned cards (from config.banned_cards) are removed from the pool
        entirely, so they can't appear in any individual.
        """
        banned = set((self.config.banned_cards or []))
        valid = set()
        for i, card in enumerate(self.candidate_pool):
            if card.name in banned:
                continue
            card_colors = set(ch for ch in (card.color_identity or '') if ch in 'WUBRG')
            if card_colors.issubset(self._commander_colors):
                valid.add(i)
        return valid

    def _resolve_locked_indices(self, locked_names: list[str]) -> list[int]:
        """
        Resolve locked card names to their pool indices.

        Locked cards that aren't in the pool are logged and skipped.
        Duplicate names in locked_names are de-duplicated (unless they're
        basic lands — those can appear multiple times, so we keep every
        mention so the user can lock "5 Forests").

        Returns list of indices (may include the same basic-land index
        multiple times if the user explicitly listed it multiple times).
        """
        result: list[int] = []
        seen_non_basic: set[int] = set()
        missing: list[str] = []

        for name in locked_names:
            if not name:
                continue
            idx = self._card_to_index.get(name)
            if idx is None:
                missing.append(name)
                continue
            if idx not in self._valid_indices:
                # Banned or color-illegal
                logger.warning(
                    f"Locked card '{name}' is not in the valid pool "
                    f"(banned, wrong color identity, or similar) — skipping"
                )
                continue

            # Non-basics dedup; basics allow duplicates
            if idx in self._basic_land_set:
                result.append(idx)
            elif idx not in seen_non_basic:
                result.append(idx)
                seen_non_basic.add(idx)

        if missing:
            logger.warning(
                f"{len(missing)} locked cards not found in pool: "
                f"{', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else "")
            )

        # Don't let locks exceed 99
        if len(result) > 99:
            logger.warning(
                f"Too many locked cards ({len(result)}); truncating to 99"
            )
            result = result[:99]

        return result

    def _load_warm_start(self, path: str) -> Optional[list[int]]:
        """Load a WarmStartDeck and resolve card names to indices."""
        from .models import WarmStartDeck
        try:
            snapshot = WarmStartDeck.from_json_file(path)
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to load warm-start from {path}: {e}")
            return None

        indices: list[int] = []
        missing = 0
        for name in snapshot.card_names:
            idx = self._card_to_index.get(name)
            if idx is None or idx not in self._valid_indices:
                missing += 1
                continue
            indices.append(idx)

        if missing:
            logger.info(
                f"Warm-start: {missing}/{len(snapshot.card_names)} cards not in pool "
                "(the GA will fill those slots with fresh picks)"
            )

        # Pad with basics if needed
        while len(indices) < 99 and self._basic_land_indices:
            indices.append(self._basic_land_indices[0])

        return indices[:99]

    def _get_basic_land_indices(self) -> list[int]:
        """
        Get indices of basic lands in the pool that match commander colors.
        Basics are special: deck can contain multiple copies.
        """
        basics = []
        for i in self._valid_indices:
            card = self.candidate_pool[i]
            if card.is_basic_land:
                # Only basics that produce a color the commander uses
                basic_ci = set(ch for ch in (card.color_identity or '') if ch in 'WUBRG')
                # Colorless basic lands (Wastes) always OK
                if not basic_ci or basic_ci.issubset(self._commander_colors):
                    basics.append(i)
        return basics

    def _categorize_cards(self) -> dict[str, list[int]]:
        """Categorize VALID pool indices by card-type for smart mutation."""
        categories = {
            'land': [], 'creature': [], 'instant': [], 'sorcery': [],
            'artifact': [], 'enchantment': [], 'planeswalker': [], 'other': [],
        }
        for i in self._valid_indices:
            card = self.candidate_pool[i]
            cat = self._get_card_category(card)
            categories[cat].append(i)
        return categories

    @staticmethod
    def _get_card_category(card: Card) -> str:
        if card.is_land:
            return 'land'
        elif card.is_creature:
            return 'creature'
        elif 'Instant' in card.types:
            return 'instant'
        elif 'Sorcery' in card.types:
            return 'sorcery'
        elif 'Artifact' in card.types:
            return 'artifact'
        elif 'Enchantment' in card.types:
            return 'enchantment'
        elif 'Planeswalker' in card.types:
            return 'planeswalker'
        return 'other'

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[PopulationStats], None]] = None,
    ) -> OptimizationResult:
        """Run the full GA optimization. Returns OptimizationResult."""
        start_time = time.time()
        logger.info(
            f"GA start: {self.config.generations} gens × {self.config.population_size} pop, "
            f"pool={len(self._valid_indices)} valid cards"
        )

        # Initial population
        self._initialize_population()
        # Initial eval (use fast if available)
        initial_mode = EvalMode.FAST if self.fast_evaluator else EvalMode.FULL
        self._evaluate_population(mode=initial_mode, force=True)
        self._record_stats(initial_mode)

        # Evolution loop
        fast_phase_end = int(self.config.generations * self.FAST_PHASE_FRACTION)
        # Exposed for _choose_replacement: value-weighted mutation draws are
        # FULL-phase only (the fast heuristic can't see combos/consistency,
        # so exploiting per-card value against it just homogenizes early).
        self._fast_phase_end = fast_phase_end if self.fast_evaluator else 0

        for gen in range(1, self.config.generations + 1):
            self.generation = gen

            # Determine eval mode for this generation
            if self.fast_evaluator and gen <= fast_phase_end:
                mode = EvalMode.FAST
            else:
                mode = EvalMode.FULL
            self._current_mode = mode

            # Breed next generation
            new_population = self._evolve()

            # Elitism: keep best N from old population
            new_population = self._apply_elitism(new_population)

            self.population = new_population
            self._evaluate_population(mode=mode)
            self._record_stats(mode)

            # Update best-ever (compare using same mode across time)
            current_best = max(self.population, key=lambda ind: ind.fitness)
            if self._is_new_best(current_best):
                self.best_ever = Individual(
                    card_indices=current_best.card_indices.copy(),
                    fitness=current_best.fitness,
                    fitness_mode=current_best.fitness_mode,
                    scores=current_best.scores,
                    generation_born=current_best.generation_born,
                    is_valid=current_best.is_valid,
                )
                self.generations_since_improvement = 0
            else:
                self.generations_since_improvement += 1

            # Progress callback
            if progress_callback:
                progress_callback(self._current_stats())

            if gen % 10 == 0 or gen == self.config.generations:
                stats = self._current_stats()
                logger.info(
                    f"Gen {gen} [{mode}]: best={stats.best_fitness:.2f} "
                    f"avg={stats.avg_fitness:.2f} diversity={stats.diversity:.2f} "
                    f"invalid={stats.invalid_count}"
                )

            # Early stopping
            if self.generations_since_improvement >= self.config.patience_generations:
                if mode == EvalMode.FAST:
                    # v0.9.27: stagnation during the FAST phase must never
                    # end the run — the full evaluator (combos, consistency,
                    # real synergy) hasn't scored a single deck yet. Observed
                    # (Doom B5): an early stop at gen 148 of 150 fast gens
                    # skipped full evaluation entirely and shipped a deck
                    # with its combo core missing. Switch to the full phase
                    # instead; _is_new_best lets full-mode fitness dethrone
                    # the fast-mode best automatically.
                    logger.info(
                        f"Fast phase stagnated at gen {gen} — switching to "
                        f"full evaluation early"
                    )
                    fast_phase_end = 0
                    self._fast_phase_end = 0
                    self.generations_since_improvement = 0
                else:
                    logger.info(
                        f"Early stop at gen {gen} "
                        f"({self.generations_since_improvement} gens without improvement)"
                    )
                    break

        # Final re-evaluation of best with full evaluator
        # (in case best was produced during fast phase)
        best = self.best_ever
        if best is None:
            # Should only happen if everything was invalid. Pick least-bad.
            best = max(self.population, key=lambda ind: ind.fitness)

        best_deck = self._indices_to_deck(best.card_indices)
        final_scores = self.evaluator.evaluate(best_deck)
        final_fitness = final_scores.total(self._weights)

        elapsed = time.time() - start_time

        telemetry = self.evaluator.build_telemetry(best_deck)

        return OptimizationResult(
            best_deck=best_deck,
            final_score=final_fitness,
            generations_run=self.generation,
            score_history=self.score_history,
            diversity_history=self.diversity_history,
            eval_mode_history=self.eval_mode_history,
            runtime_seconds=elapsed,
            config=self.config,
            card_telemetry=telemetry,
            commander_analysis=self.analysis,
        )

    def _is_new_best(self, candidate: Individual) -> bool:
        """Check if candidate beats best-ever by at least min_improvement."""
        if self.best_ever is None:
            return True
        # Only compare same-mode fitness to avoid fast>full regression
        if candidate.fitness_mode != self.best_ever.fitness_mode:
            # If candidate is full mode and best is fast, prefer full
            if candidate.fitness_mode == EvalMode.FULL and self.best_ever.fitness_mode == EvalMode.FAST:
                return True
            # Otherwise don't update (fast can't dethrone full)
            return False
        return candidate.fitness > self.best_ever.fitness + self.config.min_improvement

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def _initialize_population(self):
        """
        Create the initial population.

        Priority order:
        1. If warm-start is configured, seed N copies of it (with locks applied
           on top — locks always win)
        2. Fill remaining slots with random valid individuals (locked cards
           pre-populated into each)
        """
        self.population = []

        # Warm-start seeding
        if self._warm_start_indices is not None:
            warm_count = min(
                self.config.warm_start_copies,
                self.config.population_size,
            )
            for _ in range(warm_count):
                # Clone the warm-start, apply locks (they override), then shuffle
                ind = Individual(
                    card_indices=list(self._warm_start_indices),
                    generation_born=0,
                )
                self._apply_locks(ind)
                self.population.append(ind)

        # Fill remaining with random individuals
        while len(self.population) < self.config.population_size:
            self.population.append(self._create_random_individual())

    def _apply_locks(self, individual: Individual):
        """
        Ensure the individual contains every locked card.

        Strategy: check which locked cards are missing; for each, replace
        a non-locked, non-basic card with the locked one. Basic-land slots
        are preferred as replacement targets (least disruptive).
        """
        if not self._locked_indices:
            return

        locked_set = set(self._locked_indices)
        # For basics that appear multiple times in locked_indices,
        # we need to count occurrences
        locked_counts: dict[int, int] = {}
        for idx in self._locked_indices:
            locked_counts[idx] = locked_counts.get(idx, 0) + 1

        current = list(individual.card_indices)

        # Count current occurrences for each locked index
        for locked_idx, required_count in locked_counts.items():
            current_count = current.count(locked_idx)
            needed = required_count - current_count
            if needed <= 0:
                continue

            # Need to add `needed` copies of locked_idx. Find swap victims.
            for _ in range(needed):
                # Find a position to overwrite (prefer non-locked, non-basic)
                replace_pos = None
                # Pass 1: find a basic land not in locked set
                for pos, idx in enumerate(current):
                    if idx in self._basic_land_set and idx not in locked_set:
                        replace_pos = pos
                        break
                # Pass 2: find a non-basic not in locked set
                if replace_pos is None:
                    for pos, idx in enumerate(current):
                        if idx not in locked_set:
                            replace_pos = pos
                            break
                if replace_pos is None:
                    # Deck is entirely locked cards; can't add more
                    break
                current[replace_pos] = locked_idx

        individual.card_indices = current[:99]
        # If the deck shrunk (shouldn't but defensive), pad with basics
        while len(individual.card_indices) < 99 and self._basic_land_indices:
            individual.card_indices.append(self._basic_land_indices[0])


    def _create_random_individual(self) -> Individual:
        """
        Create a random valid deck from the candidate pool.

        Basic lands are special: they can appear multiple times. The algorithm:
        1. Pick unique non-basic lands (up to some count, say half of lands)
        2. Fill remaining land count with basics (can repeat)
        3. Pick unique non-land cards from the rest of the pool
        """
        land_indices = self._cards_by_category['land']
        land_set = set(land_indices)

        # Separate non-basic lands (unique) from basic lands (can repeat)
        non_basic_land_indices = [i for i in land_indices if i not in self._basic_land_set]
        has_basics = len(self._basic_land_indices) > 0

        # Target 37 lands
        target_lands = 37

        # Pick up to (target_lands - N) unique non-basic lands, where N = a few basics
        basic_reserve = 5 if has_basics else 0
        max_non_basic = min(target_lands - basic_reserve, len(non_basic_land_indices))
        if max_non_basic > 0:
            selected_lands = self.rng.sample(
                non_basic_land_indices,
                self.rng.randint(max(0, max_non_basic - 5), max_non_basic),
            )
        else:
            selected_lands = []

        # Fill remaining land slots with random basics (can repeat)
        remaining_land_slots = target_lands - len(selected_lands)
        if remaining_land_slots > 0 and has_basics:
            for _ in range(remaining_land_slots):
                selected_lands.append(self.rng.choice(self._basic_land_indices))

        # Pick non-land cards (unique)
        non_land_pool = [i for i in self._valid_indices if i not in land_set]
        needed = 99 - len(selected_lands)

        selected_non_lands = []
        if needed > 0 and non_land_pool:
            sample_size = min(needed, len(non_land_pool))
            selected_non_lands = self.rng.sample(non_land_pool, sample_size)

        indices = selected_lands + selected_non_lands

        # Fill to 99 if still short (e.g., tiny test pool) — pad with basics
        while len(indices) < 99 and has_basics:
            indices.append(self.rng.choice(self._basic_land_indices))

        # If no basics available and still short, pad with any valid card
        # (may create duplicates — will fail validation, but that's OK for bootstrapping)
        if len(indices) < 99:
            taken = set(indices) - self._basic_land_set  # basics can duplicate
            remaining = [i for i in self._valid_indices if i not in taken]
            self.rng.shuffle(remaining)
            for i in remaining:
                if len(indices) >= 99:
                    break
                indices.append(i)

        ind = Individual(
            card_indices=indices[:99],
            generation_born=self.generation,
        )
        # Apply locks (v0.4): ensure every locked card is present
        self._apply_locks(ind)
        return ind

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_population(self, mode: str, force: bool = False) -> None:
        """
        Evaluate every individual. Re-evaluates if mode differs from cached mode
        (this fixes the stale-fitness bug from v0.1).
        """
        for ind in self.population:
            # Skip if already evaluated in the correct mode
            if not force and ind.fitness_mode == mode and ind.fitness > 0:
                continue

            deck = self._indices_to_deck(ind.card_indices)

            if mode == EvalMode.FAST and self.fast_evaluator:
                ind.fitness = self.fast_evaluator.evaluate(deck)
                ind.scores = None  # fast eval doesn't produce full DeckScores
                ind.is_valid = ind.fitness > 0
            else:
                scores = self.evaluator.evaluate(deck)
                ind.scores = scores
                ind.is_valid = scores.is_valid
                # REJECT invalid decks entirely (the fix for issue #13)
                if not ind.is_valid:
                    ind.fitness = 0.0
                else:
                    ind.fitness = scores.total(self._weights)

            ind.fitness_mode = mode

    def _indices_to_deck(self, indices: list[int]) -> Deck:
        """Convert card indices to a Deck object."""
        cards = [self.candidate_pool[i] for i in indices]
        return Deck(
            commander=self.commander,
            cards=cards,
            generation=self.generation,
        )

    # ------------------------------------------------------------------
    # Evolution operators
    # ------------------------------------------------------------------

    def _evolve(self) -> list[Individual]:
        """Create new generation via tournament selection + crossover + mutation."""
        new_pop = []
        while len(new_pop) < self.config.population_size:
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            if self.rng.random() < self.config.crossover_rate:
                c1, c2 = self._crossover_smart(p1, p2)
            else:
                c1 = Individual(
                    card_indices=p1.card_indices.copy(),
                    generation_born=self.generation,
                )
                c2 = Individual(
                    card_indices=p2.card_indices.copy(),
                    generation_born=self.generation,
                )

            if self.rng.random() < self.config.mutation_rate:
                self._mutate(c1)
            if self.rng.random() < self.config.mutation_rate:
                self._mutate(c2)

            new_pop.extend([c1, c2])

        return new_pop[:self.config.population_size]

    def _tournament_select(self) -> Individual:
        """Tournament selection: pick best of K random individuals."""
        k = min(self.config.tournament_size, len(self.population))
        tournament = self.rng.sample(self.population, k)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover_smart(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> tuple[Individual, Individual]:
        """
        Smart uniform crossover that tracks taken indices to avoid duplicates.

        v0.1 bug: the old crossover could create children that were ~50% duplicates
        which then had to be massively replaced, effectively randomizing the child.
        v0.2: pick from each parent but always pick something not already taken.
        """
        p1_set = set(parent1.card_indices)
        p2_set = set(parent2.card_indices)
        # Union is what we have available to pick from
        available = p1_set | p2_set

        child1_indices = self._build_child(parent1, parent2, available.copy())
        child2_indices = self._build_child(parent2, parent1, available.copy())

        child1 = Individual(
            card_indices=child1_indices,
            generation_born=self.generation,
        )
        child2 = Individual(
            card_indices=child2_indices,
            generation_born=self.generation,
        )
        # v0.4: ensure locked cards survive crossover
        self._apply_locks(child1)
        self._apply_locks(child2)
        return (child1, child2)

    def _build_child(
        self,
        preferred: Individual,
        secondary: Individual,
        available: set[int],
    ) -> list[int]:
        """
        Build a child deck by picking cards without duplication (except basics).

        Strategy: for each position, flip a coin.
        - On heads, try to take preferred[i] (basic lands always allowed; others
          only if not already taken).
        - On tails, try to take secondary[i] under the same rules.
        - If neither available, pick any still-available non-dupe, or a basic.
        """
        # Non-basic indices that are already in the child
        taken_non_basic: set[int] = set()
        child: list[int] = []

        def can_take(idx: int) -> bool:
            # Basics can always be taken (duplication is legal)
            if idx in self._basic_land_set:
                return True
            # Non-basics only if not already in child
            return idx not in taken_non_basic

        def take(idx: int):
            if idx not in self._basic_land_set:
                taken_non_basic.add(idx)

        for i in range(99):
            pref_idx = preferred.card_indices[i] if i < len(preferred.card_indices) else None
            sec_idx = secondary.card_indices[i] if i < len(secondary.card_indices) else None

            pick = None
            if self.rng.random() < 0.5 and pref_idx is not None and can_take(pref_idx):
                pick = pref_idx
            elif sec_idx is not None and can_take(sec_idx):
                pick = sec_idx
            elif pref_idx is not None and can_take(pref_idx):
                pick = pref_idx

            if pick is None:
                # Fall back: pick anything from available that isn't a duplicate
                remaining = [idx for idx in available if can_take(idx)]
                if remaining:
                    pick = self.rng.choice(remaining)
                elif self._basic_land_indices:
                    # Always have basics as ultimate fallback
                    pick = self.rng.choice(self._basic_land_indices)
                else:
                    # Nothing left — use any valid non-dupe
                    candidates = [idx for idx in self._valid_indices if can_take(idx)]
                    if candidates:
                        pick = self.rng.choice(candidates)
                    else:
                        break

            take(pick)
            child.append(pick)

        # Ensure 99 cards
        while len(child) < 99:
            if self._basic_land_indices:
                child.append(self.rng.choice(self._basic_land_indices))
            else:
                candidates = [idx for idx in self._valid_indices if can_take(idx)]
                if not candidates:
                    break
                pick = self.rng.choice(candidates)
                take(pick)
                child.append(pick)

        return child[:99]

    def _choose_replacement(self, candidates: list[int]) -> int:
        """v0.9.26: pick a mutation replacement — value-weighted with
        probability GA_MUTATION_VALUE_BIAS (∝ effective-score², so the pool's
        best cards actually get PROPOSED), uniform otherwise (exploration).

        The fitness function can only keep what the operators put in front
        of it: with uniform-only draws, a strictly dominant card's entry was
        a lottery ticket (observed: Sol Ring unproposed across 300
        generations while a dominated same-category card kept its slot).

        v0.9.27: FULL-phase only. During the fast phase the heuristic can't
        see combos/consistency, so biasing proposals toward per-card value
        just homogenizes the population against the wrong objective
        (observed: gen-10 diversity fell 0.54 -> 0.39 and the fast phase
        stalled 20 points lower). Fast phase explores uniformly; the full
        phase exploits.
        """
        in_fast_phase = (
            getattr(self, "_fast_phase_end", 0) > 0
            and self.generation <= self._fast_phase_end
        )
        if (not in_fast_phase and len(candidates) > 1
                and self.rng.random() < tuning.GA_MUTATION_VALUE_BIAS):
            weights = [self._effective_scores[i] ** 2 for i in candidates]
            return self.rng.choices(candidates, weights=weights, k=1)[0]
        return self.rng.choice(candidates)

    def _mutate(self, individual: Individual):
        """
        Mutate by swapping some cards with same-category alternatives.

        Locked cards (config.locked_cards) are never mutated — we skip their
        positions. After mutation completes, we also re-apply locks to heal
        any accidental removal from prior ops.
        """
        n_swaps = max(1, int(len(individual.card_indices) * self.MUTATION_STRENGTH))

        # v0.4: positions containing locked cards are off-limits
        locked_set = set(self._locked_indices)

        # Track non-basic duplicates set
        non_basic_in_deck = set(
            idx for idx in individual.card_indices
            if idx not in self._basic_land_set
        )

        for _ in range(n_swaps):
            if not individual.card_indices:
                break
            pos = self.rng.randint(0, len(individual.card_indices) - 1)
            old_idx = individual.card_indices[pos]

            # v0.4: skip if this position holds a locked card
            if old_idx in locked_set:
                continue

            old_card = self.candidate_pool[old_idx]
            category = self._get_card_category(old_card)

            # Pick replacement from same category (basics ok, non-basics only if new)
            category_pool = self._cards_by_category.get(category, [])
            candidates = [
                i for i in category_pool
                if i in self._basic_land_set or i not in non_basic_in_deck
            ]

            if not candidates:
                candidates = [
                    i for i in self._valid_indices
                    if i in self._basic_land_set or i not in non_basic_in_deck
                ]

            if candidates:
                new_idx = self._choose_replacement(candidates)
                # Update tracking
                if old_idx not in self._basic_land_set:
                    non_basic_in_deck.discard(old_idx)
                if new_idx not in self._basic_land_set:
                    non_basic_in_deck.add(new_idx)
                individual.card_indices[pos] = new_idx

        # v0.4: re-apply locks in case any were dropped by earlier ops
        self._apply_locks(individual)

        individual.fitness_mode = EvalMode.NONE
        individual.fitness = 0.0

    def _apply_elitism(self, new_population: list[Individual]) -> list[Individual]:
        """Preserve top-K individuals from the old population."""
        if self.config.elitism_count <= 0:
            return new_population

        # Best K from old population (deep copies so we don't share references)
        elite = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        elite = elite[:self.config.elitism_count]
        elite_copies = [
            Individual(
                card_indices=ind.card_indices.copy(),
                fitness=ind.fitness,
                fitness_mode=ind.fitness_mode,
                scores=ind.scores,
                generation_born=ind.generation_born,
                is_valid=ind.is_valid,
            )
            for ind in elite
        ]

        # Replace worst of new with elite
        new_sorted = sorted(new_population, key=lambda ind: ind.fitness)
        keep = new_sorted[self.config.elitism_count:]
        return keep + elite_copies

    # ------------------------------------------------------------------
    # Statistics & reporting
    # ------------------------------------------------------------------

    def _record_stats(self, mode=None):
        """Record statistics for the current generation."""
        stats = self._current_stats()
        self.score_history.append(stats.best_fitness)
        self.diversity_history.append(stats.diversity)
        # Tag the point with the evaluator that produced it (str of the
        # EvalMode enum, or "" if unknown) so the report can separate phases.
        self.eval_mode_history.append(
            getattr(mode, "value", str(mode)) if mode is not None else ""
        )

    def _current_stats(self) -> PopulationStats:
        fitnesses = [ind.fitness for ind in self.population]
        valid_count = sum(1 for ind in self.population if ind.is_valid)

        # Diversity = unique cards used across population / total pool
        all_cards = set()
        for ind in self.population:
            all_cards.update(ind.card_indices)
        pool_size = len(self._valid_indices) or 1
        diversity = len(all_cards) / pool_size

        return PopulationStats(
            generation=self.generation,
            best_fitness=max(fitnesses) if fitnesses else 0.0,
            avg_fitness=sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            worst_fitness=min(fitnesses) if fitnesses else 0.0,
            diversity=diversity,
            invalid_count=len(self.population) - valid_count,
            improvements_since_best=self.generations_since_improvement,
            mode=str(getattr(self, "_current_mode", "")),
        )
