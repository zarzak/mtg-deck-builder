"""
Deck Builder Orchestrator - Coordinates multi-phase deck building.

Phases (numbered roughly in execution order; some are conditional):
1. Commander analysis (LLM, mockable)
2. Card pool generation (database queries)
3. EDHREC data fetch — v0.3, opt-in via use_edhrec
4. Budget filtering — v0.3, opt-in via budget_max_per_card
5. Candidate filtering (LLM selection per role)
6. Synergy pre-scoring (LLM, possibly augmented by embeddings — v0.3)
7. Locked-card injection — v0.4, ensures locked names are in the pool
8. Genetic algorithm optimization (DeckOptimizer or IslandModelOptimizer)
9. Optional LLM review pass — v0.2
10. Optional HTML report generation — v0.2 base, images added v0.4

Optional integrations the orchestrator coordinates:
- LLM (mockable for testing)
- EDHREC client (v0.3)
- Embedding-based synergy scorer (v0.3, requires sentence-transformers)
- Price source (v0.3, for budget filter)
- Scryfall card source (v0.4, for HTML report images)
- Scryfall tag client (v0.5, for art-tag flavor scoring)
- Flavor tag scorer (v0.5)

All optional integrations can be either auto-constructed from BuildConfig
or injected via constructor parameters (for tests, custom impls, or
sharing instances across multiple builds).
"""

import logging
import re
import time
from typing import Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import (
    Card, Deck, BuildConfig, CommanderAnalysis,
    OptimizationResult, ComboReport,
)
from .card_database import CardDatabase
from .llm_engine import LLMEngine, LLMConfig
from .deck_evaluator import DeckEvaluator, FastEvaluator
from .deck_optimizer import DeckOptimizer, PopulationStats
from .card_power_scorer import CardPowerScorer
from .combo_engine_detector import ComboEngineDetector
from .structural_predicates import derive_structural_predicates

logger = logging.getLogger(__name__)


@dataclass
class BuildProgress:
    """Progress event emitted during deck building. Consumers can render these."""
    phase: str        # 'analysis', 'pools', 'filtering', 'scoring', 'optimization', 'review', 'done'
    step: str         # specific step within phase
    progress: float   # 0.0 to 1.0 within this phase
    message: str
    elapsed_seconds: float = 0.0


@dataclass
class CandidatePool:
    """Organized candidate cards by role for the filtering phase."""
    ramp: list[Card] = field(default_factory=list)
    draw: list[Card] = field(default_factory=list)
    removal: list[Card] = field(default_factory=list)
    wipe: list[Card] = field(default_factory=list)
    lands: list[Card] = field(default_factory=list)
    threats: list[Card] = field(default_factory=list)
    protection: list[Card] = field(default_factory=list)
    recursion: list[Card] = field(default_factory=list)
    # v0.9.16: global power-staples — top-N by GLOBAL cached card power,
    # regardless of role or theme. The generalized fix for taxonomy holes.
    power_staples: list[Card] = field(default_factory=list)
    # `synergy` is the broad recall-union pool (up to 2500 cards). It is the
    # SOURCE pool for the Phase 2 synergy_engine pass — not a deck bucket
    # itself in the v0.9 architecture. The synergy_engine field below holds
    # the small filtered output of that pass.
    synergy: list[Card] = field(default_factory=list)
    # v0.9: cross-cutting "synergy engine" bucket. After role buckets fill,
    # this holds 15-30 strategy-defining engine pieces (Soul-Sister-style
    # triggers, cheap repeatable payoffs) that didn't make a role bucket
    # but are central to the commander's plan.
    synergy_engine: list[Card] = field(default_factory=list)

    def all_cards(self) -> list[Card]:
        """
        Deduplicated union of all *deck-input* pools.

        Note: `synergy` is the SOURCE pool for the synergy_engine pass
        (the broad recall union, up to 2500 cards) — it is intentionally
        NOT included here. The Phase 2 synergy_engine pass curates that
        pool down to a small bucket (`synergy_engine`) which IS a
        deck-input. Including `synergy` here would flood the GA with
        2500 unfiltered candidates and undo all the filtering work.
        """
        seen = set()
        out = []
        for cards in [
            self.ramp, self.draw, self.removal, self.wipe,
            self.lands, self.threats, self.protection,
            self.recursion, self.power_staples, self.synergy_engine,
        ]:
            for card in cards:
                if card.name not in seen:
                    seen.add(card.name)
                    out.append(card)
        return out

    def total_unique(self) -> int:
        return len(self.all_cards())


class DeckBuilder:
    """
    Main orchestrator for hybrid AI deck building.

    Usage (with API):
        builder = DeckBuilder("cards.csv", BuildConfig(commander_name="Lathiel, the Bounteous Dawn"))
        result = builder.build()
        print(result.best_deck.to_decklist())

    Usage (without API, using mocks for testing):
        builder = DeckBuilder(
            "cards.csv",
            BuildConfig(commander_name="Lathiel, the Bounteous Dawn"),
            llm_config=LLMConfig(mock_mode=True),
        )
        result = builder.build()
    """

    def __init__(
        self,
        card_database_path: str | Path,
        config: BuildConfig,
        llm_config: Optional[LLMConfig] = None,
        progress_callback: Optional[Callable[[BuildProgress], None]] = None,
        # Session 3 optional components — users can inject custom implementations
        edhrec_client: Optional[object] = None,
        embedding_scorer: Optional[object] = None,
        price_source: Optional[object] = None,
        # v0.4: Scryfall card/image source for HTML reports
        card_source: Optional[object] = None,
    ):
        self.config = config
        self.llm_config = llm_config or LLMConfig(model=config.llm_model)
        self.progress_callback = progress_callback
        self.card_database_path = Path(card_database_path)

        # Session 3 integrations (all optional, lazy-constructed)
        self._edhrec_client = edhrec_client
        self._embedding_scorer = embedding_scorer
        self._price_source = price_source
        # v0.4: lazy-constructed if use_images=True
        self._card_source = card_source
        # v0.5: lazy-constructed if flavor_art_tags or use_oracle_tag_validation
        self._tag_client = None
        self._flavor_tag_scorer = None

        # Lazy-loaded components
        self._db: Optional[CardDatabase] = None
        self._llm: Optional[LLMEngine] = None
        self._commander: Optional[Card] = None
        self._analysis: Optional[CommanderAnalysis] = None
        self._candidates: Optional[CandidatePool] = None
        self._synergy_cache: dict[str, float] = {}
        self._baseline_power_cache: dict[str, float] = {}
        self._edhrec_data: Optional[object] = None  # EDHRECCommanderData when loaded
        # v0.9.14: card name -> core effect class (from the LLM scoring pass);
        # feeds the consistency dimension. Empty in mock/embedding paths.
        self._card_effect_classes: dict[str, str] = {}
        # v0.9.14: the GA's final candidate pool, kept for the refinement
        # loop's alternatives list.
        self._ga_candidate_pool: list[Card] = []

        # v0.9.1: per-source recall membership, set by _build_synergy_pool
        # and consumed by _compute_synergy_hints. Empty sets when recall
        # is disabled — legacy pool building skips hint generation.
        self._edhrec_recall_names: set[str] = set()
        self._embedding_recall_names: set[str] = set()
        self._pattern_recall_names: set[str] = set()
        # v0.9.6: full per-card cosine-to-commander map (every embedded card,
        # not just the top-`limit`). Used to rank the synergy_engine pool and
        # anchor the top-tier bypass. Empty when embedding recall is disabled.
        self._embedding_recall_scores: dict[str, float] = {}
        # v0.9.33 (#26): pool-entry provenance — card name -> ordered list
        # of channels that introduced/selected it. Answers "why is this
        # card in the pool" (and, by absence, "why not") at a glance.
        self._pool_provenance: dict[str, list[str]] = {}

        # v0.9.7: LLM intrinsic card-power scores (name -> 0-100). Populated by
        # _phase_score_card_power (commander-independent, globally cached) and
        # consumed by the synergy_engine pre-rank + the baseline power cache.
        # Empty when card_power_mode == "off" or in mock mode — behavior then
        # is identical to legacy builds.
        self._card_power_scorer: Optional[object] = None
        self._card_power_scores: dict[str, float] = {}

        # v0.9.8: combo/engine detection. _combo_report holds the detected
        # combos + engine tags; _onramp_names is the set guaranteed into
        # the GA pool (Leak A). Empty when combo_mode == "off" / mock.
        self._combo_detector: Optional[object] = None
        self._synergy_score_cache: Optional[object] = None  # v0.9.31
        self._combo_report: Optional[ComboReport] = None
        self._onramp_names: set[str] = set()
        # v0.9.15: bracket partition of detected combos. Reward combos feed
        # the GA's combo dimension; banned ones (e.g. two-card infinites at
        # brackets 1-3) feed the constraint penalty instead.
        self._reward_combos: list = []
        self._banned_combos: list = []

        # v0.9.9: structural/attribute synergy (e.g. "vanilla matters").
        # Predicates from the commander analysis; matching card names get
        # recalled, on-ramped, and synergy-floored. Empty for text commanders.
        self._structural_predicates: list[str] = []
        self._structural_card_names: set[str] = set()
        # v0.9.10: True when a structural commander's synergy was scored by the
        # LLM rubric (commander-effect-aware) — the flat floor is then skipped
        # so the LLM's reasoned values stand. The floor is a fallback for the
        # embedding/mock path only.
        self._structural_scored_by_llm: bool = False

        self._start_time: Optional[float] = None

    @property
    def card_source(self):
        """Lazy-construct a card source for image/metadata lookups.

        Priority order:
        1. User-injected card_source (if provided to constructor) — respected.
        2. BulkCardSource if config.use_bulk_source is True — one-time download,
           fast lookups, preferred for serious use.
        3. ScryfallCardSource if config.use_images is True — per-card API calls,
           good for small decks or first-time runs.
        4. None otherwise.
        """
        if self._card_source is not None:
            return self._card_source

        # v0.6: bulk source preferred when enabled
        if self.config.use_bulk_source:
            from .scryfall_bulk import ScryfallBulkFetcher, BulkCardSource
            cache_dir = self.config.bulk_cache_dir or "./scryfall_bulk"
            fetcher = ScryfallBulkFetcher(
                cache_dir=cache_dir,
                offline=self.config.bulk_offline,
            )
            path = fetcher.ensure_bulk(self.config.bulk_type)
            if path is not None:
                source = BulkCardSource.load_from_file(path)
                if source is not None:
                    self._card_source = source
                    return self._card_source
            # Bulk failed; fall through to per-card source if images were also
            # requested, else return None
            logger.warning(
                "Bulk source unavailable (offline with no cache, or fetch failed)"
            )

        # v0.4: per-card source when images are enabled
        if self.config.use_images:
            from .scryfall_cards import ScryfallCardSource
            cache_dir = self.config.images_cache_dir or "./scryfall_cache"
            self._card_source = ScryfallCardSource(
                cache_dir=cache_dir,
                offline=self.config.images_offline,
            )
        return self._card_source

    @property
    def tag_client(self):
        """Lazy-construct a ScryfallTagClient if tag features are enabled.

        Activates when ANY of: flavor_art_tags is non-empty,
        use_oracle_tag_validation is set, or validate_roles_after_build
        is set. Reused across all tag-consuming features.
        """
        if self._tag_client is not None:
            return self._tag_client
        needs_tags = (
            bool(self.config.flavor_art_tags)
            or self.config.use_oracle_tag_validation
            or self.config.validate_roles_after_build  # v0.6
        )
        if not needs_tags:
            return None
        from .scryfall_tags import ScryfallTagClient
        cache_dir = self.config.tags_cache_dir or "./scryfall_tags_cache"
        self._tag_client = ScryfallTagClient(
            cache_dir=cache_dir,
            offline=self.config.tags_offline,
        )
        return self._tag_client

    @property
    def flavor_tag_scorer(self):
        """Lazy-construct a FlavorTagScorer if flavor_art_tags is set."""
        if self._flavor_tag_scorer is not None:
            return self._flavor_tag_scorer
        if not self.config.flavor_art_tags:
            return None
        from .flavor_tags import FlavorTagScorer
        client = self.tag_client
        if client is None:
            return None
        # Color-filter to commander identity if we know it
        color_id = None
        if self._commander is not None:
            color_id = self._commander.color_identity or None
        self._flavor_tag_scorer = FlavorTagScorer.create_if_configured(
            self.config.flavor_art_tags,
            client,
            color_identity=color_id,
        )
        return self._flavor_tag_scorer

    @property
    def db(self) -> CardDatabase:
        if self._db is None:
            self._report_progress("init", "loading_database", 0.0,
                                  f"Loading {self.card_database_path}...")
            self._db = CardDatabase(self.card_database_path)
            self._db.load()
            self._report_progress("init", "loading_database", 1.0,
                                  f"Loaded {self._db.card_count} cards")
        return self._db

    @property
    def llm(self) -> LLMEngine:
        if self._llm is None:
            self._llm = LLMEngine(self.llm_config)
        return self._llm

    def _report_progress(
        self,
        phase: str,
        step: str,
        progress: float,
        message: str,
    ):
        if self.progress_callback:
            elapsed = time.time() - self._start_time if self._start_time else 0.0
            self.progress_callback(BuildProgress(
                phase=phase, step=step, progress=progress,
                message=message, elapsed_seconds=elapsed,
            ))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> OptimizationResult:
        """Run the full deck building pipeline."""
        self._start_time = time.time()

        # v0.9.17: configure the Game Changer source before any bracket
        # enforcement runs. Precedence: explicit --game-changers file >
        # CSV isGameChanger column > embedded list.
        self._configure_game_changer_source()

        self._phase_commander_analysis()
        # v0.9.9: when the commander has a structural predicate, add the
        # complementary TEXT patterns BEFORE recall so payoffs that *reference*
        # the attribute (e.g. Ruxa for "vanilla") get recalled too — the
        # predicate itself only matches the attribute cards (the bodies).
        self._augment_patterns_for_structural()
        # EDHREC data fetch (optional, non-blocking). v0.3 used it for
        # synergy scoring; v0.8 also uses it for candidate recall.
        if self.config.use_edhrec or self.config.recall_use_edhrec:
            self._phase_fetch_edhrec()
        self._phase_generate_pools()
        # v0.9.9: structural/attribute recall — pull attribute-payoff cards
        # (vanilla creatures, etc.) into the pool BEFORE filtering. No-op when
        # the commander has no structural predicates.
        self._phase_structural_recall()
        # NEW v0.3: Budget filter (optional, runs before LLM filtering to save API calls)
        if self.config.budget_max_per_card is not None:
            self._phase_budget_filter()
        # v0.9.7: score intrinsic card power BEFORE filtering so it can feed
        # the synergy_engine pre-rank (recall). No-op when mode == "off".
        self._phase_score_card_power()
        # v0.9.8: detect combos/engines (uses power ranking) and pull any
        # missing combo pieces into recall, BEFORE filtering. No-op when
        # combo_mode == "off".
        self._phase_detect_combos()
        self._phase_llm_filtering()
        self._phase_synergy_scoring()
        result = self._phase_optimization()

        # v0.9.14: post-GA LLM refinement. The GA optimizes per-card averages
        # and count thresholds; this pass hands the ASSEMBLED deck to the LLM
        # for holistic set-level critique (redundancy, interaction spread,
        # role quality) and applies its swaps. No-op when refine_iterations
        # is 0 or in mock mode.
        if result.best_deck is not None:
            self._phase_llm_refinement(result)

        # v0.9.7: snow-basic normalization is decided on the FINAL deck, not
        # the candidate pool — the recall/role pools almost always contain
        # some snow-matters card, so a pool-level check never fires.
        if result.best_deck is not None:
            self._normalize_snow_basics(result.best_deck)

        if self.config.enable_llm_review:
            review = self._phase_llm_review(result)
            result.llm_review = review

        result.commander_analysis = self._analysis
        if self._combo_report is not None:
            result.combos = self._combo_report.combos

        # v0.9.15: bracket compliance audit of the FINAL deck (report-only —
        # enforcement already happened in the pool filter, GA penalty, and
        # refinement guard; this is the honest after-the-fact check).
        if result.best_deck is not None:
            from .bracket import audit_deck
            result.bracket_audit = audit_deck(
                result.best_deck,
                self._combo_report.combos if self._combo_report else [],
                getattr(self.config, "bracket", 4),
            )

        # v0.6: Post-build role validation (diagnostic, never affects score)
        if self.config.validate_roles_after_build:
            self._phase_validate_roles(result)

        # v0.9.6: total end-to-end wall-clock for the report (the GA-only
        # runtime_seconds understates a real build by ~100x).
        result.total_runtime_seconds = (
            time.time() - self._start_time if self._start_time else None
        )

        # v0.9.33 (#26): stamp pool-entry provenance onto the final deck's
        # telemetry. Done last so it survives refinement's telemetry rebuild.
        self._attach_provenance(result)

        # v0.9.16c: log the prompt-cache efficiency summary for the build.
        try:
            summary = self.llm.cache_summary()
            if summary:
                logger.info(summary)
        except Exception:
            pass

        self._report_progress("done", "complete", 1.0,
                              f"Complete! Score: {result.final_score:.1f}")
        return result

    def _attach_provenance(self, result) -> None:
        """v0.9.33 (#26): copy each card's pool-entry channels onto its
        telemetry row. A card in the final deck with EMPTY provenance entered
        via the base role skeleton (get_cards_for_role) — the default path —
        so absence is itself informative."""
        if not result or not getattr(result, "card_telemetry", None):
            return
        result.pool_provenance = dict(self._pool_provenance)
        for t in result.card_telemetry:
            prov = self._pool_provenance.get(t.name)
            if prov:
                t.provenance = list(prov)

    def quick_build(self) -> Deck:
        """
        Quick deck build without LLM or GA. For testing only.
        Uses pure heuristics to assemble a reasonable deck.
        """
        self._start_time = time.time()
        commander = self._find_commander()
        color_id = commander.color_identity

        cards = []
        seen_names = set()

        def add_cards(new_cards: list[Card], limit: int):
            added = 0
            for c in new_cards:
                if c.name in seen_names:
                    continue
                if added >= limit:
                    break
                cards.append(c)
                seen_names.add(c.name)
                added += 1

        # 37 lands
        lands = self.db.get_cards_for_role('land', color_id, limit=200)
        add_cards(lands, 37)

        # 12 ramp
        add_cards(self.db.get_cards_for_role('ramp', color_id, limit=60), 12)

        # 10 draw
        add_cards(self.db.get_cards_for_role('draw', color_id, limit=60), 10)

        # 8 removal
        add_cards(self.db.get_cards_for_role('removal', color_id, limit=60), 8)

        # Fill with threats
        threats = self.db.get_cards_for_role('threat', color_id, limit=100)
        add_cards(threats, 99 - len(cards))

        # Still short? Pad with basics if possible, otherwise any legal card
        if len(cards) < 99:
            basics = [c for c in self.db.all_cards
                      if c.is_basic_land and c.color_identity
                      and set(c.color_identity) & set(color_id)]
            if basics:
                while len(cards) < 99:
                    cards.append(basics[0])  # basics can duplicate
            else:
                all_legal = self.db.query(color_identity=color_id).cards
                remaining = [c for c in all_legal if c.name not in seen_names]
                while len(cards) < 99 and remaining:
                    cards.append(remaining.pop(0))
                    seen_names.add(cards[-1].name)

        return Deck(commander=commander, cards=cards[:99])

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _find_commander(self) -> Card:
        commander = self.db.get_by_name(self.config.commander_name)
        if commander is None:
            similar = self.db.find_similar_names(self.config.commander_name)
            if similar:
                raise ValueError(
                    f"Commander not found: '{self.config.commander_name}'.\n"
                    f"Did you mean:\n  - " + "\n  - ".join(similar)
                )
            raise ValueError(
                f"Commander not found: '{self.config.commander_name}'. "
                f"Database has {self.db.card_count} cards loaded."
            )
        return commander

    def _configure_game_changer_source(self) -> None:
        """v0.9.17: point bracket.py at the right Game Changer source.

        Order: an explicit --game-changers file (config.game_changers_file)
        replaces the embedded list; otherwise, if the loaded CSV carried an
        isGameChanger column, the per-card flags are authoritative; otherwise
        the embedded constant stands. Always resets first so one builder
        instance doesn't leak a prior override."""
        from . import bracket
        bracket.reset_game_changer_source()

        path = getattr(self.config, "game_changers_file", None)
        if path:
            names = bracket.load_game_changer_names(path)
            if names:
                bracket.set_game_changer_names(names)
                logger.info(
                    f"Game Changers: refreshed from {path} ({len(names)} cards)"
                )
                return
            logger.warning(
                f"Game Changers: could not load {path}; using embedded list"
            )

        if getattr(self.db, "has_game_changer_column", False):
            # The per-card CSV attribute is now the source (no override).
            n = sum(1 for c in self.db.all_cards if c.is_game_changer)
            logger.info(
                f"Game Changers: sourced from the CSV isGameChanger column "
                f"({n} flagged)"
            )
            return

        # No column and no override → NO Game Changer data. There's no
        # embedded list to fall back to (deleted in v0.9.18). Only matters at
        # brackets 1-3 (B4/5 have no GC limit), so warn rather than fail.
        if getattr(self.config, "bracket", 4) <= 3:
            logger.warning(
                "Game Changers: cards.csv has no isGameChanger column and no "
                "--game-changers file given — bracket GC enforcement is "
                "INACTIVE. Run `refresh-cards` to source it from MTGJSON."
            )

    def _phase_commander_analysis(self):
        self._report_progress("analysis", "finding_commander", 0.0,
                              f"Finding {self.config.commander_name}...")
        self._commander = self._find_commander()

        self._report_progress("analysis", "analyzing", 0.3,
                              f"Analyzing {self._commander.name}...")
        self._analysis = self.llm.analyze_commander(self._commander)
        mechanics_str = ", ".join(self._analysis.key_mechanics) or "(none identified)"
        self._report_progress("analysis", "complete", 1.0,
                              f"Mechanics: {mechanics_str}")
        logger.info(f"Commander analysis: {self._analysis.build_around_text}")

    def _phase_fetch_edhrec(self):
        """
        Fetch EDHREC data for the commander. Never blocks the pipeline on
        errors — EDHREC being down just means we fall back to LLM/heuristic.
        """
        self._report_progress("edhrec", "fetching", 0.0,
                              "Fetching EDHREC data...")

        if self._edhrec_client is None:
            from .edhrec_client import EDHRECClient
            self._edhrec_client = EDHRECClient(
                cache_dir=self.config.edhrec_cache_dir,
                offline=self.config.edhrec_offline,
            )

        try:
            self._edhrec_data = self._edhrec_client.fetch_commander(
                self._commander.name
            )
        except Exception as e:
            logger.warning(f"EDHREC fetch failed: {e}; continuing without")
            self._edhrec_data = None

        if self._edhrec_data is None:
            self._report_progress("edhrec", "complete", 1.0,
                                  "EDHREC unavailable (will use heuristic)")
        else:
            n = len(self._edhrec_data.cards)
            self._report_progress("edhrec", "complete", 1.0,
                                  f"EDHREC: {n} card entries")

    def _phase_budget_filter(self):
        """Filter candidate pools to stay within per-card budget."""
        if self._candidates is None:
            return  # No pools yet — shouldn't happen given call ordering

        max_price = self.config.budget_max_per_card
        if max_price is None:
            return

        self._report_progress(
            "budget", "filtering", 0.0,
            f"Applying budget: ${max_price:.2f}/card max...",
        )

        if self._price_source is None:
            from .price_source import ScryfallPriceSource
            # Cache under the EDHREC cache dir if set; otherwise memory-only
            cache_dir = None
            if self.config.edhrec_cache_dir:
                cache_dir = Path(self.config.edhrec_cache_dir) / "prices"
            self._price_source = ScryfallPriceSource(cache_dir=cache_dir)

        from .price_source import filter_cards_by_budget

        # Apply to every category in the pool
        before_total = self._candidates.total_unique()
        for field_name in ['ramp', 'draw', 'removal', 'wipe', 'lands',
                           'threats', 'protection', 'recursion', 'synergy']:
            original = getattr(self._candidates, field_name)
            filtered = filter_cards_by_budget(
                original,
                self._price_source,
                max_price_per_card=max_price,
                exclude_unknown=self.config.budget_exclude_unknown,
            )
            setattr(self._candidates, field_name, filtered)

        after_total = self._candidates.total_unique()
        self._report_progress(
            "budget", "complete", 1.0,
            f"Budget: {before_total} -> {after_total} candidates",
        )

    def _filter_late_additions_by_budget(self, cards: list, context: str) -> list:
        """Re-apply the per-card budget cap to cards added to the pool AFTER
        the budget phase ran (combo missing-piece recall, the GA on-ramp).
        Without this, those paths smuggle arbitrarily expensive cards past an
        active --budget. No-op when no budget cap is configured."""
        max_price = self.config.budget_max_per_card
        if max_price is None or not cards:
            return cards
        if self._price_source is None:
            from .price_source import ScryfallPriceSource
            cache_dir = None
            if self.config.edhrec_cache_dir:
                cache_dir = Path(self.config.edhrec_cache_dir) / "prices"
            self._price_source = ScryfallPriceSource(cache_dir=cache_dir)
        from .price_source import filter_cards_by_budget
        kept = filter_cards_by_budget(
            cards,
            self._price_source,
            max_price_per_card=max_price,
            exclude_unknown=self.config.budget_exclude_unknown,
        )
        if len(kept) != len(cards):
            dropped = {c.name for c in cards} - {c.name for c in kept}
            logger.info(
                f"Budget: dropped {len(dropped)} over-budget {context} "
                f"addition(s): {', '.join(sorted(dropped))}"
            )
        return kept

    def _phase_generate_pools(self):
        """Query the database for candidates in each role."""
        self._report_progress("pools", "starting", 0.0, "Querying database...")
        color_id = self._commander.color_identity

        # v0.9.5 (de-truncation): NO limit — every card that fills a role
        # flows into the LLM elimination tournament. Previously these were
        # capped at 300 (200 for protection/recursion, 100 for wipe), so the
        # LLM never saw cards beyond the cap. With no cap and no quality
        # pre-rank, the tournament reviews the entire role pool per the
        # "review every card" goal.
        pools = {
            'ramp': self.db.get_cards_for_role('ramp', color_id),
            'draw': self.db.get_cards_for_role('draw', color_id),
            'removal': self.db.get_cards_for_role('removal', color_id),
            'wipe': self.db.get_cards_for_role('wipe', color_id),
            'lands': self.db.get_cards_for_role('land', color_id),
            'threats': self.db.get_cards_for_role('threat', color_id),
            'protection': self.db.get_cards_for_role('protection', color_id),
            'recursion': self.db.get_cards_for_role('recursion', color_id),
        }

        candidates = CandidatePool(
            ramp=pools['ramp'],
            draw=pools['draw'],
            removal=pools['removal'],
            wipe=pools['wipe'],
            lands=pools['lands'],
            threats=pools['threats'],
            protection=pools['protection'],
            recursion=pools['recursion'],
        )

        # Synergy pool: layered recall (v0.8) or legacy substring (default).
        candidates.synergy = self._build_synergy_pool(color_id)

        self._candidates = candidates
        self._report_progress(
            "pools", "complete", 1.0,
            f"Found {candidates.total_unique()} unique candidates",
        )

    def _tag_provenance(self, cards_or_names, channel: str) -> None:
        """v0.9.33 (#26): record that `channel` introduced/selected these
        cards. Idempotent per (card, channel); preserves first-seen order."""
        for item in cards_or_names:
            name = item if isinstance(item, str) else item.name
            channels = self._pool_provenance.setdefault(name, [])
            if channel not in channels:
                channels.append(channel)

    def _build_synergy_pool(self, color_id: str) -> list:
        """
        Build the synergy candidate pool.

        If any of the v0.8 recall flags is enabled, union the configured
        sources (EDHREC, embeddings, LLM-expanded patterns) and cap the
        result at `recall_pool_cap`. Otherwise fall back to the legacy
        substring match against `analysis.synergy_keywords` so existing
        builds behave identically.

        Color-identity filtering is applied uniformly; each recall source
        also filters defensively in case its data drifts.
        """
        cfg = self.config
        recall_on = (
            cfg.recall_use_edhrec
            or cfg.recall_use_embeddings
            or cfg.recall_use_patterns
        )
        if not recall_on:
            pool = self._legacy_synergy_pool(color_id)
            self._tag_provenance(pool, "recall:keywords")
            return pool

        # v0.9.33 (#28): assembly extracted to recall_phase.build_recall_pool.
        from .recall_phase import build_recall_pool
        rr = build_recall_pool(
            db=self.db, config=cfg, analysis=self._analysis,
            edhrec_data=self._edhrec_data, color_id=color_id,
            progress=lambda stage, pct, msg: self._report_progress(
                "pools", stage, pct, msg),
        )
        # Retain the full cosine map (all embedded cards) so the
        # synergy_engine pre-rank can sort even cards that entered the
        # pool via EDHREC/patterns rather than the embedding cutoff.
        self._embedding_recall_scores = rr.embedding_scores

        # v0.9.1: per-source membership for hint computation. These sets
        # feed _compute_synergy_hints, which annotates every card in the
        # filtering phase's user prompts with [SYN+++]/[SYN++]/[SYN+] tags.
        self._edhrec_recall_names = rr.edhrec_names
        self._embedding_recall_names = rr.embedding_names
        self._pattern_recall_names = rr.pattern_names

        # v0.9.33 (#26): pool-entry provenance — only for cards that made
        # the capped union (membership in a source that got cut isn't entry).
        unioned = rr.cards
        in_union = {c.name for c in unioned}
        for names, channel in (
            (rr.edhrec_names, "recall:edhrec"),
            (rr.inclusion_names, "recall:edhrec-inclusion-b5"),
            (rr.embedding_names, "recall:embeddings"),
            (rr.pattern_names, "recall:patterns"),
        ):
            self._tag_provenance(names & in_union, channel)

        return unioned

    def _compute_synergy_hints(self) -> dict[str, str]:
        """
        Per-card synergy hint tags, derived from recall source membership.

        Each card is graded by how many of the recall sources flagged it
        (EDHREC community, embedding similarity, substring pattern match).
        The tiering is ADAPTIVE to how many sources actually contributed
        (`enabled` = sources with a non-empty recall set), so the top tier
        stays meaningful for commanders one source can't cover:

          enabled == 3 (all three produced cards):
            3 sources → "[SYN+++]"   all signals agree — commander-defining
            2 sources → "[SYN++]"    strong synergy candidate
            1 source  → "[SYN+]"     some commander-specific signal

          enabled == 2 (e.g. a NEW commander with no EDHREC data, so only
          embeddings + patterns fired):
            2 sources → "[SYN+++]"   both available signals agree
            1 source  → "[SYN+]"     single weak signal

          enabled == 1 (only one source on/productive):
            1 source  → "[SYN+]"     a lone source is weak evidence; never +++

          0 sources for a card → no entry (no special signal; let the LLM
          judge it on role-fit alone).

        Why adaptive: the old fixed rule required all THREE sources for
        [SYN+++], but EDHREC returns nothing for new/unpopular commanders —
        so the top tier (and the protect-the-best-payoffs bypass that keys
        off it) would silently be empty for exactly those commanders.
        Anchoring on "all AVAILABLE signals agree" makes the tier extensible
        to any commander while refusing to promote a single lone source to
        the top tier (which would over-inflate when only embeddings are on).

        Returns: dict mapping card.name -> hint tag. Names not in the dict
        have no tag (rendered without prefix).
        """
        edhrec = self._edhrec_recall_names
        embedding = self._embedding_recall_names
        pattern = self._pattern_recall_names

        enabled = sum(1 for s in (edhrec, embedding, pattern) if s)
        if enabled == 0:
            return {}

        all_signalled = edhrec | embedding | pattern
        hints: dict[str, str] = {}
        for name in all_signalled:
            sources = (
                (1 if name in edhrec else 0)
                + (1 if name in embedding else 0)
                + (1 if name in pattern else 0)
            )
            hints[name] = self._tier_for_sources(sources, enabled)
        return hints

    @staticmethod
    def _tier_for_sources(sources: int, enabled: int) -> str:
        """
        Map (source-hit-count, number-of-enabled-sources) → hint tag.

        See _compute_synergy_hints for the adaptive tiering rationale.
        """
        if enabled >= 3:
            if sources >= 3:
                return "[SYN+++]"
            if sources == 2:
                return "[SYN++]"
            return "[SYN+]"
        if enabled == 2:
            # All available signals agree → top tier; otherwise weak.
            return "[SYN+++]" if sources >= 2 else "[SYN+]"
        # enabled == 1: a single lone source never reaches the top tier.
        return "[SYN+]"

    # Tier → numeric rank for ordering (higher = more commander-specific).
    _TIER_RANK = {"[SYN+++]": 3, "[SYN++]": 2, "[SYN+]": 1}

    def _rank_synergy_engine_pool(
        self, pool: list, hints: dict[str, str]
    ) -> list:
        """
        Order the synergy_engine candidate pool by commander-relevance, most
        relevant first, for the v0.9.6 pre-rank + bypass.

        Sort key, descending:
          1. adaptive hint tier (+++ > ++ > + > untagged) — how many recall
             sources agreed the card is commander-specific.
          2. composite = cosine-to-commander + weight * (card_power/100) —
             commander fit ordered within a tier, with a SYNERGY-LED nudge
             from intrinsic card power (v0.9.7) so a strong card climbs but
             cosine still dominates which cards reach the shortlist/bypass.
          3. name — deterministic tie-break.

        Degrades gracefully: if embedding recall is off, every cosine is 0.0;
        if card power is off (or in mock mode) every power is 0.0 so the
        composite is just cosine — identical to the v0.9.6 behavior; if recall
        is off entirely, hints is empty and ordering is purely alphabetical
        (the pre-rank becomes a no-op rather than misordering anything).
        """
        hints = hints or {}
        cos = self._embedding_recall_scores
        power = self._card_power_scores
        weight = getattr(self.config, "card_power_recall_weight", 0.15)

        def composite(name: str) -> float:
            return cos.get(name, 0.0) + weight * (power.get(name, 0.0) / 100.0)

        return sorted(
            pool,
            key=lambda c: (
                -self._TIER_RANK.get(hints.get(c.name, ""), 0),
                -composite(c.name),
                c.name,
            ),
        )

    @staticmethod
    def _boost_synergy_by_hint(raw_score: float, hint: Optional[str]) -> float:
        """
        Combine an embedding cosine synergy score with the recall hint tag.

        The hint encodes how many independent recall sources agreed the card
        is commander-specific; the embedding cosine is a real but noisy
        per-card signal. Earlier this used `max(raw, floor)`, which SNAPPED
        nearly every tagged card to its floor (80/65/50) and erased the
        embedding nuance — two [SYN++] cards with very different cosines both
        landed on exactly 65, leaving the synergy dimension with no
        resolution.

        Instead we linearly remap the cosine score into the tier's band
        [floor, 100]:

            result = floor + (raw/100) * (100 - floor)

          [SYN+++] → band [80, 100]
          [SYN++]  → band [65, 100]
          [SYN+]   → band [50, 100]
          (none)   → floor 0 ⇒ result == raw (unchanged)

        So the floor is still respected (raw=0 ⇒ exactly floor) but a higher
        cosine always yields a higher score within the tier — restoring
        resolution. The mapping is monotonic in raw and never drops a card
        below the old `max(raw, floor)` value, so the change can only sharpen
        (not weaken) the synergy signal. Result clamped to 0-100.
        """
        floors = {"[SYN+++]": 80.0, "[SYN++]": 65.0, "[SYN+]": 50.0}
        floor = floors.get(hint or "", 0.0)
        raw = max(0.0, min(100.0, raw_score))
        result = floor + (raw / 100.0) * (100.0 - floor)
        return max(0.0, min(100.0, result))

    def _legacy_synergy_pool(self, color_id: str) -> list:
        """
        Pre-v0.8 substring-match path. Kept as the default so builds with
        all recall flags off behave identically to before.
        """
        if not self._analysis.synergy_keywords:
            return []

        commander_colors = set(ch for ch in (color_id or '') if ch in 'WUBRG')

        synergy_cards = []
        for keyword in self._analysis.synergy_keywords:
            if not keyword:
                continue
            for card in self.db.all_cards:
                if keyword.lower() in (card.text or '').lower():
                    card_colors = set(ch for ch in (card.color_identity or '')
                                      if ch in 'WUBRG')
                    if card_colors.issubset(commander_colors):
                        synergy_cards.append(card)

        seen = set()
        unique_synergy = []
        for c in synergy_cards:
            if c.name not in seen:
                seen.add(c.name)
                unique_synergy.append(c)
        return unique_synergy[:300]

    def _get_card_power_scorer(self):
        """Lazily construct the card-power scorer, or None if disabled.

        Returns None when card_power_mode == "off" or in mock mode (no real
        LLM), so callers degrade to the legacy heuristic baseline.
        """
        if getattr(self.config, "card_power_mode", "off") == "off":
            return None
        if bool(getattr(self.llm.config, "mock_mode", False)):
            return None
        if self._card_power_scorer is None:
            self._card_power_scorer = CardPowerScorer(
                self.llm,
                model=self.config.card_power_model,
                cache_dir=self.config.card_power_cache_dir,
                batch_size=self.config.card_power_batch_size,
            )
        return self._card_power_scorer

    def _phase_score_card_power(self):
        """v0.9.7: LLM intrinsic card-power scoring (commander-independent).

        Scores the synergy recall pool so power can feed the synergy_engine
        pre-rank (recall-feeding). Scores are cached globally on disk; the
        final-pool baseline pass in _phase_synergy_scoring reuses the cache.
        No-op when card_power_mode == "off" or in mock mode — legacy builds
        are unaffected.
        """
        scorer = self._get_card_power_scorer()
        if scorer is None:
            return

        pool = list(self._candidates.synergy or [])
        if not pool:
            return

        # Recall-feeding scope: top-N synergy candidates by cosine (cap=0=all).
        cap = getattr(self.config, "card_power_recall_cap", 0)
        if cap and len(pool) > cap:
            cos = self._embedding_recall_scores
            pool = sorted(pool, key=lambda c: -cos.get(c.name, 0.0))[:cap]

        self._report_progress("card_power", "scoring", 0.0,
                              f"Scoring card power ({len(pool)} cards)...")
        scores = scorer.score_cards(pool)
        self._card_power_scores.update(scores)
        if scores:
            vals = list(scores.values())
            logger.info(
                f"Card power scored: {len(scores)} cards "
                f"(min={min(vals):.0f}, max={max(vals):.0f}, "
                f"avg={sum(vals) / len(vals):.0f})"
            )
        self._report_progress("card_power", "complete", 1.0,
                              f"Card power: {len(scores)} scored")

    def _get_synergy_cache(self):
        """v0.9.31: lazy per-commander synergy-score cache. None when
        disabled (synergy_cache_dir=None) or in mock mode (tests must stay
        deterministic and file-free)."""
        if getattr(self.llm.config, "mock_mode", False):
            return None
        cache_dir = getattr(self.config, "synergy_cache_dir", None)
        if not cache_dir:
            return None
        if getattr(self, "_synergy_score_cache", None) is None:
            from .llm_engine import SYNERGY_SCORING_PROMPT
            from .synergy_cache import SynergyScoreCache
            self._synergy_score_cache = SynergyScoreCache(
                commander=self._commander.name if self._commander else "",
                model=self.llm.config.model,
                rubric=SYNERGY_SCORING_PROMPT,
                cache_dir=cache_dir,
            )
        return self._synergy_score_cache

    def _get_combo_detector(self):
        """Lazily construct the combo detector, or None if disabled/mock."""
        if getattr(self.config, "combo_mode", "off") == "off":
            return None
        if bool(getattr(self.llm.config, "mock_mode", False)):
            return None
        if self._combo_detector is None:
            self._combo_detector = ComboEngineDetector(
                self.llm,
                model=self.config.combo_model,
                cache_dir=self.config.combo_cache_dir,
                max_pool=self.config.combo_max_pool,
                signature_pass=getattr(
                    self.config, "combo_signature_pass", True),
            )
        return self._combo_detector

    def _phase_detect_combos(self):
        """v0.9.8: detect combos + engines, then pull missing combo pieces
        into recall so they become buildable.

        Runs after card-power scoring (so the pool pass sees the strongest
        cards) and before filtering (so the on-ramp + recall additions flow
        into the GA pool). No-op when combo_mode == "off" or in mock mode.
        """
        detector = self._get_combo_detector()
        if detector is None:
            return

        pool = list(self._candidates.synergy or [])
        if not pool:
            return

        # Pool pass analyzes the strongest synergy candidates first.
        power = self._card_power_scores
        pool_ranked = sorted(pool, key=lambda c: -power.get(c.name, 0.0))

        # v0.9.30: human-verified combo database (EDHREC surfaces Commander
        # Spellbook) as the deterministic backbone — the LLM passes cover
        # what databases can't (novel commanders, pool-specific engines).
        # Never blocks: any failure just means LLM-only detection.
        database_combos: list[dict] = []
        if not bool(getattr(self.llm.config, "mock_mode", False)):
            try:
                if self._edhrec_client is None:
                    from .edhrec_client import EDHRECClient
                    self._edhrec_client = EDHRECClient(
                        cache_dir=self.config.edhrec_cache_dir,
                        offline=self.config.edhrec_offline,
                    )
                database_combos = self._edhrec_client.fetch_combos(
                    self._commander.name)
            except Exception as e:
                logger.warning(f"EDHREC combo fetch failed: {e}; LLM-only")

        self._report_progress("combos", "detecting", 0.0,
                              f"Detecting combos ({len(pool_ranked)} candidates)...")
        report = detector.detect(self._analysis, pool_ranked,
                                 database_combos=database_combos)
        self._combo_report = report

        commander_colors = set(
            ch for ch in (self._commander.color_identity or "") if ch in "WUBRG"
        )

        # Prune combos that can NEVER be assembled: a piece that isn't in the
        # card DB (hallucinated/misspelled name) or is outside the commander's
        # color identity makes the whole combo unbuildable — yet it would
        # still earn near-complete partial credit in the GA fitness, paying
        # decks to hoard pieces of impossible combos (and cluttering the
        # report's "one piece away" section). The commander itself always
        # counts as present. In-memory only; the disk cache keeps the raw
        # detection so a future DB update can resurrect a pruned combo.
        commander_name = self._commander.name
        buildable = []
        pruned = 0
        for combo in report.combos:
            ok = True
            for name in combo.cards:
                if name == commander_name:
                    continue
                card = self.db.get_by_name(name)
                if card is None:
                    ok = False
                    break
                card_colors = set(
                    ch for ch in (card.color_identity or "") if ch in "WUBRG"
                )
                if not card_colors.issubset(commander_colors):
                    ok = False
                    break
            if ok:
                buildable.append(combo)
            else:
                pruned += 1
        if pruned:
            logger.info(
                f"Combo pruning: dropped {pruned} unbuildable combo(s) "
                f"(piece not in DB or off-color)"
            )
        report.combos = buildable

        # Recall feedback: resolve missing combo pieces (knowledge pass) to
        # real, color-legal DB cards and add them to the synergy pool so the
        # combo can actually be built. Respects the per-card budget cap when
        # one is active (the budget phase already ran).
        have = {c.name for c in self._candidates.synergy}
        addable = []
        for name in report.missing_pieces:
            if name in have:
                continue
            card = self.db.get_by_name(name)
            if card is None:
                continue
            card_colors = set(
                ch for ch in (card.color_identity or "") if ch in "WUBRG"
            )
            if not card_colors.issubset(commander_colors):
                continue
            addable.append(card)
        addable = self._filter_late_additions_by_budget(addable, "combo-recall")
        added = 0
        for card in addable:
            self._candidates.synergy.append(card)
            have.add(card.name)
            self._tag_provenance([card], "combo-recall")
            added += 1

        # Names to GUARANTEE into the GA pool (Leak A on-ramp): engine cards +
        # every card that participates in a detected combo, restricted to what
        # exists in our DB and is color-legal.
        onramp: set[str] = set()
        for name in list(report.engines.keys()) + list(report.all_combo_card_names()):
            card = self.db.get_by_name(name)
            if card is None:
                continue
            card_colors = set(
                ch for ch in (card.color_identity or "") if ch in "WUBRG"
            )
            if card_colors.issubset(commander_colors):
                onramp.add(name)
        # UNION (not assign) — structural recall may have already flagged
        # attribute cards (vanilla creatures, etc.) for the on-ramp; don't
        # clobber them.
        self._onramp_names |= onramp

        # v0.9.15: partition combos by bracket policy. Banned combos (e.g.
        # two-card infinites at brackets 1-3) must never be REWARDED — they
        # feed the GA's constraint penalty instead, and the compliance audit.
        from .bracket import two_card_combo_banned
        bracket = getattr(self.config, "bracket", 4)

        def _mv_of(name: str):
            card = self.db.get_by_name(name)
            return card.mana_value if card is not None else 0

        self._banned_combos = [
            c for c in report.combos
            if two_card_combo_banned(c, bracket, _mv_of)
        ]
        banned_keys = {frozenset(c.cards) for c in self._banned_combos}
        self._reward_combos = [
            c for c in report.combos if frozenset(c.cards) not in banned_keys
        ]
        if self._banned_combos:
            logger.info(
                f"Bracket {bracket}: {len(self._banned_combos)} detected "
                f"combo(s) are banned at this bracket — moved from reward "
                f"to penalty (e.g. "
                f"{' + '.join(self._banned_combos[0].cards)})"
            )

        logger.info(
            f"Combo phase: {len(report.combos)} combos, {len(report.engines)} "
            f"engines, pulled {added} missing pieces into recall, "
            f"{len(onramp)} cards flagged for the GA on-ramp"
        )
        self._report_progress("combos", "complete", 1.0,
                              f"Combos: {len(report.combos)} found")

    def _phase_llm_filtering(self):
        """LLM picks the top N candidates per role, creating a curated pool."""
        self._report_progress("filtering", "starting", 0.0,
                              "LLM filtering pools...")

        target = self.config.candidates_per_category
        already_selected: set[str] = set()

        # v0.9: synergy is NOT a peer role anymore. We fill traditional role
        # buckets first (Phase 1), each with the cross-cutting "prefer
        # synergistic candidates" instruction baked into CARD_SELECTION_PROMPT.
        # Then a Phase 2 synergy_engine pass picks strategy-defining engine
        # pieces from the broader recall pool minus what's already selected.
        # The synergy bucket itself is removed from this loop.
        categories = [
            ('ramp', self._candidates.ramp, target),
            ('draw', self._candidates.draw, target),
            ('removal', self._candidates.removal, target),
            ('threats', self._candidates.threats, target),
            ('protection', self._candidates.protection, 50),
            ('recursion', self._candidates.recursion, 50),
            ('wipe', self._candidates.wipe, 30),
            ('lands', self._candidates.lands, 80),
        ]

        filtered = CandidatePool()
        # Preserve the unioned synergy recall pool — Phase 2 reads from it.
        filtered.synergy = self._candidates.synergy
        # +1 for the synergy_engine phase below (progress reporting only).
        total_cats = len(categories) + (1 if self.config.synergy_engine_target > 0 else 0)

        # v0.9.1: per-card synergy hint tags derived from recall source
        # membership. Passed to every select_cards call so the LLM weights
        # commander-specific cards (Heliod, Soul Warden, etc.) above
        # equally-fit untagged candidates in their role bucket. Empty dict
        # when recall is disabled — legacy builds get the old un-hinted
        # prompts unchanged.
        synergy_hints = self._compute_synergy_hints()
        if synergy_hints:
            tag_dist = {"[SYN+++]": 0, "[SYN++]": 0, "[SYN+]": 0}
            for tag in synergy_hints.values():
                tag_dist[tag] = tag_dist.get(tag, 0) + 1
            logger.info(
                f"Synergy hints computed: {len(synergy_hints)} tagged cards "
                f"(+++ {tag_dist['[SYN+++]']}, "
                f"++ {tag_dist['[SYN++]']}, "
                f"+ {tag_dist['[SYN+]']})"
            )

        for i, (cat_name, pool, target_count) in enumerate(categories):
            self._report_progress(
                "filtering", cat_name,
                i / total_cats,
                f"Filtering {cat_name} ({len(pool)} → {target_count})...",
            )

            if len(pool) <= target_count:
                selected_names = [c.name for c in pool]
            else:
                selected_names = self.llm.select_cards(
                    self._analysis, pool, role=cat_name,
                    count=target_count,
                    already_selected=already_selected,
                    synergy_hints=synergy_hints or None,
                )

            name_to_card = {c.name: c for c in pool}
            selected_cards = [
                name_to_card[n] for n in selected_names if n in name_to_card
            ]

            # v0.9.15b: power bypass — the bucket's top-N by cached intrinsic
            # power join ADDITIVELY, so the tournament can't eliminate the
            # format's best role-fillers (observed: Llanowar Elves, Force of
            # Negation funnel-cut in a real cEDH run). Only fires when a
            # tournament actually cut the pool and power scores exist.
            bypass_n = getattr(self.config, "role_power_bypass", 0)
            if bypass_n > 0 and len(pool) > target_count:
                # v0.9.19: rank by the GLOBAL power cache, not the
                # recall-scoped _card_power_scores. Role buckets draw from
                # the whole DB, but _card_power_scores only covers the recall
                # union — so a bucket card outside recall was invisible to
                # the bypass no matter how strong (observed: Sol Ring pow 98
                # funnel-cut from the Jodah ramp bucket while pow-82 cards
                # were rescued). The recall-scoped dict overlays the cache so
                # fresh scores win over stale cached ones.
                power = {}
                scorer = self._get_card_power_scorer()
                if scorer is not None:
                    power.update(scorer.cached_scores())
                power.update(self._card_power_scores)
                chosen = {c.name for c in selected_cards}
                by_power = sorted(
                    (c for c in pool if c.name in power),
                    key=lambda c: -power[c.name],
                )
                rescued = [
                    c for c in by_power[:bypass_n] if c.name not in chosen
                ]
                if rescued:
                    self._tag_provenance(rescued, f"role-bypass:{cat_name}")
                    selected_cards = selected_cards + rescued
                    logger.info(
                        f"Power bypass [{cat_name}]: rescued "
                        f"{len(rescued)} top-power card(s) the tournament "
                        f"cut: {', '.join(c.name for c in rescued)}"
                    )

            self._tag_provenance(
                (c for c in selected_cards
                 if f"role-bypass:{cat_name}"
                 not in self._pool_provenance.get(c.name, [])),
                f"role:{cat_name}")
            setattr(filtered, cat_name, selected_cards)
            already_selected.update(c.name for c in selected_cards)

        # Phase 2: synergy_engine pass.
        # Input pool: the broad recall union, minus everything chosen above
        # AND minus the commander itself.
        # Output: ~synergy_engine_target strategy-defining engine pieces.
        #
        # v0.9.6: instead of running a full elimination tournament over the
        # entire recall union (slow, and a coarse early round could drop the
        # best payoffs), we PRE-RANK the pool by (adaptive hint tier,
        # cosine-to-commander). The top `synergy_engine_bypass` cards skip the
        # LLM entirely (guaranteed into the GA pool); the LLM then picks the
        # remaining slots from only the top `synergy_engine_shortlist` cards.
        if self.config.synergy_engine_target > 0 and self._candidates.synergy:
            commander_name = self._commander.name if self._commander else None
            engine_pool = [
                c for c in self._candidates.synergy
                if c.name not in already_selected
                and c.name != commander_name
            ]

            ranked = self._rank_synergy_engine_pool(engine_pool, synergy_hints)

            target = self.config.synergy_engine_target
            bypass_n = max(0, min(self.config.synergy_engine_bypass,
                                  target, len(ranked)))
            bypassed = ranked[:bypass_n]
            shortlist = ranked[bypass_n:
                               bypass_n + self.config.synergy_engine_shortlist]
            remaining = max(0, target - len(bypassed))

            self._report_progress(
                "filtering", "synergy_engine",
                (total_cats - 1) / total_cats,
                f"Synergy engine pass ({len(engine_pool)} ranked → "
                f"{bypass_n} bypass + {remaining} of {len(shortlist)} via LLM)...",
            )
            logger.info(
                f"synergy_engine pass: {len(engine_pool)} candidates "
                f"(after subtracting {len(already_selected)} already-selected); "
                f"pre-ranked → {bypass_n} bypassed straight to GA, LLM picks "
                f"{remaining} from a {len(shortlist)}-card shortlist "
                f"(target={target})"
            )
            if bypassed:
                logger.info(
                    "synergy_engine bypass (guaranteed): "
                    + ", ".join(c.name for c in bypassed)
                )

            engine_cards = list(bypassed)
            if shortlist and remaining > 0:
                engine_names = self.llm.select_synergy_engine_cards(
                    self._analysis,
                    shortlist,
                    count=remaining,
                    already_selected=already_selected,
                    synergy_hints=synergy_hints or None,
                )
                name_to_card = {c.name: c for c in shortlist}
                engine_cards += [
                    name_to_card[n] for n in engine_names if n in name_to_card
                ]

            self._tag_provenance(bypassed, "engine-bypass")
            self._tag_provenance(
                (c for c in engine_cards if c not in bypassed),
                "engine-pick")
            filtered.synergy_engine = engine_cards
            already_selected.update(c.name for c in engine_cards)

        # v0.9.8: Leak A on-ramp.
        self._apply_onramp(filtered)

        # v0.9.16: global power-staples channel — see _apply_power_staples.
        self._apply_power_staples(filtered)

        self._candidates = filtered
        self._report_progress(
            "filtering", "complete", 1.0,
            f"Filtered to {filtered.total_unique()} candidates",
        )

    def _apply_power_staples(self, filtered) -> None:
        """v0.9.16: the GLOBAL power-staples channel, IN PLACE.

        Pool entry used to require matching a hand-written role regex or
        being theme-relevant to recall — so generically-strong cards that
        are neither (stax, theft, cost reducers, and formerly tutors and
        clones) had NO channel until a category was invented for them.
        This is the general fix: the top-N color-legal cards by GLOBAL
        cached card power join the pool directly, regardless of taxonomy.

        Guardrails: engine-native signal only (the LLM's own power cache,
        not a name list); pool ENTRY only (the synergy rubric scores them
        honestly and the GA decides); the per-card budget cap and the
        bracket pool filters still apply downstream. No-op when card power
        is off/mock or the cache is empty.
        """
        limit = getattr(self.config, "power_staples_limit", 0)
        if limit <= 0:
            return
        scorer = self._get_card_power_scorer()
        if scorer is None:
            return
        cache = scorer.cached_scores()
        if not cache:
            return

        commander_name = self._commander.name if self._commander else ""
        commander_colors = set(
            ch for ch in (self._commander.color_identity or "") if ch in "WUBRG"
        )
        have = {c.name for c in filtered.all_cards()}
        ranked: list[tuple[float, str]] = []
        for name, power in cache.items():
            if name in have or name == commander_name:
                continue
            ranked.append((power, name))
        ranked.sort(key=lambda t: (-t[0], t[1]))

        staples: list[Card] = []
        for power, name in ranked:
            if len(staples) >= limit:
                break
            card = self.db.get_by_name(name)
            if card is None:
                continue
            card_colors = set(
                ch for ch in (card.color_identity or "") if ch in "WUBRG"
            )
            if not card_colors.issubset(commander_colors):
                continue
            staples.append(card)

        staples = self._filter_late_additions_by_budget(staples, "power-staples")
        if staples:
            filtered.power_staples = staples
            self._tag_provenance(staples, "power-staples")
            logger.info(
                f"Power staples: {len(staples)} top-cached-power cards "
                f"joined the pool (taxonomy-independent): "
                f"{', '.join(c.name for c in staples[:10])}"
                + (" ..." if len(staples) > 10 else "")
            )

    def _apply_onramp(self, filtered) -> None:
        """Leak A on-ramp: guarantee detected engines + combo pieces into the
        GA pool (via the synergy_engine bucket) IN PLACE.

        Engines and combo pieces otherwise fight the whole recall union for a
        handful of synergy_engine slots. all_cards() dedups by name, so a card
        already in a role bucket is unaffected; we only add the genuinely
        missing ones.
        """
        if not self._onramp_names:
            return
        commander_name = self._commander.name if self._commander else None
        have = {c.name for c in filtered.all_cards()}
        onramp_cards = []
        for name in sorted(self._onramp_names):
            if name in have or name == commander_name:
                continue
            card = self.db.get_by_name(name)
            if card is not None:
                onramp_cards.append(card)
                have.add(name)
        # On-ramp additions bypass the earlier budget phase — re-apply the cap.
        onramp_cards = self._filter_late_additions_by_budget(
            onramp_cards, "on-ramp",
        )
        if onramp_cards:
            self._tag_provenance(onramp_cards, "combo-onramp")
            filtered.synergy_engine = list(filtered.synergy_engine) + onramp_cards
            logger.info(
                f"Combo on-ramp: guaranteed {len(onramp_cards)} engine/combo "
                f"card(s) into the GA pool"
            )

    # Complementary text patterns for attribute archetypes: the predicate
    # matches the attribute cards (vanilla bodies), these match the PAYOFFS
    # that reference the attribute (Ruxa-style anthems/recursion).
    _STRUCTURAL_PATTERN_HINTS = {
        "vanilla": ["no abilities", "with no abilities", "vanilla"],
        "no_abilities": ["no abilities", "with no abilities", "vanilla"],
        "colorless": ["colorless"],
    }

    @staticmethod
    def _predicate_key(pred: str) -> str:
        """Bare predicate name (drop any :value / operator/number tail)."""
        return re.split(r"[:<>=]", pred.strip().lower(), maxsplit=1)[0].strip()

    def _augment_patterns_for_structural(self) -> None:
        """Add complementary text patterns for the commander's structural
        predicates so attribute-PAYOFFS (which have text) are recalled. Runs
        before recall; no-op when there are no predicates or mode == off."""
        if getattr(self.config, "structural_synergy_mode", "on") == "off":
            return
        if self._analysis is None:
            return
        preds = derive_structural_predicates(self._analysis)
        self._structural_predicates = preds
        extra: list[str] = []
        for p in preds:
            extra += self._STRUCTURAL_PATTERN_HINTS.get(self._predicate_key(p), [])
        if not extra:
            return
        existing = list(self._analysis.synergy_patterns or [])
        have = {e.lower() for e in existing}
        added = [e for e in extra if e.lower() not in have]
        if added:
            self._analysis.synergy_patterns = existing + added
            logger.info(
                f"Structural patterns: added {added} so attribute-payoffs "
                f"(e.g. Ruxa) are recalled"
            )

    def _phase_structural_recall(self):
        """v0.9.9: structural/attribute recall.

        For commanders whose payoff is a card ATTRIBUTE (vanilla creatures,
        colorless, low-curve, stats), the text-based recall + synergy signals
        are blind. This derives the structural predicates, pulls color-legal
        matching cards into the synergy pool, and flags them to be on-ramped
        into the GA pool. Their synergy is floored later (_apply_structural_
        boost). No-op when there are no predicates or mode == "off".
        """
        if getattr(self.config, "structural_synergy_mode", "on") == "off":
            return
        if self._analysis is None:
            return
        preds = derive_structural_predicates(self._analysis)
        self._structural_predicates = preds
        if not preds:
            return

        color_id = self._commander.color_identity if self._commander else ""
        cap = getattr(self.config, "structural_recall_cap", 80)
        matches = self.db.get_cards_matching_predicates(preds, color_id, limit=cap)

        have = {c.name for c in (self._candidates.synergy or [])}
        if matches:
            self._structural_card_names = {c.name for c in matches}
            # Add to the synergy recall pool (the GA-input universe) if missing.
            added = 0
            for c in matches:
                if c.name not in have:
                    self._candidates.synergy.append(c)
                    have.add(c.name)
                    added += 1
            self._tag_provenance(matches, "structural")
            # Guarantee them into the GA pool via the on-ramp.
            self._onramp_names |= self._structural_card_names
            logger.info(
                f"Structural recall {preds}: {len(matches)} attribute-matching "
                f"cards ({added} new to pool), flagged for on-ramp + synergy floor"
            )
        else:
            logger.info(f"Structural recall ({preds}): no matching cards")

        # v0.9.13: also pull the attribute-PAYOFF cards — text that REFERENCES
        # the attribute (Muraganda Petroglyphs' "creatures with no abilities",
        # Ruxa's recursion) — and guarantee them via the on-ramp. Previously
        # payoffs only reached the GA pool by winning the synergy_engine
        # cosine pre-rank race, and run-to-run analysis noise dropped
        # Petroglyphs (the archetype's defining payoff) from an entire build.
        # They are NOT added to _structural_card_names: they have real rules
        # text, so the LLM rubric scores them honestly — no floors needed.
        payoffs = self._structural_payoff_cards(preds, color_id, cap=30)
        if payoffs:
            new_names = []
            for c in payoffs:
                if c.name not in have:
                    self._candidates.synergy.append(c)
                    have.add(c.name)
                    new_names.append(c.name)
            self._onramp_names |= {c.name for c in payoffs}
            logger.info(
                f"Structural payoff recall {preds}: {len(payoffs)} "
                f"attribute-referencing cards ({len(new_names)} new to pool), "
                f"guaranteed via on-ramp"
            )

    def _structural_payoff_cards(self, preds, color_id, cap: int = 30) -> list:
        """Color-legal cards whose rules TEXT references the commander's
        structural attribute (via _STRUCTURAL_PATTERN_HINTS), ranked by hit
        count then mana value. These are the Ruxa/Petroglyphs class — the
        payoffs an attribute deck exists to play."""
        patterns: list[str] = []
        for p in preds or []:
            patterns += self._STRUCTURAL_PATTERN_HINTS.get(self._predicate_key(p), [])
        patterns = [pt.lower() for pt in dict.fromkeys(patterns)]
        if not patterns:
            return []
        from .candidate_recall import _within_commander_colors, _color_identity_set
        commander_colors = _color_identity_set(color_id)
        scored = []
        for card in self.db.all_cards:
            text = (card.text or "").lower()
            if not text:
                continue
            hits = sum(1 for pt in patterns if pt in text)
            if hits == 0:
                continue
            if not _within_commander_colors(card, commander_colors):
                continue
            scored.append((-hits, card.mana_value, card.name, card))
        scored.sort(key=lambda t: t[:3])
        return [t[3] for t in scored[:cap]]

    def _apply_structural_boost(self, synergy: dict) -> None:
        """v0.9.9: floor the synergy of structural-predicate matches so
        attribute-payoffs (text-less, hence ~0 from text signals) compete for
        deck slots. IN PLACE; only raises, never lowers.

        Skipped when the LLM rubric already scored these cards (v0.9.10): the
        commander-effect-aware rubric reasons each card's value (a vanilla 5/5
        under Jasmine vs a small vanilla 1/1), which a flat floor would clobber.
        The floor is the FALLBACK for the embedding/mock path only."""
        if getattr(self.config, "structural_synergy_mode", "on") == "off":
            return
        if self._structural_scored_by_llm:
            return
        names = self._structural_card_names
        if not names:
            return
        floor = getattr(self.config, "structural_boost_floor", 85.0)
        boosted = 0
        for name in names:
            if name in synergy and floor > synergy[name]:
                synergy[name] = floor
                boosted += 1
        if boosted:
            logger.info(
                f"Structural boost: floored synergy on {boosted} "
                f"attribute-matching card(s) to {floor:.0f}"
            )

    def _apply_edhrec_floor(self, synergy: dict) -> None:
        """v0.9.12: floor synergy to EDHREC's commander-distinctive signal,
        IN PLACE: synergy = max(reasoned, edhrec_floor * distinctive_0_100).

        Keys on the RAW EDHREC synergy metric (how much more often this card
        appears under THIS commander than baseline, typically 0..1) and ONLY
        when it is positive:

          - synergy <= 0 or missing → no floor. A card that merely appears in
            EDHREC data has no distinctive signal; flooring it would smear the
            reasoned rubric's low bands. Negative synergy (community avoids
            the card here) must never be a boost.
          - The inclusion-rate fallback is deliberately NOT used: inclusion is
            the precon/popularity-biased metric this project excludes.
          - positive synergy maps 0→50, +1.0→100 before the factor, so with
            the default factor 0.75 the strongest package cards floor to ~75
            (competitive) while mild signals floor to ~40s (below reasoned
            on-plan scores — no effect).

        Boost-only (max): never lowers, so pricey/unpopular cards EDHREC
        under-represents are protected. No-op when the factor is 0 or no
        EDHREC data was fetched (new/unpopular commanders)."""
        factor = getattr(self.config, "edhrec_floor", 0.0)
        if factor <= 0 or self._edhrec_data is None:
            return
        factor = max(0.0, min(1.0, factor))
        floored = 0
        for name in list(synergy.keys()):
            entry = self._edhrec_data.cards.get(name)
            if entry is None:
                continue
            raw = getattr(entry, "synergy", None)
            if raw is None or raw <= 0:
                continue  # no positive distinctive signal — never floor
            distinctive = 50.0 + 50.0 * min(1.0, raw)
            target = factor * distinctive
            if target > synergy[name]:
                synergy[name] = target
                floored += 1
        if floored:
            logger.info(
                f"EDHREC floor (factor={factor:.2f}): surfaced {floored} "
                f"community-distinctive cards"
            )

    @staticmethod
    def _body_power(card) -> Optional[float]:
        """Commander-effect-aware power for a creature whose BODY is the
        payoff (vanilla → unblockable, tribal → pumped, big-stats matters).

        The global card-power signal rates such a creature on its vacuum value
        ("a 10/10 that does nothing is mediocre"), but under a body-matters
        commander its in-play power is its combat presence. Derive that from
        stats (with a mild penalty for being expensive). Returns None for
        non-creatures or non-numeric P/T (the floor then doesn't apply)."""
        try:
            p = int(card.power)
            t = int(card.toughness)
        except (TypeError, ValueError):
            return None
        stats = p + t
        mv = max(0, card.mana_value)
        score = 12 + stats * 3.4 - max(0, mv - 5) * 2.0
        return max(0.0, min(100.0, score))

    def _apply_structural_power_floor(self, baseline: dict) -> None:
        """v0.9.11: floor the BASELINE power of attribute-matching creatures to
        their body value, so the commander's transformation isn't dragged down
        by the (commander-independent) card-power score. IN PLACE; only raises.

        Distinct from synergy: synergy = 'fits the plan'; this = 'how strong is
        the card in play once the commander transforms it'. A vanilla 10/10
        under Jasmine is legitimately high on BOTH (powerful AND on-plan)."""
        if getattr(self.config, "structural_synergy_mode", "on") == "off":
            return
        names = self._structural_card_names
        if not names:
            return
        boosted = 0
        for name in names:
            card = self.db.get_by_name(name)
            if card is None or not card.is_creature:
                continue
            bp = self._body_power(card)
            if bp is None:
                continue
            if bp > baseline.get(name, 0.0):
                baseline[name] = bp
                boosted += 1
        if boosted:
            logger.info(
                f"Structural power floor: lifted baseline on {boosted} "
                f"attribute-matching creature(s) to reflect their body"
            )

    def _apply_engine_boost(self, synergy: dict, baseline: dict) -> None:
        """v0.9.8: raise LLM-detected engines' synergy so they compete for
        deck slots. IN PLACE on `synergy`.

        - mode "floor": synergy = max(synergy, engine_boost_floor) — flat lift.
        - mode "power": synergy = max(synergy, the card's own power score) —
          quality-scaled, so strong engines rise and weak ones don't. When the
          card has NO power score (card_power_mode off), falls back to the
          flat floor — otherwise "power" mode would silently boost nothing.

        Only raises (never lowers) and only for cards already in the pool, so a
        boosted engine still has to beat the rest of the deck on total fitness;
        it's a competitive lift, not an auto-include.
        """
        mode = getattr(self.config, "engine_boost_mode", "off")
        if mode == "off" or self._combo_report is None:
            return
        engines = self._combo_report.engines
        if not engines:
            return
        floor = getattr(self.config, "engine_boost_floor", 80.0)
        boosted = 0
        for name in engines:
            if name not in synergy:
                continue
            if mode == "power":
                # A genuine (even low) power score stands — that's the point
                # of quality scaling. Only a MISSING score falls back.
                target = baseline[name] if name in baseline else floor
            else:  # "floor"
                target = floor
            if target > synergy[name]:
                synergy[name] = target
                boosted += 1
        if boosted:
            logger.info(
                f"Engine boost ({mode}): lifted synergy on {boosted} of "
                f"{len(engines)} detected engine card(s)"
            )

    def _phase_synergy_scoring(self):
        """
        Pre-compute the synergy and baseline (power) scores for all candidates.

        SYNERGY ("does this card fit the commander's plan", 0-100):
          1. Reasoned base: embeddings (topical, cheap) OR the LLM rubric
             (commander-effect-aware). Structural/attribute commanders are
             routed to the LLM here — embeddings can't reason about a
             commander transforming a card's attributes. Heuristic fills gaps.
          2. EDHREC floor (boost-only): surface the commander's community-
             DISTINCTIVE staple package without penalizing pricey cards.
          3. Engine boost: lift LLM-detected engines so repeatable payoffs
             compete for slots.
          4. Structural boost: floor attribute-payoff synergy (vanilla, etc.) —
             SKIPPED when the LLM already reasoned those cards (it's then a
             fallback for the embedding/mock path).

        BASELINE ("how strong is the card in play", 0-100):
          - LLM card power (commander-independent, globally cached) else a
            mana/staple heuristic.
          - Structural POWER floor: for attribute-matching creatures whose body
            is the payoff (Jasmine → unblockable), floor baseline to the body
            value so a vanilla 10/10 isn't dragged by its vacuum power.

        All boosts/floors are raise-only — none ever lowers a reasoned score.
        """
        self._report_progress("scoring", "starting", 0.0,
                              "Computing synergy scores...")

        from .embedding_scorer import is_embeddings_available

        all_cards = self._candidates.all_cards()
        synergy: dict[str, float] = {}
        baseline: dict[str, float] = {}

        # v0.9.4: synergy hints (recall-source signal) drive both the
        # embedding-boost and the LLM calibration. Compute once.
        scoring_hints = self._compute_synergy_hints()

        # Decide the scoring path. "auto" prefers embeddings when available
        # (eliminates ~30 LLM calls/build); "llm" forces the rubric;
        # "embedding" forces cosine-only. The legacy use_embeddings flag
        # still enables the embedding layer.
        #
        # In mock mode we deliberately SKIP embeddings: mock builds should
        # be fast and deterministic, and the LLM-mock path returns a stable
        # heuristic synergy score. Loading sentence-transformers in mock
        # mode would be slow and pointless. (The explicit use_embeddings
        # flag still forces it if a test really wants the embedding path.)
        mock_mode = bool(getattr(self.llm.config, "mock_mode", False))
        mode = getattr(self.config, "synergy_scoring_mode", "auto")

        # v0.9.10: structural/attribute commanders (vanilla, tribal, etc.) need
        # REASONED value — the card's worth once the commander's effect is
        # applied to its attributes — which embeddings (topical similarity)
        # fundamentally can't provide and actively get wrong (they over-rate
        # off-theme "big beaters"). Force the LLM rubric (now commander-effect-
        # aware) for them, unless we're in mock mode (no real LLM).
        self._structural_scored_by_llm = False
        if self._structural_predicates and not mock_mode and mode != "llm":
            logger.info(
                "Structural commander: routing synergy to the LLM rubric "
                "(value must be reasoned, not topical embedding similarity)"
            )
            mode = "llm"
        if self._structural_predicates and mode == "llm" and not mock_mode:
            self._structural_scored_by_llm = True

        embeddings_ok = is_embeddings_available()
        use_embeddings = (
            mode != "llm"
            and (embeddings_ok or self.config.use_embeddings)
            and (not mock_mode or self.config.use_embeddings)
        )

        # Layer 1: Embedding-based scoring (fast) + hint-tier boost.
        if use_embeddings:
            self._report_progress("scoring", "embeddings", 0.2,
                                  "Embedding synergy + hint boost...")
            embedding_scores = self._run_embedding_scorer(all_cards)
            for name, raw in embedding_scores.items():
                synergy[name] = self._boost_synergy_by_hint(
                    raw, scoring_hints.get(name)
                )
            logger.info(
                f"Embedding+hint scores for {len(embedding_scores)} cards "
                f"(mode={mode})"
            )

        # Layer 1.5 (v0.9.7): LLM intrinsic card power → baseline. Commander-
        # independent and globally cached, so this reuses the recall-feeding
        # scores and only newly scores final-pool cards the earlier phase
        # didn't reach. Replaces the flat mana-curve heuristic baseline (which
        # pinned Power Level ~52) and feeds the effective_synergy baseline term
        # so filler is down-weighted everywhere.
        power_scorer = self._get_card_power_scorer()
        if power_scorer is not None:
            power_scores = power_scorer.score_cards(all_cards)
            self._card_power_scores.update(power_scores)
            for name, p in power_scores.items():
                baseline[name] = p
            logger.info(
                f"Card-power baseline set for {len(power_scores)} cards "
                f"(model={self.config.card_power_model})"
            )

        # NOTE: EDHREC is no longer applied here as an OVERRIDE. It is now a
        # boost-only FLOOR applied AFTER the reasoned synergy below (embeddings/
        # LLM) — see _apply_edhrec_floor. An override here would (a) be
        # clobbered by the LLM layer in "llm" mode, and (b) replace the
        # commander-effect reasoning rather than surface staples on top of it.

        # Layer 3: LLM scoring.
        #   - mode == "llm": score every card via the rubric (overrides).
        #   - otherwise: only fill cards still missing (embeddings absent).
        if mode == "llm":
            to_score = all_cards
        else:
            to_score = [c for c in all_cards if c.name not in synergy]

        if to_score:
            # v0.9.31: per-commander synergy cache. Cache hits reuse the
            # prior run's score AND effect class (the consistency dimension
            # must not lose classes on a hit); only misses reach the LLM.
            # Keyed on card text + hint tag + rubric, so DB refreshes,
            # tag-tier changes, and prompt edits all rescore honestly.
            cache = self._get_synergy_cache()
            if cache is not None:
                hit_scores, hit_classes, to_score = cache.lookup(
                    to_score, scoring_hints)
                synergy.update(hit_scores)
                self._card_effect_classes.update(hit_classes)
                if hit_scores:
                    logger.info(
                        f"Synergy cache: {len(hit_scores)} score(s) reused, "
                        f"{len(to_score)} to score fresh"
                    )

        if to_score:
            self._report_progress("scoring", "llm", 0.6,
                                  f"LLM scoring {len(to_score)} cards...")
            # Pass synergy_hints so the LLM uses the recall pipeline's
            # per-card signal as a calibration anchor (v0.9.2).
            # batch_size 40 (v0.9.10): structural commanders route the whole
            # pool through the LLM, so fewer/larger batches cut call count
            # ~40% — the synergy payload is tiny per card and the parser is
            # now robust to messy responses.
            llm_scores = self.llm.score_synergy_batch(
                self._analysis, to_score, batch_size=40,
                synergy_hints=scoring_hints or None,
                class_sink=self._card_effect_classes,  # v0.9.14
            )
            synergy.update(llm_scores)
            cache = self._get_synergy_cache()
            if cache is not None:
                cache.store(to_score, llm_scores,
                            self._card_effect_classes, scoring_hints)
                cache.save()
            if self._card_effect_classes:
                logger.info(
                    f"Effect classes: {len(self._card_effect_classes)} cards "
                    f"tagged across "
                    f"{len(set(self._card_effect_classes.values()))} classes"
                )

        # Final heuristic fallback for anything still missing.
        for card in all_cards:
            if card.name not in synergy:
                synergy[card.name] = (
                    self.llm.quick_synergy_check_with_analysis(self._analysis, card)
                )

        # v0.9.12: EDHREC floor. Surface the commander's community-DISTINCTIVE
        # package (Skirk Prospector for Krenko, Ruxa for Jasmine) that recall
        # pulls in but the reasoned signal under-rates — by flooring synergy to
        # a fraction of the EDHREC distinctive score. Boost-only (protects
        # pricey/unpopular cards); fires only when EDHREC data is present.
        self._apply_edhrec_floor(synergy)

        # v0.9.8: engine boost. Lift LLM-detected engines' synergy so the
        # commander's core repeatable payoffs (Soul Warden, Cleric Class...)
        # compete for deck slots instead of losing to generic value cards.
        self._apply_engine_boost(synergy, baseline)

        # v0.9.9: structural boost. Floor attribute-payoff synergy (vanilla
        # creatures, etc.) — they have no text for the text-based signals to
        # score, so without this they never compete despite being the plan.
        self._apply_structural_boost(synergy)

        # v0.9.11: structural POWER floor. The commander-independent card-power
        # signal rates a vanilla 10/10 ~low (mediocre in a vacuum), which drags
        # its effective score below go-wide support. But the commander
        # transforms its BODY into the payoff (Jasmine → unblockable 10/10), so
        # for attribute-matching creatures the baseline should reflect the body
        # (stats), not the vacuum power. Commander-effect-aware, scoped to the
        # transformed cards — keeps the global card-power cache intact.
        self._apply_structural_power_floor(baseline)

        self._synergy_cache = synergy
        self._baseline_power_cache = baseline

        self._report_progress(
            "scoring", "complete", 1.0,
            f"Scored {len(self._synergy_cache)} cards "
            f"(synergy: {len(synergy)}, baseline: {len(baseline)})",
        )

    def _run_embedding_scorer(self, cards: list[Card]) -> dict[str, float]:
        """Run the embedding scorer, constructing one if needed."""
        if self._embedding_scorer is None:
            from .embedding_scorer import EmbeddingSynergyScorer, EmbeddingConfig
            cfg = EmbeddingConfig(model_name=self.config.embedding_model)
            self._embedding_scorer = EmbeddingSynergyScorer.create_if_available(
                self._analysis, config=cfg,
            )
        if self._embedding_scorer is None:
            logger.warning(
                "Embeddings requested but sentence-transformers not available; "
                "skipping embedding phase"
            )
            return {}
        try:
            return self._embedding_scorer.score_cards(cards)
        except Exception as e:
            logger.warning(f"Embedding scoring failed: {e}")
            return {}

    def _phase_optimization(self) -> OptimizationResult:
        """Run the genetic algorithm (single-population or island model)."""
        self._report_progress("optimization", "initializing", 0.0,
                              "Setting up GA...")

        candidate_cards = self._ensure_basic_lands(self._candidates.all_cards())
        # v0.4: inject locked cards that may have been filtered out
        candidate_cards = self._ensure_locked_cards(candidate_cards)
        # NOTE: snow-covered basics are normalized on the FINAL deck (see
        # _normalize_snow_basics), not filtered here — the candidate pool
        # almost always contains some snow-matters card, so a pool-level
        # check would keep snow basics for essentially every build.
        # v0.4: remove banned cards from the pool entirely
        if self.config.banned_cards:
            banned = set(self.config.banned_cards)
            candidate_cards = [c for c in candidate_cards if c.name not in banned]

        # v0.9.15: bracket hard filters — cards OUTRIGHT banned by the
        # bracket rules leave the pool entirely (budget rules like B3's
        # "up to 3 Game Changers" are enforced by the GA penalty instead,
        # since a filter can't express "best three stay").
        bracket = getattr(self.config, "bracket", 4)
        if bracket <= 3:
            from .bracket import (
                is_game_changer, is_mass_land_denial, grants_extra_turn,
            )
            before = len(candidate_cards)
            removed: list[str] = []
            # Explicitly locked cards are exempt — the official rules allow
            # thematic exceptions, and an explicit user choice outranks the
            # default enforcement.
            locked_set = set(self.config.locked_cards or [])

            def keep(c: Card) -> bool:
                if c.name in locked_set:
                    return True
                if bracket <= 2 and is_game_changer(c):
                    removed.append(c.name)
                    return False
                if is_mass_land_denial(c):  # banned at all of B1-3
                    removed.append(c.name)
                    return False
                if bracket == 1 and grants_extra_turn(c):
                    removed.append(c.name)
                    return False
                return True

            candidate_cards = [c for c in candidate_cards if keep(c)]
            if removed:
                logger.info(
                    f"Bracket {bracket} pool filter: removed "
                    f"{before - len(candidate_cards)} card(s): "
                    f"{', '.join(sorted(set(removed))[:12])}"
                    + (" ..." if len(set(removed)) > 12 else "")
                )

        # v0.9.14: keep the pool for the refinement loop's alternatives.
        self._ga_candidate_pool = candidate_cards

        # Use island model if requested
        if self.config.use_island_model:
            return self._run_island_optimization(candidate_cards)

        # Standard single-population GA
        evaluator = DeckEvaluator(
            self.config, self._analysis,
            synergy_cache=self._synergy_cache,
            baseline_power_cache=self._baseline_power_cache,
            flavor_tag_scorer=self.flavor_tag_scorer,  # v0.5
            combos=self._reward_combos,  # v0.9.8 (bracket-legal only, v0.9.15)
            card_effect_classes=self._card_effect_classes,  # v0.9.14
            banned_combos=self._banned_combos,  # v0.9.15
        )
        fast_eval = FastEvaluator(self.config, self._analysis,
                                  synergy_cache=self._synergy_cache)  # v0.9.12

        optimizer = DeckOptimizer(
            config=self.config,
            analysis=self._analysis,
            candidate_pool=candidate_cards,
            commander=self._commander,
            evaluator=evaluator,
            fast_evaluator=fast_eval,
        )

        last_mode = [""]

        def ga_progress(stats: PopulationStats):
            pct = stats.generation / self.config.generations
            # v0.9.29: the fast->full switch RESCALES fitness (heuristic ->
            # real objective) — without this marker the console shows an
            # alarming unexplained score cliff (e.g. 87.5 -> 59.9).
            if stats.mode == "full" and last_mode[0] == "fast":
                self._report_progress(
                    "optimization", "phase_switch", pct,
                    "— switching to FULL evaluation (combos/consistency "
                    "now scored; fitness rescales, not a regression) —",
                )
            last_mode[0] = stats.mode
            tag = f" [{stats.mode}]" if stats.mode else ""
            self._report_progress(
                "optimization", f"gen_{stats.generation}",
                pct,
                f"Gen {stats.generation}{tag}: best={stats.best_fitness:.1f} "
                f"avg={stats.avg_fitness:.1f} invalid={stats.invalid_count}",
            )

        result = optimizer.run(progress_callback=ga_progress)

        self._report_progress(
            "optimization", "complete", 1.0,
            f"Best score: {result.final_score:.1f}",
        )
        return result

    def _run_island_optimization(self, candidate_cards: list[Card]) -> OptimizationResult:
        """Run the island-model parallel GA."""
        from .island_optimizer import IslandModelOptimizer, IslandConfig

        self._report_progress(
            "optimization", "island_setup", 0.05,
            f"Starting island model ({self.config.num_islands} islands)...",
        )

        island_cfg = IslandConfig(
            num_islands=self.config.num_islands,
            migration_interval=self.config.island_migration_interval,
            # Default to multiprocessing; caller can override via CLI if problematic
            use_multiprocessing=True,
        )

        optimizer = IslandModelOptimizer(
            config=self.config,
            analysis=self._analysis,
            candidate_pool=candidate_cards,
            commander=self._commander,
            synergy_cache=self._synergy_cache,
            baseline_power_cache=self._baseline_power_cache,
            island_config=island_cfg,
            combos=self._reward_combos,  # v0.9.8 (bracket-legal only, v0.9.15)
            card_effect_classes=self._card_effect_classes,  # v0.9.14
            banned_combos=self._banned_combos,  # v0.9.15
        )

        result = optimizer.run()

        self._report_progress(
            "optimization", "complete", 1.0,
            f"Best score (across islands): {result.final_score:.1f}",
        )
        return result

    def _phase_llm_refinement(self, result: OptimizationResult) -> None:
        """v0.9.14 post-GA LLM refinement — implementation in refinement.py
        (extracted v0.9.16; see that module's docstring for design)."""
        from .refinement import run_refinement
        run_refinement(self, result)

    def _phase_llm_review(self, result: OptimizationResult) -> str:
        """Run one LLM review pass to identify gaps and suggest swaps."""
        self._report_progress("review", "reviewing", 0.0,
                              "Running LLM deck review...")
        review = self.llm.review_deck(result.best_deck, self._analysis)
        self._report_progress("review", "complete", 1.0,
                              "Review complete")
        return review

    def _phase_validate_roles(self, result: OptimizationResult) -> None:
        """v0.6: Cross-check regex role classification against community oracle
        tags. Diagnostic only — attaches a ValidationReport to the result but
        never changes scores or deck contents. Silently skipped if the
        tag_client can't be constructed."""
        self._report_progress("validate_roles", "validating", 0.0,
                              "Cross-checking roles against oracle tags...")
        client = self.tag_client
        if client is None:
            logger.info(
                "Role validation requested but tag client unavailable; skipping"
            )
            self._report_progress("validate_roles", "complete", 1.0,
                                  "(skipped)")
            return

        from .oracle_validation import validate_roles
        # Filter to commander's color identity to avoid reporting cards
        # that aren't legal in this deck as "missed"
        color_id = (
            self._commander.color_identity if self._commander else None
        )
        try:
            report = validate_roles(
                deck=result.best_deck,
                tag_client=client,
                color_identity=color_id,
            )
            result.role_validation_report = report
            self._report_progress(
                "validate_roles", "complete", 1.0,
                f"{report.total_disagreements} disagreements across "
                f"{len(report.roles_checked)} roles",
            )
        except Exception as e:
            # Never let validation break a build
            logger.warning(f"Role validation failed: {e}")
            self._report_progress("validate_roles", "complete", 1.0,
                                  "(failed)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_basic_lands(self, cards: list[Card]) -> list[Card]:
        """
        Ensure the candidate pool includes the commander's basic lands.

        Without this, the GA can't include multiple basics (its only way to
        deal with a sparse-land pool).
        """
        colors_needed = set(ch for ch in (self._commander.color_identity or '')
                            if ch in 'WUBRG')

        basic_by_color = {
            'W': 'Plains', 'U': 'Island', 'B': 'Swamp',
            'R': 'Mountain', 'G': 'Forest',
        }

        have_names = {c.name for c in cards}
        augmented = list(cards)

        for color in colors_needed:
            basic_name = basic_by_color.get(color)
            if basic_name and basic_name not in have_names:
                basic = self.db.get_by_name(basic_name)
                if basic:
                    augmented.append(basic)

        return augmented

    @staticmethod
    def _is_snow_basic(card: Card) -> bool:
        """A snow-covered basic land (Snow-Covered Plains/Forest/etc.)."""
        return card.is_basic_land and 'snow' in (card.supertypes or '').lower()

    @staticmethod
    def _cares_about_snow(card: Card) -> bool:
        """True if the card uses snow as a resource/payoff — it references
        snow in its rules text, or the snow-mana symbol {S} appears in its
        cost or text. A land merely *being* snow (empty text, normal cost) is
        not a payoff, so snow basics never count as their own justification."""
        text = (card.text or '').lower()
        if 'snow' in text:
            return True
        # {S} snow-mana symbol — in the casting cost or an ability.
        if '{s}' in (card.mana_cost or '').lower() or '{s}' in text:
            return True
        return False

    # Snow-covered basic -> regular-basic equivalent. Functionally identical
    # unless something cares about snow (there is no snow-walk; they keep
    # their normal land types), so swapping is always safe for a non-snow deck.
    _SNOW_TO_REGULAR = {
        "Snow-Covered Plains": "Plains",
        "Snow-Covered Island": "Island",
        "Snow-Covered Swamp": "Swamp",
        "Snow-Covered Mountain": "Mountain",
        "Snow-Covered Forest": "Forest",
        "Snow-Covered Wastes": "Wastes",
    }

    def _normalize_snow_basics(self, deck) -> None:
        """Swap snow-covered basics for regular basics IN PLACE unless the
        final deck actually runs a snow payoff.

        Decided on the finished 99 rather than the candidate pool: the recall
        and role pools almost always contain *some* snow-matters card, so a
        pool-level check keeps snow basics for nearly every build. What matters
        is whether a snow payoff made the deck — if not, snow basics add
        nothing and only leak information, so we normalize them away.
        """
        if not deck or not deck.cards:
            return
        if any(self._cares_about_snow(c) for c in deck.cards
               if not self._is_snow_basic(c)):
            return  # a real snow payoff is in the deck — keep snow basics

        swapped = 0
        for i, c in enumerate(deck.cards):
            if not self._is_snow_basic(c):
                continue
            reg_name = self._SNOW_TO_REGULAR.get(c.name)
            reg = self.db.get_by_name(reg_name) if reg_name else None
            if reg is not None:
                deck.cards[i] = reg
                swapped += 1
        if swapped:
            logger.info(
                f"Normalized {swapped} snow-covered basic(s) to regular "
                f"(no snow payoff in the final deck)"
            )

    def _ensure_locked_cards(self, cards: list[Card]) -> list[Card]:
        """
        v0.4: Ensure every card in config.locked_cards is in the pool.

        The LLM filtering phase may have dropped a user's locked card (not a
        top pick in its category). We need to add it back so the GA can
        actually lock to it.

        Cards listed in locked_cards but not found in the database produce a
        warning. Cards found but filtered out of the color identity are also
        logged (the user probably made a mistake).
        """
        locked = self.config.locked_cards or []
        if not locked:
            return cards

        have_names = {c.name for c in cards}
        augmented = list(cards)
        missing: list[str] = []
        color_violations: list[str] = []

        commander_colors = set(
            ch for ch in (self._commander.color_identity or '')
            if ch in 'WUBRG'
        )

        for name in locked:
            if name in have_names:
                continue
            card = self.db.get_by_name(name)
            if card is None:
                missing.append(name)
                continue

            # Check color identity
            card_colors = set(
                ch for ch in (card.color_identity or '') if ch in 'WUBRG'
            )
            if not card_colors.issubset(commander_colors):
                color_violations.append(name)
                continue

            augmented.append(card)
            self._tag_provenance([card], "locked")
            have_names.add(name)

        if missing:
            logger.warning(
                f"Locked cards not in database: {', '.join(missing)}"
            )
        if color_violations:
            logger.warning(
                f"Locked cards violate commander color identity "
                f"(ignored): {', '.join(color_violations)}"
            )

        return augmented


def build_deck(
    commander_name: str,
    card_database_path: str | Path,
    use_mock_llm: bool = False,
    **config_kwargs,
) -> OptimizationResult:
    """Convenience function for a one-shot build."""
    config = BuildConfig(commander_name=commander_name, **config_kwargs)
    llm_config = LLMConfig(mock_mode=True) if use_mock_llm else None
    builder = DeckBuilder(card_database_path, config, llm_config=llm_config)
    return builder.build()
