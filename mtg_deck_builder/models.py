"""
Data models for MTG Deck Builder.

Design decisions:
- Using dataclasses for simplicity and type safety
- Card uses a custom __eq__/__hash__ based only on name (intentional: we
  don't care about printing differences, only card identity)
- Deck is mutable during optimization
- Scores are stored as a structured object for multi-objective optimization
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class CardType(Enum):
    """Main card types in MTG."""
    CREATURE = "Creature"
    INSTANT = "Instant"
    SORCERY = "Sorcery"
    ARTIFACT = "Artifact"
    ENCHANTMENT = "Enchantment"
    PLANESWALKER = "Planeswalker"
    LAND = "Land"
    BATTLE = "Battle"


class DeckRole(Enum):
    """
    Functional roles cards can fill in a deck.
    Based on common EDH deck-building frameworks.
    """
    RAMP = "ramp"
    DRAW = "draw"
    REMOVAL = "removal"
    WIPE = "wipe"
    THREAT = "threat"
    PROTECTION = "protection"
    RECURSION = "recursion"
    LAND = "land"
    SYNERGY = "synergy"
    UTILITY = "utility"


# eq=False so we can provide our own equality based on name only.
# This is deliberate: the same card from different sets should be treated
# as the same card for deck-building purposes.
@dataclass(frozen=True, eq=False)
class Card:
    """Represents a single MTG card. Frozen because card data is immutable."""
    name: str
    mana_cost: str
    mana_value: int
    card_type: str
    text: str
    color_identity: str
    colors: str
    power: Optional[str] = None
    toughness: Optional[str] = None
    loyalty: Optional[str] = None
    defense: Optional[str] = None
    types: str = ""
    subtypes: str = ""
    supertypes: str = ""
    keywords: str = ""
    layout: str = "normal"
    legalities: str = ""

    # v0.9.17: official Game Changer flag. Populated from an `isGameChanger`
    # CSV column when present (MTGJSON carries this) — the automatic refresh
    # path. Default False; bracket.is_game_changer prefers this over the
    # embedded name list, so a regenerated CSV overrides the frozen constant
    # per-card.
    is_game_changer: bool = False

    # Computed/assigned scores (not from CSV). -1 means "not yet set".
    baseline_power: float = -1.0

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.name == other.name

    @property
    def is_creature(self) -> bool:
        return "Creature" in self.types

    @property
    def is_land(self) -> bool:
        return "Land" in self.types

    @property
    def is_instant_or_sorcery(self) -> bool:
        return "Instant" in self.types or "Sorcery" in self.types

    @property
    def is_basic_land(self) -> bool:
        return "Basic" in self.supertypes and self.is_land

    @property
    def is_vanilla(self) -> bool:
        """A vanilla creature has no rules text and no keywords."""
        if not self.is_creature:
            return False
        import re
        text_without_reminder = re.sub(r'\([^)]*\)', '', self.text or '').strip()
        return len(text_without_reminder) == 0 and not self.keywords

    def format_for_llm(self) -> str:
        """Format card for LLM context, optimized for token efficiency."""
        parts = [f"**{self.name}** {self.mana_cost}".strip()]
        parts.append(self.card_type)
        if self.power and self.toughness:
            parts.append(f"{self.power}/{self.toughness}")
        if self.text:
            parts.append(self.text)
        return " | ".join(p for p in parts if p)


@dataclass
class DeckScores:
    """
    Multi-dimensional scores for a deck.
    Each dimension is 0-100 scale for consistency.
    """
    mana_curve: float = 0.0
    role_coverage: float = 0.0
    synergy: float = 0.0
    power_level: float = 0.0
    creativity: float = 0.0

    # NEW v0.4: Flavor dimension
    # Rewards thematic/tribal coherence (shared subtypes with commander,
    # aesthetic alignment) distinct from mechanical synergy. Push this up
    # for tribal decks where you want type consistency, not just triggers.
    flavor: float = 0.0

    # NEW v0.9.3: Strategy density dimension.
    # Counts the fraction of non-mana-land cards whose synergy score is
    # >= 60. Where `synergy` measures *average* commander-fit across the
    # deck (capped by deck structure — lands and utility cards
    # mathematically can't push the average above ~55), `strategy_density`
    # measures HOW MANY cards are actually on-strategy. A deck full of
    # Soul Sisters and lifegain payoffs has high density even if the
    # average is dragged down by required ramp/removal pieces.
    strategy_density: float = 0.0

    # Combined synergy/power score (the "effective synergy" from our design discussion).
    effective_synergy: float = 0.0

    # v0.9.8: interaction-aware combo score (0-100). Rewards decks that
    # assemble multi-card combos (and partially, near-complete ones). Additive
    # bonus on top of the functional deck; 0 when combo detection is off.
    combo: float = 0.0

    # v0.9.14: consistency/redundancy score (0-100). Measures how well the
    # deck covers the commander's CORE EFFECT CLASSES with redundant copies
    # (e.g. "repeatable lifegain trigger" x4-6). Per-card averages can't see
    # this — to an average, the 2nd soul sister and a 5th anthem look alike;
    # to the plan, the 2nd copy of the critical effect is worth far more.
    # 0 when effect-class data is unavailable (mock / embedding-only runs).
    consistency: float = 0.0

    # Penalties (subtracted from total)
    constraint_penalty: float = 0.0

    # Diagnostic info
    role_counts: dict[str, int] = field(default_factory=dict)
    is_valid: bool = True
    violation_reasons: list[str] = field(default_factory=list)

    def total(self, weights: dict[str, float]) -> float:
        """Calculate the weighted total score, normalized to a 0-100 scale.

        v0.9.7: `creativity` is intentionally excluded from the weighted
        total. It is still computed and shown in reports as an informational
        metric, but it no longer influences deck ranking. In practice it sat
        at a flat 100 for almost every deck (the staple ratio rarely exceeds
        the target), so it only contributed a constant offset and an
        unearned "creativity" claim. Any 'creativity' key in `weights` is
        ignored here by design.

        v0.9.25: the total is the weighted AVERAGE of the 0-100 dimensions
        (weighted sum divided by the sum of active weights), so it stays on
        a 0-100 scale no matter how many additive dimensions (combo,
        consistency, flavor) are switched on. Previously those additive
        weights pushed the headline past 100 (observed: 110-117), which read
        as nonsense. Penalties stay ABSOLUTE points: base weights sum to 1.0,
        so this restores the scale the penalty constants were calibrated on.
        GA ranking is unaffected between decks with equal penalties (pure
        rescale); penalized decks bite ~20% harder, i.e. at original design
        strength.
        """
        # Fallback defaults mirror BuildConfig.score_weights — they only
        # apply if a caller passes a weights dict missing a key.
        w = {
            'mana_curve': weights.get('mana_curve', 0.10),
            'role_coverage': weights.get('role_coverage', 0.15),
            'synergy': weights.get('synergy', 0.35),
            'strategy_density': weights.get('strategy_density', 0.20),
            'power_level': weights.get('power_level', 0.20),
            'combo': weights.get('combo', 0.0),
            'consistency': weights.get('consistency', 0.0),
            'flavor': weights.get('flavor', 0.0),
        }
        score = (
            self.mana_curve * w['mana_curve'] +
            self.role_coverage * w['role_coverage'] +
            self.synergy * w['synergy'] +
            self.strategy_density * w['strategy_density'] +
            self.power_level * w['power_level'] +
            self.combo * w['combo'] +
            self.consistency * w['consistency'] +
            self.flavor * w['flavor']
        )
        weight_sum = sum(w.values())
        if weight_sum > 0:
            score /= weight_sum
        return max(0.0, score - self.constraint_penalty)


@dataclass
class CardTelemetry:
    """Per-card diagnostic data so we can see *why* a card was picked."""
    name: str
    baseline_power: float
    synergy_score: float
    effective_score: float  # baseline * base_weight + synergy * synergy_weight
    role: str
    reasoning: str = ""
    # v0.9.33 (#26): the channel(s) that put this card into the GA pool
    # (recall:edhrec, power-staples, combo-onramp, role:ramp, locked, ...).
    provenance: list[str] = field(default_factory=list)


@dataclass
class Combo:
    """A multi-card interaction (v0.9.8).

    `cards` are the exact card names that, TOGETHER, produce the effect — a
    2nd/3rd-order value the per-card scorers can't see (e.g. Spike Feeder +
    Heliod, Sun-Crowned = infinite life/counters). `payoff` is a 0-100 impact
    rating; `result` is a short human description; `source` records provenance
    (llm-pool / llm-knowledge / edhrec) for debugging and the fallback rule.
    """
    cards: list[str]
    payoff: float = 50.0
    result: str = ""
    source: str = "llm-pool"

    def size(self) -> int:
        return len(self.cards)


@dataclass
class ComboReport:
    """Output of the combo/engine detection phase (v0.9.8).

    - `combos`: interactions found among the candidate pool + known ones.
    - `engines`: name -> short note for cards whose value SCALES with the
      deck's strategy (repeatable triggers like Soul Warden) rather than a
      discrete combo. These get the enabler on-ramp (Leak A).
    - `missing_pieces`: combo cards named by the knowledge pass that aren't in
      the current candidate pool yet — pulled back into recall so the combo
      becomes buildable.
    """
    combos: list[Combo] = field(default_factory=list)
    engines: dict[str, str] = field(default_factory=dict)
    missing_pieces: list[str] = field(default_factory=list)
    # v0.9.16c: famous/known-tech combos named by the RECALL-ONLY signature
    # pass. Informational (drives recall of niche pieces via missing_pieces,
    # NOT the combo score). Each entry: {"cards": [...], "result": str}.
    signature_combos: list = field(default_factory=list)

    def all_combo_card_names(self) -> set[str]:
        names: set[str] = set()
        for c in self.combos:
            names.update(c.cards)
        return names


@dataclass
class Deck:
    """Represents a 99-card EDH deck (excluding commander). Mutable during optimization."""
    commander: Card
    cards: list[Card] = field(default_factory=list)
    scores: Optional[DeckScores] = None
    generation: int = 0

    def __post_init__(self):
        if self.scores is None:
            self.scores = DeckScores()

    @property
    def card_count(self) -> int:
        return len(self.cards)

    @property
    def is_valid(self) -> bool:
        valid, _ = self.validate()
        return valid

    def validate(self) -> tuple[bool, list[str]]:
        """
        Detailed validation returning (valid, reasons).
        """
        reasons = []

        if len(self.cards) != 99:
            reasons.append(f"Wrong card count: {len(self.cards)} (need 99)")

        # Check for duplicates (except basic lands)
        non_basic = [c for c in self.cards if "Basic" not in c.supertypes]
        seen = set()
        duplicates = []
        for card in non_basic:
            if card.name in seen:
                duplicates.append(card.name)
            seen.add(card.name)
        if duplicates:
            preview = duplicates[:3]
            extra = '...' if len(duplicates) > 3 else ''
            reasons.append(f"Duplicate non-basic cards: {preview}{extra}")

        # Check color identity compliance
        commander_colors = set(ch for ch in (self.commander.color_identity or '')
                               if ch in 'WUBRG')
        violations = []
        for card in self.cards:
            card_colors = set(ch for ch in (card.color_identity or '')
                              if ch in 'WUBRG')
            if not card_colors.issubset(commander_colors):
                violations.append(card.name)
        if violations:
            preview = violations[:3]
            extra = '...' if len(violations) > 3 else ''
            reasons.append(f"Color identity violations: {preview}{extra}")

        return (len(reasons) == 0, reasons)

    def get_cards_by_type(self, card_type: str) -> list[Card]:
        return [c for c in self.cards if card_type in c.types]

    def get_mana_curve(self) -> dict[int, int]:
        """Get count of non-land cards at each mana value."""
        curve = {}
        for card in self.cards:
            if not card.is_land:
                mv = card.mana_value
                curve[mv] = curve.get(mv, 0) + 1
        return curve

    def get_color_distribution(self) -> dict[str, int]:
        colors = {}
        for card in self.cards:
            for color in (card.colors or ''):
                if color in 'WUBRG':
                    colors[color] = colors.get(color, 0) + 1
        return colors

    def to_decklist(self) -> str:
        """Export deck as a standard decklist format, grouped by type."""
        lines = [f"Commander: {self.commander.name}", ""]

        by_type = {}
        for card in self.cards:
            main_type = card.types.split(',')[0].strip() if card.types else "Other"
            by_type.setdefault(main_type, []).append(card)

        type_order = ["Creature", "Instant", "Sorcery", "Artifact",
                      "Enchantment", "Planeswalker", "Land", "Battle"]

        seen_types = set()
        for card_type in type_order:
            if card_type in by_type:
                seen_types.add(card_type)
                cards = sorted(by_type[card_type], key=lambda c: (c.mana_value, c.name))
                lines.append(f"// {card_type}s ({len(cards)})")
                for card in cards:
                    lines.append(f"1 {card.name}")
                lines.append("")

        for card_type, cards in by_type.items():
            if card_type not in seen_types:
                cards = sorted(cards, key=lambda c: c.name)
                lines.append(f"// {card_type} ({len(cards)})")
                for card in cards:
                    lines.append(f"1 {card.name}")
                lines.append("")

        return "\n".join(lines)


@dataclass
class CommanderAnalysis:
    """LLM-generated analysis of a commander."""
    name: str
    color_identity: str
    key_mechanics: list[str]
    build_around_text: str
    evaluation_notes: str
    category_queries: dict[str, str]
    synergy_keywords: list[str]
    anti_synergy_keywords: list[str] = field(default_factory=list)

    # NEW in v0.2: LLM-recommended weight overrides for this specific commander.
    # If provided, these override the default scoring weights.
    # Example for Jasmine Boreal (vanilla-creatures-matter):
    #   {'synergy': 0.50, 'power_level': 0.05}
    recommended_weights: Optional[dict[str, float]] = None

    # NEW in v0.2: Recommended synergy/baseline balance for this commander.
    # Should be between 0.0 and 1.0.
    # - High synergy commanders (Jasmine): ~0.8
    # - Strong synergy (Lathiel): ~0.6
    # - Balanced: ~0.5
    # - Generic goodstuff commander: ~0.3
    recommended_synergy_weight: Optional[float] = None

    # NEW in v0.8 (Session 8 — candidate recall):
    # LLM-expanded substring patterns for matching card text. The plain
    # `synergy_keywords` list above is too literal (e.g. "gain life" doesn't
    # substring-match "gain 1 life"); these patterns are deliberately
    # written in shapes that survive a digit/X-normalization pass. Used by
    # the pattern-recall candidate source. Empty list = patterns disabled.
    # Example for a lifegain commander:
    #   ["gain life", "lifelink", "+1/+1 counter", "creature token",
    #    "you gained life", "soul sister", "gains life"]
    synergy_patterns: list[str] = field(default_factory=list)

    # NEW v0.9.9: STRUCTURAL synergy predicates — card-ATTRIBUTE matters, not
    # text. For archetypes whose payoff is a structural property (vanilla /
    # "no abilities", colorless, low-curve, big-stats, a creature type), the
    # text-based synergy signals are blind. These predicates reward attributes
    # directly. Empty for normal text-defined commanders. Bounded vocabulary:
    #   vanilla | no_abilities | colorless | creature | land
    #   subtype:X | type:X | supertype:X | keyword:X
    #   mv<=N | cmc>=N | power>=N | toughness<=N   (ops: <= >= == < >)
    # Example for Jasmine Boreal (vanilla matters): ["vanilla"]
    structural_predicates: list[str] = field(default_factory=list)

    # NEW v0.9.14: the strategy's CORE EFFECT CLASSES — the effects the deck
    # needs REDUNDANT copies of to function consistently. Each entry:
    #   {"name": "repeatable lifegain trigger", "min_count": 4}
    # The synergy-scoring pass tags each card with the class it fills (if
    # any); the evaluator's consistency dimension scores class coverage with
    # diminishing returns per copy. Empty = consistency dimension inactive.
    core_effect_classes: list[dict] = field(default_factory=list)


# v0.9.15: bracket-5 (cEDH) structural role targets. Sourced to cEDH
# deckbuilding norms — the format is STRUCTURALLY different at bracket 5:
# lean draw engines, cheap interaction as a first-class package (the
# 'protection' role counts counterspells/protection effects), and near-zero
# board wipes (they don't advance a combo/tempo plan). Used as the base
# targets when config.bracket == 5; user role_target_overrides still win.
#
# v0.9.33 (#32): land floor lowered 28 -> 26 to match community norms
# (cEDH trades lands for the 10-12 fast-mana rocks/dorks it runs). The land
# count is no longer enforced in isolation — deck_evaluator adds a COUPLED
# "lands + fast mana >= 38 total sources" penalty (tuning.CEDH_MANA_SOURCES
# _FLOOR), so a rock-heavy build can run leaner lands without the old model
# forcing 28 lands regardless of acceleration.
CEDH_ROLE_TARGETS: dict[str, tuple[int, int]] = {
    'ramp': (12, 26),
    'draw': (8, 14),
    'removal': (5, 10),
    'protection': (6, 14),
    'wipe': (0, 2),
    'land': (26, 31),
}


@dataclass
class BuildConfig:
    """Configuration for a deck build run."""
    commander_name: str

    # Constraints
    # NOTE: For per-card budget caps see budget_max_per_card (v0.3) below.
    # The old `budget_max` field was removed in v0.5.5 cleanup as it was
    # never read anywhere in the codebase.

    # v0.9.15: the official Commander BRACKET (1-5) replaces the old 1-10
    # power_level. Brackets set BOTH the rules regime (Game Changer limits,
    # MLD, extra turns, two-card-combo policy — see bracket.py) and the
    # build posture (weight scaling; at bracket 5 the structural templates
    # switch to cEDH shape: curve, role targets, land count).
    #   1 Exhibition | 2 Core | 3 Upgraded | 4 Optimized (default) | 5 cEDH
    # Default is 4 — "no restrictions", which matches how the engine built
    # decks before brackets existed.
    bracket: Optional[int] = None
    # v0.9.17: optional path to refresh the Game Changer list from an external
    # file (JSON array / {"names":[...]} / MTGJSON-atomic isGameChanger /
    # newline text). Overrides the embedded bracket.py constant for this run.
    # None = use the CSV isGameChanger column if present, else the embedded
    # list. See bracket.load_game_changer_names.
    game_changers_file: Optional[str] = None
    # DEPRECATED (v0.9.15): the old 1-10 scale. If given and `bracket` is
    # not, it maps onto brackets (1-2->B1, 3-4->B2, 5-7->B3, 8-9->B4,
    # 10->B5) in __post_init__.
    power_level: Optional[int] = None

    # GA Parameters
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    tournament_size: int = 3
    elitism_count: int = 2
    random_seed: Optional[int] = None

    # NEW: Early stopping
    patience_generations: int = 30
    min_improvement: float = 0.01

    # Scoring Weights (base defaults - may be overridden per-commander)
    # v0.9.7: `creativity` removed — it no longer contributes to the
    # weighted total (see DeckScores.total). `strategy_density` is now listed
    # explicitly (it was previously an invisible 0.20 supplied only by
    # total()'s default, so it never showed up in reports and made the
    # effective weights sum to 1.20). The set now sums to 1.0, so the Total
    # score is a true 0-100. Mirrors the "balanced" preset minus flavor.
    score_weights: dict[str, float] = field(default_factory=lambda: {
        'mana_curve': 0.10,
        'role_coverage': 0.15,
        'synergy': 0.35,
        'strategy_density': 0.20,
        'power_level': 0.20,
    })

    # NEW: Whether to let commander analysis override scoring weights
    commander_adaptive_weights: bool = True

    # Role Targets
    # These are long-standing format-structural heuristics (every functional
    # EDH deck needs roughly this much mana/draw/interaction).
    #
    # v0.9.14: removal SUB-TYPES ('removal_creature', 'removal_artifact' —
    # the latter covers artifacts AND enchantments) are classified and
    # TRACKED (role_counts diagnostics, report) but deliberately have NO
    # default targets: the right interaction spread is strategy/meta
    # judgment, not a universal constant. Spread is enforced by the LLM
    # refinement pass (its rubric checks it against the whole deck) and,
    # per-commander, by core effect classes. Users who want a hard floor
    # can set one explicitly: --role-target removal_creature=4,12.
    role_targets: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        'ramp': (10, 14),
        'draw': (10, 14),
        'removal': (8, 12),
        'wipe': (2, 4),
        'land': (35, 38),
    })

    # v0.9.13: penalty per card BELOW a role's minimum target, subtracted
    # from the total (via constraint_penalty). The role_coverage dimension
    # alone under-prices shortfalls: with honest synergy scoring, swapping a
    # ramp/removal card for an on-theme body gains ~0.65 total points but
    # costs only ~0.33 in role coverage — so the GA rationally starves mana
    # (observed: a 4-ramp/3-removal Jasmine deck). At 2.0/card the trade
    # flips and the GA fills minimums first. 0 disables.
    role_shortfall_penalty: float = 2.0

    # v0.9.14: quality-weighted role coverage. When True, a card counts
    # toward its role target weighted by its baseline power —
    # min(1.0, 0.5 + power/120) — so a power-20 filler counts ~0.67 and a
    # power-60+ card counts 1.0. Creates the missing gradient toward
    # Rampant-Growth-class role fillers (a 20-point power difference on one
    # card was otherwise worth ~0.06 total points: fitness-invisible).
    quality_weighted_roles: bool = True

    # v0.9.14: consistency dimension weight (additive, like combo_weight).
    # Injected into the effective weights only when the commander analysis
    # emitted core_effect_classes; 0 disables.
    consistency_weight: float = 0.12

    # v0.9.14: post-GA LLM refinement loop. The GA optimizes per-card
    # averages and count thresholds; the refinement pass hands the ACTUAL
    # assembled 99 (plus the best unused pool alternatives) to the LLM for
    # holistic critique — redundancy, interaction spread, ramp quality —
    # and applies its swaps. This is the set-level judgment no average can
    # express. 0 disables.
    refine_iterations: int = 3
    refine_max_swaps: int = 8

    # Card Pool Settings
    candidates_per_category: int = 100

    # v0.9.15b: per-role POWER BYPASS. The top-N cards by cached intrinsic
    # card power in each role bucket are guaranteed into the GA pool
    # ADDITIVELY (on top of the LLM's picks), so the selection tournament
    # can never eliminate the format's best role-fillers — a real Kinnan
    # run funnel-cut Llanowar Elves (power 78), Arcane Signet-class rocks,
    # and Force of Negation (88) before the GA ever saw them. Mirrors the
    # synergy_engine cosine bypass: not a name list, just the engine's own
    # quality signal; the GA still decides the 99. Requires card_power_mode
    # llm (no scores -> no-op). 0 disables.
    role_power_bypass: int = 15

    # v0.9.16: GLOBAL power-staples channel — the generalized answer to
    # taxonomy holes. Pool entry was gated by hand-written role regexes and
    # theme recall, so generically-strong cards that are neither (stax,
    # tutors, clones, theft, cost reducers...) had NO channel until someone
    # invented a category for them. Instead: the top-N color-legal cards by
    # GLOBAL cached card power enter the pool directly, regardless of role
    # or theme. Engine-native signal (not a name list), entry-only (honest
    # scoring + GA still decide), bracket pool filters still apply after.
    # Coverage grows with the cache (see the `power-scan` CLI command).
    # 0 disables.
    # v0.9.33 (#27): raised 40 -> 60 to reach flexible clones (Metamorph
    # pow 80, Mirage Mirror) that sit just below the old cut. Downstream
    # cost is bounded: staples come from the EXISTING power cache (zero new
    # power calls), the ~20 extra cards add ~half a synergy batch (free on
    # reruns via the synergy cache), and entry is pool-only — the GA still
    # decides on honest synergy.
    power_staples_limit: int = 60

    # LLM Settings. claude-sonnet-4-6 supports adaptive thinking (not used
    # here yet) and has the same wire shape as 4-5 for the fields we touch.
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.3

    # Synergy vs Power Balance (NOW ACTUALLY USED in DeckEvaluator)
    # Formula: effective_score = baseline * base_weight + synergy * synergy_weight
    synergy_weight: float = 0.6
    base_weight: float = 0.4

    # Creativity Settings
    staple_penalty_threshold: float = 0.5
    creativity_target: float = 0.3

    # NEW v0.2: Architectural toggles
    use_structured_ga: bool = False
    enable_llm_review: bool = False
    generate_html_report: bool = False

    # NEW v0.3 (Session 3): Optional integrations
    # All default to off/safe values so existing builds are unaffected.

    # EDHREC integration - provides real baseline/synergy data
    use_edhrec: bool = False
    edhrec_cache_dir: Optional[str] = None  # e.g. "./edhrec_cache"
    edhrec_offline: bool = False  # True for testing without network

    # Embedding-based synergy (first-pass scorer, ~1000x faster than LLM)
    use_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"

    # v0.9.4: synergy scoring mode (runtime lever).
    #   "auto"      — use embedding cosine + hint-tier boost when
    #                 sentence-transformers is available (fast, ~0 LLM
    #                 calls); fall back to LLM only if embeddings missing.
    #   "llm"       — always use the LLM rubric (max quality, ~30 calls).
    #   "embedding" — embedding+hint only, never LLM (fastest).
    # Default "auto" eliminates the ~30 LLM scoring calls per build when
    # embeddings are installed, while keeping the calibrated hint signal.
    synergy_scoring_mode: str = "auto"

    # --------------------------------------------------------------
    # v0.8 (Session 8): Layered candidate recall for the synergy pool
    # --------------------------------------------------------------
    # The historical pool builder used literal substring matching against
    # `analysis.synergy_keywords`, which silently dropped any card with a
    # numeric variant in its text ("gain 1 life" doesn't match "gain life").
    # These three flags enable independent recall sources that union into
    # the synergy pool — each catches a different class of failure mode.
    #
    # All default to False so existing builds are unaffected. Enable
    # whichever subset you trust; they're safe in any combination. When
    # any of them is True, the old substring path is replaced by the
    # union; when all are False, the legacy path runs unchanged.

    # Top-N high-synergy cards from EDHREC (community-vetted). Requires
    # `use_edhrec=True` to actually fetch — this flag only controls
    # whether the fetched data feeds the candidate pool.
    recall_use_edhrec: bool = False
    recall_edhrec_limit: int = 300

    # Top-N cards by cosine similarity to the commander's strategy text.
    # Requires sentence-transformers installed; silently falls back to
    # empty list if missing.
    recall_use_embeddings: bool = False
    recall_embedding_limit: int = 1500
    recall_embedding_cache_dir: Optional[str] = "./embedding_cache"

    # LLM-expanded substring patterns matched against digit/X-normalized
    # card text. Patterns come from CommanderAnalysis.synergy_patterns,
    # which the analyze_commander prompt now requests.
    recall_use_patterns: bool = False

    # Final cap on the unioned synergy pool. Higher = more candidates for
    # the LLM filter to choose from but slower select_cards calls.
    recall_pool_cap: int = 2500

    # --------------------------------------------------------------
    # v0.9: Phase 2 synergy_engine pass
    # --------------------------------------------------------------
    # After all traditional role buckets fill (ramp/draw/removal/threats/
    # protection/recursion/wipe/lands), a final LLM pass scans the recall
    # union MINUS already-selected cards and picks strategy-defining
    # engine pieces — typically the cheap repeatable triggers and payoffs
    # that lose head-to-head role evaluations because they're individually
    # weak (Soul Warden, Suture Priest, Trelasarra, Hardened Scales, etc.).
    #
    # Target ~25 cards by default — enough to populate the deck's
    # commander-specific engine slots without overwhelming role-buckets.
    # Set to 0 to disable the Phase 2 pass.
    synergy_engine_target: int = 25

    # v0.9.6: the synergy_engine pass used to run a full elimination
    # tournament over the ENTIRE recall union (e.g. ~2,200 cards) just to
    # pick `synergy_engine_target`, which (a) cost ~20 min of LLM calls and
    # (b) let a coarse early Haiku round eliminate the very best payoffs.
    # We now pre-rank the pool by (adaptive hint tier, cosine-to-commander)
    # and only run the LLM over the top `synergy_engine_shortlist` cards.
    synergy_engine_shortlist: int = 300

    # Of that ranked pool, the top `synergy_engine_bypass` cards skip the
    # LLM tournament entirely and go straight into the GA candidate pool, so
    # the commander's defining payoffs (highest tier + cosine) can't be
    # dropped by elimination. This is a COMMANDER-RELATIVE guarantee (top
    # cosine-to-this-commander), not a global staples list — and it only
    # ensures the cards reach the GA, which still decides the final 99.
    # Set to 0 to disable the bypass (pure LLM selection from the shortlist).
    synergy_engine_bypass: int = 12

    # v0.9.7: LLM intrinsic card-power scoring.
    # The missing "is this card actually good?" signal. Commander-independent,
    # cached globally on disk, so the first build pays and later builds reuse.
    #   off : no card-power scoring (legacy behavior — heuristic baseline only)
    #   llm : score via the LLM rubric (Sonnet recommended; Haiku too soft)
    card_power_mode: str = "off"
    card_power_model: str = "claude-sonnet-4-6"
    card_power_cache_dir: Optional[str] = "./card_power_cache"
    card_power_batch_size: int = 100
    # Pre-rank blend weight: composite = cosine + weight * (power/100). Kept
    # small so recall stays SYNERGY-LED — power nudges a card up but commander
    # fit (tier + cosine) still dominates which cards reach the shortlist.
    card_power_recall_weight: float = 0.15
    # Cap on how many synergy-pool cards get power-scored for recall-feeding
    # (top-N by cosine). 0 = score the whole pool (best quality, higher first-
    # build cost). A positive cap trades recall reach for cost/latency.
    card_power_recall_cap: int = 0

    # v0.9.8: LLM combo/engine detection + interaction-aware GA fitness.
    #   off : no combo detection (legacy behavior)
    #   llm : detect combos/engines via the LLM (pool + knowledge passes)
    # v0.9.31: per-commander synergy-score cache. LLM synergy variance
    # (±5-8 points/run) was the largest remaining source of run-to-run deck
    # churn; caching makes repeat builds near-deterministic and cuts ~30-40
    # Sonnet calls. None disables (--no-synergy-cache).
    synergy_cache_dir: Optional[str] = "./synergy_cache"

    combo_mode: str = "off"
    combo_model: str = "claude-sonnet-4-6"
    combo_cache_dir: Optional[str] = "./combo_cache"
    # v0.9.16c: RECALL-ONLY signature-combo pass — names the commander's
    # famous combos so their niche pieces (e.g. Mirror Universe for Selenia)
    # enter the pool. Provides more data without reweighting: it feeds recall
    # only, never the combo score/boost/on-ramp. One extra LLM call.
    combo_signature_pass: bool = True
    # How many of the (power-ranked) synergy-pool cards the pool pass analyzes.
    combo_max_pool: int = 350
    # Moderate by default: combos reward assembly as an ADDITIVE bonus on top
    # of a functional deck, scaled by power_level (cEDH chases them harder).
    # The combo dimension is 0-100; this is its weight in the total.
    combo_weight: float = 0.12

    # v0.9.8: engine boost. LLM-detected "engine" cards (repeatable payoffs
    # like Soul Warden) are on-ramped into the pool, but without a scoring
    # lift they lose deck slots to generic value cards. This raises their
    # synergy so they COMPETE (not auto-include). Only active with combos on.
    #   off   : on-ramp only (no scoring change)
    #   floor : floor each engine's synergy to engine_boost_floor (flat)
    #   power : floor each engine's synergy to its own card-power score
    #           (quality-scaled — strong engines rise, weak ones don't)
    # Default is "power": in A/B testing it beat the flat floor on synergy,
    # strategy-density, and assembled combos (it pulled the strong engines —
    # Heliod, Rhox Faithmender, Conclave Mentor — without inflating weak ones).
    engine_boost_mode: str = "power"
    engine_boost_floor: float = 80.0

    # v0.9.9: structural/attribute synergy. When the commander analysis emits
    # structural_predicates (e.g. ["vanilla"]), pull matching cards into recall
    # and floor their synergy so attribute-payoffs (which have no text for the
    # text-based signals to see) actually compete. No-op when no predicates.
    #   on  : apply structural recall + synergy floor
    #   off : ignore structural predicates
    # v0.9.12: EDHREC synergy FLOOR. When EDHREC data is present, a card's
    # synergy is floored to (edhrec_floor * its EDHREC distinctive-synergy):
    #   synergy = max(reasoned, edhrec_floor * edhrec_distinctive)
    # This SURFACES the commander's community staple package (Skirk Prospector,
    # Ruxa...) that recall pulls in but the reasoned signal under-rates — a
    # gentle blend was too weak to lift them into contention. Boost-only (never
    # lowers, so pricey/unpopular cards are protected); uses the DISTINCTIVE
    # metric so generic staples (Sol Ring) aren't over-boosted; 0 disables;
    # no-op for commanders with no EDHREC data.
    edhrec_floor: float = 0.75

    structural_synergy_mode: str = "on"
    # 95 (not 85): attribute payoffs (vanilla creatures) have ~0 power, so with
    # synergy_weight ~0.85 a floor of 85 only TIES the text-synergistic support
    # creatures (which self-score ~85 AND have real power, winning on the power
    # term). The floor must clear the text-synergy ceiling (~85-90) for the
    # attribute payoffs — the deck's whole reason to exist — to actually win
    # slots. Tune with --structural-floor.
    structural_boost_floor: float = 95.0
    # Cap on how many attribute-matching cards to pull in / guarantee, ranked
    # by combat stats (power+toughness) — most attribute archetypes are
    # creature beatdown, so bigger bodies first.
    structural_recall_cap: int = 80

    # Budget constraint
    budget_max_per_card: Optional[float] = None  # e.g. 10.0 USD max per card
    budget_exclude_unknown: bool = False  # drop cards with no price data
    # Optional: user can provide their own PriceSource (any duck-typed object)
    # Set via DeckBuilder constructor, not config directly.

    # Island-model parallel GA
    use_island_model: bool = False
    num_islands: int = 4
    island_migration_interval: int = 10

    # v0.4: Scryfall image integration for HTML reports
    use_images: bool = False
    images_cache_dir: Optional[str] = None  # default "./scryfall_cache"
    images_offline: bool = False  # for testing without network

    # --------------------------------------------------------------
    # v0.5 (Session 5): Scryfall Tagger integration
    # --------------------------------------------------------------

    # Art-tag-based flavor scoring. Supply a list of Scryfall art tag slugs
    # (e.g. ["forest", "mammoth", "woodland"]) and the flavor scorer will
    # reward cards whose artwork matches any of those tags — regardless of
    # the card's mechanics. This complements the tribal-subtype flavor
    # already in v0.4. Both signals combine additively when both apply.
    #
    # Requires network on first use; cached afterwards. Set
    # `tags_offline=True` for testing or if you've pre-seeded the cache.
    flavor_art_tags: list[str] = field(default_factory=list)

    # When True, DeckEvaluator will cross-check its regex-based role
    # detection against oracle tags from the Tagger project. Disagreements
    # are logged at DEBUG level — doesn't change scores, just diagnostics.
    # Useful for spotting cards that should be classified differently.
    use_oracle_tag_validation: bool = False

    # Where to cache Scryfall tag query results (separate from
    # images_cache_dir because the schemas are different)
    tags_cache_dir: Optional[str] = None  # default "./scryfall_tags_cache"
    tags_offline: bool = False

    # --------------------------------------------------------------
    # NEW v0.6 (Session 6): Scryfall bulk data + role validation
    # --------------------------------------------------------------

    # When True, the card_source property downloads (or reuses) Scryfall's
    # bulk JSON file and serves all per-card lookups from a local index
    # instead of making per-card API calls. ~130MB download, one HTTP call
    # vs potentially hundreds. Recommended for serious use.
    #
    # Independent of use_images — you can enable bulk without enabling
    # images (useful if you want fast card metadata lookups without also
    # embedding art in HTML).
    use_bulk_source: bool = False
    bulk_cache_dir: Optional[str] = None  # default "./scryfall_bulk"
    # Which bulk file to use. oracle_cards (~130MB) is usually right;
    # default_cards (~300MB) has one entry per printing if you need that;
    # unique_artwork (~200MB) is useful for art-focused tools.
    bulk_type: str = "oracle_cards"
    bulk_offline: bool = False

    # When True, and a tag_client is available, DeckBuilder runs role
    # validation after optimization: cross-checks regex-based role
    # classifications against community oracle tags. Pure diagnostic —
    # doesn't change scores or deck contents. Result is stored on the
    # OptimizationResult as `role_validation_report` for inspection.
    validate_roles_after_build: bool = False

    # --------------------------------------------------------------
    # NEW v0.4 (Session 4): Iterative refinement
    # --------------------------------------------------------------

    # Cards that MUST appear in every deck. Checked by name (case-sensitive).
    # These are injected into every individual and preserved through
    # crossover/mutation. If a locked card isn't in the candidate pool,
    # DeckBuilder adds it. Typical use: user's favorite cards, commanders
    # they're already building around, or cards they want to try.
    locked_cards: list[str] = field(default_factory=list)

    # Cards that must NEVER appear. Filtered out of the pool entirely.
    # Typical use: cards the user already owns too many of, cards they
    # dislike, or known bad matches.
    banned_cards: list[str] = field(default_factory=list)

    # Overrides for specific role counts. These MERGE into the default
    # role_targets (they don't replace them). E.g. {'removal': (10, 14)}
    # means "I want 10-14 removal spells". Roles not mentioned use defaults.
    role_target_overrides: dict[str, tuple] = field(default_factory=dict)

    # Warm-start: path to a JSON file containing a previous OptimizationResult
    # (produced by result.to_json()). When set, the GA initial population
    # includes copies of that deck, letting the GA refine rather than start
    # from scratch. Great for "this was good but needs more removal" iteration.
    warm_start_path: Optional[str] = None

    # How many copies of the warm-start deck to seed the population with.
    # Higher = more conservative (GA stays closer to the original). Default
    # is 1, which lets the GA explore around it. Set to
    # population_size to effectively freeze the deck and only tweak.
    warm_start_copies: int = 1

    # NEW v0.4: Flavor dimension
    # "Flavor" rewards tribal coherence, theme adherence, and thematic
    # choices over raw power. Separate from synergy (which rewards mechanical
    # interaction). You can push this up for a tribal deck and synergy will
    # still matter separately.
    # NOTE: for backwards compatibility, if 'flavor' isn't in score_weights
    # it'll default to 0 and effectively not contribute.

    def __post_init__(self):
        # v0.9.15: resolve the bracket. Explicit bracket wins; else the
        # deprecated power_level maps over; else default 4 (Optimized —
        # the regime the engine effectively built in before brackets).
        if self.bracket is None:
            if self.power_level is not None:
                from .bracket import power_level_to_bracket
                self.bracket = power_level_to_bracket(self.power_level)
            else:
                self.bracket = 4
        self.bracket = max(1, min(5, int(self.bracket)))

    def get_effective_weights(
        self,
        analysis: Optional[CommanderAnalysis] = None
    ) -> dict[str, float]:
        """
        Get the actual scoring weights to use, accounting for:
        1. Per-commander overrides from analysis (if enabled)
        2. Power-level-based creativity adjustment
        """
        weights = dict(self.score_weights)

        if (
            self.commander_adaptive_weights
            and analysis is not None
            and analysis.recommended_weights
        ):
            for k, v in analysis.recommended_weights.items():
                if k in weights:
                    weights[k] = v

        # v0.9.15: bracket-scaled raw-power weight. Brackets 4-5 (Optimized/
        # cEDH) weight raw power up ~x1.25; brackets 1-2 (Exhibition/Core)
        # down ~x0.7; bracket 3 (Upgraded) is neutral. Same magnitudes as
        # the old power_level >= 8 / <= 4 scaling.
        if self.bracket >= 4:
            weights['power_level'] = weights.get('power_level', 0.20) * 1.25
        elif self.bracket <= 2:
            weights['power_level'] = weights.get('power_level', 0.20) * 0.7

        # v0.9.8/v0.9.15: combo weight is an ADDITIVE bonus (not part of the
        # normalized 1.0), injected only when combo detection is enabled and
        # scaled by bracket. At brackets 1-2 the combo dimension is NOT
        # injected at all: the official rules ban two-card combos there, and
        # the casual posture ("wins telegraphed, incremental") means combo
        # assembly should not be chased — detection still runs so the
        # compliance audit and GA penalty can see violations.
        if getattr(self, "combo_mode", "off") != "off" and self.bracket >= 3:
            cw = getattr(self, "combo_weight", 0.12)
            if self.bracket >= 4:
                cw *= 1.5
            weights['combo'] = cw

        # v0.9.14: consistency weight is likewise an ADDITIVE bonus, injected
        # only when the analysis actually emitted core effect classes (no
        # data → no dimension → totals unchanged for legacy/mock runs).
        if (
            analysis is not None
            and getattr(analysis, "core_effect_classes", None)
            and getattr(self, "consistency_weight", 0.0) > 0
        ):
            weights['consistency'] = self.consistency_weight

        return weights

    def get_effective_synergy_balance(
        self,
        analysis: Optional[CommanderAnalysis] = None
    ) -> tuple[float, float]:
        """
        Returns (base_weight, synergy_weight) accounting for commander overrides.
        """
        if (
            self.commander_adaptive_weights
            and analysis is not None
            and analysis.recommended_synergy_weight is not None
        ):
            sw = max(0.0, min(1.0, analysis.recommended_synergy_weight))
            return (1.0 - sw, sw)
        return (self.base_weight, self.synergy_weight)

    def get_effective_role_targets(self) -> dict[str, tuple[int, int]]:
        """
        Returns role_targets merged with user overrides (v0.4).

        role_target_overrides wins when set. Example:
            BuildConfig(role_target_overrides={'removal': (10, 14)})
        will set the removal target to (10, 14) while leaving all other
        roles at their defaults.

        v0.9.15: at bracket 5 (cEDH) the BASE targets switch to
        CEDH_ROLE_TARGETS — the format is structurally different there
        (fewer lands, more fast mana, cheap interaction as a package).
        User overrides still merge on top.
        """
        if getattr(self, "bracket", 4) == 5:
            merged = dict(CEDH_ROLE_TARGETS)
        else:
            merged = dict(self.role_targets)
        for role, target in (self.role_target_overrides or {}).items():
            if not (isinstance(target, (list, tuple)) and len(target) == 2):
                logger.warning(
                    f"Ignoring role_target_override for {role!r}: expected "
                    f"(lo, hi) tuple/list, got {target!r}"
                )
                continue
            try:
                lo, hi = int(target[0]), int(target[1])
                if lo > hi:
                    lo, hi = hi, lo
                merged[role] = (lo, hi)
            except (TypeError, ValueError):
                logger.warning(
                    f"Ignoring role_target_override for {role!r}: could not "
                    f"parse {target!r} as integers"
                )
        return merged


@dataclass
class OptimizationResult:
    """Results from a deck optimization run."""
    best_deck: Deck
    final_score: float
    generations_run: int
    score_history: list[float]
    diversity_history: list[float]
    runtime_seconds: float          # GA optimization phase only
    config: BuildConfig

    # NEW v0.2
    card_telemetry: list[CardTelemetry] = field(default_factory=list)
    # v0.9.6: total end-to-end build wall-clock (analysis + recall + LLM
    # filtering + scoring + GA). runtime_seconds above is GA-only, which made
    # the report's "Ns runtime" wildly understate real builds. None until the
    # builder sets it.
    total_runtime_seconds: Optional[float] = None
    # v0.9.6: per-generation evaluator mode ("fast"/"full"/"") parallel to
    # score_history. The fast and full evaluators score on different scales,
    # so the report plots only the comparable full-eval segment to avoid a
    # misleading mid-run "crash" where the phases meet.
    eval_mode_history: list[str] = field(default_factory=list)
    llm_review: Optional[str] = None
    commander_analysis: Optional[CommanderAnalysis] = None
    # v0.9.8: detected combos (for the report's assembled-combos section).
    combos: list = field(default_factory=list)
    # v0.9.14: swaps applied by the post-GA LLM refinement loop, for the
    # report. Each entry: {"out": name, "in": name, "reason": str,
    # "round": int}.
    refinement_log: list = field(default_factory=list)
    # v0.9.15: bracket compliance audit of the final deck (a BracketAudit
    # from bracket.py; typed loosely to avoid import cycles).
    bracket_audit: Optional[object] = None
    # v0.9.33 (#26): full pool-entry provenance map (name -> channels) for
    # EVERY card that entered the GA pool, not just the final 99 — lets us
    # answer "why is card X missing" (it never entered, or via which channel).
    pool_provenance: dict = field(default_factory=dict)

    # NEW v0.6: Optional role validation diagnostic
    # Typed as Any in-annotation to avoid circular imports; this is a
    # ValidationReport from oracle_validation.py when populated.
    role_validation_report: Optional["object"] = None

    def to_warm_start(self) -> "WarmStartDeck":
        """
        Produce a lightweight serializable snapshot suitable for warm-starting
        a future build. Strips telemetry, scores, and config — keeps just what
        the next GA needs: commander and deck contents.
        """
        return WarmStartDeck(
            commander_name=self.best_deck.commander.name,
            card_names=[c.name for c in self.best_deck.cards],
            final_score=self.final_score,
        )

    def to_json_file(self, path: str) -> None:
        """Save warm-start snapshot to a JSON file."""
        import json as _json
        from pathlib import Path as _P
        _P(path).write_text(
            _json.dumps(self.to_warm_start().to_dict(), indent=2),
            encoding="utf-8",
        )


@dataclass
class WarmStartDeck:
    """
    A serializable deck snapshot — just names, no card objects or scores.
    Used to seed a fresh GA run with a previous deck as starting point.
    """
    commander_name: str
    card_names: list[str]
    final_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "commander_name": self.commander_name,
            "card_names": list(self.card_names),
            "final_score": self.final_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WarmStartDeck":
        return cls(
            commander_name=data["commander_name"],
            card_names=list(data.get("card_names", [])),
            final_score=float(data.get("final_score", 0.0)),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "WarmStartDeck":
        import json as _json
        from pathlib import Path as _P
        data = _json.loads(_P(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
