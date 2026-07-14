"""
Deck Evaluator - Multi-dimensional scoring function for EDH decks.

Key v0.2 fixes:
- Uses shared card_fills_role() from card_database for consistency with pool generation
- Actually implements the synergy/baseline power formula from BuildConfig
- Heuristic synergy rescaled to 0-100 range matching LLM output
- Role counting no longer over-counts ramp
- Constraint violations are tracked but the *optimizer* treats them as
  hard rejections (not the evaluator)
"""

import logging
import re
from typing import Optional
from collections import Counter

from .models import Card, Deck, DeckScores, CommanderAnalysis, BuildConfig, CardTelemetry
from .card_database import card_fills_role, COMMON_STAPLES, is_staple  # noqa: F401
from . import tuning

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Mana-only land classifier
# ----------------------------------------------------------------------
# Distinguishes lands that exist purely to produce mana (basics, duals,
# tris, fetches, tap-lands, mana-confluence-style) from utility lands
# that interact with the strategy (Maze of Ith, Strip Mine, Karn's
# Bastion, Gavony Township, Hall of Heliod's Generosity, etc.).
#
# The synergy dimension computes its average over non-mana-only lands
# so that the metric tracks actual strategy density rather than getting
# pulled toward the mid-50s by 36 basics scoring ~25 each.
#
# We classify by STRIPPING known mana-production patterns from the
# card's text and checking whether anything substantive remains. If the
# residual text is empty or trivial, it's a mana-only land. This is
# more robust than maintaining a name list (which can never cover the
# whole card pool) or relying on type-line alone.

# Patterns that we treat as "just mana production" — these get scrubbed
# from a land's text before residual analysis.
_MANA_PATTERN_REGEXES = [
    # "{T}: Add {W}." / "{T}: Add {W} or {U}." / "{T}: Add one mana of any color."
    re.compile(r"\{t\}\s*:\s*add[^.]*\.", re.IGNORECASE),
    # "{T}, Pay 1 life: Add one mana of any color." (Mana Confluence-style)
    re.compile(
        r"\{t\}\s*,\s*pay \d+ life\s*[:,.]?\s*add[^.]*\.",
        re.IGNORECASE,
    ),
    # Fetch lands: "{T}, Pay 1 life, Sacrifice this land: Search your library for a Plains or Forest..."
    re.compile(
        r"\{t\}\s*,\s*(pay \d+ life\s*,\s*)?sacrifice[^:]*"
        r":\s*search your library for[^.]*land[^.]*\.",
        re.IGNORECASE,
    ),
    # "{Color}: Add ..." (channel-style mana abilities, rare)
    re.compile(r"\{[wubrgc]\}\s*:\s*add[^.]*\.", re.IGNORECASE),
    # "This land enters tapped." / "Enters the battlefield tapped."
    re.compile(
        r"(this land|it|~) (enters|enters the battlefield) tapped\.?",
        re.IGNORECASE,
    ),
    re.compile(r"enters (the battlefield )?tapped\.?", re.IGNORECASE),
    # Reminder text in parens (some basics have reminder text)
    re.compile(r"\([^)]*\)"),
]


def is_mana_only_land(card: Card) -> bool:
    """
    True iff this card is a land whose only function is producing mana.

    Returns True for: basics, duals, tris, fetch lands, tap-lands,
    Mana Confluence-style "pay life for any color" lands, and vanilla
    enters-tapped duals.

    Returns False for ALL non-lands, and for utility lands like Strip
    Mine, Maze of Ith, Karn's Bastion, Gavony Township, Hall of Heliod's
    Generosity, Bojuka Bog — any land whose text contains a non-mana
    ability that interacts with the game state.

    Used by the synergy dimension to exclude pure mana lands from the
    average (they fundamentally can't be commander-synergistic) while
    still counting utility lands that may be strategy-defining.
    """
    if not card.card_type or "land" not in card.card_type.lower():
        return False

    text = (card.text or "").strip()
    if not text:
        return True  # vanilla land

    # Strip out every recognised mana-production pattern.
    residual = text
    for pattern in _MANA_PATTERN_REGEXES:
        residual = pattern.sub("", residual)

    # Collapse whitespace and check what's left. Anything substantive
    # means there's a non-mana ability.
    residual = re.sub(r"\s+", " ", residual).strip()
    return len(residual) < 15


# Ideal mana curve distribution for EDH midrange decks.
# Aggressive decks want more 1-2 drops; control wants more 4+.
# We store these as percentages of NON-LAND cards.
IDEAL_CURVE = {
    0: 0.02,   # 2% at 0 CMC
    1: 0.10,   # 10% at 1 CMC
    2: 0.22,   # 22% at 2 CMC
    3: 0.22,   # 22% at 3 CMC
    4: 0.18,   # 18% at 4 CMC
    5: 0.12,   # 12% at 5 CMC
    6: 0.08,   # 8% at 6 CMC
    7: 0.06,   # 6% at 7+ CMC (combined bucket)
}

# v0.9.15: bracket-5 (cEDH) curve template. cEDH decks are structurally
# low-to-the-ground (average nonland MV ~1.8-2.2): heavy 0-2 drop density
# (fast mana, dorks, cheap interaction) with a thin tail of high-impact
# top-end. The midrange IDEAL_CURVE actively punishes correct cEDH builds
# (observed: a Kinnan run scored Mana Curve 71 for the RIGHT curve).
CEDH_CURVE = {
    0: 0.08,
    1: 0.26,
    2: 0.28,
    3: 0.16,
    4: 0.10,
    5: 0.06,
    6: 0.04,
    7: 0.02,
}


# COMMON_STAPLES and is_staple now live in card_database.py (v0.9.4) so both
# pool generation and evaluation share them without a circular import.
# They're imported at the top of this module and re-exported for
# backward compatibility (existing call sites + __init__.py).


class DeckEvaluator:
    """
    Evaluates EDH decks across multiple dimensions.

    Usage:
        evaluator = DeckEvaluator(config, analysis, synergy_cache)
        scores = evaluator.evaluate(deck)
        total = scores.total(config.get_effective_weights(analysis))
    """

    # All roles we count/score against.
    # 'wipe' and 'land' are included in role_targets; others we just track.
    # v0.9.14: removal sub-types added so coverage + the shortfall penalty
    # can see the removal SPREAD, not just the count.
    TRACKED_ROLES = ('ramp', 'draw', 'removal', 'removal_creature',
                     'removal_artifact', 'wipe', 'threat',
                     'protection', 'recursion', 'land')

    def __init__(
        self,
        config: BuildConfig,
        analysis: CommanderAnalysis,
        synergy_cache: Optional[dict[str, float]] = None,
        baseline_power_cache: Optional[dict[str, float]] = None,
        flavor_tag_scorer: Optional[object] = None,
        combos: Optional[list] = None,
        card_effect_classes: Optional[dict[str, str]] = None,
        banned_combos: Optional[list] = None,
    ):
        """
        Initialize evaluator.

        Args:
            config: Build configuration
            analysis: Commander analysis (drives synergy scoring)
            synergy_cache: Pre-computed synergy scores (card name -> 0-100)
            baseline_power_cache: Pre-computed baseline power (card name -> 0-100).
                If None, uses heuristic estimation.
            flavor_tag_scorer: Optional FlavorTagScorer (v0.5) for art-tag-based
                flavor scoring. When provided, combines with tribal subtype
                scoring from v0.4.
            combos: Optional list of Combo (v0.9.8) for interaction-aware
                scoring. Empty/None disables the combo dimension. The caller
                should pass only BRACKET-LEGAL combos here (banned ones must
                not be rewarded).
            card_effect_classes: Optional {card name -> effect class name}
                (v0.9.14), produced by the LLM synergy-scoring pass. Together
                with analysis.core_effect_classes this powers the consistency
                dimension. Empty/None disables it.
            banned_combos: Optional list of Combo banned at this bracket
                (v0.9.15, e.g. two-card infinites at brackets 1-2). Fully
                assembled banned combos incur a constraint penalty.
        """
        self.config = config
        self.analysis = analysis
        self.synergy_cache: dict[str, float] = synergy_cache or {}
        self.baseline_power_cache: dict[str, float] = baseline_power_cache or {}
        self.flavor_tag_scorer = flavor_tag_scorer
        self.combos: list = combos or []
        self.card_effect_classes: dict[str, str] = card_effect_classes or {}
        self.banned_combos: list = banned_combos or []

        # Lookup effective synergy/base weights up front (may be commander-adjusted)
        self.base_weight, self.synergy_weight = config.get_effective_synergy_balance(analysis)

        # Statistics
        self.eval_count = 0
        self.synergy_cache_hits = 0
        self.baseline_cache_hits = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, deck: Deck) -> DeckScores:
        """
        Evaluate a deck across all dimensions. Returns DeckScores.
        The deck's .scores attribute is also updated in place.
        """
        self.eval_count += 1

        valid, violations = deck.validate()
        role_counts = self._count_all_roles(deck)

        scores = DeckScores(
            mana_curve=self._score_mana_curve(deck),
            role_coverage=self._score_role_coverage(deck),
            synergy=self._score_synergy(deck),
            strategy_density=self._score_strategy_density(deck),  # v0.9.3
            power_level=self._score_power_level(deck),
            creativity=self._score_creativity(deck),
            combo=self._score_combos(deck),  # v0.9.8
            consistency=self._score_consistency(deck),  # v0.9.14
            flavor=self._score_flavor(deck),  # v0.4
            constraint_penalty=(
                self._calculate_penalties(deck, violations)
                + self._role_shortfall_penalty(role_counts)  # v0.9.13
                + self._bracket_penalty(deck)  # v0.9.15
                + self._cedh_mana_base_penalty(deck)  # v0.9.33 (#32)
            ),
            is_valid=valid,
            violation_reasons=violations,
            role_counts=role_counts,
        )

        # Compute the "effective synergy" combining synergy + baseline.
        # This is what we were supposed to have from the start.
        scores.effective_synergy = self._compute_effective_synergy(deck)

        deck.scores = scores
        return scores

    def build_telemetry(self, deck: Deck) -> list[CardTelemetry]:
        """
        Generate per-card telemetry for the final deck. Lets us see
        *why* each card was picked.
        """
        telemetry = []
        for card in deck.cards:
            baseline = self._get_card_baseline(card)
            synergy = self._get_card_synergy(card)
            effective = baseline * self.base_weight + synergy * self.synergy_weight
            role = self._classify_card_role(card)
            telemetry.append(CardTelemetry(
                name=card.name,
                baseline_power=baseline,
                synergy_score=synergy,
                effective_score=effective,
                role=role,
            ))
        return telemetry

    def get_evaluation_stats(self) -> dict:
        """Get statistics about evaluation performance."""
        total_card_lookups = self.eval_count * 99  # approximate
        return {
            'total_evaluations': self.eval_count,
            'synergy_cache_hits': self.synergy_cache_hits,
            'baseline_cache_hits': self.baseline_cache_hits,
            'synergy_hit_rate': self.synergy_cache_hits / max(1, total_card_lookups),
        }

    # ------------------------------------------------------------------
    # Scoring dimensions
    # ------------------------------------------------------------------

    def _score_mana_curve(self, deck: Deck) -> float:
        """
        Score mana curve distribution (0-100).
        Compares actual distribution to IDEAL_CURVE using squared deviation.
        """
        non_lands = [c for c in deck.cards if not c.is_land]
        if not non_lands:
            return 0.0

        # Bucket 7+ CMC together (matches how the curve templates are defined)
        curve = Counter(min(c.mana_value, 7) for c in non_lands)
        total = len(non_lands)

        # v0.9.15: bracket 5 uses the cEDH curve template.
        template = (
            CEDH_CURVE if getattr(self.config, "bracket", 4) == 5
            else IDEAL_CURVE
        )
        deviation = 0.0
        for mv in range(8):
            ideal = template.get(mv, 0.04)
            actual_pct = curve.get(mv, 0) / total
            deviation += ((actual_pct - ideal) ** 2) / max(ideal, 0.01)

        # Typical deviation for reasonable decks is 0-1.5.
        # Map to 100 at deviation=0, 0 at deviation=2.
        score = max(0.0, 100 - deviation * 50)
        return score

    def _score_role_coverage(self, deck: Deck) -> float:
        """
        Score role coverage (0-100).
        Each role in config.role_targets contributes equally.
        v0.4: respects role_target_overrides via get_effective_role_targets().

        v0.9.14 (quality_weighted_roles): each card counts toward its role
        weighted by baseline power — min(1.0, 0.5 + power/120) — so a
        power-60+ card counts fully but a power-20 filler counts ~0.67.
        Without this, within-role quality differences moved the total by
        ~0.06 points (fitness-invisible), so the GA filled role slots with
        arbitrary qualifying cards (Sylvan Ranger over Rampant Growth).
        """
        scores = []
        effective_targets = self.config.get_effective_role_targets()
        quality = getattr(self.config, "quality_weighted_roles", False)
        for role, (min_target, max_target) in effective_targets.items():
            if quality:
                count = self._quality_weighted_role_count(deck, role)
            else:
                count = self._count_role(deck, role)
            scores.append(self._score_role_count(count, min_target, max_target))

        return sum(scores) / len(scores) if scores else 50.0

    def _quality_weighted_role_count(self, deck: Deck, role: str) -> float:
        """Role count with each filler weighted by its baseline power.

        Weight = min(1.0, 0.5 + power/120): power >= 60 counts fully (the
        common case — no behavior change for reasonable cards), power 20
        counts ~0.67. The GA then needs either better fillers or more weak
        ones to hit the target — a real gradient toward role QUALITY."""
        total = 0.0
        for c in deck.cards:
            if card_fills_role(c, role):
                power = self._get_card_baseline(c)
                total += min(1.0, tuning.ROLE_QUALITY_BASE
                             + power / tuning.ROLE_QUALITY_DIVISOR)
        return total

    @staticmethod
    def _score_role_count(count: int, min_t: int, max_t: int) -> float:
        """Score a single role count against its target range."""
        if count < min_t:
            # Linearly scale from 0 (count=0) to 80 (count=min_target).
            # 80 because we still want to reward partial coverage.
            return (count / min_t) * 80 if min_t > 0 else 100.0
        elif count > max_t:
            # Slight penalty for excess: each card over max is -5 points, floor at 70
            excess = count - max_t
            return max(70.0, 100 - excess * 5)
        else:
            return 100.0

    def _score_synergy(self, deck: Deck) -> float:
        """
        Average synergy score across the deck's non-mana-only cards.

        v0.9.3: lands whose only function is producing mana (basics,
        duals, tris, fetches, etc.) are EXCLUDED from this average.
        Pure mana lands fundamentally can't be commander-synergistic,
        so including them in the average pulls a strategy-defining
        deck's synergy score from ~70 down to ~47 — the metric stops
        tracking deck quality. Utility lands (Strip Mine, Karn's
        Bastion, Gavony Township, Hall of Heliod's Generosity, etc.)
        ARE counted because their non-mana abilities can be
        commander-relevant.

        Uses synergy_cache if present, otherwise heuristic. Both
        encode the LLM's tag-aware synergy score for each card.
        """
        if not deck.cards:
            return 0.0

        scoreable_cards = self._synergy_scoreable(deck)
        if not scoreable_cards:
            # Degenerate case (somehow only mana lands) — fall back to
            # the old behavior so we don't divide by zero.
            return 50.0

        total = sum(self._get_card_synergy(c) for c in scoreable_cards)
        return total / len(scoreable_cards)

    # Lands at or above this synergy are genuinely ON-THEME (Gaea's Cradle
    # in an elves deck) and count in the synergy/density averages; below it
    # they're doing a LAND'S job and are treated as mana-base. Matches the
    # "clear support" band the strategy-density threshold already uses.
    _LAND_SYNERGY_THRESHOLD = tuning.LAND_SYNERGY_THRESHOLD

    def _synergy_scoreable(self, deck: Deck) -> list[Card]:
        """Cards that count toward the synergy/density averages.

        v0.9.15c: ALL lands below the clear-support synergy threshold are
        excluded — not just mana-only lands. Previously a utility land with
        rules text (Boseiju-class, power 90, honest synergy ~35) DRAGGED the
        synergy average while a plain tapped dual (mana-only, excluded) was
        neutral — so the fitness actively repelled the format's best
        utility lands in favor of textless filler (observed in a real cEDH
        run: Eldrazi Temple and a tapped dual beat Boseiju and Tropical
        Island). A land doing a land's job owes the theme nothing; a land
        that IS on theme still counts and is still rewarded.
        """
        return [
            c for c in deck.cards
            if not c.is_land
            or self._get_card_synergy(c) >= self._LAND_SYNERGY_THRESHOLD
        ]

    def _score_strategy_density(self, deck: Deck) -> float:
        """
        On-strategy MASS of the deck's non-mana-base cards, scaled 0-100.

        v0.9.25: each card contributes linear ramp credit
        clamp((synergy - DENSITY_RAMP_LOW) / (HIGH - LOW), 0, 1) instead of
        the old binary >=60 cliff. A synergy-55 card is now worth ~half a
        synergy-80 card rather than nothing; borderline LLM-score jitter
        moves density proportionally instead of flipping whole cards; and
        the GA has a gradient inside the formerly-flat 0-59 band.

        Two infrastructure carve-outs (cards doing a MANA job owe the
        theme nothing):
        - lands below clear-support (v0.9.15c, via _synergy_scoreable) —
          excluded from synergy average AND density;
        - ramp-role cards below clear-support (v0.9.25) — excluded from
          DENSITY ONLY. They stay in the synergy average, so mediocre rocks
          still pay a real cost while Sol Ring-class rate wins on power.

        Real well-built decks land around 40-65; filler-heavy below 25.
        """
        if not deck.cards:
            return 0.0

        scoreable_cards = self._synergy_scoreable(deck)
        if tuning.RAMP_DENSITY_NEUTRAL:
            scoreable_cards = [
                c for c in scoreable_cards
                if not (self._get_card_synergy(c)
                        < tuning.CLEAR_SUPPORT_SYNERGY
                        and card_fills_role(c, 'ramp'))
            ]
        if not scoreable_cards:
            return 0.0

        lo, hi = tuning.DENSITY_RAMP_LOW, tuning.DENSITY_RAMP_HIGH
        span = hi - lo
        credit = sum(
            min(1.0, max(0.0, (self._get_card_synergy(c) - lo) / span))
            for c in scoreable_cards
        )
        return (credit / len(scoreable_cards)) * 100.0

    def _score_power_level(self, deck: Deck) -> float:
        """
        Average baseline power (0-100) over the non-mana-land cards.

        Uses baseline_power_cache (LLM card power in v0.9.7, or EDHREC) if
        present, otherwise a heuristic based on mana efficiency and staple
        status.

        v0.9.7: mana-only lands (basics, plain tap-for-mana lands) are
        EXCLUDED. They are intrinsically low-power and roughly constant across
        decks, so averaging them in drags every deck toward ~50 and masks the
        real differences between the cards that actually do something — the
        same dilution that made this dimension look static. Utility lands
        (Gaea's Cradle, High Market, etc.) are NOT mana-only and still count.
        """
        if not deck.cards:
            return 0.0
        scoreable = [c for c in deck.cards if not is_mana_only_land(c)]
        if not scoreable:
            return 0.0
        total = sum(self._get_card_baseline(c) for c in scoreable)
        return total / len(scoreable)

    def _score_creativity(self, deck: Deck) -> float:
        """
        Score creativity / originality (0-100).

        Penalizes decks that are mostly staples. Rewards interesting choices.
        This score is also auto-adjusted by power_level in get_effective_weights,
        so at high power we don't penalize staple-heavy decks as much.
        """
        if not deck.cards:
            return 0.0

        # Exclude basic lands from staple-ratio calculation (they're always there)
        non_basics = [c for c in deck.cards if not c.is_basic_land]
        if not non_basics:
            return 50.0

        staple_count = sum(1 for c in non_basics if is_staple(c))
        staple_ratio = staple_count / len(non_basics)

        target = self.config.creativity_target
        threshold = self.config.staple_penalty_threshold

        if staple_ratio <= target:
            return 100.0
        elif staple_ratio >= threshold:
            # Aggressive penalty above threshold
            excess = staple_ratio - threshold
            return max(30.0, 80 - excess * 200)
        else:
            # Smooth transition between target and threshold
            range_size = threshold - target
            position = (staple_ratio - target) / range_size
            return 100 - position * 20  # 100 -> 80 over the transition zone

    # v0.9.8: interaction-aware combo scoring tuning knobs. The near-complete
    # term is deliberately small (a gradient toward completion, not a reward
    # for hoarding fragments) and uses the BEST single near-combo, not a sum —
    # otherwise a hub card in many combos (Archangel of Thune, ~19 combos)
    # floods the partial credit and saturates the dimension on its own.
    # Values live in tuning.py (single source of truth + rationale).
    _COMBO_NEAR1 = tuning.COMBO_NEAR1
    _COMBO_NEAR2 = tuning.COMBO_NEAR2
    _COMBO_REDUNDANCY = tuning.COMBO_REDUNDANCY
    _COMBO_REDUNDANCY_CAP = tuning.COMBO_REDUNDANCY_CAP

    def _score_combos(self, deck: Deck) -> float:
        """Interaction-aware combo score (0-100).

        The COMMANDER counts as always-present (it's a permanent combo piece in
        every game), so "X + Commander" combos complete when X is in the deck.

        For each combo (a set of n cards), count k present in the deck:
          - k == n      -> full payoff (the combo is assembled)
          - k == n-1    -> payoff * _COMBO_NEAR1   (one piece away: a gradient)
          - k == n-2    -> payoff * _COMBO_NEAR2   (only when n >= 4, k >= 2)

        score = best_completed
                + REDUNDANCY * min(Σ other completed, REDUNDANCY_CAP)
                + best single near-complete contribution

        Using the BEST near (not the sum) means a hub card can't saturate the
        score from partials alone — there's a real gradient from "near" to
        "assembled" that the GA can climb. Capped to 100.

        v0.9.13: the redundancy+near extras are additionally capped at HALF
        the headroom above `best` (see below). Without this, a deck with a
        strong best combo plus a rich synergy web saturates at the 100 cap,
        and completing an even better combo adds ZERO fitness — observed in
        a real run where 21 assembled pairs pinned the score at 100 while
        the deck sat one piece away from a 95-payoff infinite (the near
        credit for that very combo was helping saturate the score). The
        headroom squeeze keeps a strict gradient: raising `best` always
        raises the score, even at saturation.
        """
        if not self.combos or not deck.cards:
            return 0.0

        names = {c.name for c in deck.cards}
        if deck.commander is not None:
            names.add(deck.commander.name)

        completed: list[float] = []
        near: list[float] = []
        for combo in self.combos:
            cards = getattr(combo, "cards", None) or []
            n = len(cards)
            if n < 2:
                continue
            payoff = float(getattr(combo, "payoff", 0.0))
            k = sum(1 for cn in cards if cn in names)
            if k >= n:
                completed.append(payoff)
            elif k == n - 1:
                near.append(payoff * self._COMBO_NEAR1)
            elif k == n - 2 and k >= 2:  # only meaningful for n >= 4
                near.append(payoff * self._COMBO_NEAR2)

        if not completed and not near:
            return 0.0

        completed.sort(reverse=True)
        best = completed[0] if completed else 0.0
        rest = sum(completed[1:])
        near_gradient = max(near) if near else 0.0
        extras = (
            self._COMBO_REDUNDANCY * min(rest, self._COMBO_REDUNDANCY_CAP)
            + near_gradient
        )
        # Headroom squeeze (v0.9.13): extras can never fill more than half
        # the gap to 100, so upgrading the BEST combo always gains fitness.
        score = best + min(
            extras, tuning.COMBO_HEADROOM_FRACTION * (100.0 - best))
        return max(0.0, min(100.0, score))

    # v0.9.14: diminishing marginal value of the i-th copy of a core effect
    # class. The 1st copy is worth full value, the 2nd 75%, decaying to a
    # 25% floor — so redundancy is rewarded strongly early (consistency!)
    # without paying linearly for flooding a single class.
    _CONSISTENCY_COPY_WEIGHTS = tuning.CONSISTENCY_COPY_WEIGHTS

    def _score_consistency(self, deck: Deck) -> float:
        """Consistency/redundancy score (0-100), v0.9.14.

        For each core effect class the commander analysis declared (e.g.
        "repeatable lifegain trigger", min_count 4), count the deck cards the
        LLM tagged with that class and credit them with diminishing marginal
        weights, normalized against the class's min_count. A class at or
        above its minimum scores 1.0; a class with ZERO copies scores 0 —
        the plan can't function without the effect at all.

        This is the set-level property per-card averages can't express: to a
        synergy average, the 2nd soul sister and a 5th anthem look alike; to
        the PLAN, the 2nd copy of the critical effect is worth far more.

        Returns 0 when effect-class data is unavailable (mock / embedding
        scoring paths) — the dimension's weight is only injected when the
        analysis emitted classes, so absent data never penalizes.
        """
        classes = getattr(self.analysis, "core_effect_classes", None) or []
        if not classes or not self.card_effect_classes or not deck.cards:
            return 0.0

        # Count deck cards per (normalized) class name.
        counts: dict[str, int] = {}
        for card in deck.cards:
            cls = self.card_effect_classes.get(card.name)
            if cls:
                key = cls.strip().lower()
                counts[key] = counts.get(key, 0) + 1

        weights = self._CONSISTENCY_COPY_WEIGHTS
        fills: list[float] = []
        for entry in classes:
            name = str(entry.get("name", "")).strip().lower()
            if not name:
                continue
            try:
                min_count = int(entry.get("min_count", 3))
            except (TypeError, ValueError):
                min_count = 3
            min_count = max(1, min(min_count, len(weights)))
            have = counts.get(name, 0)
            credited = min(have, min_count)
            earned = sum(weights[:credited])
            needed = sum(weights[:min_count])
            fills.append(earned / needed if needed > 0 else 0.0)

        if not fills:
            return 0.0
        return 100.0 * (sum(fills) / len(fills))

    def _score_flavor(self, deck: Deck) -> float:
        """
        Score thematic/tribal coherence (0-100).

        Distinct from synergy (which rewards mechanical-text keyword matches).
        Two independent signals combine here:

        1. **Tribal subtype alignment** (v0.4): creatures sharing a subtype
           with the commander boost the score.
        2. **Art-tag alignment** (v0.5): if a FlavorTagScorer is configured
           (via config.flavor_art_tags), cards whose artwork matches any of
           the user's themes boost the score — regardless of card mechanics.

        When both signals are available, we take the MAX. This is deliberate:
        if you're running a strong-tribal deck, art tags shouldn't penalize
        you; if you're running a thematic-art deck (say, all "forest" art
        under a non-tribal commander), tribal shouldn't penalize you. Max
        captures "best-aligned dimension."

        Returns 50 (neutral) when no flavor signal is available (e.g.,
        Planeswalker commander, no art tags).

        Scale:
        - 50: neutral / no signal
        - 65-75: moderate alignment
        - 80-95: strong alignment
        """
        tribal_score = self._score_flavor_tribal(deck)
        art_tag_score: Optional[float] = None
        if self.flavor_tag_scorer is not None:
            try:
                art_tag_score = self.flavor_tag_scorer.score_deck(deck)
            except Exception as e:
                # Never let flavor scoring break the evaluator, but surface
                # the failure — a silent zero here misleads users who
                # explicitly asked for flavor-tag scoring.
                logger.warning(f"Art-tag flavor scoring failed: {e}")
                art_tag_score = None

        # Combine: take the max when both available, else use whichever we have
        if art_tag_score is not None and tribal_score is not None:
            return max(tribal_score, art_tag_score)
        if art_tag_score is not None:
            return art_tag_score
        return tribal_score if tribal_score is not None else 50.0

    def _score_flavor_tribal(self, deck: Deck) -> float:
        """
        Score thematic/tribal coherence via shared subtypes with commander
        (v0.4 flavor logic; renamed in v0.5 to make room for the combined
        scorer above).
        """
        # Pull commander subtypes from the commander card. analysis doesn't
        # carry subtypes directly, so we look at the commander stored on the deck.
        commander_card = deck.commander
        if commander_card is None or not commander_card.subtypes:
            return 50.0

        # Commander's creature subtypes (e.g., "Unicorn" from Lathiel)
        cmd_subtypes = set(
            s.strip().lower()
            for s in commander_card.subtypes.split(",")
            if s.strip()
        )
        if not cmd_subtypes:
            return 50.0

        # Count creatures in deck that share ≥1 subtype with commander
        creatures = [c for c in deck.cards if c.is_creature]
        if not creatures:
            return 50.0  # can't measure tribal with no creatures

        shared_count = 0
        for card in creatures:
            card_subs = set(
                s.strip().lower()
                for s in (card.subtypes or "").split(",")
                if s.strip()
            )
            if card_subs & cmd_subtypes:
                shared_count += 1

        tribal_ratio = shared_count / len(creatures)

        # Map tribal_ratio → score
        # 0% shared: 40 (light penalty for zero theme)
        # 20% shared: 60 (decent)
        # 50% shared: 85 (strong theme)
        # 100% shared: 95 (full tribal, capped because pure tribal is rare/hard)
        if tribal_ratio <= 0.2:
            score = 40 + tribal_ratio * 100  # 40 → 60
        elif tribal_ratio <= 0.5:
            score = 60 + (tribal_ratio - 0.2) * 83  # 60 → 85 over 0.2-0.5
        else:
            score = 85 + (tribal_ratio - 0.5) * 20  # 85 → 95 over 0.5-1.0

        return max(0.0, min(100.0, score))

    def _cedh_mana_base_penalty(self, deck: Deck) -> float:
        """v0.9.33 (#32): bracket-5 COUPLED mana-source floor.

        cEDH counts total mana sources (lands + fast mana), not lands alone —
        a rock-heavy build runs fewer lands on purpose. The land role floor
        (26) already forbids going too land-light; this adds the other half
        of the community rule: lands + non-land fast mana (rocks / dorks /
        rituals, i.e. the ramp role) must clear CEDH_MANA_SOURCES_FLOOR (38).
        Together they encode "lands >= 26 AND lands + fast mana >= 38".

        Reuses role_shortfall_penalty's per-missing-source rate so the GA
        sees one consistent mana-base gradient. bracket-5 only; 0 otherwise.
        """
        if getattr(self.config, "bracket", 4) != 5 or not deck.cards:
            return 0.0
        rate = getattr(self.config, "role_shortfall_penalty", 0.0)
        if rate <= 0:
            return 0.0
        lands = sum(1 for c in deck.cards if c.is_land)
        fast_mana = sum(
            1 for c in deck.cards
            if not c.is_land and card_fills_role(c, 'ramp')
        )
        deficit = max(0, tuning.CEDH_MANA_SOURCES_FLOOR - (lands + fast_mana))
        return rate * deficit

    def _role_shortfall_penalty(self, role_counts: dict[str, int]) -> float:
        """v0.9.13: penalty for roles below their minimum target.

        The role_coverage AVERAGE under-prices shortfalls once synergy scoring
        is honest: cutting a ramp card for an on-theme body gains more in the
        synergy/density terms than it loses in coverage, so the GA rationally
        starves mana and interaction (observed: 4-ramp/3-removal decks). A
        per-missing-card penalty makes filling minimums strictly better while
        keeping a smooth gradient the GA can climb. Uses the same effective
        targets (incl. user overrides) as role coverage.
        """
        rate = getattr(self.config, "role_shortfall_penalty", 0.0)
        if rate <= 0:
            return 0.0
        shortfall = 0
        for role, (min_t, _max_t) in self.config.get_effective_role_targets().items():
            count = role_counts.get(role)
            if count is None:
                continue  # role not tracked/counted — don't guess
            shortfall += max(0, min_t - count)
        return rate * shortfall

    # v0.9.15: bracket-rule enforcement strengths (per violating card/combo).
    # Strong enough that the GA never profits from a violation; not so
    # strong that a single stray card zeroes an otherwise-good deck.
    _BRACKET_GC_EXCESS_PENALTY = tuning.BRACKET_GC_EXCESS_PENALTY
    _BRACKET_EXTRA_TURN_PENALTY = tuning.BRACKET_EXTRA_TURN_PENALTY
    _BRACKET_MLD_PENALTY = tuning.BRACKET_MLD_PENALTY
    _BRACKET_COMBO_PENALTY = tuning.BRACKET_COMBO_PENALTY

    def _bracket_penalty(self, deck: Deck) -> float:
        """v0.9.15: penalty for bracket-rule violations (brackets 1-3 only).

        Hard pool filters upstream remove outright-banned cards (Game
        Changers at B1-2, MLD at B1-3, extra turns at B1), so this penalty
        mostly enforces the BUDGET rules a filter can't express: the B3
        Game Changer limit (best 3 stay, excess penalized), the extra-turn
        chaining limit (1 allowed at B2-3), and assembly of banned two-card
        combos (each card is individually legal; the PAIR is not).
        """
        bracket = getattr(self.config, "bracket", 4)
        if bracket >= 4 or not deck.cards:
            return 0.0
        from .bracket import (
            BRACKET_RULES, is_game_changer, is_mass_land_denial,
            grants_extra_turn,
        )
        rules = BRACKET_RULES[bracket]
        penalty = 0.0

        gc = mld = extra = 0
        seen: set[str] = set()
        for c in deck.cards:
            if c.name in seen:
                continue
            seen.add(c.name)
            if is_game_changer(c):
                gc += 1
            if is_mass_land_denial(c):
                mld += 1
            if grants_extra_turn(c):
                extra += 1

        gc_limit = rules["gc_limit"]
        if gc_limit is not None and gc > gc_limit:
            penalty += self._BRACKET_GC_EXCESS_PENALTY * (gc - gc_limit)
        if not rules["mld_allowed"] and mld:
            penalty += self._BRACKET_MLD_PENALTY * mld
        et_limit = rules["extra_turn_limit"]
        if et_limit is not None and extra > et_limit:
            penalty += self._BRACKET_EXTRA_TURN_PENALTY * (extra - et_limit)

        if self.banned_combos:
            names = {c.name for c in deck.cards}
            if deck.commander is not None:
                names.add(deck.commander.name)
            for combo in self.banned_combos:
                cards = getattr(combo, "cards", None) or []
                if cards and all(n in names for n in cards):
                    penalty += self._BRACKET_COMBO_PENALTY

        return penalty

    def _calculate_penalties(
        self,
        deck: Deck,
        violations: list[str],
    ) -> float:
        """
        Calculate constraint penalty. Note: the OPTIMIZER will reject invalid
        decks entirely (fitness=0), so this penalty exists mainly for diagnostics.
        """
        if not violations:
            return 0.0

        penalty = 0.0
        for reason in violations:
            if reason.startswith("Wrong card count"):
                penalty += 20
            elif reason.startswith("Duplicate"):
                penalty += 30
            elif reason.startswith("Color identity"):
                penalty += 50
            else:
                penalty += 10
        return penalty

    def _compute_effective_synergy(self, deck: Deck) -> float:
        """
        Compute the 'effective synergy' — the core formula from our design.

        effective = baseline * base_weight + synergy * synergy_weight

        This is the average over all cards. Gives a single number that
        captures "is this the right deck for this commander, and is it
        a powerful deck on absolute terms."
        """
        if not deck.cards:
            return 0.0

        total = 0.0
        for card in deck.cards:
            baseline = self._get_card_baseline(card)
            synergy = self._get_card_synergy(card)
            total += baseline * self.base_weight + synergy * self.synergy_weight

        return total / len(deck.cards)

    # ------------------------------------------------------------------
    # Card-level lookups (with caching)
    # ------------------------------------------------------------------

    def _get_card_synergy(self, card: Card) -> float:
        """Get synergy score for a card (0-100)."""
        if card.name in self.synergy_cache:
            self.synergy_cache_hits += 1
            return self.synergy_cache[card.name]
        return self._heuristic_synergy(card)

    def _get_card_baseline(self, card: Card) -> float:
        """Get baseline power score for a card (0-100)."""
        if card.name in self.baseline_power_cache:
            self.baseline_cache_hits += 1
            return self.baseline_power_cache[card.name]
        # Card-intrinsic baseline (if set) takes precedence over heuristic
        if card.baseline_power >= 0:
            return card.baseline_power
        return self._heuristic_baseline(card)

    # ------------------------------------------------------------------
    # Heuristics (used when LLM/EDHREC data not available)
    # ------------------------------------------------------------------

    def _heuristic_synergy(self, card: Card) -> float:
        """
        Fast heuristic synergy scoring without LLM.

        Rescaled to 0-100 to match LLM output so they can be mixed safely.
        - Baseline 35 (card has *some* value in a random deck)
        - +12 per synergy keyword match (saturates around 3 matches)
        - -15 per anti-synergy keyword match
        - +10 if shares creature type with commander
        - Clamped to 0-100
        """
        score = 35.0
        card_text = (card.text or '').lower()

        # Count synergy keyword hits, but saturate (more isn't linearly better)
        hits = 0
        for keyword in self.analysis.synergy_keywords:
            if keyword and keyword.lower() in card_text:
                hits += 1
        # Diminishing returns: first hit +12, second +10, third +8, etc.
        if hits > 0:
            for i in range(hits):
                score += max(4, 12 - i * 2)

        # Anti-synergy penalty
        for keyword in self.analysis.anti_synergy_keywords:
            if keyword and keyword.lower() in card_text:
                score -= 15

        # Tribal synergy: shared subtype with commander
        # (We don't have commander subtypes here, but synergy_keywords usually
        # includes creature types for tribal commanders, handled above.)

        return max(0.0, min(100.0, score))

    def _heuristic_baseline(self, card: Card) -> float:
        """
        Heuristic baseline power (0-100) when no EDHREC data.

        Rough rules:
        - Staples: 80 baseline
        - Untapped dual lands / Command Tower: 75
        - Lands with unconditional ETB-tapped: 35
        - Lands with conditional ETB-tapped (pay 2 life, control basic, etc.): 50
        - 0-mana cards: 65 (Sol Ring-like power)
        - 1-2 mana: 55
        - 3-4 mana: 50
        - 5-6 mana: 45
        - 7+ mana: 40 (tough to cast in EDH)
        """
        if is_staple(card):
            return 80.0

        if card.is_land:
            text = (card.text or '').lower()
            # Untapped multi-color lands are premium
            multicolor = (card.color_identity and
                          len(set(c for c in card.color_identity if c in 'WUBRG')) > 1)

            if multicolor:
                # Check for ETB-tapped-unless conditions (conditional shocks, etc.)
                if 'enters tapped unless' in text or 'may pay 2 life' in text:
                    return 60.0
                # Unconditional ETB-tapped dual (Scattered Groves, Canopy Vista sometimes)
                if 'enters tapped' in text or 'enters the battlefield tapped' in text:
                    return 45.0
                # Pure untapped dual
                return 70.0

            # Mono/colorless lands
            if 'enters tapped' in text or 'enters the battlefield tapped' in text:
                return 35.0
            if card.is_basic_land:
                return 55.0
            return 55.0

        mv = card.mana_value
        if mv == 0:
            return 65.0
        elif mv <= 2:
            return 55.0
        elif mv <= 4:
            return 50.0
        elif mv <= 6:
            return 45.0
        else:
            return 40.0

    # ------------------------------------------------------------------
    # Role counting & classification (shared with card_database)
    # ------------------------------------------------------------------

    def _count_role(self, deck: Deck, role: str) -> int:
        """Count cards filling a specific role in the deck."""
        return sum(1 for c in deck.cards if card_fills_role(c, role))

    def _count_all_roles(self, deck: Deck) -> dict[str, int]:
        """Count every tracked role (for diagnostics)."""
        return {role: self._count_role(deck, role) for role in self.TRACKED_ROLES}

    def _classify_card_role(self, card: Card) -> str:
        """
        Classify a card into its primary role (for telemetry).
        Checks roles in priority order; returns the first match.
        """
        priority_order = [
            'land', 'ramp', 'draw', 'wipe', 'removal',
            'recursion', 'protection', 'threat',
        ]
        for role in priority_order:
            if card_fills_role(card, role):
                return role
        return 'synergy/other'


class FastEvaluator:
    """
    Ultra-fast evaluator for initial GA generations.

    Skips LLM-based scoring entirely; uses only heuristics.
    Good for early exploration when we just want to prune obviously-bad decks.
    """

    def __init__(self, config: BuildConfig, analysis: CommanderAnalysis,
                 synergy_cache: Optional[dict[str, float]] = None):
        self.config = config
        self.analysis = analysis
        # v0.9.12: prefer the real (pre-computed) synergy scores — which carry
        # card power, combos, structural reasoning, and the EDHREC floor — so
        # the GA's EARLY generations pursue the actual strategy instead of a
        # crude keyword heuristic (critical for structural commanders, whose
        # text-less payoffs register zero keyword hits). Falls back to the
        # keyword heuristic when no cache is provided.
        self.synergy_cache: dict[str, float] = synergy_cache or {}
        # Pre-compute synergy keyword set (lowered) for the fallback path.
        self._synergy_words = [k.lower() for k in (analysis.synergy_keywords or []) if k]

    def evaluate(self, deck: Deck) -> float:
        """
        Quick evaluation returning single fitness score.
        ~10x faster than full DeckEvaluator.
        """
        if len(deck.cards) != 99:
            return 0.0

        # Quick validity: no color identity check (assumes pool is pre-filtered)
        # Just check duplicates
        seen = set()
        for card in deck.cards:
            if card.is_basic_land:
                continue
            if card.name in seen:
                return 0.0
            seen.add(card.name)

        score = 50.0

        # v0.9.15: bracket 5 (cEDH) has a structurally different shape —
        # lower curve, fewer lands (compensated by fast mana).
        cedh = getattr(self.config, "bracket", 4) == 5
        mv_sweet = (1.4, 2.4) if cedh else (2.5, 3.5)
        mv_ok = (1.0, 3.0) if cedh else (2.0, 4.0)
        # v0.9.33 (#32): cEDH sweet spot lowered to 26-30 (fast mana
        # compensates); the coupled total-source floor lives in the full
        # evaluator, not this heuristic.
        land_sweet = (26, 30) if cedh else (35, 38)
        land_ok = (24, 32) if cedh else (33, 40)

        # Mana curve: reward reasonable average MV
        non_lands = [c for c in deck.cards if not c.is_land]
        if non_lands:
            avg_mv = sum(c.mana_value for c in non_lands) / len(non_lands)
            if mv_sweet[0] <= avg_mv <= mv_sweet[1]:
                score += 15
            elif mv_ok[0] <= avg_mv <= mv_ok[1]:
                score += 5
            else:
                score -= 10

        # Land count
        land_count = sum(1 for c in deck.cards if c.is_land)
        if land_sweet[0] <= land_count <= land_sweet[1]:
            score += 10
        elif land_ok[0] <= land_count <= land_ok[1]:
            score += 5
        else:
            score -= 10

        # Synergy signal. Prefer the real pre-computed scores (cheap dict
        # lookups); fall back to the keyword heuristic only when absent.
        if self.synergy_cache:
            nonland = [c for c in deck.cards if not c.is_land]
            if nonland:
                avg = sum(self.synergy_cache.get(c.name, 30.0)
                          for c in nonland) / len(nonland)
                score += (avg / 100.0) * 30
        elif self._synergy_words:
            hit_count = 0
            for card in deck.cards:
                text = (card.text or '').lower()
                for kw in self._synergy_words:
                    if kw in text:
                        hit_count += 1
                        break
            score += (hit_count / len(deck.cards)) * 30

        return max(0.0, min(100.0, score))
