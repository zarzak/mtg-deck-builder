"""
Central tuning constants (v0.9.16).

Every magic number the engine uses, in one place, each with its rationale
and provenance. The project rule (see the user's scoring philosophy):
deterministic constants must be sourced to either FORMAT STRUCTURE (official
rules, universal deckbuilding norms) or an OBSERVED FAILURE in a real run —
anything else is a bug. Each entry states its source.

BuildConfig fields are deliberately NOT duplicated here — user-tunable knobs
live on the config (with CLI flags); these are the engine's internals.
"""

# ----------------------------------------------------------------------
# Synergy scale landmarks (source: the SYNERGY_SCORING_PROMPT rubric bands)
# ----------------------------------------------------------------------

# The rubric's "clear support" band starts at 60. Used by: strategy-density
# threshold, the land mana-base cutoff, and effect-class tagging guidance.
CLEAR_SUPPORT_SYNERGY = 60.0

# Lands at/above CLEAR_SUPPORT_SYNERGY are genuinely on-theme and count in
# the synergy/density averages; below it they're mana-base (v0.9.15c —
# observed failure: Boseiju-class lands were repelled by the synergy drag).
LAND_SYNERGY_THRESHOLD = CLEAR_SUPPORT_SYNERGY

# ----------------------------------------------------------------------
# cEDH mana base (v0.9.33 / #32) — community-sourced (EDHREC "Guide to Mana
# in cEDH", Draftsim cEDH guide, coolstuffinc land-count builders series):
# cEDH runs FEWER lands than casual (often ~28, as low as ~26) BECAUSE it
# packs 10-12 fast-mana rocks + dorks + rituals; the tested rule of thumb is
# ~38 TOTAL mana sources (lands + fast mana). The old model forced land >= 28
# INDEPENDENT of ramp, so it couldn't express that tradeoff. We now floor
# lands at 26 (role target) AND couple total sources >= 38 (penalty below),
# so a rock-heavy build can run leaner lands without going mana-light.
# ----------------------------------------------------------------------
CEDH_LAND_FLOOR = 26            # lands never below this at bracket 5
CEDH_MANA_SOURCES_FLOOR = 38    # lands + non-land fast mana (rocks/dorks/rituals)

# v0.9.25: strategy density is a linear ramp, not a cliff. A card earns
# density credit proportional to (synergy - LOW) / (HIGH - LOW), clamped to
# [0, 1]. Replaces the binary >=60 threshold, which (a) made scores jitter
# ±1-3 total points run-to-run as ±5-8 LLM synergy noise flipped borderline
# cards, (b) gave the GA zero gradient inside the 0-59 band, and (c) counted
# a synergy-40 format staple identically to a blank. Anchors: LOW = the
# rubric's marginal/noise boundary; HIGH = the "strong synergy" band start
# (same landmark the engine-boost flat floor uses).
DENSITY_RAMP_LOW = 30.0
DENSITY_RAMP_HIGH = 80.0

# v0.9.25: mana-infrastructure neutrality for DENSITY (not the synergy
# average). Ramp-role cards below CLEAR_SUPPORT_SYNERGY are doing a mana
# rock's job — they owe the theme nothing and leave the density set, the
# same reasoning as the v0.9.15c land rule. They deliberately STAY in the
# synergy average (unlike lands): rock slots compete with spell slots, so
# the remaining drag keeps mediocre rocks out while extreme-rate ones
# (Sol Ring-class: power ~98) win on the power dimension. Observed failure:
# Sol Ring delivered to the pool and shown to refinement in two consecutive
# runs (Jodah B4, Doom B5) and declined both times on density math alone.
RAMP_DENSITY_NEUTRAL = True

# ----------------------------------------------------------------------
# Combo fitness shape (v0.9.8 + v0.9.13 headroom fix)
# ----------------------------------------------------------------------

COMBO_NEAR1 = 0.15          # one piece away: a gradient, not a reward
COMBO_NEAR2 = 0.05          # two pieces away (only for 4+-card combos)
COMBO_REDUNDANCY = 0.25     # weight on completed combos beyond the best
COMBO_REDUNDANCY_CAP = 40.0  # many wincons can't alone max the dimension
# Extras (redundancy + near) can never fill more than this fraction of the
# headroom above the best combo — keeps a strict gradient toward upgrading
# the best combo even at saturation (observed failure: 21 assembled pairs
# pinned the score at 100 while a 95-payoff infinite sat one piece away).
COMBO_HEADROOM_FRACTION = 0.5

# ----------------------------------------------------------------------
# Consistency dimension (v0.9.14)
# ----------------------------------------------------------------------

# Diminishing marginal value of the i-th copy of a core effect class: the
# 1st copy is worth full value, decaying to a 25% floor. Redundancy is
# rewarded strongly early without paying linearly for flooding one class.
CONSISTENCY_COPY_WEIGHTS = [1.0, 0.75, 0.55, 0.40, 0.32, 0.28, 0.25, 0.25]

# ----------------------------------------------------------------------
# Bracket enforcement strengths (v0.9.15; rules source: official brackets)
# ----------------------------------------------------------------------

# Per-violation GA penalties: strong enough that the GA never profits from
# a violation, not so strong that one stray card zeroes a good deck.
BRACKET_GC_EXCESS_PENALTY = 4.0
BRACKET_EXTRA_TURN_PENALTY = 4.0
BRACKET_MLD_PENALTY = 6.0
BRACKET_COMBO_PENALTY = 8.0

# The official B3 rule is "no two-card combos BEFORE TURN SIX"; deck
# construction can't see turns, so combined mana value below this proxies
# "assemblable by ~turn 6 with normal ramp" (Heliod+Ballista MV 3 = early;
# Selenia+Mirror Universe MV 11 = late/B3-legal).
EARLY_COMBO_MV = 10

# ----------------------------------------------------------------------
# Role-quality weighting (v0.9.14)
# ----------------------------------------------------------------------

# A role-filler counts toward its target weighted by min(1, base + power/div):
# power >= 60 counts fully (no change for reasonable cards); power 20 counts
# ~0.67 (observed failure: within-role quality was fitness-invisible).
ROLE_QUALITY_BASE = 0.5
ROLE_QUALITY_DIVISOR = 120.0

# ----------------------------------------------------------------------
# GA internals (v0.2)
# ----------------------------------------------------------------------

GA_MUTATION_STRENGTH = 0.05     # fraction of deck swapped per mutation

# v0.9.26: fraction of mutation replacement draws that are VALUE-WEIGHTED
# (probability ∝ effective-score²) instead of uniform. Uniform-only proposal
# meant a high-value card's entry into the deck was pure dice: observed in
# the second Doom B5 run — the GA kept Prophetic Prism (syn 50/pow 48) for
# 300 generations while Sol Ring (syn 50/pow 98, strictly dominant, fitness
# verified +0.3) sat in the same ramp category unproposed. The uniform share
# preserves exploration/diversity; squaring sharpens the bias so top cards
# are ~3x likelier than average, not ~1.5x.
# v0.9.27: applies during the FULL evaluation phase only — biasing the fast
# phase homogenized the population against a heuristic that can't see
# combos/consistency (observed: third Doom run stalled and early-stopped
# inside the fast phase, shipping a deck the full evaluator never scored).
GA_MUTATION_VALUE_BIAS = 0.5
GA_FAST_PHASE_FRACTION = 0.5    # generations on the fast evaluator

# ----------------------------------------------------------------------
# LLM selection tournament (v0.9.5/v0.9.6)
# ----------------------------------------------------------------------

# ~150 entries is the empirical comfort zone for "rank these cards" tasks;
# past it the model drops middle items.
SELECT_CARDS_CHUNK_SIZE = 150
# Pools at/below 2x chunk size take a single call — avoids the asymmetric
# shrinkage of one more elimination round.
SELECT_CARDS_MAX_SINGLE_PASS = 300
