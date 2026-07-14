# MTG Deck Builder - Changelog v0.2.0

This release addresses issues identified during a full code review after
the v0.1.0 code stabilized. All 20+ items are addressed here.

## Critical Bug Fixes

### 1. Role detection was broken (bug #2 from review)
**Problem**: `_count_role` ramp detection matched ANY card with "add" OR "mana"
OR "land" in its text. Tons of non-ramp cards matched, causing GA to think
every deck had plenty of ramp.

**Fix**: Rewrote role detection with proper per-role logic. Ramp now requires
"Add {X} mana" patterns or land-search patterns, with exclusions for clearly
non-ramp cards. Also added `protection`, `threat`, `recursion` roles.

### 2. Two-phase evaluator carried stale fitness scores (bug #10)
**Problem**: At generation 50, we switched from fast to full evaluator, but
the check `if individual.fitness == 0` meant already-evaluated individuals
kept their stale fast scores. Elite carry-overs never got re-evaluated.

**Fix**: Track which evaluator produced each fitness score via a new
`fitness_is_full` flag on Individual. Force re-evaluation when evaluator
mode changes.

### 3. Synergy/power balance formula was decorative (bug #5)
**Problem**: `synergy_weight` and `base_weight` were in BuildConfig but
never referenced in code. Our whole design discussion about Llanowar Elves
vs Grizzly Bears was never implemented.

**Fix**: Created `_compute_effective_synergy()` in DeckEvaluator that combines
LLM synergy score with baseline power using configurable weights. The formula
`effective = baseline * base_weight + synergy * synergy_weight` now actually
drives scoring. Also added `commander_adaptive_weights` option so the LLM's
commander analysis can override these defaults per-commander.

### 4. Heuristic and LLM scores used different scales (bug #4)
**Problem**: `_heuristic_synergy` uses baseline 40, +15 per keyword. LLM uses
full 0-100 range. When mixed in same deck evaluation, comparing apples to
oranges.

**Fix**: Calibrated heuristic to produce 0-100 distribution that better
matches LLM output. Baseline 35, +10-20 per match with stronger signals for
multiple matches.

### 5. Constraint penalties were too weak (issue #13)
**Problem**: Duplicate cards got 50-point penalty, color violations got 100.
These are tiny compared to 70+ score decks. GA could converge on "barely
invalid but scores well."

**Fix**: Invalid decks now get fitness of 0 in the optimizer. Penalties
still exist in DeckScores for diagnostic purposes, but invalid decks
are rejected at the GA level.

### 6. Performance: O(n²) random individual creation (bug #7)
**Problem**: `i not in land_indices` where land_indices was a list caused
O(n²) lookups during initial population creation.

**Fix**: Convert to set before lookup.

### 7. Deprecated model as default (bug identified from docs)
**Problem**: `claude-sonnet-4-20250514` is deprecated, retires June 2026.

**Fix**: Updated default to `claude-sonnet-4-5`. Added handling for Opus 4.7's
rejection of `temperature` parameter.

### 8. Card.__eq__ / __hash__ redundant override (bug #6)
**Problem**: `@dataclass(frozen=True)` auto-generates these; our override
was fragile and had subtle issues.

**Fix**: Removed custom overrides, rely on dataclass-generated ones using
only the name field via `eq=False` field config.

## Design Gaps Addressed

### 9. Commander-dependent scoring weights (issue #11)
LLM analysis now optionally returns `recommended_weights` and the scorer
uses them. Jasmine Boreal can say "power_level: 0.05, synergy: 0.50" to
flip the evaluation for vanilla creature decks.

### 10. Creativity scales with power level (issue #12)
When power_level >= 8, creativity weight is reduced automatically (high-
power decks should lean on staples). When power_level <= 4, creativity
weight is boosted (jank decks should be jank).

### 11. Better crossover handling duplicates (issue #14)
New `_crossover_v2` tracks taken indices during the sweep and picks
alternatives from the same category, avoiding the massive duplicate
replacement that degraded crossover.

### 12. Configurable early stopping (bug #8)
`patience_generations` added to BuildConfig (default 30). Also added
`min_improvement` threshold.

## Architectural Improvements

### 13. Added optional "structured GA" mode (new thought A)
New `StructuredOptimizer` decides the role mix (how many ramp, draw, etc.)
via GA, then greedily picks best cards per role. Dramatically reduces
search space while guaranteeing structural validity. Enable via
`config.use_structured_ga = True`.

### 14. LLM deck review pass (new thought B)
After GA converges, `DeckBuilder.build()` optionally runs one LLM review
pass that identifies missing answers, suggests swaps. Very cheap (one API
call), catches things the fitness function misses. Enable via
`config.enable_llm_review = True`.

### 15. Per-card telemetry and HTML report (new thought C)
`OptimizationResult` now includes `card_telemetry` with baseline, synergy,
effective score for every final card. `cli.py --report` writes an HTML
file readable on mobile.

### 16. Mock LLM mode for testing without API key
`LLMEngine` now supports `mock_mode=True` which uses deterministic heuristic
responses. Lets us test the entire pipeline without API access and enables
unit tests.

## Enhancements Added

### 17. EDHREC integration scaffolding (issue #16)
Added `edhrec_client.py` with stub that shows the correct API shape. Uses
the synergy-score endpoint (not raw inclusion) to avoid precon bias as
discussed. Not active by default.

### 18. Embedding-based synergy scaffold (issue #17)
Added optional `embedding_scorer.py`. Requires sentence-transformers; if
not installed, gracefully falls back to LLM synergy. When enabled, can
score 10,000 cards in seconds.

### 19. Budget constraint support (issue #18)
`BuildConfig.budget_max` now actually filters the candidate pool when a
price source is configured. No price source integration yet; hook is there.

### 20. Windows terminal compatibility
CLI now detects non-TTY output and uses ASCII progress indicators.

## Testing

- Added `tests/` directory with pytest tests
- `test_models.py`: validates data model behavior
- `test_card_database.py`: validates CSV loading, queries, edge cases
- `test_deck_evaluator.py`: validates each scoring dimension
- `test_deck_optimizer.py`: validates GA operations (crossover, mutation, evolution)
- `test_integration.py`: end-to-end test with mock LLM

All tests pass with the synthetic `test_cards.csv`.

## Migration Notes

The external API (DeckBuilder, BuildConfig) is mostly unchanged. Breaking
changes:
- `Card.__eq__` now uses dataclass default — if you relied on name-only
  equality, use `card.name == other.name` explicitly
- `BuildConfig.llm_model` default changed to `claude-sonnet-4-5`
- `OptimizationResult` has new `card_telemetry` field

Non-breaking additions:
- `BuildConfig.commander_adaptive_weights` (default True)
- `BuildConfig.use_structured_ga` (default False - opt-in)
- `BuildConfig.enable_llm_review` (default False - opt-in, costs extra API call)
- `BuildConfig.patience_generations` (default 30)
- `LLMConfig.mock_mode` (default False)
