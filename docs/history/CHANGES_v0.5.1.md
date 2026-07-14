# Changes in v0.5.1 (Session 5.5 — Cleanup Pass)

No new user-facing features. This is a deliberate quality pass over five
sessions of additive development to clean up drift, duplication, and dead
code that accumulated as features layered on.

If you're a user: nothing should look different. Same flags, same outputs,
same APIs. The HTML report footer now says "v0.5.1" instead of being stuck
at "v0.4". If you were getting buggy behavior from a custom card_source
that raised exceptions, your reports will now render successfully (with
those cards just lacking images) instead of crashing.

If you're reading the codebase: a lot got shorter and cleaner.

## What got fixed

### Duplication eliminated

Three modules (`scryfall_cards.py`, `scryfall_tags.py`, `price_source.py`)
had each defined their own `_safe_filename()`, `_url_quote()`, and
`_http_get()`. Implementations had drifted subtly:
- `_safe_filename` had a 100-char limit in two places, 120 in the third
- `_http_get` set User-Agent on `urllib` fallback in some, on the
  `requests` path in others, and on neither path in `price_source.py`
- `_http_get` 404-handling was inconsistent (returned None vs body)

All consolidated into a new `mtg_deck_builder/_http.py` module with:
- `http_get_text(url, timeout, log_label)` — single entry point, tries
  `requests` then falls back to `urllib`. Always sets User-Agent and
  Accept headers on both paths. Always returns body on 200/404, None
  otherwise. Never raises.
- `safe_filename(name, max_len=100)` — pre-compiled regex, defaults to
  the more common 100-char limit
- `url_quote(s)` — wraps `urllib.parse.quote` with sensible defaults

The four client modules (`scryfall_cards`, `scryfall_tags`, `price_source`,
`edhrec_client`) now have 3-line wrappers around `_http_get` instead of
30+ lines of duplicated networking code.

### Hardcoded version strings eliminated

Six different places hardcoded `"mtg-deck-builder/<version>"` as the
User-Agent header — and the versions said `0.2`, `0.2`, `0.4`, `0.4`,
`0.5`, `0.5` (whatever version the file was at when written, then never
updated). The HTML report footer also hardcoded `"v0.4"`.

All of these now read `__version__` from `__init__.py` dynamically.
Single source of truth — version bumps only need to touch `__init__.py`.

### Dead code removed

- `IslandModel` stub class in `deck_optimizer.py` (23 lines + a `TODO:
  implement actual parallelism` comment). The real `IslandModelOptimizer`
  has lived in `island_optimizer.py` since v0.3; the stub was never used
  but was sitting there waiting to silently single-thread anyone who
  imported it by mistake.
- `BuildConfig.budget_max` field. Defined since the original v0.1 sketch,
  never actually read by any code. The real budget control,
  `budget_max_per_card`, was added in v0.3.

### HTML report exception safety

`html_report.py` had five unprotected calls to `card_source.get_image_url()`
and `card_source.get_artist()`. A buggy custom card_source implementation
that raised exceptions would crash the entire HTML render — even though
the rest of the report could've been generated fine.

Added `_safe_image_url()` and `_safe_artist()` helpers that catch any
exception and return None. All five call sites updated. Worst-case
behavior is now: that one card just renders without its image; report
still completes.

Regression test added (`test_card_source_exceptions_dont_break_report`)
using an `ExplodingCardSource` mock that raises on every method call.

### Test quality improvements

Audited 328 tests for tautological assertions, missing assertions, and
"high setup, low verification" patterns. Findings:
- Zero `assert True`-equivalents
- Zero tests with no assertions at all
- ~20 tests have only one assertion despite 15+ lines of setup; most
  are fine (testing one specific thing) but two were genuinely
  under-asserting and got strengthened:

`test_warm_start_seeds_population` previously only checked that the FIRST
individual matched the warm-start, despite using `warm_start_copies=2`.
Now verifies BOTH seeded copies match, AND that the random-init tail is
actually random (not all identical decks).

`test_evaluator_takes_max_of_signals` previously only verified one
direction of the MAX behavior (low-tribal + high-art-tag = high score).
Now verifies both directions: high-tribal + low-art-tag should also
yield a high score. Without the second direction, the test would pass
with broken "always use art-tag if available" logic.

### Docstring drift

`deck_builder.py`'s module docstring listed only the v0.2 phases (1-7).
Updated to reflect the current 10-phase pipeline including EDHREC fetch
(v0.3), budget filtering (v0.3), and locked-card injection (v0.4).
Added a list of optional integrations the orchestrator coordinates.

`cli.py` module docstring said "(v0.2)" and didn't mention the `diff`
subcommand. Updated.

## Test count

- v0.5.0: 327 passing
- v0.5.1: **328 passing** (+1 regression test for exception-safe
  HTML rendering)

## Known consistency issues NOT fixed (intentional)

Documented here so they don't have to be re-discovered by a future audit:

**Lazy-construction pattern inconsistency.** Sessions 4-5 use the
`@property`-decorated lazy-construct pattern (`card_source`, `tag_client`,
`flavor_tag_scorer`). Session 3's clients (`_edhrec_client`,
`_embedding_scorer`, `_price_source`) use the older "inline lazy with
`if None` check at use site" pattern. Both work; the property pattern is
strictly nicer. Not refactored because:
- Risk: the inline pattern is touched in multiple places per client; a
  refactor needs to update each call site without breakage
- Reward: cosmetic; both patterns produce correct behavior

If we touch the EDHREC/embedding/price-source code substantively in a
future session for other reasons, fold the property-pattern conversion
into that work.

**BuildConfig field ordering.** 47 fields in roughly chronological-add
order. Image fields and tag fields ended up next to each other (good),
but refinement fields (locked/banned/warm-start) ended up after them
because they were added in a different turn of Session 4. A logical
regrouping would aid readability but would be churn for minor benefit.
Defer.

**`_http_get` kept as instance method.** Even though it's now a 3-line
wrapper, I kept it as a method (rather than removing it) so existing
tests that monkey-patch it on instances continue to work without
modification. Specifically `test_scryfall_tags.py` does
`client._http_get = fake_get` to stub out network calls in pagination
tests. Removing the method would force rewriting those tests. Cost of
keeping the method: 3 lines of trivial wrapper per client. Worth it.

## Files added

- `mtg_deck_builder/_http.py` (~130 lines)
- `CHANGES_v0.5.1.md` (this file)

## Files modified

- `mtg_deck_builder/__init__.py` — version bump
- `mtg_deck_builder/cli.py` — docstring refresh
- `mtg_deck_builder/deck_builder.py` — docstring refresh
- `mtg_deck_builder/deck_optimizer.py` — deleted dead IslandModel stub
- `mtg_deck_builder/edhrec_client.py` — _fetch_url uses shared http
- `mtg_deck_builder/html_report.py` — defensive card_source wrappers,
  dynamic version footer
- `mtg_deck_builder/models.py` — removed dead `budget_max` field
- `mtg_deck_builder/price_source.py` — _http_get and helpers use shared
  module
- `mtg_deck_builder/scryfall_cards.py` — _http_get and helpers use
  shared module
- `mtg_deck_builder/scryfall_tags.py` — _http_get, _safe_filename, and
  url quote use shared module
- `mtg_deck_builder/tests/test_html_images.py` — regression test for
  exception safety
- `mtg_deck_builder/tests/test_refinement.py` — strengthened warm-start
  seeding test
- `mtg_deck_builder/tests/test_flavor_tags.py` — strengthened max-of-
  signals test

## Total LOC change

Net negative — duplicated code was longer than the consolidated
shared module + the dead stub class + the unused config field combined.
