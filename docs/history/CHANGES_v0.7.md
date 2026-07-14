# Changes in v0.7 (Session 7)

The big one: a **FastAPI + Jinja2 + SSE** web interface for everything
the CLI does, plus a couple of UI-driven features (rendered diff page,
tag cache pre-seed helper) that didn't exist on the CLI.

If you're a CLI-only user, this release is purely additive — nothing
about the CLI flag space, BuildConfig, or core pipeline changed. The
web interface is in a new `mtg_deck_builder.web` subpackage with its
own optional dependencies (`requirements-web.txt`).

---

## What you get

```bash
pip install -r requirements-web.txt
python -m mtg_deck_builder.web --csv cards.csv --mock
# Open http://127.0.0.1:8765
```

Five pages:

- **`/`** — home with links to the four feature pages
- **`/build`** — full `BuildConfig` form (commander, GA params, weights,
  preset, all integration flags, locked/banned cards, budget). Submit
  redirects to a status page with **live SSE progress** and an "Open
  report" button when complete.
- **`/diff`** — upload two deck snapshot JSONs, get a styled HTML
  comparison page with summary stats, commander-change callout, and
  optional role grouping.
- **`/analyze`** — type a commander name, get the LLM analysis (color
  identity, key mechanics, build-around text, evaluation notes,
  synergy keywords) without running a full optimization. Useful for
  "what does the LLM think this commander wants?" before committing
  to a 90-second build.
- **`/tags`** — pre-seed the Scryfall tag cache with a list of art
  or oracle tags. Subsequent `--tags-offline` builds use the cached
  data with no network calls.

Default bind: `127.0.0.1:8765`. Single-user local tool — **do not
expose to the internet**. No auth, no multi-user isolation, no TLS.
If you want to make this internet-facing, wrap it in a proper proxy
that handles those concerns.

---

## Architecture

### New `mtg_deck_builder.web` subpackage

- **`app.py`** — FastAPI app factory and all route handlers
- **`forms.py`** — `config_from_form(form_dict) -> BuildConfig`,
  testable without FastAPI; reuses CLI's `WEIGHT_PRESETS` for
  preset definitions
- **`state.py`** — `BuildRegistry`, `BuildState`, `ProgressEvent`;
  thread-safe in-memory store of running and completed builds
- **`diff_html.py`** — `render_diff_html(result)` styled HTML output
  with the same CSS palette as the main report
- **`templates/`** — seven Jinja templates (base + home + 5 forms +
  analyze_result + tags_result)

### Background workers + SSE bridging

Builds take 30s-2min, so we can't block the FastAPI event loop. Pattern:

1. POST handler creates a `BuildState` with a UUID and stashes it in
   the `BuildRegistry`.
2. `ThreadPoolExecutor.submit(_run_build, state, config, ...)` kicks
   off the build in a worker thread.
3. The build runs synchronously; its `progress_callback` posts events
   to the `BuildState`.
4. `state.post_event()` calls `loop.call_soon_threadsafe(queue.put_nowait, event)`
   to deliver the event into the asyncio.Queue from the worker thread.
5. The SSE handler streams events from the queue, replaying any history
   first so a late-connecting consumer doesn't miss the early phases.
6. A `__done__` sentinel event signals stream end; the JS client
   reloads the status page so the server re-renders with the report
   link populated.

Single thread per build (one user, local tool — no concurrency
worries). Memory grows with every build but a session is short; restart
clears state.

### Templates

All templates extend `base.html` which provides:
- Shared nav bar with links to /build, /diff, /analyze, /tags
- The same dark palette as `html_report.py` so the in-browser experience
  matches the generated reports
- Mobile-friendly grid (`fieldset-grid` collapses to single column at
  600px)

### Form → BuildConfig

`config_from_form()` mirrors what `cli.py` does with argparse:

- Coerces strings to int/float/bool
- Splits comma-separated tags and newline-separated card lists
- Reuses `cli.WEIGHT_PRESETS` so form preset and CLI `--preset`
  produce identical weight dicts
- Handles HTML checkbox semantics (present → True, absent → False)

Every coercion path has a defensive default so a slightly malformed
form input renders the page instead of throwing.

---

## Bug fix found while wiring

Uncovered while smoke-testing `/diff`: `deck_diff._as_multiset()` only
accepted dicts in the canonical `commander_name` / `card_names` format
that `WarmStartDeck.to_dict()` produces. If a user typed JSON by hand
with the more intuitive `commander` / `cards` keys, the diff silently
returned empty results instead of failing or showing anything.

Fixed: `_as_multiset` now accepts either format. Falls through cleanly:
`commander_name` then `commander`; `card_names` then `cards`. All
existing tests still pass since they used the canonical form, but
the web UI is now forgiving of either shape.

---

## New module exports

`mtg_deck_builder.web.create_app(csv_path, mock_llm=False, artifacts_dir=None)`
`mtg_deck_builder.web.main()` — entry point for `python -m mtg_deck_builder.web`

The web subpackage uses lazy imports so importing `mtg_deck_builder` itself
doesn't pull in FastAPI. Only `from mtg_deck_builder.web import create_app`
or running `python -m mtg_deck_builder.web` actually loads the web deps.

---

## CLI flags for the web launcher

```
python -m mtg_deck_builder.web --csv PATH [options]

  --csv PATH              required; cards CSV path
  --host HOST             default 127.0.0.1
  --port PORT             default 8765
  --mock                  Server runs all builds/analyses in mock mode
  --artifacts-dir DIR     Where to write generated HTML reports
                          (default ./web_artifacts)
  --log-level LEVEL       uvicorn log level (default info)
```

---

## Test count

- v0.6.0: 378 tests
- v0.7.0: **437 tests** (+59)
  - +22 `test_web_forms.py` (coercion, weight building, full config_from_form)
  - +10 `test_web_diff_html.py` (HTML rendering, XSS escaping, role groups)
  - +11 `test_web_state.py` (ProgressEvent, BuildState, BuildRegistry)
  - +16 `test_web_routes.py` (all routes via FastAPI TestClient,
    including full build lifecycle and SSE streaming)

All web tests use `mock_llm=True` and `--tags-offline` semantics so
they're hermetic — no network, no real LLM, no flaky external deps.

---

## What I considered and didn't ship

**Real-time SSE in a multi-tab browser session.** Each tab opens its
own `EventSource` and gets independent state replay. Works fine for
single-user local use but if you wanted multiple windows watching the
same build, that's an extra design conversation.

**Persistent build history across server restarts.** In-memory only.
Session 8+ idea: SQLite-backed history with a "recent builds" page.

**A real card-picker autocomplete in the form.** I used a `<datalist>`
which gives basic native browser autocomplete — clean and zero-JS but
not as nice as a proper combobox with type-ahead filtering. Good enough.

**Streaming progress on /diff and /analyze.** Both are fast enough
(diff is instant; analyze is one LLM call) that the request-response
pattern is fine.

**Iterative weight-slider re-scoring.** The vision was: "tweak weights
and see scores update without rebuilding." Would need a separate
in-memory candidate pool + scoring API. Real Session 8+ work.

---

## New files

- `mtg_deck_builder/web/__init__.py`
- `mtg_deck_builder/web/app.py` (~415 lines, biggest single addition)
- `mtg_deck_builder/web/forms.py` (~190 lines)
- `mtg_deck_builder/web/state.py` (~125 lines)
- `mtg_deck_builder/web/diff_html.py` (~225 lines)
- `mtg_deck_builder/web/templates/*.html` (8 files)
- `mtg_deck_builder/tests/test_web_forms.py`
- `mtg_deck_builder/tests/test_web_state.py`
- `mtg_deck_builder/tests/test_web_diff_html.py`
- `mtg_deck_builder/tests/test_web_routes.py`
- `requirements-web.txt`
- `CHANGES_v0.7.md` (this file)

## Files modified

- `mtg_deck_builder/__init__.py` — version 0.7.0
- `mtg_deck_builder/deck_diff.py` — `_as_multiset()` accepts both
  canonical and intuitive dict formats

## Files unchanged

Everything else. The web subpackage is purely additive — no core
behavior changed, no field renames, no breaking API edits.

---

## Session 8+ wishlist

- **Persistent build history** (SQLite, "recent builds" sidebar)
- **Iterative weight-slider re-scoring** without rebuilding
- **Card-by-card swap suggestions** in the report (LLM "what would you
  swap?" with diff preview)
- **Multi-deck management** (named decks, tags/categories, search)
- **Cron-style scheduled validation** ("re-check role classifications
  weekly as Scryfall tagger updates")
