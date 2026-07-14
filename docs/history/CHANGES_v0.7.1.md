# Changes in v0.7.1 (Session 7.5 — Cleanup Pass)

No new user-facing features. This is a deliberate quality pass over the
v0.7 web layer (~1,000 lines added in one session — exactly when bugs
hide and drift accumulates).

If you're a CLI-only user: nothing changed for you.

If you're a web user: two visible bug fixes (SSE duplicate events, log
duplicate lines) plus a friendlier behavior on out-of-range form input.

If you're reading the codebase: a lot got hardened.

## Bugs fixed

### SSE duplicate events (real, user-visible)

`_event_generator` had a race: `post_event()` appends to BOTH
`state.events` (history) AND `state.queue` (live stream) atomically.
The old generator iterated history first, then drained the queue —
but events posted before SSE-connect were in BOTH, so they got
yielded twice.

Fixed with a snapshot-history-then-drain pattern:
1. Snapshot `len(state.events)` to get a count of items already in history
2. Yield those (catches up the late consumer)
3. Drain that many items from the queue WITHOUT yielding (those are
   the duplicates we already covered from history)
4. Then enter the main loop awaiting truly new events

Two new unit tests exercise `_event_generator` directly, constructing
a `BuildState` by hand to deterministically arrange the race:
- `test_drain_step_prevents_duplicate_events` — would FAIL with old
  code (would expect 4 events, get 7-8)
- `test_generator_handles_no_queue` — covers the previously-crashy
  case where `state.queue is None` (would AttributeError before)

The first attempt at this regression test was an integration test
through TestClient, but I verified by reverting the fix that the
integration test couldn't reliably trigger the race — the build
finishes too fast in the test environment for SSE to connect
mid-build. Direct unit tests on the generator function are
deterministic and fast.

### Duplicate log lines on the build status page

`build_status.html` had two sources of truth for the event log:
- Server-side: `{% for evt in state.events %}` rendered every event
- Client-side: SSE stream replayed history and appended each event

Result: every event appeared twice in the visible log.

Fix: the template only pre-renders events when SSE won't run (status
is already `complete` or `failed` on first page load). When SSE will
run, the JS populates the log from scratch.

### `state.queue is None` crash

`_event_generator` previously called `state.queue.get()` without
checking. If a `BuildState` was created without queue/loop wiring
(theoretically possible if someone pokes the registry directly, or a
refactor introduces a code path that creates state outside the build
form), the generator would `AttributeError`. Now degrades gracefully:
replay history, emit synthetic `__done__`, return.

## Friendlier UX

### Range clamping for GA params

`config_from_form` now clamps `population_size` to [4, 500],
`generations` to [1, 1000], `patience_generations` to [1, 1000].
Mirrors the input min/max in `build_form.html`.

Before: a typo like `population_size=0` would crash the GA and the
user got a stack trace. After: clamps to the minimum (4), build runs
normally.

5 new tests cover the clamp behavior.

## Code hygiene

### Six unused imports removed

`web/app.py` had 5 dead imports: `sys`, `Form`, `JSONResponse`,
`format_diff`, `DiffResult`. `web/state.py` had a stray `traceback`.
All removed.

### Pinned upper bounds in requirements-web.txt

The `TemplateResponse` signature drift that bit us during Session 7
wiring would have been prevented by an upper bound on `fastapi`.
Added `<1.0` upper bounds on all four web deps with a comment
explaining the rationale.

## Audit findings — investigated, no fix needed

These were checked and confirmed safe. Recording them so a future
audit doesn't waste time re-investigating:

- **All five `except Exception` blocks in `web/app.py`** log or render
  errors visibly — no silent swallows.
- **Templates have zero `|safe` or `Markup` bypasses** — Jinja
  autoescape is in effect everywhere. XSS test in
  `test_web_diff_html.py::test_xss_in_card_names_escaped` confirms
  HTML in card names gets escaped.
- **`BuildRegistry` memory growth** is intentional (single-user local
  tool, restart clears state). Documented in module docstring.
- **One `except RuntimeError: pass`** in `state.py:post_event` is
  intentional (loop-already-closed is the expected case when an SSE
  consumer goes away mid-build) and documented inline.
- **`_run_build`'s `finally` block** always posts `__done__` even on
  exception, so the SSE stream always terminates cleanly. No "stream
  hangs forever" failure mode.

## Test count

- v0.7.0: 437 tests
- v0.7.1: **444 tests** (+7)
  - +2 `_event_generator` unit tests
  - +5 form clamping tests

## Files modified

- `mtg_deck_builder/__init__.py` — version bump
- `mtg_deck_builder/web/app.py` — SSE generator hardening, dead
  imports removed
- `mtg_deck_builder/web/state.py` — dead `traceback` import removed
- `mtg_deck_builder/web/forms.py` — `_clamp` helper, GA param clamping
- `mtg_deck_builder/web/templates/build_status.html` — fix log
  duplication
- `mtg_deck_builder/tests/test_web_routes.py` — 2 new generator tests
- `mtg_deck_builder/tests/test_web_forms.py` — 5 new clamping tests
- `requirements-web.txt` — pinned upper bounds

## Total LOC change

Net positive (the new tests outweigh the removed dead code), but
that's mostly tests. Production code is roughly net-zero — fixes
added some lines, dead imports removed some.
