# Changes in v0.5 (Session 5)

v0.5 brings **Scryfall Tagger integration** and **deck diffing** — two
features that turn the v0.4 iteration loop into something genuinely
introspectable. You can now score decks by what the *art* depicts, not
just what creature types are involved, and you can see exactly what
changed when iterating.

Three new modules, all opt-in:
1. **`scryfall_tags`** — query cards by Scryfall Tagger tags
2. **`flavor_tags`** — score decks by art-tag alignment
3. **`deck_diff`** — compare two deck snapshots

---

## 1. Scryfall Tagger integration

The Scryfall community runs a [Tagger project](https://tagger.scryfall.com/)
that crowdsources two kinds of tags on every Magic card:

- **Art tags** (`art:`, `atag:`, `arttag:`) — what's depicted in the
  illustration. These describe physical/visual content: `art:mammoth`
  finds every card with a mammoth drawn on it, regardless of whether
  the card is a Mammoth, a creature, or even a creature at all. A land
  with a mammoth in the background art will match.
- **Oracle tags** (`function:`, `otag:`, `oracletag:`) — what the card
  does mechanically. Examples: `function:removal`, `function:ramp`,
  `function:counterspell-creature`. Community-curated, so often more
  precise than text-based regex matching.

Scryfall imports the Tagger data daily. We query it through the regular
`/cards/search` endpoint — no separate Tagger API needed.

### `ScryfallTagClient`

```python
from mtg_deck_builder import ScryfallTagClient

client = ScryfallTagClient(cache_dir="./tags_cache")

# Art tag — cards depicting mammoths
mammoth_cards = client.get_cards_with_art_tag("mammoth")

# Oracle tag — cards that function as removal
removal_cards = client.get_cards_with_oracle_tag("removal")

# Color-identity filter (for Lathiel — W/G only)
wg_ramp = client.get_cards_with_oracle_tag("ramp", color_identity="WG")
```

Returns lists of card *names* (not full JSON — if you need images, use
the v0.4 `ScryfallCardSource`). Cached per `(kind, tag, color_identity)`
on disk for 7 days.

Design notes:
- **Pagination cap**: max 3 pages per query (~525 cards). For most tags
  this captures everything. Configurable via `max_pages`.
- **404 handling**: Scryfall returns `object: "error"` JSON for no-match
  queries. We detect this and return `[]` rather than crashing.
- **Always returns a list**: any failure (network down, malformed JSON,
  unknown tag) yields `[]`. Never raises.
- **User-Agent and Accept** headers set per Scryfall's API guidelines.
- **Rate limiting**: 100ms minimum between requests (~10 req/s max).

---

## 2. Art-tag-based flavor scoring

The v0.4 `flavor` dimension only looked at tribal subtype matches. v0.5
adds a much richer signal: **art-tag alignment**.

```python
config = BuildConfig(
    commander_name="Lathiel, the Bounteous Dawn",
    flavor_art_tags=["mammoth", "forest", "deer"],  # "wilderness" theme
    score_weights={..., "flavor": 0.20},  # opt in to flavor scoring
)
```

CLI:
```bash
--flavor-tag mammoth --flavor-tag forest --flavor-tag deer
```

The evaluator pre-fetches the union of all matching cards once, then
scores each deck by what fraction of its cards are in that set. Mapping:

| Match ratio | Score |
|-------------|-------|
| 0%          | 40    |
| 10%         | 60    |
| 25%         | 80    |
| 50%+        | 95    |

(Piecewise linear in between. 100% match is rare in practice — basic
lands and utility artifacts usually don't have specific art tags.)

### Combining with v0.4 tribal flavor

When both signals are active (commander has a creature subtype AND
`flavor_art_tags` is set), the evaluator takes the **MAX** of the two
scores, not the average.

Rationale: if you're running a strong-tribal deck, art tags shouldn't
penalize you. If you're running a thematic-art deck under a non-tribal
commander, tribal shouldn't penalize you. "Best-aligned dimension wins"
felt more honest than dilution. A deck that excels in both gets the
benefit of either; a deck that's strong in one and weak in the other
isn't unfairly averaged down.

This also keeps tribal decks behaving exactly as they did in v0.4 if
you don't set `flavor_art_tags`.

### Color-identity filter

The scorer auto-passes the commander's color identity through to the
tag client, so a Lathiel (W/G) build only counts art-tag matches that
are W/G-legal. No more credit for `art:mammoth` on a black-bordered
Mammoth that you couldn't actually run.

### Failure modes

- Misspelled tag → empty contribution, info-level log message
- Tag client unreachable → empty contribution, build continues
- Scorer raises an exception → falls back to v0.4 tribal scoring (the
  evaluator catches it explicitly so flavor scoring can never break a
  build)

---

## 3. Diff mode

Compare two deck snapshots and see exactly what changed.

CLI:
```bash
mtg-cli --csv cards.csv diff lathiel_v1.json lathiel_v2.json
```

Output:
```
Commander: Lathiel, the Bounteous Dawn
Kept: 87   Added: 12   Removed: 12

=== Added ===
  [ramp      ] Sol Ring, Arcane Signet
  [removal   ] Swords to Plowshares, Path to Exile
  [land      ] Plains, Plains, Plains

=== Removed ===
  [ramp      ] Birds of Paradise
  [removal   ] Beast Within, Generous Gift
  [land      ] Forest, Forest
```

Python API:
```python
from mtg_deck_builder import diff_decks, format_diff, WarmStartDeck

a = WarmStartDeck.from_json_file("lathiel_v1.json")
b = WarmStartDeck.from_json_file("lathiel_v2.json")
result = diff_decks(a, b, card_db=db)  # card_db optional, enables role grouping

print(format_diff(result, show_kept=False, max_per_group=20))
print(f"Kept {result.kept_count}, added {result.added_count}, removed {result.removed_count}")
```

### Design notes

**Multiset semantics for basic lands.** Going from 15 Forests to 17
Forests shows as "2 Forests added," not "Forest is in both, no change."
Implemented via `Counter` arithmetic.

**Flexible inputs.** `diff_decks` accepts `WarmStartDeck`, `Deck`, dict,
or raw list of names. Useful for diffing live build results against
saved snapshots without converting first.

**Role grouping is optional.** Pass a `CardDatabase` to bucket the
output by role (ramp/draw/removal/wipe/land/other). Without it,
everything lands in a flat "added"/"removed" list. The CLI auto-loads
the DB if `--csv` was provided.

**Commander change detection.** If both inputs include commander info
and they differ, the diff flags it explicitly: "Commander changed:
Lathiel -> Karlov".

**Mobile-friendly truncation.** `format_diff(max_per_group=N)` caps each
role bucket to N items with "(... and K more)" — important for the
phone-only workflow.

---

## New `BuildConfig` fields (all default to off/safe)

| Field | Default | Purpose |
|-------|---------|---------|
| `flavor_art_tags` | `[]` | Art tags for flavor scoring |
| `use_oracle_tag_validation` | `False` | Reserved for Session 6 |
| `tags_cache_dir` | `None` | Tag query cache location |
| `tags_offline` | `False` | Cache-only mode |

## New CLI flags

```
--flavor-tag TAG               (repeatable)
--tags-cache-dir DIR
--tags-offline
```

## New CLI subcommand

```
mtg-cli diff FROM.json TO.json [--csv cards.csv] [--show-kept] [--max-per-group N]
```

## New modules

- `scryfall_tags.py` — `ScryfallTagClient` (tag search + cache)
- `flavor_tags.py` — `FlavorTagScorer` (art-tag deck scoring)
- `deck_diff.py` — `diff_decks`, `format_diff`, `DiffResult`

## New exports from the package

- `ScryfallTagClient`
- `FlavorTagScorer`
- `diff_decks`, `format_diff`, `DiffResult`

## Test count

- v0.4: 271 tests
- v0.5: **327 tests** (+56)
  - +18 ScryfallTagClient (offline, pagination, color filter, cache, 404, malformed JSON)
  - +16 FlavorTagScorer (factory, pre-fetching, scoring, evaluator integration)
  - +22 deck_diff (input normalization, multiset semantics, role grouping, formatting)

## Architecture notes

- **Lazy construction.** `DeckBuilder.tag_client` and
  `DeckBuilder.flavor_tag_scorer` are lazy properties — they only build
  when actually needed. `tag_client` is shared between flavor scoring
  and the (future) oracle-tag validation, so we don't double up on
  caches or rate limits.
- **Two separate Scryfall caches.** The `ScryfallCardSource` cache (per-card
  full JSON) and the `ScryfallTagClient` cache (per-tag-query lists of
  names) are kept separate because the data shapes differ. They could
  share a directory but use different filename schemas, so collisions
  are impossible.
- **Defense in depth on the evaluator.** `_score_flavor` catches any
  exception from the art-tag scorer and falls back to tribal scoring.
  The art-tag scorer itself never raises — it returns 50 (neutral) for
  any degenerate case (no matches in universe, empty deck, etc.). So
  there are two independent layers preventing flavor scoring from
  breaking a build.

## Known caveats

- **Tag accuracy depends on community coverage.** Newer cards may not
  be tagged yet. Niche tags may have small or biased samples. Treat the
  art-tag flavor signal as a hint, not gospel.
- **Oracle-tag role validation is configured but not consumed.** The
  `use_oracle_tag_validation` config field exists for Session 6 wiring;
  setting it today is a no-op.
- **First flavor-tag run is slow.** Pre-fetching all matching cards for
  N tags = N HTTP round trips (each potentially paginated). Subsequent
  runs in the same 7-day window hit cache. Pre-warm with a throwaway
  small build if you care about cold-start times.
- **Color-identity filtering is at fetch time, not score time.** If you
  change commander mid-iteration without clearing the cache, you may
  get stale matches. Clear `tags_cache_dir` or use a different one for
  different commanders.

## Session 6+ wishlist

(carrying forward + adding from this session's work)

- **Oracle-tag-augmented role detection** (`function:`/`otag:` queries
  to validate our regex role matchers as a second opinion). Was
  originally Session 5 plan; pulled to Session 6 because the user value
  is diagnostic rather than score-affecting.
- **Bulk Scryfall data downloader** — pull the daily ~130MB JSON dump
  once instead of per-card API calls. Massive speedup for first-time
  use. Would also give us local oracle-tag and art-tag data without
  per-tag fetches.
- **Diff mode HTML output** — render `DiffResult` as a styled card-by-card
  comparison page, not just terminal text.
- **Fully offline HTML reports** with base64-embedded card images.
- **Local web UI** (Flask form → DeckBuilder → existing HTML report)
  if/when we want to commit to a GUI. Iteration would still happen via
  CLI flags; the UI would be a discovery tool for the flag space.
