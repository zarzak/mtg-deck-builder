# Changes in v0.6 (Session 6)

Two features, both opt-in, both backend-focused:

1. **Scryfall bulk data** — one 130MB download instead of hundreds of
   per-card API calls
2. **Oracle-tag role validation** — use community tags to audit our
   regex-based role classifications

Plus one bug fix uncovered while wiring these together.

---

## 1. Scryfall bulk data

Scryfall publishes daily JSON dumps of their card database as "bulk
data". Every card's full JSON — images, prices, artists, oracle text,
everything — in one file. One HTTP call gets us a month's worth of
images that would've otherwise been hundreds of individual fetches.

Two new classes:

- **`ScryfallBulkFetcher`** — downloads and caches bulk files, with
  metadata-based freshness checks. Atomic writes so a partial download
  can't corrupt the cache. Graceful fallback to stale cache when
  metadata is unreachable.

- **`BulkCardSource`** — drop-in replacement for the v0.4
  `ScryfallCardSource`. Same interface (`get_card_data`, `get_image_url`,
  `get_artist`, `get_scryfall_uri`). Builds an in-memory name→JSON
  index at construction; all subsequent lookups are O(1) with zero
  network calls.

### CLI usage

```bash
# First use — downloads ~130MB, then runs normally
mtg-cli --csv cards.csv build "Lathiel" --bulk-source --images --report out.html

# Subsequent runs — instant, no network unless Scryfall has a newer version
mtg-cli --csv cards.csv build "Lathiel" --bulk-source --images --report out.html

# Pure-offline mode — use existing cache only
mtg-cli --csv cards.csv build "Lathiel" --bulk-source --bulk-offline
```

### Flags

```
--bulk-source            Use bulk downloader (recommended for serious use)
--bulk-cache-dir DIR     Where to cache the JSON (default: ./scryfall_bulk)
--bulk-type TYPE         oracle_cards | default_cards | unique_artwork
--bulk-offline           Don't download; use cache only
```

### Design notes

- **`oracle_cards` is the default** (~130MB, one entry per unique
  Oracle name). For deck-building use cases, this is the right tradeoff
  vs. `default_cards` (~300MB, one per printing) or `all_cards` (~2GB).
- **Freshness via `updated_at`**: we fetch the bulk metadata endpoint
  first, compare ISO timestamps, and only re-download when Scryfall has
  a newer version. Usually one cheap metadata call per session.
- **Double-faced card aliasing**: if a card has both "Delver of Secrets
  // Insectile Aberration" (full name) and "Delver of Secrets" (front
  face), both names find the same card object.
- **Memory**: ~300-500MB for the full bulk file held in a dict. Paid
  once at construction; every lookup after that is free. If that's too
  much, fall back to per-card `ScryfallCardSource`.

### Interface compatibility

`BulkCardSource` duck-types `ScryfallCardSource`, so HTML report
generation works with either. `DeckBuilder.card_source` property picks
bulk when `use_bulk_source=True`, per-card when `use_images=True`, and
user-injected sources beat both.

---

## 2. Oracle-tag role validation

Our role classification (`ramp`/`draw`/`removal`/`wipe`/`land`) uses
regex patterns on oracle text. The Scryfall Tagger project crowdsources
the same classification by hand. This new feature cross-checks the two
and reports disagreements.

**Diagnostic only** — doesn't change scores or deck contents. Just
tells you where your deck might be classified differently than
community consensus.

### CLI usage

```bash
mtg-cli --csv cards.csv build "Lathiel" \
    --validate-roles --tags-offline  # uses cached tags
```

Output:
```
------------------------------------------------------------
ROLE VALIDATION (regex vs. community oracle tags)
------------------------------------------------------------
Role validation: 99 cards, 4 roles
Total disagreements: 5

=== ramp / missed (tags flagged, regex did not) ===
  Some Card, Another Card

=== removal / extra (regex flagged, tags did not) ===
  Odd Card
```

### Kinds of disagreement

- **Missed**: oracle tags say the card fills this role, but our regex
  didn't flag it. Candidates for improving our role patterns.
- **Extra**: our regex flagged the card, but community tags didn't.
  Possibly false positives (e.g., a card with "destroy" in cost text
  being flagged as removal).

### Design notes

- **Conservative tag mapping**: `ROLE_TO_ORACLE_TAGS` maps our role
  names to community tags that clearly express the same concept. For
  "ramp" we check `mana-ramp`, `ramp-artifact`, `ramp-creature` —
  union wins. Tags that might or might not match a role are left out:
  false negatives on validation are less bad than false positives.
- **Color-identity scoped**: validation queries use `id<=<commander-colors>`
  so we don't report on cards that couldn't legally be in this deck.
- **Skipped vs. reported**: if a role's tag queries all fail (network
  down, no cache), the role shows as "skipped" rather than being
  treated as "all cards missed." Never generates a false disagreement.
- **Never crashes a build**: if the tag client raises, validation is
  skipped with a log line; the deck build still completes normally.
- **Always a union**: multiple community tags may exist for one
  concept. We union them — any matching tag counts.

### CLI flag

```
--validate-roles         After the build, print a role-validation report
```

Combine with the Session 5 tag flags:

```bash
--tags-cache-dir DIR     Where to cache tag query results
--tags-offline           Don't hit the network for tag data
```

---

## 3. Bug fix: color-identity format

Uncovered while wiring oracle-tag validation end-to-end. The
`ScryfallTagClient` built its `id<=` Scryfall query by lowercasing the
input color_identity. This worked for inputs like `"WG"` but not `"W,G"`
(our CSV format — comma-separated) or `"w g"` (possible in other
formats). The query string ended up with `id<=w,g` which Scryfall
doesn't recognize.

Fixed with a new `_normalize_color_identity()` helper that extracts
only WUBRG letters regardless of separators or casing, then sorts for
canonical form. `"WG"`, `"W,G"`, `"gw"`, `"G W"` all now produce
the canonical `"gw"` — same cache key, same Scryfall query.

New test `test_color_identity_format_normalization` in
`test_scryfall_tags.py` exercises eight input variants and confirms they
all hit the same cache entry.

---

## New `BuildConfig` fields (all default to off/safe)

| Field | Default | Purpose |
|-------|---------|---------|
| `use_bulk_source` | `False` | Enable bulk downloader |
| `bulk_cache_dir` | `None` | Where to cache bulk JSON |
| `bulk_type` | `"oracle_cards"` | Which bulk file |
| `bulk_offline` | `False` | Cache-only mode |
| `validate_roles_after_build` | `False` | Run role-tag audit after build |

## New CLI flags

```
--bulk-source                       Use bulk downloader
--bulk-cache-dir DIR
--bulk-type {oracle_cards,default_cards,unique_artwork}
--bulk-offline
--validate-roles                    Post-build role audit
```

## New modules

- `scryfall_bulk.py` — `ScryfallBulkFetcher`, `BulkCardSource`
- `oracle_validation.py` — `validate_roles`, `format_role_report`,
  `ValidationReport`, `RoleDisagreement`

## New exports from the package

- `ScryfallBulkFetcher`, `BulkCardSource`
- `validate_roles`, `format_role_report`, `ValidationReport`,
  `RoleDisagreement`

## `OptimizationResult` additions

- `role_validation_report: Optional[ValidationReport]` — populated
  when `validate_roles_after_build=True`

## Test count

- v0.5.1: 328 tests
- v0.6.0: **378 tests** (+50)
  - +27 scryfall_bulk (fetcher + source with stubbed HTTP, DFC aliasing)
  - +16 oracle_validation (role mapping, disagreement kinds, skipped
    roles, raising clients)
  - +6 Session 6 integration (bulk through full build, validation flag)
  - +1 color-identity normalization regression test

## Architecture decisions

- **`BulkCardSource` is an interface duck, not a subclass.** It happens
  to match `ScryfallCardSource`'s method signatures but doesn't inherit.
  Consumers don't care because Python doesn't care; the HTML report
  code just calls `.get_image_url()` and it works.
- **`validate_roles` is a standalone function, not a method on
  `DeckEvaluator`.** Validation is diagnostic, not scoring — keeping it
  separate means it can't accidentally contaminate the fitness pipeline.
- **Tag client is shared** between flavor scoring (v0.5) and role
  validation (v0.6). Same on-disk cache, one rate limit pool.
- **Role validation runs after LLM review** in the build pipeline. This
  way the report is always generated from the final deck (not a stale
  mid-optimization version), and the LLM has no opportunity to see
  validation output and try to "fix" it.

## Known caveats

- **Bulk file is ~130MB.** First-time users on slow connections will
  wait. We show a progress-ish log message with the size estimate, but
  there's no streaming progress bar. Good Session 7+ work if GUI.
- **Oracle-tag validation requires tag cache.** If the user sets
  `--validate-roles` but no tags have been fetched yet AND they're
  offline, every role will show as "skipped." Not a bug, just honest
  about what we can't validate without data.
- **Community tags can be sparse.** Newer cards may have few or no
  tags. Validation is more meaningful for established formats than for
  cards from the last set.
- **Regex patterns haven't been updated in response to validation
  findings.** The audit is there to help you identify classification
  issues — fixing the patterns is a manual judgment call.

## Session 7+ wishlist

- **GUI** (the long-planned Session 7) — local Flask form that maps to
  the CLI flag space, starting minimal
- **Regex pattern improvements** driven by the `--validate-roles`
  report on real decks
- **Diff mode HTML output** (carried over from Session 5)
- **Offline HTML reports** with base64-embedded images from
  `BulkCardSource`
