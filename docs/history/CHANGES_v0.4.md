# Changes in v0.4 (Session 4)

v0.4 focuses on **iterating on a deck you're already building** rather than
cold-starting from scratch each time. All new features are opt-in — upgrading
from v0.3 changes nothing unless you enable a flag.

Three main areas:
1. **Iterative refinement** — locks, bans, role overrides, warm-start from a prior deck
2. **Tunable weights** — CLI flags for customizing scoring priorities, plus a new `flavor` dimension
3. **Scryfall card images** — optional art/thumbnails in the HTML report

---

## 1. Iterative Refinement

The v0.3 workflow was "build a deck from scratch." v0.4 supports the natural
follow-up: "this deck was good but needs more removal" or "keep my Sol Ring
but swap out the expensive stuff."

### Locked cards

Cards that MUST appear in every candidate deck. Injected into every initial
individual and preserved through crossover/mutation.

```python
config = BuildConfig(
    commander_name="Lathiel, the Bounteous Dawn",
    locked_cards=["Sol Ring", "Soul Warden", "Archangel of Thune"],
)
```

CLI:
```bash
--lock "Sol Ring" --lock "Soul Warden"
```

Design details:
- Basic lands can be locked multiple times (`--lock Forest --lock Forest --lock Forest`)
- Non-basic duplicates dedupe automatically
- Missing cards log a warning and are skipped (not fatal)
- Cards that violate color identity are logged and skipped
- Too many locks (>99) are truncated with a warning

### Banned cards

Cards that must NEVER appear. Filtered out of the candidate pool entirely.

```python
config = BuildConfig(banned_cards=["Sol Ring", "Mana Crypt"])
```

CLI:
```bash
--ban "Sol Ring" --ban "Mana Crypt"
```

**Ban vs. lock conflict:** if you both ban and lock the same card, the ban
wins and the lock is silently dropped. Rationale: bans are the stricter
constraint. A user who changed their mind shouldn't get a crash.

### Role target overrides

Change what counts as "enough ramp" or "enough removal" for your deck.

```python
config = BuildConfig(role_target_overrides={"removal": (10, 14)})
```

CLI:
```bash
--role-target removal=10,14 --role-target ramp=8,12
```

Overrides **merge** with defaults — roles you don't specify keep their
default targets. Reversed bounds (`14,10`) are auto-fixed. Garbage input
is silently ignored.

### Warm-start from a prior deck

Save a deck, then use it as the starting point for a new run with
different parameters.

First run:
```bash
python -m mtg_deck_builder.cli --csv cards.csv build "Lathiel" \
    --seed 42 --save-deck lathiel_v1.json
```

Refine:
```bash
python -m mtg_deck_builder.cli --csv cards.csv build "Lathiel" \
    --warm-start lathiel_v1.json \
    --warm-start-copies 2 \
    --lock "Soul Warden" \
    --role-target removal=12,15 \
    --save-deck lathiel_v2.json
```

Design details:
- Warm-start files are minimal: `{commander_name, card_names, final_score}`.
  Not a full state dump — portable and human-readable.
- `warm_start_copies` controls how many copies of the prior deck seed the
  initial population. Higher = stays closer to original. Default 1 gives the
  GA room to explore from that starting point.
- Cards from the warm-start that aren't in the current pool are silently
  dropped (the GA fills those slots with fresh picks).
- **Locks trump warm-start:** if you warm-start from a deck without Sol Ring
  and then `--lock "Sol Ring"`, every copy of the warm-start gets Sol Ring
  inserted.

---

## 2. Tunable Weights + Flavor Dimension

### `flavor` dimension

New scoring dimension separate from `synergy`. Where synergy is mechanical
(does this card's text interact with the commander's keywords), flavor is
thematic (does this card fit the deck's identity).

Current implementation: scores tribal coherence — creatures sharing a
subtype with the commander boost flavor. A Unicorn commander with 50%+
Unicorn creatures scores ~85; 0% scores ~40; scaling is linear between.
Non-creatures don't count (they don't drag down tribal ratio), and
commanders without creature subtypes (e.g. Planeswalker commanders) return
neutral 50.

For backwards compat, `flavor` defaults to weight 0 in `score_weights`.
The dimension only contributes when explicitly weighted — see presets below.

Future: art-tag-based flavor (Scryfall Tagger `art:` / `atag:` queries) is
Session 5 material.

### CLI weight controls

Override individual weights:
```bash
--weight synergy=0.6 --weight creativity=0.05
```

Or pick a preset:
```bash
--preset flavor      # theme/tribal (flavor=0.20, synergy=0.30, creativity=0.15)
--preset power       # cEDH-leaning (power_level=0.35, synergy=0.25)
--preset budget      # value-focused (creativity=0.25, power_level=0.10)
--preset balanced    # the v0.2 defaults (no flavor)
```

Presets and `--weight` flags compose: preset first, then individual
overrides. Weights auto-normalize to sum to 1.0 so you don't have to.

Example: start from `flavor` preset but bump synergy higher:
```bash
--preset flavor --weight synergy=0.5
```

### Design notes on weights

- If `--preset` and `--weight` are both omitted, the default `score_weights`
  from v0.3 is used unchanged (no normalization pass).
- Unknown dimension names in `--weight` flags log a warning and are
  ignored (so a typo doesn't crash the build).
- Negative weights are rejected by the argparse validator.
- When weights are set, the CLI prints what it's using:
  `Weights (flavor): flavor=0.20, synergy=0.30, ...`

---

## 3. Scryfall Card Images

Optional: embed card art and thumbnails in the HTML report. When disabled
(default), the report renders exactly as in v0.3.

### Setup

```bash
--images                    # enable
--images-cache-dir ./cache  # where to store JSON cache (default: ./scryfall_cache)
--images-offline            # cache-only, don't hit Scryfall
```

First real run populates the cache. Subsequent runs (within 30 days) are
offline-fast. Cache is per-card JSON (~2KB each), so a 99-card deck is
under 200KB.

### What appears in the HTML report

When `--images` is set:
- **Commander art** at the top (Scryfall's `art_crop` version, no frame)
  with artist credit
- **Deck art gallery** section: grid of up to 24 card arts (prioritizing
  creatures)
- **Thumbnail column** in the per-card telemetry table (60px wide small
  images)

All `<img>` tags use `loading="lazy"` so mobile users don't download
every image upfront.

### Design notes on images

- New module `scryfall_cards.py` with `ScryfallCardSource` class; distinct
  from `ScryfallPriceSource` because the cache schemas differ (prices are
  just a number, cards are the full JSON)
- Scryfall usage compliance: User-Agent header set, Accept header included,
  rate limit ≥100ms between requests (~10 req/s max), artist credit
  displayed wherever we show `art_crop`
- Double-faced cards: returns the front-face image
- Graceful degradation: any card-level failure yields no image for that
  card, but the report still renders. Network failures while rendering
  are impossible (all images are `<img src=url>` — rendered by the
  browser, not us)
- TTL 30 days on disk cache — card data and image URLs don't change

### Dependency injection

You can inject a custom `card_source` (or a mock for testing):

```python
from mtg_deck_builder import DeckBuilder, ScryfallCardSource

# Use your own
my_source = ScryfallCardSource(cache_dir="/shared/cache", offline=True)
builder = DeckBuilder(csv_path, config, card_source=my_source)
```

Any object with `get_image_url(name, size)` and `get_artist(name)` methods
works — doesn't have to be `ScryfallCardSource`.

---

## New `BuildConfig` fields (all default to off/safe)

| Field | Default | Purpose |
|-------|---------|---------|
| `locked_cards` | `[]` | Cards that MUST appear |
| `banned_cards` | `[]` | Cards that must NEVER appear |
| `role_target_overrides` | `{}` | Per-role min/max overrides |
| `warm_start_path` | `None` | JSON file to seed population from |
| `warm_start_copies` | `1` | How many population copies of warm-start |
| `use_images` | `False` | Enable Scryfall image integration |
| `images_cache_dir` | `None` | Image JSON cache location |
| `images_offline` | `False` | Cache-only mode |

Plus `DeckScores.flavor` dimension (0-100), with default weight 0 in
`score_weights` for backwards compat.

## New CLI flags

```
--lock CARD                       (repeatable)
--ban CARD                        (repeatable)
--role-target ROLE=MIN,MAX        (repeatable)
--warm-start JSON_FILE
--warm-start-copies N
--save-deck JSON_FILE

--weight DIM=VAL                  (repeatable)
--preset {flavor,power,budget,balanced}

--images
--images-cache-dir DIR
--images-offline
```

## New modules

- `scryfall_cards.py` — `ScryfallCardSource` (full card JSON cache + images)

## New classes / helpers

- `WarmStartDeck` (models.py) — serializable deck snapshot
- `OptimizationResult.to_warm_start()` — convert result to snapshot
- `OptimizationResult.to_json_file(path)` — save warm-start to disk
- `WarmStartDeck.from_json_file(path)` — load warm-start
- `BuildConfig.get_effective_role_targets()` — merge overrides with defaults
- `DeckBuilder.card_source` property — lazy-construct `ScryfallCardSource`
- `_parse_role_target`, `_parse_weight`, `_build_weight_dict` in `cli.py`

## Test count

- v0.3: 203 tests
- v0.4: **271 tests** (+68)
  - +19 refinement (locks, bans, role overrides, warm-start)
  - +20 flavor + weights (scoring, CLI helpers)
  - +18 Scryfall card source (offline, caching, DFCs, metadata)
  - +11 HTML-with-images integration

## Known caveats

- The `flavor` dimension currently rewards *tribal* coherence only. Art-tag
  flavor (via the Scryfall Tagger's `art:`/`atag:` queries) is a richer
  signal that we haven't integrated yet. Session 5.
- HTML reports with images embed Scryfall CDN URLs rather than inlined
  base64. So: fast to generate, small file size, but needs internet on
  the *viewing* side to actually show the images. If you want a true
  offline-viewable HTML report, that's a future-session feature.
- The image TTL is 30 days. If Scryfall rotates CDN URLs within that
  window, cached links may break. Real-world this happens rarely.

## Session 5+ ideas (recorded here so we don't forget)

- **Art-tag-based flavor scoring** via Scryfall's `art:`/`atag:`/`arttag:`
  operators (user pointed out this makes real flavor scoring possible
  beyond just tribal matching)
- **Oracle-tag-augmented role detection** via `function:`/`otag:` — could
  serve as a high-quality second opinion on top of our regex role matchers
- **Bulk Scryfall data** for images/tags instead of per-card requests
  (~130MB daily dump, but gives everything at once)
- **True offline HTML reports** with base64-embedded images
- **Diff mode** — show what changed between a warm-start and the new result
