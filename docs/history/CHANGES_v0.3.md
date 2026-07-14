# Changes in v0.3 (Session 3)

v0.3 adds four opt-in integrations. All default to OFF so upgrading from v0.2
changes nothing unless you explicitly enable a feature.

## New modules

### `edhrec_client.py` — EDHREC data integration

HTTP client for fetching community deck data from EDHREC. Used to supply real
baseline-power and commander-specific synergy scores, replacing heuristic
estimates for cards EDHREC has data on.

Key design decision: we use EDHREC's **synergy** field (commander-specific
delta over baseline), not raw inclusion rates. Inclusion rates are biased by
precons — Sol Ring appears in ~98% of decks even when it's not particularly
synergistic with the commander. The synergy field is designed to neutralize
that bias.

Features:
- Disk caching with configurable TTL (default 1 week)
- Rate limiting (200ms minimum between requests)
- Offline mode for testing
- Graceful degradation: any network failure returns `None`, build continues
- No hard dependency on `requests` — falls back to `urllib` from stdlib
- 26 unit tests (all offline)

Enable via: `BuildConfig(use_edhrec=True, edhrec_cache_dir="./edhrec_cache")`
or CLI: `--edhrec --edhrec-cache-dir ./edhrec_cache`

### `embedding_scorer.py` — Fast semantic synergy

Cosine-similarity based synergy scorer using sentence-transformers. Roughly
1000× faster than LLM scoring; good as a first-pass filter before spending
tokens on an LLM review of the survivors.

Design:
- `sentence-transformers` is an **optional** dependency. `create_if_available()`
  factory returns `None` if the library isn't installed, letting callers fall
  back to LLM scoring.
- Embeds commander strategy text (from analysis) against card text; cosine
  similarity mapped to [30, 95] synergy scale.
- Default model: `all-MiniLM-L6-v2` (~25MB, fast). Configurable.
- 13 unit tests using a deterministic fake model (no ML model download needed
  for tests).

Enable via: `BuildConfig(use_embeddings=True)` + `pip install sentence-transformers`
or CLI: `--embeddings`

### `price_source.py` — Budget constraints

Protocol-based price source abstraction with multiple implementations:
- `NullPriceSource` — always returns None (testing)
- `StaticPriceSource` — dict-backed (tests, custom lists)
- `ScryfallPriceSource` — real Scryfall API with disk cache + rate limiting

Plus `filter_cards_by_budget()` helper and `deck_total_price()` summarizer.

Design note: when `exclude_unknown=False` (default), cards with no price data
are **kept** rather than dropped. Better to allow an unknown-priced card than
to stall the build because Scryfall had a hiccup.

20 unit tests (offline).

Enable via: `BuildConfig(budget_max_per_card=10.0)`
or CLI: `--budget 10`

### `island_optimizer.py` — Parallel GA

Real multiprocessing island-model implementation, replacing the v0.2 stub.
Multiple independent DeckOptimizer instances run in separate processes with
different seeds, exploring different parts of the solution space.

Features:
- Uses `multiprocessing.Pool` with `spawn` context (Windows-safe)
- Different prime-spaced seeds per island for diverse exploration
- Graceful fallback to sequential execution if multiprocessing fails
  (pickle issues, Windows spawn problems, etc.)
- Per-island results returned; best across islands is the final result
- 8 unit tests using sequential mode for speed

Enable via: `BuildConfig(use_island_model=True, num_islands=4)`
or CLI: `--islands 4`

**Startup cost warning:** Multiprocessing spawn takes 1-2s on most systems.
Only worth it for runs with ≥100 generations and population ≥30.

## Pipeline integration

`DeckBuilder.build()` now has two new phases:

1. `_phase_fetch_edhrec()` — runs after commander analysis if `use_edhrec=True`.
   Never blocks on network errors.
2. `_phase_budget_filter()` — runs after pool generation if
   `budget_max_per_card` is set. Filters before the (expensive) LLM filtering
   phase so we don't waste tokens on cards we can't afford.

`_phase_synergy_scoring()` now does **layered** scoring:
1. Start with heuristic baseline (all cards)
2. Overlay embedding scores (if enabled)
3. Overlay EDHREC data (if available) — wins when present because it's
   community-vetted
4. Fill remaining gaps with LLM batch scoring

`_phase_optimization()` branches to the island model when `use_island_model=True`.

## New `BuildConfig` fields (all default to off/safe)

| Field | Default | Purpose |
|-------|---------|---------|
| `use_edhrec` | `False` | Enable EDHREC data layer |
| `edhrec_cache_dir` | `None` | Where to cache EDHREC JSON |
| `edhrec_offline` | `False` | Cache-only mode |
| `use_embeddings` | `False` | Enable embedding-based synergy |
| `embedding_model` | `"all-MiniLM-L6-v2"` | sentence-transformers model |
| `budget_max_per_card` | `None` | USD ceiling per card |
| `budget_exclude_unknown` | `False` | Drop cards with no price data |
| `use_island_model` | `False` | Parallel GA |
| `num_islands` | `4` | Islands (when enabled) |
| `island_migration_interval` | `10` | Generations between migrations |

## New CLI flags

```
--edhrec              Enable EDHREC integration
--edhrec-cache-dir    Custom cache directory
--edhrec-offline      Don't hit the network
--embeddings          Enable embedding-based synergy
--embedding-model     sentence-transformers model name
--budget USD          Per-card budget ceiling
--budget-exclude-unknown   Drop cards with no price data
--islands N           Use N-island parallel GA
--island-migration-interval N    Migrations every N generations
```

## Dependency injection for testing

`DeckBuilder` constructor now accepts optional injected components:

```python
builder = DeckBuilder(
    csv_path, config,
    edhrec_client=my_client,    # custom or mock EDHRECClient
    embedding_scorer=my_scorer,  # custom or mock scorer
    price_source=my_prices,      # custom or mock price source
)
```

Useful for:
- Tests: inject mocks instead of hitting real services
- Air-gapped environments: inject a `StaticPriceSource` from a pre-loaded file
- Custom data sources: swap EDHREC for a local database, etc.

## Test count

- v0.2: 127 tests
- v0.3: **203 tests** (+76 net)
  - +26 EDHREC
  - +13 embeddings
  - +20 price sources
  - +8 island optimizer
  - +9 integration tests for Session 3 wiring

## Known quality caveats

- The included `test_cards.csv` is a **toy dataset** (84 cards) sufficient to
  verify plumbing but not to produce a good real deck. When you run against
  your full mtgjson-derived CSV, results will be meaningfully different.
- Tests verify **correctness** (valid 99-card outputs, no crashes, graceful
  fallbacks) not **quality** (is this a good deck?). Deck quality needs the
  LLM review pass and eyeballing the HTML report on real data.
- `FakeModel` in embedding tests uses a substring check, not real embeddings.
  Real `sentence-transformers` will pick up much subtler semantic signal than
  "does the card text contain a literal synergy keyword".

## Not done yet (ideas for Session 4+)

- Scryfall image integration (rendering cards in HTML report)
- User-tunable scoring weight sliders (preset-or-custom)
- Iterative refinement: card locks, bans, role overrides, warm-start from
  a prior deck
- A `flavor` dimension separate from synergy (for tribal/theme decks)
