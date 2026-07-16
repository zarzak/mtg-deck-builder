# MTG Deck Builder — Hybrid LLM + Genetic Algorithm

[![tests](https://github.com/zarzak/mtg-deck-builder/actions/workflows/tests.yml/badge.svg)](https://github.com/zarzak/mtg-deck-builder/actions/workflows/tests.yml)

An automated **Magic: The Gathering Commander (EDH)** deck builder that pairs
large-language-model card evaluation with a genetic algorithm to assemble and
optimize 99-card decks around any commander.

The core idea: an LLM is good at *judgment* ("is this card good for this
commander's plan?") but bad at *combinatorial search* ("which 99 of these
2,500 cards form the best deck?"), while a genetic algorithm is the reverse.
This project uses each for what it's good at — the LLM scores cards, detects
combos, and reviews the assembled deck; the GA searches the space of legal
99-card decks against a multi-objective fitness function built from those
scores.

```bash
# One command, from a commander name to an optimized, bracket-legal decklist:
python -m mtg_deck_builder.cli --csv cards.csv build "Jodah, the Unifier" \
    --bracket 4 --combos llm --card-power-mode llm \
    --report jodah.html --output jodah.txt
```

> **Status:** personal project / portfolio piece. 807 passing tests; CI runs
> on Python 3.11–3.13. Builds call the Anthropic API and cost roughly
> **$2–6** each depending on the commander's colors.

**→ See a real build:** [`examples/`](./examples) has a complete decklist and
scoring report the tool produced for a cEDH stax commander.

---

## What it does

- **Understands the commander.** An LLM analysis pass derives the strategy,
  key mechanics, synergy keywords, and structural predicates that drive
  everything downstream.
- **Finds the right candidate pool** through layered *recall*: EDHREC
  community data, semantic embedding similarity, mechanic-pattern matching,
  and attribute predicates (for commanders whose payoff is a card *property*
  text can't see, like "vanilla creatures matter").
- **Scores cards on two axes** — commander-specific **synergy** and
  commander-independent **card power** — via calibrated LLM rubrics, globally
  cached so repeat runs are cheap and deterministic.
- **Detects combos & engines** from four sources: the EDHREC / Commander
  Spellbook human-verified database, LLM pool/knowledge/deepening passes, a
  rules-verification pass, and an accumulating per-commander memory — then
  guarantees the pieces reach the optimizer.
- **Optimizes with a genetic algorithm** across seven objectives (mana curve,
  role coverage, synergy, strategy density, power level, combo assembly,
  effect-class consistency), with a fast heuristic phase followed by a full
  evaluation phase.
- **Refines the assembled deck** with an LLM pass that critiques set-level
  composition (redundancy, interaction spread, role quality) and proposes
  guarded swaps.
- **Enforces the official Commander Brackets (1–5)** — Game Changer limits,
  mass-land-denial / extra-turn / two-card-combo policy, and cEDH structural
  templates at bracket 5.
- **Ships a local web GUI** and rich HTML reports with per-card scoring
  breakdowns and pool-entry provenance.

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt          # just `anthropic` + `pytest`
export ANTHROPIC_API_KEY=sk-ant-...       # or set it in the GUI (stored encrypted)
```

### 2. Get the card database

The card data is derived from [MTGJSON](https://mtgjson.com) and is **not
committed** (it's ~12 MB and MTGJSON-licensed). Generate it in one command —
this downloads MTGJSON's `AtomicCards` and writes `cards.csv`:

```bash
python -m mtg_deck_builder.cli --csv cards.csv refresh-cards --force
```

### 3. Build a deck — CLI or GUI

```bash
# CLI:
python -m mtg_deck_builder.cli --csv cards.csv build "Kinnan, Bonder Prodigy" \
    --bracket 4 --recall-edhrec --recall-embeddings --recall-patterns \
    --card-power-mode llm --combos llm \
    --report kinnan.html --output kinnan.txt

# ...or the local web GUI (build form, deck viewer with card images,
# manual editor, database utilities):
python -m mtg_deck_builder.cli --csv cards.csv gui
# opens http://127.0.0.1:8765
```

### 4. (Optional) Pre-seed the card-power cache

`build` scores card power on demand, but you can bulk-score a color region
once so later builds are cheaper. The repo ships a pre-built cache
(`card_power_cache/`, ~32K cards); to extend or rebuild it:

```bash
python -m mtg_deck_builder.cli --csv cards.csv power-scan --dry-run   # cost estimate
python -m mtg_deck_builder.cli --csv cards.csv power-scan             # do it
```

---

## How it works — the pipeline

```
commander name
      │
      ▼
  ┌─────────────────┐   LLM: strategy, mechanics, synergy keywords,
  │ 1. Analysis     │   structural predicates, effect classes   (cached)
  └─────────────────┘
      │
      ▼
  ┌─────────────────┐   EDHREC synergy + inclusion, embedding cosine,
  │ 2. Recall       │   mechanic patterns, attribute predicates
  └─────────────────┘   → merged, capped candidate pool (~2,500)
      │
      ▼
  ┌─────────────────┐   LLM card-power (commander-independent, cached)
  │ 3. Scoring      │   + LLM synergy (commander-specific, cached)
  └─────────────────┘
      │
      ▼
  ┌─────────────────┐   database + LLM passes + rules-verify + memory;
  │ 4. Combo detect │   missing pieces pulled into recall, engines on-ramped
  └─────────────────┘
      │
      ▼
  ┌─────────────────┐   8 role buckets (LLM tournament) + power-staples
  │ 5. Pool filter  │   channel + power-bypass safety nets → GA pool
  └─────────────────┘
      │
      ▼
  ┌─────────────────┐   fast heuristic phase → full multi-objective phase;
  │ 6. GA optimize  │   value-weighted mutation, elitism, early stop
  └─────────────────┘
      │
      ▼
  ┌─────────────────┐   LLM set-level critique → guarded swaps
  │ 7. Refinement   │   (can't break role floors or bracket rules)
  └─────────────────┘
      │
      ▼
  bracket audit → HTML report + decklist
```

Every LLM judgment (analysis, synergy, card power, combos) is persisted to a
per-commander disk cache keyed by the exact prompt / card text, so a rebuild
of the same commander is near-deterministic and largely free — the caches are
what make iterating on a deck practical.

---

## Design principles

A few deliberate constraints shaped the engine (and are enforced by tests):

- **No hardcoded "auto-include" lists.** Card quality is *deduced* from the
  LLM power/synergy signals, never from a curated staples list. The Game
  Changer list (a format-rules list, not a quality signal) is the sole
  exception and is sourced from the card data itself.
- **Community data is additive, never a filter.** EDHREC surfaces candidates
  and can *raise* a card's floor; it can never exclude a card or rank one
  down. Popularity is not quality.
- **Prompts are commander-agnostic.** No commander or card names are baked
  into the scoring prompts — worked examples are shape-based — so the engine
  generalizes to commanders released after the model's training cutoff (it
  built credible decks for several such commanders during development).
- **Deterministic where facts exist, sampled where judgment is needed.**
  Format structure and observed failure modes use fixed constants; strategy
  and meta composition use LLM judgment, cached for stability.

---

## Commands

| Command         | What it does                                                        |
| --------------- | ------------------------------------------------------------------- |
| `build`         | Full pipeline: analysis → recall → scoring → GA → refinement        |
| `gui`           | Launch the local web GUI (build, view decks with images, edit)      |
| `refresh-cards` | Download MTGJSON AtomicCards and (re)build `cards.csv`               |
| `power-scan`    | Bulk-score card power into the global cache                         |
| `analyze`       | Run just the commander analysis pass                                |
| `quick`         | Heuristic-only build (no LLM, no GA) — for smoke tests              |
| `search`        | Search the card database                                            |
| `stats`         | Card-database statistics                                            |
| `diff`          | Compare two saved deck snapshots                                    |

Run any command with `--help` for its full flag set. `--mock` runs the whole
pipeline with deterministic stub responses and **no API calls** — useful for
trying the machinery without a key.

---

## Testing

```bash
python -m pytest mtg_deck_builder/tests -q      # 807 tests
```

The suite runs entirely offline (mock LLM, local fixture `test_cards.csv`) and
covers scoring, recall, the GA, bracket enforcement, combo detection, the
caches, and the GUI's server/argument layer.

---

## How this was built

This project is a **human-architected, AI-implemented** collaboration. I
designed the system and directed the work; the code itself was
written by **[Claude](https://www.anthropic.com/claude)** (Anthropic's models,
via Claude Code) under that direction.

Concretely, the human side was the engineering that *isn't* typing code:

- **Architecture** — the LLM-for-judgment / GA-for-search split, the layered
  recall pipeline, the multi-objective fitness function, the per-commander
  caching strategy, and the Commander Bracket system.
- **Design discipline** — the constraints in [Design principles](#design-principles)
  above (no auto-include lists, additive-only community data, commander-agnostic
  prompts) were requirements I set and repeatedly enforced against easier but
  worse implementations.
- **Empirical direction** — dozens of real build cycles where I reviewed the
  output, root-caused what the engine got wrong (a missing staple, a dropped
  combo, a mis-scored dimension), and decided the fix.

I'm sharing this partly as a working tool and partly as an example of what
directing an AI to build a non-trivial system looks like when the human owns
the architecture, the design decisions, and the review.

---

## Attribution

This project consumes data from several excellent community resources, used
under their respective terms. It is unaffiliated with and not endorsed by any
of them:

- **[MTGJSON](https://mtgjson.com)** — card database (`refresh-cards`).
- **[EDHREC](https://edhrec.com)** — commander synergy data and the
  human-verified combo database (which surfaces
  **[Commander Spellbook](https://commanderspellbook.com)**).
- **[Scryfall](https://scryfall.com)** — card images, prices, and oracle tags.

Magic: The Gathering is © Wizards of the Coast. This is a fan-made,
non-commercial tool.

---

## License

[MIT](./LICENSE) © 2026 Brian Klein
