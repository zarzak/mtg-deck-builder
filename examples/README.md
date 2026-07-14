# Example output

A real, unedited build produced by this tool — so you can see what it makes
without running it yourself.

### Grand Arbiter Augustin IV — cEDH stax-control (bracket 5)

- **[Decklist](./grand_arbiter_augustin_iv.txt)** — the 99, grouped by type.
- **[Full HTML report](https://htmlpreview.github.io/?https://github.com/zarzak/mtg-deck-builder/blob/main/examples/grand_arbiter_augustin_iv.html)**
  — per-card scoring breakdown, mana curve, detected combos, bracket
  compliance, and **pool-entry provenance** (which channel put each card in
  the pool). *(GitHub shows HTML as source; that link renders it.)*

**Why this one:** it's the engine's strongest result (a normalized total of
**88.8**) and a good stress test — Grand Arbiter is an off-meta stax/tax
commander released after the model's training cutoff, so the engine had no
memorized list to lean on. It nonetheless assembled a coherent stax package
(Winter Orb, Stasis, Trinisphere, Sphere of Resistance, Back to Basics,
Thalia, Drannith Magistrate, Linvala, The Tabernacle at Pendrell Vale…), a
deep counter/tax suite, and the full fast-mana package — **with no hardcoded
stax category**; those cards surfaced through the normal recall + synergy
machinery.
