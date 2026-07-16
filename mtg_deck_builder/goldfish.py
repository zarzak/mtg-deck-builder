"""
Goldfish / playtest simulator (v0.9.34, #35).

Monte-Carlo "goldfishing" of a finished deck: shuffle, draw opening hands
with mulligans, play out N turns with land drops and mana-rock acceleration,
and measure the consistency questions a static score can't answer:

  - How often is the opening hand keepable?
  - How many lands are in play by turn 3/4/5 — and how often do we miss?
  - What turn does the commander actually come down?
  - How often are the detected combos fully drawn by turn N?

Entirely deterministic given a seed, entirely local — no LLM calls, no cost.

Deliberate simplifications (documented, revisit if they mislead):
  - Colors are ignored ("colorless goldfish"): castability = total mana
    only. Color screw is real but modeling duals/fetches honestly is a
    project of its own.
  - All lands enter untapped and produce 1.
  - Mana acceleration models ARTIFACT rocks only (cast greedily, cheapest
    first; production available from the NEXT turn). Dorks and rituals are
    ignored — dorks die/summoning-sick, rituals are one-shot.
  - The player draws on every turn including turn 1 (multiplayer Commander:
    nobody skips a draw).
  - Mulligan: keep a hand with 2-5 lands; otherwise redraw at one fewer
    card, keeping whatever arrives at 5. (Simplified London.)
  - Combo assembly counts a combo as "drawn" when every piece is in the
    cards seen so far (commander counts as always available). It does not
    model casting them all in one turn.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .models import Card


# ----------------------------------------------------------------------
# Card classification helpers
# ----------------------------------------------------------------------

_WORD_COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def rock_production(card: Card) -> int:
    """How much mana an artifact rock adds per turn (0 = not a rock).

    Parses the "Add ..." line: counts {W}{U}{B}{R}{G}{C} symbols (Sol Ring's
    "Add {C}{C}" -> 2), or the "add one/two/three mana" word form. Artifacts
    only; lands and non-mana artifacts return 0.
    """
    types = (card.types or "").lower()
    if "land" in types or "artifact" not in types:
        return 0
    text = (card.text or "")
    best = 0
    for line in text.splitlines():
        low = line.lower()
        if "add" not in low:
            continue
        # Count only mana symbols AFTER the word "add" on this line
        # (excludes activation costs like "{T}:" before it).
        idx = low.find("add")
        symbols = len(re.findall(r"\{[WUBRGC]\}", line[idx:]))
        if symbols == 0:
            m = re.search(r"add (one|two|three|four|five) mana", low)
            if m:
                symbols = _WORD_COUNTS[m.group(1)]
        best = max(best, symbols)
    return best


# ----------------------------------------------------------------------
# Config / report
# ----------------------------------------------------------------------

@dataclass
class GoldfishConfig:
    trials: int = 500
    turns: int = 6
    seed: Optional[int] = 42
    keep_min_lands: int = 2
    keep_max_lands: int = 5


@dataclass
class GoldfishReport:
    trials: int = 0
    turns: int = 0
    land_count: int = 0
    rock_count: int = 0

    keep7_rate: float = 0.0          # opening 7 was keepable
    avg_mulligans: float = 0.0
    avg_lands_by_turn: dict = field(default_factory=dict)   # turn -> avg
    missed_drop_by_t3_rate: float = 0.0   # < 3 lands in play on turn 3
    avg_mana_by_turn: dict = field(default_factory=dict)    # lands + rocks

    commander_name: str = ""
    commander_mv: int = 0
    avg_commander_turn: float = 0.0       # among trials where cast
    commander_cast_by_turn: dict = field(default_factory=dict)  # turn -> P
    commander_uncast_rate: float = 0.0    # not cast within the horizon

    any_combo_drawn_by_turn: dict = field(default_factory=dict)  # turn -> P
    combo_details: list = field(default_factory=list)
    # top combos: {"cards": [...], "payoff": int, "drawn_by_final": float}

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def to_text(self) -> str:
        lines = [
            f"Goldfish simulation — {self.trials} games, "
            f"{self.turns} turns (colors ignored; artifact ramp only)",
            f"  Deck: {self.land_count} lands, {self.rock_count} mana rocks",
            f"  Keepable opening 7: {self.keep7_rate:.0%}   "
            f"avg mulligans: {self.avg_mulligans:.2f}",
            "  Lands in play:  " + "  ".join(
                f"t{t}={v:.1f}" for t, v in sorted(self.avg_lands_by_turn.items())),
            f"  Missed a land drop by turn 3: {self.missed_drop_by_t3_rate:.0%}",
            "  Mana available: " + "  ".join(
                f"t{t}={v:.1f}" for t, v in sorted(self.avg_mana_by_turn.items())),
            f"  Commander ({self.commander_name}, MV {self.commander_mv}): "
            f"avg cast turn {self.avg_commander_turn:.1f}"
            + (f", uncast within horizon {self.commander_uncast_rate:.0%}"
               if self.commander_uncast_rate > 0.005 else ""),
        ]
        if self.any_combo_drawn_by_turn:
            lines.append("  Any detected combo fully drawn: " + "  ".join(
                f"by t{t}={p:.0%}"
                for t, p in sorted(self.any_combo_drawn_by_turn.items())))
        for c in self.combo_details[:5]:
            lines.append(
                f"    {' + '.join(c['cards'])} [{c['payoff']}]: "
                f"{c['drawn_by_final']:.0%} by t{self.turns}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------

def simulate(
    cards: list[Card],
    commander: Card,
    combos: Optional[list] = None,
    config: Optional[GoldfishConfig] = None,
) -> GoldfishReport:
    """Run the Monte-Carlo goldfish over `cards` (the 99).

    `combos` is an optional list of Combo-like objects (`.cards`,
    `.payoff`); the commander counts as always available for them.
    """
    cfg = config or GoldfishConfig()
    rng = random.Random(cfg.seed)
    n = len(cards)
    if n == 0:
        return GoldfishReport(trials=0, turns=cfg.turns)

    is_land = [("land" in (c.types or "").lower()) for c in cards]
    production = [rock_production(c) for c in cards]
    mv = [max(0, int(c.mana_value or 0)) for c in cards]
    names = [c.name for c in cards]
    commander_mv = max(0, int(commander.mana_value or 0)) if commander else 0

    # Pre-index combos: piece name -> needed (commander pre-satisfied).
    combo_specs = []
    if combos:
        for combo in combos:
            pieces = set(getattr(combo, "cards", []) or [])
            pieces.discard(commander.name if commander else "")
            if pieces:
                combo_specs.append(
                    (frozenset(pieces), int(getattr(combo, "payoff", 0)),
                     list(getattr(combo, "cards", []))))

    turns = list(range(1, cfg.turns + 1))
    sum_lands = {t: 0.0 for t in turns}
    sum_mana = {t: 0.0 for t in turns}
    keep7 = 0
    mull_total = 0
    missed_t3 = 0
    commander_turns: list[int] = []
    uncast = 0
    combo_by_turn = {t: 0 for t in turns}
    combo_final_hits = [0] * len(combo_specs)

    indices = list(range(n))
    for _ in range(cfg.trials):
        rng.shuffle(indices)
        deck_pos = 0

        # --- mulligan loop (simplified London) ---
        hand_size = 7
        mulls = 0
        while True:
            hand = indices[:hand_size]
            lands_in_hand = sum(1 for i in hand if is_land[i])
            if (cfg.keep_min_lands <= lands_in_hand <= cfg.keep_max_lands
                    or hand_size <= 5):
                break
            mulls += 1
            hand_size -= 1
            rng.shuffle(indices)
        if mulls == 0:
            keep7 += 1
        mull_total += mulls
        deck_pos = hand_size

        hand = list(indices[:hand_size])
        seen = set(hand)
        lands_in_play = 0
        rock_mana = 0          # production from rocks cast on PRIOR turns
        pending_rock_mana = 0  # rocks cast this turn come online next turn
        commander_cast_turn = None

        for t in turns:
            # Draw (multiplayer: every turn draws).
            if deck_pos < n:
                card_i = indices[deck_pos]
                hand.append(card_i)
                seen.add(card_i)
                deck_pos += 1

            rock_mana += pending_rock_mana
            pending_rock_mana = 0

            # Land drop: play a land if we have one.
            land_i = next((i for i in hand if is_land[i]), None)
            if land_i is not None:
                hand.remove(land_i)
                lands_in_play += 1

            mana = lands_in_play + rock_mana
            spent = 0

            # Cast rocks greedily, cheapest first (acceleration next turn).
            for i in sorted([i for i in hand if production[i] > 0],
                            key=lambda i: mv[i]):
                if mv[i] <= mana - spent:
                    spent += mv[i]
                    pending_rock_mana += production[i]
                    hand.remove(i)

            # Cast the commander once affordable.
            if commander_cast_turn is None and commander_mv <= mana - spent:
                commander_cast_turn = t
                spent += commander_mv

            sum_lands[t] += lands_in_play
            sum_mana[t] += mana
            if t == 3 and lands_in_play < 3:
                missed_t3 += 1

            # Combo assembly (drawn-based; `seen` grows monotonically so
            # this is naturally cumulative per turn).
            if combo_specs:
                seen_names = {names[i] for i in seen}
                if any(pieces <= seen_names
                       for pieces, _p, _c in combo_specs):
                    combo_by_turn[t] += 1

        # Per-combo final-turn stats (independent of the any-combo counter).
        if combo_specs:
            seen_names = {names[i] for i in seen}
            for spec_i, (pieces, _payoff, _cards) in enumerate(combo_specs):
                if pieces <= seen_names:
                    combo_final_hits[spec_i] += 1

        if commander_cast_turn is None:
            uncast += 1
        else:
            commander_turns.append(commander_cast_turn)

    trials = cfg.trials
    report = GoldfishReport(
        trials=trials,
        turns=cfg.turns,
        land_count=sum(is_land),
        rock_count=sum(1 for p in production if p > 0),
        keep7_rate=keep7 / trials,
        avg_mulligans=mull_total / trials,
        avg_lands_by_turn={t: sum_lands[t] / trials for t in turns},
        missed_drop_by_t3_rate=missed_t3 / trials if cfg.turns >= 3 else 0.0,
        avg_mana_by_turn={t: sum_mana[t] / trials for t in turns},
        commander_name=commander.name if commander else "",
        commander_mv=commander_mv,
        avg_commander_turn=(sum(commander_turns) / len(commander_turns)
                            if commander_turns else 0.0),
        commander_uncast_rate=uncast / trials,
        any_combo_drawn_by_turn=(
            {t: combo_by_turn[t] / trials for t in turns}
            if combo_specs else {}),
    )
    # Cumulative commander-cast distribution P(cast by turn t).
    if commander_turns or uncast:
        report.commander_cast_by_turn = {
            t: sum(1 for ct in commander_turns if ct <= t) / trials
            for t in turns
        }
    if combo_specs:
        details = [
            {"cards": spec[2], "payoff": spec[1],
             "drawn_by_final": combo_final_hits[i] / trials}
            for i, spec in enumerate(combo_specs)
        ]
        details.sort(key=lambda d: (-d["drawn_by_final"], -d["payoff"]))
        report.combo_details = details
    return report
