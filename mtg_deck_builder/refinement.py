"""
Post-GA LLM refinement (v0.9.14, extracted v0.9.16).

The GA optimizes per-card averages and count thresholds; this pass hands the
ASSEMBLED 99 (plus the best unused pool alternatives, both annotated with the
engine's synergy/power signals) to the LLM for holistic set-level critique —
redundancy, interaction spread, role quality — and applies its swaps under
three deterministic guard families:

  - mechanical validity (land parity, no duplicate non-basics, locked cards);
  - role floors (a swap may never take a tracked role below its minimum);
  - bracket rules (Game Changer budget, extra-turn limit, banned-combo
    completion).

Swaps are deliberately NOT fitness-gated — the LLM's set-level judgment is
exactly what the per-card fitness cannot express — but the final score and
telemetry are honestly re-computed afterwards.

`run_refinement(builder, result)` takes the DeckBuilder for access to its
caches/config/LLM; the coupling is explicit and one-directional (this module
never mutates builder state, only the result's deck).
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import Card
from .deck_evaluator import DeckEvaluator

logger = logging.getLogger(__name__)


def run_refinement(builder, result) -> None:
    """v0.9.14: holistic LLM refinement of the assembled deck, IN PLACE.

    For up to config.refine_iterations rounds: rank the unused GA-pool
    cards by effective score, hand the deck + alternatives (annotated
    with our synergy/power signals) to the LLM, apply its validated
    swaps, and re-evaluate. Every applied swap is logged to
    result.refinement_log for the report. Swaps are mechanically
    validated (land parity, no duplicates, locked cards protected) but
    deliberately NOT fitness-gated: the LLM's set-level judgment —
    redundancy, interaction spread, role quality — is exactly what the
    per-card fitness cannot see, so fitness must not veto it. The final
    score and telemetry are honestly re-computed afterwards.
    """
    rounds = getattr(builder.config, "refine_iterations", 0)
    if rounds <= 0:
        return
    if bool(getattr(builder.llm.config, "mock_mode", False)):
        return
    deck = result.best_deck
    if not deck or not deck.cards or not builder._ga_candidate_pool:
        return

    evaluator = DeckEvaluator(
        builder.config, builder._analysis,
        synergy_cache=builder._synergy_cache,
        baseline_power_cache=builder._baseline_power_cache,
        flavor_tag_scorer=builder.flavor_tag_scorer,
        combos=builder._reward_combos,
        card_effect_classes=builder._card_effect_classes,
        banned_combos=builder._banned_combos,
    )
    weights = builder.config.get_effective_weights(builder._analysis)
    locked = set(builder.config.locked_cards or [])
    max_swaps = max(1, getattr(builder.config, "refine_max_swaps", 8))
    pool_by_name = {c.name: c for c in builder._ga_candidate_pool}

    def effective(card: Card) -> float:
        return (
            evaluator._get_card_baseline(card) * evaluator.base_weight
            + evaluator._get_card_synergy(card) * evaluator.synergy_weight
        )

    # v0.9.14b: role-floor guard. The GA enforced role minimums via the
    # shortfall penalty; refinement is deliberately NOT fitness-gated, so
    # without this guard the refiner can trade ramp/removal below their
    # floors for engine pieces (observed: a run dropped ramp 10 -> 6).
    # Format-structural floors are deterministic constraints — the
    # refiner's judgment applies WITHIN them, not to them.
    from .card_database import card_fills_role
    effective_targets = builder.config.get_effective_role_targets()
    role_counts = {
        role: sum(1 for c in deck.cards if card_fills_role(c, role))
        for role in effective_targets
    }

    def _floor_violation(out_card: Card, in_card: Card) -> Optional[str]:
        """Role whose floor the swap would break, or None if safe."""
        for role, (min_t, _mx) in effective_targets.items():
            if (
                card_fills_role(out_card, role)
                and not card_fills_role(in_card, role)
                and role_counts[role] - 1 < min_t
            ):
                return role
        return None

    def _update_role_counts(out_card: Card, in_card: Card) -> None:
        for role in effective_targets:
            role_counts[role] += (
                int(card_fills_role(in_card, role))
                - int(card_fills_role(out_card, role))
            )

    # v0.9.15: bracket guards. The pool filter keeps outright-banned
    # cards out of the alternatives, so refinement only has to guard
    # the BUDGET rules: the B3 Game Changer limit, the B2-3 extra-turn
    # limit, and completion of a banned two-card combo (each piece is
    # individually legal; the PAIR is not).
    from .bracket import (
        BRACKET_RULES, bracket_name, is_game_changer, grants_extra_turn,
    )
    bracket = getattr(builder.config, "bracket", 4)
    rules = BRACKET_RULES[bracket]
    counters = {
        "gc": sum(1 for c in deck.cards if is_game_changer(c)),
        "extra": sum(1 for c in deck.cards if grants_extra_turn(c)),
    }

    def _bracket_violation(out_card: Card, in_card: Card) -> Optional[str]:
        gc_limit = rules["gc_limit"]
        if (
            gc_limit is not None
            and is_game_changer(in_card) and not is_game_changer(out_card)
            and counters["gc"] + 1 > gc_limit
        ):
            return f"exceeds the Game Changer budget ({gc_limit})"
        et_limit = rules["extra_turn_limit"]
        if (
            et_limit is not None
            and grants_extra_turn(in_card)
            and not grants_extra_turn(out_card)
            and counters["extra"] + 1 > et_limit
        ):
            return f"exceeds the extra-turn limit ({et_limit})"
        if builder._banned_combos:
            post = {c.name for c in deck.cards} - {out_card.name}
            post.add(in_card.name)
            if builder._commander is not None:
                post.add(builder._commander.name)
            for combo in builder._banned_combos:
                pieces = getattr(combo, "cards", None) or []
                if in_card.name in pieces and all(p in post for p in pieces):
                    return (
                        "completes a bracket-banned combo: "
                        + " + ".join(pieces)
                    )
        return None

    def _update_bracket_counters(out_card: Card, in_card: Card) -> None:
        counters["gc"] += int(is_game_changer(in_card)) - int(is_game_changer(out_card))
        counters["extra"] += (
            int(grants_extra_turn(in_card)) - int(grants_extra_turn(out_card))
        )

    applied_total = 0
    for rnd in range(1, rounds + 1):
        builder._report_progress(
            "refine", f"round_{rnd}", (rnd - 1) / rounds,
            f"LLM refinement round {rnd}/{rounds}...",
        )
        # v0.9.26: roles sitting AT their floor are flagged explicitly.
        # Observed (Doom B5 rerun): 3 of 8 round-1 proposals were rejected
        # by the floor guard — the LLM saw "ramp: 12 (min 12)" but still
        # proposed swapping ramp out for non-ramp, wasting swap budget that
        # would have gone to further upgrades. Mechanical guidance only.
        role_status = (
            f"[Bracket {bracket} — {bracket_name(bracket)}] "
            + "; ".join(
                f"{role}: {role_counts[role]} (min {min_t})"
                + (" [AT FLOOR — swapping this role OUT will be rejected; "
                   "only propose same-role replacements]"
                   if role_counts[role] <= min_t else "")
                for role, (min_t, _mx) in effective_targets.items()
            )
        )
        if bracket == 5:
            role_status += (
                ". cEDH posture: low curve, fast mana, and cheap/free "
                "interaction are premium; board wipes and clunky value "
                "cards are not."
            )
        elif bracket <= 3:
            gc_l = rules["gc_limit"]
            role_status += (
                f". Bracket rules: Game Changer budget {gc_l}; no mass "
                f"land denial; no bracket-banned two-card combos."
            )
        deck_names = {c.name for c in deck.cards}
        unused = [
            c for c in pool_by_name.values()
            if c.name not in deck_names
            and c.name != (builder._commander.name if builder._commander else "")
        ]
        nonland = sorted(
            (c for c in unused if not c.is_land),
            key=effective, reverse=True,
        )[:120]
        lands = sorted(
            (c for c in unused if c.is_land),
            key=effective, reverse=True,
        )[:25]
        alternatives = nonland + lands
        if not alternatives:
            break

        # v0.9.28: per-card role tags for the tracked roles, so the AT FLOOR
        # rule is actionable — the LLM can see WHICH cards hold a floor
        # (Bladebrand counts as protection; Prism as ramp) and recognize
        # same-role upgrades as always-legal.
        card_roles = {
            c.name: roles
            for c in list(deck.cards) + alternatives
            if (roles := [r for r in effective_targets
                          if card_fills_role(c, r)])
        }

        swaps = builder.llm.refine_deck_swaps(
            builder._analysis, deck, alternatives,
            synergy=builder._synergy_cache,
            power=builder._baseline_power_cache,
            max_swaps=max_swaps,
            locked=locked,
            role_status=role_status,
            card_roles=card_roles,
        )
        if not swaps:
            logger.info(f"Refinement round {rnd}: no swaps proposed")
            break

        applied = 0
        for swap in swaps:
            out_name, in_name = swap["out"], swap["in"]
            in_card = pool_by_name.get(in_name)
            if in_card is None:
                continue
            # Locate the outgoing card (it may already have been swapped
            # away earlier this round).
            idx = next(
                (i for i, c in enumerate(deck.cards) if c.name == out_name),
                None,
            )
            if idx is None:
                continue
            out_card = deck.cards[idx]
            # Land parity keeps the mana base size tuned by the GA.
            if out_card.is_land != in_card.is_land:
                continue
            # No duplicate non-basics.
            current = {c.name for c in deck.cards}
            if in_name in current and not in_card.is_basic_land:
                continue
            # Role-floor guard: never let a swap take a functional role
            # below its minimum target.
            violated = _floor_violation(out_card, in_card)
            if violated is not None:
                logger.info(
                    f"Refinement swap rejected (round {rnd}): {out_name} "
                    f"-> {in_name} would drop '{violated}' below its "
                    f"minimum ({role_counts[violated] - 1} < "
                    f"{effective_targets[violated][0]})"
                )
                continue
            # v0.9.15: bracket guard (GC budget, extra-turn limit,
            # banned-combo completion).
            b_violation = _bracket_violation(out_card, in_card)
            if b_violation is not None:
                logger.info(
                    f"Refinement swap rejected (round {rnd}): {out_name} "
                    f"-> {in_name} {b_violation}"
                )
                continue
            deck.cards[idx] = in_card
            _update_role_counts(out_card, in_card)
            _update_bracket_counters(out_card, in_card)
            applied += 1
            applied_total += 1
            result.refinement_log.append({
                "out": out_name, "in": in_name,
                "reason": swap.get("reason", ""), "round": rnd,
            })
            logger.info(
                f"Refinement swap (round {rnd}): {out_name} -> {in_name} "
                f"({swap.get('reason', '')})"
            )
        if applied == 0:
            break

    if applied_total:
        # Honest re-evaluation of the refined deck.
        scores = evaluator.evaluate(deck)
        result.final_score = scores.total(weights)
        result.card_telemetry = evaluator.build_telemetry(deck)
        logger.info(
            f"Refinement complete: {applied_total} swap(s) applied; "
            f"re-evaluated total = {result.final_score:.2f}"
        )
    builder._report_progress(
        "refine", "complete", 1.0,
        f"Refinement: {applied_total} swap(s) applied",
    )
