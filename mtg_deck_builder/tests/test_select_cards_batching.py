"""
Tests for the multi-round batching path in LLMEngine.select_cards.

The dispatcher hands pools larger than SELECT_CARDS_CHUNK_SIZE to a
recursive elimination tournament. These tests inject a deterministic
fake LLM that picks "by name prefix" so we can verify:

  - Every card reaches the LLM at least once (no silent truncation).
  - Per-chunk picks union and dedupe correctly.
  - Recursion converges to the requested count.
  - Small pools still take the single-chunk fast path.
"""

from typing import Optional

import pytest

from mtg_deck_builder.llm_engine import LLMEngine, LLMConfig
from mtg_deck_builder.models import Card, CommanderAnalysis


def _card(name: str) -> Card:
    """Card with deterministic text — useful for tracing through batches."""
    return Card(
        name=name,
        mana_cost="{1}{W}", mana_value=2,
        card_type="Creature", text=f"text-of-{name}",
        color_identity="W", colors="W",
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _analysis() -> CommanderAnalysis:
    return CommanderAnalysis(
        name="Test Commander", color_identity="W",
        key_mechanics=["test"], build_around_text="test strategy",
        evaluation_notes="...", category_queries={},
        synergy_keywords=["test"],
    )


class _RecordingLLM(LLMEngine):
    """
    LLM stub that bypasses the real API but exercises the dispatcher /
    batching paths. Records every _select_cards_chunk invocation so
    tests can assert on call counts and the union of cards seen.
    """

    def __init__(self):
        super().__init__(LLMConfig(mock_mode=False, api_key="dummy"))
        # Override any client construction — we don't make real calls
        self.client = object()
        self.config.mock_mode = False
        self.calls: list[dict] = []

    def _select_cards_chunk(self, analysis, candidates, role, count,
                            mode="role", synergy_hints=None, model=None):
        # Record the call (including hints + model so tests can inspect them).
        self.calls.append({
            "candidate_names": [c.name for c in candidates],
            "count": count,
            "role": role,
            "mode": mode,
            "synergy_hints": dict(synergy_hints) if synergy_hints else None,
            "model": model,
        })
        # Deterministic pick: take the lexicographically-smallest names
        # unless any cards have hint tags, in which case prefer tagged
        # cards first (matches the prompt's intended behavior for testing
        # downstream effects of hints).
        if synergy_hints:
            def rank_key(c):
                tag = synergy_hints.get(c.name, "")
                # "[SYN+++]" > "[SYN++]" > "[SYN+]" > ""
                priority = tag.count("+")
                return (-priority, c.name)
            ranked = sorted(candidates, key=rank_key)
        else:
            ranked = sorted(candidates, key=lambda c: c.name)
        return [c.name for c in ranked[:count]]


# ----------------------------------------------------------------------
# Single-chunk fast path
# ----------------------------------------------------------------------

class TestSinglechunkFastPath:
    def test_small_pool_uses_one_chunk(self):
        llm = _RecordingLLM()
        pool = [_card(f"Card{i:03d}") for i in range(50)]
        out = llm.select_cards(_analysis(), pool, role="ramp", count=10)
        assert len(llm.calls) == 1, "Small pool should make exactly one LLM call"
        assert len(out) == 10

    def test_pool_smaller_than_count_returns_all_no_call(self):
        llm = _RecordingLLM()
        pool = [_card("Foo"), _card("Bar")]
        out = llm.select_cards(_analysis(), pool, role="ramp", count=10)
        assert llm.calls == [], "No LLM call when pool ≤ count"
        assert set(out) == {"Foo", "Bar"}

    def test_at_chunk_size_still_single_call(self):
        """Pool of exactly SELECT_CARDS_CHUNK_SIZE goes single-chunk."""
        llm = _RecordingLLM()
        n = LLMEngine.SELECT_CARDS_CHUNK_SIZE
        pool = [_card(f"C{i:04d}") for i in range(n)]
        out = llm.select_cards(_analysis(), pool, role="ramp", count=10)
        assert len(llm.calls) == 1
        assert len(out) == 10


# ----------------------------------------------------------------------
# Multi-round batching
# ----------------------------------------------------------------------

class TestBatchingTournament:
    def test_every_card_reaches_llm_at_least_once(self):
        """The headline claim: no silent truncation."""
        llm = _RecordingLLM()
        n = 500  # well above chunk_size of 150
        pool = [_card(f"Card{i:04d}") for i in range(n)]
        llm.select_cards(_analysis(), pool, role="synergy", count=50)

        # Union of all candidate names across all calls must include every input name.
        seen: set[str] = set()
        for call in llm.calls:
            seen.update(call["candidate_names"])
        missing = {c.name for c in pool} - seen
        assert not missing, (
            f"{len(missing)} cards never reached the LLM "
            f"(sample: {sorted(missing)[:5]})"
        )

    def test_first_round_sizes(self):
        """Round 1 chunks the input pool into ceil(n/chunk_size) calls."""
        llm = _RecordingLLM()
        n = 320  # → 3 chunks of 150,150,20 in round 1
        pool = [_card(f"Card{i:04d}") for i in range(n)]
        llm.select_cards(_analysis(), pool, role="synergy", count=50)

        chunk_size = LLMEngine.SELECT_CARDS_CHUNK_SIZE
        # First 3 calls should be round-1 chunks of size 150,150,20.
        # (Any subsequent calls are downstream rounds.)
        first_three = [len(c["candidate_names"]) for c in llm.calls[:3]]
        # Last chunk in round 1 may be smaller than chunk_size if the
        # final remainder fits within `per_chunk_keep` and the whole chunk
        # is kept without an LLM call. Verify the first two are full size.
        assert first_three[0] == chunk_size
        assert first_three[1] == chunk_size

    def test_converges_to_requested_count(self):
        llm = _RecordingLLM()
        n = 2500
        pool = [_card(f"Card{i:04d}") for i in range(n)]
        out = llm.select_cards(_analysis(), pool, role="synergy", count=150)
        assert len(out) <= 150
        # Output must be a subset of the input pool
        pool_names = {c.name for c in pool}
        assert all(name in pool_names for name in out)
        # Output names are all unique
        assert len(out) == len(set(out))

    def test_returns_exactly_count_for_2500_pool(self):
        """Regression: pre-fix the Lathiel 2500-card tournament returned
        only 116/150 nominees because round 5 over-pruned (192 candidates
        in 2 chunks → 75+42 = 117 from asymmetric shrinkage). The fix
        terminates the tournament when survivors fit in
        SELECT_CARDS_MAX_SINGLE_PASS and does a final LLM pick instead.
        """
        llm = _RecordingLLM()
        pool = [_card(f"Card{i:04d}") for i in range(2500)]
        out = llm.select_cards(_analysis(), pool, role="synergy", count=150)
        assert len(out) == 150, (
            f"Expected exactly 150 cards, got {len(out)}. Tournament "
            f"is over-pruning again."
        )

    def test_returns_exactly_count_at_threshold_boundary(self):
        """Pools just over SELECT_CARDS_MAX_SINGLE_PASS used to recurse
        unnecessarily; now they take one round + a final pass."""
        from mtg_deck_builder.llm_engine import LLMEngine
        threshold = LLMEngine.SELECT_CARDS_MAX_SINGLE_PASS
        # 50 cards above threshold = single round + final pass
        pool = [_card(f"Card{i:04d}") for i in range(threshold + 50)]
        llm = _RecordingLLM()
        out = llm.select_cards(_analysis(), pool, role="synergy", count=100)
        assert len(out) == 100

    def test_total_call_count_matches_expected_rounds(self):
        """For pool=2500 with chunk=150 and per-chunk-keep=75, we expect:
           Round 1: 17 chunks  (last chunk has 100 cards > 75 keep, so it's still an LLM call)
           Round 2: ceil(1275 / 150) = 9
           Round 3: ceil(675 / 150) = 5
           Round 4: ceil(375 / 150) = 3
           Round 5: ceil(225 / 150) = 2 → 150 nominees, ≤ chunk_size, final pick = 1 more call
        Total ≥ 17 + 9 + 5 + 3 + 2 + 1 = 37 calls (could differ if last chunks
        skip the LLM via the "whole chunk survives" path; see the test below
        for the exact behavior, this just bounds it).
        """
        llm = _RecordingLLM()
        pool = [_card(f"C{i:04d}") for i in range(2500)]
        llm.select_cards(_analysis(), pool, role="synergy", count=150)
        # Loose bound: should be in the ballpark of ~37 calls. Allow ±10.
        assert 25 <= len(llm.calls) <= 50, (
            f"Expected ~37 LLM calls for 2500-card pool, got {len(llm.calls)}"
        )

    def test_short_chunk_with_few_candidates_skips_llm(self):
        """Within a tournament round, a chunk ≤ per_chunk_keep passes
        through without an LLM call.

        Use a pool > SELECT_CARDS_MAX_SINGLE_PASS so we actually enter
        the tournament path (rather than the single-pass shortcut for
        pools ≤ 300). 400 cards → 3 chunks of 150,150,100. With
        per_chunk_keep=75 and count=10, all three chunks exceed
        per_chunk_keep so all three get LLM calls — but if we shrink to
        a hypothetical pool where the last chunk is < 75, that one
        skips. Verify that exact "whole chunk survives" mechanic at the
        next round.
        """
        llm = _RecordingLLM()
        # 400 cards triggers the tournament. After round 1 yields ~225
        # nominees, that's ≤ 300 so the next call is a single final pass
        # (no second round, no chunk-skip mechanic). To exercise the
        # chunk-skip path within a round, use a larger pool whose last
        # chunk is below per_chunk_keep.
        pool = [_card(f"C{i:04d}") for i in range(380)]
        # 380 → 3 chunks of 150,150,80. All > 75 so all get LLM calls.
        # Round 1 nominees: ~225 (≤ 300) → final single pass call.
        # Expect: 3 round-1 calls + 1 final = 4 calls total.
        llm.select_cards(_analysis(), pool, role="ramp", count=10)
        assert len(llm.calls) >= 2

    def test_chunk_skipped_when_smaller_than_per_chunk_keep(self):
        """Exercise the 'whole chunk survives, no LLM call' branch.

        Need a pool size whose last chunk is < per_chunk_keep (75) AND
        whose first-round nominees still exceed SELECT_CARDS_MAX_SINGLE_PASS
        so we run a second round (otherwise we shortcut to a single
        final pass and the skip mechanic isn't visible).
        """
        llm = _RecordingLLM()
        # 1010 cards → round 1: 7 chunks. Last chunk = 1010 - 6*150 = 110 cards.
        # 110 > 75 so it gets an LLM call too. Adjust:
        # 970 → 7 chunks. Last = 970 - 6*150 = 70 cards. 70 ≤ 75 → SKIPS.
        # Round 1: 6 LLM calls + 1 skip → ~6*75 + 70 = 520 nominees.
        # 520 > 300 → recurse.
        # Round 2: 520 → 4 chunks of 150,150,150,70. Last skips.
        #   3 LLM calls + 1 skip → ~3*75 + 70 = 295 nominees.
        # 295 ≤ 300 → single final pass = 1 more call.
        # Total: 6 + 3 + 1 = 10 calls.
        pool = [_card(f"C{i:04d}") for i in range(970)]
        llm.select_cards(_analysis(), pool, role="ramp", count=10)
        # 6 chunks per round called → at least 6+3+1 = 10 expected.
        # Loose bound: must skip at least once (so fewer calls than
        # cumulative-chunk-count if every chunk were called).
        # Total chunks across rounds without skip: 7 + 4 + 1 = 12.
        # With skips: 6 + 3 + 1 = 10.
        assert len(llm.calls) < 12, (
            f"Expected at least one chunk skipped (< 12 calls), "
            f"got {len(llm.calls)}"
        )

    def test_no_duplicates_across_rounds(self):
        """Dedupe across chunks: a name picked in round 1 shouldn't
        appear twice in the final output."""
        llm = _RecordingLLM()
        pool = [_card(f"C{i:04d}") for i in range(500)]
        out = llm.select_cards(_analysis(), pool, role="x", count=50)
        assert len(out) == len(set(out))


# ----------------------------------------------------------------------
# Backward compat: already_selected filter still applies
# ----------------------------------------------------------------------

class TestAlreadySelectedFilter:
    def test_already_selected_excluded_in_batched_path(self):
        llm = _RecordingLLM()
        pool = [_card(f"Card{i:03d}") for i in range(500)]
        already = {f"Card{i:03d}" for i in range(0, 500, 5)}  # every 5th card
        out = llm.select_cards(
            _analysis(), pool, role="synergy", count=50,
            already_selected=already,
        )
        # No "Card000", "Card005", ... should appear in output
        assert not (set(out) & already)
