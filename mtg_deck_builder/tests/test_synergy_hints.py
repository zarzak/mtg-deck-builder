"""
Tests for the v0.9.1 synergy-hint pipeline.

Headline claims under test:
  - _compute_synergy_hints grades each card by recall source count.
  - select_cards passes hints through to the user prompt builder.
  - User prompt lines for tagged cards are prefixed with the right tag.
  - Hints propagate through batched-tournament rounds (no information loss).
  - End-to-end: a card flagged by 3 sources gets [SYN+++] in the prompt.
"""

import pytest

from mtg_deck_builder.card_database import CardDatabase
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import BuildConfig, CommanderAnalysis, Card
from mtg_deck_builder.llm_engine import LLMEngine, LLMConfig


def _card(name: str, text: str = "") -> Card:
    return Card(
        name=name, mana_cost="{1}{W}", mana_value=2,
        card_type="Creature", text=text or f"text of {name}",
        color_identity="W", colors="W",
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _make_builder(test_csv_path, **config_overrides) -> DeckBuilder:
    analysis = CommanderAnalysis(
        name="Lathiel, the Bounteous Dawn",
        color_identity="G,W",
        key_mechanics=["lifegain"],
        build_around_text="gain life",
        evaluation_notes="...",
        category_queries={},
        synergy_keywords=["gain life", "lifelink"],
        synergy_patterns=["gain life", "lifelink"],
    )
    config = BuildConfig(
        commander_name=analysis.name,
        random_seed=42,
        candidates_per_category=10,
        synergy_engine_target=15,
        **config_overrides,
    )
    builder = DeckBuilder(
        card_database_path=test_csv_path,
        config=config,
        llm_config=LLMConfig(mock_mode=True),
    )
    db = builder.db
    builder._commander = db.get_by_name(analysis.name)
    builder._analysis = analysis
    return builder


# ----------------------------------------------------------------------
# Hint computation
# ----------------------------------------------------------------------

class TestComputeSynergyHints:
    def test_three_sources_yields_triple_plus(self, test_csv_path):
        b = _make_builder(test_csv_path)
        b._edhrec_recall_names = {"Heliod, Sun-Crowned"}
        b._embedding_recall_names = {"Heliod, Sun-Crowned"}
        b._pattern_recall_names = {"Heliod, Sun-Crowned"}

        hints = b._compute_synergy_hints()
        assert hints["Heliod, Sun-Crowned"] == "[SYN+++]"

    def test_two_sources_yields_double_plus(self, test_csv_path):
        # With all THREE sources active (enabled=3), a card in 2 of them is
        # the middle tier. A throwaway card keeps the embedding source
        # non-empty so enabled == 3.
        b = _make_builder(test_csv_path)
        b._edhrec_recall_names = {"Soul Warden"}
        b._pattern_recall_names = {"Soul Warden"}
        b._embedding_recall_names = {"Some Other Card"}

        hints = b._compute_synergy_hints()
        assert hints["Soul Warden"] == "[SYN++]"

    def test_two_enabled_sources_agree_yields_triple_plus(self, test_csv_path):
        # Adaptive tiering: when only 2 sources are productive (e.g. a NEW
        # commander with no EDHREC data), a card both of them flag is the top
        # tier — otherwise the [SYN+++] bypass would be empty for exactly the
        # new commanders we want it to cover.
        b = _make_builder(test_csv_path)
        b._embedding_recall_names = {"Soul Warden"}
        b._pattern_recall_names = {"Soul Warden"}
        # edhrec empty → enabled == 2

        hints = b._compute_synergy_hints()
        assert hints["Soul Warden"] == "[SYN+++]"

    def test_single_enabled_source_never_triple_plus(self, test_csv_path):
        # Only one productive source → a lone hit is weak evidence, [SYN+].
        b = _make_builder(test_csv_path)
        b._embedding_recall_names = {"Soul Warden"}
        # edhrec + pattern empty → enabled == 1

        hints = b._compute_synergy_hints()
        assert hints["Soul Warden"] == "[SYN+]"

    def test_one_source_yields_single_plus(self, test_csv_path):
        b = _make_builder(test_csv_path)
        b._pattern_recall_names = {"Random Card"}

        hints = b._compute_synergy_hints()
        assert hints["Random Card"] == "[SYN+]"

    def test_no_sources_has_no_entry(self, test_csv_path):
        b = _make_builder(test_csv_path)
        # All recall sets are empty
        hints = b._compute_synergy_hints()
        assert "Sol Ring" not in hints
        assert hints == {}

    def test_multiple_cards_get_distinct_tiers(self, test_csv_path):
        b = _make_builder(test_csv_path)
        b._edhrec_recall_names = {"A", "B", "C"}
        b._embedding_recall_names = {"A", "B"}
        b._pattern_recall_names = {"A"}

        hints = b._compute_synergy_hints()
        assert hints["A"] == "[SYN+++]"
        assert hints["B"] == "[SYN++]"
        assert hints["C"] == "[SYN+]"


# ----------------------------------------------------------------------
# User prompt annotation
# ----------------------------------------------------------------------

class TestUserPromptAnnotation:
    def test_tagged_cards_get_prefixed_in_user_prompt(self):
        """The user prompt should prefix tagged cards with their hint."""
        llm = LLMEngine(LLMConfig(mock_mode=True))

        cards = [_card("Heliod"), _card("Soul Warden"), _card("Sol Ring")]
        hints = {
            "Heliod": "[SYN+++]",
            "Soul Warden": "[SYN++]",
            # Sol Ring intentionally untagged
        }

        prompt = llm._build_select_cards_user_prompt(
            candidates=cards,
            role="test",
            count=2,
            mode="role",
            synergy_hints=hints,
        )

        # Heliod and Soul Warden should be prefixed
        assert "[SYN+++] **Heliod**" in prompt
        assert "[SYN++] **Soul Warden**" in prompt
        # Sol Ring line should NOT have a hint prefix
        sol_ring_line = next(
            line for line in prompt.split("\n") if "**Sol Ring**" in line
        )
        assert not sol_ring_line.startswith("[SYN")

    def test_no_hints_produces_clean_prompt(self):
        """Untagged calls should produce the original unannotated format."""
        llm = LLMEngine(LLMConfig(mock_mode=True))
        cards = [_card("Foo"), _card("Bar")]
        prompt = llm._build_select_cards_user_prompt(
            candidates=cards, role="test", count=1, mode="role",
            synergy_hints=None,
        )
        # No card should have a tag prefix
        for line in prompt.split("\n"):
            if line.startswith("**"):
                assert "[SYN" not in line.split("**")[0]


# ----------------------------------------------------------------------
# Hints propagate through batched rounds
# ----------------------------------------------------------------------

class _HintRecordingLLM(LLMEngine):
    """Records every _select_cards_chunk call's synergy_hints arg."""

    def __init__(self):
        super().__init__(LLMConfig(mock_mode=False, api_key="dummy"))
        self.client = object()
        self.config.mock_mode = False
        self.calls: list[dict] = []

    def _select_cards_chunk(self, analysis, candidates, role, count,
                            mode="role", synergy_hints=None, model=None):
        self.calls.append({
            "candidates": [c.name for c in candidates],
            "count": count,
            "synergy_hints": dict(synergy_hints) if synergy_hints else None,
            "model": model,
        })
        # Prefer tagged cards (mimicking what the real LLM should do)
        if synergy_hints:
            ranked = sorted(
                candidates,
                key=lambda c: (-synergy_hints.get(c.name, "").count("+"), c.name),
            )
        else:
            ranked = sorted(candidates, key=lambda c: c.name)
        return [c.name for c in ranked[:count]]


class TestHintsPropagateThroughRounds:
    def test_hints_reach_every_chunk_call(self):
        """Across a multi-round tournament, every chunk call should
        receive the synergy_hints dict."""
        llm = _HintRecordingLLM()

        # Build a pool that requires batching (> MAX_SINGLE_PASS = 300)
        pool = [_card(f"Card{i:04d}") for i in range(500)]
        hints = {"Card0042": "[SYN+++]", "Card0099": "[SYN++]"}

        analysis = CommanderAnalysis(
            name="Test", color_identity="W", key_mechanics=[],
            build_around_text="", evaluation_notes="",
            category_queries={}, synergy_keywords=[],
        )
        llm.select_cards(analysis, pool, role="synergy", count=50,
                         synergy_hints=hints)

        # Every chunk call must have received the hints
        for call in llm.calls:
            assert call["synergy_hints"] == hints, (
                "Hints were dropped somewhere in the tournament recursion"
            )

    def test_tagged_card_survives_to_final_pick(self):
        """A pool where one card has [SYN+++] should see that card
        prioritized to the top of the final output."""
        llm = _HintRecordingLLM()
        pool = [_card(f"Card{i:04d}") for i in range(400)]
        # Tag exactly one card
        hints = {"Card0042": "[SYN+++]"}

        analysis = CommanderAnalysis(
            name="Test", color_identity="W", key_mechanics=[],
            build_around_text="", evaluation_notes="",
            category_queries={}, synergy_keywords=[],
        )
        out = llm.select_cards(analysis, pool, role="x", count=10,
                               synergy_hints=hints)
        assert "Card0042" in out, (
            "Tagged card should survive elimination rounds and land in "
            f"the final 10 picks. Got: {out}"
        )


# ----------------------------------------------------------------------
# End-to-end: filtering phase passes hints
# ----------------------------------------------------------------------

class TestFilteringPhaseUsesHints:
    def test_hints_passed_to_select_cards_in_filtering(self, test_csv_path):
        """Verifies the deck_builder wires synergy_hints into every
        select_cards call during _phase_llm_filtering."""
        b = _make_builder(test_csv_path)
        # Inject recall membership so hints get computed
        b._edhrec_recall_names = {"Soul Warden"}
        b._embedding_recall_names = {"Soul Warden"}
        b._pattern_recall_names = {"Soul Warden"}

        # Replace the LLM with a recorder that captures call args
        captured = []
        original_select = b.llm.select_cards

        def spy_select_cards(analysis, candidates, role, count,
                             already_selected=None, mode="role",
                             synergy_hints=None):
            captured.append({"role": role, "synergy_hints": synergy_hints})
            return original_select(
                analysis, candidates, role, count,
                already_selected=already_selected, mode=mode,
                synergy_hints=synergy_hints,
            )

        b.llm.select_cards = spy_select_cards
        b._phase_generate_pools()
        b._candidates.synergy = []  # skip synergy_engine pass for speed
        b._phase_llm_filtering()

        # At least one call should have received the computed hints
        with_hints = [c for c in captured if c["synergy_hints"]]
        assert len(with_hints) > 0, (
            f"No select_cards call received synergy_hints; got {len(captured)} "
            f"total calls"
        )
        # The hints should include Soul Warden as [SYN+++]
        for c in with_hints:
            assert c["synergy_hints"].get("Soul Warden") == "[SYN+++]"
