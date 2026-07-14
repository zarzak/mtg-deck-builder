"""
Tests for v0.9.2: synergy_hints plumbed through the scoring phase.

Headline claims:
  - score_synergy_batch accepts a synergy_hints parameter and passes it
    through to _score_synergy_single.
  - _score_synergy_single annotates each card's user-prompt line with
    its hint tag when one is provided.
  - The deck_builder's _phase_synergy_scoring passes the computed hints
    to score_synergy_batch.
"""

import pytest
from unittest.mock import patch, MagicMock

from mtg_deck_builder.llm_engine import LLMEngine, LLMConfig
from mtg_deck_builder.models import Card, CommanderAnalysis


def _card(name: str, text: str = "test text") -> Card:
    return Card(
        name=name, mana_cost="{1}{W}", mana_value=2,
        card_type="Creature", text=text,
        color_identity="W", colors="W",
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _analysis() -> CommanderAnalysis:
    return CommanderAnalysis(
        name="Test Commander", color_identity="W",
        key_mechanics=["lifegain"], build_around_text="gain life",
        evaluation_notes="...", category_queries={},
        synergy_keywords=["gain life"],
    )


# ----------------------------------------------------------------------
# Hints flow through batch → single
# ----------------------------------------------------------------------

class TestHintsFlowThroughScoring:
    def test_score_synergy_batch_passes_hints_to_single(self):
        """The batch entry point must forward hints to each batch's
        _score_synergy_single call."""
        llm = LLMEngine(LLMConfig(mock_mode=False, api_key="dummy"))
        llm.client = object()
        llm.config.mock_mode = False

        captured = []

        def fake_single(analysis, cards, synergy_hints=None, class_sink=None):
            captured.append(dict(synergy_hints) if synergy_hints else None)
            return {c.name: 50.0 for c in cards}

        llm._score_synergy_single = fake_single

        cards = [_card(f"Card{i}") for i in range(60)]  # forces 2 batches
        hints = {"Card1": "[SYN+++]", "Card2": "[SYN++]"}
        llm.score_synergy_batch(_analysis(), cards, batch_size=30,
                                synergy_hints=hints)

        # Every batch's _score_synergy_single should have received the hints
        assert len(captured) == 2
        for call_hints in captured:
            assert call_hints == hints

    def test_score_synergy_batch_works_without_hints(self):
        """Backward compat: omitting synergy_hints should still work."""
        llm = LLMEngine(LLMConfig(mock_mode=False, api_key="dummy"))
        llm.client = object()
        llm.config.mock_mode = False

        captured = []
        def fake_single(analysis, cards, synergy_hints=None, class_sink=None):
            captured.append(synergy_hints)
            return {c.name: 50.0 for c in cards}
        llm._score_synergy_single = fake_single

        cards = [_card("A"), _card("B")]
        llm.score_synergy_batch(_analysis(), cards, batch_size=30)
        assert captured == [None]


# ----------------------------------------------------------------------
# User prompt annotation in scoring
# ----------------------------------------------------------------------

class TestScoringPromptAnnotation:
    def test_user_prompt_prefixes_tagged_cards(self):
        """The scoring user prompt should prefix each tagged card with
        its hint tag, so the LLM can use it as a calibration anchor."""
        llm = LLMEngine(LLMConfig(mock_mode=False, api_key="dummy"))
        llm.client = object()
        llm.config.mock_mode = False

        captured_prompt = []
        def fake_call_api(system_prompt, user_prompt, **kwargs):
            captured_prompt.append(user_prompt)
            # Valid JSON response
            return '{"scores": [{"name": "Heliod", "score": 90}]}'

        llm._call_api = fake_call_api

        cards = [_card("Heliod", "Lifelink"), _card("Sol Ring", "Add mana")]
        hints = {"Heliod": "[SYN+++]"}

        llm._score_synergy_single(_analysis(), cards, synergy_hints=hints)

        prompt = captured_prompt[0]
        # Heliod's line should have the tag
        assert "[SYN+++] **Heliod**" in prompt
        # Sol Ring's line should not have any tag
        sol_ring_line = next(
            line for line in prompt.split("\n") if "**Sol Ring**" in line
        )
        assert "[SYN" not in sol_ring_line.split("**")[0].lstrip("- ")

    def test_no_hints_produces_unannotated_prompt(self):
        """Without hints, the prompt should look like the original
        (un-prefixed) format — backward compatibility."""
        llm = LLMEngine(LLMConfig(mock_mode=False, api_key="dummy"))
        llm.client = object()
        llm.config.mock_mode = False

        captured_prompt = []
        def fake_call_api(system_prompt, user_prompt, **kwargs):
            captured_prompt.append(user_prompt)
            return '{"scores": [{"name": "A", "score": 50}]}'
        llm._call_api = fake_call_api

        cards = [_card("A"), _card("B")]
        llm._score_synergy_single(_analysis(), cards, synergy_hints=None)

        for line in captured_prompt[0].split("\n"):
            if "**A**" in line or "**B**" in line:
                # No [SYN tag on un-hinted lines
                assert "[SYN" not in line.split("**")[0]


# ----------------------------------------------------------------------
# End-to-end: deck_builder passes hints to scoring
# ----------------------------------------------------------------------

class TestDeckBuilderPassesHintsToScoring:
    def test_phase_synergy_scoring_uses_hints(self, test_csv_path):
        """When recall has produced source-membership data, the scoring
        phase should pass the computed hints into score_synergy_batch."""
        from mtg_deck_builder.deck_builder import DeckBuilder
        from mtg_deck_builder.models import BuildConfig

        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W", key_mechanics=["lifegain"],
            build_around_text="gain life", evaluation_notes="...",
            category_queries={}, synergy_keywords=["gain life"],
            synergy_patterns=["gain life"],
        )
        config = BuildConfig(
            commander_name=analysis.name, random_seed=42,
            candidates_per_category=5,
            synergy_engine_target=0,  # skip Phase 2 for speed
            # v0.9.4: force the LLM scoring path. In default "auto" mode
            # with embeddings installed, the embedding layer fills all
            # scores and score_synergy_batch is never called.
            synergy_scoring_mode="llm",
        )
        builder = DeckBuilder(
            card_database_path=test_csv_path,
            config=config,
            llm_config=LLMConfig(mock_mode=True),  # mock skips API
        )
        db = builder.db
        builder._commander = db.get_by_name(analysis.name)
        builder._analysis = analysis

        # Inject recall data so hints get computed
        soul_warden = db.get_by_name("Soul Warden")
        assert soul_warden is not None
        builder._edhrec_recall_names = {"Soul Warden"}
        builder._embedding_recall_names = {"Soul Warden"}
        builder._pattern_recall_names = {"Soul Warden"}

        # Spy on score_synergy_batch
        spy_calls = []
        original = builder.llm.score_synergy_batch

        def spy(analysis, cards, batch_size=25, synergy_hints=None,
                class_sink=None):
            spy_calls.append({"synergy_hints": dict(synergy_hints) if synergy_hints else None})
            return original(analysis, cards, batch_size=batch_size,
                            synergy_hints=synergy_hints,
                            class_sink=class_sink)

        builder.llm.score_synergy_batch = spy

        builder._phase_generate_pools()
        builder._phase_llm_filtering()
        # Inject Soul Warden into candidates so the scorer is forced to
        # consider it (the synergy_engine pass is disabled).
        builder._candidates.synergy_engine = [soul_warden]
        builder._phase_synergy_scoring()

        # At least one call should have received the computed hints
        with_hints = [c for c in spy_calls if c["synergy_hints"]]
        assert len(with_hints) > 0, (
            f"_phase_synergy_scoring didn't pass hints to score_synergy_batch; "
            f"got {len(spy_calls)} calls"
        )
        assert with_hints[0]["synergy_hints"].get("Soul Warden") == "[SYN+++]"
