"""
Tests for v0.9.4:
  - Role-pool quality pre-rank (get_cards_for_role sorts by quality)
  - Stickers/Attractions joke-card filter
  - _role_quality_score behavior
  - Embedding+hint synergy scoring boost (_boost_synergy_by_hint)
  - tournament_model override threading through _call_api
"""

import pytest

from mtg_deck_builder.card_database import (
    CardDatabase, _role_quality_score, is_staple, COMMON_STAPLES,
)
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import Card, CommanderAnalysis, BuildConfig
from mtg_deck_builder.llm_engine import LLMEngine, LLMConfig


def _card(name, text="", card_type="Sorcery", mv=2, ci="G",
          types="Sorcery") -> Card:
    return Card(
        name=name, mana_cost="{1}{G}", mana_value=mv,
        card_type=card_type, text=text,
        color_identity=ci, colors=ci,
        power="", toughness="", loyalty="", defense="",
        types=types, subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


# ----------------------------------------------------------------------
# _role_quality_score
# ----------------------------------------------------------------------

class TestRoleQualityScore:
    def test_staple_gets_big_bonus(self):
        # Rampant Growth is in COMMON_STAPLES
        rampant = _card("Rampant Growth",
                        "Search your library for a basic land card, put that "
                        "card onto the battlefield tapped, then shuffle.")
        gatecreeper = _card("Gatecreeper Vine",
                            "Defender. When this creature enters, you may "
                            "search your library for a basic land card or a "
                            "Gate card, reveal it, put it into your hand, "
                            "then shuffle.",
                            card_type="Creature — Plant", types="Creature")
        rs = _role_quality_score(rampant, "ramp")
        gs = _role_quality_score(gatecreeper, "ramp")
        assert rs > gs, (
            f"Rampant Growth ({rs}) should outrank Gatecreeper Vine ({gs})"
        )

    def test_ramp_battlefield_beats_hand(self):
        to_battlefield = _card("Battle Ramp",
                               "Search your library for a basic land card, put "
                               "it onto the battlefield.")
        to_hand = _card("Hand Ramp",
                        "Search your library for a basic land card, put it "
                        "into your hand.")
        assert _role_quality_score(to_battlefield, "ramp") > \
            _role_quality_score(to_hand, "ramp")

    def test_removal_exile_beats_destroy(self):
        exile = _card("Exile Spell", "Exile target creature.",
                      card_type="Instant", types="Instant")
        destroy = _card("Destroy Spell", "Destroy target creature.",
                        card_type="Sorcery", types="Sorcery")
        assert _role_quality_score(exile, "removal") > \
            _role_quality_score(destroy, "removal")

    def test_lower_mana_value_scores_higher(self):
        cheap = _card("Cheap Rock", "{T}: Add {C}{C}.", mv=1,
                      card_type="Artifact", types="Artifact")
        pricey = _card("Pricey Rock", "{T}: Add {C}{C}.", mv=6,
                       card_type="Artifact", types="Artifact")
        assert _role_quality_score(cheap, "ramp") > \
            _role_quality_score(pricey, "ramp")


# ----------------------------------------------------------------------
# get_cards_for_role pre-rank (against the real DB)
# ----------------------------------------------------------------------

class TestRolePoolPrerank:
    @pytest.fixture(scope="class")
    def db(self):
        from pathlib import Path
        csv = Path(__file__).parent.parent.parent / "test_cards.csv"
        if not csv.exists():
            pytest.skip("test CSV not found")
        d = CardDatabase(csv)
        d.load()
        return d

    def test_staples_rank_above_filler_in_role_pool(self, db):
        """If both a staple and filler fill the same role, the staple
        should appear earlier in the returned (sorted) pool."""
        # Use whatever staples are present in the test CSV.
        ramp = db.get_cards_for_role("ramp", "G,W", limit=300)
        names = [c.name for c in ramp]
        # Find any staple present and any non-staple present
        staple_positions = [i for i, n in enumerate(names) if n in COMMON_STAPLES]
        nonstaple_positions = [i for i, n in enumerate(names)
                               if n not in COMMON_STAPLES]
        if staple_positions and nonstaple_positions:
            # The best staple should rank above the worst non-staple
            assert min(staple_positions) < max(nonstaple_positions)

    def test_pool_is_sorted_by_quality_desc(self, db):
        ramp = db.get_cards_for_role("ramp", "G,W", limit=50)
        scores = [_role_quality_score(c, "ramp") for c in ramp]
        # Non-increasing (allowing ties)
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# ----------------------------------------------------------------------
# v0.9.5 de-truncation: unbounded get_cards_for_role
# ----------------------------------------------------------------------

class TestRolePoolDetruncation:
    @pytest.fixture(scope="class")
    def db(self):
        from pathlib import Path
        csv = Path(__file__).parent.parent.parent / "test_cards.csv"
        if not csv.exists():
            pytest.skip("test CSV not found")
        d = CardDatabase(csv)
        d.load()
        return d

    def test_default_returns_every_match(self, db):
        """No limit (default) returns every card that fills the role —
        never fewer than a bounded call."""
        unbounded = db.get_cards_for_role("ramp", "G,W")
        bounded = db.get_cards_for_role("ramp", "G,W", limit=3)
        assert len(unbounded) >= len(bounded)
        # Bounded picks are a subset of the full pool (no card invented/lost).
        assert {c.name for c in bounded}.issubset({c.name for c in unbounded})

    def test_unbounded_has_no_truncation(self, db):
        """The unbounded pool size equals the raw count of role matches —
        nothing is cut. Compare against a deliberately huge explicit limit."""
        unbounded = db.get_cards_for_role("ramp", "G,W")
        huge = db.get_cards_for_role("ramp", "G,W", limit=10_000)
        assert {c.name for c in unbounded} == {c.name for c in huge}

    def test_unbounded_is_name_sorted_not_quality_sorted(self, db):
        """The unbounded path applies NO quality pre-rank — it is sorted by
        name for deterministic tournament chunking. Verify name order."""
        ramp = db.get_cards_for_role("ramp", "G,W")
        names = [c.name for c in ramp]
        assert names == sorted(names), "Unbounded pool must be name-sorted"

class TestJokeCardFilter:
    def test_stickers_card_excluded(self):
        db = CardDatabase.__new__(CardDatabase)
        # Build a minimal card and test _is_valid_card directly
        sticker = _card("Carnival Elephant Meteor", "Sacrifice: Draw two.",
                        card_type="Stickers", types="Stickers")
        normal = _card("Rampant Growth", "Search your library for a land.")
        assert db._is_valid_card(sticker) is False
        assert db._is_valid_card(normal) is True

    def test_attraction_excluded(self):
        db = CardDatabase.__new__(CardDatabase)
        attraction = _card("Some Attraction", "Visit: do a thing.",
                           card_type="Artifact — Attraction",
                           types="Artifact")
        assert db._is_valid_card(attraction) is False


# ----------------------------------------------------------------------
# Embedding+hint synergy boost
# ----------------------------------------------------------------------

class TestSynergyHintBoost:
    """v0.9.6: hint tag remaps the cosine into the tier band [floor, 100]
    via floor + (raw/100)*(100-floor) — a floor that still lets cosine add
    resolution within the tier, instead of the old snap-to-floor."""

    def test_triple_plus_band_floor_respected(self):
        # raw=0 lands exactly on the [SYN+++] floor of 80.
        assert DeckBuilder._boost_synergy_by_hint(0.0, "[SYN+++]") == 80.0

    def test_triple_plus_low_raw_lifts_slightly_above_floor(self):
        # raw=20, floor=80 → 80 + 0.20*20 = 84.0 (not snapped to 80)
        assert DeckBuilder._boost_synergy_by_hint(20.0, "[SYN+++]") == 84.0

    def test_double_plus_blend(self):
        # raw=30, floor=65 → 65 + 0.30*35 = 75.5
        assert DeckBuilder._boost_synergy_by_hint(30.0, "[SYN++]") == 75.5

    def test_single_plus_blend(self):
        # raw=10, floor=50 → 50 + 0.10*50 = 55.0
        assert DeckBuilder._boost_synergy_by_hint(10.0, "[SYN+]") == 55.0

    def test_high_embedding_score_never_reduced(self):
        # raw=92, floor=65 → 65 + 0.92*35 = 97.2; the blend lifts a strong
        # cosine, and never drops it below raw.
        out = DeckBuilder._boost_synergy_by_hint(92.0, "[SYN++]")
        assert out == pytest.approx(97.2)
        assert out >= 92.0

    def test_resolution_within_tier(self):
        # The whole point of the change: two same-tier cards with different
        # cosines must get DIFFERENT scores (old max()-floor snapped both).
        lo = DeckBuilder._boost_synergy_by_hint(20.0, "[SYN++]")
        hi = DeckBuilder._boost_synergy_by_hint(80.0, "[SYN++]")
        assert hi > lo

    def test_untagged_unchanged(self):
        assert DeckBuilder._boost_synergy_by_hint(42.0, None) == 42.0

    def test_clamped_to_100(self):
        assert DeckBuilder._boost_synergy_by_hint(150.0, None) == 100.0


# ----------------------------------------------------------------------
# tournament_model override threading
# ----------------------------------------------------------------------

class _ModelCapturingLLM(LLMEngine):
    """Captures the `model` arg of each _call_api invocation and echoes the
    candidate names back so the tournament progresses to the final pick."""

    def __init__(self):
        super().__init__(LLMConfig(mock_mode=False, api_key="dummy"))
        self.client = object()
        self.config.mock_mode = False
        self.calls: list[dict] = []

    def _call_api(self, system_prompt, user_prompt, temperature=None,
                  max_tokens=None, commander_context=None, model=None):
        import re
        import json
        effective = model or self.config.model
        self.calls.append({"model": effective})
        # Echo back every card name in the prompt; the caller truncates to
        # the requested count, so this lets the tournament shrink normally.
        names = re.findall(r"\*\*([^*]+)\*\*", user_prompt)
        return json.dumps({"names": names})

    @property
    def models_used(self):
        return [c["model"] for c in self.calls]


class TestTournamentModelOverride:
    def _run(self, llm):
        analysis = CommanderAnalysis(
            name="T", color_identity="W", key_mechanics=[],
            build_around_text="", evaluation_notes="",
            category_queries={}, synergy_keywords=[],
        )
        pool = [_card(f"Card{i:04d}", "text", card_type="Creature",
                      types="Creature", ci="W") for i in range(500)]
        return llm._select_cards_batched(analysis, pool, role="synergy", count=50)

    def test_elimination_uses_cheap_model_final_uses_default(self):
        llm = _ModelCapturingLLM()
        llm.config.model = "claude-sonnet-4-6"
        llm.config.tournament_model = "claude-haiku-4-5"
        self._run(llm)

        # Elimination rounds used Haiku
        assert "claude-haiku-4-5" in llm.models_used, (
            f"Expected Haiku elimination calls; got {set(llm.models_used)}"
        )
        # The FINAL call (precision pick) used the default Sonnet model
        assert llm.models_used[-1] == "claude-sonnet-4-6", (
            f"Final pick should use default model; got {llm.models_used[-1]}"
        )

    def test_tournament_model_none_uses_default_everywhere(self):
        llm = _ModelCapturingLLM()
        llm.config.model = "claude-sonnet-4-6"
        llm.config.tournament_model = None
        self._run(llm)
        assert set(llm.models_used) == {"claude-sonnet-4-6"}
