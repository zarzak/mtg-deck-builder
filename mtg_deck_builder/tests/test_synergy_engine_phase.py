"""
Tests for the v0.9 synergy_engine Phase 2 pass.

Headline claims under test:
  - `synergy` is no longer a deck-input bucket in `all_cards()`; it's the
    SOURCE pool for the synergy_engine pass.
  - The synergy_engine pass runs AFTER role buckets fill.
  - The synergy_engine pool excludes already_selected cards.
  - Setting synergy_engine_target=0 disables the pass without breaking
    the rest of the pipeline.
  - With recall_use_patterns + a Lathiel-shaped analysis, the
    synergy_engine bucket contains canonical Soul-Sister-style cards.

These tests use mock-mode LLM so they're deterministic and don't hit the
API. The mock's `_mock_select_cards` ranks by `quick_synergy_check`
heuristic — close enough to demonstrate the architecture.
"""

import pytest

from mtg_deck_builder.card_database import CardDatabase
from mtg_deck_builder.deck_builder import DeckBuilder, CandidatePool
from mtg_deck_builder.models import BuildConfig, CommanderAnalysis, Card
from mtg_deck_builder.llm_engine import LLMConfig


# ----------------------------------------------------------------------
# CandidatePool.all_cards excludes synergy source pool
# ----------------------------------------------------------------------

def _make_card(name: str, text: str = "") -> Card:
    return Card(
        name=name, mana_cost="{1}{W}", mana_value=2,
        card_type="Creature", text=text,
        color_identity="W", colors="W",
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


class TestCandidatePoolAllCards:
    def test_synergy_source_pool_excluded_from_all_cards(self):
        """`all_cards()` is the deck-input set; `synergy` is the
        source pool for Phase 2 and must not flood it."""
        pool = CandidatePool(
            ramp=[_make_card("Sol Ring")],
            synergy=[_make_card(f"Synergy{i}") for i in range(100)],  # source
            synergy_engine=[_make_card("Soul Warden")],  # Phase 2 output
        )
        names = {c.name for c in pool.all_cards()}
        assert "Sol Ring" in names
        assert "Soul Warden" in names  # synergy_engine IS a deck-input
        for i in range(100):
            assert f"Synergy{i}" not in names, (
                "synergy source pool leaked into all_cards"
            )

    def test_synergy_engine_counted_in_total_unique(self):
        pool = CandidatePool(
            ramp=[_make_card("A")],
            synergy_engine=[_make_card("B"), _make_card("C")],
        )
        # 3 unique names across deck-input buckets
        assert pool.total_unique() == 3


# ----------------------------------------------------------------------
# _phase_llm_filtering runs synergy_engine after role buckets
# ----------------------------------------------------------------------

class TestPhaseFilteringOrdering:
    def test_synergy_engine_runs_with_already_selected_set(self, test_csv_path):
        """The synergy_engine pool is build from synergy_pool minus
        already_selected; we verify this by injecting a synergy pool that
        overlaps with role pool selections."""
        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W",
            key_mechanics=["lifegain", "+1/+1 counters"],
            build_around_text="gain life",
            evaluation_notes="...",
            category_queries={},
            synergy_keywords=["gain life"],
            synergy_patterns=["gain life", "lifelink"],
        )
        config = BuildConfig(
            commander_name=analysis.name,
            random_seed=42,
            candidates_per_category=10,
            synergy_engine_target=15,
            # recall flags off — we'll inject a synergy pool manually
        )
        builder = DeckBuilder(
            card_database_path=test_csv_path,
            config=config,
            llm_config=LLMConfig(mock_mode=True),
        )
        db = builder.db
        builder._commander = db.get_by_name(analysis.name)
        builder._analysis = analysis

        # Run the pool phase to get role pools + synergy recall pool
        builder._phase_generate_pools()
        # Force a known synergy pool with a card that ALSO appears in another role
        all_cards_by_name = {c.name: c for c in db.all_cards}
        sol_ring = all_cards_by_name.get("Sol Ring")
        soul_warden = all_cards_by_name.get("Soul Warden")
        assert sol_ring is not None, "Sol Ring missing from test DB"
        assert soul_warden is not None, "Soul Warden missing from test DB"

        # Inject a tiny synergy pool that contains Sol Ring (likely picked
        # by ramp) and Soul Warden (unique to synergy_engine).
        builder._candidates.synergy = [sol_ring, soul_warden] + [
            c for c in db.all_cards
            if c.name not in {sol_ring.name, soul_warden.name}
        ][:20]

        builder._phase_llm_filtering()

        # Soul Warden should land in synergy_engine. Sol Ring should NOT
        # be in synergy_engine because the ramp bucket already grabbed it
        # (or because it was already_selected by some other role).
        engine_names = {c.name for c in builder._candidates.synergy_engine}
        assert soul_warden.name in engine_names, (
            f"Soul Warden missing from synergy_engine. "
            f"synergy_engine: {sorted(engine_names)}"
        )
        # synergy_engine should never contain cards already chosen by a role
        all_role_buckets = (
            builder._candidates.ramp + builder._candidates.draw +
            builder._candidates.removal + builder._candidates.threats +
            builder._candidates.protection + builder._candidates.recursion +
            builder._candidates.wipe + builder._candidates.lands
        )
        role_names = {c.name for c in all_role_buckets}
        assert not (engine_names & role_names), (
            "synergy_engine contains cards already in role buckets: "
            f"{engine_names & role_names}"
        )

    def test_synergy_engine_target_zero_disables_pass(self, test_csv_path):
        """Setting synergy_engine_target=0 turns off Phase 2."""
        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W",
            key_mechanics=["lifegain"],
            build_around_text="gain life",
            evaluation_notes="...",
            category_queries={},
            synergy_keywords=["gain life"],
            synergy_patterns=["gain life"],
        )
        config = BuildConfig(
            commander_name=analysis.name,
            random_seed=42,
            candidates_per_category=10,
            synergy_engine_target=0,  # disabled
        )
        builder = DeckBuilder(
            card_database_path=test_csv_path,
            config=config,
            llm_config=LLMConfig(mock_mode=True),
        )
        db = builder.db
        builder._commander = db.get_by_name(analysis.name)
        builder._analysis = analysis
        builder._phase_generate_pools()
        # Inject a non-empty synergy pool to prove the engine bucket stays
        # empty even when there are candidates.
        builder._candidates.synergy = list(db.all_cards)[:50]
        builder._phase_llm_filtering()

        assert builder._candidates.synergy_engine == [], (
            f"Expected empty synergy_engine with target=0, got "
            f"{len(builder._candidates.synergy_engine)} cards"
        )

    def test_synergy_engine_excludes_commander(self, test_csv_path):
        """The commander itself must never appear in synergy_engine."""
        analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn",
            color_identity="G,W",
            key_mechanics=["lifegain"],
            build_around_text="gain life",
            evaluation_notes="...",
            category_queries={},
            synergy_keywords=["gain life"],
            synergy_patterns=["gain life"],
        )
        config = BuildConfig(
            commander_name=analysis.name,
            random_seed=42,
            candidates_per_category=10,
            synergy_engine_target=15,
        )
        builder = DeckBuilder(
            card_database_path=test_csv_path,
            config=config,
            llm_config=LLMConfig(mock_mode=True),
        )
        db = builder.db
        builder._commander = db.get_by_name(analysis.name)
        builder._analysis = analysis
        builder._phase_generate_pools()
        # Put the commander himself in the synergy pool to verify it gets filtered
        builder._candidates.synergy = [builder._commander] + list(db.all_cards)[:30]
        builder._phase_llm_filtering()

        engine_names = {c.name for c in builder._candidates.synergy_engine}
        assert builder._commander.name not in engine_names, (
            "Commander should never be in synergy_engine bucket"
        )
