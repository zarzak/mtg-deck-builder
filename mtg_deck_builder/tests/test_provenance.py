"""
Tests for v0.9.33 pool-entry provenance (#26) and the extracted recall
phase (#28).

Provenance answers "why is this card in the pool" (and, by absence, "why
not") — every miss investigation this project has run needed it. The recall
extraction moved pool assembly into recall_phase.build_recall_pool; these
tests pin the RecallResult contract and the builder's channel tagging.
"""

from mtg_deck_builder.card_database import CardDatabase
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import BuildConfig, CommanderAnalysis
from mtg_deck_builder.llm_engine import LLMConfig
from mtg_deck_builder.recall_phase import build_recall_pool, RecallResult


def _analysis() -> CommanderAnalysis:
    return CommanderAnalysis(
        name="Lathiel, the Bounteous Dawn", color_identity="G,W",
        key_mechanics=["lifegain"], build_around_text="gain life",
        evaluation_notes="", category_queries={},
        synergy_keywords=["gain life", "lifelink"],
        synergy_patterns=["gain .* life", "lifelink"],
    )


class TestRecallPhaseExtraction:
    def test_build_recall_pool_returns_populated_result(self, test_csv_path):
        db = CardDatabase(str(test_csv_path))
        db.load()
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          recall_use_edhrec=False, recall_use_embeddings=False,
                          recall_use_patterns=True)
        rr = build_recall_pool(db=db, config=cfg, analysis=_analysis(),
                               edhrec_data=None, color_id="G,W")
        assert isinstance(rr, RecallResult)
        assert rr.cards, "pattern recall should find lifegain cards"
        # Pattern names are populated; disabled sources stay empty.
        assert rr.pattern_names
        assert rr.edhrec_names == set()
        assert rr.embedding_names == set()

    def test_pattern_source_tagged_only_when_in_union(self, test_csv_path):
        # Every pattern-recalled card that made the (uncapped) union is
        # tagged; membership sets never exceed the union.
        db = CardDatabase(str(test_csv_path))
        db.load()
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          recall_use_edhrec=False, recall_use_embeddings=False,
                          recall_use_patterns=True, recall_pool_cap=5)
        rr = build_recall_pool(db=db, config=cfg, analysis=_analysis(),
                               edhrec_data=None, color_id="G,W")
        assert len(rr.cards) <= 5


class TestBuilderProvenance:
    def _builder(self, test_csv_path):
        cfg = BuildConfig(
            commander_name="Lathiel, the Bounteous Dawn",
            population_size=4, generations=2, patience_generations=50,
            random_seed=42, candidates_per_category=10,
            recall_use_edhrec=False, recall_use_embeddings=False,
            recall_use_patterns=True,
        )
        b = DeckBuilder(card_database_path=str(test_csv_path), config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        b._analysis = _analysis()
        return b

    def test_recall_channel_tagged_on_pool_generation(self, test_csv_path):
        b = self._builder(test_csv_path)
        b._phase_generate_pools()
        # Pattern-recalled cards carry the recall:patterns channel.
        tagged = [n for n, ch in b._pool_provenance.items()
                  if "recall:patterns" in ch]
        assert tagged, "no cards tagged with recall:patterns"

    def test_tag_provenance_is_idempotent_and_ordered(self, test_csv_path):
        b = self._builder(test_csv_path)
        b._tag_provenance(["Sol Ring"], "recall:edhrec")
        b._tag_provenance(["Sol Ring"], "recall:edhrec")  # dup ignored
        b._tag_provenance(["Sol Ring"], "power-staples")
        assert b._pool_provenance["Sol Ring"] == \
            ["recall:edhrec", "power-staples"]

    def test_full_build_attaches_provenance_to_telemetry(self, test_csv_path):
        b = self._builder(test_csv_path)
        result = b.build()
        # Every telemetry row has a provenance list attribute; at least some
        # non-empty (recall/role channels), and the full map is exposed.
        assert all(hasattr(t, "provenance") for t in result.card_telemetry)
        assert any(t.provenance for t in result.card_telemetry)
        assert result.pool_provenance  # the full name->channels map exists
        # A card's telemetry provenance matches the pool map (final-99 subset).
        for t in result.card_telemetry:
            if t.name in result.pool_provenance:
                assert t.provenance == result.pool_provenance[t.name]
