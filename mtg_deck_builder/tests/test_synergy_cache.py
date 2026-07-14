"""
Tests for the v0.9.31 per-commander synergy-score cache.

The cache must reuse scores (and effect classes) across runs, while
rescoring honestly when anything that shaped the score changes: card text,
hint tag, or the rubric itself.
"""

from mtg_deck_builder.models import Card, BuildConfig
from mtg_deck_builder.synergy_cache import SynergyScoreCache


def _card(name: str, text: str = "some text") -> Card:
    return Card(
        name=name, mana_cost="{1}", mana_value=1,
        card_type="Creature", text=text,
        color_identity="W", colors="W",
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _cache(tmp_path, rubric="RUBRIC v1") -> SynergyScoreCache:
    return SynergyScoreCache("Test Commander", "claude-sonnet-4-6",
                             rubric=rubric, cache_dir=str(tmp_path))


class TestRoundTrip:
    def test_store_save_reload_lookup(self, tmp_path):
        c1 = _cache(tmp_path)
        cards = [_card("Soul Warden"), _card("Sol Ring")]
        c1.store(cards, {"Soul Warden": 88.0, "Sol Ring": 45.0},
                 effect_classes={"Soul Warden": "lifegain trigger"},
                 hints={"Soul Warden": "[SYN+++]"})
        c1.save()

        c2 = _cache(tmp_path)  # fresh instance = "next run"
        scores, classes, misses = c2.lookup(
            cards, hints={"Soul Warden": "[SYN+++]"})
        assert scores == {"Soul Warden": 88.0, "Sol Ring": 45.0}
        assert classes == {"Soul Warden": "lifegain trigger"}
        assert misses == []

    def test_unscored_cards_never_cached(self, tmp_path):
        # A parse failure (card missing from scores) must not cache a hole.
        c = _cache(tmp_path)
        c.store([_card("A"), _card("B")], {"A": 70.0})
        scores, _, misses = c.lookup([_card("A"), _card("B")])
        assert scores == {"A": 70.0}
        assert [m.name for m in misses] == ["B"]


class TestInvalidation:
    def test_text_change_rescores(self, tmp_path):
        c = _cache(tmp_path)
        c.store([_card("A", text="old wording")], {"A": 60.0})
        _, _, misses = c.lookup([_card("A", text="new wording")])
        assert [m.name for m in misses] == ["A"]

    def test_hint_tag_change_rescores(self, tmp_path):
        # The tag anchors the rubric's score bands — same card under a
        # different tag is a different question.
        c = _cache(tmp_path)
        c.store([_card("A")], {"A": 80.0}, hints={"A": "[SYN+++]"})
        scores, _, misses = c.lookup([_card("A")], hints={"A": "[SYN+]"})
        assert scores == {} and [m.name for m in misses] == ["A"]
        # Same tag -> hit.
        scores, _, misses = c.lookup([_card("A")], hints={"A": "[SYN+++]"})
        assert scores == {"A": 80.0} and misses == []

    def test_rubric_change_discards_file(self, tmp_path):
        c1 = _cache(tmp_path, rubric="RUBRIC v1")
        c1.store([_card("A")], {"A": 55.0})
        c1.save()
        c2 = _cache(tmp_path, rubric="RUBRIC v2 (edited)")
        assert len(c2) == 0

    def test_disabled_cache_dir_is_noop(self):
        c = SynergyScoreCache("X", "m", rubric="r", cache_dir=None)
        c.store([_card("A")], {"A": 50.0})
        c.save()  # must not raise or write anywhere
        scores, _, misses = c.lookup([_card("A")])
        assert scores == {"A": 50.0}  # in-memory still works this run


class TestBuilderGating:
    def test_mock_mode_disables_cache(self, test_csv_path):
        from mtg_deck_builder.deck_builder import DeckBuilder
        from mtg_deck_builder.llm_engine import LLMConfig
        b = DeckBuilder(str(test_csv_path),
                        BuildConfig(commander_name="Lathiel, the Bounteous Dawn"),
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        assert b._get_synergy_cache() is None

    def test_disabled_dir_and_enabled_instance(self, test_csv_path):
        from mtg_deck_builder.deck_builder import DeckBuilder
        from mtg_deck_builder.llm_engine import LLMConfig
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          synergy_cache_dir=None)
        b = DeckBuilder(str(test_csv_path), cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        b.llm.config.mock_mode = False  # pretend live
        assert b._get_synergy_cache() is None  # dir disabled
        b.config.synergy_cache_dir = "./synergy_cache_test_tmp"
        try:
            cache = b._get_synergy_cache()
            assert cache is not None
            assert cache.commander == "Lathiel, the Bounteous Dawn"
            assert b._get_synergy_cache() is cache  # lazy singleton
        finally:
            import shutil
            shutil.rmtree("./synergy_cache_test_tmp", ignore_errors=True)
