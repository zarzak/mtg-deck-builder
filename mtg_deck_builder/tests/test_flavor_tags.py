"""
Tests for FlavorTagScorer (v0.5 art-tag-based flavor scoring).
"""

import pytest
import time

from mtg_deck_builder.models import Card, Deck
from mtg_deck_builder.scryfall_tags import ScryfallTagClient, TagCacheEntry
from mtg_deck_builder.flavor_tags import FlavorTagScorer


def _seeded_client(tag_to_names: dict[str, list[str]]) -> ScryfallTagClient:
    """Create an offline tag client with the given tag->names mappings."""
    client = ScryfallTagClient(offline=True)
    for tag, names in tag_to_names.items():
        key = ScryfallTagClient._cache_key("art", tag, None)
        client._memory_cache[key] = TagCacheEntry(
            tag=tag, kind="art",
            card_names=list(names),
            fetched_at=time.time(),
        )
    return client


def _card(name: str) -> Card:
    return Card(
        name=name, mana_cost="{G}", mana_value=1,
        card_type="Creature", text="",
        color_identity="G", colors="G",
        power="1", toughness="1",
        types="Creature",
    )


def _commander() -> Card:
    return Card(
        name="Test Commander",
        mana_cost="{2}{G}{W}", mana_value=4,
        card_type="Legendary Creature — Druid",
        text="",
        color_identity="GW", colors="GW",
        power="3", toughness="3",
        types="Creature", subtypes="Druid",
        supertypes="Legendary",
    )


class TestCreateIfConfigured:
    def test_returns_none_when_no_tags(self):
        client = _seeded_client({})
        assert FlavorTagScorer.create_if_configured([], client) is None

    def test_returns_none_when_no_client(self):
        assert FlavorTagScorer.create_if_configured(["mammoth"], None) is None

    def test_returns_scorer_when_both_provided(self):
        client = _seeded_client({"mammoth": ["Card A"]})
        scorer = FlavorTagScorer.create_if_configured(["mammoth"], client)
        assert scorer is not None
        assert scorer.matching_count == 1


class TestPreFetching:
    def test_union_of_multiple_tags(self):
        """Multiple tags should union to a single matching set."""
        client = _seeded_client({
            "forest": ["Elf A", "Elf B", "Tree"],
            "mammoth": ["Tree", "Mammoth X"],  # Tree overlaps
        })
        scorer = FlavorTagScorer(["forest", "mammoth"], client)
        # Union: Elf A, Elf B, Tree, Mammoth X = 4 unique
        assert scorer.matching_count == 4

    def test_empty_tag_skipped(self):
        """Empty strings in the tag list shouldn't crash."""
        client = _seeded_client({"forest": ["A", "B"]})
        scorer = FlavorTagScorer(["forest", "", "  "], client)
        assert scorer.matching_count == 2

    def test_nonexistent_tag_yields_empty_contribution(self):
        """A tag with no matches contributes nothing but doesn't crash."""
        client = _seeded_client({"forest": ["A"]})
        scorer = FlavorTagScorer(["forest", "totally_fake_tag"], client)
        assert scorer.matching_count == 1


class TestScoring:
    def test_zero_matches_returns_neutral(self):
        """If no cards in the universe match, return neutral 50."""
        client = _seeded_client({})
        scorer = FlavorTagScorer(["mammoth"], client)
        deck = Deck(commander=_commander(), cards=[_card(f"C{i}") for i in range(99)])
        assert scorer.score_deck(deck) == 50.0

    def test_empty_deck_returns_neutral(self):
        client = _seeded_client({"forest": ["A", "B"]})
        scorer = FlavorTagScorer(["forest"], client)
        deck = Deck(commander=_commander(), cards=[])
        assert scorer.score_deck(deck) == 50.0

    def test_heavy_match_scores_high(self):
        """Deck with 50% matching cards should score near 95."""
        # 50 cards match, 49 don't
        matching = [f"M{i}" for i in range(50)]
        client = _seeded_client({"forest": matching})

        scorer = FlavorTagScorer(["forest"], client)
        deck = Deck(
            commander=_commander(),
            cards=[_card(n) for n in matching] +
                  [_card(f"N{i}") for i in range(49)],
        )
        score = scorer.score_deck(deck)
        assert score >= 90, f"Expected high score for 50% match, got {score}"

    def test_moderate_match_scores_mid(self):
        """Deck with ~10% matches should be around 60."""
        matching = [f"M{i}" for i in range(10)]
        client = _seeded_client({"forest": matching + [f"Other{i}" for i in range(100)]})

        scorer = FlavorTagScorer(["forest"], client)
        deck = Deck(
            commander=_commander(),
            cards=[_card(n) for n in matching] +
                  [_card(f"N{i}") for i in range(89)],
        )
        score = scorer.score_deck(deck)
        assert 55 <= score <= 70, f"Expected mid score for ~10% match, got {score}"

    def test_zero_deck_matches_scores_low(self):
        """Deck with 0% matches (but matching set non-empty) scores low."""
        client = _seeded_client({"forest": ["Other A", "Other B", "Other C"]})

        scorer = FlavorTagScorer(["forest"], client)
        deck = Deck(
            commander=_commander(),
            cards=[_card(f"N{i}") for i in range(99)],
        )
        score = scorer.score_deck(deck)
        # 0 / 99 matches but universe non-empty -> 40
        assert 38 <= score <= 45

    def test_card_matches_helper(self):
        client = _seeded_client({"forest": ["Mammoth", "Oak"]})
        scorer = FlavorTagScorer(["forest"], client)
        assert scorer.card_matches("Mammoth") is True
        assert scorer.card_matches("Not In Tag") is False


class TestColorIdentityFilter:
    def test_color_identity_passed_through(self):
        """Scorer should pass color_identity to the tag client."""
        client = ScryfallTagClient(offline=True)
        # Seed BOTH the unfiltered and the WG-filtered cache entries
        key_any = ScryfallTagClient._cache_key("art", "mammoth", None)
        key_wg = ScryfallTagClient._cache_key("art", "mammoth", "WG")
        client._memory_cache[key_any] = TagCacheEntry(
            tag="mammoth", kind="art",
            card_names=["Red Mammoth", "White Mammoth"],
            fetched_at=time.time(),
        )
        client._memory_cache[key_wg] = TagCacheEntry(
            tag="mammoth", kind="art",
            card_names=["White Mammoth"],  # subset for WG identity
            fetched_at=time.time(),
        )
        scorer = FlavorTagScorer(["mammoth"], client, color_identity="WG")
        assert scorer.matching_count == 1
        assert scorer.card_matches("White Mammoth")
        assert not scorer.card_matches("Red Mammoth")


class TestIntegrationWithEvaluator:
    """End-to-end: tag scorer plugged into DeckEvaluator."""

    def test_evaluator_takes_max_of_signals(self):
        """Evaluator combines tribal and art-tag scores by taking the max.

        Two scenarios:
        1. Low tribal + high art-tag → max should pick art-tag (high)
        2. High tribal + low art-tag → max should pick tribal (high)
        Both scenarios should yield a high flavor score, proving MAX
        behavior rather than e.g. "use art-tag whenever available."
        """
        from mtg_deck_builder.models import BuildConfig, CommanderAnalysis
        from mtg_deck_builder.deck_evaluator import DeckEvaluator

        analysis = CommanderAnalysis(
            name="Cmdr", color_identity="GW",
            key_mechanics=[], build_around_text="",
            evaluation_notes="", category_queries={},
            synergy_keywords=[],
        )
        config = BuildConfig(commander_name="Cmdr")

        # === Scenario 1: low tribal, high art-tag ===
        unicorn_cmdr = Card(
            name="Unicorn Cmdr",
            mana_cost="{2}{G}{W}", mana_value=4,
            card_type="Legendary Creature — Unicorn",
            text="", color_identity="GW", colors="GW",
            power="2", toughness="2",
            types="Creature", subtypes="Unicorn", supertypes="Legendary",
        )
        matching_names = [f"ArtCard{i}" for i in range(99)]
        tag_client = _seeded_client({"forest": matching_names})
        scorer_high_art = FlavorTagScorer(["forest"], tag_client)
        deck_low_tribal_high_art = Deck(
            commander=unicorn_cmdr,
            cards=[
                Card(
                    name=n, mana_cost="{G}", mana_value=1,
                    card_type="Creature", text="",
                    color_identity="G", colors="G",
                    power="1", toughness="1",
                    types="Creature", subtypes="Elf",  # NOT Unicorn
                )
                for n in matching_names
            ],
        )
        evaluator1 = DeckEvaluator(
            config, analysis, flavor_tag_scorer=scorer_high_art,
        )
        s1 = evaluator1._score_flavor(deck_low_tribal_high_art)
        assert s1 > 90, f"Scenario 1 (low tribal, high art): expected >90, got {s1}"

        # === Scenario 2: high tribal, low art-tag ===
        # Different art tag that matches NOTHING in the deck
        scorer_low_art = FlavorTagScorer(
            ["fire"],
            _seeded_client({"fire": ["RedDragon1", "RedDragon2"]}),  # not in deck
        )
        deck_high_tribal_low_art = Deck(
            commander=unicorn_cmdr,
            cards=[
                Card(
                    name=f"Unicorn{i}", mana_cost="{G}", mana_value=1,
                    card_type="Creature", text="",
                    color_identity="G", colors="G",
                    power="1", toughness="1",
                    types="Creature", subtypes="Unicorn",
                )
                for i in range(99)
            ],
        )
        evaluator2 = DeckEvaluator(
            config, analysis, flavor_tag_scorer=scorer_low_art,
        )
        s2 = evaluator2._score_flavor(deck_high_tribal_low_art)
        # 100% Unicorn -> tribal ~95; art-tag ~40 (zero match but matching set non-empty)
        # Max should pick tribal
        assert s2 > 90, f"Scenario 2 (high tribal, low art): expected >90, got {s2}"

    def test_evaluator_without_scorer_falls_back_to_tribal(self):
        """No scorer = tribal-only flavor (v0.4 behavior)."""
        from mtg_deck_builder.models import BuildConfig, CommanderAnalysis
        from mtg_deck_builder.deck_evaluator import DeckEvaluator

        commander = Card(
            name="X", mana_cost="{G}", mana_value=1,
            card_type="Legendary Creature — Unicorn",
            text="", color_identity="G", colors="G",
            power="1", toughness="1",
            types="Creature", subtypes="Unicorn",
            supertypes="Legendary",
        )
        # All Unicorns -> high tribal
        deck = Deck(
            commander=commander,
            cards=[
                Card(
                    name=f"U{i}", mana_cost="{G}", mana_value=1,
                    card_type="Creature", text="",
                    color_identity="G", colors="G",
                    power="1", toughness="1",
                    types="Creature", subtypes="Unicorn",
                )
                for i in range(99)
            ],
        )
        analysis = CommanderAnalysis(
            name="X", color_identity="G", key_mechanics=[],
            build_around_text="", evaluation_notes="",
            category_queries={}, synergy_keywords=[],
        )
        config = BuildConfig(commander_name="X")
        evaluator = DeckEvaluator(config, analysis)  # no scorer
        flavor = evaluator._score_flavor(deck)
        # Fully Unicorn = high tribal
        assert flavor >= 90

    def test_evaluator_tolerates_scorer_exception(self):
        """If the scorer raises, evaluator must not crash."""
        from mtg_deck_builder.models import BuildConfig, CommanderAnalysis
        from mtg_deck_builder.deck_evaluator import DeckEvaluator

        class BrokenScorer:
            def score_deck(self, deck):
                raise RuntimeError("oops")

        commander = Card(
            name="X", mana_cost="{G}", mana_value=1,
            card_type="Legendary Creature — Elf",
            text="", color_identity="G", colors="G",
            power="1", toughness="1",
            types="Creature", subtypes="Elf", supertypes="Legendary",
        )
        deck = Deck(commander=commander, cards=[_card(f"C{i}") for i in range(99)])
        analysis = CommanderAnalysis(
            name="X", color_identity="G", key_mechanics=[],
            build_around_text="", evaluation_notes="",
            category_queries={}, synergy_keywords=[],
        )
        config = BuildConfig(commander_name="X")
        evaluator = DeckEvaluator(
            config, analysis, flavor_tag_scorer=BrokenScorer(),
        )
        # Should not raise — falls back to tribal
        score = evaluator._score_flavor(deck)
        assert 0 <= score <= 100
