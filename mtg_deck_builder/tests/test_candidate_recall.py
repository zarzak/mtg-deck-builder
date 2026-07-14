"""Tests for candidate_recall — the layered synergy-pool sources."""

from dataclasses import dataclass
from typing import Optional

import pytest

from mtg_deck_builder.models import Card, CommanderAnalysis
from mtg_deck_builder.candidate_recall import (
    _normalize_card_text,
    recall_from_edhrec,
    recall_from_embeddings,
    recall_from_patterns,
    union_candidates,
)


def _make_card(
    name: str,
    text: str = "",
    color_identity: str = "W",
    card_type: str = "Creature",
    mana_cost: str = "{1}{W}",
) -> Card:
    """Construct a frozen Card with sensible defaults."""
    return Card(
        name=name,
        mana_cost=mana_cost,
        mana_value=2,
        card_type=card_type,
        text=text,
        color_identity=color_identity,
        colors=color_identity,
        power="1",
        toughness="1",
        loyalty="",
        defense="",
        types="Creature",
        subtypes="",
        supertypes="",
        keywords="",
        layout="normal",
        legalities="commander:legal",
    )


# ----------------------------------------------------------------------
# _normalize_card_text
# ----------------------------------------------------------------------

class TestNormalizeCardText:
    def test_strips_standalone_digits(self):
        assert _normalize_card_text("you gain 1 life") == "you gain life"

    def test_strips_standalone_x(self):
        assert _normalize_card_text("Pay X life. Gain X life.") == \
            "pay life. gain life."

    def test_lowercases(self):
        assert _normalize_card_text("LIFELINK") == "lifelink"

    def test_collapses_whitespace(self):
        assert _normalize_card_text("a   b\tc\nd") == "a b c d"

    def test_preserves_slash_in_pt(self):
        # "1/1" → "/", which keeps the / between adjacent tokens
        assert _normalize_card_text("Create a 1/1 white Soldier creature token") == \
            "create a / white soldier creature token"

    def test_handles_none(self):
        assert _normalize_card_text(None) == ""

    def test_handles_empty(self):
        assert _normalize_card_text("") == ""

    def test_does_not_strip_word_internal_digits(self):
        # "+1/+1" is NOT a standalone digit — keep it. \b\d+\b only matches
        # whole-token digits surrounded by non-word characters.
        # The "+" before "1" is non-word, but "/" after the "1" is also
        # non-word, so "+1" matches \b\d+\b at "1". Same for the second 1.
        # End result: "+/+ counter".
        assert _normalize_card_text("+1/+1 counter") == "+/+ counter"


# ----------------------------------------------------------------------
# recall_from_patterns
# ----------------------------------------------------------------------

class TestRecallFromPatterns:
    def test_smoking_gun_soul_warden(self):
        """The headline failure: 'gain life' must catch 'gain 1 life'."""
        cards = [
            _make_card("Soul Warden",
                       "Whenever another creature enters, you gain 1 life."),
            _make_card("Lightning Bolt",
                       "Lightning Bolt deals 3 damage to any target.",
                       card_type="Instant"),
        ]
        out = recall_from_patterns(["gain life"], cards)
        assert [c.name for c in out] == ["Soul Warden"]

    def test_lifelink_keyword_match(self):
        cards = [
            _make_card("Heliod, Sun-Crowned",
                       "Lifelink. Whenever you gain life, put a +1/+1 counter."),
            _make_card("Vanilla Bear", "", card_type="Creature"),
        ]
        out = recall_from_patterns(["lifelink"], cards)
        assert [c.name for c in out] == ["Heliod, Sun-Crowned"]

    def test_creature_token_with_typeline_infill(self):
        """'creature token' should match the typical 'create a 1/1 X creature token' phrasing."""
        cards = [
            _make_card("Spectral Procession",
                       "Create three 1/1 white Spirit creature tokens with flying.",
                       card_type="Sorcery"),
            _make_card("Lightning Bolt",
                       "Lightning Bolt deals 3 damage to any target.",
                       card_type="Instant"),
        ]
        out = recall_from_patterns(["creature token"], cards)
        assert [c.name for c in out] == ["Spectral Procession"]

    def test_empty_patterns(self):
        cards = [_make_card("Foo", "anything")]
        assert recall_from_patterns([], cards) == []

    def test_empty_cards(self):
        assert recall_from_patterns(["gain life"], []) == []

    def test_dedupes_via_pool_order(self):
        """A card matching multiple patterns appears only once."""
        cards = [
            _make_card("Heliod",
                       "Lifelink. Whenever you gain life, put a +1/+1 counter."),
        ]
        out = recall_from_patterns(["gain life", "lifelink", "+1/+1 counter"], cards)
        assert [c.name for c in out] == ["Heliod"]

    def test_limit_caps_results(self):
        cards = [_make_card(f"Card {i}", "you gain 1 life") for i in range(20)]
        out = recall_from_patterns(["gain life"], cards, limit=5)
        assert len(out) == 5

    def test_ignores_blank_patterns(self):
        cards = [_make_card("Foo", "you gain 1 life")]
        # Blank/whitespace patterns must not match everything
        out = recall_from_patterns(["", "  ", "\t"], cards)
        assert out == []

    def test_x_normalization(self):
        # "gain X life" → "gain  life" → "gain life" after whitespace
        # collapse, so pattern "gain life" matches.
        # Note: "gains life" / "gained life" deliberately do NOT match —
        # morphology is the embedding layer's job, not patterns'.
        cards = [
            _make_card("Aetherflux Reservoir",
                       "you may pay 50 life. If you do, you gain X life."),
        ]
        out = recall_from_patterns(["gain life"], cards)
        assert [c.name for c in out] == ["Aetherflux Reservoir"]


# ----------------------------------------------------------------------
# recall_from_edhrec
# ----------------------------------------------------------------------

@dataclass
class _FakeEDHRECCard:
    name: str
    synergy: float

    def to_synergy_score(self) -> float:
        return 50.0 + self.synergy * 30.0


@dataclass
class _FakeEDHRECData:
    cards: dict

    def get_high_synergy_cards(self, min_synergy: float = 0.1):
        out = [c for c in self.cards.values() if c.synergy >= min_synergy]
        return sorted(out, key=lambda c: c.synergy, reverse=True)


class _FakeDB:
    def __init__(self, cards):
        self._by_name = {c.name: c for c in cards}

    def get_by_name(self, name: str) -> Optional[Card]:
        return self._by_name.get(name)


class TestRecallFromEDHREC:
    def test_returns_empty_when_no_data(self):
        out = recall_from_edhrec(
            edhrec_data=None,
            db=_FakeDB([]),
            commander_color_identity="G,W",
            limit=300,
        )
        assert out == []

    def test_returns_high_synergy_cards_sorted(self):
        soul_warden = _make_card("Soul Warden", "you gain 1 life")
        archangel = _make_card("Archangel of Thune", "Lifelink. ...")
        sol_ring = _make_card("Sol Ring", "Add 2 mana", color_identity="",
                              card_type="Artifact")
        db = _FakeDB([soul_warden, archangel, sol_ring])
        data = _FakeEDHRECData(cards={
            "Soul Warden": _FakeEDHRECCard("Soul Warden", 0.45),
            "Archangel of Thune": _FakeEDHRECCard("Archangel of Thune", 0.62),
            "Sol Ring": _FakeEDHRECCard("Sol Ring", 0.02),  # Below min_synergy
        })

        out = recall_from_edhrec(
            edhrec_data=data,
            db=db,
            commander_color_identity="G,W",
            limit=300,
            min_synergy=0.1,
        )

        # Sol Ring filtered out by min_synergy. Order: archangel (0.62) > soul_warden (0.45)
        assert [c.name for c in out] == ["Archangel of Thune", "Soul Warden"]

    def test_default_floor_keeps_zero_synergy_drops_negative(self):
        """v0.9.20: the default floor is 0.0 — universal staples sit at
        exactly 0.00 synergy (inclusion-minus-baseline self-cancels; Sol
        Ring was +0.00 in 24,951 Jodah decks) and popular commanders
        compress the whole distribution, so page membership is the signal.
        Only NEGATIVE synergy (actively underplayed) is excluded."""
        staple = _make_card("Sol Ring", "Add 2 mana", color_identity="",
                            card_type="Artifact")
        bad_fit = _make_card("Fog", "Prevent all combat damage",
                             color_identity="G", card_type="Instant")
        db = _FakeDB([staple, bad_fit])
        data = _FakeEDHRECData(cards={
            "Sol Ring": _FakeEDHRECCard("Sol Ring", 0.0),
            "Fog": _FakeEDHRECCard("Fog", -0.05),
        })
        out = recall_from_edhrec(
            edhrec_data=data, db=db,
            commander_color_identity="G,W", limit=300,
        )
        assert [c.name for c in out] == ["Sol Ring"]

    def test_filters_off_color_cards(self):
        """A card outside the commander's color identity is dropped."""
        red_card = _make_card("Lightning Bolt", "deals damage",
                              color_identity="R", card_type="Instant")
        db = _FakeDB([red_card])
        data = _FakeEDHRECData(cards={
            "Lightning Bolt": _FakeEDHRECCard("Lightning Bolt", 0.5),
        })

        out = recall_from_edhrec(
            edhrec_data=data,
            db=db,
            commander_color_identity="G,W",  # No red
            limit=300,
        )
        assert out == []

    def test_skips_cards_not_in_local_db(self):
        db = _FakeDB([])  # Empty database
        data = _FakeEDHRECData(cards={
            "Soul Warden": _FakeEDHRECCard("Soul Warden", 0.45),
        })

        out = recall_from_edhrec(
            edhrec_data=data,
            db=db,
            commander_color_identity="G,W",
            limit=300,
        )
        assert out == []

    def test_caps_at_limit(self):
        cards = [_make_card(f"Card {i}", "...") for i in range(10)]
        db = _FakeDB(cards)
        data = _FakeEDHRECData(cards={
            c.name: _FakeEDHRECCard(c.name, 0.5) for c in cards
        })

        out = recall_from_edhrec(
            edhrec_data=data,
            db=db,
            commander_color_identity="W",
            limit=3,
        )
        assert len(out) == 3


# ----------------------------------------------------------------------
# recall_from_embeddings
# ----------------------------------------------------------------------

# Reuse the existing FakeModel from test_embedding_scorer — same shape
# (single string → 1D, list → 2D) and "synergy axis" semantics that
# properly model "this card is on-strategy" vs "this card is irrelevant".
# Importing rather than redefining keeps the two test files in sync.
from mtg_deck_builder.tests.test_embedding_scorer import FakeModel as _FakeEmbeddingModel


class TestRecallFromEmbeddings:
    def test_returns_top_n_by_similarity(self, monkeypatch):
        """Cards mentioning synergy keywords rank above ones that don't."""
        from mtg_deck_builder import embedding_scorer as es
        from mtg_deck_builder.embedding_scorer import EmbeddingSynergyScorer

        # Patch availability and the model factory so we don't load a real
        # 25MB sentence-transformers checkpoint.
        monkeypatch.setattr(es, "is_embeddings_available", lambda: True)
        monkeypatch.setattr(
            EmbeddingSynergyScorer,
            "create_if_available",
            classmethod(lambda cls, analysis, config=None: cls(
                analysis,
                config=config,
                model=_FakeEmbeddingModel(
                    synergy_keywords=["gain life", "lifelink", "+1/+1 counter"],
                ),
            )),
        )

        analysis = CommanderAnalysis(
            name="Lifegain Commander",
            color_identity="W",
            key_mechanics=["lifegain"],
            build_around_text="gain life lifelink",
            evaluation_notes="lifegain matters",
            category_queries={},
            synergy_keywords=["lifelink", "gain life"],
        )

        # FakeModel does literal substring checks (no digit-normalization),
        # so use texts that contain the keywords as bare phrases. The real
        # sentence-transformers model would handle morphology semantically;
        # we validate the recall PIPELINE here, not embedding quality.
        cards = [
            _make_card("Soul Sister",
                       "Whenever a creature enters, you gain life."),
            _make_card("Heliod",
                       "Lifelink. +1/+1 counter on target creature."),
            _make_card("Vanilla Bear", "Trample.", color_identity="W"),
            _make_card("Plain Goblin", "Haste.", color_identity="W"),
        ]

        out = recall_from_embeddings(
            analysis=analysis, cards=cards, limit=2, cache_dir=None,
        )

        # Synergy cards rank above non-synergy cards.
        assert len(out) == 2
        names = {c.name for c in out}
        assert "Soul Sister" in names
        assert "Heliod" in names
        assert "Vanilla Bear" not in names
        assert "Plain Goblin" not in names

    def test_returns_empty_when_library_missing(self, monkeypatch):
        from mtg_deck_builder import embedding_scorer as es
        monkeypatch.setattr(es, "is_embeddings_available", lambda: False)

        analysis = CommanderAnalysis(
            name="X", color_identity="W", key_mechanics=[],
            build_around_text="", evaluation_notes="",
            category_queries={}, synergy_keywords=[],
        )
        out = recall_from_embeddings(
            analysis=analysis,
            cards=[_make_card("Foo")],
            limit=10,
            cache_dir=None,
        )
        assert out == []

    def test_returns_empty_for_empty_pool(self):
        analysis = CommanderAnalysis(
            name="X", color_identity="W", key_mechanics=[],
            build_around_text="", evaluation_notes="",
            category_queries={}, synergy_keywords=[],
        )
        out = recall_from_embeddings(
            analysis=analysis,
            cards=[],
            limit=10,
            cache_dir=None,
        )
        assert out == []


# ----------------------------------------------------------------------
# Embedding cache: cross-commander reuse
# ----------------------------------------------------------------------

class TestEmbeddingCacheCrossCommander:
    """
    Validates the answer to "what happens to embeddings when running
    against a non-Lathiel commander". Critical property: shared cards
    are embedded once, then reused. Only cards new to the second build
    pay the embedding cost.
    """

    def _patched_factory(self, monkeypatch, encode_counter):
        """Return a factory that creates a scorer using a counting fake model."""
        from mtg_deck_builder import embedding_scorer as es
        from mtg_deck_builder.embedding_scorer import EmbeddingSynergyScorer

        monkeypatch.setattr(es, "is_embeddings_available", lambda: True)

        class CountingFakeModel:
            def __init__(self, base):
                self._base = base

            def encode(self, texts, convert_to_numpy=True):
                # Count the number of TEXTS encoded (not calls), so we can
                # verify "second commander only embeds the new cards".
                if isinstance(texts, str):
                    encode_counter["texts"] += 1
                else:
                    encode_counter["texts"] += len(texts)
                encode_counter["calls"] += 1
                return self._base.encode(texts, convert_to_numpy=convert_to_numpy)

        # Reuse the well-tested FakeModel as the underlying behavior; just
        # wrap it to count.
        base = _FakeEmbeddingModel(synergy_keywords=["lifelink", "gain life"])

        monkeypatch.setattr(
            EmbeddingSynergyScorer,
            "create_if_available",
            classmethod(lambda cls, analysis, config=None: cls(
                analysis,
                config=config,
                model=CountingFakeModel(base),
            )),
        )

    def test_second_commander_only_embeds_new_cards(self, monkeypatch, tmp_path):
        encode_counter = {"texts": 0, "calls": 0}
        self._patched_factory(monkeypatch, encode_counter)

        # Shared cards (would be in either commander's color identity)
        shared = [
            _make_card("Soul Sister", "Whenever a creature enters, you gain life."),
            _make_card("Heliod", "Lifelink. +1/+1 counter."),
        ]
        # Lathiel-only cards (G,W)
        lathiel_only = [_make_card("Forest Ranger", "vanilla", color_identity="W")]
        # Atraxa-only cards (WUBG) that wouldn't appear in Lathiel's pool
        atraxa_only = [_make_card("Phyrexian Devourer", "exile cards", color_identity="W")]

        analysis_lathiel = CommanderAnalysis(
            name="Lathiel", color_identity="G,W", key_mechanics=["lifegain"],
            build_around_text="lifegain", evaluation_notes="",
            category_queries={}, synergy_keywords=["lifelink", "gain life"],
        )
        analysis_atraxa = CommanderAnalysis(
            name="Atraxa", color_identity="W,U,B,G", key_mechanics=["counters"],
            build_around_text="proliferate", evaluation_notes="",
            category_queries={}, synergy_keywords=["lifelink", "gain life"],
        )

        cache_dir = str(tmp_path / "emb_cache")

        # First build: Lathiel sees shared + lathiel_only (3 cards)
        recall_from_embeddings(
            analysis=analysis_lathiel,
            cards=shared + lathiel_only,
            limit=100,
            cache_dir=cache_dir,
        )
        first_count = encode_counter["texts"]
        # Encoded the commander query plus 3 cards = at least 3 texts.
        # (The commander query is encoded inside scorer init.)
        assert first_count >= 3

        # Reset counter and run Atraxa with shared + atraxa_only
        encode_counter["texts"] = 0
        encode_counter["calls"] = 0

        recall_from_embeddings(
            analysis=analysis_atraxa,
            cards=shared + atraxa_only,
            limit=100,
            cache_dir=cache_dir,
        )

        # Atraxa encoded its own commander query (1) plus only Phyrexian
        # Devourer (1 new card). The 2 shared cards were served from cache.
        # Total card-text encodes should be 1 (just the new card),
        # since the commander-query encode happens in a separate call
        # before _load_or_compute_card_embeddings.
        # We verify by checking the per-card encode count is exactly 1.
        # The commander encode is a single string call (counted as 1 text).
        assert encode_counter["texts"] <= 2, (
            f"Expected at most 2 encoded texts (1 commander query + "
            f"1 new card 'Phyrexian Devourer'), got {encode_counter['texts']}. "
            f"Cache is not reusing shared embeddings across commanders."
        )

    def test_cache_invalidates_when_card_text_changes(self, monkeypatch, tmp_path):
        """If a card's text changes (DB update), only that card is re-embedded."""
        encode_counter = {"texts": 0, "calls": 0}
        self._patched_factory(monkeypatch, encode_counter)

        analysis = CommanderAnalysis(
            name="X", color_identity="W", key_mechanics=[],
            build_around_text="", evaluation_notes="",
            category_queries={}, synergy_keywords=[],
        )
        cache_dir = str(tmp_path / "emb_cache")

        # First build with original text
        cards_v1 = [
            _make_card("Card A", "original text v1"),
            _make_card("Card B", "stable text"),
        ]
        recall_from_embeddings(
            analysis=analysis, cards=cards_v1, limit=10, cache_dir=cache_dir,
        )
        encode_counter["texts"] = 0

        # Second build: Card A's text changed; Card B unchanged
        cards_v2 = [
            _make_card("Card A", "different text v2"),
            _make_card("Card B", "stable text"),
        ]
        recall_from_embeddings(
            analysis=analysis, cards=cards_v2, limit=10, cache_dir=cache_dir,
        )

        # Should have encoded only the commander query (1 string) plus
        # Card A (1 text). Card B was reused from cache.
        assert encode_counter["texts"] <= 2, (
            f"Expected ≤2 encoded texts (commander + Card A), "
            f"got {encode_counter['texts']}. Stale entry not invalidated."
        )


# ----------------------------------------------------------------------
# union_candidates
# ----------------------------------------------------------------------

class TestUnionCandidates:
    def test_dedupes_across_sources(self):
        a = _make_card("A")
        b = _make_card("B")
        c = _make_card("C")

        out = union_candidates([a, b], [b, c], [a, c])
        # First-seen wins; a and b come from source 1, c from source 2
        assert [card.name for card in out] == ["A", "B", "C"]

    def test_priority_order_preserved(self):
        a, b, c, d = _make_card("A"), _make_card("B"), _make_card("C"), _make_card("D")

        # Source 1 (EDHREC priority) goes first
        out = union_candidates([a, b], [c, d])
        assert [card.name for card in out] == ["A", "B", "C", "D"]

    def test_caps_at_limit(self):
        cards = [_make_card(f"Card {i}") for i in range(10)]
        out = union_candidates(cards, [], cap=3)
        assert len(out) == 3

    def test_skips_empty_sources(self):
        a = _make_card("A")
        out = union_candidates([], [a], [])
        assert [card.name for card in out] == ["A"]

    def test_empty_input(self):
        assert union_candidates() == []
        assert union_candidates([], [], []) == []
