"""
Tests for the v0.9.12/v0.9.13 EDHREC synergy FLOOR.

EDHREC floors a card's synergy to (edhrec_floor * distinctive score), where
the distinctive score maps the RAW EDHREC synergy metric (0..1) onto 50..100.
Key semantics (v0.9.13 review fixes):

  - Only a POSITIVE raw synergy triggers the floor. A card that merely appears
    in EDHREC data (synergy 0 or missing) has no distinctive signal, and a
    NEGATIVE synergy (community avoids the card here) must never boost.
  - The inclusion-rate fallback is never used — inclusion is the
    precon/popularity-biased metric this project excludes by design.
  - Boost-only (never lowers), so pricey/unpopular cards are protected.
  - No-op when the factor is 0 or no EDHREC data was fetched.
"""

import pytest

from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import BuildConfig
from mtg_deck_builder.llm_engine import LLMConfig


class _FakeEntry:
    def __init__(self, synergy=None, inclusion_rate=None):
        self.synergy = synergy
        self.inclusion_rate = inclusion_rate


class _FakeEdhrec:
    def __init__(self, mapping):
        self.cards = dict(mapping)


def _builder(test_csv_path, floor=0.75) -> DeckBuilder:
    cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                      random_seed=42, edhrec_floor=floor)
    return DeckBuilder(card_database_path=test_csv_path, config=cfg,
                       llm_config=LLMConfig(mock_mode=True))


class TestEdhrecFloor:
    def test_surfaces_undervalued_staple(self, test_csv_path):
        # A highly distinctive card (raw synergy +0.8 → distinctive 90) is
        # floored to 0.75*90 = 67.5, lifting it into contention.
        b = _builder(test_csv_path, floor=0.75)
        b._edhrec_data = _FakeEdhrec({"Skirk Prospector": _FakeEntry(synergy=0.8)})
        syn = {"Skirk Prospector": 40.0}
        b._apply_edhrec_floor(syn)
        assert syn["Skirk Prospector"] == pytest.approx(67.5)

    def test_boost_only_never_lowers(self, test_csv_path):
        # A distinctive card whose reasoned score is already higher stays put —
        # protects pricey/unpopular cards EDHREC under-represents.
        b = _builder(test_csv_path, floor=0.75)
        b._edhrec_data = _FakeEdhrec({"Pricey Bomb": _FakeEntry(synergy=0.9)})
        syn = {"Pricey Bomb": 90.0}
        b._apply_edhrec_floor(syn)
        assert syn["Pricey Bomb"] == 90.0  # 0.75*95=71.25 < 90 -> untouched

    def test_zero_synergy_never_floored(self, test_csv_path):
        # Regression: a card EDHREC is NEUTRAL on (synergy 0 — present in the
        # data but with no distinctive signal) must not be floored at all. The
        # old to_synergy_score mapping floored these to 37.5, smearing the
        # reasoned rubric's low bands for every EDHREC-known card.
        b = _builder(test_csv_path, floor=0.75)
        b._edhrec_data = _FakeEdhrec({"Sol Ring": _FakeEntry(synergy=0.0)})
        syn = {"Sol Ring": 12.0}
        b._apply_edhrec_floor(syn)
        assert syn["Sol Ring"] == 12.0

    def test_negative_synergy_never_floored(self, test_csv_path):
        # Community actively avoids the card under this commander — that must
        # never become a boost.
        b = _builder(test_csv_path, floor=0.75)
        b._edhrec_data = _FakeEdhrec({"Bad Fit": _FakeEntry(synergy=-0.5)})
        syn = {"Bad Fit": 5.0}
        b._apply_edhrec_floor(syn)
        assert syn["Bad Fit"] == 5.0

    def test_inclusion_rate_never_used(self, test_csv_path):
        # The precon-biased inclusion metric must not leak into the floor even
        # when the synergy metric is missing.
        b = _builder(test_csv_path, floor=0.75)
        b._edhrec_data = _FakeEdhrec({
            "Precon Filler": _FakeEntry(synergy=None, inclusion_rate=0.95),
        })
        syn = {"Precon Filler": 10.0}
        b._apply_edhrec_floor(syn)
        assert syn["Precon Filler"] == 10.0

    def test_synergy_capped_at_one(self, test_csv_path):
        # Raw synergy occasionally exceeds 1.0 — the distinctive mapping caps
        # at 100 so the floor tops out at factor*100.
        b = _builder(test_csv_path, floor=0.75)
        b._edhrec_data = _FakeEdhrec({"Signature Card": _FakeEntry(synergy=2.0)})
        syn = {"Signature Card": 10.0}
        b._apply_edhrec_floor(syn)
        assert syn["Signature Card"] == pytest.approx(75.0)

    def test_only_known_cards_touched(self, test_csv_path):
        b = _builder(test_csv_path, floor=0.75)
        b._edhrec_data = _FakeEdhrec({"Known": _FakeEntry(synergy=0.8)})
        syn = {"Known": 10.0, "Unknown": 10.0}
        b._apply_edhrec_floor(syn)
        assert syn["Known"] == pytest.approx(67.5)
        assert syn["Unknown"] == 10.0  # not in EDHREC -> untouched

    def test_factor_zero_noop(self, test_csv_path):
        b = _builder(test_csv_path, floor=0.0)
        b._edhrec_data = _FakeEdhrec({"X": _FakeEntry(synergy=0.9)})
        syn = {"X": 10.0}
        b._apply_edhrec_floor(syn)
        assert syn["X"] == 10.0

    def test_no_edhrec_data_noop(self, test_csv_path):
        b = _builder(test_csv_path, floor=0.75)
        b._edhrec_data = None  # new/unpopular commander
        syn = {"X": 10.0}
        b._apply_edhrec_floor(syn)
        assert syn["X"] == 10.0
