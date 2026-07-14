"""
Tests for oracle_validation (v0.6).
All offline — stub out tag_client methods.
"""

import pytest
import time

from mtg_deck_builder.models import Card, Deck
from mtg_deck_builder.scryfall_tags import ScryfallTagClient, TagCacheEntry
from mtg_deck_builder.oracle_validation import (
    validate_roles, format_role_report,
    RoleDisagreement, ValidationReport, ROLE_TO_ORACLE_TAGS,
)


def _make_card(
    name: str,
    text: str = "",
    card_type: str = "Creature",
    types: str = "Creature",
) -> Card:
    return Card(
        name=name, mana_cost="{G}", mana_value=1,
        card_type=card_type, text=text,
        color_identity="G", colors="G",
        power="1", toughness="1", types=types,
    )


def _commander() -> Card:
    return Card(
        name="Cmdr",
        mana_cost="{2}{G}", mana_value=3,
        card_type="Legendary Creature",
        text="", color_identity="G", colors="G",
        power="2", toughness="2",
        types="Creature", supertypes="Legendary",
    )


def _seed_tag_client(oracle_tag_to_names: dict[str, list[str]]) -> ScryfallTagClient:
    """Create a tag client with seeded oracle-tag cache entries."""
    client = ScryfallTagClient(offline=True)
    for tag, names in oracle_tag_to_names.items():
        key = ScryfallTagClient._cache_key("oracle", tag, None)
        client._memory_cache[key] = TagCacheEntry(
            tag=tag, kind="oracle",
            card_names=list(names),
            fetched_at=time.time(),
        )
    return client


class TestRoleMapping:
    def test_has_mappings_for_core_roles(self):
        for role in ("ramp", "draw", "removal", "wipe"):
            assert role in ROLE_TO_ORACLE_TAGS
            assert ROLE_TO_ORACLE_TAGS[role]

    def test_unknown_role_in_report_is_skipped(self):
        tag_client = _seed_tag_client({})
        deck = Deck(commander=_commander(), cards=[_make_card("A")])
        report = validate_roles(
            deck, tag_client, roles_to_check=["nonexistent_role"]
        )
        assert "nonexistent_role" in report.skipped_roles
        assert report.disagreements == []


class TestValidationBasics:
    def test_empty_disagreements_when_everything_matches(self):
        """If regex says the card is ramp AND oracle tag agrees, no disagreement."""
        deck = Deck(
            commander=_commander(),
            cards=[
                _make_card("Sol Ring", text="Add {C}{C}"),  # regex: ramp
            ],
        )
        tag_client = _seed_tag_client({
            "mana-ramp": ["Sol Ring"],
        })
        report = validate_roles(
            deck, tag_client, roles_to_check=["ramp"],
        )
        assert report.total_disagreements == 0

    def test_missed_disagreement(self):
        """Oracle tag flags a card as ramp; our regex didn't."""
        # Card with text that doesn't match ramp regex patterns
        deck = Deck(
            commander=_commander(),
            cards=[
                _make_card("Weird Ramp", text="Obscure ramp effect"),
            ],
        )
        tag_client = _seed_tag_client({"mana-ramp": ["Weird Ramp"]})
        report = validate_roles(
            deck, tag_client, roles_to_check=["ramp"],
        )
        # Oracle tag flagged it, regex didn't → "missed"
        missed = [d for d in report.disagreements if d.kind == "missed"]
        assert len(missed) == 1
        assert missed[0].card_name == "Weird Ramp"
        assert missed[0].role == "ramp"

    def test_extra_disagreement(self):
        """Our regex flags a card as ramp; oracle tags didn't."""
        deck = Deck(
            commander=_commander(),
            cards=[
                _make_card("Fake Ramp", text="Add {G}"),  # regex thinks ramp
            ],
        )
        # Oracle tags have no ramp cards at all
        tag_client = _seed_tag_client({"mana-ramp": ["Unrelated Card"]})
        report = validate_roles(
            deck, tag_client, roles_to_check=["ramp"],
        )
        extras = [d for d in report.disagreements if d.kind == "extra"]
        assert len(extras) == 1
        assert extras[0].card_name == "Fake Ramp"

    def test_per_role_summary(self):
        deck = Deck(
            commander=_commander(),
            cards=[
                _make_card("Ramp Card", text="Add {G}"),  # regex: ramp
                _make_card("Weird", text="nothing"),
            ],
        )
        tag_client = _seed_tag_client({
            "mana-ramp": ["Weird"],  # only flags the "weird" one
        })
        report = validate_roles(
            deck, tag_client, roles_to_check=["ramp"],
        )
        missed, extra = report.per_role_summary["ramp"]
        # Ramp Card: regex says yes, tags say no → extra
        # Weird: regex says no, tags say yes → missed
        assert missed == 1
        assert extra == 1


class TestSkippedRoles:
    def test_role_with_no_tag_data_is_skipped(self):
        """If all queries return empty, role is skipped (not reported as all-missed)."""
        deck = Deck(
            commander=_commander(),
            cards=[_make_card("A"), _make_card("B")],
        )
        # Tag client has NO oracle-tag entries for removal at all
        tag_client = _seed_tag_client({})
        report = validate_roles(
            deck, tag_client, roles_to_check=["removal"],
        )
        assert "removal" in report.skipped_roles
        assert report.disagreements == []

    def test_raising_tag_client_doesnt_crash(self):
        """A tag client that raises on query shouldn't crash validation."""
        class RaisingClient:
            def get_cards_with_oracle_tag(self, tag, color_identity=None):
                raise RuntimeError("network down")

        deck = Deck(
            commander=_commander(),
            cards=[_make_card("A", text="Add {G}")],
        )
        # Should not raise
        report = validate_roles(
            deck, RaisingClient(), roles_to_check=["ramp"],
        )
        # All queries raised → no successful fetch → role is skipped
        assert "ramp" in report.skipped_roles

    def test_partial_raising_doesnt_skip_role_if_some_tag_works(self):
        """If one tag raises but another succeeds, the role is still validated."""
        class PartialClient:
            def __init__(self):
                self.calls = 0
            def get_cards_with_oracle_tag(self, tag, color_identity=None):
                self.calls += 1
                if tag == "mana-ramp":
                    raise RuntimeError("flaky")
                if tag == "ramp-artifact":
                    return ["Sol Ring"]
                return []

        deck = Deck(commander=_commander(), cards=[_make_card("Sol Ring", text="Add {C}")])
        report = validate_roles(deck, PartialClient(), roles_to_check=["ramp"])
        # mana-ramp raised but ramp-artifact worked, so role was validated
        assert "ramp" not in report.skipped_roles


class TestMultipleRoles:
    def test_validates_multiple_roles_independently(self):
        deck = Deck(
            commander=_commander(),
            cards=[
                _make_card("Card A"),
                _make_card("Card B"),
            ],
        )
        tag_client = _seed_tag_client({
            "mana-ramp": ["Card A"],
            "removal": ["Card B"],
        })
        report = validate_roles(
            deck, tag_client, roles_to_check=["ramp", "removal"],
        )
        # Both roles have disagreements
        assert "ramp" in report.per_role_summary
        assert "removal" in report.per_role_summary


class TestColorIdentityFilter:
    def test_color_identity_passed_through(self):
        captured = []
        class SpyClient:
            def get_cards_with_oracle_tag(self, tag, color_identity=None):
                captured.append((tag, color_identity))
                return ["Card"] if tag == "mana-ramp" else []

        deck = Deck(commander=_commander(), cards=[_make_card("Card", text="Add {G}")])
        validate_roles(
            deck, SpyClient(), roles_to_check=["ramp"],
            color_identity="WG",
        )
        # Every call should have received the color identity
        assert all(ci == "WG" for _, ci in captured), f"got {captured}"


class TestReportFormatting:
    def test_format_no_disagreements(self):
        report = ValidationReport(
            roles_checked=["ramp"],
            cards_checked=10,
        )
        text = format_role_report(report)
        assert "No disagreements found" in text

    def test_format_with_disagreements(self):
        report = ValidationReport(
            roles_checked=["ramp"],
            cards_checked=2,
            disagreements=[
                RoleDisagreement("Card X", "ramp", "missed"),
                RoleDisagreement("Card Y", "ramp", "extra"),
            ],
        )
        text = format_role_report(report)
        assert "missed" in text
        assert "extra" in text
        assert "Card X" in text
        assert "Card Y" in text

    def test_format_shows_skipped_roles(self):
        report = ValidationReport(
            roles_checked=["ramp", "removal"],
            cards_checked=0,
            skipped_roles=["ramp"],
        )
        text = format_role_report(report)
        assert "Skipped" in text
        assert "ramp" in text

    def test_max_per_kind_truncation(self):
        report = ValidationReport(
            roles_checked=["ramp"],
            cards_checked=100,
            disagreements=[
                RoleDisagreement(f"Card{i}", "ramp", "missed")
                for i in range(25)
            ],
        )
        text = format_role_report(report, max_per_kind=5)
        assert "and 20 more" in text


class TestIntegration:
    """Round-trip with realistic regex behavior from card_database."""

    def test_real_regex_on_real_ramp_cards(self):
        """Known-ramp cards should NOT show up as disagreements when tags agree."""
        # Sol Ring and Rampant Growth both match our ramp regex
        ramp_deck_cards = [
            _make_card("Sol Ring", text="{T}: Add {C}{C}."),
            _make_card(
                "Rampant Growth",
                text="Search your library for a basic land card, put it onto the battlefield tapped",
                card_type="Sorcery", types="Sorcery",
            ),
        ]
        deck = Deck(commander=_commander(), cards=ramp_deck_cards)
        tag_client = _seed_tag_client({
            "mana-ramp": ["Sol Ring", "Rampant Growth"],
        })
        report = validate_roles(deck, tag_client, roles_to_check=["ramp"])
        # Regex and tags agree on both → no disagreements
        assert report.total_disagreements == 0
