"""
Tests for deck_diff (v0.5).
"""

import pytest

from mtg_deck_builder.deck_diff import (
    diff_decks, format_diff, DiffResult, _extract_names_and_commander,
)
from mtg_deck_builder.models import WarmStartDeck, Deck, Card


def _make_card(name: str) -> Card:
    return Card(
        name=name, mana_cost="{G}", mana_value=1,
        card_type="Creature", text="",
        color_identity="G", colors="G",
        power="1", toughness="1", types="Creature",
    )


class TestInputNormalization:
    def test_warm_start_deck(self):
        ws = WarmStartDeck(
            commander_name="Lathiel",
            card_names=["A", "B", "C"],
        )
        names, commander = _extract_names_and_commander(ws)
        assert names == ["A", "B", "C"]
        assert commander == "Lathiel"

    def test_deck_object(self):
        commander = _make_card("CMD")
        deck = Deck(commander=commander, cards=[_make_card("A"), _make_card("B")])
        names, commander_name = _extract_names_and_commander(deck)
        assert names == ["A", "B"]
        assert commander_name == "CMD"

    def test_dict_input(self):
        data = {"commander_name": "CMD", "card_names": ["A", "B"]}
        names, commander = _extract_names_and_commander(data)
        assert names == ["A", "B"]
        assert commander == "CMD"

    def test_list_input(self):
        names, commander = _extract_names_and_commander(["A", "B", "C"])
        assert names == ["A", "B", "C"]
        assert commander is None

    def test_bad_input_raises(self):
        with pytest.raises(TypeError):
            _extract_names_and_commander(42)


class TestDiffLogic:
    def test_identical_decks_no_changes(self):
        a = ["Card A", "Card B", "Card C"]
        b = ["Card A", "Card B", "Card C"]
        result = diff_decks(a, b)
        assert result.added_count == 0
        assert result.removed_count == 0
        assert result.kept_count == 3

    def test_complete_swap(self):
        a = ["A", "B", "C"]
        b = ["X", "Y", "Z"]
        result = diff_decks(a, b)
        assert set(result.added) == {"X", "Y", "Z"}
        assert set(result.removed) == {"A", "B", "C"}
        assert result.kept == []

    def test_partial_swap(self):
        a = ["A", "B", "C", "D"]
        b = ["A", "B", "X", "Y"]
        result = diff_decks(a, b)
        assert sorted(result.kept) == ["A", "B"]
        assert set(result.added) == {"X", "Y"}
        assert set(result.removed) == {"C", "D"}

    def test_basic_land_multiset_semantics(self):
        """Going from 15 Forests to 17 Forests should show 2 added Forests."""
        a = ["Forest"] * 15 + ["Sol Ring"]
        b = ["Forest"] * 17 + ["Sol Ring"]
        result = diff_decks(a, b)
        assert result.added.count("Forest") == 2
        assert result.removed.count("Forest") == 0
        assert "Sol Ring" in result.kept

    def test_basic_land_reduction(self):
        """Going from 17 Forests to 15 Forests -> 2 removed."""
        a = ["Forest"] * 17
        b = ["Forest"] * 15
        result = diff_decks(a, b)
        assert result.removed.count("Forest") == 2
        assert result.kept.count("Forest") == 15

    def test_kept_count_uses_min(self):
        """kept count for a duplicated card = min(from, to)."""
        a = ["Forest"] * 10
        b = ["Forest"] * 5
        result = diff_decks(a, b)
        assert result.kept_count == 5
        assert result.removed_count == 5
        assert result.added_count == 0


class TestCommanderChange:
    def test_same_commander(self):
        a = WarmStartDeck(commander_name="Lathiel", card_names=["A"])
        b = WarmStartDeck(commander_name="Lathiel", card_names=["A"])
        result = diff_decks(a, b)
        assert not result.commander_changed
        assert result.commander_from == "Lathiel"
        assert result.commander_to == "Lathiel"

    def test_different_commander_flagged(self):
        a = WarmStartDeck(commander_name="Lathiel", card_names=["A"])
        b = WarmStartDeck(commander_name="Karlov", card_names=["A"])
        result = diff_decks(a, b)
        assert result.commander_changed

    def test_none_commander_not_flagged(self):
        """Diffing a raw list (no commander) shouldn't trigger commander_changed."""
        result = diff_decks(["A"], ["A"])
        assert not result.commander_changed


class TestRoleGrouping:
    def test_grouping_when_db_provided(self, test_csv_path):
        """With a CardDatabase, added/removed get role-bucketed."""
        from mtg_deck_builder.card_database import CardDatabase
        db = CardDatabase(test_csv_path)

        # Fake diff with Sol Ring added (ramp) and a basic removed
        a = ["Forest", "Card X"]
        b = ["Forest", "Sol Ring"]
        result = diff_decks(a, b, card_db=db)
        # Sol Ring should be in added_by_role under 'ramp' (Sol Ring is in the
        # test CSV and matches ramp patterns)
        assert result.added_by_role  # not empty
        # The bucket should be the ramp bucket (assuming Sol Ring is in test_cards.csv)
        all_added = []
        for role, names in result.added_by_role.items():
            all_added.extend(names)
        assert "Sol Ring" in all_added

    def test_no_grouping_without_db(self):
        a = ["A"]
        b = ["B"]
        result = diff_decks(a, b)
        assert result.added_by_role == {}
        assert result.removed_by_role == {}


class TestFormatDiff:
    def test_format_identical(self):
        result = diff_decks(["A"], ["A"])
        text = format_diff(result)
        assert "(No changes.)" in text or "Kept: 1" in text

    def test_format_with_changes(self):
        result = diff_decks(["A", "B"], ["A", "C"])
        text = format_diff(result)
        assert "Added" in text
        assert "Removed" in text
        assert "C" in text
        assert "B" in text

    def test_commander_change_shown(self):
        a = WarmStartDeck(commander_name="X", card_names=["A"])
        b = WarmStartDeck(commander_name="Y", card_names=["A"])
        text = format_diff(diff_decks(a, b))
        assert "Commander changed" in text
        assert "X" in text and "Y" in text

    def test_max_per_group_limits_output(self):
        """Large added list should be truncated with '... and N more'."""
        a = []
        b = [f"Card{i}" for i in range(100)]
        result = diff_decks(a, b)
        text = format_diff(result, max_per_group=10)
        assert "and 90 more" in text

    def test_show_kept_included_when_flag_set(self):
        result = diff_decks(["A", "B", "C"], ["A", "B", "C"])
        with_kept = format_diff(result, show_kept=True)
        assert "Kept ===" in with_kept
        without_kept = format_diff(result, show_kept=False)
        assert "Kept ===" not in without_kept


class TestJSONRoundTrip:
    def test_diff_two_json_files(self, tmp_path):
        """End-to-end: save two WarmStartDecks, load and diff."""
        a = WarmStartDeck(
            commander_name="Lathiel",
            card_names=["Sol Ring", "Forest", "Plains"],
        )
        b = WarmStartDeck(
            commander_name="Lathiel",
            card_names=["Sol Ring", "Forest", "Island"],
        )
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        import json
        path_a.write_text(json.dumps(a.to_dict()))
        path_b.write_text(json.dumps(b.to_dict()))

        loaded_a = WarmStartDeck.from_json_file(str(path_a))
        loaded_b = WarmStartDeck.from_json_file(str(path_b))
        result = diff_decks(loaded_a, loaded_b)
        assert "Island" in result.added
        assert "Plains" in result.removed
        assert "Sol Ring" in result.kept
