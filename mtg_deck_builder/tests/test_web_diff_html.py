"""Tests for web.diff_html — DiffResult → HTML."""

import pytest

from mtg_deck_builder.deck_diff import DiffResult
from mtg_deck_builder.web.diff_html import render_diff_html


def _empty_diff():
    return DiffResult(
        commander_from="Cmdr A",
        commander_to="Cmdr A",
        kept=["Sol Ring", "Forest"],
        added=[],
        removed=[],
    )


class TestRenderDiffHtml:
    def test_returns_full_html_doc(self):
        html = render_diff_html(_empty_diff())
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_no_changes_message(self):
        html = render_diff_html(_empty_diff())
        assert "No cards added" in html
        assert "No cards removed" in html

    def test_added_cards_listed(self):
        r = DiffResult(
            commander_from="X", commander_to="X", kept=[],
            added=["Lightning Bolt", "Counterspell"],
            removed=[],
        )
        html = render_diff_html(r)
        assert "Lightning Bolt" in html
        assert "Counterspell" in html
        assert "+2" in html

    def test_removed_cards_listed(self):
        r = DiffResult(
            commander_from="X", commander_to="X", kept=[],
            added=[],
            removed=["Path to Exile"],
        )
        html = render_diff_html(r)
        assert "Path to Exile" in html
        # Unicode minus
        assert "−1" in html

    def test_commander_change_callout(self):
        r = DiffResult(
            commander_from="Old Cmdr", commander_to="New Cmdr",
            kept=[], added=[], removed=[],
        )
        html = render_diff_html(r)
        assert "Commander changed" in html
        assert "Old Cmdr" in html
        assert "New Cmdr" in html

    def test_no_commander_change_when_same(self):
        r = DiffResult(
            commander_from="Same", commander_to="Same",
            kept=[], added=[], removed=[],
        )
        html = render_diff_html(r)
        assert "Commander changed" not in html

    def test_filenames_in_meta(self):
        html = render_diff_html(
            _empty_diff(), before_name="old.json", after_name="new.json",
        )
        assert "old.json" in html
        assert "new.json" in html

    def test_xss_in_card_names_escaped(self):
        r = DiffResult(
            commander_from="X", commander_to="X", kept=[],
            added=["<script>alert('xss')</script>"],
            removed=[],
        )
        html = render_diff_html(r)
        # The literal <script> tag should NOT appear in the output
        assert "<script>" not in html
        # But the escaped form should
        assert "&lt;script&gt;" in html

    def test_role_groups_used_when_provided(self):
        r = DiffResult(
            commander_from="X", commander_to="X", kept=[],
            added=["Sol Ring", "Forest"],
            removed=[],
            added_by_role={
                "ramp": ["Sol Ring"],
                "land": ["Forest"],
            },
        )
        html = render_diff_html(r)
        assert "ramp" in html.lower()
        assert "land" in html.lower()

    def test_unchanged_count_displayed(self):
        r = DiffResult(
            commander_from="X", commander_to="X",
            kept=["A", "B", "C"],
            added=["D"],
            removed=[],
        )
        html = render_diff_html(r)
        # 3 unchanged should be in the summary
        assert ">3<" in html
