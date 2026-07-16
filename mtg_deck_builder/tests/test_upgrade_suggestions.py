"""
Tests for v0.9.34 (#36): card-centric upgrade suggestions — single cards
that would complete detected combos the final deck is one piece away from.

Sourced from _reward_combos (bracket-banned combos excluded upstream);
suggested cards must be in the DB, color-legal, and within budget.
Surfaced in the HTML report and as parser-safe '#' comments in the
decklist output.
"""

from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.html_report import _render_combos_section
from mtg_deck_builder.llm_engine import LLMConfig
from mtg_deck_builder.models import (
    BuildConfig, Combo, Deck, OptimizationResult,
)


def _builder(test_csv_path, **cfg_kw):
    cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                      random_seed=42, **cfg_kw)
    b = DeckBuilder(card_database_path=str(test_csv_path), config=cfg,
                    llm_config=LLMConfig(mock_mode=True))
    b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
    return b


def _result(b, deck_names) -> OptimizationResult:
    cards = [b.db.get_by_name(n) for n in deck_names]
    assert all(c is not None for c in cards), "fixture card missing"
    deck = Deck(commander=b._commander, cards=cards)
    return OptimizationResult(
        best_deck=deck, final_score=50.0, generations_run=1,
        score_history=[], diversity_history=[], runtime_seconds=0.0,
        config=b.config,
    )


class TestComputeUpgradeSuggestions:
    def test_one_missing_grouped_and_ranked(self, test_csv_path):
        b = _builder(test_csv_path)
        # Deck has Sun Titan; two combos each missing only Heliod ->
        # grouped under one suggestion, ranked by best payoff.
        b._reward_combos = [
            Combo(cards=["Sun Titan", "Heliod, Sun-Crowned"],
                  result="Infinite life", payoff=95),
            Combo(cards=["Soul Warden", "Heliod, Sun-Crowned"],
                  result="Value engine", payoff=70),
            Combo(cards=["Sun Titan", "Soul Warden"],
                  result="assembled already", payoff=60),
        ]
        result = _result(b, ["Sun Titan", "Soul Warden"])
        b._compute_upgrade_suggestions(result)
        assert len(result.upgrade_suggestions) == 1
        s = result.upgrade_suggestions[0]
        assert s["card"] == "Heliod, Sun-Crowned"
        assert s["best_payoff"] == 95
        assert len(s["completes"]) == 2  # both combos grouped
        assert s["completes"][0]["payoff"] == 95  # ranked within group

    def test_assembled_and_two_missing_excluded(self, test_csv_path):
        b = _builder(test_csv_path)
        b._reward_combos = [
            # Both pieces missing -> not a one-card suggestion.
            Combo(cards=["Heliod, Sun-Crowned", "Archangel of Thune"], payoff=90),
        ]
        result = _result(b, ["Sun Titan"])
        b._compute_upgrade_suggestions(result)
        assert result.upgrade_suggestions == []

    def test_commander_counts_as_present(self, test_csv_path):
        b = _builder(test_csv_path)
        # "Commander + X" with X absent -> X is a valid one-card suggestion.
        b._reward_combos = [
            Combo(cards=["Lathiel, the Bounteous Dawn",
                         "Archangel of Thune"], payoff=88),
        ]
        result = _result(b, ["Sun Titan"])
        b._compute_upgrade_suggestions(result)
        assert [s["card"] for s in result.upgrade_suggestions] == \
            ["Archangel of Thune"]

    def test_off_color_and_unknown_cards_dropped(self, test_csv_path):
        b = _builder(test_csv_path)  # Lathiel is G/W
        b._reward_combos = [
            Combo(cards=["Sun Titan", "Phyrexian Arena"], payoff=90),  # B
            Combo(cards=["Sun Titan", "Not A Real Card"], payoff=99),
        ]
        result = _result(b, ["Sun Titan"])
        b._compute_upgrade_suggestions(result)
        assert result.upgrade_suggestions == []

    def test_over_budget_suggestion_dropped(self, test_csv_path):
        b = _builder(test_csv_path, budget_max_per_card=5.0)

        class _Prices:
            def get_price(self, name):
                return {"Heliod, Sun-Crowned": 30.0}.get(name)

        b._price_source = _Prices()
        b._reward_combos = [
            Combo(cards=["Sun Titan", "Heliod, Sun-Crowned"], payoff=95),
        ]
        result = _result(b, ["Sun Titan"])
        b._compute_upgrade_suggestions(result)
        assert result.upgrade_suggestions == []

    def test_banned_combos_never_suggested(self, test_csv_path):
        # The method reads _reward_combos, which the combo phase has already
        # stripped of bracket-banned pairs — a banned combo sitting only in
        # _banned_combos must not generate a suggestion.
        b = _builder(test_csv_path, bracket=2)
        b._reward_combos = []
        b._banned_combos = [
            Combo(cards=["Sun Titan", "Heliod, Sun-Crowned"], payoff=95),
        ]
        result = _result(b, ["Sun Titan"])
        b._compute_upgrade_suggestions(result)
        assert result.upgrade_suggestions == []


class TestSurfacing:
    def test_report_renders_grouped_table(self):
        combos = [Combo(cards=["A", "B"], payoff=90)]
        suggestions = [{"card": "B", "best_payoff": 90,
                        "completes": [{"with": ["A"], "result": "win",
                                       "payoff": 90}]}]
        html_out = _render_combos_section(combos, {"A"},
                                          upgrade_suggestions=suggestions)
        assert "Upgrade suggestions" in html_out
        assert "<b>B</b>" in html_out
        # Legacy fallback (no suggestions computed) keeps the old view.
        legacy = _render_combos_section(combos, {"A"},
                                        upgrade_suggestions=None)
        assert "One piece away" in legacy

    def test_decklist_comments_are_parser_safe(self, tmp_path):
        # '#' suggestion lines must not change the parsed deck.
        from mtg_deck_builder.gui import parse_decklist
        deck_txt = (
            "Commander: X\n\n// Creatures (2)\n1 A\n1 B\n\n"
            "# ---- Upgrade suggestions (one card completes a detected "
            "combo) ----\n"
            "# Consider: Heliod, Sun-Crowned — completes Sun Titan + "
            "Heliod, Sun-Crowned (payoff 95)\n"
        )
        f = tmp_path / "x_deck.txt"
        f.write_text(deck_txt, encoding="utf-8")
        deck = parse_decklist(f)
        assert deck["total"] == 2  # comments ignored
