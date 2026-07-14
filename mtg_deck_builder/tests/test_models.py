"""Tests for data models."""

import pytest
from mtg_deck_builder.models import (
    Card, Deck, DeckScores, BuildConfig, CommanderAnalysis, CardTelemetry,
)


class TestCard:
    def test_card_equality_by_name(self):
        """Cards with same name should be equal even if other fields differ."""
        c1 = Card(name="Sol Ring", mana_cost="{1}", mana_value=1,
                  card_type="Artifact", text="", color_identity="", colors="")
        c2 = Card(name="Sol Ring", mana_cost="{1}", mana_value=1,
                  card_type="Artifact", text="different text", color_identity="", colors="")
        assert c1 == c2
        assert hash(c1) == hash(c2)

    def test_card_inequality(self):
        c1 = Card(name="Sol Ring", mana_cost="{1}", mana_value=1,
                  card_type="Artifact", text="", color_identity="", colors="")
        c2 = Card(name="Arcane Signet", mana_cost="{2}", mana_value=2,
                  card_type="Artifact", text="", color_identity="", colors="")
        assert c1 != c2
        assert hash(c1) != hash(c2)

    def test_card_set_membership(self):
        """Cards should work in sets (hash consistency)."""
        c1 = Card(name="Sol Ring", mana_cost="{1}", mana_value=1,
                  card_type="Artifact", text="", color_identity="", colors="")
        c2 = Card(name="Sol Ring", mana_cost="{1}", mana_value=1,
                  card_type="Artifact", text="other", color_identity="", colors="")
        s = {c1, c2}
        assert len(s) == 1

    def test_is_creature(self, db):
        assert db.get_by_name("Birds of Paradise").is_creature
        assert not db.get_by_name("Sol Ring").is_creature

    def test_is_land(self, db):
        assert db.get_by_name("Forest").is_land
        assert not db.get_by_name("Sol Ring").is_land

    def test_is_basic_land(self, db):
        assert db.get_by_name("Forest").is_basic_land
        assert not db.get_by_name("Command Tower").is_basic_land

    def test_is_vanilla(self, db):
        """Grizzly Bears has no text — truly vanilla."""
        assert db.get_by_name("Grizzly Bears").is_vanilla
        # Birds of Paradise has Flying keyword — not vanilla
        assert not db.get_by_name("Birds of Paradise").is_vanilla

    def test_is_not_vanilla_non_creature(self, db):
        """Non-creatures can't be vanilla regardless of text."""
        assert not db.get_by_name("Forest").is_vanilla  # land, not creature
        assert not db.get_by_name("Sol Ring").is_vanilla  # artifact, not creature

    def test_format_for_llm(self, db):
        s = db.get_by_name("Llanowar Elves").format_for_llm()
        assert "Llanowar Elves" in s
        assert "{G}" in s
        assert "Creature" in s


class TestDeck:
    def test_valid_deck(self, lathiel, db):
        """A deck with 99 unique cards (or basic duplicates) and correct color ID is valid."""
        # Use cards from the test DB
        pool = [c for c in db.all_cards if c.name != lathiel.name]
        wg_pool = [c for c in pool if set(
            ch for ch in (c.color_identity or "") if ch in "WUBRG"
        ).issubset({"W", "G"})]
        # Pick 99 unique cards, padding with Forest if needed
        cards = wg_pool[:99]
        forest = db.get_by_name("Forest")
        while len(cards) < 99:
            cards.append(forest)
        deck = Deck(commander=lathiel, cards=cards)
        valid, reasons = deck.validate()
        assert valid, f"Deck should be valid but got: {reasons}"

    def test_wrong_card_count_invalid(self, lathiel):
        deck = Deck(commander=lathiel, cards=[])
        valid, reasons = deck.validate()
        assert not valid
        assert any("card count" in r.lower() for r in reasons)

    def test_color_identity_violation(self, lathiel, db):
        """A deck with a blue card under a W/G commander should be invalid."""
        forest = db.get_by_name("Forest")
        island = db.get_by_name("Island")
        cards = [island] + [forest] * 98
        deck = Deck(commander=lathiel, cards=cards)
        valid, reasons = deck.validate()
        assert not valid
        assert any("color identity" in r.lower() for r in reasons)

    def test_duplicate_non_basic(self, lathiel, db):
        """Duplicate non-basics = invalid."""
        sol_ring = db.get_by_name("Sol Ring")
        forest = db.get_by_name("Forest")
        # 2 Sol Rings + 97 Forests = duplicate non-basic
        cards = [sol_ring, sol_ring] + [forest] * 97
        deck = Deck(commander=lathiel, cards=cards)
        valid, reasons = deck.validate()
        assert not valid
        assert any("duplicate" in r.lower() for r in reasons)

    def test_basic_duplicates_ok(self, lathiel, db):
        """Duplicate BASIC lands are legal."""
        forest = db.get_by_name("Forest")
        plains = db.get_by_name("Plains")
        # 99 basics, with some duplicates
        cards = [forest] * 50 + [plains] * 49
        deck = Deck(commander=lathiel, cards=cards)
        valid, reasons = deck.validate()
        assert valid, f"Basic duplicates should be valid, got: {reasons}"

    def test_mana_curve(self, lathiel, db):
        """get_mana_curve excludes lands."""
        forest = db.get_by_name("Forest")
        sol_ring = db.get_by_name("Sol Ring")  # MV 1
        cultivate = db.get_by_name("Cultivate")  # MV 3
        cards = [forest] * 37 + [sol_ring] + [cultivate]
        while len(cards) < 99:
            cards.append(forest)
        deck = Deck(commander=lathiel, cards=cards)
        curve = deck.get_mana_curve()
        assert 1 in curve and curve[1] >= 1
        assert 3 in curve and curve[3] >= 1
        # No lands in curve
        assert 0 not in curve or curve[0] == 0 or all(
            not c.is_land for c in deck.cards if c.mana_value == 0 and not c.is_land
        )

    def test_to_decklist(self, lathiel, db):
        forest = db.get_by_name("Forest")
        sol_ring = db.get_by_name("Sol Ring")
        cards = [forest] * 98 + [sol_ring]
        deck = Deck(commander=lathiel, cards=cards)
        output = deck.to_decklist()
        assert "Commander: Lathiel" in output
        assert "Sol Ring" in output
        assert "Forest" in output


class TestBuildConfig:
    def test_default_weights(self):
        # v0.9.15: the default bracket is 4 (Optimized), which scales the
        # power_level weight x1.25 (0.20 -> 0.25) — so the sum is ~1.05,
        # not 1.0. At bracket 3 (neutral scaling) the set sums to 1.0.
        cfg = BuildConfig(commander_name="Test")
        assert cfg.bracket == 4
        weights = cfg.get_effective_weights()
        assert weights["power_level"] == pytest.approx(0.25)
        neutral = BuildConfig(commander_name="Test", bracket=3)
        nw = neutral.get_effective_weights()
        assert abs(sum(nw.values()) - 1.0) < 0.01
        assert "synergy" in weights
        # v0.9.7: strategy_density is now an explicit default weight, and
        # creativity is no longer scored.
        assert "strategy_density" in weights
        assert "creativity" not in weights

    def test_commander_override_weights(self):
        """If analysis has recommended_weights, those should win (then
        bracket scaling applies on top — x1.25 power at bracket 4)."""
        cfg = BuildConfig(commander_name="Test", bracket=3)
        analysis = CommanderAnalysis(
            name="T", color_identity="G",
            key_mechanics=[], build_around_text="",
            evaluation_notes="", category_queries={},
            synergy_keywords=[],
            recommended_weights={"synergy": 0.60, "power_level": 0.10},
        )
        weights = cfg.get_effective_weights(analysis)
        assert weights["synergy"] == 0.60
        assert weights["power_level"] == 0.10  # bracket 3 = neutral scaling
        # Untouched weights should come from the default (v0.9.7: 0.10)
        assert weights["mana_curve"] == 0.10
        # At bracket 4 the recommendation is scaled x1.25.
        cfg4 = BuildConfig(commander_name="Test", bracket=4)
        assert cfg4.get_effective_weights(analysis)["power_level"] == \
            pytest.approx(0.125)

    def test_commander_override_disabled(self):
        """adaptive_weights=False should ignore analysis overrides."""
        cfg = BuildConfig(commander_name="Test", commander_adaptive_weights=False)
        analysis = CommanderAnalysis(
            name="T", color_identity="G",
            key_mechanics=[], build_around_text="",
            evaluation_notes="", category_queries={},
            synergy_keywords=[],
            recommended_weights={"synergy": 0.99},
        )
        weights = cfg.get_effective_weights(analysis)
        assert weights["synergy"] == 0.35  # default, not 0.99

    def test_power_level_scaling_high(self):
        """High power_level weights raw power more (v0.9.7: no longer via creativity)."""
        cfg = BuildConfig(commander_name="Test", power_level=9)
        weights = cfg.get_effective_weights()
        default = BuildConfig(commander_name="Test", power_level=7).get_effective_weights()
        assert weights["power_level"] > default["power_level"]
        # creativity is no longer a weighted dimension
        assert "creativity" not in weights

    def test_power_level_scaling_low(self):
        """Low power_level weights raw power less (v0.9.7: no longer via creativity)."""
        cfg = BuildConfig(commander_name="Test", power_level=3)
        weights = cfg.get_effective_weights()
        default = BuildConfig(commander_name="Test", power_level=7).get_effective_weights()
        assert weights["power_level"] < default["power_level"]
        assert "creativity" not in weights

    def test_synergy_balance_default(self):
        cfg = BuildConfig(commander_name="Test")
        base, syn = cfg.get_effective_synergy_balance()
        assert base == 0.4
        assert syn == 0.6

    def test_synergy_balance_from_analysis(self):
        cfg = BuildConfig(commander_name="Test")
        analysis = CommanderAnalysis(
            name="T", color_identity="G",
            key_mechanics=[], build_around_text="",
            evaluation_notes="", category_queries={},
            synergy_keywords=[],
            recommended_synergy_weight=0.85,
        )
        base, syn = cfg.get_effective_synergy_balance(analysis)
        assert abs(syn - 0.85) < 0.001
        assert abs(base - 0.15) < 0.001


class TestDeckScores:
    def test_total_weighted_excludes_creativity(self):
        # v0.9.7: creativity is informational only and must NOT affect total.
        s = DeckScores(
            mana_curve=80, role_coverage=70, synergy=60,
            power_level=50, creativity=40,
        )
        weights = {"mana_curve": 0.2, "role_coverage": 0.2, "synergy": 0.2,
                   "power_level": 0.2, "creativity": 0.2}
        total = s.total(weights)
        # creativity (40 * 0.2 = 8) is excluded; strategy_density defaults to
        # 0.20 weight but its score is 0 here, so it contributes nothing.
        expected = (80 + 70 + 60 + 50) * 0.2
        assert abs(total - expected) < 0.01

    def test_creativity_weight_is_ignored(self):
        # Cranking the creativity weight changes nothing — it's not scored.
        s = DeckScores(synergy=100, creativity=100)
        low = s.total({"synergy": 1.0, "creativity": 0.0})
        high = s.total({"synergy": 1.0, "creativity": 10.0})
        assert low == high

    def test_total_with_penalty(self):
        s = DeckScores(
            mana_curve=100, role_coverage=100, synergy=100,
            power_level=100, creativity=100, constraint_penalty=30,
        )
        weights = {"mana_curve": 0.2, "role_coverage": 0.2, "synergy": 0.2,
                   "power_level": 0.2, "creativity": 0.2}
        total = s.total(weights)
        # creativity excluded: (100*4)*0.2 = 80, minus 30 penalty = 50.
        assert total == 50.0

    def test_total_floors_at_zero(self):
        s = DeckScores(constraint_penalty=500)
        weights = {"synergy": 1.0}
        assert s.total(weights) == 0.0


class TestCardTelemetry:
    def test_telemetry_dataclass(self):
        t = CardTelemetry(
            name="Sol Ring", baseline_power=80, synergy_score=45,
            effective_score=59, role="ramp",
        )
        assert t.name == "Sol Ring"
        assert t.role == "ramp"
