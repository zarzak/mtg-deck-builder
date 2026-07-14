"""Tests for web.forms — form dict → BuildConfig mapping."""

import pytest

from mtg_deck_builder.web.forms import (
    config_from_form, build_score_weights,
    _coerce_int, _coerce_optional_int, _coerce_optional_float, _coerce_bool,
    _split_lines, _split_csv, _parse_weights,
)


class TestCoercionHelpers:
    def test_int_with_default(self):
        assert _coerce_int("10", 5) == 10
        assert _coerce_int("", 5) == 5
        assert _coerce_int(None, 5) == 5
        assert _coerce_int("garbage", 5) == 5

    def test_optional_int(self):
        assert _coerce_optional_int("42") == 42
        assert _coerce_optional_int("") is None
        assert _coerce_optional_int(None) is None
        assert _coerce_optional_int("garbage") is None

    def test_optional_float(self):
        assert _coerce_optional_float("1.5") == 1.5
        assert _coerce_optional_float("") is None
        assert _coerce_optional_float("NaNgarbage") is None

    def test_bool(self):
        assert _coerce_bool("on") is True
        assert _coerce_bool("true") is True
        assert _coerce_bool(True) is True
        assert _coerce_bool("") is False
        assert _coerce_bool(None) is False
        assert _coerce_bool("off") is False

    def test_split_lines(self):
        assert _split_lines("a\nb\n\n  c  \n") == ["a", "b", "c"]
        assert _split_lines("") == []
        assert _split_lines(None) == []

    def test_split_csv(self):
        assert _split_csv("a, b, c") == ["a", "b", "c"]
        assert _split_csv("") == []
        assert _split_csv("  x  ,  y  ") == ["x", "y"]

    def test_parse_weights(self):
        form = {
            "weight_synergy": "0.4",
            "weight_flavor": "0.2",
            "weight_bogus": "abc",  # invalid, dropped
            "commander_name": "ignored",
        }
        weights = _parse_weights(form)
        assert weights == {"synergy": 0.4, "flavor": 0.2}


class TestBuildScoreWeights:
    def test_no_preset_no_overrides_returns_none(self):
        assert build_score_weights(None, {}) is None
        assert build_score_weights("", {}) is None

    def test_preset_only(self):
        # v0.9.7: creativity removed from scoring; the flavor preset's flavor
        # weight is restored to 0.25 (it had been trimmed to make room for
        # strategy_density + creativity).
        w = build_score_weights("flavor", {})
        assert w is not None
        assert w["flavor"] == 0.25
        assert w["strategy_density"] == 0.25
        assert "creativity" not in w

    def test_overrides_only_use_balanced_base(self):
        w = build_score_weights(None, {"synergy": 0.99})
        assert w is not None
        assert w["synergy"] == 0.99
        # v0.9.3 balanced preset defaults
        assert w["mana_curve"] == 0.10
        assert w["strategy_density"] == 0.20

    def test_preset_plus_override(self):
        w = build_score_weights("flavor", {"synergy": 0.01})
        assert w["synergy"] == 0.01  # override wins
        assert w["flavor"] == 0.25   # preset preserved (v0.9.7)
        assert w["strategy_density"] == 0.25

    def test_unknown_dim_dropped(self):
        w = build_score_weights(None, {"bogus_dim": 0.5})
        assert "bogus_dim" not in w


class TestConfigFromForm:
    def test_missing_commander_raises(self):
        with pytest.raises(ValueError):
            config_from_form({})

    def test_minimal_form(self):
        cfg = config_from_form({"commander_name": "Lathiel"})
        assert cfg.commander_name == "Lathiel"
        # Defaults preserved
        assert cfg.population_size == 50
        assert cfg.generations == 100

    def test_numeric_fields(self):
        cfg = config_from_form({
            "commander_name": "Lathiel",
            "population_size": "20",
            "generations": "15",
            "random_seed": "42",
            "budget_max_per_card": "5.50",
        })
        assert cfg.population_size == 20
        assert cfg.generations == 15
        assert cfg.random_seed == 42
        assert cfg.budget_max_per_card == 5.50

    def test_checkbox_fields(self):
        cfg = config_from_form({
            "commander_name": "Lathiel",
            "use_images": "on",
            "use_bulk_source": "on",
            "bulk_offline": "on",
            "validate_roles": "on",
        })
        assert cfg.use_images is True
        assert cfg.use_bulk_source is True
        assert cfg.bulk_offline is True
        assert cfg.validate_roles_after_build is True
        # Unchecked boxes stay False
        assert cfg.use_edhrec is False

    def test_list_fields(self):
        cfg = config_from_form({
            "commander_name": "Lathiel",
            "flavor_tags": "forest, mammoth, deer",
            "locked_cards": "Sol Ring\nLightning Greaves",
            "banned_cards": "Path to Exile",
        })
        assert cfg.flavor_art_tags == ["forest", "mammoth", "deer"]
        assert cfg.locked_cards == ["Sol Ring", "Lightning Greaves"]
        assert cfg.banned_cards == ["Path to Exile"]

    def test_preset_applied(self):
        cfg = config_from_form({
            "commander_name": "Lathiel",
            "preset": "flavor",
        })
        # v0.9.7: flavor preset has flavor=0.25 (creativity removed from
        # scoring freed up weight to restore flavor's emphasis).
        assert cfg.score_weights["flavor"] == 0.25

    def test_weight_override_beats_preset(self):
        cfg = config_from_form({
            "commander_name": "Lathiel",
            "preset": "flavor",
            "weight_synergy": "0.01",
        })
        assert cfg.score_weights["synergy"] == 0.01

    def test_no_weights_submitted_leaves_defaults(self):
        cfg = config_from_form({"commander_name": "Lathiel"})
        # Default BuildConfig score_weights (no flavor key)
        assert "flavor" not in cfg.score_weights or cfg.score_weights.get("flavor", 0) == 0

    def test_bulk_type_passthrough(self):
        cfg = config_from_form({
            "commander_name": "Lathiel",
            "bulk_type": "default_cards",
        })
        assert cfg.bulk_type == "default_cards"

    def test_bulk_type_default_when_empty(self):
        cfg = config_from_form({"commander_name": "Lathiel"})
        assert cfg.bulk_type == "oracle_cards"

    def test_population_size_clamped_low(self):
        """v0.7.1: out-of-range GA params are clamped, not rejected."""
        cfg = config_from_form({
            "commander_name": "Lathiel", "population_size": "0",
        })
        assert cfg.population_size == 4  # clamped to min

    def test_population_size_clamped_high(self):
        cfg = config_from_form({
            "commander_name": "Lathiel", "population_size": "999999",
        })
        assert cfg.population_size == 500  # clamped to max

    def test_population_size_negative_clamped(self):
        cfg = config_from_form({
            "commander_name": "Lathiel", "population_size": "-5",
        })
        assert cfg.population_size == 4

    def test_generations_clamped(self):
        cfg = config_from_form({
            "commander_name": "Lathiel", "generations": "0",
        })
        assert cfg.generations == 1

    def test_generations_in_range_passes_through(self):
        cfg = config_from_form({
            "commander_name": "Lathiel", "generations": "50",
        })
        assert cfg.generations == 50
