"""
Tests for the v0.9.18 MTGJSON refresh (Python port of mtg-deck-extract.js).

Covers transformation parity with the .js (field order, array joining,
legalities, escaping, layout skip, cardVersions[0]) plus the new
isGameChanger column and round-trip through the CardDatabase loader.
"""

import json

from mtg_deck_builder import mtgjson_refresh as mr
from mtg_deck_builder.card_database import CardDatabase
from mtg_deck_builder.bracket import is_game_changer, reset_game_changer_source


def _atomic(**cards):
    """Build an MTGJSON-atomic `data` dict from name -> printing dict."""
    return {name: [printing] for name, printing in cards.items()}


class TestExtractRows:
    def test_basic_field_mapping_and_gc_column(self):
        data = _atomic(**{
            "Sol Ring": {
                "manaCost": "{1}", "manaValue": 1.0, "type": "Artifact",
                "text": "{T}: Add {C}{C}.", "colorIdentity": [],
                "colors": [], "types": ["Artifact"], "layout": "normal",
                "legalities": {"commander": "Legal", "modern": "Banned"},
                "isGameChanger": True,
            },
        })
        rows, n = mr.extract_rows(data)
        assert n == 1
        row = rows[0]
        # 17 base fields + isGameChanger = 18 cells.
        assert len(row) == len(mr._FIELDS) + 1
        cells = dict(zip(mr._FIELDS + ["isGameChanger"], row))
        assert cells["name"] == "Sol Ring"
        assert cells["manaValue"] == "1"           # .0 dropped like JS
        assert cells["types"] == "Artifact"        # array joined
        assert cells["legalities"] == "commander"  # only Legal, lowercased
        assert cells["isGameChanger"] == "true"

    def test_layout_skip(self):
        data = _atomic(
            Real={"layout": "normal", "type": "Creature"},
            Tok={"layout": "token", "type": "Token Creature"},
            Emb={"layout": "emblem", "type": "Emblem"},
        )
        rows, n = mr.extract_rows(data)
        assert n == 1
        assert rows[0][0] == "Real"

    def test_takes_first_printing(self):
        data = {"Multi": [
            {"layout": "normal", "type": "First", "isGameChanger": True},
            {"layout": "normal", "type": "Second"},
        ]}
        rows, _ = mr.extract_rows(data)
        assert dict(zip(mr._FIELDS, rows[0]))["type"] == "First"
        assert rows[0][-1] == "true"  # first printing's GC flag

    def test_escaping_order(self):
        # Text with backslash, pipe, and newline must escape in .js order.
        data = _atomic(Weird={
            "layout": "normal", "type": "Instant",
            "text": "a\\b|c\nd",
        })
        rows, _ = mr.extract_rows(data)
        text = dict(zip(mr._FIELDS, rows[0]))["text"]
        assert text == "a\\\\b\\|c\\nd"

    def test_missing_gc_defaults_false(self):
        data = _atomic(Plain={"layout": "normal", "type": "Land"})
        rows, _ = mr.extract_rows(data)
        assert rows[0][-1] == "false"


class TestWriteAndRoundTrip:
    def test_written_csv_loads_and_carries_gc(self, tmp_path):
        data = _atomic(**{
            "Rhystic Study": {
                "manaCost": "{2}{U}", "manaValue": 3.0, "type": "Enchantment",
                "text": "Whenever an opponent casts a spell...",
                "colorIdentity": ["U"], "colors": ["U"],
                "types": ["Enchantment"], "layout": "normal",
                "legalities": {"commander": "Legal"}, "isGameChanger": True,
            },
            "Grizzly Bears": {
                "manaCost": "{1}{G}", "manaValue": 2.0, "type": "Creature",
                "text": "", "colorIdentity": ["G"], "colors": ["G"],
                "power": "2", "toughness": "2", "types": ["Creature"],
                "subtypes": ["Bear"], "layout": "normal",
                "legalities": {"commander": "Legal"}, "isGameChanger": False,
            },
        })
        out = tmp_path / "cards.csv"
        count = mr.write_csv(data, out)
        assert count == 2

        db = CardDatabase(str(out))
        db.load()
        assert db.has_game_changer_column is True
        rhystic = db.get_by_name("Rhystic Study")
        bears = db.get_by_name("Grizzly Bears")
        assert rhystic.is_game_changer is True
        assert bears.is_game_changer is False
        assert rhystic.mana_value == 3
        # And the bracket source honors the CSV flag directly.
        try:
            assert is_game_changer(rhystic)
        finally:
            reset_game_changer_source()

    def test_refresh_from_local_file(self, tmp_path):
        atomic = tmp_path / "AtomicCards.json"
        atomic.write_text(json.dumps({"meta": {}, "data": _atomic(
            Card={"layout": "normal", "type": "Instant",
                  "legalities": {"commander": "Legal"}},
        )}), encoding="utf-8")
        out = tmp_path / "cards.csv"
        count = mr.refresh(output_path=str(out), atomic_json_path=str(atomic))
        assert count == 1
        assert out.exists()
