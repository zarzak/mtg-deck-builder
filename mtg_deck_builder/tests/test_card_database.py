"""Tests for CardDatabase and card_fills_role."""

import pytest
import tempfile
from pathlib import Path

from mtg_deck_builder.card_database import CardDatabase, card_fills_role


class TestCardDatabaseLoading:
    def test_load_count(self, db):
        """Test CSV should load 83+ cards."""
        assert db.card_count >= 80

    def test_get_by_name(self, db):
        c = db.get_by_name("Sol Ring")
        assert c is not None
        assert c.name == "Sol Ring"

    def test_get_by_name_case_insensitive(self, db):
        assert db.get_by_name("sol ring") is not None
        assert db.get_by_name("SOL RING") is not None

    def test_get_by_name_strip_whitespace(self, db):
        assert db.get_by_name("  Sol Ring  ") is not None

    def test_get_by_name_not_found(self, db):
        assert db.get_by_name("Nonexistent Card") is None

    def test_handles_special_names(self, db):
        """Names with apostrophes and punctuation should work."""
        assert db.get_by_name("Kodama's Reach") is not None
        assert db.get_by_name("Teferi's Protection") is not None

    def test_csv_with_comments(self):
        """Loader should skip lines starting with #."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("# A comment line\n")
            f.write("# Another comment\n")
            f.write("name|manaCost|manaValue|type|text|colorIdentity|colors|power|"
                    "toughness|loyalty|defense|types|subtypes|supertypes|keywords|"
                    "layout|legalities\n")
            f.write("Test Card|{1}|1|Artifact|Test||||||Artifact|||||normal|commander\n")
            path = f.name

        try:
            db = CardDatabase(path)
            db.load()
            assert db.card_count == 1
            assert db.get_by_name("Test Card") is not None
        finally:
            Path(path).unlink()

    def test_csv_with_bom(self):
        """Loader should handle UTF-8 BOM."""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as f:
            f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
            f.write(b"name|manaCost|manaValue|type|text|colorIdentity|colors|"
                    b"power|toughness|loyalty|defense|types|subtypes|supertypes|"
                    b"keywords|layout|legalities\n")
            f.write(b"Test|{1}|1|Artifact|||||||Artifact|||||normal|commander\n")
            path = f.name
        try:
            db = CardDatabase(path)
            db.load()
            assert db.card_count == 1
        finally:
            Path(path).unlink()

    def test_empty_legalities_excluded_when_column_present(self):
        """v0.9.13 regression: with a legalities column present (MTGJSON-
        derived data), an EMPTY value means the card is legal in NO format
        (banned like Mox Emerald, un-set like HONK!) and must be excluded.
        The old lenient rule admitted ~2,100 such cards."""
        header = ("name|manaCost|manaValue|type|text|colorIdentity|colors|"
                  "power|toughness|loyalty|defense|types|subtypes|supertypes|"
                  "keywords|layout|legalities\n")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(header)
            f.write("Legal Card|{1}|1|Artifact|||||||Artifact|||||normal|commander,legacy\n")
            f.write("Banned Card|{0}|0|Artifact|||||||Artifact|||||normal|\n")
            path = f.name
        try:
            db = CardDatabase(path)
            db.load()
            assert db.get_by_name("Legal Card") is not None
            assert db.get_by_name("Banned Card") is None
        finally:
            Path(path).unlink()

    def test_same_face_dfc_name_normalized(self):
        """v0.9.16b regression: 'Sol Ring // Sol Ring' promo printings must
        collapse onto the plain name — a real run ended up with two Sol
        Rings that duplicate validation couldn't see."""
        header = ("name|manaCost|manaValue|type|text|colorIdentity|colors|"
                  "power|toughness|loyalty|defense|types|subtypes|supertypes|"
                  "keywords|layout|legalities\n")
        def _row(name, types="Artifact", ci=""):
            # 17 fields matching the header exactly.
            return "|".join([
                name, "{1}", "1", types, "Some text.", ci, ci,
                "", "", "", "", types, "", "", "", "normal", "commander",
            ]) + "\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(header)
            f.write(_row("Sol Ring"))
            f.write(_row("Sol Ring // Sol Ring"))
            f.write(_row("Wear // Tear", types="Instant", ci="R,W"))
            path = f.name
        try:
            db = CardDatabase(path)
            db.load()
            assert db.card_count == 2  # variant collapsed onto Sol Ring
            assert db.get_by_name("Sol Ring") is not None
            assert db.get_by_name("Sol Ring // Sol Ring") is None
            assert db.get_by_name("Wear // Tear") is not None  # real split OK
        finally:
            Path(path).unlink()

    def test_game_changer_column_read(self):
        """v0.9.17: an isGameChanger CSV column populates Card.is_game_changer
        and flags the DB as authoritative."""
        header = ("name|manaCost|manaValue|type|text|colorIdentity|colors|"
                  "power|toughness|loyalty|defense|types|subtypes|supertypes|"
                  "keywords|layout|legalities|isGameChanger\n")

        def _row(name, gc):
            return "|".join([
                name, "{1}", "1", "Artifact", "t.", "", "",
                "", "", "", "", "Artifact", "", "", "", "normal",
                "commander", gc,
            ]) + "\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(header)
            f.write(_row("Gamechanging Card", "true"))
            f.write(_row("Normal Card", "false"))
            path = f.name
        try:
            db = CardDatabase(path)
            db.load()
            assert db.has_game_changer_column is True
            assert db.get_by_name("Gamechanging Card").is_game_changer is True
            assert db.get_by_name("Normal Card").is_game_changer is False
        finally:
            Path(path).unlink()

    def test_fixture_game_changer_column(self, db):
        # v0.9.18: the shared fixture CSV now carries the isGameChanger
        # column (marked from the authoritative refreshed data). Sol Ring is
        # NOT a Game Changer; Teferi's Protection IS.
        assert db.has_game_changer_column is True
        assert db.get_by_name("Sol Ring").is_game_changer is False
        assert db.get_by_name("Teferi's Protection").is_game_changer is True

    def test_empty_legalities_lenient_when_column_absent(self):
        """A minimal CSV without a legalities column keeps the lenient
        behavior — otherwise the whole database would be excluded."""
        header = ("name|manaCost|manaValue|type|text|colorIdentity|colors|"
                  "power|toughness|loyalty|defense|types|subtypes|supertypes|"
                  "keywords|layout\n")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(header)
            f.write("No Legality Data|{1}|1|Artifact|||||||Artifact|||||normal\n")
            path = f.name
        try:
            db = CardDatabase(path)
            db.load()
            assert db.get_by_name("No Legality Data") is not None
        finally:
            Path(path).unlink()


class TestCardDatabaseQuery:
    def test_query_by_color_identity(self, db):
        result = db.query(color_identity="WG")
        # All results should be W/G-legal
        for c in result.cards:
            card_colors = set(ch for ch in (c.color_identity or "") if ch in "WUBRG")
            assert card_colors.issubset({"W", "G"}), f"{c.name} has color {c.color_identity}"

    def test_query_by_type(self, db):
        result = db.query(card_types=["Creature"])
        for c in result.cards:
            assert "Creature" in c.types

    def test_query_text_pattern(self, db):
        result = db.query(text_pattern="gain.*life")
        # Should match at least some cards
        assert result.total_matches >= 3

    def test_query_combination(self, db):
        """Multiple filters combined with AND logic."""
        result = db.query(color_identity="WG", card_types=["Creature"])
        for c in result.cards:
            card_colors = set(ch for ch in (c.color_identity or "") if ch in "WUBRG")
            assert card_colors.issubset({"W", "G"})
            assert "Creature" in c.types

    def test_colorless_card_passes_all_color_ids(self, db):
        """A colorless card (Sol Ring) should match any color identity."""
        sol_ring = db.get_by_name("Sol Ring")
        # Sol Ring has empty color identity, should be valid for WG deck
        assert db._matches_color_identity(sol_ring, "WG")
        assert db._matches_color_identity(sol_ring, "")

    def test_search_by_name(self, db):
        results = db.search_by_name("lathiel")
        assert len(results) >= 1
        assert any("lathiel" in r.name.lower() for r in results)

    def test_find_similar_names(self, db):
        similar = db.find_similar_names("lathie")
        assert any("lathiel" in n.lower() for n in similar)


class TestCardFillsRole:
    """The critical test suite — these validate the shared role detection
    that both pool generation and evaluator depend on."""

    def test_ramp_mana_rocks(self, db):
        assert card_fills_role(db.get_by_name("Sol Ring"), "ramp")
        assert card_fills_role(db.get_by_name("Arcane Signet"), "ramp")
        assert card_fills_role(db.get_by_name("Selesnya Signet"), "ramp")

    def test_ramp_land_tutors(self, db):
        assert card_fills_role(db.get_by_name("Cultivate"), "ramp")
        assert card_fills_role(db.get_by_name("Rampant Growth"), "ramp")
        assert card_fills_role(db.get_by_name("Kodama's Reach"), "ramp")
        assert card_fills_role(db.get_by_name("Nature's Lore"), "ramp")
        assert card_fills_role(db.get_by_name("Farseek"), "ramp")

    def test_ramp_creatures(self, db):
        assert card_fills_role(db.get_by_name("Birds of Paradise"), "ramp")
        assert card_fills_role(db.get_by_name("Llanowar Elves"), "ramp")
        assert card_fills_role(db.get_by_name("Sakura-Tribe Elder"), "ramp")
        assert card_fills_role(db.get_by_name("Wood Elves"), "ramp")
        assert card_fills_role(db.get_by_name("Farhaven Elf"), "ramp")

    def test_ramp_additional_land(self, db):
        """Cards like Oracle of Mul Daya that grant extra land drops."""
        assert card_fills_role(db.get_by_name("Oracle of Mul Daya"), "ramp")

    def test_not_ramp(self, db):
        """Vanilla creatures, removal spells, etc. should NOT be counted as ramp."""
        assert not card_fills_role(db.get_by_name("Grizzly Bears"), "ramp")
        assert not card_fills_role(db.get_by_name("Forest"), "ramp")  # land not ramp
        assert not card_fills_role(db.get_by_name("Swords to Plowshares"), "ramp")
        assert not card_fills_role(db.get_by_name("Wrath of God"), "ramp")

    def test_draw(self, db):
        assert card_fills_role(db.get_by_name("Harmonize"), "draw")
        assert card_fills_role(db.get_by_name("Concentrate"), "draw")
        assert card_fills_role(db.get_by_name("Sign in Blood"), "draw")
        assert card_fills_role(db.get_by_name("Phyrexian Arena"), "draw")
        assert card_fills_role(db.get_by_name("Windfall"), "draw")
        assert card_fills_role(db.get_by_name("Beast Whisperer"), "draw")
        assert card_fills_role(db.get_by_name("Sylvan Library"), "draw")

    def test_not_draw(self, db):
        assert not card_fills_role(db.get_by_name("Grizzly Bears"), "draw")
        assert not card_fills_role(db.get_by_name("Sol Ring"), "draw")

    def test_removal_single_target(self, db):
        assert card_fills_role(db.get_by_name("Swords to Plowshares"), "removal")
        assert card_fills_role(db.get_by_name("Path to Exile"), "removal")
        assert card_fills_role(db.get_by_name("Generous Gift"), "removal")
        assert card_fills_role(db.get_by_name("Beast Within"), "removal")
        assert card_fills_role(db.get_by_name("Nature's Claim"), "removal")

    def test_wipe_not_removal(self, db):
        """Wipes should NOT be counted as single-target removal."""
        assert not card_fills_role(db.get_by_name("Wrath of God"), "removal")
        assert not card_fills_role(db.get_by_name("Day of Judgment"), "removal")

    def test_wipe(self, db):
        assert card_fills_role(db.get_by_name("Wrath of God"), "wipe")
        assert card_fills_role(db.get_by_name("Day of Judgment"), "wipe")

    def test_not_wipe(self, db):
        """Single-target spells should not be counted as wipes."""
        assert not card_fills_role(db.get_by_name("Swords to Plowshares"), "wipe")

    def test_threat_stat_based(self, db):
        """Big creatures qualify as threats."""
        assert card_fills_role(db.get_by_name("Elesh Norn, Grand Cenobite"), "threat")  # 4/7
        assert card_fills_role(db.get_by_name("Sun Titan"), "threat")  # 6/6
        assert card_fills_role(db.get_by_name("Felidar Sovereign"), "threat")  # 4/6

    def test_threat_text_based(self, db):
        """Cards with 'you win the game' text qualify as threats."""
        assert card_fills_role(db.get_by_name("Felidar Sovereign"), "threat")

    def test_not_threat(self, db):
        """Small creatures and non-creatures should not be threats."""
        assert not card_fills_role(db.get_by_name("Grizzly Bears"), "threat")  # 2/2
        assert not card_fills_role(db.get_by_name("Savannah Lions"), "threat")  # 2/1

    def test_land(self, db):
        assert card_fills_role(db.get_by_name("Forest"), "land")
        assert card_fills_role(db.get_by_name("Command Tower"), "land")
        assert card_fills_role(db.get_by_name("Temple Garden"), "land")

    def test_not_land(self, db):
        assert not card_fills_role(db.get_by_name("Sol Ring"), "land")
        assert not card_fills_role(db.get_by_name("Grizzly Bears"), "land")

    def test_protection(self, db):
        assert card_fills_role(db.get_by_name("Teferi's Protection"), "protection")
        assert card_fills_role(db.get_by_name("Heroic Intervention"), "protection")
        assert card_fills_role(db.get_by_name("Selfless Spirit"), "protection")

    def test_recursion(self, db):
        assert card_fills_role(db.get_by_name("Eternal Witness"), "recursion")
        assert card_fills_role(db.get_by_name("Sun Titan"), "recursion")

    def test_invalid_role(self, db):
        """Unknown role should return False, not crash."""
        assert card_fills_role(db.get_by_name("Sol Ring"), "nonexistent_role") is False


class TestGetCardsForRole:
    def test_wg_ramp(self, db):
        ramp = db.get_cards_for_role("ramp", "WG", limit=100)
        # All should be WG-legal and fill ramp role
        assert len(ramp) >= 5
        for c in ramp:
            card_colors = set(ch for ch in (c.color_identity or "") if ch in "WUBRG")
            assert card_colors.issubset({"W", "G"})

    def test_includes_sol_ring_as_ramp(self, db):
        """Sol Ring is colorless, should be WG-legal ramp."""
        ramp = db.get_cards_for_role("ramp", "WG", limit=100)
        names = {c.name for c in ramp}
        assert "Sol Ring" in names

    def test_respects_limit(self, db):
        result = db.get_cards_for_role("ramp", "WG", limit=3)
        assert len(result) <= 3

    def test_wg_lands(self, db):
        lands = db.get_cards_for_role("land", "WG")
        assert len(lands) >= 3  # At minimum Forest, Plains, Command Tower
        for c in lands:
            assert c.is_land
