"""
Tests for v0.9.9 structural / attribute synergy.

The text-based synergy signals are blind to cards whose payoff is a structural
ATTRIBUTE (vanilla creatures have no text). These predicates reward attributes
directly. Covers the evaluator grammar, prose-cue derivation, the DB query, and
the builder's recall + synergy-floor wiring.
"""

import pytest

from mtg_deck_builder.structural_predicates import (
    card_matches_predicate, card_matches_any, derive_structural_predicates,
)
from mtg_deck_builder.deck_builder import DeckBuilder
from mtg_deck_builder.models import Card, CommanderAnalysis, BuildConfig
from mtg_deck_builder.llm_engine import LLMConfig


def _card(name="X", text="", types="Creature", subtypes="", supertypes="",
          keywords="", colors="G", mv=3, power="3", toughness="3") -> Card:
    return Card(
        name=name, mana_cost="{2}{G}", mana_value=mv,
        card_type=f"{supertypes} {types} — {subtypes}".strip(), text=text,
        color_identity=colors, colors=colors,
        power=power, toughness=toughness, loyalty="", defense="",
        types=types, subtypes=subtypes, supertypes=supertypes,
        keywords=keywords, layout="normal", legalities="commander:legal",
    )


# ----------------------------------------------------------------------
# Predicate grammar
# ----------------------------------------------------------------------

class TestPredicateGrammar:
    def test_vanilla(self):
        van = _card(text="", keywords="")
        nonvan = _card(text="Trample.", keywords="Trample")
        assert card_matches_predicate(van, "vanilla") is True
        assert card_matches_predicate(van, "no_abilities") is True
        assert card_matches_predicate(nonvan, "vanilla") is False

    def test_colorless(self):
        assert card_matches_predicate(_card(colors=""), "colorless") is True
        assert card_matches_predicate(_card(colors="G"), "colorless") is False

    def test_mana_value(self):
        c = _card(mv=2)
        assert card_matches_predicate(c, "mv<=2") is True
        assert card_matches_predicate(c, "mv<2") is False
        assert card_matches_predicate(c, "cmc>=6") is False
        assert card_matches_predicate(c, "mv==2") is True

    def test_power_toughness(self):
        c = _card(power="5", toughness="1")
        assert card_matches_predicate(c, "power>=4") is True
        assert card_matches_predicate(c, "toughness<=1") is True
        assert card_matches_predicate(c, "power<4") is False

    def test_nonnumeric_power_never_matches(self):
        c = _card(power="*", toughness="*")
        assert card_matches_predicate(c, "power>=4") is False

    def test_subtype_type_keyword(self):
        c = _card(subtypes="Bear, Druid", types="Creature", keywords="Trample")
        assert card_matches_predicate(c, "subtype:Bear") is True
        assert card_matches_predicate(c, "subtype:Elf") is False
        assert card_matches_predicate(c, "type:Creature") is True
        assert card_matches_predicate(c, "keyword:Trample") is True

    def test_multiword_keyword_and_subtype(self):
        # v0.9.13: multi-word values ("First strike", "Time Lord") must match
        # as full phrases — the old whitespace tokenizer split them apart so
        # "keyword:first strike" could never match anything.
        c = _card(keywords="First strike, Lifelink", subtypes="Time Lord, Doctor")
        assert card_matches_predicate(c, "keyword:first strike") is True
        assert card_matches_predicate(c, "keyword:lifelink") is True
        assert card_matches_predicate(c, "subtype:time lord") is True
        assert card_matches_predicate(c, "subtype:doctor") is True
        assert card_matches_predicate(c, "keyword:double strike") is False

    def test_unknown_predicate_is_false(self):
        assert card_matches_predicate(_card(), "wibble") is False
        assert card_matches_predicate(_card(), "") is False

    def test_matches_any(self):
        c = _card(colors="", text="")  # colorless vanilla
        assert card_matches_any(c, ["mv<=1", "colorless"]) is True
        assert card_matches_any(c, ["mv<=1", "subtype:Elf"]) is False


# ----------------------------------------------------------------------
# Prose-cue derivation
# ----------------------------------------------------------------------

class TestDerive:
    def _an(self, build="", notes="", preds=None):
        return CommanderAnalysis(
            name="C", color_identity="G", key_mechanics=[],
            build_around_text=build, evaluation_notes=notes,
            category_queries={}, synergy_keywords=[],
            structural_predicates=preds or [],
        )

    def test_explicit_predicates_kept(self):
        assert "vanilla" in derive_structural_predicates(
            self._an(preds=["vanilla"]))

    def test_derives_vanilla_from_prose(self):
        an = self._an(build="rewards a deck full of vanilla creatures")
        assert "vanilla" in derive_structural_predicates(an)

    def test_derives_from_no_abilities_phrase(self):
        an = self._an(build="creatures with no abilities are the payoff")
        assert "vanilla" in derive_structural_predicates(an)

    def test_evaluation_notes_aside_does_not_trigger(self):
        # Regression (real Lathiel run): evaluation_notes saying "creatures
        # that are otherwise vanilla or french-vanilla are better here" must
        # NOT flip the structural machinery for a lifegain commander. Cues
        # only count in build_around_text (the core strategy).
        an = self._an(
            build="gain life in large bursts and distribute +1/+1 counters",
            notes="creatures that are otherwise vanilla or french-vanilla "
                  "are significantly better here than their stats suggest",
        )
        assert derive_structural_predicates(an) == []

    def test_normal_commander_no_predicates(self):
        an = self._an(build="gain life and make tokens")
        assert derive_structural_predicates(an) == []

    def test_colorless_in_passing_no_predicate(self):
        # v0.9.13: the bare word "colorless" is not specific enough — nearly
        # any analysis can mention "colorless mana rocks" in passing, and a
        # false positive here flips the entire structural machinery.
        an = self._an(notes="even colorless mana rocks like Sol Ring are fine")
        assert "colorless" not in derive_structural_predicates(an)

    def test_colorless_matters_derives_predicate(self):
        an = self._an(build="rewards colorless creatures like Eldrazi")
        assert "colorless" in derive_structural_predicates(an)


# ----------------------------------------------------------------------
# DB query
# ----------------------------------------------------------------------

class TestDbQuery:
    def test_returns_only_matching(self, test_csv_path):
        from mtg_deck_builder.card_database import CardDatabase
        db = CardDatabase(test_csv_path)
        db.load()
        out = db.get_cards_matching_predicates(["vanilla"], "G,W", limit=50)
        # Everything returned is actually vanilla (and we didn't crash).
        assert all(card_matches_predicate(c, "vanilla") for c in out)

    def test_empty_predicates_returns_empty(self, test_csv_path):
        from mtg_deck_builder.card_database import CardDatabase
        db = CardDatabase(test_csv_path)
        db.load()
        assert db.get_cards_matching_predicates([], "G", limit=10) == []


# ----------------------------------------------------------------------
# Builder wiring (recall + boost)
# ----------------------------------------------------------------------

class TestBuilderWiring:
    def _builder(self, test_csv_path, mode="on") -> DeckBuilder:
        # Lathiel is present in the small test CSV; the structural predicate is
        # injected directly, so the commander identity (G,W) is all we need.
        cfg = BuildConfig(commander_name="Lathiel, the Bounteous Dawn",
                          random_seed=42, structural_synergy_mode=mode,
                          structural_boost_floor=85.0)
        b = DeckBuilder(card_database_path=test_csv_path, config=cfg,
                        llm_config=LLMConfig(mock_mode=True))
        b._commander = b.db.get_by_name("Lathiel, the Bounteous Dawn")
        b._analysis = CommanderAnalysis(
            name="Lathiel, the Bounteous Dawn", color_identity="G,W",
            key_mechanics=[], build_around_text="vanilla creatures matter",
            evaluation_notes="", category_queries={}, synergy_keywords=[],
            structural_predicates=["vanilla"],
        )
        return b

    def test_structural_boost_floors_synergy(self, test_csv_path):
        b = self._builder(test_csv_path)
        b._structural_card_names = {"Grizzly Bears", "Other"}
        syn = {"Grizzly Bears": 5.0, "Other": 92.0, "Untouched": 50.0}
        b._apply_structural_boost(syn)
        assert syn["Grizzly Bears"] == 85.0   # floored up
        assert syn["Other"] == 92.0           # already above floor
        assert syn["Untouched"] == 50.0       # not a structural match

    def test_body_power_scales_with_stats(self):
        big = _card(power="10", toughness="10", mv=5)
        small = _card(power="1", toughness="1", mv=1)
        bp_big = DeckBuilder._body_power(big)
        bp_small = DeckBuilder._body_power(small)
        assert bp_big > bp_small
        assert 60 <= bp_big <= 100      # a 10/10 is a strong body
        assert bp_small < 30            # a 1/1 is not

    def test_body_power_none_for_noncreature_or_star(self):
        assert DeckBuilder._body_power(
            _card(types="Land", power="", toughness="")) is None
        assert DeckBuilder._body_power(
            _card(power="*", toughness="*")) is None

    def test_power_floor_lifts_matching_creature(self, test_csv_path):
        b = self._builder(test_csv_path)
        cmd = "Lathiel, the Bounteous Dawn"   # a creature present in the test DB
        b._structural_card_names = {cmd}
        baseline = {cmd: 5.0}
        b._apply_structural_power_floor(baseline)
        assert baseline[cmd] > 5.0            # floored up to its body power

    def test_power_floor_only_raises(self, test_csv_path):
        b = self._builder(test_csv_path)
        cmd = "Lathiel, the Bounteous Dawn"
        b._structural_card_names = {cmd}
        baseline = {cmd: 99.0}                # already high
        b._apply_structural_power_floor(baseline)
        assert baseline[cmd] == 99.0          # never lowered

    def test_power_floor_off_mode_noop(self, test_csv_path):
        b = self._builder(test_csv_path, mode="off")
        b._structural_card_names = {"Lathiel, the Bounteous Dawn"}
        baseline = {"Lathiel, the Bounteous Dawn": 5.0}
        b._apply_structural_power_floor(baseline)
        assert baseline["Lathiel, the Bounteous Dawn"] == 5.0

    def test_floor_skipped_when_scored_by_llm(self, test_csv_path):
        # v0.9.10: when the LLM rubric reasoned the values, the flat floor is
        # skipped so it doesn't clobber the nuanced scores.
        b = self._builder(test_csv_path)
        b._structural_scored_by_llm = True
        b._structural_card_names = {"Grizzly Bears"}
        syn = {"Grizzly Bears": 30.0}
        b._apply_structural_boost(syn)
        assert syn["Grizzly Bears"] == 30.0  # untouched — LLM is authoritative

    def test_boost_off_mode_noop(self, test_csv_path):
        b = self._builder(test_csv_path, mode="off")
        b._structural_card_names = {"Grizzly Bears"}
        syn = {"Grizzly Bears": 5.0}
        b._apply_structural_boost(syn)
        assert syn["Grizzly Bears"] == 5.0

    def test_augment_patterns_for_structural(self, test_csv_path):
        # When the commander has a "vanilla" predicate, complementary text
        # patterns are added so attribute-payoffs (Ruxa) get recalled.
        b = self._builder(test_csv_path)
        b._analysis.synergy_patterns = ["existing"]
        b._augment_patterns_for_structural()
        pats = [p.lower() for p in b._analysis.synergy_patterns]
        assert "no abilities" in pats
        assert "vanilla" in pats
        assert "existing" in pats  # didn't clobber existing patterns

    def test_augment_noop_for_text_commander(self, test_csv_path):
        b = self._builder(test_csv_path)
        b._analysis.build_around_text = "gain life and make tokens"
        b._analysis.structural_predicates = []
        b._analysis.synergy_patterns = ["gain life"]
        b._augment_patterns_for_structural()
        assert b._analysis.synergy_patterns == ["gain life"]

    def test_combo_detect_unions_onramp_not_clobber(self, test_csv_path):
        # Regression: _phase_detect_combos must UNION its on-ramp names, not
        # overwrite the structural recall's flags (which run earlier).
        from mtg_deck_builder.models import ComboReport, Combo
        b = self._builder(test_csv_path)
        b._phase_generate_pools()
        b.config.combo_mode = "llm"
        b._onramp_names = {"Grizzly Bears"}  # pretend structural flagged it

        class _Stub:
            def detect(self, analysis, pool, edhrec_fallback=None):
                return ComboReport(
                    combos=[Combo(cards=["Sol Ring", "Mana Vault"], payoff=90)],
                    engines={"Sol Ring": "n"})
        b._get_combo_detector = lambda: _Stub()
        b._phase_detect_combos()
        # Structural flag survived; combo cards were added on top.
        assert "Grizzly Bears" in b._onramp_names

    def test_payoff_recall_guarantees_petroglyphs_class(self, test_csv_path):
        # v0.9.13 regression: attribute-PAYOFF cards (text referencing the
        # attribute — Muraganda Petroglyphs, Ruxa) must be pulled into the
        # pool and on-ramped, not left to win the synergy_engine cosine
        # pre-rank race (which dropped Petroglyphs from a real build).
        b = self._builder(test_csv_path)
        b._phase_generate_pools()
        b._phase_structural_recall()
        pool = {c.name for c in b._candidates.synergy}
        assert "Muraganda Petroglyphs" in pool
        assert "Ruxa, Patient Professor" in pool
        assert "Muraganda Petroglyphs" in b._onramp_names
        assert "Ruxa, Patient Professor" in b._onramp_names
        # Payoffs are NOT floored like attribute bodies — they have real
        # text and get honest LLM scores.
        assert "Muraganda Petroglyphs" not in b._structural_card_names

    def test_recall_pulls_vanilla_into_pool_and_onramp(self, test_csv_path):
        b = self._builder(test_csv_path)
        b._phase_generate_pools()
        before = {c.name for c in b._candidates.synergy}
        b._phase_structural_recall()
        after = {c.name for c in b._candidates.synergy}
        assert b._structural_predicates == ["vanilla"]
        # If the test DB has any G/W vanilla creatures, they're flagged for the
        # on-ramp and present in the pool.
        assert b._structural_card_names == (
            b._structural_card_names & after
        )  # every flagged card is in the pool
        assert b._onramp_names >= b._structural_card_names
