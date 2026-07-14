"""
Tests for v0.9.8 LLM combo/engine detection (the SOURCE layer).

Covers:
  - pool pass + knowledge pass parsing into a ComboReport;
  - knowledge-pass combo cards absent from the pool become missing_pieces;
  - dedupe keeps the higher-payoff combo for a duplicate card set;
  - per-commander disk cache round-trips and invalidates on pool change;
  - mock mode yields an empty report (no crash);
  - EDHREC fallback fires ONLY when the LLM produced nothing.

No real API: a fake LLM returns canned JSON keyed on which pass called it.
"""

import json

import pytest

from mtg_deck_builder.combo_engine_detector import (
    ComboEngineDetector, _extract_json_object,
)
from mtg_deck_builder.models import Card, CommanderAnalysis, Combo, ComboReport


def _card(name: str) -> Card:
    return Card(
        name=name, mana_cost="{1}{W}", mana_value=2,
        card_type="Creature", text=f"text {name}",
        color_identity="W", colors="W",
        power="1", toughness="1", loyalty="", defense="",
        types="Creature", subtypes="", supertypes="", keywords="",
        layout="normal", legalities="commander:legal",
    )


def _analysis() -> CommanderAnalysis:
    return CommanderAnalysis(
        name="Lathiel, the Bounteous Dawn", color_identity="G,W",
        key_mechanics=["lifegain"], build_around_text="gain life -> counters",
        evaluation_notes="...", category_queries={},
        synergy_keywords=["gain life", "+1/+1 counter"],
    )


class _Cfg:
    def __init__(self, mock=False):
        self.mock_mode = mock


class _FakeLLM:
    """Returns canned JSON per pass; counts calls."""

    def __init__(self, pool_resp: str, knowledge_resp: str, mock=False,
                 verify_resp: str = '{"invalid": []}',
                 signature_resp: str = '{"combos": []}',
                 deepen_resp: str = '{"combos": [], "engines": []}',
                 rate_resp: str = '{"ratings": []}'):
        self.config = _Cfg(mock)
        self.pool_resp = pool_resp
        self.knowledge_resp = knowledge_resp
        self.verify_resp = verify_resp
        self.signature_resp = signature_resp
        self.deepen_resp = deepen_resp
        self.rate_resp = rate_resp
        self.calls = 0
        self.verify_calls = 0
        self.signature_calls = 0
        self.deepen_calls = 0
        self.rate_calls = 0

    def _call_api(self, system, user, temperature=None, max_tokens=None,
                  model=None):
        self.calls += 1
        if "RULES EXPERT" in system:  # v0.9.15b verification pass
            self.verify_calls += 1
            return self.verify_resp
        if "combo historian" in system:  # v0.9.16c signature pass
            self.signature_calls += 1
            return self.signature_resp
        if "COMBO POWER RATER" in system:  # v0.9.30 database rating pass
            self.rate_calls += 1
            return self.rate_resp
        if "ALREADY FOUND" in user:  # v0.9.30 deepening pass
            self.deepen_calls += 1
            return self.deepen_resp
        return self.pool_resp if "candidate cards" in system else self.knowledge_resp


POOL_JSON = json.dumps({
    "combos": [
        {"cards": ["Spike Feeder", "Archangel of Thune"],
         "payoff": 95, "result": "infinite life + counters"},
    ],
    "engines": [{"name": "Soul Warden", "note": "scales with creatures"}],
})

KNOWLEDGE_JSON = json.dumps({
    "combos": [
        # Heliod is NOT in the pool below -> becomes a missing piece.
        {"cards": ["Spike Feeder", "Heliod, Sun-Crowned"],
         "payoff": 96, "result": "infinite life"},
    ],
    "engines": [{"name": "Walking Ballista", "note": "combo payoff"}],
})


# ----------------------------------------------------------------------
# JSON extraction
# ----------------------------------------------------------------------

class TestExtractJsonObject:
    def test_plain(self):
        assert _extract_json_object('{"a":1}') == {"a": 1}

    def test_fenced_and_prose(self):
        text = 'Sure:\n```json\n{"a":1}\n```\ndone'
        assert _extract_json_object(text) == {"a": 1}

    def test_raises_without_object(self):
        with pytest.raises(ValueError):
            _extract_json_object("no object")


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------

class TestDetect:
    def _detector(self, tmp_path, **kw):
        llm = _FakeLLM(POOL_JSON, KNOWLEDGE_JSON, **kw)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        return det, llm

    def test_pool_and_knowledge_merged(self, tmp_path):
        det, _ = self._detector(tmp_path)
        pool = [_card("Spike Feeder"), _card("Archangel of Thune")]
        report = det.detect(_analysis(), pool)
        sets = {frozenset(c.cards) for c in report.combos}
        assert frozenset({"Spike Feeder", "Archangel of Thune"}) in sets
        assert frozenset({"Spike Feeder", "Heliod, Sun-Crowned"}) in sets
        assert "Soul Warden" in report.engines
        assert "Walking Ballista" in report.engines

    def test_missing_pieces_from_knowledge(self, tmp_path):
        det, _ = self._detector(tmp_path)
        pool = [_card("Spike Feeder"), _card("Archangel of Thune")]
        report = det.detect(_analysis(), pool)
        # Heliod isn't in the pool -> flagged to pull into recall.
        assert "Heliod, Sun-Crowned" in report.missing_pieces
        # Spike Feeder IS in the pool -> not a missing piece.
        assert "Spike Feeder" not in report.missing_pieces

    def test_dedupe_keeps_higher_payoff(self, tmp_path):
        pool_resp = json.dumps({"combos": [
            {"cards": ["A", "B"], "payoff": 60, "result": "x"}], "engines": []})
        know_resp = json.dumps({"combos": [
            {"cards": ["B", "A"], "payoff": 90, "result": "x"}], "engines": []})
        det = ComboEngineDetector(_FakeLLM(pool_resp, know_resp),
                                  cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("A"), _card("B")])
        ab = [c for c in report.combos if frozenset(c.cards) == frozenset({"A", "B"})]
        assert len(ab) == 1
        assert ab[0].payoff == 90.0

    def test_salvages_truncated_response(self, tmp_path):
        # Simulate a max_tokens truncation: a valid wrapper that gets cut off
        # mid-array. The two complete combos before the cut must be recovered.
        truncated = (
            '{"combos": ['
            '{"cards": ["A", "B"], "payoff": 90, "result": "infinite life"},'
            '{"cards": ["C", "D"], "payoff": 80, "result": "infinite mana"},'
            '{"cards": ["E", "F"], "payoff": 70, "resu'  # <- cut here
        )
        det = ComboEngineDetector(
            _FakeLLM(truncated, '{"combos":[],"engines":[]}'),
            cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("A"), _card("B")])
        sets = {frozenset(c.cards) for c in report.combos}
        assert frozenset({"A", "B"}) in sets
        assert frozenset({"C", "D"}) in sets   # both complete ones salvaged
        assert frozenset({"E", "F"}) not in sets  # the cut-off one is dropped

    def test_two_card_minimum(self, tmp_path):
        pool_resp = json.dumps({"combos": [
            {"cards": ["Solo"], "payoff": 80, "result": "nope"}], "engines": []})
        det = ComboEngineDetector(_FakeLLM(pool_resp, '{"combos":[],"engines":[]}'),
                                  cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("Solo")])
        assert report.combos == []  # single-card "combo" dropped

    def test_mock_mode_empty(self, tmp_path):
        det = ComboEngineDetector(
            _FakeLLM(POOL_JSON, KNOWLEDGE_JSON, mock=True), cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("Spike Feeder")])
        assert report.combos == [] and report.engines == {}

    def test_edhrec_fallback_only_when_empty(self, tmp_path):
        empty = '{"combos":[],"engines":[]}'
        det = ComboEngineDetector(_FakeLLM(empty, empty), cache_dir=str(tmp_path))
        fb = ComboReport(combos=[Combo(cards=["X", "Y"], payoff=70,
                                       source="edhrec")])
        report = det.detect(_analysis(), [_card("X")], edhrec_fallback=lambda: fb)
        assert any(c.source == "edhrec" for c in report.combos)

    def test_fallback_not_used_when_llm_succeeds(self, tmp_path):
        det, _ = self._detector(tmp_path)
        called = {"v": False}

        def fb():
            called["v"] = True
            return ComboReport()
        det.detect(_analysis(), [_card("Spike Feeder")], edhrec_fallback=fb)
        assert called["v"] is False  # LLM produced combos -> no fallback


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

class TestSignaturePass:
    """v0.9.16c: the signature pass is RECALL-ONLY — it pulls niche combo
    pieces into missing_pieces without touching the combo score/on-ramp."""

    SIG = json.dumps({"combos": [
        {"cards": ["Selenia, Dark Angel", "Mirror Universe"],
         "result": "Flip life totals at 1 life to kill"},
        {"cards": ["Exquisite Blood", "Sanguine Bond"], "result": "Infinite"},
    ]})

    def test_niche_piece_pulled_into_recall_only(self, tmp_path):
        # Mirror Universe is NOT in the pool -> becomes a recall target.
        # Crucially it is NOT added to report.combos (no scoring weight).
        llm = _FakeLLM('{"combos":[],"engines":[]}',
                       '{"combos":[],"engines":[]}', signature_resp=self.SIG)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("Exquisite Blood"),
                                          _card("Sanguine Bond")])
        assert "Mirror Universe" in report.missing_pieces
        # Signature combos recorded (informational) but NOT scored.
        assert any("Mirror Universe" in c["cards"]
                   for c in report.signature_combos)
        assert all("Mirror Universe" not in c.cards for c in report.combos)
        assert llm.signature_calls == 1

    def test_toggle_off_skips_pass(self, tmp_path):
        llm = _FakeLLM('{"combos":[],"engines":[]}',
                       '{"combos":[],"engines":[]}', signature_resp=self.SIG)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path),
                                  signature_pass=False)
        report = det.detect(_analysis(), [_card("Exquisite Blood")])
        assert llm.signature_calls == 0
        assert "Mirror Universe" not in report.missing_pieces

    def test_signature_does_not_inflate_combo_count(self, tmp_path):
        # A signature combo whose pieces are ALL in the pool still must not
        # be added to report.combos (recall-only).
        llm = _FakeLLM('{"combos":[],"engines":[]}',
                       '{"combos":[],"engines":[]}', signature_resp=self.SIG)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("Exquisite Blood"),
                                          _card("Sanguine Bond"),
                                          _card("Selenia, Dark Angel"),
                                          _card("Mirror Universe")])
        assert report.combos == []  # nothing scored from signature
        assert len(report.signature_combos) == 2


class TestComboMemory:
    """v0.9.29: combos are facts about cards — a fresh detection (pool-hash
    miss) merges the previous VERIFIED cache instead of discarding it.
    Regression: Doomsday+Citadel [92] detected in three consecutive runs,
    absent from the fourth's fresh LLM sample, silently cut from the deck."""

    OLD = json.dumps({"combos": [
        {"cards": ["Doomsday", "Bolas's Citadel"],
         "result": "Stack a pile, cast it for life", "payoff": 92}],
        "engines": [{"name": "Old Engine", "description": "engine desc"}]})
    NEW = json.dumps({"combos": [
        {"cards": ["Spike Feeder", "Archangel of Thune"],
         "result": "Infinite life", "payoff": 95}], "engines": []})

    def test_fresh_detection_merges_previous_verified(self, tmp_path):
        # Run 1: old combo detected and cached (verified).
        llm = _FakeLLM(self.OLD, '{"combos":[],"engines":[]}')
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        pool1 = [_card("Doomsday"), _card("Bolas's Citadel")]
        det.detect(_analysis(), pool1)
        # Run 2: pool changed (hash miss) and the fresh sample does NOT
        # re-find the old combo.
        llm2 = _FakeLLM(self.NEW, '{"combos":[],"engines":[]}')
        det2 = ComboEngineDetector(llm2, cache_dir=str(tmp_path))
        pool2 = [_card("Spike Feeder"), _card("Archangel of Thune")]
        report = det2.detect(_analysis(), pool2)
        combo_sets = {frozenset(c.cards) for c in report.combos}
        assert frozenset({"Doomsday", "Bolas's Citadel"}) in combo_sets
        assert frozenset({"Spike Feeder", "Archangel of Thune"}) in combo_sets
        # Merged pieces absent from the new pool become recall targets.
        assert "Doomsday" in report.missing_pieces
        # And the union is what got cached for next time.
        det3 = ComboEngineDetector(_FakeLLM("x", "y"),
                                   cache_dir=str(tmp_path))
        cached = det3.detect(_analysis(), pool2)  # hash hit -> no LLM calls
        assert len(cached.combos) == 2

    def test_unverified_previous_cache_not_merged(self, tmp_path):
        import os
        llm = _FakeLLM(self.OLD, '{"combos":[],"engines":[]}')
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        det.detect(_analysis(), [_card("Doomsday")])
        # Stamp the cache unverified (simulates a failed verify run).
        path = [f for f in os.listdir(tmp_path)][0]
        import json as _json
        p = tmp_path / path
        data = _json.loads(p.read_text(encoding="utf-8"))
        data["verified"] = False
        p.write_text(_json.dumps(data), encoding="utf-8")
        llm2 = _FakeLLM(self.NEW, '{"combos":[],"engines":[]}')
        det2 = ComboEngineDetector(llm2, cache_dir=str(tmp_path))
        report = det2.detect(_analysis(), [_card("Spike Feeder"),
                                           _card("Archangel of Thune")])
        assert {frozenset(c.cards) for c in report.combos} == {
            frozenset({"Spike Feeder", "Archangel of Thune"})}


class TestDatabaseCombos:
    """v0.9.30: EDHREC/Commander Spellbook combos merge as a deterministic,
    pre-verified backbone — rated by one LLM call, never re-verified."""

    DB = [{"cards": ["Doomsday", "Bolas's Citadel"], "decks": 9000},
          {"cards": ["Spike Feeder", "Archangel of Thune"], "decks": 5000}]
    RATE = json.dumps({"ratings": [
        {"cards": ["Doomsday", "Bolas's Citadel"], "payoff": 93}]})

    def test_ingested_rated_and_deduped(self, tmp_path):
        # LLM already found Spike+Archangel — the database copy must dedupe;
        # Doomsday+Citadel is new and gets the rated payoff.
        llm = _FakeLLM(POOL_JSON, '{"combos":[],"engines":[]}',
                       rate_resp=self.RATE)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        pool = [_card("Spike Feeder"), _card("Archangel of Thune")]
        report = det.detect(_analysis(), pool, database_combos=self.DB)
        sets = {frozenset(c.cards): c for c in report.combos}
        assert len(sets) == 2  # deduped
        dd = sets[frozenset({"Doomsday", "Bolas's Citadel"})]
        assert dd.payoff == 93 and dd.source == "database"
        assert llm.rate_calls == 1
        # Pieces absent from the pool become recall targets.
        assert "Doomsday" in report.missing_pieces

    def test_mock_mode_flat_payoff_no_calls(self, tmp_path):
        llm = _FakeLLM(POOL_JSON, KNOWLEDGE_JSON, mock=True)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("Spike Feeder")],
                            database_combos=self.DB)
        db = [c for c in report.combos if c.source == "database"]
        assert len(db) == 2 and all(c.payoff == 80 for c in db)
        assert llm.calls == 0  # mock: no LLM passes, no rating call

    def test_cache_hit_still_merges_new_database_combos(self, tmp_path):
        llm = _FakeLLM(POOL_JSON, KNOWLEDGE_JSON, rate_resp=self.RATE)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        pool = [_card("Spike Feeder"), _card("Archangel of Thune")]
        det.detect(_analysis(), pool)          # populates the cache
        calls_after_first = llm.calls
        det2 = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        report = det2.detect(_analysis(), pool, database_combos=self.DB)
        assert frozenset({"Doomsday", "Bolas's Citadel"}) in {
            frozenset(c.cards) for c in report.combos}
        # Only the rating call was spent (no detection passes re-run).
        assert llm.calls == calls_after_first + 1


class TestCache:
    def test_cache_hit_skips_llm(self, tmp_path):
        llm = _FakeLLM(POOL_JSON, KNOWLEDGE_JSON)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        pool = [_card("Spike Feeder"), _card("Archangel of Thune")]
        det.detect(_analysis(), pool)
        # pool + knowledge + deepening (v0.9.30) + signature (v0.9.16c)
        # + verification (v0.9.15b)
        assert llm.calls == 5
        # Re-detect, same pool -> cache hit, no new calls.
        det2 = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        det2.detect(_analysis(), pool)
        assert llm.calls == 5

    def test_unverified_cache_invalidated(self, tmp_path):
        # v0.9.15b: caches written before the verification pass may contain
        # rules-invalid combos — they must be treated as misses.
        llm = _FakeLLM(POOL_JSON, KNOWLEDGE_JSON)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        pool = [_card("Spike Feeder"), _card("Archangel of Thune")]
        det.detect(_analysis(), pool)
        # Strip the verified flag to simulate a pre-upgrade cache.
        import glob
        path = glob.glob(str(tmp_path / "*.json"))[0]
        data = json.loads(open(path, encoding="utf-8").read())
        del data["verified"]
        open(path, "w", encoding="utf-8").write(json.dumps(data))
        n = llm.calls
        det.detect(_analysis(), pool)
        assert llm.calls > n  # re-detected


# ----------------------------------------------------------------------
# v0.9.15b: verification sub-pass
# ----------------------------------------------------------------------

class TestVerification:
    def test_invalid_combo_rejected(self, tmp_path):
        # Regression (real Kinnan run): the pool pass emitted "Food Chain +
        # Kinnan" (payoff 92) — rules-invalid, since Kinnan only doubles
        # mana from TAPPING permanents. The verification pass must drop it
        # while keeping the genuine combo.
        pool_resp = json.dumps({"combos": [
            {"cards": ["Food Chain", "Kinnan"], "payoff": 92,
             "result": "infinite mana (bogus)"},
            {"cards": ["Spike Feeder", "Archangel of Thune"], "payoff": 95,
             "result": "infinite life"},
        ], "engines": []})
        # NOTE: verification runs after _dedupe, which sorts by payoff
        # descending — Spike Feeder (95) is #1, Food Chain (92) is #2.
        llm = _FakeLLM(pool_resp, '{"combos":[],"engines":[]}',
                       verify_resp='{"invalid": [2]}')
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("Food Chain"),
                                          _card("Spike Feeder")])
        sets = {frozenset(c.cards) for c in report.combos}
        assert frozenset({"Food Chain", "Kinnan"}) not in sets
        assert frozenset({"Spike Feeder", "Archangel of Thune"}) in sets
        assert llm.verify_calls == 1

    def test_verification_failure_keeps_all(self, tmp_path):
        # A garbled verification response must never LOSE genuine combos.
        llm = _FakeLLM(POOL_JSON, KNOWLEDGE_JSON,
                       verify_resp="not json at all")
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("Spike Feeder")])
        assert len(report.combos) == 2  # both survive

    def test_failed_verification_not_cached_as_verified(self, tmp_path):
        # Regression (real Kinnan run): a failed verification pass used to
        # stamp the cache verified=True anyway, so the next run would trust
        # unverified combos. It must cache verified=False (-> cache miss).
        llm = _FakeLLM(POOL_JSON, KNOWLEDGE_JSON,
                       verify_resp="not json at all")
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        pool = [_card("Spike Feeder")]
        det.detect(_analysis(), pool)
        import glob
        path = glob.glob(str(tmp_path / "*.json"))[0]
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["verified"] is False
        # Same pool again -> unverified cache is a MISS -> full re-run.
        n = llm.calls
        det.detect(_analysis(), pool)
        assert llm.calls > n

    def test_verification_parses_prose_with_mana_symbols(self, tmp_path):
        # Regression (real Kinnan run): the verify response quoted card text
        # containing "{T}: Add..." — first-brace extraction landed on {T}
        # and raised "Expecting property name". The parser must fall back to
        # scanning every balanced dict for the verdict.
        pool_resp = json.dumps({"combos": [
            {"cards": ["A", "B"], "payoff": 90, "result": "real"},
            {"cards": ["C", "D"], "payoff": 80, "result": "fake"},
        ], "engines": []})
        noisy_verdict = (
            'Combo 2 does not work: {T}: Add {C} is a mana ability and the '
            'trigger never fires.\n{"invalid": [2]}'
        )
        llm = _FakeLLM(pool_resp, '{"combos":[],"engines":[]}',
                       verify_resp=noisy_verdict)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        report = det.detect(_analysis(), [_card("A"), _card("C")])
        sets = {frozenset(c.cards) for c in report.combos}
        assert frozenset({"A", "B"}) in sets
        assert frozenset({"C", "D"}) not in sets  # verdict applied
        import glob
        path = glob.glob(str(tmp_path / "*.json"))[0]
        assert json.loads(open(path, encoding="utf-8").read())["verified"] is True

    def test_verified_report_cached(self, tmp_path):
        pool_resp = json.dumps({"combos": [
            {"cards": ["A", "B"], "payoff": 90, "result": "real"},
            {"cards": ["C", "D"], "payoff": 80, "result": "fake"},
        ], "engines": []})
        llm = _FakeLLM(pool_resp, '{"combos":[],"engines":[]}',
                       verify_resp='{"invalid": [2]}')
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        pool = [_card("A"), _card("B"), _card("C"), _card("D")]
        det.detect(_analysis(), pool)
        # Cache holds only the verified combo, flagged verified.
        import glob
        path = glob.glob(str(tmp_path / "*.json"))[0]
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["verified"] is True
        assert len(data["combos"]) == 1
        assert set(data["combos"][0]["cards"]) == {"A", "B"}

    def test_empty_report_not_cached(self, tmp_path):
        # A failed/empty detection must not poison the cache.
        empty = '{"combos":[],"engines":[]}'
        llm = _FakeLLM(empty, empty)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        det.detect(_analysis(), [_card("A")])
        n = llm.calls
        # Re-run: no cached empty report -> detection runs again.
        det.detect(_analysis(), [_card("A")])
        assert llm.calls > n

    def test_pool_change_invalidates(self, tmp_path):
        llm = _FakeLLM(POOL_JSON, KNOWLEDGE_JSON)
        det = ComboEngineDetector(llm, cache_dir=str(tmp_path))
        det.detect(_analysis(), [_card("Spike Feeder")])
        n = llm.calls
        # Different pool -> different hash -> re-run.
        det.detect(_analysis(), [_card("Spike Feeder"), _card("Brand New Card")])
        assert llm.calls > n
