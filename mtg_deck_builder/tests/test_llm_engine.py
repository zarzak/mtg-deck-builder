"""Tests for LLMEngine — primarily in mock mode since we don't want API calls in tests."""

import pytest
from mtg_deck_builder.llm_engine import LLMEngine, LLMConfig
from mtg_deck_builder.models import Deck


class TestMockMode:
    def test_mock_analyze_lathiel(self, mock_llm, lathiel):
        """Mock analysis should identify Lathiel's lifegain mechanics."""
        analysis = mock_llm.analyze_commander(lathiel)
        assert analysis.name == lathiel.name
        assert any("life" in m.lower() for m in analysis.key_mechanics)

    def test_mock_analyze_karlov(self, mock_llm, karlov):
        """Mock analysis should identify Karlov's lifegain + counters."""
        analysis = mock_llm.analyze_commander(karlov)
        assert any("life" in m.lower() or "counter" in m.lower()
                   for m in analysis.key_mechanics)

    def test_mock_analyze_caches(self, mock_llm, lathiel):
        """Second call should return cached result (same object)."""
        a1 = mock_llm.analyze_commander(lathiel)
        a2 = mock_llm.analyze_commander(lathiel)
        assert a1 is a2

    def test_mock_select_cards(self, mock_llm, lathiel_analysis, db):
        """Mock card selection should return deterministic high-synergy cards."""
        candidates = db.all_cards[:30]
        selected = mock_llm.select_cards(
            lathiel_analysis, candidates, role="synergy", count=5,
        )
        assert len(selected) <= 5
        # All selections should be valid names from the pool
        candidate_names = {c.name for c in candidates}
        for name in selected:
            assert name in candidate_names

    def test_mock_select_respects_already_selected(self, mock_llm, lathiel_analysis, db):
        """Already-selected cards should be excluded."""
        candidates = db.all_cards[:30]
        first = candidates[0].name
        selected = mock_llm.select_cards(
            lathiel_analysis, candidates, role="synergy", count=10,
            already_selected={first},
        )
        assert first not in selected

    def test_mock_score_synergy_batch(self, mock_llm, lathiel_analysis, db):
        cards = db.all_cards[:15]
        scores = mock_llm.score_synergy_batch(lathiel_analysis, cards)
        assert len(scores) == len(cards)
        for name, score in scores.items():
            assert 0 <= score <= 100

    def test_mock_explain_deck(self, mock_llm, lathiel_analysis, lathiel, db):
        forest = db.get_by_name("Forest")
        deck = Deck(commander=lathiel, cards=[forest] * 99)
        explanation = mock_llm.explain_deck(deck, lathiel_analysis)
        assert "MOCK" in explanation or len(explanation) > 10

    def test_mock_review_deck(self, mock_llm, lathiel_analysis, lathiel, db):
        forest = db.get_by_name("Forest")
        deck = Deck(commander=lathiel, cards=[forest] * 99)
        review = mock_llm.review_deck(deck, lathiel_analysis)
        assert len(review) > 10


class TestQuickSynergyCheck:
    def test_quick_check_returns_0_100(self, mock_llm, db):
        commander = db.get_by_name("Lathiel, the Bounteous Dawn")
        for card in db.all_cards[:20]:
            score = mock_llm.quick_synergy_check(commander, card)
            assert 0 <= score <= 100

    def test_quick_check_with_analysis(self, mock_llm, lathiel_analysis, db):
        """Cards with synergy keywords should score higher."""
        soul_warden = db.get_by_name("Soul Warden")  # has "gain 1 life"
        grizzly = db.get_by_name("Grizzly Bears")  # no synergy text

        ss = mock_llm.quick_synergy_check_with_analysis(lathiel_analysis, soul_warden)
        gs = mock_llm.quick_synergy_check_with_analysis(lathiel_analysis, grizzly)
        assert ss > gs


class TestCacheSummary:
    """v0.9.16c: per-build cache-efficiency accounting."""

    def test_none_when_no_calls(self):
        eng = LLMEngine(LLMConfig(mock_mode=True))
        assert eng.cache_summary() is None

    def test_hit_rate_math_and_per_model_split(self):
        eng = LLMEngine(LLMConfig(mock_mode=True))
        # Simulate accounting the way _call_api records it.
        eng._cache_stats = {
            "claude-sonnet-4-6": {"calls": 4, "fresh_in": 3000,
                                  "cache_read": 9000, "cache_create": 3000},
            "claude-haiku-4-5": {"calls": 10, "fresh_in": 10000,
                                 "cache_read": 0, "cache_create": 0},
        }
        s = eng.cache_summary()
        # Sonnet: 9000/(9000+3000) = 75%
        assert "claude-sonnet-4-6" in s and "75%" in s
        # Haiku: 0% (can't cache our prefix)
        assert "claude-haiku-4-5" in s and "0%" in s
        # Overall: 9000/(9000+13000) = 41%
        assert "OVERALL hit rate: 41%" in s


class TestCachePadding:
    """v0.9.19: prefix padding past the model minimum + extended TTL."""

    def _capture_call(self, config: LLMConfig, model: str,
                      system: str, context):
        """Run _call_api against a fake client; return captured kwargs."""
        eng = LLMEngine(config)
        eng.config.mock_mode = False
        captured = {}

        class _Usage:
            input_tokens = 10
            output_tokens = 5
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        class _Block:
            type = "text"
            text = "{}"

        class _Resp:
            usage = _Usage()
            content = [_Block()]
            stop_reason = "end_turn"

        class _Messages:
            @staticmethod
            def create(**kw):
                captured.update(kw)
                return _Resp()

        class _Client:
            messages = _Messages()

        eng.client = _Client()
        eng._call_api(system, "user prompt", model=model,
                      commander_context=context)
        return captured

    def test_haiku_short_prefix_gets_pad_and_clears_minimum(self):
        cfg = LLMConfig(mock_mode=True)
        # ~2.9K-token prefix (the real tournament shape) — below Haiku's 4096.
        kw = self._capture_call(cfg, "claude-haiku-4-5",
                                "R" * 6000, "C" * 5600)
        blocks = kw["system"]
        texts = [b["text"] for b in blocks]
        assert any("PADDING" in t for t in texts)
        # Padded prefix comfortably clears 4096 estimated tokens.
        assert sum(len(t) for t in texts) // 4 > 4096
        # Pad sits between rubric and commander context; final block keeps
        # the breakpoint.
        assert "PADDING" in texts[1]
        assert "cache_control" in blocks[-1]

    def test_sonnet_long_prefix_not_padded(self):
        cfg = LLMConfig(mock_mode=True)
        kw = self._capture_call(cfg, "claude-sonnet-4-6",
                                "R" * 8000, "C" * 8000)  # ~4K tokens > 2048
        assert not any("PADDING" in b["text"] for b in kw["system"])

    def test_pad_deterministic_across_calls(self):
        cfg = LLMConfig(mock_mode=True)
        kw1 = self._capture_call(cfg, "claude-haiku-4-5", "R" * 6000, "C" * 5600)
        kw2 = self._capture_call(cfg, "claude-haiku-4-5", "R" * 6000, "C" * 5600)
        assert kw1["system"] == kw2["system"]  # byte-identical -> cache hit

    def test_ttl_1h_sets_header_and_ttl(self):
        cfg = LLMConfig(mock_mode=True)  # cache_ttl defaults to "1h"
        kw = self._capture_call(cfg, "claude-sonnet-4-6",
                                "R" * 8000, "C" * 8000)
        assert kw["extra_headers"]["anthropic-beta"].startswith(
            "extended-cache-ttl")
        for b in kw["system"]:
            if "cache_control" in b:
                assert b["cache_control"]["ttl"] == "1h"

    def test_ttl_5m_no_header_no_ttl_field(self):
        cfg = LLMConfig(mock_mode=True, cache_ttl="5m")
        kw = self._capture_call(cfg, "claude-sonnet-4-6",
                                "R" * 8000, "C" * 8000)
        assert "extra_headers" not in kw
        for b in kw["system"]:
            if "cache_control" in b:
                assert b["cache_control"] == {"type": "ephemeral"}

    def test_padding_disabled_by_config(self):
        cfg = LLMConfig(mock_mode=True, cache_pad_to_minimum=False)
        kw = self._capture_call(cfg, "claude-haiku-4-5", "R" * 6000, "C" * 5600)
        assert not any("PADDING" in b["text"] for b in kw["system"])


class TestJSONParser:
    def test_parse_plain_json(self):
        assert LLMEngine._parse_json_defensively('{"a": 1}') == {"a": 1}

    def test_parse_from_prose(self):
        """LLM sometimes includes preamble."""
        text = 'Here is your result:\n{"a": 1}\nDone.'
        assert LLMEngine._parse_json_defensively(text) == {"a": 1}

    def test_parse_from_code_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert LLMEngine._parse_json_defensively(text) == {"a": 1}

    def test_parse_from_bare_fence(self):
        text = "```\n{\"a\": 1}\n```"
        assert LLMEngine._parse_json_defensively(text) == {"a": 1}

    def test_parse_nested_braces(self):
        """The parser should extract the correct outer object."""
        text = '{"outer": {"inner": "value"}}'
        result = LLMEngine._parse_json_defensively(text)
        assert result == {"outer": {"inner": "value"}}

    def test_parse_prose_with_stray_braces(self):
        """Regression: a reasoning preamble containing mana symbols like
        {G}{W} must not make the parser latch onto the first stray brace and
        give up before the real JSON object."""
        text = (
            "I need to evaluate these cards for Jasmine Boreal.\n"
            "1. Produces {G}{W} mana for casting vanilla creatures.\n"
            "**Key question**: do these help? Let me score them:\n"
            '{"scores": [{"name": "Grizzly Bears", "score": 85}]}'
        )
        result = LLMEngine._parse_json_defensively(text)
        assert result == {"scores": [{"name": "Grizzly Bears", "score": 85}]}

    def test_parse_invalid_returns_none(self):
        assert LLMEngine._parse_json_defensively("not json at all") is None

    def test_parse_empty_returns_none(self):
        assert LLMEngine._parse_json_defensively("") is None


class TestNameCanonicalization:
    """v0.9.13: LLM-returned names must map back to exact candidate names.

    The model drifts on case and sometimes echoes the [SYN+++] hint-tag
    prefix inside the name field; every downstream consumer does exact-case
    dict lookups, so an un-canonicalized name silently drops the pick (card
    selection) or the score (synergy scoring)."""

    def test_strips_hint_tag_and_fixes_case(self):
        m = {"soul warden": "Soul Warden"}
        assert LLMEngine._canonicalize_card_name(
            "[SYN+++] Soul Warden", m) == "Soul Warden"
        assert LLMEngine._canonicalize_card_name(
            "[SYN+] soul warden", m) == "Soul Warden"
        assert LLMEngine._canonicalize_card_name("SOUL WARDEN", m) == "Soul Warden"
        assert LLMEngine._canonicalize_card_name("Soul Warden", m) == "Soul Warden"

    def test_unknown_or_nonstring_returns_none(self):
        m = {"soul warden": "Soul Warden"}
        assert LLMEngine._canonicalize_card_name("Grizzly Bears", m) is None
        assert LLMEngine._canonicalize_card_name(None, m) is None
        assert LLMEngine._canonicalize_card_name(42, m) is None

    def _patched_engine(self, response: str) -> LLMEngine:
        eng = LLMEngine(LLMConfig(mock_mode=True))
        eng.config.mock_mode = False  # take the real parse path
        eng._call_api = lambda *a, **k: response
        return eng

    def test_score_synergy_keeps_tagged_names(self, lathiel_analysis, db):
        # The scoring prompt explicitly allows the hint-tag prefix in the
        # name field — the parsed score must land on the real card name
        # instead of being silently replaced by the heuristic.
        eng = self._patched_engine(
            '{"scores": [{"name": "[SYN+++] soul warden", "score": 88}]}'
        )
        card = db.get_by_name("Soul Warden")
        out = eng._score_synergy_single(lathiel_analysis, [card])
        assert out["Soul Warden"] == 88.0

    def test_select_cards_chunk_canonicalizes_and_dedupes(self, lathiel_analysis, db):
        eng = self._patched_engine(
            '{"names": ["[SYN++] soul warden", "soul warden", "Not A Card"]}'
        )
        cards = [db.get_by_name("Soul Warden"),
                 db.get_by_name("Heliod, Sun-Crowned"),
                 db.get_by_name("Sun Titan")]
        out = eng._select_cards_chunk(lathiel_analysis, cards, "threats", 2)
        assert out == ["Soul Warden"]  # canonical, deduped, hallucination dropped


class TestAnalysisCache:
    """v0.9.32: persistent commander-analysis cache — the keystone of
    run-to-run stability (analysis variance cascades into recall, hint
    tags, and effect classes)."""

    RESP = ('{"key_mechanics": ["lifegain"], "build_around_text": "Gain life",'
            ' "evaluation_notes": "n", "category_queries": {},'
            ' "synergy_keywords": ["life"]}')

    def _engine(self, tmp_path):
        eng = LLMEngine(LLMConfig(mock_mode=True))
        eng.config.mock_mode = False
        eng.config.analysis_cache_dir = str(tmp_path)
        calls = []
        eng._call_api = (lambda system_prompt, user_prompt, **kw:
                         calls.append(1) or self.RESP)
        return eng, calls

    def test_disk_roundtrip_across_engine_instances(self, tmp_path, db):
        lathiel = db.get_by_name("Lathiel, the Bounteous Dawn")
        e1, calls1 = self._engine(tmp_path)
        a1 = e1.analyze_commander(lathiel)
        assert calls1 == [1]  # one real call
        # "Next run": fresh engine, same cache dir → zero calls.
        e2, calls2 = self._engine(tmp_path)
        a2 = e2.analyze_commander(lathiel)
        assert calls2 == []
        assert a2.build_around_text == a1.build_around_text
        assert a2.synergy_keywords == a1.synergy_keywords

    def test_commander_text_change_invalidates(self, tmp_path, db):
        from dataclasses import replace
        lathiel = db.get_by_name("Lathiel, the Bounteous Dawn")
        e1, _ = self._engine(tmp_path)
        e1.analyze_commander(lathiel)
        # Same name, different card text (DB refresh) → prompt hash differs.
        changed = replace(lathiel, text=(lathiel.text or "") + " Flying.")
        e2, calls2 = self._engine(tmp_path)
        e2.analyze_commander(changed)
        assert calls2 == [1]  # re-analyzed, not reused

    def test_mock_mode_never_touches_disk(self, tmp_path, db):
        import os
        lathiel = db.get_by_name("Lathiel, the Bounteous Dawn")
        eng = LLMEngine(LLMConfig(mock_mode=True))
        eng.config.analysis_cache_dir = str(tmp_path / "sub")
        eng.analyze_commander(lathiel)
        assert not os.path.exists(str(tmp_path / "sub"))

    def test_unknown_fields_dropped_on_load(self, tmp_path, db):
        import json as _json, glob
        lathiel = db.get_by_name("Lathiel, the Bounteous Dawn")
        e1, _ = self._engine(tmp_path)
        e1.analyze_commander(lathiel)
        # Simulate a file written by a NEWER version with an extra field.
        path = glob.glob(str(tmp_path / "*.json"))[0]
        data = _json.loads(open(path, encoding="utf-8").read())
        data["analysis"]["field_from_the_future"] = 42
        open(path, "w", encoding="utf-8").write(_json.dumps(data))
        e2, calls2 = self._engine(tmp_path)
        a = e2.analyze_commander(lathiel)
        assert calls2 == []  # still a hit; unknown field ignored
        assert a.name == lathiel.name


class TestRefinementRoleTags:
    """v0.9.28: refine_deck_swaps annotates cards with their tracked role
    memberships so the AT FLOOR rule is actionable."""

    def test_role_tags_rendered_in_prompt(self, lathiel_analysis, db):
        eng = LLMEngine(LLMConfig(mock_mode=True))
        eng.config.mock_mode = False
        captured = {}

        def fake_call(system_prompt, user_prompt, **kw):
            captured["prompt"] = user_prompt
            return '{"swaps": []}'

        eng._call_api = fake_call
        sol = db.get_by_name("Sol Ring")
        warden = db.get_by_name("Soul Warden")
        deck = Deck(commander=db.get_by_name("Lathiel, the Bounteous Dawn"),
                    cards=[sol])
        eng.refine_deck_swaps(
            lathiel_analysis, deck, alternatives=[warden],
            synergy={"Sol Ring": 50.0, "Soul Warden": 80.0},
            power={"Sol Ring": 98.0, "Soul Warden": 60.0},
            card_roles={"Sol Ring": ["ramp"], "Soul Warden": ["threat"]},
        )
        assert "[syn=50 pow=98 role:ramp] **Sol Ring**" in captured["prompt"]
        assert "role:threat] **Soul Warden**" in captured["prompt"]

    def test_untagged_cards_unchanged(self, lathiel_analysis, db):
        eng = LLMEngine(LLMConfig(mock_mode=True))
        eng.config.mock_mode = False
        captured = {}
        eng._call_api = (lambda system_prompt, user_prompt, **kw:
                         captured.update(prompt=user_prompt) or '{"swaps": []}')
        sol = db.get_by_name("Sol Ring")
        deck = Deck(commander=db.get_by_name("Lathiel, the Bounteous Dawn"),
                    cards=[sol])
        eng.refine_deck_swaps(
            lathiel_analysis, deck, alternatives=[],
            synergy={"Sol Ring": 50.0}, power={"Sol Ring": 98.0},
        )
        assert "[syn=50 pow=98] **Sol Ring**" in captured["prompt"]


class TestTemperatureHandling:
    def test_temperature_rejection_detection(self):
        """Opus 4.7 is flagged as rejecting temperature."""
        from mtg_deck_builder.llm_engine import _model_rejects_temperature
        assert _model_rejects_temperature("claude-opus-4-7")
        assert _model_rejects_temperature("claude-opus-4-7-20260416")
        assert not _model_rejects_temperature("claude-sonnet-4-6")
        assert not _model_rejects_temperature("claude-haiku-4-5")


class TestNoAPIKeyFallback:
    def test_no_key_falls_back_to_mock(self, monkeypatch):
        """Without an API key, engine should fall back to mock mode silently."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        engine = LLMEngine(LLMConfig())  # No api_key, no env var
        assert engine.config.mock_mode
        assert engine.client is None
