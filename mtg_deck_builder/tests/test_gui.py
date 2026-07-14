"""
Tests for the v0.9.20 local web GUI.

The GUI shells out to the CLI, so the critical unit is the payload -> argv
translation (build_argv / power_scan_argv / refresh_argv). A live-server
smoke test covers routing, state JSON, and the one-job-at-a-time guard —
without ever spawning a real subprocess.
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import mtg_deck_builder.gui as gui_mod
from mtg_deck_builder.gui import (
    BUILD_DEFAULTS, JobManager, api_key_status, build_argv, build_prefix,
    get_api_key, make_handler, parse_decklist, power_scan_argv,
    read_deck_meta, refresh_argv, safe_relpath, save_deck, search_cards,
    set_api_key, slugify,
)


class TestSlugify:
    def test_commander_names(self):
        assert slugify("Jodah, the Unifier") == "jodah_the_unifier"
        assert slugify("Selenia, Dark Angel") == "selenia_dark_angel"
        assert slugify("") == "deck"


class TestBuildArgv:
    def test_usual_defaults_match_manual_command(self):
        argv = build_argv({"commander": "Jodah, the Unifier"}, "cards.csv")
        joined = " ".join(argv)
        # The GUI's simple mode reproduces the user's standard invocation.
        assert "--csv cards.csv" in joined
        assert "build Jodah, the Unifier" in joined
        assert "--bracket 4" in joined
        assert "--seed 42" in joined
        assert "--generations 300" in joined and "--population 100" in joined
        assert "--recall-edhrec " in joined and "--recall-embeddings" in joined
        assert "--recall-patterns" in joined
        assert "--synergy-scoring-mode llm" in joined
        assert "--tournament-model claude-haiku-4-5" in joined
        assert "--card-power-mode llm" in joined
        assert "--combos llm" in joined
        assert "--report jodah_the_unifier_deck_report.html" in joined
        assert "--output jodah_the_unifier_deck.txt" in joined
        assert "--log-file jodah_the_unifier_build.log" in joined

    def test_commander_required(self):
        with pytest.raises(ValueError):
            build_argv({}, "cards.csv")

    def test_toggles_and_options(self):
        argv = build_argv({
            "commander": "X",
            "recall_edhrec": False,
            "signature_pass": False,
            "quality_roles": False,
            "seed": "",
            "budget": 5,
            "budget_exclude_unknown": True,
            "locks": "Sol Ring, Command Tower",
            "bans": "Armageddon",
            "extra_flags": "--islands 4",
        }, "cards.csv")
        joined = " ".join(argv)
        assert "--recall-edhrec " not in joined + " "
        assert "--no-signature-pass" in joined
        assert "--no-quality-roles" in joined
        assert "--seed" not in joined  # blank = random
        assert "--budget 5.0" in joined
        assert "--budget-exclude-unknown" in joined
        assert argv[argv.index("--lock") + 1] == "Sol Ring"
        assert argv.count("--lock") == 2
        assert argv[argv.index("--ban") + 1] == "Armageddon"
        assert "--islands 4" in joined

    def test_defaults_dict_is_not_mutated(self):
        before = dict(BUILD_DEFAULTS)
        build_argv({"commander": "X", "generations": 50}, "cards.csv")
        assert BUILD_DEFAULTS == before


class TestUtilityArgv:
    def test_refresh(self):
        argv = refresh_argv("cards.csv")
        assert argv[-2:] == ["refresh-cards", "--force"]
        assert "--csv" in argv

    def test_power_scan_colors_and_dry_run(self):
        argv = power_scan_argv({"colors": "WU", "dry_run": True,
                                "batch_size": 100}, "cards.csv")
        joined = " ".join(argv)
        assert "power-scan" in joined
        assert "--colors WU" in joined
        assert "--dry-run" in joined
        assert "--batch-size 100" in joined

    def test_power_scan_whole_db_real_run(self):
        argv = power_scan_argv({"colors": "", "dry_run": False}, "cards.csv")
        joined = " ".join(argv)
        assert "--colors" not in joined  # empty = whole DB
        assert "--dry-run" not in joined


class _FakeJobs(JobManager):
    """JobManager that never spawns a subprocess."""

    def __init__(self):
        super().__init__()
        self.started = []
        self._busy = False

    @property
    def running(self):
        return self._busy

    def start(self, argv, kind, label, extra_env=None):
        if self._busy:
            return False
        self.started.append((kind, argv, extra_env or {}))
        self._busy = True
        return True


@pytest.fixture()
def gui_server():
    jobs = _FakeJobs()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                make_handler("cards.csv", jobs))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, jobs
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, obj):
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


SAMPLE_DECK = """Commander: Jodah, the Unifier

// Creatures (2)
1 Kinnan, Bonder Prodigy
1 Sisay, Weatherlight Captain

// Lands (3)
2 Island
1 Command Tower
"""


class TestDeckParser:
    def test_parse_sections_counts_total(self, tmp_path):
        f = tmp_path / "jodah_deck.txt"
        f.write_text(SAMPLE_DECK, encoding="utf-8")
        deck = parse_decklist(f)
        assert deck["commander"] == "Jodah, the Unifier"
        assert [s["name"] for s in deck["sections"]] == ["Creatures", "Lands"]
        assert deck["sections"][1]["cards"][0] == {"count": 2, "name": "Island"}
        assert deck["total"] == 5

    def test_junk_lines_skipped(self, tmp_path):
        f = tmp_path / "x_deck.txt"
        f.write_text("Commander: X\n\ngarbage line\n// S (1)\n1 Card\n",
                     encoding="utf-8")
        deck = parse_decklist(f)
        assert deck["total"] == 1


class TestApiKeyEncryption:
    """v0.9.22: the key is DPAPI-wrapped at rest (Windows) and never stored
    as recognizable plaintext by set_api_key."""

    def test_roundtrip(self):
        settings = {}
        set_api_key(settings, "sk-ant-roundtrip-9876")
        assert get_api_key(settings) == "sk-ant-roundtrip-9876"
        # No plaintext field, and the blob doesn't contain the raw key.
        assert "api_key" not in settings
        assert "sk-ant-roundtrip" not in json.dumps(settings)

    def test_dpapi_used_on_windows(self):
        import sys
        settings = {}
        set_api_key(settings, "sk-x-1")
        expected = "dpapi" if sys.platform == "win32" else "plain"
        assert settings["api_key_scheme"] == expected

    def test_clear(self):
        settings = {}
        set_api_key(settings, "sk-x-2")
        set_api_key(settings, "")
        assert get_api_key(settings) == ""
        assert "api_key_enc" not in settings

    def test_legacy_plaintext_still_readable(self):
        assert get_api_key({"api_key": "sk-legacy-77"}) == "sk-legacy-77"


class TestApiKeyStatus:
    def test_saved_key_wins_over_environment(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key-0000")
        st = api_key_status({"api_key": "sk-saved-key-9999"})
        assert st["source"] == "saved" and st["masked"] == "…9999"

    def test_environment_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key-4321")
        st = api_key_status({})
        assert st["source"] == "environment" and st["masked"] == "…4321"

    def test_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert api_key_status({})["source"] == "missing"


class TestStorageLayout:
    """v0.9.23: decks/<slug>/<timestamp>_* layout + sandboxed subpaths."""

    def test_build_prefix_shape(self):
        import time
        ts = time.localtime()
        p = build_prefix("Jodah, the Unifier", ts)
        assert p.startswith("decks/jodah_the_unifier/")
        assert p.endswith(time.strftime("%Y-%m-%d_%H%M%S", ts))

    def test_build_argv_honors_out_prefix(self):
        argv = build_argv({"commander": "X"}, "cards.csv",
                          out_prefix="decks/x/2026-07-10_120000")
        joined = " ".join(argv)
        assert "--output decks/x/2026-07-10_120000_deck.txt" in joined
        assert "--report decks/x/2026-07-10_120000_deck_report.html" in joined
        assert "--log-file decks/x/2026-07-10_120000_build.log" in joined

    def test_safe_relpath_sandbox(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "decks" / "s").mkdir(parents=True)
        (tmp_path / "decks" / "s" / "a_deck.txt").write_text("x")
        assert safe_relpath("decks/s/a_deck.txt") is not None
        assert safe_relpath("../outside.txt") is None
        assert safe_relpath("..%2F..%2Fetc") is None
        assert safe_relpath("C:/Windows/system.ini") is None
        assert safe_relpath("") is None

    def test_list_outputs_both_locations(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "legacy_deck.txt").write_text("Commander: X\n")
        sub = tmp_path / "decks" / "jodah_the_unifier"
        sub.mkdir(parents=True)
        (sub / "2026-07-10_120000_deck.txt").write_text("Commander: J\n")
        (sub / "2026-07-10_120000_build.log").write_text("log\n")
        entries = gui_mod._list_outputs()
        by_path = {e["path"]: e for e in entries}
        assert by_path["legacy_deck.txt"]["slug"] == "legacy"
        deck = by_path["decks/jodah_the_unifier/2026-07-10_120000_deck.txt"]
        assert deck["slug"] == "jodah_the_unifier" and deck["kind"] == "deck"
        log = by_path["decks/jodah_the_unifier/2026-07-10_120000_build.log"]
        assert log["kind"] == "log"

    def test_read_deck_meta_sidecar_preferred(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a_deck.txt").write_text("Commander: X\n")
        (tmp_path / "a_meta.json").write_text(
            json.dumps({"bracket": 5, "commander": "X"}))
        meta = read_deck_meta(gui_mod.Path("a_deck.txt"))
        assert meta["bracket"] == 5

    def test_read_deck_meta_log_tail_fallback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "b_deck.txt").write_text("Commander: X\n")
        (tmp_path / "b_build.log").write_text(
            "noise\n" * 200 + "ROLE STATUS [Bracket 3 — Upgraded] ramp: 11\n")
        meta = read_deck_meta(gui_mod.Path("b_deck.txt"))
        assert meta == {"bracket": 3, "source": "log"}

    def test_read_deck_meta_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "c_deck.txt").write_text("Commander: X\n")
        assert read_deck_meta(gui_mod.Path("c_deck.txt")) is None

    def test_save_deck_into_subfolder(self, db, monkeypatch, tmp_path):
        monkeypatch.setattr(gui_mod, "_db", db)
        monkeypatch.chdir(tmp_path)
        save_deck("unused.csv", "decks/x/edited_deck.txt", "X",
                  [{"count": 1, "name": "Sol Ring"}])
        assert (tmp_path / "decks" / "x" / "edited_deck.txt").exists()


class TestCardSearch:
    def test_prefix_beats_substring_and_limit(self, db, monkeypatch):
        monkeypatch.setattr(gui_mod, "_db", db)
        results = search_cards("unused.csv", "sol", limit=10)
        names = [r["name"] for r in results]
        assert "Sol Ring" in names
        assert names.index("Sol Ring") == 0  # prefix match ranks first

    def test_commanders_only_filter(self, db, monkeypatch):
        monkeypatch.setattr(gui_mod, "_db", db)
        results = search_cards("unused.csv", "lathiel", commanders_only=True)
        assert any("Lathiel" in r["name"] for r in results)
        # A non-legendary artifact never shows up in commander search.
        results = search_cards("unused.csv", "sol ring", commanders_only=True)
        assert results == []

    def test_short_query_returns_nothing(self, db, monkeypatch):
        monkeypatch.setattr(gui_mod, "_db", db)
        assert search_cards("unused.csv", "s") == []


class TestSaveDeck:
    def _cards(self):
        return [{"count": 1, "name": "Sol Ring"},
                {"count": 1, "name": "Soul Warden"},
                {"count": 2, "name": "Forest"}]

    def test_save_derives_sections_and_writes(self, db, monkeypatch, tmp_path):
        monkeypatch.setattr(gui_mod, "_db", db)
        monkeypatch.chdir(tmp_path)
        result = save_deck("unused.csv", "test_deck.txt",
                           "Lathiel, the Bounteous Dawn", self._cards())
        text = (tmp_path / "test_deck.txt").read_text(encoding="utf-8")
        assert "Commander: Lathiel, the Bounteous Dawn" in text
        # Sections re-derived from card types, not caller input.
        assert "// Artifacts (1)" in text and "1 Sol Ring" in text
        assert "// Creatures (1)" in text and "1 Soul Warden" in text
        assert "// Lands (2)" in text and "2 Forest" in text
        # 4 cards != 99 -> warned, not blocked.
        assert any("4 cards" in w for w in result["warnings"])
        # Round-trips through the parser.
        deck = parse_decklist(tmp_path / "test_deck.txt")
        assert deck["total"] == 4

    def test_backup_created_on_overwrite(self, db, monkeypatch, tmp_path):
        monkeypatch.setattr(gui_mod, "_db", db)
        monkeypatch.chdir(tmp_path)
        save_deck("unused.csv", "test_deck.txt", "X", self._cards())
        save_deck("unused.csv", "test_deck.txt", "X",
                  [{"count": 1, "name": "Sol Ring"}])
        assert (tmp_path / "test_deck.bak").exists()
        assert "Soul Warden" in (tmp_path / "test_deck.bak").read_text(
            encoding="utf-8")

    def test_unknown_card_rejected_nothing_written(self, db, monkeypatch,
                                                   tmp_path):
        monkeypatch.setattr(gui_mod, "_db", db)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="Not A Real Card"):
            save_deck("unused.csv", "test_deck.txt", "X",
                      [{"count": 1, "name": "Not A Real Card"}])
        assert not (tmp_path / "test_deck.txt").exists()

    def test_duplicate_nonbasic_warned_basics_allowed(self, db, monkeypatch,
                                                      tmp_path):
        monkeypatch.setattr(gui_mod, "_db", db)
        monkeypatch.chdir(tmp_path)
        result = save_deck("unused.csv", "test_deck.txt", "X",
                           [{"count": 2, "name": "Sol Ring"},
                            {"count": 5, "name": "Forest"}])
        warned = " ".join(result["warnings"])
        assert "Sol Ring" in warned
        assert "Forest" not in warned

    def test_bad_filename_rejected(self, db, monkeypatch, tmp_path):
        monkeypatch.setattr(gui_mod, "_db", db)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="_deck.txt"):
            save_deck("unused.csv", "evil.py", "X", self._cards())


class TestServer:
    def test_page_and_state(self, gui_server):
        base, _ = gui_server
        status, body = _get(base + "/")
        assert status == 200 and b"MTG Deck Builder" in body
        status, body = _get(base + "/api/state")
        state = json.loads(body)
        assert state["csv"] == "cards.csv"
        assert state["defaults"]["bracket"] == 4
        assert state["job"]["running"] is False

    def test_build_starts_job_and_second_is_rejected(self, gui_server,
                                                     tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # meta sidecar lands here, not the repo
        base, jobs = gui_server
        status, resp = _post(base + "/api/build", {"commander": "Test Cmd"})
        assert status == 200 and resp["started"]
        assert jobs.started[0][0] == "build"
        # v0.9.23: outputs are anchored under decks/<slug>/<timestamp>_*
        # and a meta sidecar records the build parameters.
        argv = jobs.started[0][1]
        out = argv[argv.index("--output") + 1]
        assert out.startswith("decks/test_cmd/") and out.endswith("_deck.txt")
        metas = list((tmp_path / "decks" / "test_cmd").glob("*_meta.json"))
        assert len(metas) == 1
        meta = json.loads(metas[0].read_text(encoding="utf-8"))
        assert meta["commander"] == "Test Cmd" and meta["bracket"] == 4
        # One job at a time — builds cost credits.
        status, resp = _post(base + "/api/build", {"commander": "Another"})
        assert status == 409

    def test_build_without_commander_400(self, gui_server):
        base, _ = gui_server
        status, resp = _post(base + "/api/build", {})
        assert status == 400 and "commander" in resp["error"]

    def test_file_serving_is_sandboxed(self, gui_server):
        base, _ = gui_server
        # Traversal collapses to basename; disallowed extension -> 404.
        status, _ = _get(base + "/files/..%5C..%5Csecrets.py")
        assert status == 404

    def test_settings_roundtrip_and_masking(self, gui_server, tmp_path,
                                            monkeypatch):
        import mtg_deck_builder.gui as gui_mod
        monkeypatch.setattr(gui_mod, "SETTINGS_FILE",
                            tmp_path / "gui_settings.json")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        base, _ = gui_server
        status, resp = _post(base + "/api/settings",
                             {"api_key": "sk-test-secret-1234"})
        assert status == 200
        assert resp["api_key"]["source"] == "saved"
        assert resp["api_key"]["masked"] == "…1234"
        # The full key never travels back to the page.
        assert "sk-test-secret" not in json.dumps(resp)
        # Empty clears it.
        status, resp = _post(base + "/api/settings", {"api_key": ""})
        assert resp["api_key"]["source"] == "missing"

    def test_saved_key_injected_into_job_env(self, gui_server, tmp_path,
                                             monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_mod, "SETTINGS_FILE", tmp_path / "s.json")
        gui_mod.save_settings({"api_key": "sk-inject-42"})
        base, jobs = gui_server
        _post(base + "/api/build", {"commander": "X"})
        assert jobs.started[0][2].get("ANTHROPIC_API_KEY") == "sk-inject-42"

    def test_deck_endpoint(self, gui_server, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "jodah_deck.txt").write_text(SAMPLE_DECK,
                                                 encoding="utf-8")
        base, _ = gui_server
        status, body = _get(base + "/api/deck?file=jodah_deck.txt&images=0")
        deck = json.loads(body)
        assert status == 200
        assert deck["commander"] == "Jodah, the Unifier"
        assert deck["images"] == {}  # images=0 -> no Scryfall traffic
        # Only *_deck.txt basenames are served.
        status, _ = _get(base + "/api/deck?file=secrets.py")
        assert status == 404
