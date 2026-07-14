"""Route-level tests for the FastAPI app via TestClient."""

import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(test_csv_path):
    from mtg_deck_builder.web.app import create_app
    from mtg_deck_builder.web.state import reset_registry_for_tests
    reset_registry_for_tests()
    return create_app(str(test_csv_path), mock_llm=True)


@pytest.fixture
def client(app):
    return TestClient(app)


# ---- Static GET routes ----

class TestPages:
    def test_home(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "MTG Deck Builder" in r.text
        # All four feature panels visible
        for label in ("Build", "Diff", "Analyze", "Tags"):
            assert label in r.text

    def test_build_form(self, client):
        r = client.get("/build")
        assert r.status_code == 200
        assert "Commander" in r.text

    def test_diff_form(self, client):
        r = client.get("/diff")
        assert r.status_code == 200
        assert "Diff" in r.text

    def test_analyze_form(self, client):
        r = client.get("/analyze")
        assert r.status_code == 200

    def test_tags_form(self, client):
        r = client.get("/tags")
        assert r.status_code == 200

    def test_404_for_unknown_build(self, client):
        r = client.get("/build/nonexistent-id")
        assert r.status_code == 404


# ---- Build POST + status flow ----

class TestBuildFlow:
    def test_post_build_redirects_to_status(self, client):
        r = client.post("/build", data={
            "commander_name": "Lathiel, the Bounteous Dawn",
            "population_size": "4",
            "generations": "2",
            "random_seed": "42",
        }, follow_redirects=False)
        assert r.status_code == 303
        location = r.headers["location"]
        assert location.startswith("/build/")

    def test_post_build_missing_commander_400(self, client):
        r = client.post("/build", data={"population_size": "4"})
        assert r.status_code == 400

    def test_full_build_lifecycle(self, client):
        """End-to-end: POST build, poll status, fetch report."""
        r = client.post("/build", data={
            "commander_name": "Lathiel, the Bounteous Dawn",
            "population_size": "4",
            "generations": "2",
            "random_seed": "42",
        }, follow_redirects=False)
        assert r.status_code == 303
        location = r.headers["location"]
        build_id = location.split("/")[-1]

        # Poll until complete
        for _ in range(100):  # ~30s budget
            r2 = client.get(location)
            assert r2.status_code == 200
            if "complete" in r2.text.lower() or "failed" in r2.text.lower():
                break
            time.sleep(0.3)

        # Confirm we got to a terminal state
        body = r2.text.lower()
        assert "complete" in body or "failed" in body, "Build never finished"

        # Report should be available
        r3 = client.get(f"/build/{build_id}/report")
        assert r3.status_code == 200
        assert b"<html" in r3.content


# ---- SSE ----

class TestSSE:
    def test_sse_streams_events(self, client):
        # Kick off a build
        r = client.post("/build", data={
            "commander_name": "Lathiel, the Bounteous Dawn",
            "population_size": "4",
            "generations": "2",
            "random_seed": "42",
        }, follow_redirects=False)
        build_id = r.headers["location"].split("/")[-1]

        # Give the build a moment to produce events
        time.sleep(0.5)

        events_seen = 0
        done_seen = False
        with client.stream("GET", f"/build/{build_id}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    events_seen += 1
                    if "__done__" in line:
                        done_seen = True
                        break
                if events_seen > 200:  # safety cap
                    break

        assert events_seen > 0, "No SSE events received"
        assert done_seen, "Done sentinel not seen"

    def test_sse_unknown_build_404(self, client):
        r = client.get("/build/nonexistent/events")
        assert r.status_code == 404


class TestSSEGenerator:
    """Direct unit tests for _event_generator. Constructs a BuildState
    by hand so we can deterministically arrange the race that triggers
    the duplicate-events bug from v0.7.0."""

    def _collect(self, gen):
        """Run an async generator to completion in a fresh loop."""
        import asyncio
        out = []
        async def runner():
            async for item in gen:
                out.append(item)
        asyncio.run(runner())
        return out

    def test_drain_step_prevents_duplicate_events(self):
        """v0.7.1 regression: _event_generator must NOT yield events
        twice when they're present in both state.events and state.queue
        at SSE-connect time."""
        import asyncio
        from mtg_deck_builder.web.app import _event_generator
        from mtg_deck_builder.web.state import BuildState, ProgressEvent

        # Set up a state with events in BOTH the history list AND the queue
        # — exactly the race the worker thread creates when SSE connects
        # mid-build.
        async def setup_and_collect():
            loop = asyncio.get_event_loop()
            state = BuildState(build_id="x", commander_name="X")
            state.loop = loop
            state.queue = asyncio.Queue()

            # Simulate worker pushing 3 events (history + queue both filled)
            for i in range(3):
                evt = ProgressEvent(
                    phase=f"phase{i}", status="step", fraction=i / 3.0,
                    message=f"msg {i}",
                )
                state.events.append(evt)
                state.queue.put_nowait(evt)

            # Simulate the build finishing right after SSE connects
            done = ProgressEvent(
                phase="__done__", status="complete",
                fraction=1.0, message="complete",
            )
            state.events.append(done)
            state.queue.put_nowait(done)

            out = []
            async for chunk in _event_generator(state):
                out.append(chunk)
            return out

        chunks = asyncio.run(setup_and_collect())

        # Each event should appear exactly once
        data_chunks = [c for c in chunks if c.startswith("data: ")]
        # 3 phaseN events + 1 __done__ = 4 unique events
        assert len(data_chunks) == 4, (
            f"Expected 4 events, got {len(data_chunks)} "
            f"(would be 7-8 with the duplicate bug)"
        )
        # Each phase appears exactly once
        for i in range(3):
            matches = [c for c in data_chunks if f'"phase{i}"' in c]
            assert len(matches) == 1, f"phase{i} appeared {len(matches)} times"

    def test_generator_handles_no_queue(self):
        """If a state was created without queue/loop wiring, the generator
        must degrade gracefully to history-only replay rather than crashing."""
        import asyncio
        from mtg_deck_builder.web.app import _event_generator
        from mtg_deck_builder.web.state import BuildState, ProgressEvent

        state = BuildState(build_id="x", commander_name="X")
        state.events = [
            ProgressEvent(phase="p", status="s", fraction=0.5, message="hi"),
        ]
        # state.queue stays None (default)

        async def runner():
            out = []
            async for chunk in _event_generator(state):
                out.append(chunk)
            return out

        chunks = asyncio.run(runner())
        # Should yield the 1 history event + 1 synthetic __done__
        data_chunks = [c for c in chunks if c.startswith("data: ")]
        assert len(data_chunks) == 2
        assert "__done__" in data_chunks[-1]


# ---- Diff ----

class TestDiff:
    def test_diff_shows_differences(self, client):
        snap1 = {"commander": "X", "cards": ["Sol Ring", "Forest", "Plains"]}
        snap2 = {"commander": "X", "cards": ["Sol Ring", "Forest", "Lightning Greaves"]}
        r = client.post(
            "/diff",
            files={
                "before": ("old.json", json.dumps(snap1), "application/json"),
                "after": ("new.json", json.dumps(snap2), "application/json"),
            },
        )
        assert r.status_code == 200
        assert "Lightning Greaves" in r.text  # added
        assert "Plains" in r.text  # removed

    def test_diff_invalid_json(self, client):
        r = client.post(
            "/diff",
            files={
                "before": ("bad.json", "not json", "application/json"),
                "after": ("ok.json", "{}", "application/json"),
            },
        )
        assert r.status_code == 400

    def test_diff_canonical_warmstart_format(self, client):
        """The official commander_name + card_names format also works."""
        snap1 = {"commander_name": "X", "card_names": ["A", "B"]}
        snap2 = {"commander_name": "X", "card_names": ["B", "C"]}
        r = client.post(
            "/diff",
            files={
                "before": ("old.json", json.dumps(snap1), "application/json"),
                "after": ("new.json", json.dumps(snap2), "application/json"),
            },
        )
        assert r.status_code == 200
        assert "C" in r.text  # added
        assert "A" in r.text  # removed


# ---- Analyze ----

class TestAnalyze:
    def test_analyze_returns_analysis(self, client):
        r = client.post("/analyze", data={
            "commander_name": "Lathiel, the Bounteous Dawn",
        })
        assert r.status_code == 200
        # In mock mode the analysis still has a color identity
        assert "color identity" in r.text.lower()

    def test_analyze_missing_commander_400(self, client):
        r = client.post("/analyze", data={})
        assert r.status_code == 400


# ---- Tags ----

class TestTags:
    def test_tags_empty_400(self, client):
        r = client.post("/tags", data={"tags": "", "kind": "art"})
        assert r.status_code == 400

    def test_tags_offline_no_cache_returns_results(self, client, tmp_path):
        """With an empty cache and no network, each tag still produces a row."""
        r = client.post("/tags", data={
            "tags": "forest\nmammoth",
            "kind": "art",
            "cache_dir": str(tmp_path),
        })
        # Should NOT be 500 — offline absence is a valid result row
        assert r.status_code == 200
        assert "forest" in r.text
        assert "mammoth" in r.text
