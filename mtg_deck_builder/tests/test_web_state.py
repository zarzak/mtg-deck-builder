"""Tests for web.state — BuildState, BuildRegistry, ProgressEvent."""

import pytest

from mtg_deck_builder.web.state import (
    BuildRegistry, BuildState, ProgressEvent,
)


class TestProgressEvent:
    def test_to_dict(self):
        e = ProgressEvent(
            phase="analysis", status="done", fraction=0.5, message="hi"
        )
        d = e.to_dict()
        assert d["phase"] == "analysis"
        assert d["status"] == "done"
        assert d["fraction"] == 0.5
        assert d["message"] == "hi"
        assert "timestamp" in d


class TestBuildState:
    def test_post_event_appends_to_history(self):
        state = BuildState(build_id="abc", commander_name="X")
        assert state.events == []
        state.post_event(ProgressEvent(
            phase="p", status="s", fraction=0.1, message="m",
        ))
        assert len(state.events) == 1

    def test_post_event_without_queue_doesnt_crash(self):
        """queue=None is the common case before SSE consumer attaches."""
        state = BuildState(build_id="abc", commander_name="X")
        state.queue = None
        # Should not raise
        state.post_event(ProgressEvent(
            phase="p", status="s", fraction=0.1, message="m",
        ))


class TestBuildRegistry:
    def test_create_and_get(self):
        r = BuildRegistry()
        state = r.create(commander_name="Test Cmdr")
        assert state.build_id
        assert state.status == "pending"
        assert state.commander_name == "Test Cmdr"

        fetched = r.get(state.build_id)
        assert fetched is state

    def test_get_unknown_returns_none(self):
        r = BuildRegistry()
        assert r.get("nonexistent-id") is None

    def test_all_sorted_newest_first(self):
        import time
        r = BuildRegistry()
        s1 = r.create("A")
        time.sleep(0.01)
        s2 = r.create("B")
        all_ = r.all()
        assert all_[0] is s2  # newest
        assert all_[1] is s1

    def test_delete_returns_true_if_existed(self):
        r = BuildRegistry()
        s = r.create("X")
        assert r.delete(s.build_id) is True
        assert r.get(s.build_id) is None

    def test_delete_missing_returns_false(self):
        r = BuildRegistry()
        assert r.delete("nonexistent") is False

    def test_unique_ids(self):
        r = BuildRegistry()
        ids = {r.create(f"C{i}").build_id for i in range(10)}
        assert len(ids) == 10
