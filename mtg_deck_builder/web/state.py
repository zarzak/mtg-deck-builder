"""
In-memory state for running / completed builds.

Each build gets a UUID. State progresses:
    pending -> running -> complete | failed

Progress events from the DeckBuilder's progress_callback are appended
to a per-build list AND pushed into an asyncio.Queue that the SSE
endpoint drains. The list persists for page loads after completion;
the queue only has events until consumers pop them.

Thread safety: DeckBuilder runs in a thread; FastAPI handlers run in
the asyncio event loop. We use threading.Lock around the state dict
and asyncio.run_coroutine_threadsafe to push into the queue from the
worker thread.

No database. Restart loses everything. Fine for a local single-user
tool. Memory grows with every build; a future cleanup task could
expire old builds, but Session 7 scope says single-session is enough.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..models import OptimizationResult


@dataclass
class ProgressEvent:
    """One entry from the DeckBuilder progress_callback."""
    phase: str
    status: str
    fraction: float  # 0.0-1.0
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "status": self.status,
            "fraction": self.fraction,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class BuildState:
    """Full state for one build job."""
    build_id: str
    status: str = "pending"  # pending | running | complete | failed
    commander_name: str = ""
    events: list[ProgressEvent] = field(default_factory=list)
    result: Optional["OptimizationResult"] = None
    error: Optional[str] = None
    report_path: Optional[str] = None  # absolute path to generated HTML
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # Per-build asyncio.Queue for live SSE streaming. Created by the
    # handler that kicks off the build; the worker thread pushes into
    # it via run_coroutine_threadsafe.
    queue: Optional[asyncio.Queue] = None
    loop: Optional[asyncio.AbstractEventLoop] = None

    def post_event(self, event: ProgressEvent) -> None:
        """Append to history and push to SSE queue (if a consumer exists)."""
        self.events.append(event)
        if self.queue is not None and self.loop is not None:
            # Schedule put_nowait on the event loop. put_nowait is
            # synchronous so no coroutine is left dangling if the loop
            # is gone — we just swallow the RuntimeError.
            try:
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait, event,
                )
            except RuntimeError:
                # Loop closed; skip — page already gone
                pass


class BuildRegistry:
    """Thread-safe store of all builds."""

    def __init__(self):
        self._builds: dict[str, BuildState] = {}
        self._lock = threading.Lock()

    def create(self, commander_name: str) -> BuildState:
        build_id = str(uuid.uuid4())
        state = BuildState(build_id=build_id, commander_name=commander_name)
        with self._lock:
            self._builds[build_id] = state
        return state

    def get(self, build_id: str) -> Optional[BuildState]:
        with self._lock:
            return self._builds.get(build_id)

    def all(self) -> list[BuildState]:
        with self._lock:
            return sorted(
                self._builds.values(),
                key=lambda s: s.started_at,
                reverse=True,
            )

    def delete(self, build_id: str) -> bool:
        with self._lock:
            return self._builds.pop(build_id, None) is not None


# Global registry. The web app attaches this to app.state at startup.
_registry: Optional[BuildRegistry] = None


def get_registry() -> BuildRegistry:
    global _registry
    if _registry is None:
        _registry = BuildRegistry()
    return _registry


def reset_registry_for_tests() -> None:
    """Test helper — clear all builds without affecting the global instance."""
    global _registry
    _registry = BuildRegistry()
