"""
FastAPI application for the MTG Deck Builder web interface.

Routes are grouped by concern. Each route handler is kept thin —
business logic (form parsing, running builds, rendering diffs) lives
in sibling modules so this file stays readable.

The event loop is asyncio, but DeckBuilder is synchronous. We run
builds in a ThreadPoolExecutor and bridge progress events from the
worker thread to the handler's asyncio.Queue via
run_coroutine_threadsafe. Single worker thread per build — multi-user
concurrency isn't a concern.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import (
    HTMLResponse, RedirectResponse,
    StreamingResponse, FileResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..models import BuildConfig
from ..llm_engine import LLMConfig
from ..card_database import CardDatabase
from ..deck_builder import DeckBuilder
from ..html_report import generate_html_report
from ..deck_diff import diff_decks
from ..scryfall_tags import ScryfallTagClient
from .forms import config_from_form
from .state import BuildRegistry, BuildState, ProgressEvent, get_registry
from .diff_html import render_diff_html

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------

def create_app(
    csv_path: str,
    mock_llm: bool = False,
    artifacts_dir: Optional[str] = None,
) -> FastAPI:
    """
    Create a fully configured FastAPI app.

    Args:
        csv_path: Path to the cards CSV. Required; the server loads
            the database once at startup and reuses it across requests.
        mock_llm: If True, all builds/analyses use mock LLM (no API key
            needed). Good for testing, and for users who want to poke
            around without spending API credits.
        artifacts_dir: Where to write HTML reports and other per-build
            artifacts. Defaults to ./web_artifacts.
    """
    from mtg_deck_builder import __version__ as _pkg_version
    app = FastAPI(title="MTG Deck Builder", version=_pkg_version)
    app.state.csv_path = csv_path
    app.state.mock_llm = mock_llm
    app.state.artifacts_dir = Path(artifacts_dir or "./web_artifacts")
    app.state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    app.state.registry = get_registry()
    app.state.executor = ThreadPoolExecutor(max_workers=2)

    # Preload the card database so we can populate the commander dropdown
    logger.info(f"Loading card database from {csv_path}")
    db = CardDatabase(csv_path)
    db.load()
    app.state.db = db
    app.state.commander_names = sorted(
        c.name for c in db.all_cards
        if "Legendary" in (c.supertypes or "")
        and "Creature" in (c.types or "")
    )
    logger.info(f"Loaded {len(app.state.commander_names)} commanders")

    # Jinja templates live next to this file
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates

    # Static artifacts (HTML reports, per-build files)
    app.mount(
        "/artifacts",
        StaticFiles(directory=str(app.state.artifacts_dir)),
        name="artifacts",
    )

    _register_routes(app)
    return app


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return app.state.templates.TemplateResponse(
            request,
            "home.html",
            { "csv": app.state.csv_path},
        )

    # -- Build --

    @app.get("/build", response_class=HTMLResponse)
    async def build_form(request: Request):
        return app.state.templates.TemplateResponse(
            request,
            "build_form.html",
            {
                "commanders": app.state.commander_names,
                "mock_llm": app.state.mock_llm,
            },
        )

    @app.post("/build")
    async def build_submit(request: Request):
        form = await request.form()
        form_dict = {k: v for k, v in form.items()}
        try:
            config = config_from_form(form_dict)
        except ValueError as e:
            return app.state.templates.TemplateResponse(
                request,
                "build_form.html",
                {
                    "commanders": app.state.commander_names,
                    "mock_llm": app.state.mock_llm,
                    "error": str(e),
                    "form": form_dict,
                },
                status_code=400,
            )

        registry: BuildRegistry = app.state.registry
        state = registry.create(commander_name=config.commander_name)

        # Attach the current event loop so the worker thread can push
        # progress events back
        state.loop = asyncio.get_running_loop()
        state.queue = asyncio.Queue()

        # Kick off the build in a background thread
        want_mock = app.state.mock_llm or form_dict.get("mock") == "on"
        app.state.executor.submit(
            _run_build,
            state=state,
            config=config,
            csv_path=app.state.csv_path,
            mock_llm=want_mock,
            artifacts_dir=app.state.artifacts_dir,
        )

        return RedirectResponse(url=f"/build/{state.build_id}", status_code=303)

    @app.get("/build/{build_id}", response_class=HTMLResponse)
    async def build_status(request: Request, build_id: str):
        state = app.state.registry.get(build_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Build not found")
        return app.state.templates.TemplateResponse(
            request,
            "build_status.html",
            {
                "build_id": build_id,
                "state": state,
                "report_url": _report_url(state) if state.report_path else None,
            },
        )

    @app.get("/build/{build_id}/events")
    async def build_events(build_id: str):
        """SSE endpoint — streams progress events."""
        state = app.state.registry.get(build_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Build not found")
        return StreamingResponse(
            _event_generator(state),
            media_type="text/event-stream",
        )

    @app.get("/build/{build_id}/report", response_class=HTMLResponse)
    async def build_report(build_id: str):
        state = app.state.registry.get(build_id)
        if state is None or not state.report_path:
            raise HTTPException(
                status_code=404,
                detail="Report not available (build still running or failed)",
            )
        return FileResponse(state.report_path, media_type="text/html")

    # -- Diff --

    @app.get("/diff", response_class=HTMLResponse)
    async def diff_form(request: Request):
        return app.state.templates.TemplateResponse(
            request,
            "diff_form.html",
            {},
        )

    @app.post("/diff", response_class=HTMLResponse)
    async def diff_submit(
        request: Request,
        before: UploadFile = File(...),
        after: UploadFile = File(...),
    ):
        try:
            before_data = json.loads(await before.read())
            after_data = json.loads(await after.read())
        except json.JSONDecodeError as e:
            return app.state.templates.TemplateResponse(
                request,
                "diff_form.html",
                { "error": f"Invalid JSON: {e}"},
                status_code=400,
            )
        try:
            result = diff_decks(before_data, after_data)
        except Exception as e:
            return app.state.templates.TemplateResponse(
                request,
                "diff_form.html",
                { "error": f"Diff failed: {e}"},
                status_code=400,
            )
        html = render_diff_html(
            result,
            before_name=before.filename,
            after_name=after.filename,
        )
        return HTMLResponse(content=html)

    # -- Analyze --

    @app.get("/analyze", response_class=HTMLResponse)
    async def analyze_form(request: Request):
        return app.state.templates.TemplateResponse(
            request,
            "analyze_form.html",
            {
                "commanders": app.state.commander_names,
                "mock_llm": app.state.mock_llm,
            },
        )

    @app.post("/analyze", response_class=HTMLResponse)
    async def analyze_submit(request: Request):
        form = await request.form()
        name = (form.get("commander_name") or "").strip()
        if not name:
            return app.state.templates.TemplateResponse(
                request,
                "analyze_form.html",
                {
                    "commanders": app.state.commander_names,
                    "mock_llm": app.state.mock_llm,
                    "error": "commander_name is required",
                },
                status_code=400,
            )
        want_mock = app.state.mock_llm or form.get("mock") == "on"
        # Run the synchronous analysis in the threadpool so we don't
        # block the event loop
        try:
            loop = asyncio.get_running_loop()
            analysis = await loop.run_in_executor(
                app.state.executor,
                _run_analyze, name, want_mock, app.state.csv_path,
            )
        except Exception as e:
            return app.state.templates.TemplateResponse(
                request,
                "analyze_form.html",
                {
                    "commanders": app.state.commander_names,
                    "mock_llm": app.state.mock_llm,
                    "error": f"Analysis failed: {e}",
                },
                status_code=500,
            )
        return app.state.templates.TemplateResponse(
            request,
            "analyze_result.html",
            {
                "commander": name,
                "analysis": analysis,
            },
        )

    # -- Tag cache pre-seeding --

    @app.get("/tags", response_class=HTMLResponse)
    async def tags_form(request: Request):
        return app.state.templates.TemplateResponse(
            request,
            "tags_form.html",
            {},
        )

    @app.post("/tags", response_class=HTMLResponse)
    async def tags_submit(request: Request):
        form = await request.form()
        tag_lines = [
            line.strip()
            for line in (form.get("tags") or "").splitlines()
            if line.strip()
        ]
        kind = (form.get("kind") or "art").strip()
        if kind not in ("art", "oracle"):
            kind = "art"
        cache_dir = (form.get("cache_dir") or "").strip() or None

        if not tag_lines:
            return app.state.templates.TemplateResponse(
                request,
                "tags_form.html",
                { "error": "No tags provided"},
                status_code=400,
            )

        # Run pre-seed in threadpool (may hit network)
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                app.state.executor,
                _run_prefetch_tags, tag_lines, kind, cache_dir,
            )
        except Exception as e:
            return app.state.templates.TemplateResponse(
                request,
                "tags_form.html",
                { "error": f"Pre-fetch failed: {e}"},
                status_code=500,
            )

        return app.state.templates.TemplateResponse(
            request,
            "tags_result.html",
            {
                "kind": kind,
                "results": results,
            },
        )


# ----------------------------------------------------------------------
# Helpers (work done outside the event loop)
# ----------------------------------------------------------------------

def _run_build(
    state: BuildState,
    config: BuildConfig,
    csv_path: str,
    mock_llm: bool,
    artifacts_dir: Path,
) -> None:
    """Runs in a worker thread. Never raises into FastAPI."""
    state.status = "running"

    def progress_cb(bp):
        """Real DeckBuilder passes a BuildProgress dataclass."""
        state.post_event(ProgressEvent(
            phase=bp.phase,
            status=bp.step,
            fraction=bp.progress,
            message=bp.message,
        ))

    try:
        builder = DeckBuilder(
            csv_path, config,
            llm_config=LLMConfig(mock_mode=mock_llm),
            progress_callback=progress_cb,
        )
        result = builder.build()
        state.result = result

        # Write HTML report next to the build_id
        report_path = artifacts_dir / f"{state.build_id}.html"
        generate_html_report(
            result,
            report_path,
            card_source=builder.card_source,
        )
        state.report_path = str(report_path)
        state.status = "complete"
    except Exception as e:
        state.status = "failed"
        state.error = f"{type(e).__name__}: {e}"
        state.post_event(ProgressEvent(
            phase="error", status="failed", fraction=1.0,
            message=state.error,
        ))
        logger.error(
            "Build %s failed:\n%s",
            state.build_id, traceback.format_exc(),
        )
    finally:
        import time as _t
        state.finished_at = _t.time()
        # Push a sentinel so the SSE generator knows we're done
        state.post_event(ProgressEvent(
            phase="__done__", status=state.status,
            fraction=1.0, message=state.status,
        ))


def _run_analyze(commander_name: str, mock_llm: bool, csv_path: str):
    """Threadpool: run just the LLM commander-analysis phase.

    We need a Card object, not just a name — so look up from the DB.
    """
    from ..llm_engine import LLMEngine
    from ..card_database import CardDatabase
    db = CardDatabase(csv_path)
    db.load()
    card = db.get_by_name(commander_name)
    if card is None:
        raise ValueError(f"Commander not found in database: {commander_name!r}")
    llm = LLMEngine(LLMConfig(mock_mode=mock_llm))
    return llm.analyze_commander(card)


def _run_prefetch_tags(
    tag_names: list[str], kind: str, cache_dir: Optional[str],
) -> list[dict]:
    """
    Threadpool: pre-fetch tag cache entries.

    Returns a list of {tag, count, error} dicts for display.
    """
    client = ScryfallTagClient(cache_dir=cache_dir)
    results = []
    for name in tag_names:
        try:
            if kind == "art":
                cards = client.get_cards_with_art_tag(name)
            else:
                cards = client.get_cards_with_oracle_tag(name)
            results.append({
                "tag": name, "count": len(cards), "error": None,
            })
        except Exception as e:
            results.append({
                "tag": name, "count": 0, "error": str(e),
            })
    return results


async def _event_generator(state: BuildState):
    """SSE format: each event is 'data: <json>\\n\\n'.

    Strategy:
    1. Snapshot current event history and yield it (catches up a late
       consumer to whatever's already happened).
    2. Drain anything currently in the queue ONCE without yielding —
       these are duplicates of events 1 already yielded from history,
       since post_event() pushes to both list and queue atomically.
    3. Then enter the main loop: await new events from the queue.

    This avoids the race where event E1 is in both state.events AND
    state.queue at the moment the SSE connection opens — without the
    drain step, we'd yield E1 twice.
    """
    # If no queue was attached (e.g., status page opened for a build
    # that was created but never had its loop/queue set up), we can
    # only replay what's already in events. This shouldn't happen via
    # the UI flow but better to degrade gracefully than crash.
    if state.queue is None:
        for evt in state.events:
            yield _sse_event(evt)
            if evt.phase == "__done__":
                return
        # Nothing more we can do without a queue
        yield _sse_event(ProgressEvent(
            phase="__done__", status="closed", fraction=1.0,
            message="(no live stream available)",
        ))
        return

    # Step 1: snapshot and yield history
    history_count = len(state.events)
    for evt in state.events[:history_count]:
        yield _sse_event(evt)
        if evt.phase == "__done__":
            return

    # Step 2: drain duplicates from the queue (events that were posted
    # before the snapshot was taken AND are also in our snapshot)
    drained = 0
    while drained < history_count:
        try:
            state.queue.get_nowait()
            drained += 1
        except asyncio.QueueEmpty:
            break

    # Step 3: stream new events from the queue
    while True:
        try:
            evt = await asyncio.wait_for(state.queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            # Keep-alive comment so proxies don't drop the connection
            yield ": keepalive\n\n"
            continue
        yield _sse_event(evt)
        if evt.phase == "__done__":
            return


def _sse_event(evt: ProgressEvent) -> str:
    return f"data: {json.dumps(evt.to_dict())}\n\n"


def _report_url(state: BuildState) -> str:
    return f"/build/{state.build_id}/report"


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    """python -m mtg_deck_builder.web ..."""
    parser = argparse.ArgumentParser(
        description="Launch the MTG Deck Builder web interface"
    )
    parser.add_argument("--csv", required=True, help="Path to cards CSV")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM for all requests (no API key needed)")
    parser.add_argument(
        "--artifacts-dir", default="./web_artifacts",
        help="Where to write generated HTML reports",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = create_app(
        csv_path=args.csv,
        mock_llm=args.mock,
        artifacts_dir=args.artifacts_dir,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
