"""
mtg_deck_builder.web — FastAPI + SSE web interface (v0.7).

Launch:
    python -m mtg_deck_builder.web --csv cards.csv

Binds to 127.0.0.1 by default. Single-user local tool — no auth, no
multi-user isolation, no deployment considerations. If you want to
expose this beyond your own machine, wrap it in something that does
TLS + auth. That's not Session 7.

Pages:
- `/`            — home; links to build/diff/analyze/tags
- `/build`       — build form; POST starts a background build
- `/build/{id}`  — live progress page (SSE) and, when done, the HTML report
- `/diff`        — upload two deck snapshots, get a rendered diff
- `/analyze`     — commander name → LLM analysis without a full build
- `/tags`        — pre-seed tag cache with a list of flavor tags
"""

# Deferred imports so `from mtg_deck_builder.web.forms import ...` works
# without requiring fastapi to be installed if you just want to test
# the form layer.


def create_app(*args, **kwargs):
    from .app import create_app as _ca
    return _ca(*args, **kwargs)


def main(*args, **kwargs):
    from .app import main as _main
    return _main(*args, **kwargs)


__all__ = ["create_app", "main"]
