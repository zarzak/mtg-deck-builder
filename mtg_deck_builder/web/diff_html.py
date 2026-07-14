"""
Render a DiffResult as styled HTML.

Wishlist item carried over from Session 5 ("Diff mode HTML output").
Produces a single-page view suitable for mobile and desktop — uses the
same visual palette as the main HTML report (system fonts, dark-ish
neutral colors, card thumbnails where available).

Not a full report page — no images here (we don't have a card_source
handy in the diff view), just name-based lists with role grouping
when the diff result supplies it.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..deck_diff import DiffResult


# Minimal CSS lifted from html_report.py to keep visual consistency.
# Inline so the diff page renders standalone with no external assets.
_CSS = """
:root {
  --bg: #0f1115;
  --panel: #151821;
  --border: #262b36;
  --text: #e6e9ef;
  --muted: #9aa3b2;
  --accent: #7aa2ff;
  --added: #3fb88a;
  --removed: #e05a5a;
  --unchanged: #7a8296;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1rem;
  background: var(--bg); color: var(--text);
  font-family: -apple-system, system-ui, sans-serif;
  font-size: 16px; line-height: 1.5;
}
.container { max-width: 900px; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 0.5rem 0; }
h2 { font-size: 1.1rem; margin: 1.5rem 0 0.5rem 0; color: var(--accent); }
.meta {
  color: var(--muted); font-size: 0.9rem;
  padding: 0.5rem 0; border-bottom: 1px solid var(--border);
  margin-bottom: 1rem;
}
.meta code {
  background: var(--panel); padding: 2px 6px; border-radius: 3px;
  font-size: 0.85rem;
}
.summary {
  display: flex; gap: 1rem; flex-wrap: wrap;
  margin-bottom: 1rem;
}
.stat {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  flex: 1 1 150px;
  text-align: center;
}
.stat .n { font-size: 1.6rem; font-weight: 600; }
.stat .label { color: var(--muted); font-size: 0.8rem; }
.stat.added .n { color: var(--added); }
.stat.removed .n { color: var(--removed); }
.stat.unchanged .n { color: var(--unchanged); }
.role-group {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  margin-bottom: 0.75rem;
}
.role-group h3 {
  margin: 0 0 0.5rem 0; font-size: 0.95rem; color: var(--muted);
  font-weight: 500; text-transform: uppercase; letter-spacing: 0.03em;
}
ul.cards { list-style: none; padding: 0; margin: 0; }
ul.cards li {
  padding: 0.25rem 0; border-bottom: 1px solid var(--border);
}
ul.cards li:last-child { border-bottom: none; }
.added .marker { color: var(--added); }
.removed .marker { color: var(--removed); }
.marker {
  display: inline-block; min-width: 1.5rem;
  font-family: monospace; font-weight: 600;
}
.commander-change {
  background: #2a1a1a; border-left: 3px solid var(--removed);
  padding: 0.75rem 1rem; margin: 0 0 1rem 0;
  border-radius: 4px;
}
"""


def render_diff_html(
    result: "DiffResult",
    before_name: Optional[str] = None,
    after_name: Optional[str] = None,
) -> str:
    """
    Render a DiffResult as a full standalone HTML page.

    Args:
        result: The DiffResult to render.
        before_name: Display label for the "before" snapshot (e.g. a
            filename). Falls back to "before" if not provided.
        after_name: Same for after.

    Output includes:
    - Summary stats (added, removed, unchanged counts)
    - Commander change callout (if commanders differ)
    - Added cards (grouped by role if available, flat list otherwise)
    - Removed cards (same treatment)
    """
    before_label = html.escape(before_name or "before")
    after_label = html.escape(after_name or "after")

    added = list(result.added)
    removed = list(result.removed)
    kept_count = len(result.kept)

    # Commander change
    commander_from = result.commander_from
    commander_to = result.commander_to
    commander_changed = (
        commander_from is not None
        and commander_to is not None
        and commander_from != commander_to
    )

    # Role groupings (empty dicts if no CardDatabase was provided to diff_decks)
    added_by_role = result.added_by_role
    removed_by_role = result.removed_by_role

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Deck Diff</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        '<div class="container">',
        "<h1>Deck Diff</h1>",
        f'<div class="meta">Comparing <code>{before_label}</code> → '
        f'<code>{after_label}</code></div>',
    ]

    # Commander change callout
    if commander_changed:
        parts.append(
            f'<div class="commander-change">'
            f'<strong>Commander changed:</strong> '
            f'{html.escape(commander_from)} → '
            f'{html.escape(commander_to)}'
            f'</div>'
        )

    # Summary stats
    parts.append('<div class="summary">')
    parts.append(
        f'<div class="stat added"><div class="n">+{len(added)}</div>'
        f'<div class="label">added</div></div>'
    )
    parts.append(
        f'<div class="stat removed"><div class="n">−{len(removed)}</div>'
        f'<div class="label">removed</div></div>'
    )
    parts.append(
        f'<div class="stat unchanged"><div class="n">{kept_count}</div>'
        f'<div class="label">unchanged</div></div>'
    )
    parts.append('</div>')

    # Added section
    parts.append("<h2>Added</h2>")
    if not added:
        parts.append('<p class="meta">No cards added.</p>')
    elif added_by_role:
        parts.extend(_render_role_groups(added_by_role, "added", "+"))
    else:
        parts.append(_render_flat_list(added, "added", "+"))

    # Removed section
    parts.append("<h2>Removed</h2>")
    if not removed:
        parts.append('<p class="meta">No cards removed.</p>')
    elif removed_by_role:
        parts.extend(_render_role_groups(removed_by_role, "removed", "−"))
    else:
        parts.append(_render_flat_list(removed, "removed", "−"))

    parts.append("</div></body></html>")
    return "\n".join(parts)


def _render_flat_list(names: list[str], css_class: str, marker: str) -> str:
    items = "".join(
        f'<li class="{css_class}">'
        f'<span class="marker">{marker}</span>'
        f'{html.escape(n)}'
        f'</li>'
        for n in names
    )
    return f'<ul class="cards">{items}</ul>'


def _render_role_groups(
    by_role: dict, css_class: str, marker: str
) -> list[str]:
    out: list[str] = []
    # Stable role ordering; anything not in this list goes to "other"
    role_order = [
        "land", "ramp", "draw", "removal", "wipe",
        "creature", "enchantment", "artifact", "planeswalker",
        "other",
    ]
    seen: set[str] = set()

    def emit(role: str, names: list):
        if not names:
            return
        out.append('<div class="role-group">')
        out.append(f"<h3>{html.escape(role)} ({len(names)})</h3>")
        out.append(_render_flat_list(sorted(names), css_class, marker))
        out.append("</div>")

    for role in role_order:
        if role in by_role:
            emit(role, by_role[role])
            seen.add(role)
    # Catch any roles not in the standard order
    for role, names in by_role.items():
        if role not in seen:
            emit(role, names)
    return out
