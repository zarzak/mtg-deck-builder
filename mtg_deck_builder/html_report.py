"""
HTML report generator for deck build results.

Produces a single-file HTML report suitable for reading on mobile.
Designed to work offline (no external CDN resources).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

from .models import OptimizationResult, CardTelemetry


# Minimal inline CSS — no external dependencies for true offline use.
# Uses system fonts and is readable on small screens.
_CSS = """
:root {
  --bg: #fafafa; --fg: #222; --muted: #666; --accent: #3b6ea5;
  --card-bg: #fff; --border: #e2e2e2; --good: #3aa35b; --bad: #c0392b;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #1a1a1a; --fg: #eee; --muted: #999; --accent: #6a9ace;
          --card-bg: #262626; --border: #333; --good: #4cc773; --bad: #e06055; }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  line-height: 1.5; }
.container { max-width: 820px; margin: 0 auto; padding: 1rem; }
h1, h2, h3 { margin-top: 2rem; margin-bottom: 0.5rem; }
h1 { font-size: 1.6rem; }
h2 { font-size: 1.3rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
h3 { font-size: 1.05rem; color: var(--accent); }
.subtitle { color: var(--muted); font-size: 0.9rem; margin-top: -0.5rem; }
.card { background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
.score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.5rem; }
.score { padding: 0.5rem; border-radius: 6px; background: var(--bg); text-align: center; }
.score .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.05em; }
.score .value { font-size: 1.4rem; font-weight: 600; }
.score.final { grid-column: 1 / -1; background: var(--accent); color: #fff; }
.score.final .label { color: rgba(255,255,255,0.8); }
.bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-top: 4px; }
.bar-fill { height: 100%; background: var(--accent); transition: width .3s; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { padding: 0.4rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border); }
th { font-weight: 600; color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.prov { font-size: 0.78rem; color: var(--muted); font-family: monospace; }
.role-pill { display: inline-block; padding: 2px 8px; border-radius: 10px;
  background: var(--border); font-size: 0.7rem; color: var(--muted); }
.role-pill.synergy-other { background: #fef3c7; color: #92400e; }
@media (prefers-color-scheme: dark) {
  .role-pill.synergy-other { background: #3d3020; color: #e3c77c; }
}
/* v0.4: card image support */
td.card-thumb { width: 70px; padding: 4px; }
td.card-thumb img { border-radius: 4px; display: block; max-width: 60px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.5rem; }
.gallery-card { text-align: center; font-size: 0.75rem; }
.gallery-card img { width: 100%; border-radius: 6px; display: block; margin-bottom: 2px; }
.gallery-card .cname { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.art-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.5rem; margin-top: 0.5rem; }
.art-strip img { width: 100%; border-radius: 6px; display: block; }
.art-strip .credit { font-size: 0.7rem; color: var(--muted); margin-top: 2px; }
details { margin: 0.5rem 0; }
summary { cursor: pointer; font-weight: 600; padding: 0.3rem 0; }
.decklist { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.82rem; background: var(--bg); padding: 1rem; border-radius: 6px;
  white-space: pre-wrap; overflow-x: auto; }
.meta { font-size: 0.85rem; color: var(--muted); }
.meta-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.3rem; }
.spark { display: inline-block; height: 30px; vertical-align: middle; }
.review { white-space: pre-wrap; }
"""


def _comparable_score_history(
    score_history: list[float],
    eval_mode_history: Optional[list[str]],
) -> list[float]:
    """
    Return the slice of `score_history` that is safe to plot on one axis.

    The optimizer evaluates early generations with a cheap "fast" evaluator
    and later ones with the "full" evaluator; the two score on different
    scales, so a single sparkline over the raw history shows a spurious
    cliff at the phase boundary. When per-point mode tags are available we
    keep only the full-eval points (the meaningful optimization tail, which
    is monotonic under elitism). With no tags (older results) or no full-eval
    points, fall back to the whole history.
    """
    if not eval_mode_history or len(eval_mode_history) != len(score_history):
        return score_history
    full = [
        v for v, m in zip(score_history, eval_mode_history)
        if str(m).lower() == "full"
    ]
    return full or score_history


def _sparkline_svg(values: list[float], width: int = 200, height: int = 30) -> str:
    """Render a tiny SVG sparkline for the score history."""
    if not values:
        return ""
    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v if max_v > min_v else 1.0
    n = len(values)
    points = []
    for i, v in enumerate(values):
        x = (i / max(1, n - 1)) * width
        y = height - ((v - min_v) / span) * height
        points.append(f"{x:.1f},{y:.1f}")
    pts_attr = " ".join(points)
    return (
        f'<svg class="spark" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="{pts_attr}" />'
        f'</svg>'
    )


def _score_tile(label: str, value: float, is_final: bool = False) -> str:
    cls = "score final" if is_final else "score"
    pct = max(0, min(100, value))
    return (
        f'<div class="{cls}">'
        f'  <div class="label">{html.escape(label)}</div>'
        f'  <div class="value">{value:.1f}</div>'
        f'  <div class="bar"><div class="bar-fill" style="width: {pct:.1f}%"></div></div>'
        f'</div>'
    )


# ----------------------------------------------------------------------
# Defensive wrappers around card_source method calls.
# A buggy or partially-broken card_source must NEVER crash report
# generation — at worst, a card just renders without its image.
# ----------------------------------------------------------------------

def _safe_image_url(card_source, name: str, size: str) -> Optional[str]:
    """Get image URL from card_source; return None on any failure."""
    if card_source is None:
        return None
    try:
        return card_source.get_image_url(name, size=size)
    except Exception:
        # Not even a debug log here — a busted card_source could spam logs
        # with hundreds of identical errors during one report generation.
        return None


def _safe_artist(card_source, name: str) -> Optional[str]:
    """Get artist from card_source; return None on any failure."""
    if card_source is None:
        return None
    try:
        return card_source.get_artist(name)
    except Exception:
        return None


def _render_telemetry_table(
    telemetry: list[CardTelemetry],
    card_source=None,
) -> str:
    """
    Render the per-card scoring table, sorted by effective score.

    If card_source is provided (a ScryfallCardSource), adds a small image
    thumbnail column on the left.
    """
    sorted_t = sorted(telemetry, key=lambda t: t.effective_score, reverse=True)
    rows = []
    show_images = card_source is not None
    for t in sorted_t:
        role_class = t.role.replace('/', '-').replace(' ', '-')
        img_cell = ""
        if show_images:
            img_url = _safe_image_url(card_source, t.name, "small")
            if img_url:
                img_cell = (
                    f'<td class="card-thumb">'
                    f'<img loading="lazy" src="{html.escape(img_url)}" '
                    f'alt="{html.escape(t.name)}" width="60" />'
                    f"</td>"
                )
            else:
                img_cell = '<td class="card-thumb"></td>'
        # v0.9.33 (#26): the channel(s) that put this card into the pool.
        prov = ", ".join(getattr(t, "provenance", []) or []) or "role-skeleton"
        rows.append(
            "<tr>"
            + img_cell
            + f"<td>{html.escape(t.name)}</td>"
            + f'<td><span class="role-pill {html.escape(role_class)}">'
              f"{html.escape(t.role)}</span></td>"
            + f'<td class="num">{t.baseline_power:.0f}</td>'
            + f'<td class="num">{t.synergy_score:.0f}</td>'
            + f'<td class="num"><b>{t.effective_score:.0f}</b></td>'
            + f'<td class="prov">{html.escape(prov)}</td>'
            + "</tr>"
        )

    header_cells = []
    if show_images:
        header_cells.append('<th style="width: 70px;"></th>')
    header_cells += [
        "<th>Card</th>", "<th>Role</th>",
        "<th>Baseline</th>", "<th>Synergy</th>", "<th>Effective</th>",
        "<th>Pool entry</th>",
    ]
    return (
        "<table>"
        "<thead><tr>" + "".join(header_cells) + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def _render_role_counts_table(role_counts: dict[str, int]) -> str:
    rows = []
    for role, count in sorted(role_counts.items()):
        rows.append(
            f"<tr><td>{html.escape(role.title())}</td>"
            f'<td class="num">{count}</td></tr>'
        )
    return (
        "<table>"
        "<thead><tr><th>Role</th><th>Count</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def _render_combos_section(combos, deck_names: set[str]) -> str:
    """Render the v0.9.8 combo section: which detected combos the final deck
    fully assembled or came one piece short of. Empty string when combo
    detection didn't run (so the section vanishes for legacy builds)."""
    if not combos:
        return ""

    complete, near = [], []
    for combo in combos:
        cards = list(getattr(combo, "cards", []) or [])
        n = len(cards)
        if n < 2:
            continue
        present = [c for c in cards if c in deck_names]
        k = len(present)
        missing = [c for c in cards if c not in deck_names]
        payoff = int(getattr(combo, "payoff", 0))
        result = getattr(combo, "result", "") or ""
        if k >= n:
            complete.append((payoff, cards, result))
        elif k == n - 1:
            near.append((payoff, present, missing, result))

    if not complete and not near:
        return (
            "<h2>Combos</h2><div class=\"card\">"
            "<p class='meta'>No detected combos were assembled in the final "
            "deck.</p></div>"
        )

    parts = ["<h2>Combos</h2><div class=\"card\">"]
    if complete:
        complete.sort(reverse=True)
        rows = "".join(
            f"<tr><td>{html.escape(' + '.join(cards))}</td>"
            f"<td>{html.escape(res)}</td>"
            f'<td class="num">{payoff}</td></tr>'
            for payoff, cards, res in complete
        )
        parts.append(
            "<h3 style='margin:0 0 .3rem'>Assembled "
            f"({len(complete)})</h3><table><thead><tr><th>Combo</th>"
            "<th>Result</th><th>Payoff</th></tr></thead><tbody>"
            + rows + "</tbody></table>"
        )
    if near:
        near.sort(reverse=True)
        rows = "".join(
            f"<tr><td>{html.escape(' + '.join(present))}</td>"
            f"<td>+ {html.escape(', '.join(missing))}</td>"
            f'<td class="num">{payoff}</td></tr>'
            for payoff, present, missing, res in near[:15]
        )
        parts.append(
            "<h3 style='margin:.6rem 0 .3rem'>One piece away "
            f"({len(near)})</h3><table><thead><tr><th>Have</th>"
            "<th>Missing</th><th>Payoff</th></tr></thead><tbody>"
            + rows + "</tbody></table>"
        )
    parts.append("</div>")
    return "".join(parts)


def _render_commander_art(deck, card_source) -> str:
    """Render commander art at the top of the report (optional)."""
    if card_source is None:
        return ""
    commander = deck.commander
    art_url = _safe_image_url(card_source, commander.name, "art_crop")
    if not art_url:
        return ""
    artist = _safe_artist(card_source, commander.name) or "Unknown artist"
    return f"""
    <div class="art-strip">
      <div>
        <img loading="lazy" src="{html.escape(art_url)}"
             alt="{html.escape(commander.name)} art" />
        <div class="credit">Art by {html.escape(artist)}</div>
      </div>
    </div>
    """


def _render_art_gallery(deck, card_source, limit: int = 24) -> str:
    """
    Render a grid of art crops for the deck. Shows up to `limit` creatures
    (skipping lands and duplicates) so the gallery stays a reasonable size.
    """
    if card_source is None:
        return ""

    # Dedupe and prefer creatures + unique non-basics for visual interest
    seen = set()
    shown = []
    # Creatures first
    for card in deck.cards:
        if card.name in seen:
            continue
        if card.is_creature:
            seen.add(card.name)
            shown.append(card)
            if len(shown) >= limit:
                break
    # Fill remaining slots with non-land, non-basic cards
    if len(shown) < limit:
        for card in deck.cards:
            if card.name in seen:
                continue
            if not card.is_land:
                seen.add(card.name)
                shown.append(card)
                if len(shown) >= limit:
                    break

    if not shown:
        return ""

    entries = []
    for card in shown:
        art_url = _safe_image_url(card_source, card.name, "art_crop")
        if not art_url:
            continue
        artist = _safe_artist(card_source, card.name) or ""
        artist_html = (
            f'<div class="credit">{html.escape(artist)}</div>'
            if artist else ""
        )
        entries.append(
            f'<div class="gallery-card">'
            f'<img loading="lazy" src="{html.escape(art_url)}" '
            f'alt="{html.escape(card.name)}" />'
            f'<div class="cname" title="{html.escape(card.name)}">'
            f"{html.escape(card.name)}</div>"
            f"{artist_html}</div>"
        )

    if not entries:
        return ""

    return (
        '<div class="gallery">' + "".join(entries) + "</div>"
    )


def generate_html_report(
    result: OptimizationResult,
    output_path: str | Path,
    card_source=None,
) -> Path:
    """
    Generate a self-contained HTML report for the optimization result.

    Args:
        result: The OptimizationResult from DeckBuilder.build()
        output_path: Where to write the HTML file
        card_source: Optional ScryfallCardSource (or similar duck-typed object
            with .get_image_url(name, size) and .get_artist(name)).
            When provided, the report embeds commander art, an art gallery,
            and thumbnails in the telemetry table.

    Returns:
        Path to the written HTML file
    """
    output_path = Path(output_path)
    deck = result.best_deck
    scores = deck.scores
    config = result.config
    analysis = result.commander_analysis

    # Header
    title = f"{deck.commander.name} — Deck Build Report"

    # Summary meta. Prefer the total end-to-end wall-clock; fall back to the
    # GA-only runtime for older results that didn't record it.
    total_runtime = getattr(result, "total_runtime_seconds", None)
    if total_runtime is not None:
        runtime_str = f"{total_runtime:.0f}s total ({result.runtime_seconds:.1f}s GA)"
    else:
        runtime_str = f"{result.runtime_seconds:.1f}s runtime"
    meta_parts = [
        f"{len(deck.cards)} cards",
        f"{result.generations_run} generations",
        runtime_str,
        f"seed={config.random_seed}" if config.random_seed is not None else "no seed",
    ]
    meta_str = " · ".join(meta_parts)

    # Commander analysis section
    analysis_html = ""
    if analysis:
        kw_html = ", ".join(html.escape(k) for k in (analysis.synergy_keywords or [])) or "(none)"
        mech_html = ", ".join(html.escape(m) for m in (analysis.key_mechanics or [])) or "(none)"
        analysis_html = f"""
        <div class="card">
          <h3>Strategy</h3>
          <p>{html.escape(analysis.build_around_text or '')}</p>
          <h3>Key Mechanics</h3>
          <p>{mech_html}</p>
          <h3>Synergy Keywords</h3>
          <p class="meta">{kw_html}</p>
          <h3>Evaluation Notes</h3>
          <p class="meta">{html.escape(analysis.evaluation_notes or '')}</p>
        </div>
        """

    # Score tiles
    score_tiles = []
    score_tiles.append(_score_tile("Total", result.final_score, is_final=True))
    if scores:
        score_tiles.append(_score_tile("Mana Curve", scores.mana_curve))
        score_tiles.append(_score_tile("Role Coverage", scores.role_coverage))
        score_tiles.append(_score_tile("Synergy", scores.synergy))
        # v0.9.3: strategy density — what % of non-mana-land cards are
        # strongly on-strategy (synergy score >= 60).
        score_tiles.append(_score_tile("Strategy Density", scores.strategy_density))
        score_tiles.append(_score_tile("Power Level", scores.power_level))
        # v0.9.14: consistency (core-effect-class redundancy) — only shown
        # when effect-class data produced a score.
        if getattr(scores, "consistency", 0.0) > 0:
            score_tiles.append(_score_tile("Consistency", scores.consistency))
        # v0.9.7: creativity is informational only — it does not feed the
        # weighted total (see DeckScores.total).
        score_tiles.append(_score_tile("Creativity (unscored)", scores.creativity))
        score_tiles.append(_score_tile("Effective Synergy", scores.effective_synergy))
        # v0.9.8: combo dimension — only shown when combo detection ran.
        if getattr(result, "combos", None):
            score_tiles.append(_score_tile("Combos", scores.combo))

    # Progress sparkline. The GA's fast and full evaluators score on
    # different scales, so plotting the raw history shows a misleading
    # "crash" where the phases meet. Plot only the comparable full-eval
    # segment when we know which points came from which evaluator; fall
    # back to the whole history for older results without mode tags.
    spark_values = _comparable_score_history(
        result.score_history,
        getattr(result, "eval_mode_history", None),
    )
    sparkline = _sparkline_svg(spark_values) if spark_values else ""

    # Role counts table
    role_counts_html = (
        _render_role_counts_table(scores.role_counts)
        if scores and scores.role_counts else "<p class='meta'>No role counts recorded.</p>"
    )

    # v0.9.8: assembled / near-complete combos in the final deck. The
    # commander counts as present (always in play) to match the fitness.
    deck_names = {c.name for c in deck.cards}
    if deck.commander is not None:
        deck_names.add(deck.commander.name)
    combos_section = _render_combos_section(
        getattr(result, "combos", None), deck_names
    )

    # Telemetry table (with card thumbnails if card_source present)
    telemetry_html = (
        _render_telemetry_table(result.card_telemetry, card_source=card_source)
        if result.card_telemetry else "<p class='meta'>No per-card telemetry.</p>"
    )

    # v0.4: Commander art and deck gallery (only if card_source provided)
    commander_art_html = _render_commander_art(deck, card_source)
    gallery_html = _render_art_gallery(deck, card_source, limit=24)
    gallery_section = ""
    if gallery_html:
        gallery_section = f"""
        <h2>Deck Art Gallery</h2>
        <div class="card">{gallery_html}</div>
        """

    # Decklist
    decklist_text = deck.to_decklist()

    # v0.9.15: bracket compliance panel
    bracket_html = ""
    audit = getattr(result, "bracket_audit", None)
    if audit is not None:
        from .bracket import bracket_name
        gc_str = ", ".join(html.escape(n) for n in audit.game_changers) or "(none)"
        limit_str = "unlimited" if audit.gc_limit is None else str(audit.gc_limit)
        rows = [
            f"<div class=\"meta-row\"><span><b>Game Changers "
            f"({len(audit.game_changers)}/{limit_str}):</b> {gc_str}</span></div>",
        ]
        if audit.mld_cards:
            rows.append(
                f"<div class=\"meta-row\"><span><b>Mass land denial:</b> "
                f"{html.escape(', '.join(audit.mld_cards))}</span></div>"
            )
        if audit.extra_turn_cards:
            rows.append(
                f"<div class=\"meta-row\"><span><b>Extra turns:</b> "
                f"{html.escape(', '.join(audit.extra_turn_cards))}</span></div>"
            )
        if audit.two_card_combos:
            combo_strs = [
                html.escape(c["desc"]) + (" <b>[early]</b>" if c["early"] else "")
                for c in audit.two_card_combos
            ]
            rows.append(
                "<div class=\"meta-row\"><span><b>Two-card combos:</b> "
                + "; ".join(combo_strs) + "</span></div>"
            )
        if audit.compliant:
            verdict = ('<span style="color: var(--good); font-weight: 600;">'
                       'COMPLIANT with target bracket</span>')
        else:
            items = "".join(f"<li>{html.escape(v)}</li>" for v in audit.violations)
            verdict = (f'<span style="color: var(--bad); font-weight: 600;">'
                       f'VIOLATIONS</span><ul>{items}</ul>')
        bracket_html = f"""
        <h2>Bracket Compliance — target {audit.bracket}
            ({html.escape(bracket_name(audit.bracket))})</h2>
        <div class="card">
          {''.join(rows)}
          <div class="meta-row"><span>{verdict}</span></div>
          <div class="meta-row"><span class="meta">Effective bracket by
          contents: {audit.effective_bracket}{
            " (4 and 5 are identical by contents; 5 is a metagame declaration)"
            if audit.effective_bracket == 4 else ""}</span></div>
        </div>
        """

    # v0.9.14: refinement swaps applied by the post-GA LLM pass
    refinement_html = ""
    refinement_log = getattr(result, "refinement_log", None)
    if refinement_log:
        rows = "".join(
            f"<tr><td>{html.escape(s.get('out', ''))}</td>"
            f"<td>{html.escape(s.get('in', ''))}</td>"
            f"<td class=\"meta\">{html.escape(s.get('reason', ''))}</td></tr>"
            for s in refinement_log
        )
        refinement_html = f"""
        <h2>LLM Refinement ({len(refinement_log)} swaps)</h2>
        <div class="card">
          <p class="meta">Set-level swaps applied after the GA — redundancy,
          interaction spread, and role-quality fixes the per-card fitness
          cannot see.</p>
          <table><thead><tr><th>Out</th><th>In</th><th>Reason</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>
        """

    # Optional LLM review
    review_html = ""
    if result.llm_review:
        review_html = f"""
        <h2>LLM Review</h2>
        <div class="card">
          <div class="review">{html.escape(result.llm_review)}</div>
        </div>
        """

    # Config summary
    weights = config.get_effective_weights(analysis)
    # v0.9.7: creativity is no longer a scored dimension — don't display it
    # as a weight even if a commander recommendation still lists it.
    weights_str = ", ".join(
        f"{k}={v:.2f}" for k, v in weights.items() if k != "creativity"
    )
    base_w, syn_w = config.get_effective_synergy_balance(analysis)

    config_html = f"""
    <h2>Configuration</h2>
    <details>
      <summary>Build settings</summary>
      <div class="card">
        <div class="meta-row"><span><b>Population</b> {config.population_size}</span>
            <span><b>Generations</b> {config.generations}</span>
            <span><b>Bracket</b> {getattr(config, 'bracket', 4)}</span></div>
        <div class="meta-row"><span><b>Scoring weights:</b> {html.escape(weights_str)}</span></div>
        <div class="meta-row"><span><b>Synergy balance:</b>
            baseline={base_w:.2f}, synergy={syn_w:.2f}</span></div>
        <div class="meta-row"><span><b>Model:</b> {html.escape(config.llm_model)}</span></div>
      </div>
    </details>
    """

    # Full HTML document
    # Read version dynamically — single source of truth in __init__.py
    try:
        from . import __version__ as _pkg_version
    except Exception:
        _pkg_version = "unknown"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">{html.escape(meta_str)}</div>

  {commander_art_html}

  <h2>Scores</h2>
  <div class="card">
    <div class="score-grid">
      {''.join(score_tiles)}
    </div>
    {'<h3 style="margin-top:1rem;">Best fitness over generations</h3>' if sparkline else ''}
    {sparkline}
  </div>

  {analysis_html}

  <h2>Role Coverage</h2>
  <div class="card">{role_counts_html}</div>

  {combos_section}

  {gallery_section}

  <h2>Decklist</h2>
  <div class="card">
    <div class="decklist">{html.escape(decklist_text)}</div>
  </div>

  {bracket_html}

  {refinement_html}

  <h2>Per-Card Telemetry</h2>
  <details open>
    <summary>{len(result.card_telemetry)} cards with scoring breakdown</summary>
    <div class="card">{telemetry_html}</div>
  </details>

  {review_html}

  {config_html}

  <div style="text-align:center; color: var(--muted); font-size: 0.75rem;
              margin-top: 3rem; padding-top: 2rem; border-top: 1px solid var(--border);">
    Generated by mtg_deck_builder v{_pkg_version}
  </div>
</div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
