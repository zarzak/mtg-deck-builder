#!/usr/bin/env python3
"""
CLI interface for MTG Deck Builder.

Examples:
  Optimized build:
    python -m mtg_deck_builder.cli build "Lathiel, the Bounteous Dawn" \\
        --csv cards.csv --generations 100 --report deck_report.html

  Mock-mode build (no API key):
    python -m mtg_deck_builder.cli build "Lathiel, the Bounteous Dawn" \\
        --csv cards.csv --mock --generations 30

  Commander analysis:
    python -m mtg_deck_builder.cli analyze "Jasmine Boreal" --csv cards.csv

  Quick build (heuristic-only, no GA):
    python -m mtg_deck_builder.cli quick "Karlov of the Ghost Council" --csv cards.csv

  Diff two saved deck snapshots:
    python -m mtg_deck_builder.cli --csv cards.csv diff old.json new.json
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .models import BuildConfig
from .deck_builder import DeckBuilder, BuildProgress
from .llm_engine import LLMConfig
from .card_database import CardDatabase
from .html_report import generate_html_report


# ----------------------------------------------------------------------
# v0.4 helper parsers for repeatable/structured flags
# ----------------------------------------------------------------------

# Weight presets for --preset
# Weights auto-normalize so they don't need to sum to 1.0.
# v0.9.7: `creativity` removed from scoring — it no longer appears here.
WEIGHT_PRESETS = {
    "balanced": {
        # v0.9.7 defaults (mirrors BuildConfig.score_weights): sums to 1.0.
        "mana_curve": 0.10, "role_coverage": 0.15, "synergy": 0.35,
        "strategy_density": 0.20, "power_level": 0.20,
        "flavor": 0.0,
    },
    "flavor": {
        # Theme/tribal: elevate flavor + density, de-emphasize raw power
        "mana_curve": 0.10, "role_coverage": 0.10, "synergy": 0.20,
        "strategy_density": 0.25, "power_level": 0.10,
        "flavor": 0.25,
    },
    "power": {
        # cEDH-leaning: push power level and role coverage
        "mana_curve": 0.10, "role_coverage": 0.20, "synergy": 0.15,
        "strategy_density": 0.15, "power_level": 0.40,
        "flavor": 0.0,
    },
    "budget": {
        # Value-optimization: maximize on-strategy value per card, power less so
        "mana_curve": 0.10, "role_coverage": 0.15, "synergy": 0.30,
        "strategy_density": 0.30, "power_level": 0.10,
        "flavor": 0.0,
    },
}


def _parse_role_target(s: str) -> tuple[str, tuple[int, int]]:
    """
    Parse 'role=min,max' into (role, (min, max)).
    Raises argparse.ArgumentTypeError on bad input.
    """
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--role-target expects ROLE=MIN,MAX format; got {s!r}"
        )
    role, rest = s.split("=", 1)
    role = role.strip().lower()
    if "," not in rest:
        raise argparse.ArgumentTypeError(
            f"--role-target expects ROLE=MIN,MAX format; got {s!r}"
        )
    lo_s, hi_s = rest.split(",", 1)
    try:
        lo, hi = int(lo_s.strip()), int(hi_s.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--role-target min/max must be integers; got {s!r}"
        )
    if lo > hi:
        lo, hi = hi, lo
    return role, (lo, hi)


def _parse_weight(s: str) -> tuple[str, float]:
    """
    Parse 'dim=val' into (dim, val).
    Raises argparse.ArgumentTypeError on bad input.
    """
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--weight expects DIM=VAL format; got {s!r}"
        )
    dim, val_s = s.split("=", 1)
    dim = dim.strip().lower()
    try:
        val = float(val_s.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--weight value must be a number; got {s!r}"
        )
    if val < 0:
        raise argparse.ArgumentTypeError(
            f"--weight value must be non-negative; got {val}"
        )
    return dim, val


def _build_weight_dict(
    preset_name: str | None,
    weight_overrides: list[str],
) -> dict[str, float] | None:
    """
    Build a score_weights dict from --preset + --weight flags.

    Order of precedence (later wins):
    1. Preset (if --preset)
    2. Individual --weight overrides

    Weights are auto-normalized to sum to 1.0 so users don't have to.
    Returns None if neither flag was used (caller keeps defaults).
    """
    if not preset_name and not weight_overrides:
        return None

    # Start with preset if given, else defaults
    weights = dict(WEIGHT_PRESETS[preset_name or "balanced"])

    # Apply individual overrides
    known_dims = set(weights.keys())
    for s in weight_overrides:
        dim, val = _parse_weight(s)
        if dim not in known_dims:
            logging.warning(
                f"Unknown weight dimension {dim!r}; "
                f"known: {sorted(known_dims)}"
            )
            continue
        weights[dim] = val

    # Normalize to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """
    Configure logging for the CLI.

    Console handler respects --verbose (DEBUG) vs. default (WARNING).
    File handler, when --log-file is set, always captures DEBUG so that
    forensic detail (full LLM responses on parse failure, prompt traces, etc.)
    is preserved without spamming the console.
    """
    console_level = logging.DEBUG if verbose else logging.WARNING

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    # Reset any prior handlers (e.g., from prior basicConfig in the same process)
    for h in list(root.handlers):
        root.removeHandler(h)

    # Root must be at the lowest level we plan to emit anywhere; per-handler
    # levels then filter what each destination actually shows.
    root.setLevel(logging.DEBUG if log_file else console_level)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            root.addHandler(fh)
            logging.getLogger(__name__).info(
                f"Debug log: {log_file} (full LLM responses captured here)"
            )
        except OSError as e:
            print(f"warning: could not open --log-file {log_file!r}: {e}",
                  file=sys.stderr)


def _use_unicode() -> bool:
    """
    Check if we can safely print unicode (progress bars, emoji).
    Returns False on Windows consoles that don't handle UTF-8 well.
    """
    if os.environ.get("MTG_ASCII"):
        return False
    # Windows cmd.exe historically has encoding issues
    if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower():
        return True
    if sys.platform == "win32":
        return False
    return True


class ProgressRenderer:
    """Renders progress updates. Falls back to ASCII on non-UTF-8 terminals."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.unicode = _use_unicode()
        self._last_phase = None

    def __call__(self, progress: BuildProgress):
        if self.quiet:
            return

        phase = progress.phase
        if phase != self._last_phase:
            # New phase — print a header
            if self._last_phase is not None:
                # Clear the progress line
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
            symbol = "==" if self.unicode else "=="
            print(f"\n{symbol} {phase.upper()}")
            self._last_phase = phase

        bar_width = 24
        filled = int(bar_width * progress.progress)

        if self.unicode:
            bar = "█" * filled + "░" * (bar_width - filled)
        else:
            bar = "#" * filled + "-" * (bar_width - filled)

        elapsed = f"{progress.elapsed_seconds:.0f}s" if progress.elapsed_seconds else ""
        # On a non-UTF-8 stream (Windows cp1252, piped stdout) the message may
        # carry unicode like the "→" we use in phase labels; writing it raw
        # raises UnicodeEncodeError and aborts the whole build. Sanitize the
        # message — not just the bar glyphs — when we're in ASCII mode.
        message = progress.message
        if not self.unicode:
            message = (
                message.replace("→", "->")
                .encode("ascii", "replace")
                .decode("ascii")
            )
        line = f"  [{bar}] {message[:52]:<52} {elapsed}"

        sys.stdout.write("\r" + line[:80])
        sys.stdout.flush()

        if progress.progress >= 1.0 and phase == "done":
            print()  # newline after final progress


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def cmd_build(args):
    """Build an optimized deck."""
    print(f"Building deck for: {args.commander}")
    print("=" * 60)

    # v0.9.15: deprecated --power-level warning
    if getattr(args, "power_level", None) is not None and \
            getattr(args, "bracket", None) is None:
        from .bracket import power_level_to_bracket, bracket_name
        b = power_level_to_bracket(args.power_level)
        print(f"NOTE: --power-level is deprecated; {args.power_level} maps to "
              f"--bracket {b} ({bracket_name(b)})")

    # v0.4: parse refinement and weight flags
    role_overrides: dict[str, tuple] = {}
    for s in getattr(args, "role_target", []) or []:
        role, target = _parse_role_target(s)
        role_overrides[role] = target

    preset = getattr(args, "preset", None)
    weight_overrides = getattr(args, "weight", []) or []
    score_weights = _build_weight_dict(preset, weight_overrides)

    config_kwargs = dict(
        commander_name=args.commander,
        generations=args.generations,
        population_size=args.population,
        random_seed=args.seed,
        bracket=getattr(args, "bracket", None),
        power_level=args.power_level,  # DEPRECATED fallback, see BuildConfig
        game_changers_file=getattr(args, "game_changers", None),
        llm_model=args.model,
        enable_llm_review=args.review,
        generate_html_report=bool(args.report),
        # Session 3 toggles
        use_edhrec=getattr(args, "edhrec", False),
        # Cache dir must default whenever EITHER flag triggers the fetch
        # (see the `recall_use_edhrec` comment below) — keying only off
        # --edhrec left --recall-edhrec-only runs fetching over the network
        # with no disk cache, so every rerun re-fetched from EDHREC instead
        # of reusing yesterday's data.
        edhrec_cache_dir=getattr(args, "edhrec_cache_dir", None) or (
            "./edhrec_cache"
            if getattr(args, "edhrec", False) or getattr(args, "recall_edhrec", False)
            else None
        ),
        edhrec_offline=getattr(args, "edhrec_offline", False),
        use_embeddings=getattr(args, "embeddings", False),
        embedding_model=getattr(args, "embedding_model", "all-MiniLM-L6-v2"),
        # v0.8: Layered candidate recall for the synergy pool. Independent
        # flags so the user can A/B each source. recall_use_edhrec implies
        # the EDHREC fetch even if --edhrec wasn't passed.
        recall_use_edhrec=getattr(args, "recall_edhrec", False),
        recall_edhrec_limit=getattr(args, "recall_edhrec_limit", 300),
        recall_use_embeddings=getattr(args, "recall_embeddings", False),
        recall_embedding_limit=getattr(args, "recall_embedding_limit", 1500),
        recall_embedding_cache_dir=getattr(args, "recall_embedding_cache_dir", None) or (
            "./embedding_cache"
            if getattr(args, "recall_embeddings", False)
            else None
        ),
        recall_use_patterns=getattr(args, "recall_patterns", False),
        recall_pool_cap=getattr(args, "recall_pool_cap", 2500),
        # v0.9.4: runtime levers
        synergy_scoring_mode=getattr(args, "synergy_scoring_mode", "auto"),
        synergy_engine_target=getattr(args, "synergy_engine_target", 25),
        synergy_engine_shortlist=getattr(args, "synergy_engine_shortlist", 300),
        synergy_engine_bypass=getattr(args, "synergy_engine_bypass", 12),
        # v0.9.7: LLM intrinsic card-power scoring
        card_power_mode=getattr(args, "card_power_mode", "off"),
        card_power_model=getattr(args, "card_power_model", "claude-sonnet-4-6"),
        card_power_cache_dir=getattr(args, "card_power_cache_dir",
                                     "./card_power_cache"),
        card_power_recall_weight=getattr(args, "card_power_recall_weight", 0.15),
        card_power_recall_cap=getattr(args, "card_power_recall_cap", 0),
        role_power_bypass=getattr(args, "role_power_bypass", 15),
        power_staples_limit=getattr(args, "power_staples_limit", 60),
        # v0.9.8: combo detection + interaction fitness
        combo_mode=getattr(args, "combo_mode", "off"),
        combo_model=getattr(args, "combo_model", "claude-sonnet-4-6"),
        combo_cache_dir=getattr(args, "combo_cache_dir", "./combo_cache"),
        combo_signature_pass=not getattr(args, "no_signature_pass", False),
        synergy_cache_dir=(None if getattr(args, "no_synergy_cache", False)
                           else "./synergy_cache"),
        combo_weight=getattr(args, "combo_weight", 0.12),
        combo_max_pool=getattr(args, "combo_max_pool", 350),
        engine_boost_mode=getattr(args, "engine_boost_mode", "power"),
        engine_boost_floor=getattr(args, "engine_boost_floor", 80.0),
        edhrec_floor=getattr(args, "edhrec_floor", 0.75),
        structural_synergy_mode=getattr(args, "structural_synergy_mode", "on"),
        structural_boost_floor=getattr(args, "structural_boost_floor", 95.0),
        budget_max_per_card=getattr(args, "budget", None),
        budget_exclude_unknown=getattr(args, "budget_exclude_unknown", False),
        use_island_model=bool(getattr(args, "islands", None)),
        num_islands=getattr(args, "islands", None) or 4,
        island_migration_interval=getattr(args, "island_migration_interval", 10),
        # Session 4: refinement
        locked_cards=list(getattr(args, "lock", []) or []),
        banned_cards=list(getattr(args, "ban", []) or []),
        role_target_overrides=role_overrides,
        role_shortfall_penalty=getattr(args, "role_shortfall_penalty", 2.0),
        # v0.9.14: set-level fitness + refinement
        quality_weighted_roles=not getattr(args, "no_quality_roles", False),
        consistency_weight=getattr(args, "consistency_weight", 0.12),
        refine_iterations=getattr(args, "refine", 3),
        refine_max_swaps=getattr(args, "refine_max_swaps", 8),
        warm_start_path=getattr(args, "warm_start", None),
        warm_start_copies=getattr(args, "warm_start_copies", 1),
        # Session 4: images
        use_images=getattr(args, "images", False),
        images_cache_dir=getattr(args, "images_cache_dir", None),
        images_offline=getattr(args, "images_offline", False),
        # Session 5: flavor tags
        flavor_art_tags=list(getattr(args, "flavor_tag", []) or []),
        tags_cache_dir=getattr(args, "tags_cache_dir", None),
        tags_offline=getattr(args, "tags_offline", False),
        # Session 6: bulk data + role validation
        use_bulk_source=getattr(args, "bulk_source", False),
        bulk_cache_dir=getattr(args, "bulk_cache_dir", None),
        bulk_type=getattr(args, "bulk_type", "oracle_cards"),
        bulk_offline=getattr(args, "bulk_offline", False),
        validate_roles_after_build=getattr(args, "validate_roles", False),
    )
    if score_weights is not None:
        config_kwargs["score_weights"] = score_weights

    config = BuildConfig(**config_kwargs)

    # Show the user what we're doing with their overrides
    if config.locked_cards:
        print(f"Locked cards: {', '.join(config.locked_cards)}")
    if config.banned_cards:
        print(f"Banned cards: {', '.join(config.banned_cards)}")
    if config.role_target_overrides:
        print(f"Role overrides: {config.role_target_overrides}")
    if score_weights:
        wp = preset if preset else "custom"
        w_display = ", ".join(f"{k}={v:.2f}" for k, v in sorted(config.score_weights.items()))
        print(f"Weights ({wp}): {w_display}")
    if config.warm_start_path:
        print(f"Warm-start: {config.warm_start_path} (×{config.warm_start_copies})")
    if config.flavor_art_tags:
        print(f"Flavor art tags: {', '.join(config.flavor_art_tags)}")

    llm_config = LLMConfig(
        model=args.model,
        temperature=0.3,
        mock_mode=args.mock,
        # v0.9.4: cheaper model for tournament elimination rounds.
        tournament_model=getattr(args, "tournament_model", "claude-haiku-4-5"),
        # v0.9.32: persistent commander-analysis cache.
        analysis_cache_dir=(None
                            if getattr(args, "no_analysis_cache", False)
                            else "./analysis_cache"),
    )

    progress_cb = ProgressRenderer(quiet=args.quiet)

    builder = DeckBuilder(
        card_database_path=args.csv,
        config=config,
        llm_config=llm_config,
        progress_callback=progress_cb,
    )

    start = time.time()
    try:
        result = builder.build()
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"  Final Score:  {result.final_score:.2f}")
    print(f"  Generations:  {result.generations_run}")
    print(f"  Runtime:      {elapsed:.1f}s")
    print(f"  Valid:        {result.best_deck.is_valid}")

    if result.best_deck.scores:
        s = result.best_deck.scores
        print("\n  Score Breakdown:")
        print(f"    Mana Curve:        {s.mana_curve:.1f}")
        print(f"    Role Coverage:     {s.role_coverage:.1f}")
        print(f"    Synergy:           {s.synergy:.1f}")
        print(f"    Strategy Density:  {s.strategy_density:.1f}")
        print(f"    Power Level:       {s.power_level:.1f}")
        if s.consistency > 0:
            print(f"    Consistency:       {s.consistency:.1f}")
        print(f"    Creativity:        {s.creativity:.1f}")
        if s.flavor > 0:
            print(f"    Flavor:            {s.flavor:.1f}")
        print(f"    Effective Synergy: {s.effective_synergy:.1f}")

        if s.role_counts:
            print("\n  Role Counts:")
            for role, count in sorted(s.role_counts.items()):
                print(f"    {role:15s}: {count}")

    # Output decklist
    decklist = result.best_deck.to_decklist()
    if args.output:
        Path(args.output).write_text(decklist, encoding="utf-8")
        print(f"\nDecklist written to {args.output}")
    else:
        print("\n" + "-" * 60)
        print("DECKLIST")
        print("-" * 60)
        print(decklist)

    # HTML report
    if args.report:
        # v0.4: pass card_source for image-embedded reports if --images was set
        report_path = generate_html_report(
            result, args.report,
            card_source=builder.card_source,  # None if --images wasn't set
        )
        print(f"\nHTML report: {report_path}")

    # Export formats
    if args.moxfield:
        mf = _export_moxfield(result.best_deck)
        Path(args.moxfield).write_text(mf, encoding="utf-8")
        print(f"Moxfield export: {args.moxfield}")

    if args.archidekt:
        ar = _export_archidekt(result.best_deck)
        Path(args.archidekt).write_text(ar, encoding="utf-8")
        print(f"Archidekt export: {args.archidekt}")

    # v0.4: save the deck as a warm-start snapshot for future iteration
    save_path = getattr(args, "save_deck", None)
    if save_path:
        result.to_json_file(save_path)
        print(f"Warm-start snapshot saved: {save_path}")
        print(f"  (use --warm-start {save_path} on a future run to iterate)")

    # v0.9.15: bracket compliance audit
    audit = getattr(result, "bracket_audit", None)
    if audit is not None:
        from .bracket import bracket_name
        print("\n" + "-" * 60)
        print(f"BRACKET COMPLIANCE (target: {audit.bracket} — "
              f"{bracket_name(audit.bracket)})")
        print("-" * 60)
        gc_str = ", ".join(audit.game_changers) or "(none)"
        limit_str = ("unlimited" if audit.gc_limit is None
                     else str(audit.gc_limit))
        print(f"  Game Changers ({len(audit.game_changers)}/{limit_str}): {gc_str}")
        if audit.mld_cards:
            print(f"  Mass land denial: {', '.join(audit.mld_cards)}")
        if audit.extra_turn_cards:
            print(f"  Extra-turn cards: {', '.join(audit.extra_turn_cards)}")
        if audit.two_card_combos:
            print("  Two-card combos present:")
            for c in audit.two_card_combos:
                print(f"    - {c['desc']}" + ("  [early]" if c["early"] else ""))
        if audit.compliant:
            print("  Verdict: COMPLIANT with target bracket")
        else:
            print("  Verdict: VIOLATIONS —")
            for v in audit.violations:
                print(f"    ! {v}")
        eff = audit.effective_bracket
        suffix = (" (brackets 4 and 5 are identical by contents; 5 is a "
                  "metagame declaration)" if eff == 4 else "")
        print(f"  Effective bracket (by contents): {eff}{suffix}")

    # v0.9.14: refinement swaps applied by the post-GA LLM pass
    if getattr(result, "refinement_log", None):
        print("\n" + "-" * 60)
        print(f"LLM REFINEMENT ({len(result.refinement_log)} swaps)")
        print("-" * 60)
        for s in result.refinement_log:
            print(f"  - {s['out']}  ->  {s['in']}   ({s.get('reason', '')})")

    # LLM review
    if result.llm_review:
        print("\n" + "-" * 60)
        print("LLM REVIEW")
        print("-" * 60)
        print(result.llm_review)

    # v0.6: Role validation report (diagnostic)
    if result.role_validation_report is not None:
        from .oracle_validation import format_role_report
        print("\n" + "-" * 60)
        print("ROLE VALIDATION (regex vs. community oracle tags)")
        print("-" * 60)
        print(format_role_report(result.role_validation_report))

    return 0


def cmd_quick(args):
    """Quick build without optimization (for testing)."""
    print(f"Quick build: {args.commander}")
    print("=" * 60)

    config = BuildConfig(commander_name=args.commander)
    builder = DeckBuilder(args.csv, config, llm_config=LLMConfig(mock_mode=True))

    try:
        deck = builder.quick_build()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\nCards: {len(deck.cards)}")
    print(f"Valid: {deck.is_valid}")

    if not deck.is_valid:
        valid, reasons = deck.validate()
        for r in reasons:
            print(f"  - {r}")

    if args.output:
        Path(args.output).write_text(deck.to_decklist(), encoding="utf-8")
        print(f"\nDecklist written to {args.output}")
    else:
        print("\n" + deck.to_decklist())

    return 0


def cmd_analyze(args):
    """Analyze a commander without building a deck."""
    print(f"Analyzing: {args.commander}")
    print("=" * 60)

    config = BuildConfig(commander_name=args.commander)
    llm_config = LLMConfig(mock_mode=args.mock)
    builder = DeckBuilder(args.csv, config, llm_config=llm_config)

    try:
        commander = builder.db.get_by_name(args.commander)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not commander:
        similar = builder.db.find_similar_names(args.commander)
        print(f"Commander not found: {args.commander}", file=sys.stderr)
        if similar:
            print(f"Did you mean: {', '.join(similar)}", file=sys.stderr)
        return 1

    print(f"\n{commander.name} {commander.mana_cost}")
    print(f"{commander.card_type}")
    if commander.power and commander.toughness:
        print(f"{commander.power}/{commander.toughness}")
    if commander.text:
        print(f"\n{commander.text}")

    print("\n" + "-" * 60)
    print("AI Analysis")
    print("-" * 60)

    analysis = builder.llm.analyze_commander(commander)
    print(f"\nKey Mechanics: {', '.join(analysis.key_mechanics or []) or '(none)'}")
    print(f"\nStrategy:\n  {analysis.build_around_text}")
    print(f"\nEvaluation Notes:\n  {analysis.evaluation_notes}")
    print(f"\nSynergy Keywords: {', '.join(analysis.synergy_keywords or []) or '(none)'}")
    if analysis.anti_synergy_keywords:
        print(f"Avoid: {', '.join(analysis.anti_synergy_keywords)}")

    if analysis.recommended_weights:
        print("\nRecommended Scoring Weights (commander-adaptive):")
        for k, v in analysis.recommended_weights.items():
            print(f"  {k:15s}: {v:.2f}")

    if analysis.recommended_synergy_weight is not None:
        print(f"\nRecommended Synergy Balance: {analysis.recommended_synergy_weight:.2f}")
        print(f"  (baseline weight: {1 - analysis.recommended_synergy_weight:.2f})")

    return 0


def cmd_search(args):
    """Search the card database."""
    db = CardDatabase(args.csv)
    db.load()

    print(f"Searching for: {args.query}")
    print("=" * 60)

    result = db.search_text([args.query], color_identity=args.colors)
    print(f"\nFound {result.total_matches} matching cards")
    print("-" * 60)

    for card in result.cards[:args.limit]:
        print(f"\n{card.name} {card.mana_cost}")
        print(f"  {card.card_type}")
        if card.text:
            text = card.text[:100] + "..." if len(card.text) > 100 else card.text
            print(f"  {text}")

    return 0


def cmd_stats(args):
    """Database statistics."""
    db = CardDatabase(args.csv)
    db.load()

    print(f"Database Statistics: {args.csv}")
    print("=" * 60)
    print(f"Total cards: {db.card_count}")

    print("\nBy Type:")
    for t in ["Creature", "Instant", "Sorcery", "Artifact",
              "Enchantment", "Planeswalker", "Land"]:
        count = len(db.query(card_types=[t]).cards)
        print(f"  {t:15s}: {count}")

    print("\nBy Color Identity:")
    for c in ["W", "U", "B", "R", "G", ""]:
        label = c if c else "(colorless)"
        result = db.query(color_identity=c)
        print(f"  {label:15s}: {result.total_matches}")

    return 0


def cmd_gui(args):
    """v0.9.20: local web GUI (127.0.0.1 only). Simple mode with the usual
    defaults, advanced panel for every knob, plus refresh-cards and
    power-scan buttons."""
    from .gui import serve
    serve(csv_path=args.csv, port=getattr(args, "port", 8765),
          open_browser=not getattr(args, "no_browser", False))
    return 0


def cmd_refresh_cards(args):
    """v0.9.18: refresh the card CSV from MTGJSON AtomicCards (Python port of
    the extraction pipeline). Downloads (or reads a local) AtomicCards.json,
    extracts the pipe-delimited CSV including the isGameChanger column, and
    writes it — by default to the --csv path (in place)."""
    from . import mtgjson_refresh

    output = getattr(args, "output", None) or args.csv
    source = getattr(args, "source", None) or mtgjson_refresh.DEFAULT_SOURCE_URL
    atomic = getattr(args, "atomic_json", None)

    out_path = Path(output)
    if out_path.exists() and not getattr(args, "force", False):
        # Safety: don't silently clobber the live database.
        print(f"Refusing to overwrite existing {output} without --force.",
              file=sys.stderr)
        print(f"  (writes a fresh card DB from MTGJSON; back up {output} first "
              f"if you want to keep it, then re-run with --force)",
              file=sys.stderr)
        return 2

    print(f"Refreshing card database -> {output}")
    if atomic:
        print(f"  Source: local file {atomic}")
    else:
        print(f"  Source: {source}")
    try:
        count = mtgjson_refresh.refresh(
            output_path=output,
            source_url=source,
            atomic_json_path=atomic,
        )
    except Exception as e:
        print(f"\nRefresh failed: {e}", file=sys.stderr)
        return 1
    print(f"Done. Wrote {count} cards to {output} (with isGameChanger column).")
    print("  The Game Changer list is now sourced from this CSV.")
    return 0


def cmd_power_scan(args):
    """v0.9.16: deliberately bulk-score a color region into the GLOBAL
    card-power cache (feeds the power-staples channel and the role bypass).
    Cost-transparent: prints the uncached count and estimated calls before
    scoring; --dry-run stops there."""
    from .card_power_scorer import CardPowerScorer

    db = CardDatabase(args.csv)
    db.load()

    colors = (args.colors or "").upper()
    color_set = set(ch for ch in colors if ch in "WUBRG")
    if colors and not color_set and colors != "C":
        print(f"Unrecognized --colors {args.colors!r}; use e.g. GU or WUBRG "
              f"or C for colorless-only", file=sys.stderr)
        return 2
    if colors:
        cards = [
            c for c in db.all_cards
            if set(ch for ch in (c.color_identity or "") if ch in "WUBRG")
            <= color_set
        ]
        scope = f"color identity <= {colors}"
    else:
        cards = db.all_cards
        scope = "entire database"

    llm_config = LLMConfig(model=args.model)
    from .llm_engine import LLMEngine
    engine = LLMEngine(llm_config)
    if engine.config.mock_mode:
        print("No API key available — power-scan needs a real model.",
              file=sys.stderr)
        return 1
    scorer = CardPowerScorer(engine, model=args.model,
                             cache_dir=args.cache_dir,
                             batch_size=args.batch_size)
    cached = scorer.cached_scores()
    todo = [c for c in cards if c.name not in cached]
    batches = (len(todo) + args.batch_size - 1) // args.batch_size
    print(f"Scope: {scope} — {len(cards)} cards; already cached: "
          f"{len(cards) - len(todo)}; to score: {len(todo)} "
          f"(~{batches} calls to {args.model})")
    if args.dry_run:
        print("(dry run — nothing scored)")
        return 0
    if not todo:
        print("Cache already covers this scope.")
        return 0
    scored = scorer.score_cards(cards)
    print(f"Done. Global cache now holds {len(scorer.cached_scores())} "
          f"scores; this scope covered: {sum(1 for c in cards if c.name in scored)}"
          f"/{len(cards)}")
    return 0


def cmd_diff(args):
    """v0.5: Compare two warm-start deck JSON files."""
    from .models import WarmStartDeck
    from .deck_diff import diff_decks, format_diff

    try:
        from_deck = WarmStartDeck.from_json_file(args.from_path)
    except (OSError, ValueError) as e:
        print(f"Error reading {args.from_path}: {e}", file=sys.stderr)
        return 2

    try:
        to_deck = WarmStartDeck.from_json_file(args.to_path)
    except (OSError, ValueError) as e:
        print(f"Error reading {args.to_path}: {e}", file=sys.stderr)
        return 2

    # If --csv was provided, use it for role grouping
    card_db = None
    csv_path = getattr(args, "csv", None)
    if csv_path:
        try:
            card_db = CardDatabase(csv_path)
            card_db.load()
        except Exception as e:
            # Don't fail the diff just because the DB load failed
            print(f"(warning: couldn't load card DB for role grouping: {e})",
                  file=sys.stderr)
            card_db = None

    result = diff_decks(from_deck, to_deck, card_db=card_db)
    print(format_diff(
        result,
        show_kept=args.show_kept,
        max_per_group=args.max_per_group,
    ))
    return 0


# ----------------------------------------------------------------------
# Export formats
# ----------------------------------------------------------------------

def _export_moxfield(deck) -> str:
    lines = [f"1 {deck.commander.name} *CMDR*"]
    # Count copies (for basics)
    from collections import Counter
    counts = Counter(c.name for c in deck.cards)
    for name, count in sorted(counts.items()):
        lines.append(f"{count} {name}")
    return "\n".join(lines)


def _export_archidekt(deck) -> str:
    lines = [f"1x {deck.commander.name} [Commander]"]
    from collections import Counter
    counts = Counter(c.name for c in deck.cards)
    for name, count in sorted(counts.items()):
        lines.append(f"{count}x {name}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mtg-deck-builder",
        description="MTG EDH Deck Builder — hybrid AI + genetic algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment:\n"
            "  ANTHROPIC_API_KEY  Your API key (or use --mock for testing)\n"
            "  MTG_ASCII          Set to 1 to force ASCII output (for terminals\n"
            "                     without UTF-8 support)\n"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging on the console")
    parser.add_argument("--log-file",
                        help="Write a full DEBUG-level log to this file. "
                             "Captures complete LLM responses (including the "
                             "full text of any unparseable JSON) for forensics. "
                             "Console output is unaffected by this flag.")
    parser.add_argument("--csv", required=True,
                        help="Path to the MTG card CSV database")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # build
    build_p = subparsers.add_parser("build", help="Build an optimized deck")
    build_p.add_argument("commander", help="Commander name (case-insensitive)")
    build_p.add_argument("-g", "--generations", type=int, default=100)
    build_p.add_argument("-p", "--population", type=int, default=50)
    build_p.add_argument("-s", "--seed", type=int, default=None)
    build_p.add_argument("--bracket", type=int, default=None, metavar="N",
                         choices=[1, 2, 3, 4, 5],
                         help="Official Commander bracket 1-5 (default: 4). "
                              "1 Exhibition / 2 Core / 3 Upgraded / "
                              "4 Optimized / 5 cEDH. Brackets 1-3 enforce "
                              "the official rules (Game Changer limits, no "
                              "mass land denial, two-card-combo policy); "
                              "bracket 5 switches to cEDH structural "
                              "templates (curve, lands, interaction).")
    build_p.add_argument("--power-level", type=int, default=None, metavar="N",
                         help="DEPRECATED: old 1-10 scale; maps onto "
                              "--bracket (1-2>B1, 3-4>B2, 5-7>B3, 8-9>B4, "
                              "10>B5). Use --bracket instead.")
    build_p.add_argument("--game-changers", metavar="FILE",
                         help="Refresh the Game Changer list from an external "
                              "file (JSON array, {\"names\":[...]}, "
                              "MTGJSON-atomic isGameChanger, or newline text), "
                              "overriding the embedded list. If omitted, an "
                              "isGameChanger column in the CSV is used when "
                              "present.")
    build_p.add_argument("--model", default="claude-sonnet-4-6",
                         help="Claude model to use")
    build_p.add_argument("--mock", action="store_true",
                         help="Use mock LLM (no API calls)")
    build_p.add_argument("--review", action="store_true",
                         help="Run an LLM review pass on the final deck")
    build_p.add_argument("-o", "--output", help="Write decklist to file")
    build_p.add_argument("--report", help="Write HTML report to file")
    build_p.add_argument("--moxfield", help="Export in Moxfield format")
    build_p.add_argument("--archidekt", help="Export in Archidekt format")
    build_p.add_argument("-q", "--quiet", action="store_true",
                         help="Suppress progress output")

    # Session 3: Optional integrations
    build_p.add_argument("--edhrec", action="store_true",
                         help="Use EDHREC data for synergy/baseline scoring")
    build_p.add_argument("--edhrec-cache-dir", metavar="DIR",
                         help="Directory for EDHREC JSON cache (default: ./edhrec_cache)")
    build_p.add_argument("--edhrec-offline", action="store_true",
                         help="Don't hit the network; use cache only")
    build_p.add_argument("--embeddings", action="store_true",
                         help="Use sentence-transformers for fast synergy scoring "
                         "(requires `pip install sentence-transformers`)")
    build_p.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                         help="sentence-transformers model name")

    # v0.8: Layered candidate recall (each source independent and unioned)
    build_p.add_argument("--recall-edhrec", action="store_true",
                         help="Source synergy candidates from EDHREC top-N "
                              "(community-vetted; auto-fetches data)")
    build_p.add_argument("--recall-edhrec-limit", type=int, default=300,
                         metavar="N",
                         help="EDHREC top-N candidate cap (default: 300)")
    build_p.add_argument("--recall-embeddings", action="store_true",
                         help="Source synergy candidates by embedding "
                              "similarity to commander strategy (requires "
                              "sentence-transformers)")
    build_p.add_argument("--recall-embedding-limit", type=int, default=1500,
                         metavar="N",
                         help="Embedding-recall top-N cap (default: 1500)")
    build_p.add_argument("--recall-embedding-cache-dir", metavar="DIR",
                         help="Directory for cached card embeddings "
                              "(default: ./embedding_cache)")
    build_p.add_argument("--recall-patterns", action="store_true",
                         help="Source synergy candidates via LLM-expanded "
                              "substring patterns (with digit/X normalization) "
                              "— catches 'gain 1 life' for 'gain life'")
    build_p.add_argument("--recall-pool-cap", type=int, default=2500,
                         metavar="N",
                         help="Final cap on the unioned synergy pool "
                              "(default: 2500)")

    # v0.9.4: runtime levers
    build_p.add_argument("--synergy-scoring-mode",
                         choices=["auto", "llm", "embedding"], default="auto",
                         help="How to score per-card synergy. 'auto' (default): "
                              "embedding cosine + hint boost when available "
                              "(fast, ~0 LLM scoring calls). 'llm': force the "
                              "calibrated rubric (max quality, ~30 calls). "
                              "'embedding': cosine+hint only, never LLM.")
    build_p.add_argument("--tournament-model", metavar="MODEL",
                         default="claude-haiku-4-5",
                         help="Model for the card-selection tournament's "
                              "elimination rounds (coarse filtering). The final "
                              "precision pick always uses --model. Pass the same "
                              "value as --model to disable the split. "
                              "(default: claude-haiku-4-5)")
    build_p.add_argument("--synergy-engine-target", type=int, default=25,
                         metavar="N",
                         help="How many strategy-defining cards the Phase 2 "
                              "synergy_engine pass should select (0 disables it). "
                              "(default: 25)")
    build_p.add_argument("--synergy-engine-shortlist", type=int, default=300,
                         metavar="N",
                         help="Phase 2 pre-ranks the recall pool by (hint tier, "
                              "cosine-to-commander) and only runs the LLM over "
                              "the top N — instead of the whole ~2500-card pool. "
                              "(default: 300)")
    build_p.add_argument("--synergy-engine-bypass", type=int, default=12,
                         metavar="N",
                         help="The top N pre-ranked Phase 2 cards skip the LLM "
                              "and go straight into the GA pool, so the "
                              "commander's best payoffs can't be eliminated. "
                              "0 disables the bypass. (default: 12)")

    # v0.9.7: LLM intrinsic card-power scoring.
    build_p.add_argument("--card-power-mode", choices=["off", "llm"],
                         default="off",
                         help="LLM intrinsic card-power scoring (commander-"
                              "independent, globally cached). 'llm' enables a "
                              "real 'is this card good?' signal feeding the "
                              "Power Level dimension + synergy_engine recall. "
                              "(default: off)")
    build_p.add_argument("--card-power-model", default="claude-sonnet-4-6",
                         metavar="MODEL",
                         help="Model for card-power scoring. Sonnet recommended; "
                              "Haiku compresses the mid-band. (default: "
                              "claude-sonnet-4-6)")
    build_p.add_argument("--card-power-cache-dir", default="./card_power_cache",
                         metavar="DIR",
                         help="Disk cache for card-power scores (keyed by model "
                              "+ card text). (default: ./card_power_cache)")
    build_p.add_argument("--card-power-recall-weight", type=float, default=0.15,
                         metavar="W",
                         help="Synergy-led pre-rank blend: composite = cosine + "
                              "W*(power/100). Small keeps commander fit dominant. "
                              "(default: 0.15)")
    build_p.add_argument("--card-power-recall-cap", type=int, default=0,
                         metavar="N",
                         help="Cap how many synergy-pool cards get power-scored "
                              "for recall (top-N by cosine). 0 = score all (best "
                              "quality, higher first-build cost). (default: 0)")
    build_p.add_argument("--power-staples", dest="power_staples_limit",
                         type=int, default=40, metavar="N",
                         help="Global power-staples channel: top-N color-legal "
                              "cards by GLOBAL cached card power join the pool "
                              "regardless of role/theme (the general fix for "
                              "taxonomy holes: stax, theft, cost reducers...). "
                              "Grow the cache with the power-scan command. "
                              "0 disables. (default: 40)")
    build_p.add_argument("--role-power-bypass", type=int, default=15,
                         metavar="N",
                         help="Top-N cards by cached card power in each role "
                              "bucket join the GA pool additively — the "
                              "selection tournament can't cut them (fixes "
                              "funnel-cut staples like Llanowar Elves). "
                              "Needs --card-power-mode llm. 0 disables. "
                              "(default: 15)")

    # v0.9.8: LLM combo/engine detection + interaction-aware fitness.
    build_p.add_argument("--combos", dest="combo_mode",
                         choices=["off", "llm"], default="off",
                         help="LLM combo/engine detection (pool + knowledge "
                              "passes, cached per-commander). Enables the "
                              "enabler on-ramp + interaction-aware GA fitness "
                              "that rewards assembling multi-card combos. "
                              "(default: off)")
    build_p.add_argument("--combo-model", default="claude-sonnet-4-6",
                         metavar="MODEL",
                         help="Model for combo detection. (default: "
                              "claude-sonnet-4-6)")
    build_p.add_argument("--combo-cache-dir", default="./combo_cache",
                         metavar="DIR",
                         help="Disk cache for detected combos (per commander). "
                              "(default: ./combo_cache)")
    build_p.add_argument("--combo-weight", type=float, default=0.12,
                         metavar="W",
                         help="Weight of the 0-100 combo dimension in the total "
                              "(additive bonus, scaled by --power-level). "
                              "(default: 0.12)")
    build_p.add_argument("--no-analysis-cache", action="store_true",
                         help="Disable the persistent commander-analysis "
                              "cache (v0.9.32). By default a commander's "
                              "analysis is reused across runs — the keystone "
                              "of run-to-run stability; disable to force a "
                              "fresh analysis (e.g. after a model upgrade).")
    build_p.add_argument("--no-synergy-cache", action="store_true",
                         help="Disable the per-commander synergy-score cache "
                              "(v0.9.31). By default repeat builds reuse "
                              "prior LLM synergy scores — near-deterministic "
                              "and ~30-40 fewer calls; disable to force a "
                              "full fresh scoring pass.")
    build_p.add_argument("--no-signature-pass", action="store_true",
                         help="Disable the recall-only signature-combo pass "
                              "(names the commander's famous combos so niche "
                              "pieces like Mirror Universe enter the pool; "
                              "recall only, never reweights). Needs --combos.")
    build_p.add_argument("--combo-max-pool", type=int, default=350,
                         metavar="N",
                         help="How many power-ranked synergy cards the combo "
                              "pool pass analyzes. (default: 350)")
    build_p.add_argument("--engine-boost", dest="engine_boost_mode",
                         choices=["off", "floor", "power"], default="power",
                         help="Lift LLM-detected engines' synergy so they "
                              "compete for slots (needs --combos llm). "
                              "power: floor at the card's own power score "
                              "(quality-scaled, recommended); floor: flat "
                              "floor. (default: power)")
    build_p.add_argument("--engine-floor", dest="engine_boost_floor",
                         type=float, default=80.0, metavar="N",
                         help="Synergy floor for --engine-boost floor. "
                              "(default: 80)")
    build_p.add_argument("--edhrec-floor", dest="edhrec_floor",
                         type=float, default=0.75, metavar="F",
                         help="Floor a card's synergy to F * its EDHREC "
                              "distinctive-synergy (boost-only). Surfaces the "
                              "commander's community staple package without "
                              "penalizing pricey/unpopular cards or over-"
                              "boosting generic staples; no-op without EDHREC "
                              "data. 0 disables. (default: 0.75)")
    build_p.add_argument("--structural-synergy", dest="structural_synergy_mode",
                         choices=["on", "off"], default="on",
                         help="Attribute-based synergy for commanders whose "
                              "payoff is a card property the text signals can't "
                              "see (vanilla matters, colorless, low-curve...). "
                              "Pulls matching cards into recall + floors their "
                              "synergy. No-op without structural predicates. "
                              "(default: on)")
    build_p.add_argument("--structural-floor", dest="structural_boost_floor",
                         type=float, default=95.0, metavar="N",
                         help="Synergy floor for structural-predicate matches. "
                              "High by design: attribute payoffs have ~0 power, "
                              "so the floor must clear the text-synergy ceiling "
                              "(~85-90) for them to win slots. (default: 95)")

    build_p.add_argument("--budget", type=float, metavar="USD",
                         help="Maximum USD price per card (uses Scryfall API)")
    build_p.add_argument("--budget-exclude-unknown", action="store_true",
                         help="Drop cards without price data (instead of keeping them)")
    build_p.add_argument("--islands", type=int, metavar="N",
                         help="Use island-model parallel GA with N islands")
    build_p.add_argument("--island-migration-interval", type=int, default=10,
                         help="Generations between island migrations")

    # Session 4: Iterative refinement
    build_p.add_argument("--lock", action="append", default=[], metavar="CARD",
                         help="Card name to lock into every deck (repeat for multiple, "
                              "e.g. --lock 'Sol Ring' --lock 'Soul Warden')")
    build_p.add_argument("--ban", action="append", default=[], metavar="CARD",
                         help="Card name to exclude from the pool (repeat for multiple)")
    build_p.add_argument("--role-target", action="append", default=[],
                         metavar="ROLE=MIN,MAX",
                         help="Override a role target. Ex: --role-target removal=10,14 "
                              "(repeat for multiple roles)")
    build_p.add_argument("--role-shortfall-penalty", type=float, default=2.0,
                         metavar="P",
                         help="Fitness penalty per card BELOW a role's minimum "
                              "target. Stops the GA starving ramp/removal in "
                              "favor of on-theme cards. 0 disables. "
                              "(default: 2.0)")
    # v0.9.14: set-level fitness + LLM refinement
    build_p.add_argument("--refine", type=int, default=3, metavar="N",
                         help="Post-GA LLM refinement rounds: the assembled "
                              "deck + best unused alternatives go to the LLM "
                              "for holistic swaps (redundancy, interaction "
                              "spread, role quality). 0 disables. (default: 2)")
    build_p.add_argument("--refine-max-swaps", type=int, default=8, metavar="N",
                         help="Max swaps per refinement round. (default: 8)")
    build_p.add_argument("--consistency-weight", type=float, default=0.12,
                         metavar="W",
                         help="Weight of the consistency/redundancy dimension "
                              "(additive, like combo). Active only when the "
                              "commander analysis emits core effect classes "
                              "and synergy scoring runs on the LLM. 0 "
                              "disables. (default: 0.12)")
    build_p.add_argument("--no-quality-roles", action="store_true",
                         help="Disable quality-weighted role coverage (v0.9.14: "
                              "role fillers count toward targets weighted by "
                              "card power, so weak fillers need backup).")
    build_p.add_argument("--warm-start", metavar="JSON_FILE",
                         help="Seed the GA with a previous deck (produced by "
                              "--save-deck on an earlier run)")
    build_p.add_argument("--warm-start-copies", type=int, default=1,
                         help="Number of warm-start copies to seed population with "
                              "(higher = stays closer to original)")
    build_p.add_argument("--save-deck", metavar="JSON_FILE",
                         help="Save the resulting deck as a warm-start snapshot "
                              "(for future iterative runs)")

    # Session 4: Tunable weights
    build_p.add_argument("--weight", action="append", default=[], metavar="DIM=VAL",
                         help="Override a scoring weight. Ex: --weight synergy=0.6 "
                              "--weight power_level=0.3 (weights auto-renormalize to 1.0). "
                              "Note: 'creativity' is informational only and not scored.")
    build_p.add_argument("--preset", choices=["flavor", "power", "budget", "balanced"],
                         help="Apply a weight preset (sets --weight values). "
                              "flavor: theme-focused; power: cEDH-leaning; "
                              "budget: value-per-card (synergy + density); "
                              "balanced: defaults")

    # Session 4: Scryfall images for HTML report
    build_p.add_argument("--images", action="store_true",
                         help="Fetch card art/images from Scryfall and embed them "
                              "in the HTML report (requires internet unless cached). "
                              "Only affects --report output.")
    build_p.add_argument("--images-cache-dir", metavar="DIR",
                         help="Directory for Scryfall image JSON cache "
                              "(default: ./scryfall_cache)")
    build_p.add_argument("--images-offline", action="store_true",
                         help="Only use cached image data; don't hit Scryfall")

    # Session 5: Scryfall Tagger integration
    build_p.add_argument("--flavor-tag", action="append", default=[], metavar="TAG",
                         help="Art tag for flavor scoring, e.g. --flavor-tag mammoth "
                              "--flavor-tag forest. Cards whose artwork matches any "
                              "tag boost the flavor score (uses Scryfall Tagger data). "
                              "Combines with tribal-subtype flavor (max wins).")
    build_p.add_argument("--tags-cache-dir", metavar="DIR",
                         help="Directory for Scryfall tag-query cache "
                              "(default: ./scryfall_tags_cache)")
    build_p.add_argument("--tags-offline", action="store_true",
                         help="Only use cached tag data; don't hit Scryfall")

    # Session 6: Bulk data + role validation
    build_p.add_argument("--bulk-source", action="store_true",
                         help="Use Scryfall bulk data (one ~130MB download) instead "
                              "of per-card API calls for images/metadata. Much faster "
                              "for deck builds that use card data heavily.")
    build_p.add_argument("--bulk-cache-dir", metavar="DIR",
                         help="Directory for Scryfall bulk data cache "
                              "(default: ./scryfall_bulk)")
    build_p.add_argument("--bulk-type", default="oracle_cards",
                         choices=["oracle_cards", "default_cards", "unique_artwork"],
                         help="Which bulk file to use (default: oracle_cards)")
    build_p.add_argument("--bulk-offline", action="store_true",
                         help="Only use cached bulk data; don't download")
    build_p.add_argument("--validate-roles", action="store_true",
                         help="After the build, cross-check regex role "
                              "classifications against community oracle tags and "
                              "print a disagreement report. Diagnostic only — "
                              "doesn't change the deck.")

    build_p.set_defaults(func=cmd_build)

    # quick
    quick_p = subparsers.add_parser("quick", help="Quick heuristic-only build")
    quick_p.add_argument("commander")
    quick_p.add_argument("-o", "--output")
    quick_p.set_defaults(func=cmd_quick)

    # analyze
    analyze_p = subparsers.add_parser("analyze", help="Analyze a commander")
    analyze_p.add_argument("commander")
    analyze_p.add_argument("--mock", action="store_true",
                           help="Use mock LLM")
    analyze_p.set_defaults(func=cmd_analyze)

    # search
    search_p = subparsers.add_parser("search", help="Search the card database")
    search_p.add_argument("query", help="Text pattern (regex)")
    search_p.add_argument("--colors", help='Color identity filter (e.g. "WG")')
    search_p.add_argument("--limit", type=int, default=20)
    search_p.set_defaults(func=cmd_search)

    # stats
    stats_p = subparsers.add_parser("stats", help="Database statistics")
    stats_p.set_defaults(func=cmd_stats)

    # v0.9.20: local web GUI
    gui_p = subparsers.add_parser(
        "gui", help="Launch the local web GUI (simple mode + advanced knobs, "
                    "refresh-cards and power-scan buttons)")
    gui_p.add_argument("--port", type=int, default=8765)
    gui_p.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open the browser")
    gui_p.set_defaults(func=cmd_gui)

    # v0.9.18: refresh the card CSV from MTGJSON
    refresh_p = subparsers.add_parser(
        "refresh-cards",
        help="Download AtomicCards from MTGJSON and (re)build the card CSV "
             "with an isGameChanger column",
    )
    refresh_p.add_argument("--output", metavar="FILE",
                           help="Where to write the CSV (default: the --csv "
                                "path, in place)")
    refresh_p.add_argument("--source", metavar="URL",
                           help="MTGJSON source URL (default: AtomicCards.json.gz)")
    refresh_p.add_argument("--atomic-json", metavar="FILE",
                           help="Use a local AtomicCards.json[.gz] instead of "
                                "downloading")
    refresh_p.add_argument("--force", action="store_true",
                           help="Overwrite the output file if it already exists")
    refresh_p.set_defaults(func=cmd_refresh_cards)

    # v0.9.16: bulk power-scan into the global cache
    scan_p = subparsers.add_parser(
        "power-scan",
        help="Bulk-score cards into the GLOBAL card-power cache "
             "(feeds the power-staples channel + role bypass)",
    )
    scan_p.add_argument("--colors", metavar="WUBRG",
                        help="Color-identity scope, e.g. GU (default: whole DB)")
    scan_p.add_argument("--model", default="claude-sonnet-4-6")
    scan_p.add_argument("--cache-dir", default="./card_power_cache")
    scan_p.add_argument("--batch-size", type=int, default=100)
    scan_p.add_argument("--dry-run", action="store_true",
                        help="Show uncached count + estimated calls, score nothing")
    scan_p.set_defaults(func=cmd_power_scan)

    # v0.5: diff two warm-start snapshots
    diff_p = subparsers.add_parser(
        "diff",
        help="Compare two deck snapshots (JSON files from --save-deck)",
    )
    diff_p.add_argument("from_path", help="First deck JSON")
    diff_p.add_argument("to_path", help="Second deck JSON")
    diff_p.add_argument(
        "--show-kept", action="store_true",
        help="Also list cards that appear in both decks (usually noisy)",
    )
    diff_p.add_argument(
        "--max-per-group", type=int, default=20,
        help="Max cards to show per role bucket (default: 20)",
    )
    diff_p.set_defaults(func=cmd_diff)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    setup_logging(args.verbose, log_file=getattr(args, "log_file", None))

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        if args.verbose:
            logging.exception("Unhandled error")
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
