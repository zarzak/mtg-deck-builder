"""
Map submitted form data to a BuildConfig.

This is the web equivalent of cli.py's argparse → BuildConfig flow.
Lives in its own module so it's testable without spinning up FastAPI.

Form fields are always strings (multipart form / query string). This
module handles coercion to the right types and provides sensible
defaults for anything omitted.

NOTE: `mock` is a form field used at the LLMConfig level, not here.
config_from_form() ignores it and leaves that concern to callers.
"""

from __future__ import annotations

from typing import Any

from ..models import BuildConfig

# Reuse the CLI's preset definitions so form presets and CLI --preset
# produce identical weight dicts. Single source of truth.
from ..cli import WEIGHT_PRESETS


def _coerce_int(value: Any, default: int) -> int:
    if value in (None, "", "none"):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an int to [lo, hi]. Friendlier than rejecting out-of-range
    submits: a user who types population=0 gets a usable build instead
    of an error page."""
    return max(lo, min(hi, value))


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, "", "none"):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, "", "none"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _coerce_bool(value: Any) -> bool:
    """HTML checkbox semantics: present → True, absent → False."""
    return value in ("on", "true", True, "1", 1)


def _split_lines(value: Any) -> list[str]:
    """Split a textarea into stripped non-empty lines."""
    if not value:
        return []
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _split_csv(value: Any) -> list[str]:
    """Split 'a, b, c' into ['a', 'b', 'c']."""
    if not value:
        return []
    return [s.strip() for s in str(value).split(",") if s.strip()]


def _parse_weights(form: dict) -> dict[str, float]:
    """Collect weight_<dim>=<float> fields into a dim→weight dict."""
    weights = {}
    for key, value in form.items():
        if not key.startswith("weight_"):
            continue
        dim = key[len("weight_"):]
        w = _coerce_optional_float(value)
        if w is not None:
            weights[dim] = w
    return weights


def build_score_weights(
    preset: str | None,
    weight_overrides: dict[str, float],
) -> dict[str, float] | None:
    """
    Combine a preset name + per-dim overrides into a score_weights dict.

    Mirrors cli._build_weight_dict logic. Returns None if neither was
    provided (caller keeps BuildConfig defaults).
    """
    if not preset and not weight_overrides:
        return None

    # Start with preset if given, else balanced
    weights = dict(WEIGHT_PRESETS[preset or "balanced"])
    known_dims = set(weights.keys())

    for dim, val in weight_overrides.items():
        if dim in known_dims:
            weights[dim] = val
        # Silently drop unknown dims (errors would confuse web users)

    return weights


def config_from_form(form: dict) -> BuildConfig:
    """
    Build a BuildConfig from a form dict.

    Required: `commander_name`. Everything else has sensible defaults
    matching BuildConfig defaults.

    Raises ValueError if commander_name is missing — callers should
    catch and render a form error.
    """
    commander_name = (form.get("commander_name") or "").strip()
    if not commander_name:
        raise ValueError("commander_name is required")

    kwargs: dict = dict(
        commander_name=commander_name,
        # GA params: clamp to sane ranges so a typo doesn't crash the build.
        # Mirrors the input min/max in build_form.html.
        population_size=_clamp(
            _coerce_int(form.get("population_size"), 50), 4, 500,
        ),
        generations=_clamp(
            _coerce_int(form.get("generations"), 100), 1, 1000,
        ),
        patience_generations=_clamp(
            _coerce_int(form.get("patience_generations"), 20), 1, 1000,
        ),
        random_seed=_coerce_optional_int(form.get("random_seed")),
        # Integrations
        use_images=_coerce_bool(form.get("use_images")),
        images_offline=_coerce_bool(form.get("images_offline")),
        use_edhrec=_coerce_bool(form.get("use_edhrec")),
        edhrec_offline=_coerce_bool(form.get("edhrec_offline")),
        use_embeddings=_coerce_bool(form.get("use_embeddings")),
        # v0.5 tags
        flavor_art_tags=_split_csv(form.get("flavor_tags")),
        tags_offline=_coerce_bool(form.get("tags_offline")),
        # v0.6 bulk + validation
        use_bulk_source=_coerce_bool(form.get("use_bulk_source")),
        bulk_type=(form.get("bulk_type") or "oracle_cards"),
        bulk_offline=_coerce_bool(form.get("bulk_offline")),
        validate_roles_after_build=_coerce_bool(form.get("validate_roles")),
        # LLM review
        enable_llm_review=_coerce_bool(form.get("enable_llm_review")),
        # Budget
        budget_max_per_card=_coerce_optional_float(
            form.get("budget_max_per_card")
        ),
        # Refinement
        locked_cards=_split_lines(form.get("locked_cards")),
        banned_cards=_split_lines(form.get("banned_cards")),
    )

    # Weights — only include if preset or per-dim override was submitted
    preset = (form.get("preset") or "").strip() or None
    overrides = _parse_weights(form)
    score_weights = build_score_weights(preset, overrides)
    if score_weights is not None:
        kwargs["score_weights"] = score_weights

    return BuildConfig(**kwargs)
