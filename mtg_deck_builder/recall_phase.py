"""
Recall-pool assembly (v0.9.33) — extracted from deck_builder (#28).

Unions the configured recall sources (EDHREC synergy, bracket-5 EDHREC
inclusion, embedding similarity, analysis patterns) into the synergy
candidate pool, and reports per-source membership so the builder can
(a) compute hint tags and (b) record pool-entry PROVENANCE (#26) — the
"why is this card in/not in the pool" answer that every miss investigation
this project has needed.

Pure assembly: no scoring, no LLM calls (embeddings are local).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from .models import Card, CommanderAnalysis

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """The unioned pool plus per-source membership (for hints + provenance)."""
    cards: list[Card] = field(default_factory=list)
    edhrec_names: set[str] = field(default_factory=set)
    inclusion_names: set[str] = field(default_factory=set)
    embedding_names: set[str] = field(default_factory=set)
    pattern_names: set[str] = field(default_factory=set)
    # Full cosine map over ALL embedded cards (not just the recall cutoff) —
    # feeds the synergy_engine pre-rank.
    embedding_scores: dict[str, float] = field(default_factory=dict)


def build_recall_pool(
    db,
    config,
    analysis: CommanderAnalysis,
    edhrec_data,
    color_id: str,
    progress: Optional[Callable[[str, float, str], None]] = None,
) -> RecallResult:
    """Union the enabled recall sources, capped at config.recall_pool_cap.

    Order priority (earlier wins pool position): EDHREC synergy → bracket-5
    EDHREC inclusion → embeddings → patterns. Color-identity filtering is
    applied uniformly; each source also filters defensively.
    """
    from .candidate_recall import (
        recall_from_edhrec, recall_from_edhrec_inclusion,
        recall_from_embeddings_with_scores,
        recall_from_patterns, union_candidates,
        _within_commander_colors, _color_identity_set,
    )

    def _report(stage: str, pct: float, msg: str) -> None:
        if progress:
            progress(stage, pct, msg)

    result = RecallResult()
    commander_colors = _color_identity_set(color_id)
    legal_pool = [
        c for c in db.all_cards
        if _within_commander_colors(c, commander_colors)
    ]
    logger.info(
        f"Synergy recall: {len(legal_pool)} cards in color identity "
        f"{color_id!r} (from {len(db.all_cards)} total)"
    )

    edhrec_cards: list = []
    edhrec_inclusion_cards: list = []
    embedding_cards: list = []
    pattern_cards: list = []

    if config.recall_use_edhrec:
        _report("edhrec_recall", 0.2, "EDHREC top-N recall...")
        edhrec_cards = recall_from_edhrec(
            edhrec_data=edhrec_data,
            db=db,
            commander_color_identity=color_id,
            limit=config.recall_edhrec_limit,
        )
        # v0.9.15c: at bracket 5 ONLY, also recall by meta-consensus
        # INCLUSION — the official B5 definition builds from existing cEDH
        # lists, and generically-good staples (clones, free interaction)
        # have low DISTINCTIVE synergy by definition, so the synergy-metric
        # recall above can never see them. Pool entry only; no hint tags;
        # honest scoring decides.
        if getattr(config, "bracket", 4) == 5:
            edhrec_inclusion_cards = recall_from_edhrec_inclusion(
                edhrec_data=edhrec_data,
                db=db,
                commander_color_identity=color_id,
            )

    if config.recall_use_embeddings:
        _report("embedding_recall", 0.4, "Embedding similarity recall...")
        from .embedding_scorer import EmbeddingConfig
        embedding_cards, embedding_scores = recall_from_embeddings_with_scores(
            analysis=analysis,
            cards=legal_pool,
            limit=config.recall_embedding_limit,
            cache_dir=config.recall_embedding_cache_dir,
            embedding_config=EmbeddingConfig(
                model_name=config.embedding_model),
        )
        result.embedding_scores = embedding_scores

    if config.recall_use_patterns:
        _report("pattern_recall", 0.7, "Pattern recall...")
        pattern_cards = recall_from_patterns(
            patterns=analysis.synergy_patterns or [],
            cards=legal_pool,
        )
        if not analysis.synergy_patterns:
            logger.warning(
                "recall_use_patterns=True but analysis.synergy_patterns is "
                "empty. Either the LLM didn't return patterns or you're in "
                "mock mode without configured patterns. Falling back to "
                "synergy_keywords as patterns."
            )
            pattern_cards = recall_from_patterns(
                patterns=analysis.synergy_keywords or [],
                cards=legal_pool,
            )

    # EDHREC first (most expensive signal to replicate), then the bracket-5
    # inclusion layer, then embeddings, then pattern-fill.
    result.cards = union_candidates(
        edhrec_cards,
        edhrec_inclusion_cards,
        embedding_cards,
        pattern_cards,
        cap=config.recall_pool_cap,
    )
    logger.info(
        f"Synergy recall union: {len(result.cards)} cards "
        f"(edhrec={len(edhrec_cards)}, embeddings={len(embedding_cards)}, "
        f"patterns={len(pattern_cards)}, cap={config.recall_pool_cap})"
    )

    result.edhrec_names = {c.name for c in edhrec_cards}
    result.inclusion_names = {c.name for c in edhrec_inclusion_cards}
    result.embedding_names = {c.name for c in embedding_cards}
    result.pattern_names = {c.name for c in pattern_cards}
    return result
