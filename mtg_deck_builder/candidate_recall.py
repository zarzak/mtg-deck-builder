"""
Candidate-recall sources for the synergy pool.

Three independent layers, each opt-in via a BuildConfig flag, designed to
be unioned together so any single layer's blind spots are covered by the
others:

  recall_from_edhrec(...)    — community-vetted top-N for the commander.
                               Uses EDHREC's commander-specific synergy
                               signal (not raw inclusion rate, which is
                               biased by precons people don't upgrade).

  recall_from_embeddings(...) — semantic similarity between commander
                               strategy text and each card's text. Catches
                               synonyms, paraphrases, and rules-text shapes
                               keyword matching can't see.

  recall_from_patterns(...)   — substring match of LLM-generated patterns
                               against card text after digit/X normalization.
                               Fixes the "gain 1 life" vs "gain life" case
                               and other literal-match failures.

  union_candidates(...)       — combine sources with EDHREC-first priority,
                               dedupe by name, cap at the configured size.

Each recall function takes only the data it needs and returns
`list[Card]` sorted by per-source relevance. They never raise on empty
input or missing dependencies — a missing optional library or a None
EDHREC payload silently yields an empty list, so the caller can union
across whatever's available.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from .models import Card, CommanderAnalysis

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Color-identity helper
# ----------------------------------------------------------------------

def _color_identity_set(s: Optional[str]) -> set[str]:
    """Extract WUBRG letters from a color identity string."""
    return {c for c in (s or "").upper() if c in "WUBRG"}


def _within_commander_colors(card: Card, commander_colors: set[str]) -> bool:
    """True if the card's color identity is a subset of the commander's."""
    return _color_identity_set(card.color_identity).issubset(commander_colors)


# ----------------------------------------------------------------------
# Source 1: EDHREC
# ----------------------------------------------------------------------

def recall_from_edhrec(
    edhrec_data,                     # EDHRECCommanderData | None
    db,                              # CardDatabase
    commander_color_identity: str,
    limit: int = 300,
    min_synergy: float = 0.0,
) -> list[Card]:
    """
    Pull top-N high-synergy cards for the commander from EDHREC data.

    Returns an empty list if `edhrec_data` is None (offline mode, network
    error, commander not found). Skips card names that aren't in the local
    database — the user's card pool may not match what EDHREC has seen.

    Color-identity filtered defensively: EDHREC should already only return
    legal cards for the commander, but third-party data can drift.

    Args:
        edhrec_data: Result of EDHRECClient.fetch_commander(). May be None.
        db: CardDatabase for name → Card resolution.
        commander_color_identity: e.g. "G,W" or "WUBRG".
        limit: Maximum cards to return.
        min_synergy: Minimum EDHREC synergy score to include. EDHREC's
            synergy is typically in [-1, +1]. v0.9.20: default dropped from
            0.1 to 0.0 — synergy is inclusion-minus-baseline, so popular
            commanders compress toward zero (measured: Jodah's top synergy
            0.49 vs 0.81-0.84 for niche commanders; only 31% of its page
            cleared 0.1) and universal staples land at exactly 0.00 (Sol
            Ring: 24,951 Jodah inclusions, synergy +0.00). Page membership
            is already EDHREC's curation (~230 notable cards); recall is
            additive, so the only cards worth excluding are NEGATIVE-synergy
            ones (actively underplayed for this commander).

    Returns:
        list[Card] sorted by EDHREC synergy descending, capped at limit.
    """
    if edhrec_data is None:
        return []

    commander_colors = _color_identity_set(commander_color_identity)
    high_synergy = edhrec_data.get_high_synergy_cards(min_synergy=min_synergy)

    out: list[Card] = []
    skipped_unknown = 0
    skipped_color = 0
    for entry in high_synergy:
        card = db.get_by_name(entry.name)
        if card is None:
            skipped_unknown += 1
            continue
        if not _within_commander_colors(card, commander_colors):
            skipped_color += 1
            continue
        out.append(card)
        if len(out) >= limit:
            break

    logger.info(
        f"EDHREC recall: {len(out)} cards "
        f"(skipped {skipped_unknown} not-in-db, {skipped_color} off-color, "
        f"min_synergy={min_synergy})"
    )
    return out


def recall_from_edhrec_inclusion(
    edhrec_data,                     # EDHRECCommanderData | None
    db,                              # CardDatabase
    commander_color_identity: str,
    limit: int = 100,
    min_inclusion: float = 0.3,
) -> list:
    """v0.9.15c — BRACKET 5 ONLY (caller-enforced): top-N by INCLUSION rate.

    The synergy-metric recall above deliberately skips generically-good
    cards (clones, free interaction, fast mana have LOW distinctive synergy
    for every commander — that's what generic means), which leaves flexible
    staples with no channel into a cEDH pool. The official bracket-5
    definition says such decks are "built using existing cEDH knowledge,
    tools, and/or deck lists" — so meta-consensus inclusion is
    bracket-faithful there.

    Philosophy guardrails: this is additive pool ENTRY only — never a
    filter, never a quality signal; recalled cards get NO synergy hint tag
    (they are not commander-specific), are scored honestly by the LLM
    rubric, and still have to win GA slots on merit.
    """
    if edhrec_data is None:
        return []
    commander_colors = _color_identity_set(commander_color_identity)
    ranked = sorted(
        (e for e in edhrec_data.cards.values()
         if e.inclusion_rate is not None and e.inclusion_rate >= min_inclusion),
        key=lambda e: -(e.inclusion_rate or 0),
    )
    out = []
    for entry in ranked:
        card = db.get_by_name(entry.name)
        if card is None:
            continue
        if not _within_commander_colors(card, commander_colors):
            continue
        out.append(card)
        if len(out) >= limit:
            break
    logger.info(
        f"EDHREC inclusion recall (bracket 5): {len(out)} cards "
        f"(min_inclusion={min_inclusion})"
    )
    return out


# ----------------------------------------------------------------------
# Source 2: Embeddings
# ----------------------------------------------------------------------

def recall_from_embeddings(
    analysis: CommanderAnalysis,
    cards: list[Card],
    limit: int = 1500,
    cache_dir: Optional[str] = None,
    embedding_config: Optional[object] = None,  # EmbeddingConfig
) -> list[Card]:
    """
    Score the given card pool by semantic similarity to the commander
    strategy and return the top-`limit` cards.

    Thin wrapper over `recall_from_embeddings_with_scores` that discards the
    per-card cosine map — kept for callers that only need the ranked cards.

    Returns an empty list if sentence-transformers is not installed —
    callers should already handle that gracefully via the use_embedding_recall
    flag.

    Args:
        analysis: LLM-generated commander analysis (provides the query text).
        cards: Color-pre-filtered candidate pool. We embed all of them once.
        limit: Maximum cards to return.
        cache_dir: Disk cache location for card embeddings. None = no cache.
        embedding_config: Optional EmbeddingConfig override.

    Returns:
        list[Card] sorted by cosine similarity descending, capped at limit.
    """
    out, _scores = recall_from_embeddings_with_scores(
        analysis=analysis,
        cards=cards,
        limit=limit,
        cache_dir=cache_dir,
        embedding_config=embedding_config,
    )
    return out


def recall_from_embeddings_with_scores(
    analysis: CommanderAnalysis,
    cards: list[Card],
    limit: int = 1500,
    cache_dir: Optional[str] = None,
    embedding_config: Optional[object] = None,  # EmbeddingConfig
) -> tuple[list[Card], dict[str, float]]:
    """
    Like `recall_from_embeddings`, but ALSO returns the full per-card cosine
    similarity map for EVERY scored card — not just the top-`limit` returned.

    The cosine-to-commander value is the one recall signal that is available
    and identical in form for any commander (no network, no popularity data),
    so downstream phases use it to rank the synergy pool and to anchor the
    "protect the best payoffs" bypass. Crucially we keep the score for cards
    OUTSIDE the top-`limit` too, so a card that entered the synergy pool via
    EDHREC or patterns (but ranked below the embedding cutoff) still has a
    commander-relevance score to sort by.

    Returns:
        (cards, scores) where:
          - cards: list[Card] sorted by cosine descending, capped at limit
            (identical to recall_from_embeddings).
          - scores: {card.name -> cosine in [-1, 1]} for every card we were
            able to embed. Empty dict if embeddings are unavailable.
    """
    if not cards:
        return [], {}

    from .embedding_scorer import (
        EmbeddingSynergyScorer, EmbeddingConfig, is_embeddings_available,
    )

    if not is_embeddings_available():
        logger.info(
            "recall_from_embeddings: sentence-transformers not installed; "
            "returning empty list. Install with: pip install sentence-transformers"
        )
        return [], {}

    cfg = embedding_config or EmbeddingConfig()

    # Reuse the existing scorer to embed the commander; it already builds
    # the right query string from the analysis.
    scorer = EmbeddingSynergyScorer.create_if_available(analysis, config=cfg)
    if scorer is None:
        return [], {}

    # Embed all cards in one batch call. The cache layer below avoids
    # re-embedding cards we've already seen.
    name_to_embedding = _load_or_compute_card_embeddings(
        scorer=scorer,
        cards=cards,
        cache_dir=cache_dir,
        model_name=cfg.model_name,
    )

    if not name_to_embedding:
        return [], {}

    # Cosine similarity vs commander, sort, take top-N
    import numpy as np
    cmd_emb = np.asarray(scorer._commander_embedding, dtype="float32")
    cmd_norm = np.linalg.norm(cmd_emb) or 1.0

    scored: list[tuple[float, Card]] = []
    score_map: dict[str, float] = {}
    for card in cards:
        emb = name_to_embedding.get(card.name)
        if emb is None:
            continue
        v = np.asarray(emb, dtype="float32")
        denom = (np.linalg.norm(v) * cmd_norm) or 1.0
        sim = float(np.dot(cmd_emb, v) / denom)
        scored.append((sim, card))
        score_map[card.name] = sim

    scored.sort(key=lambda t: t[0], reverse=True)
    out = [c for _, c in scored[:limit]]

    if scored:
        logger.info(
            f"Embedding recall: {len(out)} cards from {len(cards)} pool "
            f"(top sim={scored[0][0]:.3f}, cutoff sim={scored[min(limit-1, len(scored)-1)][0]:.3f})"
        )
    return out, score_map


def _load_or_compute_card_embeddings(
    scorer,                           # EmbeddingSynergyScorer
    cards: list[Card],
    cache_dir: Optional[str],
    model_name: str,
) -> dict[str, object]:
    """
    Return {card.name -> embedding ndarray}, loading from disk where
    possible and computing+caching only the cards that aren't already there.

    The cache is **per-model, accumulating, and self-healing**:

    - One cache file per `model_name` (e.g. emb_all-MiniLM-L6-v2.pkl).
      Different model = different file. Same model across different
      commanders = same file, growing as new cards get introduced.
    - Each cached entry stores the embedding plus a hash of the embed-text
      that produced it. If a card's text changes (DB update), the hash
      mismatches and that single card is re-embedded — no need to clear
      the whole cache.
    - When this function is called with cards already in the cache, we
      do zero embedding work and just return the relevant subset.

    Building Lathiel (G,W) and then Atraxa (WUBG) reuses every shared card
    embedded for Lathiel; only cards exclusive to Atraxa's color identity
    pay the embedding cost.
    """
    import hashlib
    import os
    import pickle
    import re as _re

    def _text_hash(card) -> bytes:
        # 8 bytes of SHA-256 over the embed-text. Detects content changes
        # without inflating the cache file size much.
        return hashlib.sha256(
            scorer._build_card_text(card).encode("utf-8")
        ).digest()[:8]

    cache: dict[str, dict] = {}
    cache_path: Optional[str] = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # File name is just the model — collisions across DB versions are
        # handled by per-card text hashing below, not the file name.
        safe_model = _re.sub(r"[^A-Za-z0-9._-]", "_", model_name) or "model"
        cache_path = os.path.join(cache_dir, f"emb_{safe_model}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    cache = loaded
                    logger.info(
                        f"Loaded {len(cache)} cached embeddings from "
                        f"{cache_path}"
                    )
            except Exception as e:
                logger.warning(
                    f"Embedding cache read failed ({cache_path}); "
                    f"rebuilding from scratch: {e}"
                )
                cache = {}

    # Figure out which of the requested cards are missing or stale.
    to_embed: list[Card] = []
    for c in cards:
        entry = cache.get(c.name)
        if not isinstance(entry, dict):
            to_embed.append(c)
            continue
        if entry.get("h") != _text_hash(c):
            # Card text changed since cached — re-embed.
            to_embed.append(c)

    if to_embed:
        logger.info(
            f"Embedding {len(to_embed)} new/changed cards "
            f"(cache held {len(cache)}, requested pool={len(cards)}, "
            f"model={model_name})…"
        )
        texts = [scorer._build_card_text(c) for c in to_embed]
        try:
            embeddings = scorer._model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            embeddings = []

        for card, emb in zip(to_embed, embeddings):
            cache[card.name] = {"h": _text_hash(card), "emb": emb}

        if cache_path:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Wrote {len(cache)} total embeddings to {cache_path}"
                )
            except Exception as e:
                logger.warning(f"Embedding cache write failed: {e}")
    else:
        logger.info(
            f"All {len(cards)} requested embeddings served from cache "
            f"(no compute work this build)"
        )

    # Return only the embeddings the caller asked for.
    out: dict[str, object] = {}
    for c in cards:
        entry = cache.get(c.name)
        if isinstance(entry, dict) and "emb" in entry:
            out[c.name] = entry["emb"]
    return out


# ----------------------------------------------------------------------
# Source 3: LLM-expanded patterns + normalized substring match
# ----------------------------------------------------------------------

# Standalone digits (1, 23) and standalone X get normalized out so
# "gain 1 life" / "gain X life" both match the pattern "gain life".
_DIGIT_OR_X = re.compile(r"\b(?:\d+|[Xx])\b")
_WHITESPACE = re.compile(r"\s+")


def _normalize_card_text(text: Optional[str]) -> str:
    """
    Normalize card text for substring matching.

    Steps:
      1. lowercase
      2. drop standalone digits  ("gain 1 life" → "gain  life")
      3. drop standalone X        ("gain X life" → "gain  life")
      4. collapse whitespace
    """
    if not text:
        return ""
    s = text.lower()
    s = _DIGIT_OR_X.sub("", s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip()


def recall_from_patterns(
    patterns: Iterable[str],
    cards: list[Card],
    limit: Optional[int] = None,
) -> list[Card]:
    """
    Substring-match each pattern against each card's text after digit/X
    normalization, return the matching cards ranked by relevance.

    Ranking: a card matching N distinct patterns is more on-strategy than
    one matching 1, so we sort by hit-count descending. Ties broken by
    shorter card text (focused single-purpose cards beat 3-paragraph
    legendary creatures), then by name for determinism.

    This matters when patterns include broad ones like "creature token"
    that match thousands of cards. Without the ranking, narrower matches
    (e.g. "gain life" hitting Soul Warden) get squeezed out of the cap by
    a flood of generic token-makers.

    Args:
        patterns: Substring patterns (post-normalization). Empty patterns
            are ignored. The matcher does NOT use regex; it's literal
            substring after normalization.
        cards: Candidate pool to match against.
        limit: Maximum cards to return. None = no limit.

    Returns:
        list[Card] sorted by descending pattern-hit count.
    """
    norm_patterns = [
        _normalize_card_text(p) for p in patterns if p and p.strip()
    ]
    norm_patterns = [p for p in norm_patterns if p]
    if not norm_patterns:
        return []

    # Score each card by how many distinct patterns it hits.
    scored: list[tuple[int, int, str, Card]] = []
    for card in cards:
        body = _normalize_card_text(card.text)
        if not body:
            continue
        hits = sum(1 for p in norm_patterns if p in body)
        if hits == 0:
            continue
        # (hit_count, -len(body), name, card) — more hits first; on tie,
        # shorter text first; final tie-break on name for determinism.
        scored.append((hits, -len(body), card.name, card))

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    out = [t[3] for t in scored]
    if limit is not None:
        out = out[:limit]

    if scored:
        top_hits = scored[0][0]
        logger.info(
            f"Pattern recall: {len(out)} cards from {len(cards)} pool "
            f"using {len(norm_patterns)} patterns "
            f"(top hit count={top_hits}, returned={len(out)})"
        )
    else:
        logger.info(
            f"Pattern recall: 0 cards from {len(cards)} pool "
            f"using {len(norm_patterns)} patterns"
        )
    return out


# ----------------------------------------------------------------------
# Union: combine sources with EDHREC-first priority
# ----------------------------------------------------------------------

def union_candidates(
    *sources: list[Card],
    cap: int = 2500,
) -> list[Card]:
    """
    Merge ordered candidate lists into a single deduplicated list.

    Order priority is the order sources are passed: cards from earlier
    sources win the position. This is intentionally NOT round-robin —
    EDHREC (when first) gets the top of the pool because its
    community-vetted signal is the most expensive to replicate.

    Dedupe is by Card.name (case-sensitive). Caps at `cap` after dedupe.

    Args:
        *sources: list[Card] from each recall layer. Sources contributing
            an empty list are silently skipped.
        cap: Max output size.

    Returns:
        list[Card] of size <= cap, no duplicates.
    """
    seen: set[str] = set()
    out: list[Card] = []
    for src_idx, src in enumerate(sources):
        for card in src:
            if card.name in seen:
                continue
            seen.add(card.name)
            out.append(card)
            if len(out) >= cap:
                logger.info(
                    f"union_candidates: hit cap={cap} during source #{src_idx + 1}"
                )
                return out

    logger.info(
        f"union_candidates: {len(out)} cards from "
        f"{sum(1 for s in sources if s)} non-empty sources (cap={cap})"
    )
    return out
