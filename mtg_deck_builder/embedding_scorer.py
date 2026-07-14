"""
Embedding-based synergy scoring.

Computes synergy by cosine similarity between a commander's "strategy text"
(derived from commander text + analysis) and each card's text representation.

Why: LLM synergy scoring is great but expensive. For 500+ candidates we're
paying real tokens. Embeddings are ~1000x faster and cheap enough to re-run
during experimentation. They're not as smart as an LLM but good enough as
a first-pass filter.

Implementation notes:
- `sentence-transformers` is an optional dependency. If not installed, this
  module returns None from scorer factory. Caller falls back to LLM/heuristic.
- Default model: `all-MiniLM-L6-v2` (~25MB, fast, decent quality). Users can
  swap for bigger models.
- Cosine similarity typically lives in [0, 1] for our use case. We linearly
  map to [30, 95] so scores are comparable to LLM/heuristic 0-100 output.

This is a scaffold — the actual semantic quality depends heavily on:
  (a) what text we embed (see _build_commander_query and _build_card_text)
  (b) which model we use
Both are easy to swap independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

from .models import Card, CommanderAnalysis

logger = logging.getLogger(__name__)


def is_embeddings_available() -> bool:
    """Check whether sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    # Where similarity [0, 1] maps on the 0-100 synergy scale
    score_floor: float = 30.0
    score_ceiling: float = 95.0


class EmbeddingSynergyScorer:
    """
    Score synergy via semantic similarity.

    Usage:
        scorer = EmbeddingSynergyScorer.create_if_available(analysis)
        if scorer:
            scores = scorer.score_cards(cards)  # dict name -> 0-100

    Returns None from create_if_available() if sentence-transformers isn't
    installed — caller should fall back to LLM scoring.
    """

    def __init__(
        self,
        analysis: CommanderAnalysis,
        config: Optional[EmbeddingConfig] = None,
        model: Any = None,  # injected for testing
    ):
        self.analysis = analysis
        self.config = config or EmbeddingConfig()
        self._model = model  # actual sentence_transformers.SentenceTransformer
        self._commander_embedding = None
        # Precompute commander embedding on first use
        if self._model is not None:
            self._commander_embedding = self._embed_commander()

    @classmethod
    def create_if_available(
        cls,
        analysis: CommanderAnalysis,
        config: Optional[EmbeddingConfig] = None,
    ) -> Optional["EmbeddingSynergyScorer"]:
        """
        Factory that returns None if sentence-transformers isn't installed.
        Callers should gracefully fall back to LLM scoring in that case.
        """
        if not is_embeddings_available():
            logger.info(
                "sentence-transformers not installed; "
                "embedding synergy disabled. Install with: "
                "pip install sentence-transformers"
            )
            return None

        try:
            from sentence_transformers import SentenceTransformer
            cfg = config or EmbeddingConfig()
            logger.info(f"Loading sentence-transformers model: {cfg.model_name}")
            model = SentenceTransformer(cfg.model_name)
            return cls(analysis, config=cfg, model=model)
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            return None

    def score_cards(self, cards: list[Card]) -> dict[str, float]:
        """
        Score a batch of cards by semantic similarity to the commander.

        Returns dict mapping card.name -> synergy score (0-100).
        Much faster than LLM scoring; good for bulk initial filtering.
        """
        if self._model is None or self._commander_embedding is None:
            # Scorer was constructed without a model (shouldn't happen with
            # factory usage, but handle defensively)
            return {c.name: 50.0 for c in cards}

        texts = [self._build_card_text(c) for c in cards]
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.warning(f"Embedding encode failed: {e}")
            return {c.name: 50.0 for c in cards}

        # Cosine similarity between each card and the commander
        scores = {}
        for card, card_emb in zip(cards, embeddings):
            sim = self._cosine_similarity(self._commander_embedding, card_emb)
            scores[card.name] = self._similarity_to_score(sim)
        return scores

    def score_card(self, card: Card) -> float:
        """Score a single card (convenience)."""
        return self.score_cards([card])[card.name]

    # ------------------------------------------------------------------
    # Text construction
    # ------------------------------------------------------------------

    def _embed_commander(self):
        """Embed the commander's strategy text."""
        query_text = self._build_commander_query()
        return self._model.encode(query_text, convert_to_numpy=True)

    def _build_commander_query(self) -> str:
        """
        Build the text we embed for the commander.

        Includes: the commander's build-around description (from analysis),
        key mechanics, and synergy keywords. This is what we're "searching for"
        semantically in the card pool.

        The commander's NAME is deliberately EXCLUDED: name tokens carry no
        strategy semantics but produce spurious cosine matches against cards
        that merely share a word — a "Jasmine Boreal of the Seven" query
        ranked "Jasmine Dragon Tea Shop" (an off-plan land) into the
        guaranteed GA bypass purely on the shared "Jasmine" token. Because
        the LLM also writes the commander's name INTO build_around_text
        ("Jasmine wants..."), name occurrences are scrubbed from the prose
        too — the full name, and (case-sensitively, to spare common nouns)
        its leading name word.
        """
        import re as _re
        parts = [
            self.analysis.build_around_text or "",
            "Key mechanics: " + ", ".join(self.analysis.key_mechanics or []),
            "Synergy: " + ", ".join(self.analysis.synergy_keywords or []),
        ]
        query = " ".join(p for p in parts if p)
        name = (self.analysis.name or "").strip()
        if name:
            query = query.replace(name, " ")
            first = name.split(",")[0].split()[0] if name.split() else ""
            if len(first) >= 4:
                query = _re.sub(rf"\b{_re.escape(first)}\b", " ", query)
        return query

    @staticmethod
    def _build_card_text(card: Card) -> str:
        """Build the text we embed for a single card.

        The card NAME is excluded (v0.9.13): names carry almost no gameplay
        semantics but create spurious cosine matches against queries that
        share a word (the "Jasmine Dragon Tea Shop" problem). Rules text,
        type line, and keywords are what the strategy match should key on.
        NOTE: changing this invalidates the per-card embedding cache (the
        cache hashes this text) — the next build re-embeds once (CPU-only).
        """
        parts = [card.card_type or "", card.text or ""]
        if card.keywords:
            parts.append("Keywords: " + card.keywords.replace(",", ", "))
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Similarity math
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity between two numpy vectors."""
        # sentence-transformers returns numpy arrays; we avoid importing numpy
        # at module load (it's a transitive dep of sentence-transformers)
        import numpy as np
        a_arr = np.asarray(a, dtype="float32")
        b_arr = np.asarray(b, dtype="float32")
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) or 1.0
        return float(np.dot(a_arr, b_arr) / denom)

    def _similarity_to_score(self, sim: float) -> float:
        """
        Map cosine similarity to a 0-100 synergy score.

        Similarity typically lives in [0, 0.8] for meaningful matches;
        [0.3, 0.6] is a common range. We linearly interpolate into a
        [floor, ceiling] range.
        """
        # Clamp to [0, 1] to be safe
        sim = max(0.0, min(1.0, sim))
        floor = self.config.score_floor
        ceiling = self.config.score_ceiling
        return floor + sim * (ceiling - floor)
