"""
Per-commander synergy-score cache (v0.9.31).

The LLM synergy pass re-scored every card on every run (~30-40 Sonnet calls),
and its ±5-8 point sampling variance was the largest remaining source of
run-to-run deck churn: scores shift → pool shifts → combo-cache misses →
different decks from identical settings. This cache makes repeat builds of
the same commander reuse prior scores — near-deterministic scoring and a
large cost cut — mirroring the global card-power cache's accumulate-forever
philosophy, scoped per commander because synergy is commander-relative.

Invalidation is exact, not heuristic. An entry is reused only when ALL of:
  - the card's TEXT hash matches (DB refresh with new wording → rescore);
  - the card's HINT TAG matches ([SYN+++] anchors the rubric's score bands,
    so the same card under a different tag is a different question);
  - the file's RUBRIC hash matches (prompt edits invalidate the whole file);
  - the model matches (part of the filename).

Effect-class tags ride along with each score: the consistency dimension
needs them, and a cache hit must not silently lose a card's class.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Optional

from .models import Card

logger = logging.getLogger(__name__)


def _text_hash(card: Card) -> str:
    h = hashlib.sha256()
    for part in (card.text or "", card.card_type or "",
                 card.mana_cost or ""):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def rubric_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


class SynergyScoreCache:
    """Disk-backed {card -> (score, effect_class)} for one commander+model."""

    def __init__(self, commander: str, model: str, rubric: str,
                 cache_dir: Optional[str] = "./synergy_cache"):
        self.commander = commander
        self.model = model
        self.rubric_hash = rubric_hash(rubric)
        self.cache_dir = cache_dir
        self._entries: dict[str, dict] = {}
        self._dirty = False
        self._load()

    # -- persistence ---------------------------------------------------

    def _path(self) -> Optional[str]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9._-]", "_",
                      f"{self.commander}_{self.model}")
        return os.path.join(self.cache_dir, f"synergy_{slug}.json")

    def _load(self) -> None:
        path = self._path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("rubric_hash") != self.rubric_hash:
                logger.info(
                    f"Synergy cache for {self.commander}: rubric changed — "
                    f"discarding {len(data.get('cards', {}))} stale entries"
                )
                return
            self._entries = dict(data.get("cards", {}))
        except Exception as e:
            logger.warning(f"Synergy cache read failed ({path}): {e}")

    def save(self) -> None:
        path = self._path()
        if not path or not self._dirty:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"rubric_hash": self.rubric_hash,
                           "cards": self._entries}, f, indent=1)
            self._dirty = False
        except Exception as e:
            logger.warning(f"Synergy cache write failed ({path}): {e}")

    # -- lookup / store --------------------------------------------------

    def lookup(
        self,
        cards: list[Card],
        hints: Optional[dict[str, str]] = None,
    ) -> tuple[dict[str, float], dict[str, str], list[Card]]:
        """Split `cards` into cache hits and misses.

        Returns (scores, effect_classes, misses): scores/classes for the
        hits, and the cards the LLM still has to score.
        """
        hints = hints or {}
        scores: dict[str, float] = {}
        classes: dict[str, str] = {}
        misses: list[Card] = []
        for card in cards:
            entry = self._entries.get(card.name)
            if (entry is not None
                    and entry.get("h") == _text_hash(card)
                    and entry.get("t") == (hints.get(card.name) or "")):
                scores[card.name] = float(entry["s"])
                if entry.get("c"):
                    classes[card.name] = entry["c"]
            else:
                misses.append(card)
        return scores, classes, misses

    def store(
        self,
        cards: list[Card],
        scores: dict[str, float],
        effect_classes: Optional[dict[str, str]] = None,
        hints: Optional[dict[str, str]] = None,
    ) -> None:
        """Record freshly-scored cards (cards without a score are skipped —
        a parse failure must not cache a hole)."""
        hints = hints or {}
        effect_classes = effect_classes or {}
        for card in cards:
            score = scores.get(card.name)
            if score is None:
                continue
            self._entries[card.name] = {
                "h": _text_hash(card),
                "t": hints.get(card.name) or "",
                "s": float(score),
                "c": effect_classes.get(card.name) or None,
            }
            self._dirty = True

    def __len__(self) -> int:
        return len(self._entries)
