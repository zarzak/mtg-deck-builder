"""
LLM-based intrinsic card-power scoring (v0.9.7).

Rates each card's STANDALONE power/efficiency on a 0-100 scale, independent of
any commander — the missing "is this card actually good?" signal that the
mana-curve heuristic (everything 50-55) can't provide. This feeds two places:

  1. baseline_power_cache → the Power Level dimension + the effective_synergy
     baseline term, so filler is down-weighted everywhere.
  2. the synergy_engine pre-rank → strong cards climb into the shortlist/bypass
     (recall-feeding), blended synergy-led so commander fit still dominates.

Proven viable with claude-sonnet-4-6 (see poc_card_power.py). Haiku compressed
the mid-band and was too soft on filler, so Sonnet is the default. Because
power is commander-independent, scores are cached globally on disk keyed by
model + card-text hash — like the embedding cache, the first build pays and
every later build reuses; cards whose oracle text changed are re-scored.

EDHREC is intentionally NOT used here: the whole point is a judgment signal
free of community/preconstructed-deck bias. An optional EDHREC swing term may
be layered on later as a *weighting* component, never a filter.
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


# The rubric proven in poc_card_power.py. Absolute calibration anchors keep
# scores comparable across batches without per-batch anchor seeding.
POWER_RUBRIC = """You are a world-class Magic: The Gathering Commander (EDH) evaluator.

Rate each card's INTRINSIC POWER on a 0-100 scale: how strong and efficient the
card is in a generic, well-built Commander deck, judged ONLY on the card itself.

Critical rules:
- Judge the card STANDALONE. Do NOT assume any particular commander, combo, or
  synergy package. A card that is only good alongside a specific build should be
  rated on its own merit, not its ceiling.
- Reward efficiency: low mana cost for high impact, card advantage, flexibility,
  interaction, evasion, and effects that scale.
- Penalize: high cost for low impact, narrow conditions, do-nothing stats,
  win-more effects, and cards that are essentially Limited-only filler.
- Do NOT use real-world price, rarity, or how "popular"/famous the card is.
  Judge the gameplay text. An obscure card with a strong cheap effect should
  score high; a famous-but-clunky card should not get a pass.

Calibration anchors (internalize these bands):
- 90-100: Format-warping efficiency outliers (e.g. Sol Ring, Cyclonic Rift).
- 75-89:  Premium staples, excellent in most decks (e.g. Swords to Plowshares).
- 60-74:  Solid, commonly-played, clearly above replacement.
- 45-59:  Playable role-filler; fine but easily swapped out.
- 30-44:  Weak / marginal; mostly Limited-quality in EDH.
- 0-29:   Near-unplayable in Commander (vanilla stats, do-nothing).

Return ONLY a JSON array, one object per card, in any order:
[{"name": "<exact card name>", "power": <int 0-100>}]
No prose, no markdown fences."""


def _extract_json_array(text: str) -> list:
    """Pull the first top-level JSON array out of an LLM response."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


class CardPowerScorer:
    """
    Score intrinsic card power via the LLM, with a global disk cache.

    Usage:
        scorer = CardPowerScorer(llm_engine, model="claude-sonnet-4-6",
                                 cache_dir="./card_power_cache")
        scores = scorer.score_cards(cards)  # dict name -> 0-100 float
    """

    def __init__(
        self,
        llm,
        model: str = "claude-sonnet-4-6",
        cache_dir: Optional[str] = "./card_power_cache",
        batch_size: int = 100,
    ):
        self.llm = llm
        self.model = model
        self.cache_dir = cache_dir
        self.batch_size = max(1, batch_size)
        self._cache: dict[str, dict] = {}
        self._cache_path: Optional[str] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    @staticmethod
    def _card_text(card: Card) -> str:
        """The exact text the model sees — also what we hash for staleness."""
        return card.format_for_llm()

    @classmethod
    def _text_hash(cls, card: Card) -> str:
        return hashlib.sha256(
            cls._card_text(card).encode("utf-8")
        ).hexdigest()[:16]

    def _load_cache(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.cache_dir:
            return
        import pickle
        os.makedirs(self.cache_dir, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", self.model) or "model"
        self._cache_path = os.path.join(self.cache_dir, f"power_{safe_model}.pkl")
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    self._cache = loaded
                    logger.info(
                        f"Loaded {len(self._cache)} cached card-power scores "
                        f"from {self._cache_path}"
                    )
            except Exception as e:
                logger.warning(
                    f"Card-power cache read failed ({self._cache_path}); "
                    f"rebuilding: {e}"
                )
                self._cache = {}

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        import pickle
        try:
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Card-power cache write failed: {e}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def cached_scores(self) -> dict[str, float]:
        """The ENTIRE global cache as {name: power} — no API calls.

        v0.9.16: powers the global power-staples recall channel. The cache
        accumulates across every run (and via the `power-scan` CLI command),
        so coverage grows monotonically; cards not yet scored simply aren't
        visible to cache-driven channels until they are.
        """
        self._load_cache()
        return {
            name: float(entry["p"])
            for name, entry in self._cache.items()
            if isinstance(entry, dict) and "p" in entry
        }

    def score_cards(self, cards: list[Card]) -> dict[str, float]:
        """Return {name: power 0-100} for every requested card.

        Cache hits are free; only new/changed cards hit the API. Results for
        the whole request are returned regardless of how many were cached.
        """
        self._load_cache()

        # Dedupe by name while preserving the Card objects for scoring.
        unique: dict[str, Card] = {}
        for c in cards:
            unique.setdefault(c.name, c)

        to_score: list[Card] = []
        for name, c in unique.items():
            entry = self._cache.get(name)
            if not isinstance(entry, dict) or entry.get("h") != self._text_hash(c):
                to_score.append(c)

        if to_score:
            logger.info(
                f"Card-power: scoring {len(to_score)} new/changed cards "
                f"(cache held {len(self._cache)}, requested {len(unique)}, "
                f"model={self.model})"
            )
            for i in range(0, len(to_score), self.batch_size):
                batch = to_score[i:i + self.batch_size]
                scored = self._score_batch(batch)
                for c in batch:
                    if c.name in scored:
                        self._cache[c.name] = {
                            "h": self._text_hash(c),
                            "p": scored[c.name],
                        }
            self._save_cache()

        out: dict[str, float] = {}
        for name, c in unique.items():
            entry = self._cache.get(name)
            if isinstance(entry, dict) and "p" in entry:
                out[name] = float(entry["p"])
        return out

    def _score_batch(self, cards: list[Card]) -> dict[str, float]:
        """Score one batch via a single LLM call. Robust to partial/malformed
        responses — unparseable entries are simply skipped (caller leaves them
        uncached, so the heuristic baseline fills the gap)."""
        lines = [c.format_for_llm() for c in cards]
        user_prompt = (
            f"Rate these {len(cards)} cards.\n\n" + "\n\n".join(lines)
        )
        try:
            raw = self.llm._call_api(
                POWER_RUBRIC, user_prompt,
                temperature=0.0, max_tokens=4000, model=self.model,
            )
            results = _extract_json_array(raw)
        except Exception as e:
            logger.warning(f"Card-power batch failed ({len(cards)} cards): {e}")
            return {}

        out: dict[str, float] = {}
        for r in results:
            try:
                name = r["name"]
                power = max(0.0, min(100.0, float(r["power"])))
            except (KeyError, TypeError, ValueError):
                continue
            out[name] = power
        return out
