"""
LLM combo / engine detection (v0.9.8) — the SOURCE layer for combo-aware
deck building.

Per-card scorers (synergy, card power) judge cards in isolation, so they miss
2nd/3rd-order value: cards that are mediocre alone but game-ending together
(Spike Feeder + Heliod, Sun-Crowned = infinite life), and "engine" cards whose
worth scales with the deck (Soul Warden in a go-wide lifegain shell).

This module produces a ComboReport that downstream layers consume:
  - the enabler on-ramp guarantees engines + combo pieces reach the GA pool;
  - the interaction-aware GA fitness rewards decks that actually assemble them.

Detection is "both, pool-first":
  1. POOL pass   — find combos AMONG the candidate pool (buildable right now).
  2. KNOWLEDGE pass — list known combos for the commander's archetype from the
     model's own knowledge; pieces not yet in the pool become `missing_pieces`
     so recall can pull them in.

LLM is the PRIMARY source. EDHREC's /combos data is a FALLBACK only (used when
LLM detection is disabled or fails), honoring the project rule that EDHREC is
never a primary input. Results are cached per-commander on disk and re-run only
when the pool changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Callable, Optional

from .models import Card, CommanderAnalysis, Combo, ComboReport

logger = logging.getLogger(__name__)


_POOL_SYSTEM = """You are a world-class Magic: The Gathering Commander (EDH) combo analyst.

You will be given a commander and a list of candidate cards. Find:
  A) COMBOS — sets of 2+ of the listed cards that, together, produce a powerful
     or game-winning result (infinite life, infinite damage, infinite tokens,
     a lock, etc.). Only use card names from the provided list. Two-card combos
     are most valuable; include strong 3-card combos too.
  B) ENGINES — single cards whose value SCALES with the deck's strategy
     (repeatable triggers / payoffs like "whenever a creature enters" or
     "whenever you gain life"), as opposed to being strong in isolation.

Judge relevance to THIS commander's strategy. Be precise: do not invent
interactions that don't actually work.

Return AT MOST the 40 strongest combos and 30 most important engines (most
impactful first). Be concise — keep "result" under 8 words.

Return ONLY this JSON (no prose, no fences):
{"combos": [{"cards": ["Exact Name A","Exact Name B"], "payoff": <0-100>, "result": "<short>"}],
 "engines": [{"name": "Exact Name", "note": "<why it scales, <=10 words>"}]}"""

_VERIFY_SYSTEM = """You are a Magic: The Gathering RULES EXPERT verifying machine-detected combos.

For each numbered combo below, decide whether those exact cards, together on
the battlefield (the deck's commander is always available), can actually
produce the claimed result under the real comprehensive rules. REJECT only
when the claim genuinely fails:
  - the interaction does not function by the rules (e.g. an ability that
    triggers on "tapping a permanent for mana" does NOT apply to mana
    produced by sacrificing or exiling something as a cost);
  - the arithmetic doesn't deliver the claim (a loop that nets zero mana is
    not "infinite mana"; a commander re-cast tax breaks a claimed loop);
  - a SPECIFIC unlisted card is required for the combo to exist at all.

Do NOT reject for these:
  - needing a GENERIC resource the deck class reliably provides (e.g.
    "infinite with any 3+ mana of rocks/dorks", "with any large creature",
    "with any untap effect") — that is a valid combo as claimed;
  - honest VALUE-ENGINE claims that are accurate but not infinite
    (repeatable extra activations, steady advantage) — engines are valid;
  - minor overstatement of magnitude when the interaction itself is real.

When uncertain, KEEP the combo — this pass exists to remove rules-impossible
phantoms, not to curate power level.

Return ONLY JSON (no prose, no fences): {"invalid": [<numbers of rejected combos>]}"""


_KNOWLEDGE_SYSTEM = """You are a world-class Magic: The Gathering Commander (EDH) combo analyst.

Given a commander and its strategy, list the most important KNOWN combos and
engine cards for that commander/archetype from your own knowledge — including
pieces a typical list runs even if not named below. Prefer compact, reliable
combos (2-3 cards). Use exact, real card names.

Return AT MOST 40 combos and 30 engines (most important first). Be concise.

Return ONLY this JSON (no prose, no fences):
{"combos": [{"cards": ["Exact Name A","Exact Name B"], "payoff": <0-100>, "result": "<short>"}],
 "engines": [{"name": "Exact Name", "note": "<why it scales, <=10 words>"}]}"""


_DB_RATE_SYSTEM = """You are a Magic: The Gathering Commander (EDH) COMBO POWER RATER.

You will be given a commander's strategy and a list of HUMAN-VERIFIED combos
(each already known to work by the rules — do not re-verify them). For each,
rate its payoff for THIS commander's deck on a 0-100 scale:
90-100 = wins the game outright or generates a game-winning resource loop;
70-89 = massive advantage that usually wins shortly;
40-69 = strong value engine, not game-ending;
0-39 = marginal or anti-synergistic with this commander's plan.

Respond with JSON only:
{"ratings": [{"cards": ["Name A", "Name B"], "payoff": 95}, ...]}
Include every combo you were given, using the exact card names provided."""

_SIGNATURE_SYSTEM = """You are a world-class Magic: The Gathering Commander (EDH) combo historian.

Name this commander's SIGNATURE combos — the famous, known-tech interactions
that experienced players and cEDH/meta lists specifically build this commander
around. INCLUDE the pieces whose ONLY reason to be played is this exact
interaction (niche cards that look weak in a vacuum but are the commander's
defining combo). These are often missed by generic "good cards" analysis — that
is exactly why we ask. Use exact, real card names. If the commander has no
famous signature combo, return an empty list rather than inventing one.

Return AT MOST 20 combos (most iconic first). Be concise.

Return ONLY this JSON (no prose, no fences):
{"combos": [{"cards": ["Exact Name A","Exact Name B"], "result": "<short>"}]}"""


def _extract_json_object(text: str) -> dict:
    """Pull the first top-level JSON object out of an LLM response."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _iter_json_dicts(text: str):
    """Yield every balanced {...} substring (at any nesting depth) that parses
    as a JSON dict. Robust to TRUNCATION: a cut-off trailing object simply
    fails to close and is skipped, so we still recover every complete object
    before the cut. Used to salvage combos/engines from a truncated response."""
    starts: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            start = starts.pop()
            try:
                obj = json.loads(text[start:i + 1])
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


class ComboEngineDetector:
    """Detect combos + engines for a commander via the LLM, cached on disk."""

    def __init__(
        self,
        llm,
        model: str = "claude-sonnet-4-6",
        cache_dir: Optional[str] = "./combo_cache",
        max_pool: int = 350,
        signature_pass: bool = True,
        knowledge_deepen: bool = True,
    ):
        self.llm = llm
        self.model = model
        self.cache_dir = cache_dir
        self.max_pool = max(1, max_pool)
        # v0.9.30: second knowledge pass that SEES what was already found and
        # asks only for what's missing ("let the process finish" — a single
        # temp-0 sample provably misses known lines run-to-run; observed:
        # Doomsday+Citadel absent from one sample after three hits).
        self.knowledge_deepen = knowledge_deepen
        # v0.9.16c: the signature pass is a RECALL-ONLY aid — it names the
        # commander's famous combos so their niche pieces (Mirror Universe
        # for Selenia) get pulled into the pool. It deliberately does NOT
        # feed the combo SCORE, engine boost, or on-ramp: it provides more
        # data without reweighting anything (the GA/refinement still decide).
        self.signature_pass = signature_pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        analysis: CommanderAnalysis,
        pool_cards: list[Card],
        edhrec_fallback: Optional[Callable[[], ComboReport]] = None,
        database_combos: Optional[list[dict]] = None,
    ) -> ComboReport:
        """Return a ComboReport for `analysis` over `pool_cards`.

        `pool_cards` should already be scoped/ranked by the caller (we analyze
        at most `max_pool` of them in the pool pass). `edhrec_fallback`, if
        given, is called ONLY when the LLM layer produced nothing (disabled,
        mock, or error) — never as a primary source.
        """
        pool = pool_cards[:self.max_pool]
        pool_names = {c.name for c in pool_cards}  # full pool for membership

        cached = self._load_cache(analysis.name, pool_names)
        if cached is not None:
            # v0.9.30: a cache hit must still pick up database combos the
            # cached run didn't know (EDHREC data refreshes weekly). No-op
            # (and zero LLM calls) when nothing new.
            if database_combos:
                before = len(cached.combos)
                try:
                    self._ingest_database_combos(analysis, database_combos,
                                                 cached, pool_names)
                    if len(cached.combos) > before:
                        self._save_cache(analysis.name, pool_names, cached,
                                         verified=True)
                except Exception as e:
                    logger.warning(f"Database combo ingest failed: {e}")
            return cached

        report = ComboReport()
        llm_ok = self.llm is not None and not getattr(
            self.llm.config, "mock_mode", False
        )

        if llm_ok:
            # Isolate the passes — a failure in one must not lose the other
            # (the knowledge pass is what reliably finds the key combos).
            try:
                self._pool_pass(analysis, pool, report)
            except Exception as e:
                logger.warning(f"Combo pool pass failed: {e}")
            try:
                self._knowledge_pass(analysis, pool_names, report)
            except Exception as e:
                logger.warning(f"Combo knowledge pass failed: {e}")
            if self.knowledge_deepen:
                try:
                    self._knowledge_deepen_pass(analysis, pool_names, report)
                except Exception as e:
                    logger.warning(f"Combo deepening pass failed: {e}")
            if self.signature_pass:
                try:
                    self._signature_recall_pass(analysis, pool_names, report)
                except Exception as e:
                    logger.warning(f"Combo signature pass failed: {e}")

        if not report.combos and not report.engines and edhrec_fallback:
            logger.info("Combo detection: falling back to EDHREC combo data")
            try:
                report = edhrec_fallback() or ComboReport()
            except Exception as e:
                logger.warning(f"EDHREC combo fallback failed: {e}")

        self._dedupe(report)
        # v0.9.15b: verification sub-pass. The pool pass occasionally invents
        # interactions that don't work by the rules (observed: "Food Chain +
        # Kinnan" at payoff 92 — Kinnan only doubles mana from TAPPING, Food
        # Chain's mana comes from an exile cost). A bogus commander-inclusive
        # pair counts as permanently assembled, so it pollutes the combo
        # score AND on-ramps its pieces. One strict rules-check call prunes
        # them before anything downstream sees the report.
        verified_ok = True
        if llm_ok and report.combos:
            try:
                self._verify_pass(report, pool)
            except Exception as e:
                verified_ok = False
                logger.warning(f"Combo verification failed ({e}); keeping all")
        # v0.9.30: human-verified database combos (EDHREC/Commander
        # Spellbook) merge in AFTER our verification pass — they're already
        # rules-verified by humans, so they must not burn verify budget or
        # risk over-pruning. The LLM passes remain for what databases can't
        # cover: brand-new commanders and pool-specific engines.
        if database_combos:
            try:
                self._ingest_database_combos(analysis, database_combos,
                                             report, pool_names)
            except Exception as e:
                logger.warning(f"Database combo ingest failed: {e}")

        # v0.9.29: combo MEMORY. Detected combos are facts about cards, not
        # about this run's pool sample — fold previously-verified combos back
        # in so per-commander knowledge accumulates monotonically (same
        # philosophy as the global power cache) instead of churning with LLM
        # sampling variance. Observed: Doomsday+Citadel [payoff 92] detected
        # in three consecutive Doom runs, absent from the fourth's fresh
        # sample → never on-ramped → silently cut from the deck.
        if llm_ok and verified_ok:
            try:
                self._merge_previous_verified(analysis.name, report,
                                              pool_names)
                self._dedupe(report)
            except Exception as e:
                logger.warning(f"Combo memory merge failed: {e}")

        # Don't cache an empty result — a failed/truncated detection shouldn't
        # poison the cache and suppress a later successful run. A FAILED
        # verification must likewise not stamp the cache as verified, or the
        # next run would trust unverified combos.
        if report.combos or report.engines:
            self._save_cache(analysis.name, pool_names, report,
                             verified=verified_ok)
        logger.info(
            f"Combo detection: {len(report.combos)} combos, "
            f"{len(report.engines)} engines, "
            f"{len(report.missing_pieces)} missing combo pieces"
        )
        return report

    # ------------------------------------------------------------------
    # LLM passes
    # ------------------------------------------------------------------

    def _pool_pass(self, analysis, pool: list[Card], report: ComboReport) -> None:
        if not pool:
            return
        lines = [c.format_for_llm() for c in pool]
        user = (
            f"Commander: {analysis.name}\n"
            f"Strategy: {analysis.build_around_text}\n\n"
            f"Candidate cards ({len(pool)}):\n" + "\n".join(lines)
        )
        raw = self._call(_POOL_SYSTEM, user)
        self._ingest_text(raw, report, source="llm-pool")

    def _knowledge_pass(self, analysis, pool_names: set[str],
                        report: ComboReport) -> None:
        kw = ", ".join(analysis.synergy_keywords or [])
        user = (
            f"Commander: {analysis.name}\n"
            f"Color identity: {analysis.color_identity}\n"
            f"Strategy: {analysis.build_around_text}\n"
            f"Key mechanics: {kw}\n"
        )
        raw = self._call(_KNOWLEDGE_SYSTEM, user)
        before = len(report.combos)
        self._ingest_text(raw, report, source="llm-knowledge")
        # Knowledge-pass combo cards not already in the pool become recall
        # targets so the combo can actually be built.
        missing: set[str] = set()
        for combo in report.combos[before:]:
            for name in combo.cards:
                if name not in pool_names:
                    missing.add(name)
        report.missing_pieces = sorted(
            set(report.missing_pieces) | missing
        )

    def _signature_recall_pass(self, analysis, pool_names: set[str],
                               report: ComboReport) -> None:
        """v0.9.16c: RECALL-ONLY signature-combo pass.

        Asks for the commander's famous/known-tech combos and pulls any
        piece NOT already in the pool into `missing_pieces` (the recall
        channel) — nothing more. It does NOT add to `report.combos`, so it
        never touches the combo score, engine boost, or on-ramp: it provides
        MORE DATA (making niche combo cards buildable) without reweighting
        the deck. The recalled card then competes on its honest synergy
        score like any other; the GA/refinement decide whether it makes the
        99. Signature combos are recorded on `report.signature_combos` for
        the report (informational only).
        """
        user = (
            f"Commander: {analysis.name}\n"
            f"Color identity: {analysis.color_identity}\n"
            f"Strategy: {analysis.build_around_text}\n"
        )
        raw = self._call(_SIGNATURE_SYSTEM, user)
        try:
            obj = _extract_json_object(raw)
        except Exception:
            obj = next(
                (d for d in _iter_json_dicts(raw) if "combos" in d), None,
            ) or {}
        pieces: set[str] = set()
        recorded: list[dict] = []
        for c in obj.get("combos", []) or []:
            try:
                cards = [str(n).strip() for n in c["cards"] if str(n).strip()]
            except (KeyError, TypeError):
                continue
            if len(cards) < 2:
                continue
            recorded.append({"cards": cards, "result": str(c.get("result", ""))})
            for name in cards:
                if name not in pool_names:
                    pieces.add(name)
        if pieces:
            report.missing_pieces = sorted(
                set(report.missing_pieces) | pieces
            )
        report.signature_combos = recorded
        logger.info(
            f"Combo signature pass: {len(recorded)} signature combos named, "
            f"{len(pieces)} niche piece(s) pulled into recall "
            f"(recall-only, no scoring weight)"
        )

    def _verify_pass(self, report: ComboReport, pool: list[Card]) -> None:
        """v0.9.15b: strict rules verification of detected combos IN PLACE.

        Sends every combo (with oracle text for pieces we have) to the LLM
        with a hard rules rubric; combos it marks invalid are dropped and
        logged. On any parse/API failure the report is left untouched —
        verification must never LOSE genuine combos to a transient error.
        """
        text_by_name = {c.name: (c.text or "") for c in pool}
        lines = []
        for i, combo in enumerate(report.combos, 1):
            parts = []
            for n in combo.cards:
                t = text_by_name.get(n, "")
                parts.append(f"{n} [{t[:220]}]" if t else n)
            claimed = combo.result or "a powerful/game-winning result"
            lines.append(f"{i}. {' + '.join(parts)} => claimed: {claimed}")
        user = (
            "Verify these detected combos (number => pieces => claimed "
            "result):\n" + "\n".join(lines)
        )
        raw = self._call(_VERIFY_SYSTEM, user)
        # Robust parse: the response may quote card text containing mana
        # symbols ("{T}: Add..."), so first-brace-to-last-brace extraction
        # can land on a non-JSON brace (the same failure class as the
        # {G}{W}-preamble bug in synergy scoring). Try the clean extraction,
        # then fall back to scanning EVERY balanced dict for one with the
        # "invalid" key.
        try:
            obj = _extract_json_object(raw)
            if "invalid" not in obj:
                raise ValueError("no 'invalid' key in extracted object")
        except Exception:
            obj = next(
                (d for d in _iter_json_dicts(raw) if "invalid" in d), None,
            )
            if obj is None:
                raise ValueError(
                    f"no verification verdict found in response "
                    f"({raw[:120]!r}...)"
                )
        invalid = set()
        for x in obj.get("invalid", []) or []:
            try:
                invalid.add(int(x))
            except (TypeError, ValueError):
                continue
        if not invalid:
            logger.info("Combo verification: all combos passed")
            return
        kept, dropped = [], []
        for i, combo in enumerate(report.combos, 1):
            (dropped if i in invalid else kept).append(combo)
        for combo in dropped:
            logger.info(
                f"Combo verification REJECTED: {' + '.join(combo.cards)} "
                f"(claimed: {combo.result})"
            )
        report.combos = kept
        logger.info(
            f"Combo verification: kept {len(kept)}, rejected {len(dropped)}"
        )

    def _call(self, system: str, user: str) -> str:
        """Return the raw LLM text (parsing is done by _ingest_text, which is
        tolerant of truncation). 8000 tokens gives the bounded combo/engine
        list ample room."""
        return self.llm._call_api(
            system, user, temperature=0.0, max_tokens=8000, model=self.model
        )

    def _ingest_text(self, text: str, report: ComboReport, source: str) -> None:
        """Parse a combo/engine response into the report.

        Prefers the clean wrapper object; if that fails (e.g. the response was
        truncated at the token cap), SALVAGES every complete combo/engine
        object that did make it through, so a partial response still yields
        most of its combos instead of nothing.
        """
        try:
            obj = _extract_json_object(text)
            if isinstance(obj, dict) and ("combos" in obj or "engines" in obj):
                self._ingest(obj, report, source)
                return
        except Exception:
            pass
        # Salvage: bucket recovered objects by their keys.
        combos, engines = [], []
        for d in _iter_json_dicts(text):
            if "combos" in d or "engines" in d:  # a (rare) inner wrapper
                self._ingest(d, report, source)
                return
            if "cards" in d:
                combos.append(d)
            elif "name" in d:
                engines.append(d)
        if combos or engines:
            logger.info(
                f"Combo {source}: salvaged {len(combos)} combos / "
                f"{len(engines)} engines from a truncated/garbled response"
            )
        self._ingest({"combos": combos, "engines": engines}, report, source)

    @staticmethod
    def _ingest(obj: dict, report: ComboReport, source: str) -> None:
        for c in obj.get("combos", []) or []:
            try:
                cards = [str(n).strip() for n in c["cards"] if str(n).strip()]
                if len(cards) < 2:
                    continue
                payoff = max(0.0, min(100.0, float(c.get("payoff", 50))))
            except (KeyError, TypeError, ValueError):
                continue
            report.combos.append(Combo(
                cards=cards, payoff=payoff,
                result=str(c.get("result", "")), source=source,
            ))
        for e in obj.get("engines", []) or []:
            try:
                name = str(e["name"]).strip()
            except (KeyError, TypeError):
                continue
            if name:
                report.engines.setdefault(name, str(e.get("note", "")))

    @staticmethod
    def _dedupe(report: ComboReport) -> None:
        """Collapse duplicate combos (same card set), keeping the highest
        payoff and preferring a pool source over knowledge over edhrec."""
        rank = {"llm-pool": 3, "llm-knowledge": 2, "edhrec": 1}
        best: dict[frozenset, Combo] = {}
        for c in report.combos:
            key = frozenset(c.cards)
            cur = best.get(key)
            if cur is None or (c.payoff, rank.get(c.source, 0)) > (
                cur.payoff, rank.get(cur.source, 0)
            ):
                best[key] = c
        report.combos = sorted(
            best.values(), key=lambda c: (-c.payoff, c.cards[0])
        )

    # ------------------------------------------------------------------
    # Cache (per-commander, invalidated when the pool changes)
    # ------------------------------------------------------------------

    def _knowledge_deepen_pass(self, analysis, pool_names: set[str],
                               report: ComboReport) -> None:
        """v0.9.30: second knowledge pass that sees what the first found and
        asks ONLY for what's missing. A single temp-0 sample provably drops
        known lines between runs; showing the found list and asking 'what
        else?' recovers the tail without re-listing duplicates."""
        found = "\n".join(
            " + ".join(c.cards) for c in report.combos[:60]
        ) or "(none found yet)"
        kw = ", ".join(analysis.synergy_keywords or [])
        user = (
            f"Commander: {analysis.name}\n"
            f"Color identity: {analysis.color_identity}\n"
            f"Strategy: {analysis.build_around_text}\n"
            f"Key mechanics: {kw}\n\n"
            f"COMBOS ALREADY FOUND (do NOT repeat these):\n{found}\n\n"
            f"List well-known combos and engines for this commander that are "
            f"MISSING from the list above. If nothing notable is missing, "
            f"return empty lists."
        )
        raw = self._call(_KNOWLEDGE_SYSTEM, user)
        before = len(report.combos)
        self._ingest_text(raw, report, source="llm-knowledge-deepen")
        missing: set[str] = set()
        for combo in report.combos[before:]:
            missing.update(n for n in combo.cards if n not in pool_names)
        if missing:
            report.missing_pieces = sorted(
                set(report.missing_pieces) | missing)
        if len(report.combos) > before:
            logger.info(
                f"Combo deepening pass: +{len(report.combos) - before} "
                f"combo(s) the first sample missed"
            )

    def _ingest_database_combos(self, analysis, database_combos: list[dict],
                                report: ComboReport,
                                pool_names: set[str]) -> None:
        """v0.9.30: merge human-verified database combos (EDHREC/Commander
        Spellbook) as a deterministic backbone.

        Rules-verification is skipped (humans already did it); payoff is
        rated by ONE batched LLM call — strategy judgment, per philosophy —
        with a flat fallback when the LLM is unavailable. Deck counts are
        deliberately ignored: popularity must never become a score. Pieces
        missing from the pool join missing_pieces (recall/on-ramp)."""
        have = {frozenset(c.cards) for c in report.combos}
        fresh: list[Combo] = []
        for entry in database_combos[:150]:
            cards = [str(n) for n in entry.get("cards", []) if n]
            if len(cards) < 2 or frozenset(cards) in have:
                continue
            have.add(frozenset(cards))
            fresh.append(Combo(
                cards=cards,
                result="Human-verified combo (EDHREC/Commander Spellbook)",
                payoff=80,  # flat default; rated below when the LLM is live
                source="database",
            ))
        if not fresh:
            return

        llm_ok = self.llm is not None and not getattr(
            self.llm.config, "mock_mode", False)
        if llm_ok:
            try:
                listing = "\n".join(
                    f"- {' + '.join(c.cards)}" for c in fresh)
                user = (
                    f"Commander: {analysis.name}\n"
                    f"Strategy: {analysis.build_around_text}\n\n"
                    f"Verified combos to rate:\n{listing}"
                )
                raw = self._call(_DB_RATE_SYSTEM, user)
                by_key = {}
                for d in _iter_json_dicts(raw):
                    for r in d.get("ratings", []) or []:
                        if isinstance(r, dict) and r.get("cards"):
                            by_key[frozenset(map(str, r["cards"]))] = r.get(
                                "payoff")
                for combo in fresh:
                    rated = by_key.get(frozenset(combo.cards))
                    if isinstance(rated, (int, float)):
                        combo.payoff = max(0, min(100, int(rated)))
            except Exception as e:
                logger.warning(f"Database combo rating failed ({e}); "
                               f"using flat payoff")

        missing: set[str] = set()
        for combo in fresh:
            report.combos.append(combo)
            missing.update(n for n in combo.cards if n not in pool_names)
        if missing:
            report.missing_pieces = sorted(
                set(report.missing_pieces) | missing)
        logger.info(
            f"Database combos: merged {len(fresh)} human-verified combo(s) "
            f"(EDHREC/Commander Spellbook), {len(missing)} piece(s) to recall"
        )

    def _merge_previous_verified(self, commander: str, report: ComboReport,
                                 pool_names: set[str]) -> int:
        """v0.9.29: fold the previous VERIFIED cache into a fresh report.

        Reads the existing per-commander cache file directly (ignoring its
        pool hash — that mismatch is exactly why we're here) and merges any
        combo/engine/signature entry the fresh passes didn't re-find. Merged
        combo pieces missing from the current pool join `missing_pieces`,
        mirroring the knowledge pass, so the on-ramp can rebuild them.
        Unverified caches are never merged. Returns combos merged."""
        path = self._cache_path(commander)
        if not path or not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return 0
        if not data.get("verified"):
            return 0

        have = {frozenset(c.cards) for c in report.combos}
        missing: set[str] = set()
        merged = 0
        for entry in data.get("combos", []):
            try:
                combo = Combo(**entry)
            except TypeError:
                continue
            key = frozenset(combo.cards)
            if not combo.cards or key in have:
                continue
            report.combos.append(combo)
            have.add(key)
            merged += 1
            missing.update(n for n in combo.cards if n not in pool_names)
        if missing:
            report.missing_pieces = sorted(
                set(report.missing_pieces) | missing
            )
        for name, desc in (data.get("engines") or {}).items():
            report.engines.setdefault(name, desc)
        seen_sig = {tuple(sorted(sc.get("cards", [])))
                    for sc in report.signature_combos
                    if isinstance(sc, dict)}
        for sc in data.get("signature_combos", []):
            if (isinstance(sc, dict)
                    and tuple(sorted(sc.get("cards", []))) not in seen_sig):
                report.signature_combos.append(sc)
        if merged:
            logger.info(
                f"Combo memory: merged {merged} previously-verified combo(s) "
                f"from earlier runs (LLM sampling variance protection)"
            )
        return merged

    def _pool_hash(self, pool_names: set[str]) -> str:
        h = hashlib.sha256()
        for n in sorted(pool_names):
            h.update(n.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    def _cache_path(self, commander: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", f"{commander}_{self.model}")
        return os.path.join(self.cache_dir, f"combos_{slug}.json")

    def _load_cache(self, commander: str, pool_names: set[str]) -> Optional[ComboReport]:
        path = self._cache_path(commander)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("pool_hash") != self._pool_hash(pool_names):
                return None  # pool changed → re-detect
            if not data.get("verified"):
                # v0.9.15b: cache predates the verification sub-pass and may
                # contain rules-invalid combos — re-detect (and re-verify).
                return None
            report = ComboReport(
                combos=[Combo(**c) for c in data.get("combos", [])],
                engines=dict(data.get("engines", {})),
                missing_pieces=list(data.get("missing_pieces", [])),
                signature_combos=list(data.get("signature_combos", [])),
            )
            if not report.combos and not report.engines:
                return None  # stale empty cache (e.g. an old failed run) → re-detect
            logger.info(
                f"Loaded cached combo report for {commander} "
                f"({len(report.combos)} combos) from {path}"
            )
            return report
        except Exception as e:
            logger.warning(f"Combo cache read failed ({path}): {e}")
            return None

    def _save_cache(self, commander: str, pool_names: set[str],
                    report: ComboReport, verified: bool = True) -> None:
        path = self._cache_path(commander)
        if not path:
            return
        try:
            data = {
                "pool_hash": self._pool_hash(pool_names),
                # v0.9.15b: only True when the verification sub-pass actually
                # completed. Unverified caches are treated as misses on load.
                "verified": bool(verified),
                "combos": [vars(c) for c in report.combos],
                "engines": report.engines,
                "missing_pieces": report.missing_pieces,
                "signature_combos": report.signature_combos,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Combo cache write failed ({path}): {e}")
