"""
LLM Engine - Wrapper for Anthropic Claude API.

Key v0.2 fixes:
- Updated default model to claude-sonnet-4-6 (old default is deprecated)
- Handles Opus 4.7's rejection of temperature parameter
- Added mock_mode for testing without an API key
- More robust JSON parsing that handles partial responses, code fences, etc.
- Commander analysis now requests recommended_weights and recommended_synergy_weight
- Prompt caching on system prompts for cost savings
"""

import dataclasses
import hashlib
import json
import logging
import os
import re
from typing import Optional
from dataclasses import dataclass

from .models import Card, CommanderAnalysis, Deck
from . import tuning

logger = logging.getLogger(__name__)

# Models that reject `temperature` / `top_p` / `top_k` parameters.
# See https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7
_NO_TEMPERATURE_MODELS = {
    'claude-opus-4-7',
    'claude-opus-4-7-20260416',  # placeholder if dated variant appears
}


def _model_rejects_temperature(model_name: str) -> bool:
    for marker in _NO_TEMPERATURE_MODELS:
        if model_name.startswith(marker):
            return True
    return False


# ----------------------------------------------------------------------
# System prompts (cached across calls via prompt caching for cost savings)
# ----------------------------------------------------------------------

COMMANDER_ANALYSIS_PROMPT = """You are an expert Magic: The Gathering deck builder specializing in EDH/Commander format.

Your job is to analyze commanders and produce structured JSON guidance that drives
an automated deck optimizer.

For each commander:
1. Identify primary mechanics and synergies.
2. Note how this commander changes card evaluation (e.g., a commander that rewards
   creatures with no abilities makes vanilla creatures genuinely good; typical
   power heuristics are inverted).
3. Provide keywords/text patterns to find synergy cards in a card database.
4. Recommend scoring weights that reflect how this commander should be built.

Always respond with VALID JSON ONLY. No prose before or after the JSON object."""


CARD_SELECTION_PROMPT = """You are an expert Magic: The Gathering deck builder
working in the EDH/Commander format.

Your task: evaluate a candidate pool of cards and pick the best N for a
specific role (ramp, removal, draw, finisher, synergy_core, etc.) in a deck
led by a specific commander. The commander's strategy, evaluation notes, and
synergy keywords are provided in the Commander Context block below — treat
that block as the source of truth for what this deck wants.

# How to choose

Balance three concerns simultaneously:

1. Commander-specific synergy. The Commander Context describes what the deck
   is trying to do. A card that mechanically advances that plan is worth more
   than a generic strong card. If the commander rewards lifegain, a card with
   "whenever you gain life" beats a card with raw stat efficiency. If the
   commander rewards graveyard fill, a self-mill card beats a removal spell
   for the synergy_core role.

2. Baseline card power. Even a synergy deck needs cards that function on
   their own. A card that only works in a long combo line, with no
   stand-alone value, is a liability when the combo piece is missing.

3. Mana efficiency and flexibility. Lower mana value is generally better at
   the same effect. Cards that have multiple modes or scale with the game
   state are more resilient than narrow one-shot effects.

# How to apply this to each role

- ramp: prefer efficient, cheap (0-3 mana) acceleration, judged on RATE —
  mana spent vs. mana produced, untapped vs. tapped, color fixing — not on
  name recognition. The best SHAPE depends on the deck: rocks/dorks for
  raw speed, land-based ramp when the strategy cares about lands or land
  types. Among similarly-efficient options, prefer ones that incidentally
  synergize (e.g. a mana-dork that gains life for a lifegain commander).
- removal: prefer instant-speed and unconditional. The deck needs answers
  regardless of strategy. Among similarly-effective removal, prefer pieces
  that synergize (lifelink removal creatures for a lifegain commander,
  graveyard-fill removal for a recursion commander, etc.).
- card_draw: prefer repeatable draw and engines tied to commander text over
  one-shot card draw, when possible.
- threats / finishers: top-end creatures and game-closing pieces. Synergy
  with the commander should be high here — these are the reason the deck
  wins. Prefer the payoffs the Commander Context identifies as
  strategy-defining over generically large creatures.
- protection: counterspells, removal-protection, indestructibility.
  Cheap, flexible, instant-speed first; synergy as tiebreaker.
- recursion: graveyard-reanimation and value loops. Heavy synergy weighting
  for commanders whose strategy uses the graveyard at all.
- lands: utility lands, fetches, duals. Synergy mostly doesn't apply here.

# Synergy hints (pre-computed per-card signals)

Each candidate's user-prompt line may begin with a SYNERGY HINT TAG. These
are NOT noise — they're the output of a separate recall pipeline that
checks each card against three independent commander-specific signals:

  • EDHREC (community-vetted high-synergy cards for this commander)
  • Embedding similarity (semantic match to commander strategy text)
  • Substring pattern match (literal text overlap with strategy keywords)

The tags encode how many of these three sources flagged the card:

  [SYN+++]  All three sources agree. ALMOST CERTAINLY commander-defining.
            Pick these aggressively when filling any role they're eligible
            for. A 3-mana 2/3 with [SYN+++] is the threats bucket's
            target over a 4-mana 4/4 with no tag.

  [SYN++]   Two of three sources flag this. Strong synergy candidate;
            weight heavily as a tiebreaker.

  [SYN+]    One source. Some signal — give it a small edge over a
            similarly-fit untagged candidate.

  (no tag)  None of the synergy signals fired. This does NOT mean the
            card is bad — many universally-good cards (efficient mana
            rocks, cheap unconditional interaction, mana fixing)
            correctly have no synergy tag because they're good in ANY
            deck rather than commander-specific. Pick them on role merit.

# Synergy preference (cross-cutting)

**For every role above:** synergy hint tags are a STRONG signal. When two
candidates are equally role-appropriate, the one with the higher synergy
tag is the right pick. This is especially important in roles where the
"best raw card" heuristic would steer you wrong:

- A 3-mana creature tagged [SYN+++] in the threats bucket should beat an
  untagged 5-mana bomb — the recall sources are telling you the tagged
  card is what THIS commander wants.
- A cheap creature with weak combat stats tagged [SYN+++] should beat an
  untagged efficient beater in any creature-eligible role — its stats
  lose on raw threat value but win decisively on commander-specific
  signal.

Do NOT drop functionally-necessary non-synergy cards (efficient ramp,
removal, mana-fixing lands) just because they have no synergy tag. Those
are SUPPOSED to have no tag — they're universally good, not commander-
specific. The tag system flags "this is unusually good FOR THIS
COMMANDER," not "this is good." Functional role-fillers and synergy
picks coexist in a well-built deck.

A note about cheap "engine" pieces (1-2 mana creatures or enchantments
with a single repeatable strategy trigger but weak combat stats): these
are core to many strategies but tend to lose head-to-head against
splashier creatures in the "threats" role evaluation. They will be
picked up by a separate synergy_engine pass after role-selection
completes — you don't need to force them in at this stage. Focus on
filling the requested role well.

# Worked example (attribute-matters commander)

Imagine the commander rewards "creatures with no abilities". The candidate
pool contains:
- A 2-mana 2/2 with no rules text. In a normal deck this is worthless.
  Here it scores HIGH for synergy_core because its vanilla-ness is the
  payoff.
- A cheap generic removal spell. Strong in any deck. Scores normally
  for the removal role; would be a poor synergy_core pick.
- A cheap creature that fills a role (e.g. ramp) AND has the rewarded
  attribute. Scores HIGH for ramp AND synergy_core — pulls double duty.

The same pool with a different commander would rank these completely
differently. Always check the Commander Context first.

# Worked example (graveyard-recursion commander)

Imagine the commander cares about cards leaving the graveyard. The pool
contains:
- A 1-mana creature that self-mills on enter and on death. HIGH for
  synergy_core, modest as generic value.
- A premium 2-mana counterspell. Fine for the protection role at a
  moderate score, but irrelevant for synergy_core.
- A cheap threat whose stats scale with cards in graveyards. HIGH for
  finisher AND synergy_core — it uses the same resource the commander
  cares about.

# What NOT to pick

- Cards that explicitly turn off the commander's strategy (e.g. Rest in
  Peace in a graveyard deck — these are anti-synergy and should be
  filtered, not selected).
- Cards already in the deck. The user message will list "already_selected"
  context implicitly via the candidate pool — you'll only see remaining
  candidates, but verify by name.
- Off-color cards. The commander's color identity is in the Commander
  Context; if a candidate has a color not in that identity, exclude it
  entirely. Don't try to "fix" with hybrid cards unless the candidate
  pool deliberately includes them.
- Don't second-guess format legality: the pool is pre-filtered upstream,
  so every candidate you see is legal. Judge fit, not legality.

# Tie-breakers

When two cards seem equally good for the role:
1. Prefer the cheaper mana value.
2. Prefer the card that adds a redundant copy of an existing effect the
   deck wants more of (e.g. a second copy of a repeatable draw engine).
3. Prefer the card with broader applicability (not strictly better in
   only one matchup).
4. Prefer the card that scales with the late game over the one that's
   only strong on turn three.

# Decision algorithm

Apply this order to each candidate and decide independently:

1. Read the candidate's printed type, mana cost, and rules text.
2. Cross-reference against the role you've been asked to fill. Does the
   candidate plausibly fulfill that role at all? If not, reject.
3. Cross-reference against the Commander Context. Does the candidate
   advance, enable, or amplify the strategy? Strong fit → keep, mark as
   high-priority. No interaction → keep at lower priority for the role.
   Active conflict (anti-synergy keyword present in text) → reject.
4. Compare against your running short-list for this role. If the
   candidate is strictly worse than something already on the list and
   adds no new angle, replace nothing. Otherwise insert it in priority
   order.
5. Stop when you have exactly the requested count.

Do NOT spend tokens explaining your reasoning in the output — the rubric
above is the reasoning. The downstream optimizer only consumes names.

# Response constraints

- Return EXACTLY the requested count of names. Not more, not fewer.
- Each name MUST appear verbatim in the candidate list from the user
  message, including punctuation and capitalization. Card names are
  case-sensitive in the matching step downstream.
- Do not include the commander itself.
- Do not include cards already in the "already_selected" set if it is
  visible in the user message.
- Do not invent names. If you cannot find {count} suitable candidates,
  return what you have rather than padding with hallucinations.
- Do not include reasons, scores, or commentary. The schema is "names"
  only — additional fields will be silently dropped and waste tokens.

# Output format

Respond with VALID JSON ONLY. No prose before, no commentary after, no
markdown code fences. Use exactly this shape:

{
    "names": ["Card Name 1", "Card Name 2", "..."]
}

Each entry MUST be the exact card name as it appeared in the candidate list
in the user message — do not paraphrase, abbreviate, or invent names. The
list length should match the count requested in the user message."""


SYNERGY_ENGINE_PROMPT = """You are an expert Magic: The Gathering deck builder
in the EDH/Commander format. Your task here is DIFFERENT from filling a
standard role bucket like "ramp" or "removal."

The deck-builder pipeline has already filled the traditional role buckets
(ramp, draw, removal, threats, protection, recursion, lands) by picking the
most role-appropriate cards from per-role candidate pools. Those buckets
preferred cards that incidentally synergize with the commander, but they
optimized FOR THE ROLE first.

Your job is the cross-cutting synergy_engine pass: select the strategy-
defining "engine pieces" that the role-based filtering would skip because
they're weak in any single role but central to what this commander wants
to do.

# What you are looking for

The Commander Context below describes this commander's plan in detail.
Identify the recurring TRIGGER and PAYOFF patterns that define the
strategy, and pick the cheap, redundant pieces of those patterns that
nobody would draft as "best creature" or "best instant" in isolation.

Typical SHAPES by archetype (judge candidates by their rules text, not by
name recognition — the Commander Context defines this deck's strategy):

- Lifegain commander → 1-2 mana creatures or enchantments with a single
  repeatable "whenever [a creature enters / you gain life], ..." trigger.
  Weak in combat. Strategy-core.

- Token commander → pieces whose only job is to multiply token production
  (doublers, "create an additional token" effects) or pay off going wide.

- Spellslinger → cheap permanents that turn each instant/sorcery cast
  into value (a token, a ping, a treasure, a draw).

- Graveyard / recursion → self-mill enablers, cheap "return from
  graveyard" value pieces, and payoffs that scale with the graveyard.

- +1/+1 counters → counter-doublers, proliferate sources, and payoffs
  that trigger when counters are placed.

# Synergy hint tags

Each candidate's line may begin with a SYNERGY HINT TAG. These are the
output of a recall pipeline that scored each card against three
independent commander-specific signals (EDHREC community data,
embedding similarity to commander strategy text, substring pattern
match). The tag tells you how many of the three sources flagged it:

  [SYN+++]  All three sources — almost certainly commander-defining.
            Pick aggressively.
  [SYN++]   Two sources — strong synergy candidate.
  [SYN+]    One source — some commander-specific signal.
  (no tag)  No synergy signal. This card is in the pool only because
            it leaked through some other filter; usually safe to skip
            in this pass (the synergy_engine bucket is specifically for
            tagged cards).

Treat hint tags as the PRIMARY signal for ranking in this pass.

# How to choose

Pick cards from the candidate pool that:

1. Have a synergy hint tag. [SYN+++] cards are top priority; [SYN++]
   next; [SYN+] if the slate isn't already full. Untagged cards are
   usually not the right answer for this pass — leave them.

2. Mechanically slot into the commander's primary trigger or payoff
   loop. Mana cost doesn't matter — a 1-mana trigger creature and a
   5-mana payoff bomb are both valid synergy_engine picks if they're
   the recall pipeline's top picks for the strategy. The role buckets
   that ran earlier don't reliably catch high-synergy creatures
   (their heuristics rank on raw power, not on commander-fit), so
   THIS pass is where they should land.

3. Add NEW pattern coverage, not just more of the same. If your
   running list already has 3 "whenever a creature enters you gain 1
   life" creatures, the 4th is redundant — prefer a counter-doubler
   or a payoff trigger instead. Diversity within the strategy.

4. Are NOT in already_selected. The code filters those out of the
   pool before you see it, but be defensive.

# What to avoid

- Cards that have NO meaningful interaction with the commander's strategy.
  Generic value cards (a vanilla 4/4 for 4, a "draw a card" cantrip with
  no triggered-effect angle) don't belong here even though they'd be
  fine in any deck — they're better filled by the role buckets that ran
  earlier. This pass is specifically for STRATEGY-DEFINING cards.

- Cards that actively work against the strategy. Look at the
  anti_synergy_keywords list in the Commander Context — anything matching
  those is a hard skip.

Note on overlap with role buckets: the code pre-filters out anything the
role buckets already picked, so don't try to second-guess what was
selected — just pick the most strategy-defining cards available to you
from THIS pool. If an expensive strategy-payoff bomb is in your pool and
not in already_selected, that means no role bucket grabbed it — it's
fair game for you. Don't assume "the threats bucket would have picked
it" for any card you're seeing; it's only in your candidate list because
the earlier buckets DIDN'T pick it.

# How to fill the slate (worked shape)

Suppose the Commander Context describes a trigger→payoff strategy and
your pool contains (hypothetically):

  [SYN+++] a 3-mana payoff that triggers directly off the commander's text
  [SYN+++] a 5-mana bomb payoff for the same strategy
  [SYN+++] a 1-mana creature with the strategy's core repeatable trigger
  [SYN++]  a 2-mana redundant copy of that trigger
  [SYN++]  a 1-mana effect-doubler for the strategy's resource
  [SYN+]   a colorless piece with a narrower tie-in
           an untagged generic utility card — skip
           an untagged off-color card — skip

Good picks: take all the [SYN+++] cards first, then the [SYN++] cards,
then fill with [SYN+] up to count.

DO NOT skip the 5-mana bomb as "too expensive" — the recall pipeline
specifically tagged it as commander-defining. Mana cost is not your
filter here; the hint tags are.

DO skip untagged cards in this pass — they're either role-bucket
material that leaked into the pool, or generic cards with no
commander-specific signal.

# Response constraints

- Return at most `count` card names. If you can find fewer than `count`
  truly strategy-defining engine pieces, return fewer — padding with
  generic cards just dilutes the deck.
- Each name MUST appear verbatim in the candidate list from the user
  message, including punctuation and capitalization.
- Do not include any card listed in "already_selected" — those are
  already in the deck (the code already filters them out of the pool,
  but be defensive).
- Do not include the commander itself.
- Do not invent names.
- Do not include reasons, scores, or commentary. The schema is "names"
  only.

# Output format

Respond with VALID JSON ONLY. No prose before, no commentary after, no
markdown code fences. Use exactly this shape:

{
    "names": ["Card Name 1", "Card Name 2", "..."]
}

Each entry MUST be a verbatim card name from the candidate list. Aim for
the requested count but err on the side of fewer-but-better picks if the
pool genuinely doesn't have more strategy-defining engines."""


SYNERGY_SCORING_PROMPT = """You are an expert Magic: The Gathering deck builder
scoring cards for synergy with a specific commander in EDH/Commander format.

The Commander Context block below describes the commander's strategy, the
key mechanics it rewards, and the synergy keywords that flag good fits. Use
that block as the source of truth for what counts as "synergy" — do NOT
fall back to generic card power when scoring synergy.

# CALIBRATION (the most important section — read carefully)

The default failure mode for this task is clustering most scores in the
50-65 band ("might be relevant, why not?"). This is WRONG. The full 0-100
range must be used or downstream optimizers can't distinguish a strategy-
defining card from generic filler.

**Hard rule: a card's score MUST be backed by a specific, citable
mechanical connection between the card's rules text and the commander's
plan as described in the Commander Context.** No connection found = score
in the 10-30 band. Tangential connection = 30-50. Explicit synergy
keyword match = 60+. Multiple keyword matches or commander text directly
triggered = 80+.

# Synergy hint tags (when present)

If a card's line in the user prompt starts with a tag like [SYN+++],
[SYN++], or [SYN+], that tag is the output of an independent recall
pipeline that checked the card against EDHREC community data, embedding
similarity to the commander's strategy text, and substring patterns from
the commander's synergy keywords. The tags are STRONG calibration
anchors. Use them as score-band priors:

  Tag        → required score band      Why
  -----------------------------------------------------------------
  [SYN+++]   → 80-95                    Three independent sources agree
                                        this card is commander-defining.
                                        Score in the 80-95 band; reserve
                                        90+ for the cards that literally
                                        trigger the commander's text.
  [SYN++]    → 60-80                    Two of three sources flag it.
                                        Strong synergy candidate.
  [SYN+]     → 40-60                    One source. Real but weaker
                                        signal — there's a connection
                                        but it's narrower.
  (no tag)   → 10-30                    No recall source flagged the
                                        card. Default to this band UNLESS
                                        you can identify a specific
                                        mechanical interaction the
                                        recall sources missed (rare).

A card with [SYN+++] scoring 50 would mean you're disagreeing with three
independent signals — only do this if there is a strong concrete reason
(e.g. anti-synergy text overrides the tag).

# The 0-100 rubric

- 0-15   Anti-synergy. The card actively fights the strategy. Examples by
         shape (not specific cards): a "no creatures" card in a creature-
         based deck; a "give opponents lifegain" card in a lifegain-
         strategy deck (yes, it gains life — for the wrong player); a
         "discard" card in a deck that wants a full hand.

- 16-30  No commander-specific signal. The card is playable in many decks
         but neither its rules text NOR its attributes interact with this
         commander's plan. A vanilla creature with no triggered ability
         (UNLESS the commander rewards vanilla/attributes — then see the
         commander-effect rule, it belongs at 80+), a generic removal
         spell that doesn't carry a synergy rider, a mana rock with no
         tie to the strategy. THIS IS THE DEFAULT BAND FOR UNTAGGED
         CARDS. If you can't cite a specific connection — textual or
         attribute — the card belongs here.

- 31-45  Tangential connection. The card has SOME structural fit but
         doesn't trigger the commander or match a synergy keyword
         directly. A token-maker in a deck where creature count matters
         loosely; a card-draw spell in a deck that benefits from cards
         in hand at the margin.

- 46-60  Clear support. The card matches a synergy keyword from the
         Commander Context or feeds the commander's loop indirectly, but
         doesn't trigger or amplify the commander's text directly. A
         lifegain rider in a lifegain deck (gains 1 life once, not a
         repeatable trigger); a counter-producer in a +1/+1 deck that
         doesn't itself benefit from counters.

- 61-80  Strong synergy. Directly triggers the commander's text or
         repeatedly enables a primary payoff. Match multiple synergy
         keywords from the Commander Context. The kind of card someone
         building this commander would explicitly seek out.

- 81-95  Strategy-defining. The card is one of the "this commander
         exists to play this card" picks. Removing it noticeably weakens
         the deck. Multiple keyword matches AND triggers commander text
         AND repeatable.

- 96-100 Reserved for cards that are functionally inseparable from the
         commander's win condition. Use sparingly — most strategies have
         3-5 cards in this band, no more.

# Commander-effect rule (READ FIRST — overrides the examples below)

Synergy is the card's value IN PLAY once THIS commander's OWN ability is
applied to it — not merely whether the card's printed text mentions the
strategy. Many commanders add value to a card's ATTRIBUTES, not its text.
Always evaluate the RESULTING card:

- If the commander rewards an attribute this card HAS — e.g. a "vanilla
  creatures matter" commander that makes vanilla creatures unblockable or buffs
  them; a "+1/+1 counters" commander that grows small creatures; a tribal
  commander that pumps its creature type — then a card with that attribute is a
  PRIMARY PAYOFF. Score it 80+ even if its printed text is blank. A vanilla 5/5
  under such a commander is a 5/5 unblockable threat: strategy-defining, NOT
  16-30. The empty text is the point. This OVERRIDES the "vanilla creature ->
  no signal" example below.
- If the commander does NOT improve this card — e.g. a creature WITH abilities
  under a "vanilla matters" commander gets nothing from the commander — score
  it on standalone merit ONLY. Do NOT inflate it for sharing a color/type or
  being "a big creature in a beatdown deck." That is legality, not synergy.
- A card already strong WITHOUT the commander (innate evasion, etc.) is good
  and robust — score on its own merit. All else equal, prefer the card that is
  good on its own over one that NEEDS the commander to be good.

# How to score (procedure)

For each card:
1. Note its hint tag (if any). This sets the score band before you
   look at the rules text.
2. Read the rules text AND the attributes (types, subtypes, P/T, mana value,
   colors, whether it is vanilla).
3. Find the SPECIFIC connection to the commander plan — EITHER a textual one
   (keyword match, triggered ability that fires off the commander's actions,
   repeated payoff) OR an ATTRIBUTE the commander rewards (per the
   commander-effect rule). Write a one-sentence justification in your head.
4. If you can't find a specific connection — textual OR attribute — score in
   the 10-30 band regardless of how strong the card is in absolute terms.
   Generic power without synergy belongs LOW here.
5. If the card matches an anti_synergy_keyword from the Commander
   Context, drop the score below 20 even if there's surface-level fit.

# Common pitfalls to avoid

- The biggest failure mode: rationalizing "this is a creature in a
  creature deck" or "this is in-colors and playable" as moderate
  synergy. That's not synergy — that's legality. Untagged playable
  creatures with no commander interaction belong in 16-30, not 50-65.
- Don't anchor on raw mana value. A 6-mana spell with the perfect
  synergy can still score 90+; a 1-mana spell with no synergy still
  scores in the 16-30 band.
- Don't reward color-fixing or basic ramp as synergy. Mana fixing is a
  deckbuilding necessity, not a strategy contribution. Score on the
  0-100 synergy rubric only; the deck builder handles mana base
  separately.
- Don't conflate "good in the deck" with "synergy". A premium removal
  spell is good IN the deck but the synergy score should be 16-30
  unless the strategy is built around removal.
- Be willing to score below 15 for active anti-synergy. Be willing to
  score above 90 for true strategy-defining cards.

# Effect-class tagging (when the Commander Context lists core effect classes)

If the Commander Context contains a "Core effect classes" section, add a
"class" field to any card that CLEARLY fills one of those classes — using
the exact class name from that section. Most cards fill no class: omit the
field (or use null) rather than stretching. A card can only carry ONE class
(its primary function for this deck).

# Output format

Do ALL reasoning SILENTLY — do not write it out. Your response must BEGIN with
the character `{` and contain nothing before it: no preamble, no restating the
commander's abilities, no per-card explanation, no markdown fences. Output
VALID JSON ONLY, in exactly this shape:

{
    "scores": [
        {"name": "Card Name 1", "score": 78, "class": "<a Core-effect-class name>"},
        {"name": "Card Name 2", "score": 22}
    ]
}

The "name" field MUST match the card name from the user message verbatim
(including any tag prefix is OK — only the card name part needs to
match). The "score" field must be an integer from 0 to 100. The "class"
field is optional and must be an exact core-effect-class name when present.
Include every card from the user message exactly once."""


DECK_EXPLANATION_PROMPT = """You are an expert Magic: The Gathering deck builder explaining a deck.

Produce a clear, concise 2-3 paragraph explanation covering:
1. The overall strategy
2. Key synergies and win conditions
3. How to pilot the deck

Write in plain prose suitable for a player unfamiliar with the commander."""


DECK_REVIEW_PROMPT = """You are an expert Magic: The Gathering deck reviewer.

Review this optimized deck and identify:
1. What's missing (e.g., lacking enchantment removal, no answer to graveyards)
2. What's over-represented or redundant
3. 3-5 specific card swap suggestions, each with a brief reason

Respond as plain prose — this is a final review, not structured data."""


DECK_REFINEMENT_PROMPT = """You are an expert Magic: The Gathering deck builder
performing a final REFINEMENT pass on an assembled 99-card Commander deck.

The deck was assembled by an optimizer that maximizes per-card averages and
role counts. That optimizer is good at theme density but BLIND to set-level
composition — the things a competent human checks last:

1. REDUNDANCY / CONSISTENCY. Does the deck run enough copies of each effect
   its plan depends on? If the strategy needs a repeatable engine trigger
   online by turn 3, one copy is not enough — a human runs 3-5 functional
   duplicates. Look for critical effects present as singletons that the
   alternatives list could reinforce.
2. INTERACTION SPREAD. Is removal balanced across threat types (creatures
   vs artifacts/enchantments vs flexible)? A pile of narrow answers to one
   permanent type is a composition failure even if the total count is fine.
3. ROLE QUALITY. Are the ramp/draw/removal slots filled with efficient
   versions of those effects, or with weak stand-ins? Prefer cheaper,
   unconditional, or higher-rate versions from the alternatives.
4. MISSING PAYOFFS. Is a strategy-defining payoff or combo piece sitting in
   the alternatives list while a weaker card holds its slot?
5. FILLER. Cards that do little for this deck's plan and aren't functional
   role-fillers — the first cuts.

Each card line includes our precomputed signals: syn=<synergy 0-100 for this
commander> pow=<intrinsic card power 0-100>, and, where applicable,
role:<comma-separated functional roles the card counts toward>. The role
tags are how the deck's role minimums are ACTUALLY counted — a card may
hold a role floor even if its text doesn't look like that role to you.
Use the signals as evidence, not gospel — your set-level judgment is
exactly what those per-card numbers cannot express, and overriding them
WITH A REASON is your job.

# Rules for swaps

- "out" MUST be a card currently in the deck; "in" MUST come from the
  ALTERNATIVES list. Never invent names; copy them verbatim.
- Swap lands only for lands, and nonlands only for nonlands (the deck's
  land count is already tuned).
- Do NOT take a functional role below its minimum. The user message
  includes a ROLE STATUS line with current counts vs minimum targets;
  if a role is at or below its minimum, only swap those cards
  like-for-like (e.g. weak ramp for better ramp). CHECK THE role: TAG of
  the card you're cutting — for any of its roles marked AT FLOOR, the
  incoming card must carry the same role tag. Swaps that would drop a
  role below its floor are mechanically rejected anyway — don't waste
  your swap budget on them.
- Do not remove the commander (it is not in the 99) or any card marked
  [LOCKED].
- Suggest a swap ONLY when it clearly improves the deck's function. Fewer,
  higher-confidence swaps beat many marginal ones. Zero swaps is a valid
  answer for a well-composed deck.

# Output format

Respond with VALID JSON ONLY. No prose before or after, no markdown fences:

{
    "swaps": [
        {"out": "Card In Deck", "in": "Card From Alternatives", "reason": "<15 words max>"}
    ]
}"""


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for LLM engine."""
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.3
    api_key: Optional[str] = None

    # v0.9.4 (runtime lever): cheaper model for the elimination ROUNDS of
    # the card-selection tournament (synergy_engine + any large role pool).
    # Those rounds are coarse "keep the top half of this chunk" filtering
    # where Haiku is plenty; the FINAL precision pick still uses `model`
    # (Sonnet). Set to None to use `model` for every round.
    tournament_model: Optional[str] = "claude-haiku-4-5"

    # v0.2 additions
    mock_mode: bool = False
    """When True, uses deterministic heuristic responses instead of calling the API.
    Great for unit tests and CI."""

    use_prompt_caching: bool = True
    """Enable prompt caching on system prompts. Recommended for cost savings."""

    cache_ttl: str = "1h"
    """v0.9.19: cache-entry TTL, "5m" or "1h". The Jodah run measured Sonnet at
    19% hit rate because the 5-minute TTL expired during the 33-minute Haiku
    tournament between Sonnet phases; 1h spans a whole build (and the user's
    ~20-min iteration cadence). Writes cost 2x base vs 1.25x for 5m — a few
    cents on the ~17K tokens a build writes, repaid by one avoided re-write."""

    cache_pad_to_minimum: bool = True

    analysis_cache_dir: Optional[str] = "./analysis_cache"
    """v0.9.32: persist commander analyses to disk. The analysis was cached
    only in-memory, so every run re-generated it — and its variance (strategy
    text, keywords, category queries) cascaded into recall, hint tags, and
    effect classes, defeating the per-commander synergy cache across
    sessions. Keyed on the FULL prompt hash, so commander-text changes (DB
    refresh) and prompt edits invalidate automatically. None disables."""
    """v0.9.19: models silently refuse to cache prefixes below their minimum
    (Haiku 4.5: 4096 tokens, Sonnet: 2048). The Jodah run measured Haiku at 0%
    hit rate across 225 tournament calls with an identical ~2.9K-token prefix
    — all billed at full price. When True, a deterministic filler block pads
    the prefix past the minimum so it caches (pad rides at the 0.1x cached
    rate after the first write; measured saving ~$0.5/five-color build)."""


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class LLMEngine:
    """
    Wrapper for Anthropic Claude API with prompt caching and mock mode.

    Usage:
        engine = LLMEngine()
        analysis = engine.analyze_commander(commander_card)
        selections = engine.select_cards(analysis, candidates, role="ramp", count=15)
        scores = engine.score_synergy_batch(analysis, cards)

        # For testing without API:
        engine = LLMEngine(LLMConfig(mock_mode=True))
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._cache: dict[str, object] = {}
        # One-shot guard so we only warn once per build if the system prefix
        # is below the model's minimum cacheable size.
        self._warned_cache_silent = False
        # v0.9.16c: per-model cache accounting for the end-of-build summary.
        self._cache_stats: dict[str, dict] = {}

        if self.config.mock_mode:
            self.client = None
            logger.info("LLMEngine in MOCK MODE (no API calls will be made)")
        else:
            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning(
                    "No ANTHROPIC_API_KEY found; falling back to mock mode. "
                    "Set ANTHROPIC_API_KEY to use the real API."
                )
                self.config.mock_mode = True
                self.client = None
            else:
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=api_key)
                except ImportError:
                    logger.error("anthropic package not installed; using mock mode")
                    self.config.mock_mode = True
                    self.client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_commander(self, commander: Card) -> CommanderAnalysis:
        """Analyze a commander and return structured guidance."""
        cache_key = f"analysis_{commander.name}"
        if cache_key in self._cache:
            logger.debug(f"Using cached analysis for {commander.name}")
            return self._cache[cache_key]  # type: ignore

        if self.config.mock_mode:
            analysis = self._mock_analyze_commander(commander)
            self._cache[cache_key] = analysis
            return analysis

        prompt = f"""Analyze this EDH commander:

{commander.format_for_llm()}

Respond with JSON matching this EXACT schema:
{{
    "name": "{commander.name}",
    "color_identity": "{commander.color_identity}",
    "key_mechanics": ["..."],
    "build_around_text": "2-3 sentences describing the strategy",
    "evaluation_notes": "How should cards be evaluated differently for this commander? What's normally bad that's good here?",
    "category_queries": {{
        "synergy_core": "space-separated search keywords",
        "synergy_support": "space-separated search keywords",
        "payoffs": "space-separated search keywords"
    }},
    "synergy_keywords": ["keyword or phrase 1", "keyword or phrase 2"],
    "synergy_patterns": ["gain life", "lifelink", "+1/+1 counter", "creature token", "..."],
    "structural_predicates": [],
    "core_effect_classes": [
        {{"name": "short effect-class name", "min_count": 4}},
        {{"name": "another effect class", "min_count": 5}}
    ],
    "anti_synergy_keywords": ["things to avoid"],
    "recommended_weights": {{
        "mana_curve": 0.10,
        "role_coverage": 0.15,
        "synergy": 0.35,
        "strategy_density": 0.20,
        "power_level": 0.20
    }},
    "recommended_synergy_weight": 0.6
}}

Guidance for recommended_weights: adjust the defaults above based on HOW MUCH this
commander warps normal card evaluation. A commander whose payoff inverts normal
card quality (e.g. "vanilla creatures matter") should push synergy +
strategy_density up and power_level down. A generic goodstuff commander should
push power_level up and strategy_density down (because there's no "strategy"
to be dense on). Use ONLY the five dimensions shown above (no "creativity" —
it is not scored). Weights should sum to 1.0 — any weight you take from one
dimension must go to another.

`strategy_density` is a new dimension that counts the fraction of non-mana-land
cards strongly on-strategy. It's most useful for commanders with a recognizable
build-around theme (lifegain, tokens, +1/+1 counters, spellslinger, graveyard,
etc.) — push it to 0.20-0.30. For goodstuff commanders without a coherent
theme push it down to 0.05-0.10.

Guidance for recommended_synergy_weight: a 0-1 value. 0.8+ for commanders that
massively invert evaluation (attribute-matters archetypes). 0.5-0.7 for strong
single-theme commanders (lifegain, tokens, spellslinger). 0.3-0.5 for goodstuff
commanders. This is the weight given to commander-specific synergy vs raw card
power.

Guidance for structural_predicates: USUALLY EMPTY. Only fill this when the
commander's payoff is a structural ATTRIBUTE of cards rather than their rules
text — because text-based matching is blind to such cards. The classic case is
"vanilla creatures matter": a vanilla creature has NO text, so
synergy_patterns can never match it; emit ["vanilla"] instead. Bounded
vocabulary (use only these forms):
  vanilla | no_abilities | colorless | creature | land
  subtype:Bear | type:Creature | supertype:Legendary | keyword:Trample
  mv<=2 | cmc>=6 | power>=4 | toughness<=1   (operators: <= >= == < >)
Examples: vanilla matters -> ["vanilla"]; colorless/Eldrazi -> ["colorless"];
low-curve aggro payoff -> ["mv<=2"]; big-creatures matter -> ["power>=4"].
TRIBAL commanders (Vampires/Dragons/Elves/Slivers matter) SHOULD emit the type
as a predicate -> ["subtype:Vampire"]. The tribe name as a text keyword only
catches lords/payoffs that spell it out; the predicate also guarantees plain
and French-vanilla tribe members (whose only tribal signal is the type line)
are recalled and scored. Use the tribe's exact creature subtype.
For normal text-defined commanders (lifegain, tokens, spellslinger, etc.)
leave this an empty list — synergy_patterns already covers them.

Guidance for core_effect_classes: the 4-8 effect CLASSES this strategy needs
REDUNDANT copies of to function consistently — the packages a competent human
would deliberately run multiples of. Think consistency: which effects must be
online reliably by the mid-game, and how many copies does that take? Each
entry has a short reusable "name" describing a FUNCTION, not a card name —
derive the classes from THIS commander's plan (engine triggers, payoffs,
enablers, doublers) plus whatever interaction shape this strategy leans on
(e.g. "creature removal" for a deck that must keep boards clear, "graveyard
enabler" for a recursion deck). "min_count" (1-8) is how many copies a
well-built deck runs. Include the strategy's engine class(es), its payoff
class(es), and any support class that must not be a singleton. The per-card
scoring pass will tag candidate cards with these exact class names, so keep
names short, distinct, and functional.

Guidance for synergy_patterns: short substring patterns (1-4 words each)
that appear inside actual MTG card rules text and indicate a card is
relevant to this commander's strategy. The downstream matcher applies
case-insensitive substring matching after normalizing standalone digits
and standalone X tokens out of card text — so "gain life" will correctly
match "gain 1 life" / "gain X life", and "creature token" will match
"1/1 white Soldier creature token". You do NOT need to enumerate numeric
variants yourself. DO include:

- All MTG ability keywords relevant to the strategy (lifelink, vigilance,
  proliferate, scry, surveil, etc.) as bare single-word patterns.
- Common rules-text fragments the strategy depends on (e.g. for lifegain:
  "gain life", "you gained", "lifelink"; for tokens: "creature token",
  "create a", "populate"; for spellslinger: "whenever you cast",
  "instant or sorcery").
- Mechanical phrases that are quasi-tribal (e.g. "+1/+1 counter",
  "graveyard", "discard a card", "exile target").
- Avoid full sentences and avoid patterns longer than ~4 words —
  longer patterns fail to match when the card has slight variations.

Aim for 8-15 patterns covering both the synergy_core mechanics and the
secondary support mechanics. Each pattern catches a CLASS of cards.
Patterns ARE allowed to overlap — overlap is harmless because the matcher
deduplicates."""

        # v0.9.32: disk cache — the full prompt hash covers both prompt
        # edits AND commander-text changes (the card text is embedded in
        # the user prompt), so a hit is guaranteed to be the answer to the
        # SAME question.
        prompt_hash = hashlib.sha256(
            (COMMANDER_ANALYSIS_PROMPT + "\x00" + prompt).encode("utf-8")
        ).hexdigest()[:16]
        disk = self._load_analysis_cache(commander.name, prompt_hash)
        if disk is not None:
            logger.info(f"Analysis cache: reused stored analysis for "
                        f"{commander.name}")
            self._cache[cache_key] = disk
            return disk

        response = self._call_api(
            system_prompt=COMMANDER_ANALYSIS_PROMPT,
            user_prompt=prompt,
            temperature=0.3,
        )

        data = self._parse_json_defensively(response) or {}
        analysis = self._build_analysis_from_dict(commander, data)
        self._cache[cache_key] = analysis
        self._save_analysis_cache(commander.name, prompt_hash, analysis)
        return analysis

    # -- v0.9.32: commander-analysis disk cache -------------------------

    def _analysis_cache_path(self, commander_name: str) -> Optional[str]:
        cache_dir = getattr(self.config, "analysis_cache_dir", None)
        if not cache_dir:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9._-]", "_",
                      f"{commander_name}_{self.config.model}")
        return os.path.join(cache_dir, f"analysis_{slug}.json")

    def _load_analysis_cache(self, commander_name: str,
                             prompt_hash: str) -> Optional[CommanderAnalysis]:
        path = self._analysis_cache_path(commander_name)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("prompt_hash") != prompt_hash:
                return None  # prompt or commander text changed → re-analyze
            # Schema-defensive reconstruction: unknown fields (from a newer
            # version) are dropped; missing required fields raise and the
            # entry is treated as a miss.
            valid = {f.name for f in dataclasses.fields(CommanderAnalysis)}
            payload = {k: v for k, v in data.get("analysis", {}).items()
                       if k in valid}
            return CommanderAnalysis(**payload)
        except Exception as e:
            logger.warning(f"Analysis cache read failed ({path}): {e}")
            return None

    def _save_analysis_cache(self, commander_name: str, prompt_hash: str,
                             analysis: CommanderAnalysis) -> None:
        path = self._analysis_cache_path(commander_name)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"prompt_hash": prompt_hash,
                           "analysis": dataclasses.asdict(analysis)},
                          f, indent=1)
        except Exception as e:
            logger.warning(f"Analysis cache write failed ({path}): {e}")

    # ------------------------------------------------------------------
    # Card selection (with multi-round batching for large pools)
    # ------------------------------------------------------------------

    # The LLM's empirical comfort zone for "rank these cards" tasks is
    # ~150 entries — past that it starts dropping middle items and
    # overweighting the head/tail. We use this as the chunk size for the
    # elimination tournament's per-round splits.
    SELECT_CARDS_CHUNK_SIZE = tuning.SELECT_CARDS_CHUNK_SIZE

    # Tournament termination threshold: pools at or below this size take a
    # single LLM call instead of another elimination round. Set at 2x the
    # chunk-size sweet spot — Sonnet 4.6 still handles 300-card lists
    # reliably, and stopping the tournament early avoids the asymmetric
    # shrinkage that happens when the last chunk is smaller than
    # per_chunk_keep (e.g. 192 cards → 2 chunks of 150+42 → 75+42 = 117
    # survivors, undershooting count=150 by 33 cards).
    SELECT_CARDS_MAX_SINGLE_PASS = tuning.SELECT_CARDS_MAX_SINGLE_PASS

    # Selection modes — controls system prompt + user prompt template.
    # "role" is the standard per-role filter; "synergy_engine" is the
    # Phase 2 pass that grabs strategy-defining cross-cutting engine pieces.
    _MODE_ROLE = "role"
    _MODE_SYNERGY_ENGINE = "synergy_engine"

    def select_cards(
        self,
        analysis: CommanderAnalysis,
        candidates: list[Card],
        role: str,
        count: int = 20,
        already_selected: Optional[set[str]] = None,
        mode: str = "role",
        synergy_hints: Optional[dict[str, str]] = None,
    ) -> list[str]:
        """
        Select top N cards from candidates for a specific role.

        Dispatcher: small pools take a single LLM call (the original
        behavior); large pools (e.g. the v0.8 unioned synergy pool, up to
        2500 cards) get a multi-round elimination tournament so the LLM
        actually considers every card rather than seeing only the first
        SELECT_CARDS_CHUNK_SIZE.

        `mode` chooses the prompt pair:
          - "role" (default): traditional role bucket — CARD_SELECTION_PROMPT
            with per-role guidance, synergy as tiebreaker.
          - "synergy_engine": Phase 2 cross-cutting pass — SYNERGY_ENGINE_PROMPT
            asking for strategy-defining engine pieces, "fewer-but-better"
            target.

        `synergy_hints` (v0.9.1): optional dict of card.name → hint tag
        ("[SYN+++]", "[SYN++]", "[SYN+]"). When provided, each candidate's
        user-prompt line is prefixed with its hint so the LLM weights
        commander-specific cards above generic role-fit. The hint tag
        encoding is explained in the system prompts.

        Same return contract: list[str] of card names, length ≤ count.
        """
        if not candidates:
            return []

        already_selected = already_selected or set()
        candidates = [c for c in candidates if c.name not in already_selected]

        if len(candidates) <= count:
            return [c.name for c in candidates]

        if self.config.mock_mode:
            return self._mock_select_cards(analysis, candidates, role, count)

        if len(candidates) <= self.SELECT_CARDS_MAX_SINGLE_PASS:
            return self._select_cards_chunk(
                analysis, candidates, role, count,
                mode=mode, synergy_hints=synergy_hints,
            )

        # Pool too large for a single LLM call — run elimination rounds.
        return self._select_cards_batched(
            analysis, candidates, role, count,
            mode=mode, synergy_hints=synergy_hints,
        )

    def select_synergy_engine_cards(
        self,
        analysis: CommanderAnalysis,
        candidates: list[Card],
        count: int = 25,
        already_selected: Optional[set[str]] = None,
        synergy_hints: Optional[dict[str, str]] = None,
    ) -> list[str]:
        """
        Phase 2 cross-cutting synergy pass.

        Wrapper around select_cards that hands the LLM a different prompt
        pair — one that asks for strategy-defining engine pieces (Soul
        Sister equivalents, cheap repeatable triggers, payoff
        enchantments) rather than role-fit. Designed to run AFTER all
        traditional role buckets have filled, picking up the cards that
        the role filter would skip because they're individually weak.
        """
        return self.select_cards(
            analysis=analysis,
            candidates=candidates,
            role="synergy_engine",
            count=count,
            already_selected=already_selected,
            mode=self._MODE_SYNERGY_ENGINE,
            synergy_hints=synergy_hints,
        )

    def _build_select_cards_user_prompt(
        self,
        candidates: list[Card],
        role: str,
        count: int,
        mode: str,
        synergy_hints: Optional[dict[str, str]] = None,
    ) -> str:
        """Build the per-call user prompt for either selection mode."""
        hints = synergy_hints or {}
        lines = []
        for c in candidates:
            tag = hints.get(c.name)
            if tag:
                lines.append(f"{tag} {c.format_for_llm()}")
            else:
                lines.append(c.format_for_llm())
        formatted = "\n".join(lines)
        if mode == self._MODE_SYNERGY_ENGINE:
            return f"""Pick up to {count} strategy-defining engine cards from
this pool. Refer to the Commander Context AND the SYNERGY_ENGINE instructions
in the system prompt for what counts as an engine piece.

Candidates ({len(candidates)} cards):
{formatted}

Respond with JSON containing ONLY the names — no reasons, no commentary:
{{
    "names": ["Card Name 1", "Card Name 2", ...]
}}

Return up to {count} names that appear verbatim in the candidate list above.
Fewer is fine if the pool doesn't actually contain enough true engine pieces —
quality over quantity at this stage."""
        # default "role" mode
        return f"""Select the top {count} cards from this pool for the "{role}" role.

Refer to the Commander Context in the system prompt for strategy, evaluation
notes, and synergy keywords. Apply the cross-cutting synergy preference
described there: prefer synergistic cards as a tiebreaker among role-fit
candidates, but keep functionally-necessary role cards (efficient ramp,
removal, fixing) even when untagged.

Candidates ({len(candidates)} cards):
{formatted}

Respond with JSON containing ONLY the names — no reasons, no commentary:
{{
    "names": ["Card Name 1", "Card Name 2", ...]
}}

Select exactly {count} card names that appear verbatim in the candidate list above."""

    def _select_cards_chunk(
        self,
        analysis: CommanderAnalysis,
        candidates: list[Card],
        role: str,
        count: int,
        mode: str = "role",
        synergy_hints: Optional[dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> list[str]:
        """
        One LLM call: pick top `count` from `candidates`.

        Caller guarantees len(candidates) > count (otherwise no selection
        is needed) and len(candidates) <= SELECT_CARDS_MAX_SINGLE_PASS.
        Within that range the LLM reliably picks the requested count
        without dropping middle items.

        `model` overrides the engine's default model for this call —
        used to run elimination rounds on a cheaper model.
        """
        system_prompt = (
            SYNERGY_ENGINE_PROMPT if mode == self._MODE_SYNERGY_ENGINE
            else CARD_SELECTION_PROMPT
        )
        prompt = self._build_select_cards_user_prompt(
            candidates, role, count, mode, synergy_hints=synergy_hints,
        )

        # Budget: ~10 tokens per card name + JSON overhead.
        budget = max(2048, count * 40)

        response = self._call_api(
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=budget,
            commander_context=self._format_commander_context(analysis),
            model=model,
        )

        data = self._parse_json_defensively(response) or {}

        # Accept either the new lean schema {"names": [...]} or the legacy
        # {"selections": [{"name": ..., "reason": ...}]} schema in case the
        # model falls back to its training-data format.
        raw_names: list[str] = []
        if isinstance(data.get("names"), list):
            raw_names = [n for n in data["names"] if isinstance(n, str)]
        elif isinstance(data.get("selections"), list):
            raw_names = [
                s["name"] for s in data["selections"]
                if isinstance(s, dict) and isinstance(s.get("name"), str)
            ]

        # Canonicalize back to the exact candidate names (strip hint-tag
        # echoes, fix case drift) and dedupe — the raw LLM spelling would
        # fail the exact-case dict lookups downstream and silently drop
        # the pick.
        lower_to_canonical = {c.name.lower(): c.name for c in candidates}
        valid_names: list[str] = []
        seen: set[str] = set()
        for n in raw_names:
            canonical = self._canonicalize_card_name(n, lower_to_canonical)
            if canonical is not None and canonical not in seen:
                seen.add(canonical)
                valid_names.append(canonical)
        return valid_names[:count]

    def _select_cards_batched(
        self,
        analysis: CommanderAnalysis,
        candidates: list[Card],
        role: str,
        count: int,
        mode: str = "role",
        synergy_hints: Optional[dict[str, str]] = None,
    ) -> list[str]:
        """
        Multi-round elimination tournament for pools too large for a
        single LLM call.

        Each round splits the current pool into chunks of at most
        SELECT_CARDS_CHUNK_SIZE. Each chunk asks the LLM to keep the top
        half. Survivors get unioned. We recurse until the survivor pool
        fits in SELECT_CARDS_MAX_SINGLE_PASS, then a final single LLM
        call picks exactly `count`.

        The "top half" rule converges fast (~5 rounds for a 2500-card
        pool with chunk=150) without being so aggressive that good cards
        get dropped early. Every card the user enabled via candidate
        recall reaches the LLM at least once.

        Termination at MAX_SINGLE_PASS (2x chunk_size) rather than
        chunk_size matters: pools in the 150-300 range used to recurse
        one more round and lose 20-30 cards to asymmetric shrinkage
        (when the last chunk was smaller than per_chunk_keep it passed
        through whole, undershooting `count`). Stopping early and doing
        a single slightly-enlarged LLM pick is both cheaper AND returns
        exactly `count`.
        """
        chunk_size = self.SELECT_CARDS_CHUNK_SIZE
        # Per-chunk survival count: top half. This is the shrinkage rate;
        # capping at `count` would disable filtering when count equals
        # chunk_size (which causes infinite recursion). The final-count
        # cut happens in the base case via _select_cards_chunk.
        per_chunk_keep = max(chunk_size // 2, 1)

        chunks = [
            candidates[i:i + chunk_size]
            for i in range(0, len(candidates), chunk_size)
        ]

        logger.info(
            f"select_cards[{role}]: batched round, {len(candidates)} "
            f"candidates → {len(chunks)} chunks (chunk_size={chunk_size}, "
            f"per_chunk_keep={per_chunk_keep})"
        )

        # Elimination rounds run on the cheaper tournament model (coarse
        # "keep the top half" filtering); the final precision pick uses
        # the engine's default model. tournament_model=None disables this.
        round_model = self.config.tournament_model

        nominees: list[Card] = []
        seen: set[str] = set()
        for idx, chunk in enumerate(chunks):
            if len(chunk) <= per_chunk_keep:
                # Whole chunk survives — no LLM call needed.
                kept_names = [c.name for c in chunk]
            else:
                kept_names = self._select_cards_chunk(
                    analysis, chunk, role, per_chunk_keep,
                    mode=mode, synergy_hints=synergy_hints,
                    model=round_model,
                )
            chunk_index = {c.name: c for c in chunk}
            for name in kept_names:
                if name in seen:
                    continue
                card = chunk_index.get(name)
                if card is None:
                    continue
                seen.add(name)
                nominees.append(card)

        logger.info(
            f"select_cards[{role}]: round produced {len(nominees)} "
            f"nominees (target={count}, chunk_size={chunk_size}, "
            f"round_model={round_model or self.config.model})"
        )

        if len(nominees) <= count:
            # Tournament eliminated more than expected; just return what's left.
            return [c.name for c in nominees]

        if len(nominees) <= self.SELECT_CARDS_MAX_SINGLE_PASS:
            # Final round: one slightly-enlarged LLM call on the DEFAULT
            # (higher-quality) model to pick `count`. Better than recursing,
            # which would over-prune the last 150-300 cards via asymmetric
            # chunk shrinkage.
            return self._select_cards_chunk(
                analysis, nominees, role, count,
                mode=mode, synergy_hints=synergy_hints,
                model=None,  # default model for the precision pick
            )

        # Still too large — recurse for another elimination round.
        return self._select_cards_batched(
            analysis, nominees, role, count,
            mode=mode, synergy_hints=synergy_hints,
        )

    def score_synergy_batch(
        self,
        analysis: CommanderAnalysis,
        cards: list[Card],
        batch_size: int = 30,
        synergy_hints: Optional[dict[str, str]] = None,
        class_sink: Optional[dict[str, str]] = None,
    ) -> dict[str, float]:
        """
        Score synergy for multiple cards. Returns dict name -> 0-100.

        `synergy_hints` (v0.9.2): optional dict of card.name → hint tag.
        When provided, each card's line in the scoring user prompt is
        prefixed with its hint tag and the system prompt instructs the
        LLM to use the tag as a strong calibration anchor. This fixes
        the "filler scores 50-60" problem where the LLM clusters generic
        cards in the moderate band when it lacks an external anchor.

        `class_sink` (v0.9.14): optional dict the caller provides; when the
        analysis declared core_effect_classes, the LLM tags each card with
        the class it fills and those tags are written into this dict
        (card name -> class name). Feeds the consistency dimension.
        """
        if not cards:
            return {}

        all_scores: dict[str, float] = {}
        for i in range(0, len(cards), batch_size):
            batch = cards[i : i + batch_size]
            scores = self._score_synergy_single(
                analysis, batch, synergy_hints=synergy_hints,
                class_sink=class_sink,
            )
            all_scores.update(scores)

        return all_scores

    def explain_deck(self, deck: Deck, analysis: CommanderAnalysis) -> str:
        """Generate a prose explanation of the deck."""
        if self.config.mock_mode:
            return (
                f"[MOCK] {analysis.name} deck focused on "
                f"{', '.join(analysis.key_mechanics or ['its core strategy'])}. "
                f"The deck runs {len(deck.cards)} cards. "
                f"{analysis.build_around_text}"
            )

        card_list = deck.to_decklist()
        prompt = f"""Explain this {analysis.name} EDH deck.

Refer to the Commander Context in the system prompt for strategy.

Decklist:
{card_list}

Provide a 2-3 paragraph explanation covering strategy, key synergies, and piloting tips."""

        return self._call_api(
            system_prompt=DECK_EXPLANATION_PROMPT,
            user_prompt=prompt,
            temperature=0.5,
            commander_context=self._format_commander_context(analysis),
        )

    def review_deck(self, deck: Deck, analysis: CommanderAnalysis) -> str:
        """Run one LLM-based review pass identifying gaps and suggesting swaps."""
        if self.config.mock_mode:
            return (
                f"[MOCK REVIEW] The {analysis.name} deck appears well-constructed with "
                f"{len(deck.cards)} cards. Consider running more targeted removal and "
                f"ensuring enough synergy pieces for {', '.join(analysis.key_mechanics or [])}. "
                f"(This is a mock response.)"
            )

        card_list = deck.to_decklist()
        prompt = f"""Review this {analysis.name} deck for improvements.

Refer to the Commander Context in the system prompt for strategy and
evaluation notes.

Decklist:
{card_list}

Identify what's missing, what's over-represented, and suggest 3-5 specific swaps with reasons."""

        return self._call_api(
            system_prompt=DECK_REVIEW_PROMPT,
            user_prompt=prompt,
            temperature=0.4,
            commander_context=self._format_commander_context(analysis),
        )

    def refine_deck_swaps(
        self,
        analysis: CommanderAnalysis,
        deck: Deck,
        alternatives: list[Card],
        synergy: Optional[dict[str, float]] = None,
        power: Optional[dict[str, float]] = None,
        max_swaps: int = 8,
        locked: Optional[set[str]] = None,
        role_status: Optional[str] = None,
        card_roles: Optional[dict[str, list[str]]] = None,
    ) -> list[dict]:
        """v0.9.14: holistic refinement pass over the ASSEMBLED deck.

        Hands the LLM the actual 99 plus the best unused pool alternatives
        (both annotated with our synergy/power signals) and asks for concrete
        swaps judged on set-level composition — redundancy, interaction
        spread, role quality — the properties the GA's per-card averages
        cannot express. Returns a list of {"out","in","reason"} dicts with
        names canonicalized; mechanical validation (duplicates, land parity)
        is the caller's job. Empty list in mock mode or on parse failure.

        card_roles (v0.9.28) maps card name -> tracked role memberships.
        Without it, the LLM knows a role is AT FLOOR but not WHICH cards
        hold that floor — observed: 6 of 7 round-2 proposals rejected
        because cards the classifier counts as 'protection' (Bladebrand,
        Toxin Analysis) read as cuttable filler. Tagging makes the floor
        rule actionable and same-role upgrades recognizable.
        """
        if self.config.mock_mode or not deck or not deck.cards:
            return []

        synergy = synergy or {}
        power = power or {}
        locked = locked or set()
        card_roles = card_roles or {}

        def _line(card: Card) -> str:
            s = synergy.get(card.name)
            p = power.get(card.name)
            sig = (
                f"syn={s:.0f} pow={p:.0f}" if s is not None and p is not None
                else f"syn={s:.0f}" if s is not None
                else f"pow={p:.0f}" if p is not None
                else "no-signal"
            )
            roles = card_roles.get(card.name)
            if roles:
                sig += f" role:{','.join(roles)}"
            lock = " [LOCKED]" if card.name in locked else ""
            return f"[{sig}]{lock} {card.format_for_llm()}"

        deck_lines = "\n".join(_line(c) for c in deck.cards)
        alt_lines = "\n".join(_line(c) for c in alternatives)

        status_block = (
            f"\nROLE STATUS (current count vs minimum target — do not drop a "
            f"role below its floor):\n{role_status}\n"
            if role_status else ""
        )
        prompt = f"""Commander: {analysis.name}
{status_block}
DECK (the current 99):
{deck_lines}

ALTERNATIVES (unused candidates you may swap IN):
{alt_lines}

Propose up to {max_swaps} swaps per the system instructions. Respond with
JSON only: {{"swaps": [{{"out": "...", "in": "...", "reason": "..."}}]}}"""

        response = self._call_api(
            system_prompt=DECK_REFINEMENT_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=2000,
            commander_context=self._format_commander_context(analysis),
        )
        data = self._parse_json_defensively(response) or {}
        raw = data.get("swaps")
        if not isinstance(raw, list):
            return []

        deck_map = {c.name.lower(): c.name for c in deck.cards}
        alt_map = {c.name.lower(): c.name for c in alternatives}
        out_swaps: list[dict] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            out_name = self._canonicalize_card_name(entry.get("out"), deck_map)
            in_name = self._canonicalize_card_name(entry.get("in"), alt_map)
            if out_name is None or in_name is None:
                continue
            if out_name in locked:
                continue
            out_swaps.append({
                "out": out_name,
                "in": in_name,
                "reason": str(entry.get("reason", ""))[:200],
            })
            if len(out_swaps) >= max_swaps:
                break
        return out_swaps

    def quick_synergy_check(self, commander: Card, card: Card) -> float:
        """
        Fast heuristic synergy (0-100) without an API call.

        Used when we need synergy for a card not in the LLM cache.
        """
        score = 45.0  # neutral baseline

        commander_text = (commander.text or "").lower()
        card_text = (card.text or "").lower()

        # Basic pattern matching across commander/card text.
        # Patterns use plain substrings rather than regex to avoid
        # having to escape MTG-specific characters like '+' in "+1/+1".
        # (commander_substring, card_substring, bonus)
        bonus_patterns = [
            ("gain life", "gain", 18),
            ("lifelink", "lifelink", 15),
            ("+1/+1 counter", "+1/+1 counter", 15),
            ("proliferate", "+1/+1 counter", 15),
            ("creature token", "token", 10),
            ("whenever", "enters the battlefield", 10),
            ("dies", "dies", 12),
            ("graveyard", "graveyard", 10),
            ("artifact", "artifact", 8),
            ("enchantment", "enchantment", 8),
        ]

        for commander_str, card_str, bonus in bonus_patterns:
            if commander_str in commander_text and card_str in card_text:
                score += bonus

        # Tribal synergy: if commander has a subtype that's in card text
        if commander.subtypes:
            for subtype in commander.subtypes.split(","):
                subtype = subtype.strip().lower()
                if subtype and len(subtype) > 3 and subtype in card_text:
                    score += 8

        return max(0.0, min(100.0, score))

    def cache_summary(self) -> Optional[str]:
        """v0.9.16c: human-readable cache-efficiency summary for the build,
        or None if no real API calls were made. Hit rate is measured as
        Anthropic does: cache_read / (cache_read + fresh_input) — the share
        of input tokens served from cache. Broken out per model so the
        Haiku-tournament (which can't cache our prefix) is visible."""
        if not self._cache_stats:
            return None
        lines = ["Prompt cache summary (this build):"]
        tot_read = tot_fresh = tot_create = 0
        for model, st in sorted(self._cache_stats.items()):
            read, fresh, create = (
                st["cache_read"], st["fresh_in"], st["cache_create"])
            tot_read += read
            tot_fresh += fresh
            tot_create += create
            denom = read + fresh
            rate = (read / denom * 100.0) if denom else 0.0
            lines.append(
                f"  {model}: {st['calls']} calls, hit rate {rate:.0f}% "
                f"(read {read:,} / fresh {fresh:,} / wrote {create:,})"
            )
        denom = tot_read + tot_fresh
        overall = (tot_read / denom * 100.0) if denom else 0.0
        lines.append(f"  OVERALL hit rate: {overall:.0f}%")
        return "\n".join(lines)

    # v0.9.19: minimum cacheable prefix per model family. Below these the API
    # silently ignores cache_control (measured: Haiku 0% across 225 identical
    # tournament prefixes in the Jodah build).
    _CACHE_MIN_TOKENS = {"haiku": 4096}
    _CACHE_MIN_DEFAULT = 2048
    # One sentence ≈ 23 tokens, repeated as needed. Deterministic: identical
    # bytes every call, so the padded prefix cache-hits.
    _CACHE_PAD_SENTENCE = (
        "This block is deliberate padding to satisfy the minimum cacheable "
        "prefix size of the prompt cache; it carries no instructions and "
        "should be ignored entirely. "
    )

    def _cache_pad_block(self, blocks: list, model: str) -> Optional[dict]:
        """Return a filler system block that lifts the prefix past the
        model's minimum cacheable size, or None if it's already over.

        Token counts are estimated at ~4 chars/token, then padded to 1.4x the
        minimum: overshooting costs fractions of a cent (pad tokens ride at
        the 0.1x cached-read rate after the first write), while undershooting
        silently disables caching for the whole call — so err high.
        """
        if not getattr(self.config, "cache_pad_to_minimum", True):
            return None
        family_min = self._CACHE_MIN_DEFAULT
        for family, mn in self._CACHE_MIN_TOKENS.items():
            if family in (model or ""):
                family_min = mn
                break
        est_tokens = sum(len(b.get("text", "")) for b in blocks) // 4
        target = int(family_min * 1.4)
        if est_tokens >= target:
            return None
        deficit_chars = (target - est_tokens) * 4
        reps = deficit_chars // len(self._CACHE_PAD_SENTENCE) + 1
        return {"type": "text",
                "text": "[CACHE ALIGNMENT PADDING — IGNORE THIS BLOCK]\n"
                        + self._CACHE_PAD_SENTENCE * reps}

    # ------------------------------------------------------------------
    # Internal: API calls
    # ------------------------------------------------------------------

    @staticmethod
    def _format_commander_context(analysis: CommanderAnalysis) -> str:
        """
        Render the per-commander stable context block.

        Field order is fixed so byte-identical output across calls — that's
        the prefix-match invariant prompt caching relies on. List fields
        (synergy_keywords, key_mechanics, anti_synergy_keywords) are emitted
        in the order they came back from the analysis JSON, which is itself
        cached for the build.
        """
        parts = ["# Commander Context",
                 "",
                 "This block is identical across every card-selection and",
                 "synergy-scoring call in this build. Use it as the source",
                 "of truth for what this commander wants.",
                 "",
                 f"## Commander: {analysis.name}",
                 f"Color identity: {analysis.color_identity}"]

        if analysis.build_around_text:
            parts += ["", "## Strategy", analysis.build_around_text]

        if analysis.evaluation_notes:
            parts += ["",
                      "## How this commander warps card evaluation",
                      "(Cards that look mediocre in a vacuum may be excellent",
                      "here; cards that look strong may be irrelevant. Apply",
                      "this lens to every scoring decision below.)",
                      "",
                      analysis.evaluation_notes]

        if analysis.key_mechanics:
            parts += ["", "## Key mechanics"]
            parts += [f"- {m}" for m in analysis.key_mechanics]

        if analysis.synergy_keywords:
            parts += ["",
                      "## Synergy keywords",
                      "Cards whose rules text matches any of these keywords",
                      "are likely strong fits. The deeper the match, the",
                      "stronger the synergy:"]
            parts += [f"- {k}" for k in analysis.synergy_keywords]

        if analysis.synergy_patterns:
            parts += ["",
                      "## Synergy text patterns",
                      "Short substring patterns (post digit/X normalization)",
                      "the candidate-recall layer used to source cards. If a",
                      "card was added because of these, it likely belongs in",
                      "the synergy_core or synergy_support roles unless its",
                      "color identity or mana cost makes it unviable:"]
            parts += [f"- {p}" for p in analysis.synergy_patterns]

        if getattr(analysis, "core_effect_classes", None):
            parts += ["",
                      "## Core effect classes",
                      "The effect classes this strategy needs redundant",
                      "copies of. When asked to tag cards, use these EXACT",
                      "class names (a card that clearly fills one of these",
                      "roles gets tagged with it; most cards fill none):"]
            parts += [
                f"- {c.get('name')} (deck wants ~{c.get('min_count')}+)"
                for c in analysis.core_effect_classes
                if isinstance(c, dict) and c.get("name")
            ]

        if analysis.anti_synergy_keywords:
            parts += ["",
                      "## Anti-synergy keywords",
                      "Cards mentioning these usually fight the strategy",
                      "and should be downgraded:"]
            parts += [f"- {k}" for k in analysis.anti_synergy_keywords]

        return "\n".join(parts)

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        commander_context: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Call the Anthropic API and return response text.

        max_tokens overrides config.max_tokens for a single call. Use this for
        bulk-output calls (e.g. selecting 150 cards) where the default 4096
        budget would truncate the response and break JSON parsing.

        model overrides config.model for a single call. Used by the
        elimination-round tournament to run coarse filtering on a cheaper
        model (Haiku) while the final precision pick stays on Sonnet.

        commander_context is the per-build stable context block (commander
        analysis, strategy, synergy keywords). When provided, it's appended
        as a second system block and the cache_control marker is placed on
        it, so the entire {rubric + commander} prefix is cached and reused
        across the dozens of select_cards / score_synergy calls in one build.
        Without it, the rubric alone is too short (~150 tokens) to clear
        Sonnet's 2048-token minimum cacheable prefix.
        """
        if self.config.mock_mode or not self.client:
            logger.warning("_call_api called in mock mode; returning empty JSON")
            return "{}"

        try:
            import anthropic  # noqa: F401

            effective_model = model or self.config.model
            effective_max_tokens = max_tokens or self.config.max_tokens
            kwargs = {
                "model": effective_model,
                "max_tokens": effective_max_tokens,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            # System prompt with optional caching. The cache_control marker
            # caches everything from the start of the prefix up to and
            # including the marked block (tools → system, in render order).
            # We always put it on the LAST system block so the maximal
            # stable prefix is cached.
            if self.config.use_prompt_caching:
                cc = {"type": "ephemeral"}
                if self.config.cache_ttl and self.config.cache_ttl != "5m":
                    # v0.9.19: extended TTL (see LLMConfig.cache_ttl). Needs
                    # the extended-cache-ttl beta header (harmless if GA).
                    cc = {"type": "ephemeral", "ttl": self.config.cache_ttl}
                    kwargs["extra_headers"] = {
                        "anthropic-beta": "extended-cache-ttl-2025-04-11"
                    }
                blocks = [{"type": "text", "text": system_prompt}]
                if commander_context:
                    blocks.append({"type": "text", "text": commander_context})
                    # v0.9.16c: TWO breakpoints. The last marks the full
                    # {rubric + commander} prefix (reused across every call in
                    # THIS build). The first marks the frozen rubric ALONE, so
                    # it stays a cache hit across DIFFERENT builds/commanders
                    # within the TTL window (the rubric text never changes) —
                    # the "we run the same prompts a lot" win. Costs nothing:
                    # the rubric is a strict prefix of the full block.
                    blocks[0]["cache_control"] = dict(cc)
                # v0.9.19: pad below-minimum prefixes so they actually cache.
                # The pad sits just before the final block so it's inside the
                # cached prefix (and after the rubric, so the rubric-only
                # breakpoint stays byte-identical across calls).
                pad = self._cache_pad_block(blocks, effective_model)
                if pad is not None:
                    if len(blocks) == 1:
                        blocks.append(pad)
                    else:
                        blocks.insert(len(blocks) - 1, pad)
                blocks[-1]["cache_control"] = dict(cc)
                kwargs["system"] = blocks
            else:
                if commander_context:
                    kwargs["system"] = f"{system_prompt}\n\n{commander_context}"
                else:
                    kwargs["system"] = system_prompt

            # Temperature handling (Opus 4.7 rejects this parameter).
            # Check the EFFECTIVE model — a Haiku tournament round and a
            # Sonnet final pick may differ.
            if not _model_rejects_temperature(effective_model):
                kwargs["temperature"] = (
                    temperature if temperature is not None else self.config.temperature
                )

            response = self.client.messages.create(**kwargs)

            # Log usage info. cache_creation is the (premium-priced) write,
            # cache_read is the (discounted) hit. If both stay at 0 across
            # repeated calls with the same prefix, the prefix is below the
            # model's minimum cacheable size and the marker is being silently
            # ignored — pad the system block to fix.
            usage = response.usage
            cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
            cache_create = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            logger.debug(
                f"API call [{effective_model}]: in={usage.input_tokens} "
                f"out={usage.output_tokens} cache_read={cache_read} "
                f"cache_create={cache_create}"
            )

            # v0.9.16c: accumulate cache stats so a build can report its
            # actual hit rate (measure before optimizing — the "low hit
            # rate" question). Per-model so the Haiku-tournament vs Sonnet
            # split is visible.
            st = self._cache_stats.setdefault(
                effective_model,
                {"calls": 0, "fresh_in": 0, "cache_read": 0, "cache_create": 0},
            )
            st["calls"] += 1
            st["fresh_in"] += usage.input_tokens
            st["cache_read"] += cache_read
            st["cache_create"] += cache_create

            # Warn ONCE per engine if caching is enabled but the prefix is
            # silently uncacheable (both create and read stay 0). The user
            # explicitly opted into use_prompt_caching, so a silent failure
            # is misleading — the marker is set but the API rejected it for
            # being below the minimum cacheable size.
            #
            # Skip the warning for tournament-model (Haiku) calls: Haiku 4.5
            # needs a 4096-token cacheable prefix vs Sonnet's 2048, so our
            # ~2900-token prefix legitimately won't cache there — and Haiku
            # uncached is still cheaper than cached Sonnet, so it's not a
            # problem worth flagging. Only warn for the PRIMARY model, where
            # a silent cache failure actually wastes expensive tokens.
            is_tournament_model = (
                self.config.tournament_model is not None
                and effective_model == self.config.tournament_model
                and effective_model != self.config.model
            )
            if (
                self.config.use_prompt_caching
                and commander_context  # only meaningful when we tried to cache
                and not self._warned_cache_silent
                and not is_tournament_model
                and cache_read == 0
                and cache_create == 0
            ):
                logger.warning(
                    f"Prompt caching is enabled but cache_control was "
                    f"silently ignored on this call (cache_read=0, "
                    f"cache_create=0; whole prompt billed at full price: "
                    f"{usage.input_tokens} tokens). The system prefix is "
                    f"below the model's minimum cacheable size. For Sonnet "
                    f"4.6 the minimum is 2048 tokens; for Opus 4.6/4.7 it "
                    f"is 4096. Either pad the system prompt above that "
                    f"threshold or set LLMConfig(use_prompt_caching=False) "
                    f"to suppress this warning."
                )
                self._warned_cache_silent = True

            # Detect truncation. The Anthropic API returns stop_reason="max_tokens"
            # when it hit the cap before finishing. This almost always produces
            # invalid JSON for our structured-response calls, so flag it loudly
            # rather than letting the JSON parser fail mysteriously downstream.
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "max_tokens":
                logger.warning(
                    f"LLM response hit max_tokens cap ({effective_max_tokens}); "
                    f"output was truncated. Consider raising max_tokens for this "
                    f"call site or reducing the requested item count."
                )

            # Extract text from response
            for block in response.content:
                if getattr(block, 'type', None) == 'text':
                    return block.text
            # Fallback: try first block
            if response.content:
                return getattr(response.content[0], 'text', '')
            return ""

        except Exception as e:
            logger.error(f"API call failed: {e}")
            # Don't crash — return empty so JSON parser falls through to defaults
            return "{}"

    # ------------------------------------------------------------------
    # Internal: Parsing
    # ------------------------------------------------------------------

    # Hint tags the recall pipeline prefixes onto candidate lines. The LLM
    # sometimes echoes them back inside the "name" field, so parsing must
    # strip them before matching.
    _HINT_TAG_PREFIX_RE = re.compile(r"^\s*\[SYN\+{1,3}\]\s*")

    @classmethod
    def _canonicalize_card_name(
        cls, raw: object, lower_to_canonical: dict[str, str],
    ) -> Optional[str]:
        """Map an LLM-returned card name back to the exact candidate name.

        The LLM occasionally drifts on case ("soul warden") or echoes the
        synergy hint tag prefix ("[SYN+++] Soul Warden"). Every downstream
        consumer looks names up in exact-case dicts, so an un-canonicalized
        name silently drops the card (a selection) or the score (synergy).
        Returns None when the name doesn't match any candidate at all.
        """
        if not isinstance(raw, str):
            return None
        cleaned = cls._HINT_TAG_PREFIX_RE.sub("", raw).strip()
        return lower_to_canonical.get(cleaned.lower())

    @staticmethod
    def _parse_json_defensively(text: str) -> Optional[dict]:
        """
        Parse JSON from a potentially messy LLM response.

        Handles:
        - Markdown code blocks (```json ... ```)
        - Leading/trailing prose
        - Partial responses (extracts first {...} block)
        """
        if not text:
            return None

        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try stripping markdown code blocks
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                pass

        # Try EVERY balanced {...} block, not just the first. A reasoning
        # preamble often contains stray braces (e.g. "{G}{W}" mana symbols),
        # so latching onto the first "{" parses garbage and bails before the
        # real JSON object later in the text. Return the first block that
        # parses to a dict.
        for start in (i for i, ch in enumerate(text) if ch == "{"):
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break  # this start didn't yield a dict; try next "{"

        # Show enough context at WARNING for diagnosis (start AND end — failures
        # often happen near a stray quote, trailing comma, or unescaped pipe at
        # the very end of the response). Full text goes to DEBUG for forensic
        # analysis when the user passes --log-file.
        snippet_len = 400
        if len(text) <= snippet_len * 2 + 20:
            preview = text
        else:
            preview = (
                f"{text[:snippet_len]}\n"
                f"... [truncated {len(text) - snippet_len * 2} chars] ...\n"
                f"{text[-snippet_len:]}"
            )
        logger.warning(
            f"Could not parse JSON from LLM response ({len(text)} chars). "
            f"Preview:\n{preview}"
        )
        logger.debug(f"Full unparseable response:\n{text}")
        return None

    @staticmethod
    def _parse_effect_classes(raw) -> list[dict]:
        """Defensively parse core_effect_classes (v0.9.14): keep only entries
        with a non-empty string name; clamp min_count to 1-8 (default 3);
        cap at 8 classes; dedupe by normalized name."""
        out: list[dict] = []
        seen: set[str] = set()
        for entry in (raw or []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            key = name.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                min_count = int(entry.get("min_count", 3))
            except (TypeError, ValueError):
                min_count = 3
            out.append({
                "name": name.strip(),
                "min_count": max(1, min(min_count, 8)),
            })
            if len(out) >= 8:
                break
        return out

    def _build_analysis_from_dict(
        self,
        commander: Card,
        data: dict,
    ) -> CommanderAnalysis:
        """Build a CommanderAnalysis from parsed JSON, with sensible fallbacks."""
        return CommanderAnalysis(
            name=data.get("name", commander.name),
            color_identity=data.get("color_identity", commander.color_identity),
            key_mechanics=data.get("key_mechanics") or [],
            build_around_text=data.get("build_around_text")
                or (commander.text[:200] if commander.text else "Build around the commander."),
            evaluation_notes=data.get("evaluation_notes", "Evaluate cards normally."),
            category_queries=data.get("category_queries") or {},
            synergy_keywords=data.get("synergy_keywords") or [],
            synergy_patterns=[
                p for p in (data.get("synergy_patterns") or [])
                if isinstance(p, str) and p.strip()
            ],
            structural_predicates=[
                p for p in (data.get("structural_predicates") or [])
                if isinstance(p, str) and p.strip()
            ],
            core_effect_classes=self._parse_effect_classes(
                data.get("core_effect_classes")
            ),
            anti_synergy_keywords=data.get("anti_synergy_keywords") or [],
            recommended_weights=(
                data.get("recommended_weights")
                if isinstance(data.get("recommended_weights"), dict)
                else None
            ),
            recommended_synergy_weight=(
                float(data["recommended_synergy_weight"])
                if isinstance(data.get("recommended_synergy_weight"), (int, float))
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Internal: Synergy scoring batch
    # ------------------------------------------------------------------

    def _score_synergy_single(
        self,
        analysis: CommanderAnalysis,
        cards: list[Card],
        synergy_hints: Optional[dict[str, str]] = None,
        class_sink: Optional[dict[str, str]] = None,
    ) -> dict[str, float]:
        """Score a single batch of cards for synergy."""
        if not cards:
            return {}

        if self.config.mock_mode:
            return {c.name: self.quick_synergy_check_with_analysis(analysis, c) for c in cards}

        # Prefix each card line with its hint tag if available. The tag
        # is a strong calibration anchor — the system prompt instructs
        # the LLM to use specific score bands per tag tier so filler
        # cards (no tag) can't sneak into the moderate band.
        hints = synergy_hints or {}
        lines = []
        for c in cards:
            tag = hints.get(c.name)
            if tag:
                lines.append(f"- {tag} {c.format_for_llm()}")
            else:
                lines.append(f"- {c.format_for_llm()}")
        formatted = "\n".join(lines)

        # Commander strategy/keywords live in the cached system block.
        prompt = f"""Score these cards for synergy with {analysis.name}.

Refer to the Commander Context in the system prompt for strategy and
synergy keywords. Use the 0-100 rubric from the system instructions —
respect the score-band anchors for tagged cards.

Cards to Score:
{formatted}

Respond with JSON:
{{
    "scores": [
        {{"name": "Card Name", "score": 75}},
        ...
    ]
}}"""

        response = self._call_api(
            system_prompt=SYNERGY_SCORING_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            commander_context=self._format_commander_context(analysis),
        )

        data = self._parse_json_defensively(response)
        if not data:
            # Fallback to heuristic
            return {c.name: self.quick_synergy_check_with_analysis(analysis, c) for c in cards}

        # Canonicalize returned names: the scoring prompt explicitly allows
        # the hint-tag prefix in the "name" field, and case can drift. An
        # un-canonicalized key would miss the exact-name lookup below and the
        # LLM's reasoned score would be silently replaced by the heuristic.
        lower_to_canonical = {c.name.lower(): c.name for c in cards}
        # v0.9.14: valid effect-class names (normalized) for tag validation.
        known_classes = {
            str(c.get("name", "")).strip().lower(): str(c.get("name", "")).strip()
            for c in (getattr(analysis, "core_effect_classes", None) or [])
            if isinstance(c, dict) and c.get("name")
        }
        result: dict[str, float] = {}
        for entry in data.get("scores", []):
            if isinstance(entry, dict) and "name" in entry and "score" in entry:
                canonical = self._canonicalize_card_name(
                    entry["name"], lower_to_canonical,
                )
                if canonical is None:
                    continue
                try:
                    result[canonical] = max(0.0, min(100.0, float(entry["score"])))
                except (TypeError, ValueError):
                    continue
                # Effect-class tag (optional field) — validate against the
                # analysis's declared classes; drop anything unknown.
                if class_sink is not None and known_classes:
                    raw_cls = entry.get("class")
                    if isinstance(raw_cls, str):
                        cls = known_classes.get(raw_cls.strip().lower())
                        if cls:
                            class_sink[canonical] = cls

        # Fill in any cards the LLM missed using heuristic
        for card in cards:
            if card.name not in result:
                result[card.name] = self.quick_synergy_check_with_analysis(analysis, card)

        return result

    def quick_synergy_check_with_analysis(
        self,
        analysis: CommanderAnalysis,
        card: Card,
    ) -> float:
        """
        Heuristic synergy check using a CommanderAnalysis (instead of raw commander).
        More accurate than quick_synergy_check since it uses the analysis's
        identified synergy keywords.
        """
        score = 45.0
        card_text = (card.text or "").lower()

        for keyword in (analysis.synergy_keywords or []):
            if keyword and keyword.lower() in card_text:
                score += 10

        for keyword in (analysis.anti_synergy_keywords or []):
            if keyword and keyword.lower() in card_text:
                score -= 15

        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Internal: Mock responses (used in mock_mode)
    # ------------------------------------------------------------------

    def _mock_analyze_commander(self, commander: Card) -> CommanderAnalysis:
        """Deterministic mock analysis that inspects the commander's text."""
        text = (commander.text or "").lower()

        mechanics = []
        keywords = []
        if "life" in text and ("gain" in text or "lifelink" in text):
            mechanics.append("lifegain")
            keywords.extend(["gain life", "gain 1 life", "lifelink"])
        if "+1/+1 counter" in text:
            mechanics.append("+1/+1 counters")
            keywords.append("+1/+1 counter")
        if "token" in text:
            mechanics.append("tokens")
            keywords.extend(["create.*token", "creature token"])
        if "graveyard" in text:
            mechanics.append("graveyard")
            keywords.append("graveyard")
        if "whenever.*cast" in text or "cast" in text:
            mechanics.append("spells")
            keywords.append("whenever you cast")

        if not mechanics:
            mechanics = ["commander damage", "general combat"]
            keywords = ["flying", "trample"]

        # Mock pattern set: stripped of numerics, short fragments — what
        # the real LLM is now asked to produce. Keep in sync with prompt.
        mock_patterns: list[str] = []
        for k in keywords:
            # Just reuse the keywords; mock doesn't try to be clever.
            mock_patterns.append(k)
        # Add a few generic fragments by mechanic so pattern recall has
        # something useful to match on in mock-mode tests.
        if "lifegain" in mechanics:
            mock_patterns += ["lifelink", "gain life"]
        if "+1/+1 counters" in mechanics:
            mock_patterns += ["+1/+1 counter", "proliferate"]
        if "tokens" in mechanics:
            mock_patterns += ["creature token", "populate"]

        return CommanderAnalysis(
            name=commander.name,
            color_identity=commander.color_identity,
            key_mechanics=mechanics,
            build_around_text=(
                f"Mock analysis: {commander.name} wants to {', '.join(mechanics)}. "
                f"Build a deck that supports these themes."
            ),
            evaluation_notes=(
                f"[MOCK] Evaluate cards normally with bonuses for {', '.join(mechanics)}."
            ),
            category_queries={"synergy_core": " ".join(keywords[:3])},
            synergy_keywords=keywords,
            synergy_patterns=list(dict.fromkeys(mock_patterns)),  # dedupe, keep order
            anti_synergy_keywords=[],
            recommended_weights=None,  # Use defaults
            recommended_synergy_weight=None,
        )

    def _mock_select_cards(
        self,
        analysis: CommanderAnalysis,
        candidates: list[Card],
        role: str,
        count: int,
    ) -> list[str]:
        """Deterministic mock card selection based on synergy scoring."""
        scored = [
            (self.quick_synergy_check_with_analysis(analysis, c), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c.name for _, c in scored[:count]]
