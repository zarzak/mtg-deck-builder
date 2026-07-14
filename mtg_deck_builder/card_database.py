"""
Card Database - Load and query MTG cards from pipe-delimited CSV.

Design decisions:
- Load entire CSV into memory (MTG has ~33k cards, fits easily)
- Build indexes for fast filtering by color identity, types, etc.
- Support text search for synergy-related queries
- Robust CSV parsing: handles BOM, comments, quoted fields
"""

import csv
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from io import StringIO

from .models import Card

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a card database query."""
    cards: list[Card]
    query_description: str
    total_matches: int


# ----------------------------------------------------------------------
# Format staples (moved here from deck_evaluator in v0.9.4 so both pool
# generation and evaluation can use it without a circular import).
# ----------------------------------------------------------------------
# A curated list of format-defining EDH cards. Used as a strong signal in
# the role-pool quality pre-rank (so famous staples like Sol Ring and
# Rampant Growth aren't cut by truncation) and in the creativity penalty.
# NOT exhaustive — it's a quality FLOOR. The role heuristic
# (_role_quality_score) handles cards not on this list.
COMMON_STAPLES = frozenset({
    # ---- Colorless / artifacts ----
    "Sol Ring", "Arcane Signet", "Command Tower", "Mind Stone",
    "Lightning Greaves", "Swiftfoot Boots", "Skullclamp",
    "Reliquary Tower", "Myriad Landscape", "Fellwar Stone",
    "Thought Vessel", "Wayfarer's Bauble", "Solemn Simulacrum",
    "Burnished Hart", "Everflowing Chalice", "Worn Powerstone",
    "Hedron Archive", "Thran Dynamo", "Gilded Lotus", "Mind's Eye",
    "Idol of Oblivion", "Bonder's Ornament",
    "Commander's Sphere", "The Great Henge",
    # mana rocks (signets/talismans across guilds)
    "Azorius Signet", "Dimir Signet", "Rakdos Signet", "Gruul Signet",
    "Selesnya Signet", "Orzhov Signet", "Izzet Signet", "Golgari Signet",
    "Boros Signet", "Simic Signet",
    "Talisman of Unity", "Talisman of Conviction", "Talisman of Progress",
    "Talisman of Hierarchy", "Talisman of Resilience", "Talisman of Impulse",
    "Talisman of Creativity", "Talisman of Dominance", "Talisman of Indulgence",
    "Talisman of Curiosity",

    # ---- Green ----
    "Cultivate", "Kodama's Reach", "Rampant Growth", "Nature's Lore",
    "Beast Within", "Nature's Claim", "Heroic Intervention",
    "Eternal Witness", "Birds of Paradise", "Llanowar Elves",
    "Elvish Mystic", "Sakura-Tribe Elder", "Three Visits",
    "Farseek", "Wood Elves", "Farhaven Elf", "Sylvan Library",
    "Fyndhorn Elves", "Fauna Shaman", "Wild Growth", "Utopia Sprawl",
    "Explore", "Skyshroud Claim", "Migration Path", "Circuitous Route",
    "Harrow", "Search for Tomorrow",
    "Beanstalk Giant", "Springbloom Druid", "Ranger's Path",
    "Return of the Wildspeaker", "Beast Whisperer", "Guardian Project",
    "Tireless Provisioner", "Ramunap Excavator", "Crop Rotation",
    "Carpet of Flowers", "Vorinclex, Voice of Hunger",
    "Song of the Dryads", "Krosan Grip", "Return to Nature",
    "Tooth and Nail", "Genesis Wave",

    # ---- White ----
    "Swords to Plowshares", "Path to Exile", "Generous Gift",
    "Sun Titan", "Smothering Tithe", "Teferi's Protection",
    "Wrath of God", "Day of Judgment", "Akroma's Will",
    "Enlightened Tutor", "Idyllic Tutor", "Selfless Spirit",
    "Land Tax", "Esper Sentinel", "Mother of Runes", "Giver of Runes",
    "Approach of the Second Sun", "Austere Command", "Farewell",
    "Aura Shards", "Grasp of Fate", "Forsake the Worldly",
    "Settle the Wreckage", "Comeuppance", "Unbreakable Formation",
    "Flawless Maneuver", "Ravos, Soultender", "Archaeomancer's Map",
    "Welcoming Vampire", "Mentor of the Meek", "Containment Priest",
    "Cathars' Crusade", "Anointed Procession",

    # ---- Black ----
    "Demonic Tutor", "Vampiric Tutor", "Sign in Blood",
    "Toxic Deluge", "Phyrexian Arena", "Diabolic Intent",
    "Village Rites", "Deadly Rollick", "Read the Bones",
    "Night's Whisper", "Bolas's Citadel", "Reanimate", "Animate Dead",
    "Necromancy", "Damnation", "Bontu's Last Reckoning",
    "Dark Ritual", "Liliana of the Veil", "Grim Tutor",

    # ---- Blue ----
    "Rhystic Study", "Cyclonic Rift", "Counterspell",
    "Swan Song", "Mystic Remora", "Windfall", "Concentrate",
    "Fierce Guardianship", "Brainstorm", "Ponder", "Preordain",
    "Fact or Fiction", "Mystical Tutor", "Pongify", "Rapid Hybridization",
    "Negate", "Dovin's Veto", "An Offer You Can't Refuse",
    "Arcane Denial", "Frantic Search", "Treasure Cruise",
    "Mystic Confluence", "Propaganda",

    # ---- Red ----
    "Chaos Warp", "Blasphemous Act", "Dockside Extortionist",
    "Faithless Looting", "Wheel of Fortune", "Deflecting Swat",
    "Vandalblast", "Abrade", "Jeska's Will", "Underworld Breach",
    "Reckless Fireweaver", "Goblin Bombardment", "Birgi, God of Storytelling",
    "Light Up the Stage", "Seething Song", "Past in Flames",

    # ---- Multicolor / staple gold cards ----
    "Selesnya Charm", "Assassin's Trophy", "Putrefy", "Mortify",
    "Anguished Unmaking", "Despark", "Vindicate", "Growth Spiral",
    "Kaya's Guile", "Fractured Identity",

    # ---- Common utility lands ----
    "Bojuka Bog", "Rogue's Passage", "Bonders' Enclave",
    "War Room", "Field of the Dead", "Gaea's Cradle", "Cabal Coffers",
    "Nykthos, Shrine to Nyx", "Karn's Bastion", "Yavimaya, Cradle of Growth",
})


def is_staple(card: Card) -> bool:
    """Check if a card is considered a format staple."""
    return card.name in COMMON_STAPLES


# ----------------------------------------------------------------------
# Role detection patterns.
#
# Design philosophy (learned the hard way):
#   1. Prefer BROAD, simple patterns with EXCLUSIONS over narrow patterns.
#      A narrow pattern misses new cards; a broad+excluded pattern is
#      self-healing — if a new card looks like ramp, it'll be counted.
#   2. NEVER rely on specific word counts, number words, or exact templates.
#      WotC invents new phrasing constantly. Patterns must be template-free.
#   3. When in doubt, use two simple alternatives rather than one complex
#      pattern with lookaheads.
#   4. Exclusions should be very conservative — only exclude when we're
#      confident it's a false positive (e.g., "destroy all creatures" is
#      a wipe not a single-target removal).
# ----------------------------------------------------------------------
ROLE_PATTERNS: dict[str, dict] = {
    'ramp': {
        'include': [
            # "Add {X}..." — mana producers
            r'\badd\s+\{',
            # "Add one/two/... mana" — word-based mana production
            r'\badd\b.*\bmana\b',
            # Land tutors. We match either the word "land" OR any basic land type.
            # Using (?:land|forest|plains|island|swamp|mountain)s? keeps this
            # template-free and catches "search for a Forest card" etc.
            r'search your library for .*\b(?:land|forest|plains|island|swamp|mountain)s?\b',
            # "Play an additional land" / "play additional land"
            r'play an? additional land',
            r'additional land',
            # Domri-style "put a land card from your hand onto the battlefield"
            r'put .*\bland\b.*onto the battlefield',
        ],
        'exclude': [
            # These match "add...mana" but aren't ramp
            r'counter target',
            r'destroy target',
            r'exile target',
            # Effects that add mana to OPPONENT's pool aren't ramp for us
            r'opponent.*adds?',
        ],
        'max_mv': 5,  # Ramp worth counting is cheap (exclude 6+cmc "ramp")
    },
    'draw': {
        'include': [
            # Most draw effects: "draw a card", "draw two cards", "draw X cards",
            # "you draw cards" — match any "draw" followed by some word(s) + "card(s)"
            # The \b word boundary prevents matching "overdraw" etc.
            r'\bdraw[s]?\b.{0,30}\bcards?\b',
            # Wheel effects: "each player draws"
            r'each player draws',
        ],
        'exclude': [
            # When ONLY the opponent draws, that's not advantage for us
            # This is conservative: we only exclude clearly-opponent-only draw.
            r'^opponent(?:s)? draws?\b',
            r'target opponent draws',
        ],
    },
    'removal': {
        'include': [
            # Destroy/exile a single target
            r'\bdestroy target\b',
            r'\bexile target\b',
            # -N/-N effects on a target
            r'target creature gets -\d+/-\d+',
            # Damage to target
            r'deals\s+\d+\s+damage\s+to\s+(?:target|any)',
            # Planeswalker removal abilities
            r'−\d+:.*\b(?:destroy|exile)\b',  # Unicode minus
            r'-\d+:.*\b(?:destroy|exile)\b',  # ASCII minus
        ],
        'exclude': [
            # These are wipes, not single-target
            r'\bdestroy all\b',
            r'\bexile all\b',
            r'\bdestroy each\b',
            r'\bexile each\b',
        ],
    },
    # v0.9.14: removal SUB-TYPES. The flat 'removal' count let the GA satisfy
    # its target with six enchantment-hate spells and zero creature removal
    # (observed). These sub-roles give role coverage + the shortfall penalty
    # visibility into the spread. They deliberately OVERLAP 'removal' — a
    # "destroy target permanent" card fills removal, removal_creature, AND
    # removal_artifact simultaneously (flexibility is worth more).
    'removal_creature': {
        'include': [
            r'\b(?:destroy|exile) target\b[^.]{0,40}\bcreature\b',
            r'\b(?:destroy|exile) target (?:nonland |non-land )?permanent\b',
            r'target creature gets -\d+/-\d+',
            r'deals\s+\d+\s+damage\s+to\s+(?:target creature|any target)',
        ],
        'exclude': [
            r'\bdestroy all\b', r'\bexile all\b', r'\bdestroy each\b',
            r'\bexile each\b',
            # "your" scoping: sac/removal of OWN creatures isn't interaction
            r'target creature you control',
        ],
    },
    'removal_artifact': {
        'include': [
            r'\b(?:destroy|exile) target\b[^.]{0,40}\b(?:artifact|enchantment)\b',
            r'\b(?:destroy|exile) target (?:nonland |non-land )?permanent\b',
        ],
        'exclude': [
            r'\bdestroy all\b', r'\bexile all\b', r'\bdestroy each\b',
            r'\bexile each\b',
            r'target (?:artifact|enchantment) you control',
        ],
    },
    'wipe': {
        'include': [
            # "destroy all creatures/permanents/nonland/etc."
            r'\bdestroy all\b',
            r'\bexile all\b',
            r'\bdestroy each\b',
            # Mass -N/-N
            r'all creatures get -\d',
            r'each creature gets -\d',
            # Mass sacrifice
            r'each player sacrifices',
            r'each opponent sacrifices',
        ],
        'exclude': [
            # "Destroy all creatures with flying" etc. still count as wipes,
            # so no exclusions here — if it says "destroy all", it's a wipe.
        ],
    },
    'threat': {
        # Threats are identified structurally, not by text.
        # Custom logic: big creatures, planeswalkers, or explicit win conditions.
        'include': [
            # Win the game effects
            r'\byou win the game\b',
        ],
        'exclude': [],
        'custom_creature_threat': True,
    },
    'protection': {
        'include': [
            # Counterspells
            r'counter target.*\bspell\b',
            # Protection-granting effects (not the keyword on creatures, but
            # effects that grant protection-like abilities to your stuff)
            r'gains? (?:hexproof|indestructible|protection|shroud)',
            # v0.9.15d: equipment/aura grants use "has", not "gains" —
            # Lightning Greaves/Swiftfoot Boots-class cards were orphans.
            r'(?:equipped|enchanted) creature (?:has|gains) [^.]*'
            r'(?:hexproof|indestructible|protection|shroud)',
            r'creatures you control (?:have|gain) [^.]*'
            r'(?:hexproof|indestructible|protection|shroud)',
            r'gains? .* (?:until end of turn)',
            r'phase out',
            r'prevent all',
        ],
        'exclude': [
            # "Creatures your opponents control have hexproof" is not protection for US
            r'opponents? control.*hexproof',
            r'opponents? control.*indestructible',
        ],
    },
    'recursion': {
        'include': [
            # Simple: any "return from [a] graveyard" language
            r'return .*\bfrom\b.*\bgraveyard\b',
            # Self-recursion: "return X to your hand" when X is from graveyard
            r'from .*graveyard.*\bto\b.*\b(?:hand|battlefield)\b',
            r'graveyard.*(?:to|onto) (?:the|your) (?:battlefield|hand)',
        ],
        'exclude': [
            # "Return to hand" without graveyard is bounce, not recursion
            # (already excluded by pattern structure)
        ],
    },
    'land': {
        'include': [],
        'exclude': [],
        'custom_is_land': True,
    },
}


def _compile_patterns(role_def: dict) -> tuple[list, list]:
    """Compile regex patterns for a role definition.

    Uses DOTALL so . matches newlines (rules text often has line breaks).
    Uses MULTILINE so ^ matches after newlines (for opponent-only detection).
    """
    flags = re.IGNORECASE | re.DOTALL | re.MULTILINE
    includes = [re.compile(p, flags) for p in role_def.get('include', [])]
    excludes = [re.compile(p, flags) for p in role_def.get('exclude', [])]
    return includes, excludes


# Pre-compile at module load time for performance
_COMPILED_ROLE_PATTERNS = {
    role: _compile_patterns(role_def)
    for role, role_def in ROLE_PATTERNS.items()
}


def card_fills_role(card: Card, role: str) -> bool:
    """
    Check if a card fills a given role. Exposed as a module-level function
    so DeckEvaluator can use the same logic as pool generation.

    Args:
        card: Card to check
        role: Role name (ramp, draw, removal, wipe, threat, protection, recursion, land)

    Returns:
        True if card fills the role
    """
    if role not in ROLE_PATTERNS:
        return False

    role_def = ROLE_PATTERNS[role]

    # Special case: land
    if role_def.get('custom_is_land'):
        return card.is_land

    # Special case: threat (stat-based OR text-based)
    if role_def.get('custom_creature_threat'):
        # Planeswalkers are threats
        if 'Planeswalker' in card.types:
            return True
        # Big creatures are threats
        if card.is_creature and card.power:
            try:
                power_val = int(card.power)
                # Power 4+ that costs at least 3 mana is a "real threat"
                if power_val >= 4 and card.mana_value >= 3:
                    return True
                # Power 6+ regardless of cost (mostly for cheated-in cards)
                if power_val >= 6:
                    return True
            except ValueError:
                pass  # Power like "*" can't be parsed
        # Also match text-based win conditions (Felidar Sovereign, etc.)
        text = card.text or ''
        if text:
            includes, excludes = _COMPILED_ROLE_PATTERNS[role]
            if any(p.search(text) for p in includes) and not any(p.search(text) for p in excludes):
                return True
        return False

    # For all other roles: text-based matching
    # Skip lands for non-land roles (lands that tap for mana would match "ramp" otherwise)
    if card.is_land:
        return False

    # Mana value constraints
    if 'max_mv' in role_def and card.mana_value > role_def['max_mv']:
        return False
    if 'min_mv' in role_def and card.mana_value < role_def['min_mv']:
        return False

    text = card.text or ''
    if not text:
        return False

    includes, excludes = _COMPILED_ROLE_PATTERNS[role]

    # Must match at least one include pattern
    if not any(p.search(text) for p in includes):
        return False

    # Must not match any exclude pattern
    if any(p.search(text) for p in excludes):
        return False

    return True


def _role_quality_score(card: Card, role: str) -> float:
    """
    Local, network-free estimate of how GOOD a card is at filling a role.

    This is NOT a synergy score — it's a generic "card power for this role"
    estimate used purely to ORDER candidates before truncation, so the LLM
    filter sees the strongest options rather than the alphabetically-first
    ones. It deliberately uses only card-intrinsic signals (no EDHREC, no
    commander context) so it works identically for every commander.

    Signals (additive):
      - Known format staple: large bonus (the curated COMMON_STAPLES floor).
      - Mana efficiency: cheaper is better at the same effect.
      - Role-specific text quality: e.g. ramp that puts lands "onto the
        battlefield" (true ramp) beats "into your hand" (mere fixing);
        removal at instant speed / exile beats sorcery-speed / destroy.

    Returns a float roughly in [0, 100]; only relative order matters.
    """
    score = 0.0
    text = (card.text or "").lower()
    mv = card.mana_value

    # 1. Staple floor — the single strongest signal we have locally.
    if card.name in COMMON_STAPLES:
        score += 60.0

    # 2. Mana efficiency. Cheaper interaction is generally stronger in EDH.
    #    Scale so a 1-mana card gets ~+18, a 6-mana card ~+3. Lands (mv 0)
    #    are handled by their own role and shouldn't be penalized as "free".
    if not card.is_land:
        score += max(0.0, 20.0 - mv * 3.0)

    # 3. Role-specific text quality.
    if role == "ramp":
        # True ramp (lands to battlefield, mana production) > hand-fixing.
        if "onto the battlefield" in text:
            score += 18.0
        elif "into your hand" in text or "to your hand" in text:
            score += 4.0
        if re.search(r"\badd\b.*\bmana\b", text) or re.search(r"\badd\s+\{", text):
            score += 14.0  # mana rocks/dorks
        if "search your library" in text and "land" in text:
            score += 8.0
        # Net-positive fast mana ("add {c}{c}", "add two", "add three")
        if re.search(r"add\s+\{[wubrgc]\}\{[wubrgc]\}", text) or "add three" in text:
            score += 10.0

    elif role == "removal":
        if "exile target" in text:
            score += 14.0
        elif "destroy target" in text:
            score += 10.0
        # Instant speed is more flexible than sorcery speed.
        if "instant" in (card.card_type or "").lower():
            score += 8.0
        # Unconditional removal (no "with", "that has", etc. narrowing) is better.
        if re.search(r"(destroy|exile) target (creature|permanent)\b", text) and \
           not re.search(r"target .* (with|that|of mana value)", text):
            score += 6.0

    elif role == "draw":
        # Repeatable / engine draw beats one-shot.
        if "at the beginning of" in text or "whenever" in text:
            score += 12.0
        if re.search(r"draw\s+(two|three|x)\s+cards?", text):
            score += 6.0

    elif role == "wipe":
        if "destroy all" in text or "exile all" in text:
            score += 12.0

    elif role in ("threat", "protection", "recursion"):
        # Generic: lower-cost evasive/value bodies tend to be better, but
        # we lean on the mana-efficiency term above and the staple floor.
        if "flying" in text or "trample" in text or "menace" in text:
            score += 4.0

    return score


class CardDatabase:
    """Load and query MTG cards from a pipe-delimited CSV."""

    COLUMN_MAP = {
        'name': 'name',
        'manaCost': 'mana_cost',
        'manaValue': 'mana_value',
        'type': 'card_type',
        'text': 'text',
        'colorIdentity': 'color_identity',
        'colors': 'colors',
        'power': 'power',
        'toughness': 'toughness',
        'loyalty': 'loyalty',
        'defense': 'defense',
        'types': 'types',
        'subtypes': 'subtypes',
        'supertypes': 'supertypes',
        'keywords': 'keywords',
        'layout': 'layout',
        'legalities': 'legalities',
    }

    # v0.9.17: optional MTGJSON Game Changer column. Handled outside
    # COLUMN_MAP because it needs bool coercion and multiple header spellings.
    # Absent in older CSVs (then all cards default is_game_changer=False and
    # bracket.py falls back to the embedded list).
    _GAME_CHANGER_COLUMNS = ('isGameChanger', 'is_game_changer', 'gameChanger',
                             'game_changer')
    _TRUTHY = frozenset({'1', 'true', 'yes', 'y', 't'})

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self._cards: list[Card] = []
        self._by_name: dict[str, Card] = {}
        self._loaded = False
        # Whether the CSV actually has a legalities column. Decides how an
        # EMPTY legalities value is interpreted in _is_valid_card: with the
        # column present (MTGJSON-derived data), empty means the card is
        # legal in NO format (banned / un-set / memorabilia) and must be
        # excluded; with the column absent entirely, we stay lenient so a
        # minimal CSV isn't wiped out.
        self._has_legalities_column = False
        # v0.9.17: True when the CSV carried an isGameChanger column, in which
        # case the per-card flags are authoritative (bracket.py ignores the
        # embedded name list).
        self.has_game_changer_column = False

    def load(self) -> int:
        """Load cards from CSV file. Returns number of cards loaded."""
        if self._loaded:
            logger.info("Database already loaded, skipping reload")
            return len(self._cards)

        logger.info(f"Loading cards from {self.csv_path}")

        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            # Skip comment lines (lines starting with #)
            lines = []
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines.append(line)

            if not lines:
                logger.error("No data lines found in CSV")
                return 0

            # Detect delimiter
            header_line = lines[0]
            if '\t' in header_line and '|' not in header_line:
                delimiter = '\t'
            elif '|' in header_line:
                delimiter = '|'
            else:
                delimiter = ','

            logger.debug(f"Detected delimiter: {repr(delimiter)}")

            csv_content = ''.join(lines)
            reader = csv.DictReader(
                StringIO(csv_content),
                delimiter=delimiter,
                quotechar='"',
                doublequote=True,
            )

            if reader.fieldnames:
                reader.fieldnames = [fn.strip() for fn in reader.fieldnames]
                logger.debug(f"Columns: {reader.fieldnames}")
                self._has_legalities_column = 'legalities' in reader.fieldnames
                self.has_game_changer_column = any(
                    c in reader.fieldnames for c in self._GAME_CHANGER_COLUMNS
                )

            loaded_count = 0
            skipped_count = 0

            for row in reader:
                try:
                    row = {
                        k.strip(): v.strip() if isinstance(v, str) else v
                        for k, v in row.items() if k
                    }
                    card = self._parse_row(row)
                    if card and self._is_valid_card(card):
                        # Prevent duplicate names (different printings)
                        if card.name.lower() not in self._by_name:
                            self._cards.append(card)
                            self._by_name[card.name.lower()] = card
                            loaded_count += 1
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse: {row.get('name', 'unknown')}: {e}")
                    skipped_count += 1

        self._loaded = True
        logger.info(f"Loaded {loaded_count} cards, skipped {skipped_count}")
        return len(self._cards)

    def _parse_row(self, row: dict) -> Optional[Card]:
        """Parse a CSV row into a Card object."""
        card_data = {}
        for csv_col, card_attr in self.COLUMN_MAP.items():
            value = row.get(csv_col, '')

            if card_attr == 'mana_value':
                try:
                    value = int(float(value)) if value else 0
                except ValueError:
                    value = 0
            elif value is None:
                value = ''

            card_data[card_attr] = value

        # Convert \n to actual newlines in text
        if card_data.get('text'):
            card_data['text'] = card_data['text'].replace('\\n', '\n')

        # v0.9.17: Game Changer flag from the CSV, if the column exists.
        gc_val = ''
        for col in self._GAME_CHANGER_COLUMNS:
            if col in row:
                gc_val = row.get(col) or ''
                break
        card_data['is_game_changer'] = str(gc_val).strip().lower() in self._TRUTHY

        # v0.9.16b: normalize same-face double-sided names ("Sol Ring //
        # Sol Ring" — Secret Lair-style promo printings that MTGJSON stores
        # as distinct entries). Left as-is they masquerade as different
        # cards, and a deck can end up with two Sol Rings that duplicate
        # validation can't see (observed in a real run once power-scan
        # covered the variants). Real MDFCs (different faces) are untouched.
        name = card_data.get('name') or ''
        if ' // ' in name:
            faces = [f.strip() for f in name.split(' // ')]
            if len(set(faces)) == 1:
                card_data['name'] = faces[0]

        try:
            return Card(**card_data)
        except TypeError as e:
            logger.debug(f"Failed to create Card: {e}")
            return None

    # Un-set / acorn card types that are technically "commander-legal" in
    # the data but require sticker/attraction sideboard mechanics that don't
    # belong in a normal deck. Carnival Elephant Meteor (a "Stickers" card)
    # is the example that leaked into a build.
    _JOKE_CARD_TYPES = ('stickers', 'sticker', 'attraction', 'contraption')

    def _is_valid_card(self, card: Card) -> bool:
        """Check if card should be included in database."""
        if card.layout in ('token', 'emblem', 'art_series'):
            return False
        if not card.name:
            return False
        # v0.9.4: drop un-set acorn cards (Stickers/Attractions/Contraptions).
        # They appear "commander-legal" in legalities data but need
        # sticker-sheet / attraction-deck mechanics absent from real play.
        card_type_lower = (card.card_type or "").lower()
        if any(jt in card_type_lower for jt in self._JOKE_CARD_TYPES):
            return False
        # Legality. When the CSV has a legalities column (MTGJSON-derived
        # data lists the formats a card is legal in), an EMPTY value means
        # the card is legal NOWHERE — banned (Mox Emerald), un-set/acorn
        # (HONK!), Conspiracy/Vanguard-only, etc. Those must be excluded;
        # the old "be lenient when empty" rule let ~2,100 such cards into
        # the pool. Leniency is kept only when the column doesn't exist at
        # all (a minimal CSV without legality data).
        if card.legalities:
            if 'commander' not in card.legalities.lower():
                return False
        elif self._has_legalities_column:
            return False
        return True

    def get_by_name(self, name: str) -> Optional[Card]:
        """Get a card by exact name (case-insensitive)."""
        return self._by_name.get(name.lower().strip())

    def search_by_name(self, partial_name: str, limit: int = 10) -> list[Card]:
        """Search for cards with names containing the given string."""
        partial = partial_name.lower().strip()
        matches = [
            card for name, card in self._by_name.items()
            if partial in name
        ]
        return matches[:limit]

    def find_similar_names(self, name: str, limit: int = 5) -> list[str]:
        """Find card names similar to the given name. Useful for debugging."""
        name_lower = name.lower().strip()
        words = [w for w in name_lower.split() if len(w) > 2]

        if not words:
            # Try single-word substring match
            matches = [n for n in self._by_name.keys() if name_lower in n]
            return [self._by_name[m].name for m in matches[:limit]]

        candidates = []
        for word in words:
            for card_name in self._by_name.keys():
                if word in card_name and card_name not in candidates:
                    candidates.append(card_name)

        def similarity(card_name: str) -> int:
            return sum(1 for w in words if w in card_name)

        candidates.sort(key=similarity, reverse=True)
        return [self._by_name[c].name for c in candidates[:limit]]

    def query(
        self,
        color_identity: Optional[str] = None,
        card_types: Optional[list[str]] = None,
        text_pattern: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        max_mana_value: Optional[int] = None,
        min_mana_value: Optional[int] = None,
        is_creature: Optional[bool] = None,
        is_land: Optional[bool] = None,
        exclude_names: Optional[set[str]] = None,
        custom_filter: Optional[Callable[[Card], bool]] = None,
        limit: Optional[int] = None,
    ) -> QueryResult:
        """Query cards with multiple filter criteria."""
        if not self._loaded:
            self.load()

        results = self._cards
        query_parts = []

        if color_identity is not None:
            results = [c for c in results if self._matches_color_identity(c, color_identity)]
            query_parts.append(f"color_identity⊆{color_identity}")

        if card_types:
            results = [c for c in results if any(t in c.types for t in card_types)]
            query_parts.append(f"types∈{card_types}")

        if text_pattern:
            pattern = re.compile(text_pattern, re.IGNORECASE)
            results = [c for c in results if pattern.search(c.text or '')]
            query_parts.append(f"text~{text_pattern}")

        if keywords:
            results = [c for c in results if any(kw.lower() in c.keywords.lower() for kw in keywords)]
            query_parts.append(f"keywords∈{keywords}")

        if max_mana_value is not None:
            results = [c for c in results if c.mana_value <= max_mana_value]
            query_parts.append(f"mv≤{max_mana_value}")

        if min_mana_value is not None:
            results = [c for c in results if c.mana_value >= min_mana_value]
            query_parts.append(f"mv≥{min_mana_value}")

        if is_creature is not None:
            if is_creature:
                results = [c for c in results if c.is_creature]
            else:
                results = [c for c in results if not c.is_creature]
            query_parts.append(f"is_creature={is_creature}")

        if is_land is not None:
            if is_land:
                results = [c for c in results if c.is_land]
            else:
                results = [c for c in results if not c.is_land]
            query_parts.append(f"is_land={is_land}")

        if exclude_names:
            exclude_lower = {n.lower() for n in exclude_names}
            results = [c for c in results if c.name.lower() not in exclude_lower]
            query_parts.append(f"excluding {len(exclude_names)} cards")

        if custom_filter:
            results = [c for c in results if custom_filter(c)]
            query_parts.append("custom_filter")

        total = len(results)

        if limit is not None:
            results = results[:limit]

        query_desc = " AND ".join(query_parts) if query_parts else "all cards"

        return QueryResult(
            cards=results,
            query_description=query_desc,
            total_matches=total
        )

    def _matches_color_identity(self, card: Card, target_identity: str) -> bool:
        """
        Check if card's color identity is a subset of target identity.

        In EDH, a card's color identity must be a subset of the commander's.
        """
        # Extract just W/U/B/R/G characters (ignore commas and whitespace)
        card_ci = set(c for c in (card.color_identity or '') if c in 'WUBRG')
        target_ci = set(c for c in (target_identity or '') if c in 'WUBRG')

        # Colorless card can go in any deck
        if not card_ci:
            return True

        return card_ci.issubset(target_ci)

    def search_text(
        self,
        patterns: list[str],
        color_identity: Optional[str] = None,
        operator: str = "OR",
    ) -> QueryResult:
        """Search card text for multiple patterns."""
        if not self._loaded:
            self.load()

        compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

        def matches(card: Card) -> bool:
            text = card.text or ''
            if operator == "OR":
                return any(p.search(text) for p in compiled)
            else:
                return all(p.search(text) for p in compiled)

        results = [c for c in self._cards if matches(c)]

        if color_identity:
            results = [c for c in results if self._matches_color_identity(c, color_identity)]

        return QueryResult(
            cards=results,
            query_description=f"text search: {operator}({patterns})",
            total_matches=len(results)
        )

    def get_cards_for_role(
        self,
        role: str,
        color_identity: str,
        limit: Optional[int] = None,
    ) -> list[Card]:
        """
        Get candidate cards for a specific deck role. Uses the same
        `card_fills_role` logic as deck evaluation for consistency.

        v0.9.5 (de-truncation): `limit` now defaults to None, which returns
        EVERY matching card with NO quality pre-rank — just name-sorted for
        deterministic downstream tournament chunking. This is the LLM-filter
        path: the elimination tournament reviews every card in the role, so
        there is no need (and no desire) to pre-rank or cut the pool.

        When an explicit `limit` IS given, candidates are sorted by a local
        quality heuristic (`_role_quality_score`) and truncated. This bounded
        path serves the non-LLM baseline/simple assembly, where we DO want
        the best `limit` cards by heuristic since no LLM reviews the rest.

        Why de-truncate: the old `limit=300` returned the first 300 matches
        and silently cut format staples (Sol Ring, Rampant Growth, Land Tax).
        The v0.9.4 quality pre-rank floated staples to the top so they
        survived truncation; full de-truncation removes the cut entirely so
        the LLM simply sees everything in the role.
        """
        if not self._loaded:
            self.load()

        results = []
        for card in self._cards:
            if not self._matches_color_identity(card, color_identity):
                continue
            if card_fills_role(card, role):
                results.append(card)

        if limit is None:
            # Full de-truncation: every match, no quality pre-rank. Name
            # sort only, for deterministic tournament chunking across runs.
            results.sort(key=lambda c: c.name)
            return results

        # Bounded path (baseline/simple assembly): quality pre-rank, then cut.
        results.sort(key=lambda c: (-_role_quality_score(c, role), c.name))
        return results[:limit]

    def get_cards_matching_predicates(
        self,
        predicates,
        color_identity: str,
        limit: Optional[int] = None,
    ) -> list[Card]:
        """v0.9.9: color-legal cards matching ANY structural predicate.

        Used by the structural-synergy recall: for "vanilla matters" etc.,
        these cards have no text and so are invisible to the text-based recall
        sources — this pulls them in directly by attribute. Ranked by combat
        stats (power+toughness desc) since attribute archetypes are usually
        creature beatdown; bigger bodies first when capped.
        """
        if not self._loaded:
            self.load()
        from .structural_predicates import card_matches_any

        preds = [p for p in (predicates or []) if str(p).strip()]
        if not preds:
            return []

        def _stat(c: Card) -> int:
            tot = 0
            for v in (c.power, c.toughness):
                try:
                    tot += int(v)
                except (TypeError, ValueError):
                    pass
            return tot

        results = [
            c for c in self._cards
            if self._matches_color_identity(c, color_identity)
            and card_matches_any(c, preds)
        ]
        results.sort(key=lambda c: (-_stat(c), c.mana_value, c.name))
        return results[:limit] if limit else results

    @property
    def card_count(self) -> int:
        return len(self._cards)

    @property
    def all_cards(self) -> list[Card]:
        return self._cards.copy()
