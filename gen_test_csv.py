#!/usr/bin/env python3
"""Generate a clean test CSV for MTG deck builder testing."""

# Each row is a tuple of 17 fields matching the header order.
# Use 'NL' as placeholder for newlines within text fields; we'll convert
# to literal backslash-n when writing.
NL = "\\n"

rows = [
    ("Lathiel, the Bounteous Dawn", "{2}{G}{W}", "4", "Legendary Creature — Unicorn",
     f"Lifelink{NL}At the beginning of each end step, if you gained life this turn, distribute up to that many +1/+1 counters among any number of other target creatures.",
     "G,W", "G,W", "2", "2", "", "", "Creature", "Unicorn", "Legendary", "Lifelink", "normal", "commander"),

    ("Jasmine Boreal", "{2}{G}{W}", "4", "Legendary Creature — Human Cleric",
     "", "G,W", "G,W", "3", "4", "", "", "Creature", "Human,Cleric", "Legendary", "", "normal", "commander"),

    ("Karlov of the Ghost Council", "{1}{W}{B}", "3", "Legendary Creature — Spirit Advisor",
     f"Whenever you gain life, put two +1/+1 counters on Karlov of the Ghost Council.{NL}{{W}}{{B}}, Exile another target creature you control: Exile target creature. Activate this ability only if Karlov of the Ghost Council has ten or more +1/+1 counters on it.",
     "B,W", "B,W", "2", "2", "", "", "Creature", "Spirit,Advisor", "Legendary", "", "normal", "commander"),

    ("Sol Ring", "{1}", "1", "Artifact", "{T}: Add {C}{C}.",
     "", "", "", "", "", "", "Artifact", "", "", "", "normal", "commander"),

    ("Arcane Signet", "{2}", "2", "Artifact",
     "{T}: Add one mana of any color in your commander's color identity.",
     "", "", "", "", "", "", "Artifact", "", "", "", "normal", "commander"),

    ("Command Tower", "", "0", "Land",
     "{T}: Add one mana of any color in your commander's color identity.",
     "", "", "", "", "", "", "Land", "", "", "", "normal", "commander"),

    ("Cultivate", "{2}{G}", "3", "Sorcery",
     "Search your library for up to two basic land cards, reveal those cards, and put one onto the battlefield tapped and the other into your hand. Then shuffle.",
     "G", "G", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Kodama's Reach", "{2}{G}", "3", "Sorcery",
     "Search your library for up to two basic land cards, reveal those cards, put one onto the battlefield tapped and the other into your hand. Then shuffle.",
     "G", "G", "", "", "", "", "Sorcery", "Arcane", "", "", "normal", "commander"),

    ("Rampant Growth", "{1}{G}", "2", "Sorcery",
     "Search your library for a basic land card, put that card onto the battlefield tapped, then shuffle.",
     "G", "G", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Birds of Paradise", "{G}", "1", "Creature — Bird",
     f"Flying{NL}{{T}}: Add one mana of any color.",
     "G", "G", "0", "1", "", "", "Creature", "Bird", "", "Flying", "normal", "commander"),

    ("Llanowar Elves", "{G}", "1", "Creature — Elf Druid",
     "{T}: Add {G}.",
     "G", "G", "1", "1", "", "", "Creature", "Elf,Druid", "", "", "normal", "commander"),

    ("Elvish Mystic", "{G}", "1", "Creature — Elf Druid",
     "{T}: Add {G}.",
     "G", "G", "1", "1", "", "", "Creature", "Elf,Druid", "", "", "normal", "commander"),

    ("Swords to Plowshares", "{W}", "1", "Instant",
     "Exile target creature. Its controller gains life equal to its power.",
     "W", "W", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Path to Exile", "{W}", "1", "Instant",
     "Exile target creature. Its controller may search their library for a basic land card, put that card onto the battlefield tapped, then shuffle.",
     "W", "W", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Generous Gift", "{2}{W}", "3", "Instant",
     "Destroy target permanent. Its controller creates a 3/3 green Elephant creature token.",
     "W", "W", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Beast Within", "{2}{G}", "3", "Instant",
     "Destroy target permanent. Its controller creates a 3/3 green Beast creature token.",
     "G", "G", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Wrath of God", "{2}{W}{W}", "4", "Sorcery",
     "Destroy all creatures. They can't be regenerated.",
     "W", "W", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Day of Judgment", "{2}{W}{W}", "4", "Sorcery",
     "Destroy all creatures.",
     "W", "W", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Grizzly Bears", "{1}{G}", "2", "Creature — Bear",
     "",
     "G", "G", "2", "2", "", "", "Creature", "Bear", "", "", "normal", "commander"),

    ("Savannah Lions", "{W}", "1", "Creature — Cat",
     "",
     "W", "W", "2", "1", "", "", "Creature", "Cat", "", "", "normal", "commander"),

    ("Watchwolf", "{G}{W}", "2", "Creature — Wolf",
     "",
     "G,W", "G,W", "3", "3", "", "", "Creature", "Wolf", "", "", "normal", "commander"),

    ("Soul's Attendant", "{W}", "1", "Creature — Human Cleric",
     "Whenever a creature enters the battlefield, you may gain 1 life.",
     "W", "W", "1", "2", "", "", "Creature", "Human,Cleric", "", "", "normal", "commander"),

    ("Soul Warden", "{W}", "1", "Creature — Human Cleric",
     "Whenever another creature enters the battlefield, you gain 1 life.",
     "W", "W", "1", "1", "", "", "Creature", "Human,Cleric", "", "", "normal", "commander"),

    ("Ajani's Pridemate", "{1}{W}", "2", "Creature — Cat Soldier",
     "Whenever you gain life, you may put a +1/+1 counter on Ajani's Pridemate.",
     "W", "W", "1", "1", "", "", "Creature", "Cat,Soldier", "", "", "normal", "commander"),

    ("Archangel of Thune", "{3}{W}{W}", "5", "Creature — Angel",
     f"Flying, vigilance, lifelink{NL}Whenever you gain life, put a +1/+1 counter on each creature you control.",
     "W", "W", "3", "4", "", "", "Creature", "Angel", "", "Flying,Vigilance,Lifelink", "normal", "commander"),

    ("Trelasarra, Moon Dancer", "{G}{W}", "2", "Legendary Creature — Human Cleric",
     f"Trelasarra, Moon Dancer enters with two +1/+1 counters on it if you gained life this turn.{NL}Whenever you scry, you gain 1 life.",
     "G,W", "G,W", "1", "1", "", "", "Creature", "Human,Cleric", "Legendary", "", "normal", "commander"),

    ("Essence Warden", "{G}", "1", "Creature — Elf Shaman",
     "Whenever another creature enters the battlefield, you gain 1 life.",
     "G", "G", "1", "1", "", "", "Creature", "Elf,Shaman", "", "", "normal", "commander"),

    ("Heliod, Sun-Crowned", "{1}{W}{W}", "3", "Legendary Enchantment Creature — God",
     f"Indestructible{NL}As long as your devotion to white is less than five, Heliod isn't a creature.{NL}Whenever you gain life, put a +1/+1 counter on target creature or enchantment you control.{NL}{{1}}{{W}}: Target creature gains lifelink until end of turn.",
     "W", "W", "5", "5", "", "", "Creature,Enchantment", "God", "Legendary", "Indestructible", "normal", "commander"),

    ("Sun Titan", "{4}{W}{W}", "6", "Creature — Giant",
     f"Vigilance{NL}Whenever Sun Titan enters the battlefield or attacks, you may return target permanent card with mana value 3 or less from your graveyard to the battlefield.",
     "W", "W", "6", "6", "", "", "Creature", "Giant", "", "Vigilance", "normal", "commander"),

    ("Rhox Faithmender", "{4}{W}", "5", "Creature — Rhino Monk",
     f"Lifelink{NL}If you would gain life, you gain twice that much life instead.",
     "W", "W", "2", "5", "", "", "Creature", "Rhino,Monk", "", "Lifelink", "normal", "commander"),

    ("Felidar Sovereign", "{4}{W}{W}", "6", "Creature — Cat Beast",
     f"Vigilance, lifelink{NL}At the beginning of your upkeep, if you have 40 or more life, you win the game.",
     "W", "W", "4", "6", "", "", "Creature", "Cat,Beast", "", "Vigilance,Lifelink", "normal", "commander"),

    ("Crested Sunmare", "{3}{W}{W}", "5", "Creature — Horse",
     f"Other Horses you control have indestructible.{NL}At the beginning of each end step, if you gained life this turn, create a 5/5 white Horse creature token.",
     "W", "W", "5", "5", "", "", "Creature", "Horse", "", "", "normal", "commander"),

    ("Phyrexian Arena", "{1}{B}{B}", "3", "Enchantment",
     "At the beginning of your upkeep, you lose 1 life and draw a card.",
     "B", "B", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Sign in Blood", "{B}{B}", "2", "Sorcery",
     "Target player draws two cards and loses 2 life.",
     "B", "B", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Harmonize", "{2}{G}{G}", "4", "Sorcery",
     "Draw three cards.",
     "G", "G", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Concentrate", "{2}{U}{U}", "4", "Sorcery",
     "Draw three cards.",
     "U", "U", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Windfall", "{2}{U}", "3", "Sorcery",
     "Each player discards their hand, then draws cards equal to the greatest number of cards a player discarded this way.",
     "U", "U", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Nature's Lore", "{G}", "1", "Sorcery",
     "Search your library for a Forest card, put it onto the battlefield, then shuffle.",
     "G", "G", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Farseek", "{1}{G}", "2", "Sorcery",
     "Search your library for a Plains, Island, Swamp, or Mountain card, put it onto the battlefield tapped, then shuffle.",
     "G", "G", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Sakura-Tribe Elder", "{1}{G}", "2", "Creature — Snake Shaman",
     "Sacrifice Sakura-Tribe Elder: Search your library for a basic land card, put that card onto the battlefield tapped, then shuffle.",
     "G", "G", "1", "1", "", "", "Creature", "Snake,Shaman", "", "", "normal", "commander"),

    ("Forest", "", "0", "Basic Land — Forest", "({T}: Add {G}.)", "G", "", "", "", "", "", "Land", "Forest", "Basic", "", "normal", "commander"),
    ("Plains", "", "0", "Basic Land — Plains", "({T}: Add {W}.)", "W", "", "", "", "", "", "Land", "Plains", "Basic", "", "normal", "commander"),
    ("Island", "", "0", "Basic Land — Island", "({T}: Add {U}.)", "U", "", "", "", "", "", "Land", "Island", "Basic", "", "normal", "commander"),
    ("Swamp", "", "0", "Basic Land — Swamp", "({T}: Add {B}.)", "B", "", "", "", "", "", "Land", "Swamp", "Basic", "", "normal", "commander"),
    ("Mountain", "", "0", "Basic Land — Mountain", "({T}: Add {R}.)", "R", "", "", "", "", "", "Land", "Mountain", "Basic", "", "normal", "commander"),

    ("Temple Garden", "", "0", "Land — Forest Plains",
     f"({{T}}: Add {{G}} or {{W}}.){NL}As Temple Garden enters, you may pay 2 life. If you don't, it enters tapped.",
     "G,W", "", "", "", "", "", "Land", "Forest,Plains", "", "", "normal", "commander"),

    ("Sunpetal Grove", "", "0", "Land",
     f"Sunpetal Grove enters tapped unless you control a Forest or Plains.{NL}{{T}}: Add {{G}} or {{W}}.",
     "G,W", "", "", "", "", "", "Land", "", "", "", "normal", "commander"),

    ("Canopy Vista", "", "0", "Land — Forest Plains",
     f"Canopy Vista enters tapped unless you control two or more basic lands.{NL}{{T}}: Add {{G}} or {{W}}.",
     "G,W", "", "", "", "", "", "Land", "Forest,Plains", "", "", "normal", "commander"),

    ("Fortified Village", "", "0", "Land",
     f"Fortified Village enters tapped unless you control a Forest or Plains.{NL}{{T}}: Add {{G}} or {{W}}.",
     "G,W", "", "", "", "", "", "Land", "", "", "", "normal", "commander"),

    ("Scattered Groves", "", "0", "Land — Forest Plains",
     f"Scattered Groves enters tapped.{NL}{{T}}: Add {{G}} or {{W}}.{NL}Cycling {{2}}",
     "G,W", "", "", "", "", "", "Land", "Forest,Plains", "", "Cycling", "normal", "commander"),

    ("Eternal Witness", "{1}{G}{G}", "3", "Creature — Human Shaman",
     "When Eternal Witness enters the battlefield, you may return target card from your graveyard to your hand.",
     "G", "G", "2", "1", "", "", "Creature", "Human,Shaman", "", "", "normal", "commander"),

    ("Reclamation Sage", "{2}{G}", "3", "Creature — Elf Shaman",
     "When Reclamation Sage enters the battlefield, you may destroy target artifact or enchantment.",
     "G", "G", "2", "1", "", "", "Creature", "Elf,Shaman", "", "", "normal", "commander"),

    ("Elesh Norn, Grand Cenobite", "{5}{W}{W}{W}", "8", "Legendary Creature — Phyrexian Praetor",
     f"Vigilance{NL}Other creatures you control get +2/+2.{NL}Creatures your opponents control get -2/-2.",
     "W", "W", "4", "7", "", "", "Creature", "Phyrexian,Praetor", "Legendary", "Vigilance", "normal", "commander"),

    ("Smothering Tithe", "{3}{W}", "4", "Enchantment",
     "Whenever an opponent draws a card, that player may pay {2}. If the player doesn't, you create a Treasure token.",
     "W", "W", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Beast Whisperer", "{2}{G}{G}", "4", "Creature — Elf Druid",
     "Whenever you cast a creature spell, draw a card.",
     "G", "G", "2", "3", "", "", "Creature", "Elf,Druid", "", "", "normal", "commander"),

    ("Guardian Project", "{3}{G}", "4", "Enchantment",
     "Whenever a nontoken creature enters the battlefield under your control, if it has a different name than each other creature you control and each card in your graveyard, draw a card.",
     "G", "G", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Mirari's Wake", "{4}{G}{W}", "6", "Enchantment",
     f"Creatures you control get +1/+1.{NL}Whenever you tap a land for mana, add one mana of any type that land produced.",
     "G,W", "G,W", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Cathars' Crusade", "{3}{W}{W}", "5", "Enchantment",
     "Whenever a creature enters the battlefield under your control, put a +1/+1 counter on each creature you control.",
     "W", "W", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Anointed Procession", "{3}{W}", "4", "Enchantment",
     "If one or more tokens would be created under your control, twice that many of those tokens are created instead.",
     "W", "W", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Trostani, Selesnya's Voice", "{2}{G}{W}", "4", "Legendary Creature — Dryad",
     f"Whenever another creature enters the battlefield under your control, you gain life equal to that creature's toughness.{NL}{{1}}{{G}}{{W}}, {{T}}: Populate.",
     "G,W", "G,W", "2", "5", "", "", "Creature", "Dryad", "Legendary", "", "normal", "commander"),

    ("Selfless Spirit", "{1}{W}", "2", "Creature — Spirit",
     f"Flying{NL}Sacrifice Selfless Spirit: Creatures you control gain indestructible until end of turn.",
     "W", "W", "2", "1", "", "", "Creature", "Spirit", "", "Flying", "normal", "commander"),

    ("Lightning Greaves", "{2}", "2", "Artifact — Equipment",
     f"Equipped creature has haste and shroud.{NL}Equip {{0}}",
     "", "", "", "", "", "", "Artifact", "Equipment", "", "Equip", "normal", "commander"),

    ("Swiftfoot Boots", "{2}", "2", "Artifact — Equipment",
     f"Equipped creature has hexproof and haste.{NL}Equip {{1}}",
     "", "", "", "", "", "", "Artifact", "Equipment", "", "Equip", "normal", "commander"),

    ("Selesnya Signet", "{2}", "2", "Artifact",
     "{1}, {T}: Add {G}{W}.",
     "G,W", "", "", "", "", "", "Artifact", "", "", "", "normal", "commander"),

    ("Selesnya Charm", "{G}{W}", "2", "Instant",
     f"Choose one —{NL}• Target creature gets +2/+2 and gains trample until end of turn.{NL}• Exile target creature with power 5 or greater.{NL}• Create a 2/2 white Knight creature token with vigilance.",
     "G,W", "G,W", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Enlightened Tutor", "{W}", "1", "Instant",
     "Search your library for an artifact or enchantment card, reveal that card, put it on top of your library, then shuffle.",
     "W", "W", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Idyllic Tutor", "{2}{W}", "3", "Sorcery",
     "Search your library for an enchantment card, reveal it, put it into your hand, then shuffle.",
     "W", "W", "", "", "", "", "Sorcery", "", "", "", "normal", "commander"),

    ("Nature's Claim", "{G}", "1", "Instant",
     "Destroy target artifact or enchantment. Its controller gains 4 life.",
     "G", "G", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Exquisite Blood", "{4}{B}", "5", "Enchantment",
     "Whenever an opponent loses life, you gain that much life.",
     "B", "B", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Sanguine Bond", "{3}{B}{B}", "5", "Enchantment",
     "Whenever you gain life, target opponent loses that much life.",
     "B", "B", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Bojuka Bog", "", "0", "Land — Swamp",
     f"Bojuka Bog enters tapped.{NL}When Bojuka Bog enters the battlefield, exile all cards from target player's graveyard.",
     "B", "", "", "", "", "", "Land", "Swamp", "", "", "normal", "commander"),

    ("Reliquary Tower", "", "0", "Land",
     f"You have no maximum hand size.{NL}{{T}}: Add {{C}}.",
     "", "", "", "", "", "", "Land", "", "", "", "normal", "commander"),

    ("Myriad Landscape", "", "0", "Land",
     f"Myriad Landscape enters tapped.{NL}{{T}}: Add {{C}}.{NL}{{2}}, {{T}}, Sacrifice Myriad Landscape: Search your library for up to two basic land cards that share a land type, put them onto the battlefield tapped, then shuffle.",
     "", "", "", "", "", "", "Land", "", "", "", "normal", "commander"),

    ("Terramorphic Expanse", "", "0", "Land",
     "{T}, Sacrifice Terramorphic Expanse: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.",
     "", "", "", "", "", "", "Land", "", "", "", "normal", "commander"),

    ("Evolving Wilds", "", "0", "Land",
     "{T}, Sacrifice Evolving Wilds: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.",
     "", "", "", "", "", "", "Land", "", "", "", "normal", "commander"),

    ("Mother of Runes", "{W}", "1", "Creature — Human Cleric",
     "{T}: Target creature you control gains protection from the color of your choice until end of turn. Activate only if you control a Plains.",
     "W", "W", "1", "1", "", "", "Creature", "Human,Cleric", "", "", "normal", "commander"),

    ("Sylvan Library", "{1}{G}", "2", "Enchantment",
     "At the beginning of your draw step, you may draw two additional cards. If you do, choose two cards in your hand drawn this turn. For each of those cards, pay 4 life or put the card on top of your library.",
     "G", "G", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),

    ("Wood Elves", "{2}{G}", "3", "Creature — Elf Scout",
     "When Wood Elves enters the battlefield, search your library for a Forest card, put that card onto the battlefield, then shuffle.",
     "G", "G", "1", "1", "", "", "Creature", "Elf,Scout", "", "", "normal", "commander"),

    ("Farhaven Elf", "{2}{G}", "3", "Creature — Elf Druid",
     "When Farhaven Elf enters the battlefield, you may search your library for a basic land card, put that card onto the battlefield tapped, then shuffle.",
     "G", "G", "1", "1", "", "", "Creature", "Elf,Druid", "", "", "normal", "commander"),

    ("Oracle of Mul Daya", "{3}{G}", "4", "Creature — Elf Shaman",
     f"Play with the top card of your library revealed.{NL}You may play an additional land on each of your turns.{NL}You may play lands from the top of your library.",
     "G", "G", "2", "2", "", "", "Creature", "Elf,Shaman", "", "", "normal", "commander"),

    ("Akroma's Will", "{3}{W}{W}", "5", "Instant",
     f"Choose one. Each creature you control gains the chosen abilities until end of turn.{NL}• Flying, double strike, vigilance, and lifelink.{NL}• Protection from each color.",
     "W", "W", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Heroic Intervention", "{1}{G}", "2", "Instant",
     "Permanents you control gain hexproof and indestructible until end of turn.",
     "G", "G", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Teferi's Protection", "{2}{W}", "3", "Instant",
     "Until your next turn, your life total can't change, you gain protection from everything, and all permanents you control phase out.",
     "W", "W", "", "", "", "", "Instant", "", "", "", "normal", "commander"),

    ("Cryptolith Rite", "{1}{G}", "2", "Enchantment",
     'Creatures you control have "{T}: Add one mana of any color."',
     "G", "G", "", "", "", "", "Enchantment", "", "", "", "normal", "commander"),
]

HEADERS = ["name", "manaCost", "manaValue", "type", "text", "colorIdentity",
           "colors", "power", "toughness", "loyalty", "defense", "types",
           "subtypes", "supertypes", "keywords", "layout", "legalities"]


def main():
    out = []
    out.append("# MTG TEST DATABASE - for unit tests")
    out.append("# Not a comprehensive dataset, just enough to test the pipeline")
    out.append("|".join(HEADERS))

    for row in rows:
        assert len(row) == len(HEADERS), f"Row has {len(row)} fields, expected {len(HEADERS)}: {row[0]}"
        # Check no field contains actual newlines or pipes
        for i, field in enumerate(row):
            assert '\n' not in field, f"Actual newline in field {i} of {row[0]!r}"
            assert '|' not in field, f"Pipe in field {i} of {row[0]!r}"
        out.append("|".join(row))

    content = "\n".join(out) + "\n"
    with open("/home/claude/test_cards.csv", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
