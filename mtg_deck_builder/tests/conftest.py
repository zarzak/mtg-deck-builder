"""Shared pytest fixtures for all tests."""

import pytest
from pathlib import Path

from mtg_deck_builder.card_database import CardDatabase
from mtg_deck_builder.models import BuildConfig, CommanderAnalysis
from mtg_deck_builder.llm_engine import LLMEngine, LLMConfig


# Resolve the test CSV path (in repo root)
TEST_CSV = Path(__file__).parent.parent.parent / "test_cards.csv"


@pytest.fixture(scope="session")
def test_csv_path() -> Path:
    """Path to the shared test CSV file."""
    if not TEST_CSV.exists():
        pytest.skip(f"Test CSV not found at {TEST_CSV}")
    return TEST_CSV


@pytest.fixture(scope="session")
def db(test_csv_path) -> CardDatabase:
    """Loaded card database (shared across tests)."""
    db = CardDatabase(test_csv_path)
    db.load()
    return db


@pytest.fixture
def lathiel(db):
    """Lathiel commander card."""
    c = db.get_by_name("Lathiel, the Bounteous Dawn")
    assert c is not None, "Lathiel not in test database"
    return c


@pytest.fixture
def jasmine(db):
    """Jasmine Boreal commander card (vanilla-creatures-matter)."""
    c = db.get_by_name("Jasmine Boreal")
    assert c is not None, "Jasmine Boreal not in test database"
    return c


@pytest.fixture
def karlov(db):
    """Karlov commander card."""
    c = db.get_by_name("Karlov of the Ghost Council")
    assert c is not None, "Karlov not in test database"
    return c


@pytest.fixture
def lathiel_analysis(lathiel) -> CommanderAnalysis:
    """Mock analysis tuned for Lathiel."""
    return CommanderAnalysis(
        name=lathiel.name,
        color_identity=lathiel.color_identity,
        key_mechanics=["lifegain", "+1/+1 counters"],
        build_around_text="Gain life, distribute +1/+1 counters.",
        evaluation_notes="Lifegain synergies are key.",
        category_queries={},
        synergy_keywords=["gain life", "gain 1 life", "lifelink", "+1/+1 counter"],
        anti_synergy_keywords=["lose life"],
    )


@pytest.fixture
def jasmine_analysis(jasmine) -> CommanderAnalysis:
    """Mock analysis tuned for Jasmine (vanilla creatures matter)."""
    return CommanderAnalysis(
        name=jasmine.name,
        color_identity=jasmine.color_identity,
        key_mechanics=["vanilla creatures", "combat"],
        build_around_text="Vanilla creatures become good in this deck.",
        evaluation_notes="Cards like Grizzly Bears (pure stats, no text) are valuable here.",
        category_queries={},
        synergy_keywords=["vanilla"],
        anti_synergy_keywords=[],
        recommended_weights={
            "mana_curve": 0.15,
            "role_coverage": 0.15,
            "synergy": 0.50,
            "power_level": 0.05,
            "creativity": 0.15,
        },
        recommended_synergy_weight=0.80,
    )


@pytest.fixture
def config_lathiel() -> BuildConfig:
    """Small-scale config for quick tests."""
    return BuildConfig(
        commander_name="Lathiel, the Bounteous Dawn",
        population_size=8,
        generations=10,
        patience_generations=20,
        random_seed=42,
    )


@pytest.fixture
def mock_llm() -> LLMEngine:
    """LLM engine in mock mode."""
    return LLMEngine(LLMConfig(mock_mode=True))


@pytest.fixture
def wg_pool(db):
    """Cards legal in W/G color identity (for Lathiel/Jasmine tests)."""
    from mtg_deck_builder.models import Card

    def is_wg_legal(c: Card) -> bool:
        cs = set(ch for ch in (c.color_identity or "") if ch in "WUBRG")
        return cs.issubset({"W", "G"})

    return [c for c in db.all_cards if is_wg_legal(c)]
