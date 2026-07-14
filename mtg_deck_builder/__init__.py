"""
MTG Deck Builder - Hybrid AI + Genetic Algorithm EDH Deck Builder

The current version is in ``__version__`` below. Recent work (v0.9.x) added the
LLM card-power signal, combo/engine detection, structural/attribute synergy
(commander-effect-aware), and the EDHREC distinctive-synergy floor.
"""

from .models import (
    Card,
    Deck,
    DeckScores,
    CardTelemetry,
    BuildConfig,
    CommanderAnalysis,
    OptimizationResult,
    DeckRole,
    CardType,
    WarmStartDeck,
)

from .card_database import (
    CardDatabase,
    QueryResult,
    card_fills_role,
    ROLE_PATTERNS,
)

from .llm_engine import LLMEngine, LLMConfig

from .deck_evaluator import DeckEvaluator, FastEvaluator, is_staple

from .deck_optimizer import DeckOptimizer, PopulationStats, Individual, EvalMode

from .deck_builder import DeckBuilder, BuildProgress, CandidatePool, build_deck

from .html_report import generate_html_report

# Session 3 additions (optional integrations)
from .edhrec_client import (
    EDHRECClient, EDHRECCardData, EDHRECCommanderData,
)
from .embedding_scorer import (
    EmbeddingSynergyScorer, EmbeddingConfig, is_embeddings_available,
)
from .price_source import (
    PriceSource, NullPriceSource, StaticPriceSource, ScryfallPriceSource,
    filter_cards_by_budget, deck_total_price,
)
from .island_optimizer import IslandModelOptimizer, IslandConfig

# v0.4 additions
from .scryfall_cards import ScryfallCardSource

# v0.5 additions
from .scryfall_tags import ScryfallTagClient
from .flavor_tags import FlavorTagScorer
from .deck_diff import diff_decks, format_diff, DiffResult

# v0.6 additions
from .scryfall_bulk import ScryfallBulkFetcher, BulkCardSource
from .oracle_validation import (
    validate_roles, format_role_report,
    ValidationReport, RoleDisagreement,
)

__version__ = "0.9.33"

__all__ = [
    # Models
    "Card",
    "Deck",
    "DeckScores",
    "CardTelemetry",
    "BuildConfig",
    "CommanderAnalysis",
    "OptimizationResult",
    "DeckRole",
    "CardType",
    # Database
    "CardDatabase",
    "QueryResult",
    "card_fills_role",
    "ROLE_PATTERNS",
    # LLM
    "LLMEngine",
    "LLMConfig",
    # Evaluation
    "DeckEvaluator",
    "FastEvaluator",
    "is_staple",
    # Optimization
    "DeckOptimizer",
    "PopulationStats",
    "Individual",
    "EvalMode",
    # Orchestration
    "DeckBuilder",
    "BuildProgress",
    "CandidatePool",
    "build_deck",
    # Reporting
    "generate_html_report",
    # Session 3: EDHREC
    "EDHRECClient",
    "EDHRECCardData",
    "EDHRECCommanderData",
    # Session 3: Embeddings
    "EmbeddingSynergyScorer",
    "EmbeddingConfig",
    "is_embeddings_available",
    # Session 3: Pricing
    "PriceSource",
    "NullPriceSource",
    "StaticPriceSource",
    "ScryfallPriceSource",
    "filter_cards_by_budget",
    "deck_total_price",
    # Session 3: Island model
    "IslandModelOptimizer",
    "IslandConfig",
    # Session 4: Scryfall images
    "ScryfallCardSource",
    # Session 4: Warm-start persistence
    "WarmStartDeck",
    # Session 5: Scryfall Tagger integration
    "ScryfallTagClient",
    "FlavorTagScorer",
    # Session 5: Diff mode
    "diff_decks",
    "format_diff",
    "DiffResult",
    # Session 6: Bulk data
    "ScryfallBulkFetcher",
    "BulkCardSource",
    # Session 6: Oracle-tag role validation
    "validate_roles",
    "format_role_report",
    "ValidationReport",
    "RoleDisagreement",
]
