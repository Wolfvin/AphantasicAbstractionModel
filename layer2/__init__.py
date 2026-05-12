"""
AAM Layer 2 — Cognitive Runtime

Aphantasic Abstraction Model — knowledge without imagery.

Modules:
  context   : Scoped knowledge + internet search + source trust
  situation : Chat history as semantic graph memory
  predictive: Prediction → observation → belief update → anomaly detection
  pattern   : Pattern completion + narrative generation from graph

Bridge:
  bridge    : Unified adapter for PyO3 Rust core (AbstractionBridge)
  llm       : Generate natural narrative FROM graph reasoning chain
  web_search: Live web search with caching
"""

__version__ = "1.0.0-alpha"

from .bridge import AbstractionBridge, RsvsBridge, get_bridge, is_rust_core_available
from .embedding import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    OpenAIProvider,
    FallbackEmbeddingProvider,
    get_embedding_provider,
    cosine_similarity,
)
from .llm import generate_narrative, generate_narrative_via_sdk, generate_narrative_fallback
from .context import ContextLayer
from .situation import SituationLayer
from .predictive import PredictiveEngine, Prediction, Anomaly, BeliefUpdate
from .pattern import PatternOutput, ReasoningStep, PatternResult
from .web_search import WebSearchEngine
