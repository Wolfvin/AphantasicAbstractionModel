"""
AAM Layer 2 — Cognitive Runtime

Aphantasic Abstraction Model — knowledge without imagery.

Modules:
  context       : Scoped knowledge + internet search + source trust
  scope_control : Hierarchical scope management for RSVS graph and cognitive runtime
  chat_index    : Semantic chat index — conversations as a graph of meaning
  situation     : Chat history as semantic graph memory
  predictive    : Prediction → observation → belief update → anomaly detection
  prediction_loop: Explicit predict/observe/update lifecycle with state tracking
  pattern       : Pattern completion + narrative generation from graph
  temporal      : Temporal tracking layer — when did things happen and what's still relevant?
  possibility_generator: Enumerate all possible interpretations from RSVS graph
  hypothesis_combinator: Combine complementary hypotheses into hybrids (A × B)
  coder_layer   : Code understanding as structured knowledge

Bridge:
  bridge    : Unified adapter for PyO3 Rust core (V12PipelineBridge)
  llm       : Generate natural narrative FROM graph reasoning chain
  web_search: Live web search with caching
"""

__version__ = "1.1.0-alpha"

from .bridge import (
    V12PipelineBridge,
    AbstractionBridge,  # backward-compat alias for V12PipelineBridge
    RsvsBridge,         # backward-compat alias for V12PipelineBridge
    get_bridge,
    is_rust_core_available,
)
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
from .prediction_loop import PredictionLoop, CycleResult, CycleTracker
from .pattern import PatternOutput, ReasoningStep, PatternResult
from .temporal import TemporalTracker, TemporalRecord
from .scope_control import ScopeControl, ScopeConfig, ScopeAuditEntry
from .chat_index import SemanticChatIndex, ChatNode, ChatEdge, ConversationGraph
from .web_search import WebSearchEngine
from .possibility_generator import PossibilityGenerator, GeneratedPossibility
from .hypothesis_combinator import HypothesisCombinator, HybridResult, ComplementarityScore
