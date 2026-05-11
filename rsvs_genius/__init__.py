"""
RSVS Genius — Cognitive Layers on top of RSVS

"The Genius Who Remembers Everything"

Layers:
1. ContextLayer — Internet search + scope filter + source trust
2. SituationLayer — Chat history as semantic memory
3. PredictiveEngine — Prediction + belief update + anomaly detection
4. PatternOutput — Pattern completion + narrative generation
5. CoderLayer — Code understanding as structured knowledge (Coder version)
6. PolicyEngine — Rule-based compliance checking for tax/regulation
7. GeniusPipeline — Wire everything together

Bridge:
- RsvsBridge — Unified adapter for PyO3 Rust core and Python fallback
- LLMBridge — Generate natural narrative FROM graph reasoning chain
- WebSearchEngine — Live web search via z-ai-web-dev-sdk with caching
"""

__version__ = "0.5.0"

from .rsvs_bridge import RsvsBridge, get_bridge, is_rust_core_available
from .llm_bridge import generate_narrative, generate_narrative_via_sdk, generate_narrative_fallback
from .context_layer import ContextLayer
from .situation_layer import SituationLayer
from .predictive_engine import PredictiveEngine, Prediction, Anomaly, BeliefUpdate
from .pattern_output import PatternOutput, ReasoningStep, PatternResult
from .coder_layer import CoderLayer, CodeElement, CodeAnalysisResult
from .policy_engine import PolicyEngine, PolicyRule, PolicyViolation
from .web_search import WebSearchEngine
from .pipeline import GeniusPipeline, GeniusResponse
