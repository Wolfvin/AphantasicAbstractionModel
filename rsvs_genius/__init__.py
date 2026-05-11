"""
RSVS Genius — Cognitive Layers on top of RSVS

"The Genius Who Remembers Everything"

Layers:
1. ContextLayer — Internet search + scope filter + source trust
2. SituationLayer — Chat history as semantic memory
3. PredictiveEngine — Prediction + belief update + anomaly detection
4. PatternOutput — Pattern completion + narrative generation
5. GeniusPipeline — Wire everything together

Bridge:
- RsvsBridge — Unified adapter for PyO3 Rust core and Python fallback
"""

__version__ = "0.2.0"

from .rsvs_bridge import RsvsBridge, get_bridge, is_rust_core_available
from .context_layer import ContextLayer
from .situation_layer import SituationLayer
from .predictive_engine import PredictiveEngine, Prediction, Anomaly, BeliefUpdate
from .pattern_output import PatternOutput, ReasoningStep, PatternResult
from .pipeline import GeniusPipeline, GeniusResponse
