"""
Gate 2: Regime Detection — Layer 2 Validation Gate

Meta-principle from trading:
  "Market behavior changes. Same strategy can work today and die tomorrow."
  Bull market: buy dip works. Crash: buy dip = suicide.
  Regime = current environment. Need detection first.

Applied to AAM:
  Cognitive environment changes. Same reasoning strategy can work
  for one query and fail for another. Need to detect the CURRENT
  regime before choosing a reasoning approach.

  Without this gate: System uses one strategy for everything.
  With this gate: System adapts reasoning based on detected regime.

Regime types:
  - factual:     Pure factual lookup — "What is X?"
  - analytical:  Deep analysis required — "Why does X cause Y?"
  - creative:    Open-ended, generative — "Suggest alternatives for X"
  - crisis:      High-stakes, urgent — "Is this anomaly dangerous?"
  - exploratory: Exploring unknowns — "What might be related to X?"

Analogi:
  Like weather. You don't wear same clothes for sunny, storm, snow.
  Market same. AAM same. Regime determines strategy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CognitiveRegime(Enum):
    """Current cognitive regime — determines reasoning strategy."""
    FACTUAL = "factual"           # Pure factual lookup
    ANALYTICAL = "analytical"     # Deep analysis / causal reasoning
    CREATIVE = "creative"         # Open-ended, generative
    CRISIS = "crisis"             # High-stakes, anomaly-driven
    EXPLORATORY = "exploratory"   # Exploring unknowns
    UNCERTAIN = "uncertain"       # Cannot determine regime


class RegimeVerdict(Enum):
    """Verdict from regime detection gate."""
    STABLE = "stable"         # Regime clearly identified, proceed
    SHIFTING = "shifting"     # Regime changing, adjust strategy
    UNKNOWN = "unknown"       # Cannot determine regime, use cautious defaults


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """Current regime state — the 'weather' of the cognitive environment.

    Attributes:
        regime: The detected cognitive regime.
        confidence: How confident we are in this regime detection.
        indicators: What indicators triggered this regime.
        strategy: Recommended reasoning strategy for this regime.
        risk_level: Risk level for this regime (affects output caution).
        stability: How stable this regime is (0.0 = volatile, 1.0 = stable).
    """
    regime: CognitiveRegime = CognitiveRegime.UNCERTAIN
    confidence: float = 0.5
    indicators: list[str] = field(default_factory=list)
    strategy: str = "default"
    risk_level: float = 0.5
    stability: float = 0.5

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "indicators": self.indicators,
            "strategy": self.strategy,
            "risk_level": round(self.risk_level, 4),
            "stability": round(self.stability, 4),
        }


@dataclass
class RegimeTransition:
    """Record of a regime transition.

    Attributes:
        from_regime: Previous regime.
        to_regime: New regime.
        timestamp: When the transition occurred.
        trigger: What caused the transition.
    """
    from_regime: CognitiveRegime
    to_regime: CognitiveRegime
    timestamp: float = 0.0
    trigger: str = ""

    def to_dict(self) -> dict:
        return {
            "from_regime": self.from_regime.value,
            "to_regime": self.to_regime.value,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
        }


# ---------------------------------------------------------------------------
# Regime Detection Gate
# ---------------------------------------------------------------------------

class RegimeDetectionGate:
    """Gate 2: Regime Detection — Detect current cognitive environment.

    This gate examines the current context and determines what regime
    (environment/mode) the system is operating in. Different regimes
    require different reasoning strategies.

    Analogi:
      Jin Soun tahu kapan dia sedang menghadapi situasi damai
      vs situasi bahaya. Strateginya berbeda:
      - Damai: observasi tenang, analisis mendalam
      - Bahaya: respons cepat, keputusan terbatas
      - Investigasi: pencarian petunjuk, cross-reference intensif

    Regime-strategy mapping:
      FACTUAL:     Direct graph lookup, minimal reasoning
      ANALYTICAL:  Full reasoning chain, causal analysis, MCTS
      CREATIVE:    Relaxed constraints, compositional exploration
      CRISIS:      Conservative, high-evidence threshold, fast response
      EXPLORATORY: Wide recall, cross-reference, anomaly hunting

    Usage:
        gate = RegimeDetectionGate()
        state = gate.detect("Why does rain cause floods?")
        # state.regime = ANALYTICAL, strategy = "deep_causal"
    """

    # Regime-strategy mapping
    _REGIME_STRATEGIES: dict[CognitiveRegime, str] = {
        CognitiveRegime.FACTUAL: "direct_lookup",
        CognitiveRegime.ANALYTICAL: "deep_causal",
        CognitiveRegime.CREATIVE: "compositional_explore",
        CognitiveRegime.CRISIS: "conservative_fast",
        CognitiveRegime.EXPLORATORY: "wide_recall",
        CognitiveRegime.UNCERTAIN: "cautious_default",
    }

    # Regime risk levels
    _REGIME_RISK: dict[CognitiveRegime, float] = {
        CognitiveRegime.FACTUAL: 0.2,
        CognitiveRegime.ANALYTICAL: 0.4,
        CognitiveRegime.CREATIVE: 0.6,
        CognitiveRegime.CRISIS: 0.9,
        CognitiveRegime.EXPLORATORY: 0.5,
        CognitiveRegime.UNCERTAIN: 0.7,
    }

    def __init__(self) -> None:
        self._current_state = RegimeState(regime=CognitiveRegime.UNCERTAIN)
        self._history: list[RegimeState] = []
        self._transitions: list[RegimeTransition] = []
        self._regime_counts: dict[CognitiveRegime, int] = {}

    def detect(
        self,
        query: str,
        context: dict | None = None,
        active_senses: list[dict] | None = None,
        anomalies: list[dict] | None = None,
    ) -> RegimeState:
        """Detect the current cognitive regime.

        Examines the query, context, active senses, and anomalies
        to determine what regime the system is operating in.

        Args:
            query: The current query or trigger.
            context: Optional context dict.
            active_senses: Currently active senses from situation layer.
            anomalies: Currently detected anomalies.

        Returns:
            RegimeState with detected regime and recommended strategy.
        """
        context = context or {}
        active_senses = active_senses or []
        anomalies = anomalies or []

        indicators: list[str] = []
        regime_scores: dict[CognitiveRegime, float] = {
            r: 0.0 for r in CognitiveRegime
        }

        # Signal 1: Query pattern analysis
        query_lower = query.lower().strip()
        query_signals = self._analyze_query_patterns(query_lower)
        for regime, score in query_signals.items():
            regime_scores[regime] += score * 0.4
        indicators.extend(
            f"query_pattern:{r.value}" for r, s in query_signals.items() if s > 0.3
        )

        # Signal 2: Anomaly-driven regime detection
        if anomalies:
            anomaly_severity = len(anomalies) * 0.3
            regime_scores[CognitiveRegime.CRISIS] += min(1.0, anomaly_severity) * 0.3
            indicators.append(f"anomalies_detected:{len(anomalies)}")

        # Signal 3: Active senses density
        if len(active_senses) > 10:
            # Many active senses = exploratory state
            regime_scores[CognitiveRegime.EXPLORATORY] += 0.2
            indicators.append("high_sense_density")

        # Signal 4: Context signals
        if context:
            context_type = context.get("type", "")
            if context_type == "urgent":
                regime_scores[CognitiveRegime.CRISIS] += 0.5
                indicators.append("context:urgent")
            elif context_type == "creative":
                regime_scores[CognitiveRegime.CREATIVE] += 0.5
                indicators.append("context:creative")

        # Determine dominant regime
        best_regime = max(regime_scores, key=lambda r: regime_scores[r])
        best_score = regime_scores[best_regime]

        # If no regime has strong signal, default to UNCERTAIN
        if best_score < 0.15:
            best_regime = CognitiveRegime.UNCERTAIN
            best_score = 0.3

        # Compute confidence
        sorted_scores = sorted(regime_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, 0.5 + margin * 2.0)
        else:
            confidence = 0.5

        # Compute stability
        stability = self._compute_stability(best_regime)

        # Build regime state
        new_state = RegimeState(
            regime=best_regime,
            confidence=confidence,
            indicators=indicators,
            strategy=self._REGIME_STRATEGIES.get(best_regime, "cautious_default"),
            risk_level=self._REGIME_RISK.get(best_regime, 0.5),
            stability=stability,
        )

        # Track regime transition
        if self._current_state.regime != new_state.regime:
            transition = RegimeTransition(
                from_regime=self._current_state.regime,
                to_regime=new_state.regime,
                timestamp=time.time(),
                trigger=query[:100],
            )
            self._transitions.append(transition)
            if len(self._transitions) > 500:
                self._transitions = self._transitions[-200:]

            logger.info(
                "Regime transition: %s → %s (trigger: '%s')",
                transition.from_regime.value,
                transition.to_regime.value,
                query[:60],
            )

        self._current_state = new_state
        self._history.append(new_state)
        if len(self._history) > 500:
            self._history = self._history[-200:]

        self._regime_counts[new_state.regime] = (
            self._regime_counts.get(new_state.regime, 0) + 1
        )

        return new_state

    def _analyze_query_patterns(self, query: str) -> dict[CognitiveRegime, float]:
        """Analyze query text for regime indicators.

        Different query patterns suggest different cognitive regimes:
        - "What is X?" → FACTUAL
        - "Why does X cause Y?" → ANALYTICAL
        - "Suggest/Imagine/Create X" → CREATIVE
        - "Is X dangerous/wrong/broken?" → CRISIS
        - "What might be related to X?" → EXPLORATORY
        """
        scores: dict[CognitiveRegime, float] = {r: 0.0 for r in CognitiveRegime}

        # FACTUAL indicators
        factual_patterns = [
            "what is", "who is", "when did", "where is", "how many",
            "define", "definition of", "meaning of", "tell me about",
            "apa itu", "siapa", "kapan", "dimana", "berapa",
        ]
        for p in factual_patterns:
            if p in query:
                scores[CognitiveRegime.FACTUAL] += 0.5
                break

        # ANALYTICAL indicators
        analytical_patterns = [
            "why does", "how does", "what causes", "explain why",
            "analyze", "compare", "evaluate", "reason for",
            "kenapa", "mengapa", "bagaimana", "analisis",
        ]
        for p in analytical_patterns:
            if p in query:
                scores[CognitiveRegime.ANALYTICAL] += 0.5
                break

        # CREATIVE indicators
        creative_patterns = [
            "suggest", "imagine", "create", "design", "generate",
            "propose", "what if", "alternative", "brainstorm",
            "usulkan", "bayangkan", "buat", "alternatif",
        ]
        for p in creative_patterns:
            if p in query:
                scores[CognitiveRegime.CREATIVE] += 0.5
                break

        # CRISIS indicators
        crisis_patterns = [
            "dangerous", "risk", "threat", "urgent", "emergency",
            "critical", "broken", "wrong", "error", "fail",
            "bahaya", "risiko", "ancaman", "darurat", "gagal",
        ]
        for p in crisis_patterns:
            if p in query:
                scores[CognitiveRegime.CRISIS] += 0.5
                break

        # EXPLORATORY indicators
        exploratory_patterns = [
            "what might", "what could", "related to", "connected to",
            "explore", "investigate", "find connections",
            "hubungan", "kaitan", "selidiki", "jelajahi",
        ]
        for p in exploratory_patterns:
            if p in query:
                scores[CognitiveRegime.EXPLORATORY] += 0.5
                break

        return scores

    def _compute_stability(self, current_regime: CognitiveRegime) -> float:
        """Compute regime stability based on recent history.

        If the regime has been the same for a while, it's stable.
        If it's been changing frequently, it's volatile.
        """
        if len(self._history) < 3:
            return 0.5

        recent = self._history[-10:]
        same_regime_count = sum(1 for s in recent if s.regime == current_regime)
        return same_regime_count / len(recent)

    @property
    def current_regime(self) -> RegimeState:
        """Get the current regime state."""
        return self._current_state

    def get_stats(self) -> dict:
        """Get gate statistics."""
        return {
            "total_detections": len(self._history),
            "total_transitions": len(self._transitions),
            "regime_distribution": {
                r.value: c for r, c in self._regime_counts.items()
            },
            "current_regime": self._current_state.regime.value,
            "current_strategy": self._current_state.strategy,
        }
