"""
Gate 4: Statistical Edge — Layer 4 Validation Gate

Meta-principle from trading:
  "Why should this strategy make money at all?"
  Without edge, everything else pointless.
  EV = (win_rate x avg_win) - (loss_rate x avg_loss)
  Positive EV = edge. Negative EV = dead strategy.
  Pretty reasoning != edge. Cool story. Backtest says: negative EV. Trash.

Applied to AAM:
  "Why should this reasoning path be trusted?"
  Before outputting a conclusion, validate that the reasoning
  path has POSITIVE EXPECTED VALUE — i.e., that this type of
  reasoning has historically produced correct conclusions.

  Without this gate: Pretty reasoning can be completely wrong.
  With this gate: Only reasoning paths with statistical backing pass.

Implementation:
  - Track reasoning path types and their historical accuracy
  - Compute expected value for each reasoning path
  - Reject paths with negative EV (no statistical edge)
  - Weight confidence by historical edge strength

Analogi:
  Jin Soun: "Aku mengikuti pola reasoning X. Sejarah menunjukkan
  pola ini benar 70% dari waktu, dengan average confidence 0.6.
  EV = positif. Reasoning ini punya edge."

  Tanpa edge: "Cool story, backtest says: trash."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReasoningPath:
    """A reasoning path type that can be tracked for edge calculation.

    Attributes:
        path_type: The type of reasoning (deduction, induction, etc.).
        regime: The regime this path was used in.
        step_types: The step types in this reasoning chain.
    """
    path_type: str
    regime: str = ""
    step_types: list[str] = field(default_factory=list)

    def cache_key(self) -> str:
        """Generate a cache key for this reasoning path."""
        steps_key = "|".join(sorted(self.step_types)) if self.step_types else "none"
        return f"{self.path_type}:{self.regime}:{steps_key}"


@dataclass
class EdgeAssessment:
    """Result of statistical edge assessment.

    Attributes:
        path: The reasoning path that was assessed.
        has_edge: Whether this path has positive expected value.
        expected_value: The expected value of this reasoning path.
        win_rate: Historical win rate for this path type.
        avg_confidence: Average confidence when this path is correct.
        sample_size: Number of historical observations.
        verdict: PASS (positive edge), CAUTION (marginal edge),
                 REJECT (negative edge or insufficient data).
        reason: Human-readable explanation.
    """
    path: ReasoningPath | None = None
    has_edge: bool = False
    expected_value: float = 0.0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    sample_size: int = 0
    verdict: str = "reject"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "path_type": self.path.path_type if self.path else "unknown",
            "has_edge": self.has_edge,
            "expected_value": round(self.expected_value, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "sample_size": self.sample_size,
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass
class PathOutcome:
    """Record of a reasoning path outcome for tracking.

    Attributes:
        path_key: Cache key of the reasoning path.
        correct: Whether the conclusion was correct.
        confidence: Confidence assigned to the conclusion.
        timestamp: When this outcome was recorded.
    """
    path_key: str
    correct: bool
    confidence: float = 0.5
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Statistical Edge Gate
# ---------------------------------------------------------------------------

class StatisticalEdgeGate:
    """Gate 4: Statistical Edge — Validate reasoning has positive EV.

    This gate checks whether a reasoning path has statistically
    significant positive expected value before allowing its conclusions
    to be output with high confidence.

    Without this gate:
      Pretty reasoning can be completely wrong. AI explanation:
      "This stock appears promising due to strong innovation."
      Cool story. Backtest says: negative EV. Trash.

    With this gate:
      Only reasoning paths with demonstrated positive EV pass.
      New paths get cautious defaults until proven.

    Edge sources (in cognitive terms):
      - Momentum anomaly: Recent patterns tend to continue
      - Mean reversion: Extreme patterns tend to correct
      - Causal chains: Cause-effect reasoning has high accuracy
      - Anomaly-driven: Anomalies predict important conclusions
      - Compositional: Composing known facts has moderate accuracy

    Usage:
        gate = StatisticalEdgeGate()

        # Before outputting a conclusion:
        path = ReasoningPath(path_type="deduction", step_types=["extract", "compose", "ground"])
        assessment = gate.assess(path)
        if assessment.verdict == "reject":
            # This reasoning path has no statistical edge — be cautious
    """

    # Minimum sample size before we trust the statistics
    _MIN_SAMPLE_SIZE = 5

    # Minimum EV to consider a path as having edge
    _MIN_EDGE_EV = 0.05

    # Default win rate for untested paths (conservative)
    _DEFAULT_WIN_RATE = 0.4

    # Default confidence for untested paths
    _DEFAULT_CONFIDENCE = 0.3

    def __init__(self) -> None:
        self._outcomes: dict[str, list[PathOutcome]] = {}
        self._total_assessments: int = 0

    def assess(
        self,
        path: ReasoningPath,
        current_confidence: float = 0.5,
    ) -> EdgeAssessment:
        """Assess whether a reasoning path has statistical edge.

        Computes the expected value of this reasoning path based on
        historical outcomes. If the path has positive EV, it passes.
        If negative EV or insufficient data, it's flagged or rejected.

        EV = (win_rate x avg_confidence_when_correct) -
             (loss_rate x avg_confidence_when_wrong)

        Simplified: EV = win_rate - (1 - win_rate) = 2*win_rate - 1
        Positive EV means win_rate > 0.5.

        Args:
            path: The reasoning path to assess.
            current_confidence: Current confidence in the conclusion.

        Returns:
            EdgeAssessment with verdict and EV metrics.
        """
        self._total_assessments += 1
        path_key = path.cache_key()

        # Get historical outcomes for this path type
        outcomes = self._outcomes.get(path_key, [])
        sample_size = len(outcomes)

        # If insufficient data, use conservative defaults
        if sample_size < self._MIN_SAMPLE_SIZE:
            # Use priors based on path type
            prior_win_rate = self._get_prior_win_rate(path.path_type)
            ev = 2 * prior_win_rate - 1

            return EdgeAssessment(
                path=path,
                has_edge=ev > 0,
                expected_value=ev,
                win_rate=prior_win_rate,
                avg_confidence=self._DEFAULT_CONFIDENCE,
                sample_size=sample_size,
                verdict="caution",
                reason=(
                    f"Insufficient data ({sample_size} samples) for path "
                    f"'{path.path_type}'. Using prior win rate {prior_win_rate:.0%}. "
                    f"EV={ev:.3f}. More observations needed."
                ),
            )

        # Compute historical statistics
        wins = [o for o in outcomes if o.correct]
        losses = [o for o in outcomes if not o.correct]

        win_rate = len(wins) / sample_size if sample_size > 0 else 0.0
        avg_conf_when_correct = (
            sum(o.confidence for o in wins) / len(wins) if wins else 0.0
        )
        avg_conf_when_wrong = (
            sum(o.confidence for o in losses) / len(losses) if losses else 0.0
        )

        # Compute expected value
        # EV = P(correct) * avg_conf_correct - P(wrong) * avg_conf_wrong
        ev = (win_rate * avg_conf_when_correct) - ((1 - win_rate) * avg_conf_when_wrong)

        # Also compute simplified EV for comparison
        simplified_ev = 2 * win_rate - 1

        # Use the more conservative estimate
        conservative_ev = min(ev, simplified_ev)

        # Determine verdict
        if conservative_ev >= self._MIN_EDGE_EV and sample_size >= self._MIN_SAMPLE_SIZE:
            verdict = "pass"
            reason = (
                f"Path '{path.path_type}' has positive edge: "
                f"EV={conservative_ev:.3f}, win_rate={win_rate:.0%}, "
                f"sample_size={sample_size}. Reasoning path is statistically valid."
            )
        elif conservative_ev > -self._MIN_EDGE_EV and sample_size >= self._MIN_SAMPLE_SIZE:
            verdict = "caution"
            reason = (
                f"Path '{path.path_type}' has marginal edge: "
                f"EV={conservative_ev:.3f}, win_rate={win_rate:.0%}, "
                f"sample_size={sample_size}. Proceed with reduced confidence."
            )
        else:
            verdict = "reject"
            reason = (
                f"Path '{path.path_type}' has no edge: "
                f"EV={conservative_ev:.3f}, win_rate={win_rate:.0%}, "
                f"sample_size={sample_size}. "
                f"Pretty reasoning != edge. This path is not statistically valid."
            )

        return EdgeAssessment(
            path=path,
            has_edge=conservative_ev > 0,
            expected_value=conservative_ev,
            win_rate=win_rate,
            avg_confidence=avg_conf_when_correct,
            sample_size=sample_size,
            verdict=verdict,
            reason=reason,
        )

    def record_outcome(
        self,
        path: ReasoningPath,
        correct: bool,
        confidence: float = 0.5,
    ) -> None:
        """Record whether a reasoning path produced a correct conclusion.

        This is how we build the statistical database for edge calculation.
        Without recording outcomes, we can never know if a reasoning
        path actually has edge.

        Args:
            path: The reasoning path that was used.
            correct: Whether the conclusion was actually correct.
            confidence: The confidence that was assigned.
        """
        path_key = path.cache_key()
        outcome = PathOutcome(
            path_key=path_key,
            correct=correct,
            confidence=confidence,
            timestamp=time.time(),
        )

        if path_key not in self._outcomes:
            self._outcomes[path_key] = []
        self._outcomes[path_key].append(outcome)

        # Keep bounded
        if len(self._outcomes[path_key]) > 1000:
            self._outcomes[path_key] = self._outcomes[path_key][-500:]

        logger.debug(
            "Edge outcome recorded: path='%s', correct=%s, confidence=%.2f",
            path.path_type, correct, confidence,
        )

    def _get_prior_win_rate(self, path_type: str) -> float:
        """Get prior win rate for a path type (before we have data).

        These are conservative priors based on the general reliability
        of different reasoning types.

        Deduction and causal reasoning are generally more reliable
        than induction and abduction.
        """
        priors = {
            "deduction": 0.55,
            "induction": 0.45,
            "abduction": 0.40,
            "anomaly_driven": 0.50,
            "composition": 0.45,
            "analogy": 0.40,
            "causal": 0.55,
            "pattern": 0.45,
        }
        return priors.get(path_type, self._DEFAULT_WIN_RATE)

    def get_stats(self) -> dict:
        """Get gate statistics."""
        total_outcomes = sum(len(v) for v in self._outcomes.values())
        path_types = len(self._outcomes)

        return {
            "total_assessments": self._total_assessments,
            "total_outcomes_recorded": total_outcomes,
            "path_types_tracked": path_types,
            "paths_with_edge": sum(
                1 for k, v in self._outcomes.items()
                if len(v) >= self._MIN_SAMPLE_SIZE
                and sum(1 for o in v if o.correct) / len(v) > 0.5
            ),
        }
