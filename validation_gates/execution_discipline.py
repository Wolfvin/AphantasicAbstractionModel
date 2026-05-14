"""
Gate 5: Execution Discipline — Layer 5 Validation Gate

Meta-principle from trading:
  "Even good strategy can die here. Because execution != theory."
  Strategy says buy at 100, actually fill at 101.8. Profit destroyed.
  Killers: slippage, fees, latency, emotion, risk violation.

  Discipline means: always follow system.
  - Stop losses respected
  - Size limits respected
  - No revenge trading, no FOMO, no random overrides

Applied to AAM:
  "Even good reasoning can die at output. Because output != reasoning."
  Reasoning says confidence 60%, output says "DEFINITELY TRUE!".
  Killers: hallucination, overstatement, missing caveats,
  ignoring evidence gaps, overriding system signals.

  Discipline means: always follow the system's signals.
  - Confidence caps respected (don't overstate)
  - Evidence minimums respected (don't answer without evidence)
  - No hallucination (don't fabricate beyond evidence)
  - No override (don't ignore anomaly flags)
  - Risk limits respected (don't output in crisis mode without high confidence)

Implementation:
  - Check confidence against output threshold
  - Verify minimum evidence requirements
  - Enforce regime-specific output rules
  - Detect and prevent hallucination indicators
  - Apply risk-adjusted output limits

Analogi:
  Jin Soun: "Aku 60% yakin. Jangan bilang 'pasti benar'.
  Bilang 'kemungkinan besar, tapi perlu verifikasi'."
  Tanpa disiplin: "PASTI BENAR! 100%!" — YOLO.
  Dengan disiplin: Output menghormati signal. No override.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DisciplineRule:
    """A rule that must be followed for disciplined execution.

    Attributes:
        rule_id: Unique identifier for this rule.
        description: Human-readable description.
        category: Category of the rule (confidence, evidence, risk, etc.).
        severity: "error" (block output), "warning" (flag but proceed).
        check: A callable that returns True if the rule is satisfied.
    """
    rule_id: str
    description: str
    category: str = "general"
    severity: str = "error"
    check: Any = None  # Callable that returns (passed: bool, message: str)


@dataclass
class DisciplineVerdict:
    """Result of execution discipline gate evaluation.

    Attributes:
        allowed: Whether the output is allowed to proceed.
        confidence_cap: Maximum confidence allowed for this output.
        required_caveats: Caveats that must be included in the output.
        violations: Rules that were violated.
        warnings: Rules that generated warnings.
        adjusted_confidence: Confidence after discipline adjustments.
        hallucination_risk: Estimated hallucination risk (0.0-1.0).
        verdict: PASS (all rules satisfied), ADJUST (adjustments made),
                 BLOCK (output blocked by hard rules).
        reason: Human-readable explanation.
    """
    allowed: bool = True
    confidence_cap: float = 1.0
    required_caveats: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    adjusted_confidence: float = 0.5
    hallucination_risk: float = 0.0
    verdict: str = "pass"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "confidence_cap": round(self.confidence_cap, 4),
            "required_caveats": self.required_caveats,
            "violations": self.violations,
            "warnings": self.warnings,
            "adjusted_confidence": round(self.adjusted_confidence, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "verdict": self.verdict,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Execution Discipline Gate
# ---------------------------------------------------------------------------

class ExecutionDisciplineGate:
    """Gate 5: Execution Discipline — Enforce output rules.

    This gate ensures that the final output follows the system's
    signals, regardless of how "tempting" it might be to overstate
    or override. This is the last checkpoint before output.

    Without this gate:
      "Strategy says sell, but maybe rebound" → BOOM.
      "Reasoning says 60% confident, but let me say DEFINITELY" → YOLO.

    With this gate:
      Output respects confidence levels.
      Evidence minimums are enforced.
      Risk limits are respected.
      No hallucination, no override, no FOMO.

    Rules enforced:
      1. CONFIDENCE_CAP: Cannot output above confidence cap
      2. MIN_EVIDENCE: Must have minimum evidence to output
      3. HALLUCINATION_CHECK: Output must not go beyond evidence
      4. REGIME_RISK: Crisis mode requires higher confidence to output
      5. ANOMALY_RESPECT: Cannot ignore anomaly flags
      6. CAVEAT_ENFORCEMENT: Low confidence must include caveats

    Usage:
        gate = ExecutionDisciplineGate()
        verdict = gate.enforce(
            confidence=0.6,
            evidence_count=3,
            output_text="It is definitely true that...",
            regime="factual",
        )
        if not verdict.allowed:
            # Output blocked — must revise
    """

    # Minimum evidence nodes required to output
    _MIN_EVIDENCE_NODES = 1

    # Minimum confidence required to output at all
    _MIN_CONFIDENCE_TO_OUTPUT = 0.1

    # Confidence thresholds for caveat requirements
    _CAVEAT_THRESHOLD_LOW = 0.4
    _CAVEAT_THRESHOLD_MEDIUM = 0.6

    # Crisis mode requires higher confidence
    _CRISIS_MIN_CONFIDENCE = 0.7

    # Hallucination risk patterns
    _HALLUCINATION_PATTERNS = [
        r"\bdefinitely\b",
        r"\babsolutely\b",
        r"\bcertainly\b",
        r"\bwithout doubt\b",
        r"\b100%\b",
        r"\bguaranteed\b",
        r"\bproven\b",
        r"\bunquestionably\b",
    ]

    def __init__(self) -> None:
        self._enforcement_history: list[DisciplineVerdict] = []
        self._custom_rules: list[DisciplineRule] = []

    def enforce(
        self,
        confidence: float,
        evidence_count: int = 0,
        output_text: str = "",
        regime: str = "",
        anomalies: list[dict] | None = None,
        reasoning_steps: int = 0,
        calibrated_confidence: float | None = None,
    ) -> DisciplineVerdict:
        """Enforce discipline rules on the proposed output.

        Checks the output against all discipline rules and returns
        a verdict indicating whether the output is allowed, needs
        adjustment, or should be blocked.

        Args:
            confidence: Raw confidence from reasoning.
            evidence_count: Number of evidence nodes backing the output.
            output_text: The proposed output text.
            regime: Current cognitive regime.
            anomalies: List of detected anomalies.
            reasoning_steps: Number of reasoning steps.
            calibrated_confidence: Confidence after calibration (if available).

        Returns:
            DisciplineVerdict with enforcement results.
        """
        anomalies = anomalies or []
        effective_confidence = calibrated_confidence or confidence

        violations: list[dict] = []
        warnings: list[dict] = []
        caveats: list[str] = []
        confidence_cap = 1.0
        hallucination_risk = 0.0

        # ---- Rule 1: CONFIDENCE CAP ----
        # Cannot exceed confidence cap based on evidence
        if evidence_count == 0:
            confidence_cap = 0.3
            caveats.append("No direct evidence — low confidence cap applied.")
            warnings.append({
                "rule": "CONFIDENCE_CAP",
                "message": "No evidence nodes — confidence capped at 0.3",
                "severity": "warning",
            })
        elif evidence_count < 3:
            confidence_cap = 0.7
            caveats.append("Limited evidence — moderate confidence cap.")
        elif evidence_count < 5:
            confidence_cap = 0.85
        # 5+ evidence = no cap

        # ---- Rule 2: MINIMUM EVIDENCE ----
        if evidence_count < self._MIN_EVIDENCE_NODES:
            violations.append({
                "rule": "MIN_EVIDENCE",
                "message": f"Insufficient evidence ({evidence_count} < {self._MIN_EVIDENCE_NODES})",
                "severity": "error",
            })

        # ---- Rule 3: MINIMUM CONFIDENCE ----
        if effective_confidence < self._MIN_CONFIDENCE_TO_OUTPUT:
            violations.append({
                "rule": "MIN_CONFIDENCE",
                "message": f"Confidence too low ({effective_confidence:.2f} < {self._MIN_CONFIDENCE_TO_OUTPUT})",
                "severity": "error",
            })

        # ---- Rule 4: HALLUCINATION CHECK ----
        hallucination_risk = self._assess_hallucination_risk(
            output_text, effective_confidence, evidence_count,
        )
        if hallucination_risk > 0.7:
            violations.append({
                "rule": "HALLUCINATION_CHECK",
                "message": f"High hallucination risk ({hallucination_risk:.2f}): "
                           f"output goes beyond evidence",
                "severity": "error",
            })
        elif hallucination_risk > 0.4:
            warnings.append({
                "rule": "HALLUCINATION_CHECK",
                "message": f"Moderate hallucination risk ({hallucination_risk:.2f})",
                "severity": "warning",
            })
            caveats.append("Claims should be verified against available evidence.")

        # ---- Rule 5: REGIME RISK ----
        if regime == "crisis" and effective_confidence < self._CRISIS_MIN_CONFIDENCE:
            violations.append({
                "rule": "REGIME_RISK",
                "message": f"Crisis regime requires {self._CRISIS_MIN_CONFIDENCE:.0%} "
                           f"confidence, got {effective_confidence:.0%}",
                "severity": "error",
            })
            caveats.append("HIGH RISK: Low-confidence output in crisis mode.")
        elif regime == "crisis":
            caveats.append("Crisis mode — exercise caution with this output.")

        # ---- Rule 6: ANOMALY RESPECT ----
        if anomalies and effective_confidence > 0.7:
            warnings.append({
                "rule": "ANOMALY_RESPECT",
                "message": f"{len(anomalies)} anomaly(ies) detected — "
                           f"confidence should be reduced",
                "severity": "warning",
            })
            # Reduce confidence cap when anomalies are present
            anomaly_penalty = min(0.3, len(anomalies) * 0.1)
            confidence_cap = max(0.3, confidence_cap - anomaly_penalty)
            caveats.append("Anomalies detected — conclusions should be treated as provisional.")

        # ---- Rule 7: CAVEAT ENFORCEMENT ----
        if effective_confidence < self._CAVEAT_THRESHOLD_LOW:
            caveats.append("LOW CONFIDENCE: This conclusion is uncertain and should be verified.")
        elif effective_confidence < self._CAVEAT_THRESHOLD_MEDIUM:
            caveats.append("MODERATE CONFIDENCE: Additional evidence may change this conclusion.")

        # ---- Custom rules ----
        for rule in self._custom_rules:
            if rule.check:
                try:
                    passed, message = rule.check(confidence, evidence_count, output_text)
                    if not passed:
                        if rule.severity == "error":
                            violations.append({
                                "rule": rule.rule_id,
                                "message": message,
                                "severity": rule.severity,
                            })
                        else:
                            warnings.append({
                                "rule": rule.rule_id,
                                "message": message,
                                "severity": rule.severity,
                            })
                except Exception as exc:
                    logger.debug("Custom rule %s failed: %s", rule.rule_id, exc)

        # ---- Compute adjusted confidence ----
        adjusted = min(effective_confidence, confidence_cap)
        if anomalies:
            anomaly_penalty = min(0.3, len(anomalies) * 0.05)
            adjusted = max(0.1, adjusted - anomaly_penalty)
        if hallucination_risk > 0.3:
            adjusted *= (1.0 - hallucination_risk * 0.3)

        # ---- Determine verdict ----
        if violations:
            verdict_str = "block"
            allowed = False
            reason = (
                f"Output BLOCKED: {len(violations)} violation(s). "
                + "; ".join(v.get("message", "") for v in violations[:3])
            )
        elif warnings or confidence_cap < effective_confidence:
            verdict_str = "adjust"
            allowed = True
            reason = (
                f"Output ADJUSTED: confidence capped at {confidence_cap:.0%}, "
                f"{len(warnings)} warning(s), {len(caveats)} caveat(s) required."
            )
        else:
            verdict_str = "pass"
            allowed = True
            reason = "Output passes all discipline rules."

        result = DisciplineVerdict(
            allowed=allowed,
            confidence_cap=confidence_cap,
            required_caveats=caveats,
            violations=violations,
            warnings=warnings,
            adjusted_confidence=adjusted,
            hallucination_risk=hallucination_risk,
            verdict=verdict_str,
            reason=reason,
        )

        # Track history
        self._enforcement_history.append(result)
        if len(self._enforcement_history) > 1000:
            self._enforcement_history = self._enforcement_history[-500:]

        logger.debug(
            "ExecutionDisciplineGate: verdict=%s, confidence=%.2f→%.2f, "
            "evidence=%d, hallucination_risk=%.2f",
            verdict_str, effective_confidence, adjusted,
            evidence_count, hallucination_risk,
        )

        return result

    def add_rule(self, rule: DisciplineRule) -> None:
        """Add a custom discipline rule."""
        self._custom_rules.append(rule)

    # ------------------------------------------------------------------
    # Hallucination assessment
    # ------------------------------------------------------------------

    def _assess_hallucination_risk(
        self,
        output_text: str,
        confidence: float,
        evidence_count: int,
    ) -> float:
        """Assess the risk that the output is hallucinating.

        Hallucination risk indicators:
        1. Overconfident language (definitely, absolutely, etc.)
        2. Confidence higher than evidence supports
        3. Output contains specific claims not backed by evidence
        4. Output goes beyond the available information

        Returns a risk score between 0.0 (no risk) and 1.0 (high risk).
        """
        risk = 0.0

        if not output_text:
            return 0.0

        # Check 1: Overconfident language
        overconfident_count = 0
        for pattern in self._HALLUCINATION_PATTERNS:
            if re.search(pattern, output_text, re.IGNORECASE):
                overconfident_count += 1

        if overconfident_count > 0:
            risk += min(0.4, overconfident_count * 0.15)

        # Check 2: Confidence-evidence mismatch
        if evidence_count <= 1 and confidence > 0.7:
            risk += 0.3  # High confidence with little evidence = suspicious

        if evidence_count == 0 and confidence > 0.5:
            risk += 0.3  # Any confidence without evidence = suspicious

        # Check 3: Excessive specificity without evidence
        # Count specific claims (numbers, names, dates)
        specific_claims = len(re.findall(r'\b\d+\.?\d*\b', output_text))
        if specific_claims > 3 and evidence_count < 3:
            risk += 0.15

        return max(0.0, min(1.0, risk))

    def get_stats(self) -> dict:
        """Get gate statistics."""
        total = len(self._enforcement_history)
        if total == 0:
            return {"total_enforcements": 0}

        pass_count = sum(1 for v in self._enforcement_history if v.verdict == "pass")
        adjust_count = sum(1 for v in self._enforcement_history if v.verdict == "adjust")
        block_count = sum(1 for v in self._enforcement_history if v.verdict == "block")

        return {
            "total_enforcements": total,
            "pass_count": pass_count,
            "adjust_count": adjust_count,
            "block_count": block_count,
            "block_rate": block_count / total,
            "avg_hallucination_risk": (
                sum(v.hallucination_risk for v in self._enforcement_history) / total
            ),
        }
