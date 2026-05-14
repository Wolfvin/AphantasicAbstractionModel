"""
Gate 3: Uncertainty Calibration — Layer 3 Validation Gate

Meta-principle from trading:
  "Seberapa yakin kita?"
  Model bilang "BUY NVDA" — tapi confidence 51% atau 95%?
  Huge difference. Kalau confidence salah calibrated = bad.

  Overconfidence: Model says 95%, reality 55%. Dangerous.
  Underconfidence: Model says 55%, reality 80%. Miss opportunities.

  Good calibration: If model says 70%, across 100 similar cases,
  ~70 should be correct. That's calibration.

Applied to AAM:
  Every reasoning step produces a confidence score. But is that
  confidence CALIBRATED? Does 70% confidence actually mean 70%
  accuracy over historical cases?

  Without this gate: Confidence is just a number — not grounded in reality.
  With this gate: Confidence is calibrated against historical accuracy.

  Why important? Because position sizing (response boldness).
  High confidence: detailed, assertive response.
  Low confidence: cautious, hedged response.
  No calibration: YOLO.

Implementation:
  - Track historical confidence vs actual correctness
  - Compute calibration curve (expected vs observed accuracy)
  - Apply calibration correction to raw confidence scores
  - Flag overconfident and underconfident patterns
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRecord:
    """Record of a single confidence-accuracy observation.

    Used to build the calibration curve: for each confidence bucket,
    what was the actual accuracy?

    Attributes:
        predicted_confidence: The confidence the model assigned.
        actual_correct: Whether the prediction was actually correct.
        timestamp: When this observation was recorded.
        context: What context this observation was in.
        regime: What regime this observation was in.
    """
    predicted_confidence: float
    actual_correct: bool
    timestamp: float = 0.0
    context: str = ""
    regime: str = ""

    def to_dict(self) -> dict:
        return {
            "predicted_confidence": round(self.predicted_confidence, 4),
            "actual_correct": self.actual_correct,
            "timestamp": self.timestamp,
            "context": self.context,
            "regime": self.regime,
        }


@dataclass
class CalibrationCurve:
    """Calibration curve — expected vs observed accuracy.

    Binned by confidence level, shows what the actual accuracy
    was for each confidence bucket.

    Attributes:
        buckets: Dict mapping bucket center → (predicted, observed, count).
        total_observations: Total number of calibration records.
        calibration_error: Expected Calibration Error (ECE).
    """
    buckets: dict[float, dict] = field(default_factory=dict)
    total_observations: int = 0
    calibration_error: float = 0.0

    def to_dict(self) -> dict:
        return {
            "buckets": {str(k): v for k, v in self.buckets.items()},
            "total_observations": self.total_observations,
            "calibration_error": round(self.calibration_error, 4),
        }


@dataclass
class CalibrationResult:
    """Result of uncertainty calibration gate evaluation.

    Attributes:
        raw_confidence: The original confidence from reasoning.
        calibrated_confidence: The calibrated confidence after correction.
        calibration_applied: Whether calibration correction was applied.
        overconfidence_flag: True if system tends to be overconfident.
        underconfidence_flag: True if system tends to be underconfident.
        confidence_bucket: Which bucket this confidence falls in.
        bucket_accuracy: Historical accuracy for this bucket.
        verdict: PASS (well-calibrated), ADJUST (correction applied),
                 FLAG (significant miscalibration).
        reason: Human-readable explanation.
    """
    raw_confidence: float = 0.5
    calibrated_confidence: float = 0.5
    calibration_applied: bool = False
    overconfidence_flag: bool = False
    underconfidence_flag: bool = False
    confidence_bucket: float = 0.5
    bucket_accuracy: float = 0.5
    verdict: str = "pass"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "calibration_applied": self.calibration_applied,
            "overconfidence_flag": self.overconfidence_flag,
            "underconfidence_flag": self.underconfidence_flag,
            "confidence_bucket": round(self.confidence_bucket, 2),
            "bucket_accuracy": round(self.bucket_accuracy, 4),
            "verdict": self.verdict,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Uncertainty Calibration Gate
# ---------------------------------------------------------------------------

class UncertaintyCalibrationGate:
    """Gate 3: Uncertainty Calibration — Calibrate confidence against reality.

    This gate tracks whether the system's confidence scores are
    well-calibrated — i.e., when the system says "70% confident",
    is it actually correct about 70% of the time?

    Without calibration, confidence is just a number. With calibration,
    confidence is grounded in historical accuracy, enabling:

    - Proper response boldness (high confidence = assertive)
    - Risk-adjusted decisions (low confidence = cautious)
    - Overconfidence detection (dangerous! prevent YOLO reasoning)

    Analogi:
      Jin Soun berkata "Aku 90% yakin Ju Jangmok adalah pencuri."
      Tapi sejarah menunjukkan: setiap kali dia bilang 90% yakin,
      dia cuma bener 60% dari waktu. OVERCONFIDENT.
      Setelah kalibrasi: "Aku 60% yakin" — jauh lebih jujur.

    Usage:
        gate = UncertaintyCalibrationGate()

        # After reasoning produces a confidence score:
        result = gate.calibrate(raw_confidence=0.9)

        # Later, when you know if the prediction was correct:
        gate.record_outcome(predicted_confidence=0.9, actual_correct=False)
    """

    # Number of confidence buckets for calibration curve
    _NUM_BUCKETS = 10

    # Minimum observations before calibration kicks in
    _MIN_OBSERVATIONS = 5

    # Threshold for flagging overconfidence
    _OVERCONFIDENCE_THRESHOLD = 0.2

    # Threshold for flagging underconfidence
    _UNDERCONFIDENCE_THRESHOLD = 0.2

    def __init__(self) -> None:
        self._records: list[CalibrationRecord] = []
        self._curve: CalibrationCurve | None = None
        self._global_bias: float = 0.0  # Positive = overconfident, negative = underconfident

    def calibrate(
        self,
        raw_confidence: float,
        regime: str = "",
        context: str = "",
    ) -> CalibrationResult:
        """Calibrate a raw confidence score against historical accuracy.

        If we have enough historical data, we adjust the raw confidence
        based on the calibration curve. If the system historically
        overestimates at this confidence level, we reduce it.

        Args:
            raw_confidence: The raw confidence from reasoning (0.0-1.0).
            regime: Current cognitive regime (for regime-specific calibration).
            context: Context description (for context-specific calibration).

        Returns:
            CalibrationResult with calibrated confidence and flags.
        """
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # Find the confidence bucket
        bucket = self._get_bucket(raw_confidence)

        # Get historical accuracy for this bucket
        bucket_accuracy = self._get_bucket_accuracy(bucket)

        # Apply calibration if we have enough data
        calibration_applied = False
        calibrated = raw_confidence

        if len(self._records) >= self._MIN_OBSERVATIONS:
            # Compute calibration correction
            correction = bucket_accuracy - raw_confidence

            # Apply smoothing — don't over-correct based on limited data
            bucket_count = self._get_bucket_count(bucket)
            smooth_factor = min(1.0, bucket_count / 50.0)  # Full correction at 50+ observations

            if abs(correction) > 0.05:  # Only correct if significant
                calibrated = raw_confidence + correction * smooth_factor
                calibrated = max(0.0, min(1.0, calibrated))
                calibration_applied = True

        # Detect overconfidence/underconfidence
        overconfidence = (bucket_accuracy - raw_confidence) < -self._OVERCONFIDENCE_THRESHOLD
        underconfidence = (bucket_accuracy - raw_confidence) > self._UNDERCONFIDENCE_THRESHOLD

        # Determine verdict
        if not calibration_applied:
            verdict = "pass"
            reason = "Insufficient calibration data — using raw confidence."
        elif abs(calibrated - raw_confidence) < 0.1:
            verdict = "pass"
            reason = f"Well-calibrated at {raw_confidence:.0%} (bucket accuracy: {bucket_accuracy:.0%})."
        elif abs(calibrated - raw_confidence) < 0.2:
            verdict = "adjust"
            reason = (
                f"Minor calibration: {raw_confidence:.0%} → {calibrated:.0%} "
                f"(bucket accuracy: {bucket_accuracy:.0%})."
            )
        else:
            verdict = "flag"
            reason = (
                f"Significant miscalibration: {raw_confidence:.0%} → {calibrated:.0%} "
                f"(bucket accuracy: {bucket_accuracy:.0%}). "
                f"{'OVERCONFIDENT' if overconfidence else 'UNDERCONFIDENT'}."
            )

        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_applied=calibration_applied,
            overconfidence_flag=overconfidence,
            underconfidence_flag=underconfidence,
            confidence_bucket=bucket,
            bucket_accuracy=bucket_accuracy,
            verdict=verdict,
            reason=reason,
        )

    def record_outcome(
        self,
        predicted_confidence: float,
        actual_correct: bool,
        regime: str = "",
        context: str = "",
    ) -> None:
        """Record whether a prediction was correct for calibration tracking.

        This is how the calibration curve is built — by recording
        what happened for each confidence level.

        Args:
            predicted_confidence: The confidence that was assigned.
            actual_correct: Whether the prediction was actually correct.
            regime: The regime at the time of prediction.
            context: Context description.
        """
        record = CalibrationRecord(
            predicted_confidence=predicted_confidence,
            actual_correct=actual_correct,
            timestamp=time.time(),
            context=context,
            regime=regime,
        )
        self._records.append(record)

        # Keep bounded
        if len(self._records) > 5000:
            self._records = self._records[-2000:]

        # Recompute calibration curve periodically
        if len(self._records) % 10 == 0:
            self._recompute_curve()

        logger.debug(
            "Calibration recorded: predicted=%.2f, actual=%s, regime=%s",
            predicted_confidence, actual_correct, regime,
        )

    def get_calibration_curve(self) -> CalibrationCurve:
        """Get the current calibration curve.

        Returns:
            CalibrationCurve with bucket-by-bucket accuracy data.
        """
        if self._curve is None:
            self._recompute_curve()
        return self._curve or CalibrationCurve()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _get_bucket(self, confidence: float) -> float:
        """Get the bucket center for a confidence value."""
        bucket_size = 1.0 / self._NUM_BUCKETS
        bucket_idx = min(int(confidence / bucket_size), self._NUM_BUCKETS - 1)
        return (bucket_idx + 0.5) * bucket_size

    def _get_bucket_accuracy(self, bucket: float) -> float:
        """Get historical accuracy for a confidence bucket."""
        if not self._records:
            return 0.5  # No data = assume neutral

        bucket_size = 1.0 / self._NUM_BUCKETS
        bucket_min = bucket - bucket_size / 2
        bucket_max = bucket + bucket_size / 2

        in_bucket = [
            r for r in self._records
            if bucket_min <= r.predicted_confidence < bucket_max
        ]

        if not in_bucket:
            # No data for this bucket — interpolate from neighbors
            return self._interpolate_bucket_accuracy(bucket)

        correct_count = sum(1 for r in in_bucket if r.actual_correct)
        return correct_count / len(in_bucket)

    def _get_bucket_count(self, bucket: float) -> int:
        """Get number of observations in a confidence bucket."""
        bucket_size = 1.0 / self._NUM_BUCKETS
        bucket_min = bucket - bucket_size / 2
        bucket_max = bucket + bucket_size / 2

        return sum(
            1 for r in self._records
            if bucket_min <= r.predicted_confidence < bucket_max
        )

    def _interpolate_bucket_accuracy(self, bucket: float) -> float:
        """Interpolate bucket accuracy from neighboring buckets."""
        bucket_size = 1.0 / self._NUM_BUCKETS

        # Find nearest buckets with data
        for offset in [1, -1, 2, -2]:
            neighbor = bucket + offset * bucket_size
            if 0 <= neighbor <= 1.0:
                count = self._get_bucket_count(neighbor)
                if count > 0:
                    return self._get_bucket_accuracy(neighbor)

        # No data at all — return bucket center as best guess
        return bucket

    def _recompute_curve(self) -> None:
        """Recompute the full calibration curve."""
        bucket_size = 1.0 / self._NUM_BUCKETS
        buckets: dict[float, dict] = {}
        total_error = 0.0
        total_count = 0

        for i in range(self._NUM_BUCKETS):
            bucket_center = (i + 0.5) * bucket_size
            bucket_min = i * bucket_size
            bucket_max = (i + 1) * bucket_size

            in_bucket = [
                r for r in self._records
                if bucket_min <= r.predicted_confidence < bucket_max
            ]

            if in_bucket:
                predicted = bucket_center
                observed = sum(1 for r in in_bucket if r.actual_correct) / len(in_bucket)
                count = len(in_bucket)
                error = abs(predicted - observed)

                buckets[bucket_center] = {
                    "predicted": round(predicted, 4),
                    "observed": round(observed, 4),
                    "count": count,
                    "error": round(error, 4),
                }

                total_error += error * count
                total_count += count

        ece = total_error / total_count if total_count > 0 else 0.0

        self._curve = CalibrationCurve(
            buckets=buckets,
            total_observations=len(self._records),
            calibration_error=ece,
        )

    def get_stats(self) -> dict:
        """Get gate statistics."""
        curve = self.get_calibration_curve()
        return {
            "total_records": len(self._records),
            "calibration_error": curve.calibration_error,
            "global_bias": round(self._global_bias, 4),
            "buckets_with_data": len(curve.buckets),
        }
