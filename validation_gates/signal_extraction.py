"""
Gate 1: Signal Extraction — Layer 0/1 Validation Gate

Meta-principle from trading:
  "Mengambil sinyal bermakna dari noise market."
  Market = chaos. Tidak semua movement = opportunity.
  Signal extraction = "mana informasi yang benar-benar punya predictive value?"

Applied to AAM:
  Raw input = chaos. Tidak semua data = meaningful signal.
  Gate ini membedakan SINYAL dari NOISE sebelum masuk ke RSVS graph.

  Tanpa gate ini: AI cuma lihat "chart goes brrrr" — semua data dianggap sama.
  Dengan gate ini: Hanya informasi yang punya predictive value yang masuk graph.

Implementation:
  - Signal quality scoring: berapa banyak relasi bermakna yang bisa diekstrak?
  - Noise ratio: berapa banyak data yang tidak membentuk relasi?
  - Signal confidence: seberapa kuat sinyal yang diekstrak?
  - Multi-signal convergence: apakah multiple signals mengkonfirmasi satu sama lain?

Gate verdict:
  PASS  → data masuk ke RSVS graph
  WEAK  → data masuk tapi dengan reduced confidence
  REJECT → data ditolak — hanya noise, tidak ada sinyal
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

class SignalVerdict(Enum):
    """Verdict from signal extraction gate."""
    PASS = "pass"        # Strong signal — ingest with full confidence
    WEAK = "weak"        # Weak signal — ingest with reduced confidence
    REJECT = "reject"    # No meaningful signal — do not ingest


class SignalType(Enum):
    """Classification of signal type extracted from input."""
    RELATIONAL = "relational"     # Subject-predicate relation extracted
    CATEGORICAL = "categorical"   # "X is Y" type classification
    CAUSAL = "causal"             # "X causes Y" causal chain
    TEMPORAL = "temporal"         # Time-based relation
    COMPARATIVE = "comparative"   # "X is more/less than Y"
    FACTUAL = "factual"           # Raw fact without clear relation


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSignal:
    """A single extracted signal from raw input.

    Unlike a PerceptualTuple (which is a structured relation),
    an ExtractedSignal captures the SIGNAL QUALITY — is this
    information that has predictive value, or is it noise?

    Attributes:
        content: The signal content (text).
        signal_type: What type of signal this is.
        confidence: Signal confidence (0.0-1.0).
        predictive_value: Estimated predictive value of this signal.
        source_modality: Where this signal came from.
        converges_with: Other signals that confirm this one.
    """
    content: str
    signal_type: SignalType = SignalType.FACTUAL
    confidence: float = 0.5
    predictive_value: float = 0.5
    source_modality: str = "text"
    converges_with: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "signal_type": self.signal_type.value,
            "confidence": round(self.confidence, 4),
            "predictive_value": round(self.predictive_value, 4),
            "source_modality": self.source_modality,
            "converges_with": self.converges_with,
        }


@dataclass
class SignalResult:
    """Result of signal extraction gate evaluation.

    The complete result of running raw input through the signal
    extraction gate. Contains the verdict, extracted signals,
    noise ratio, and quality metrics.

    Attributes:
        verdict: PASS, WEAK, or REJECT.
        signals: List of extracted signals.
        noise_ratio: Fraction of input deemed noise (0.0-1.0).
        signal_quality: Overall signal quality score (0.0-1.0).
        convergence_score: How much signals confirm each other (0.0-1.0).
        confidence_modifier: Multiplier for downstream confidence.
        reason: Human-readable explanation of the verdict.
    """
    verdict: SignalVerdict = SignalVerdict.REJECT
    signals: list[ExtractedSignal] = field(default_factory=list)
    noise_ratio: float = 1.0
    signal_quality: float = 0.0
    convergence_score: float = 0.0
    confidence_modifier: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "signals": [s.to_dict() for s in self.signals],
            "noise_ratio": round(self.noise_ratio, 4),
            "signal_quality": round(self.signal_quality, 4),
            "convergence_score": round(self.convergence_score, 4),
            "confidence_modifier": round(self.confidence_modifier, 4),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Signal Extraction Gate
# ---------------------------------------------------------------------------

class SignalExtractionGate:
    """Gate 1: Signal Extraction — Filter signal from noise.

    This gate examines raw input before it enters the RSVS graph
    and determines whether it contains meaningful signal or is
    primarily noise. This prevents the graph from being polluted
    with low-quality, uninformative data.

    Analogi:
      Market = radio penuh static noise (zzzzzzzz).
      Signal extraction = menemukan suara manusia di tengah noise.

      BTC naik 0.5% = bisa jadi noise (random fluctuation).
      BTC breakout + volume 4x + funding neutral + macro bullish = SIGNAL.

      Tanpa gate ini, AAM memproses SEMUA input sama —
      "chart goes brrrr" dan "Federal Reserve rate decision" diperlakukan sama.
      Dengan gate ini, hanya yang punya predictive value yang masuk.

    Usage:
        gate = SignalExtractionGate()
        result = gate.evaluate("BTC breakout resistance with 4x volume")
        if result.verdict != SignalVerdict.REJECT:
            # Proceed with ingestion
            confidence = result.confidence_modifier
        else:
            # Skip — noise only
    """

    # Minimum signal quality to pass
    _MIN_QUALITY_PASS = 0.4

    # Minimum signal quality for WEAK verdict
    _MIN_QUALITY_WEAK = 0.15

    # Minimum number of signals to consider input non-trivial
    _MIN_SIGNAL_COUNT = 1

    def __init__(
        self,
        min_quality_pass: float = 0.4,
        min_quality_weak: float = 0.15,
    ) -> None:
        self._min_quality_pass = min_quality_pass
        self._min_quality_weak = min_quality_weak

        # Historical tracking for adaptive thresholds
        self._history: list[SignalResult] = []
        self._signal_acceptance_rate: float = 0.5

    def evaluate(
        self,
        raw_input: str,
        perceptual_tuples: list | None = None,
        context: dict | None = None,
    ) -> SignalResult:
        """Evaluate raw input for signal quality.

        Examines the input and determines:
        1. How many meaningful signals can be extracted?
        2. What fraction of the input is noise?
        3. Do the signals converge (confirm each other)?
        4. What is the overall signal quality?

        Args:
            raw_input: The raw text input to evaluate.
            perceptual_tuples: Optional pre-extracted PerceptualTuples
                from Layer 0. If provided, these are used to assess
                signal quality more accurately.
            context: Optional context dict with additional metadata.

        Returns:
            SignalResult with verdict and quality metrics.
        """
        context = context or {}

        # Step 1: Extract signals from raw input
        signals = self._extract_signals(raw_input, perceptual_tuples)

        # Step 2: Compute noise ratio
        noise_ratio = self._compute_noise_ratio(raw_input, signals)

        # Step 3: Assess signal quality
        signal_quality = self._compute_signal_quality(signals, raw_input)

        # Step 4: Check signal convergence
        convergence_score = self._compute_convergence(signals)

        # Step 5: Determine verdict
        verdict, confidence_modifier, reason = self._determine_verdict(
            signals, noise_ratio, signal_quality, convergence_score,
        )

        result = SignalResult(
            verdict=verdict,
            signals=signals,
            noise_ratio=noise_ratio,
            signal_quality=signal_quality,
            convergence_score=convergence_score,
            confidence_modifier=confidence_modifier,
            reason=reason,
        )

        # Track history
        self._history.append(result)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        logger.debug(
            "SignalExtractionGate: verdict=%s, quality=%.3f, noise=%.3f, "
            "convergence=%.3f, signals=%d",
            verdict.value, signal_quality, noise_ratio,
            convergence_score, len(signals),
        )

        return result

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def _extract_signals(
        self,
        raw_input: str,
        perceptual_tuples: list | None = None,
    ) -> list[ExtractedSignal]:
        """Extract meaningful signals from raw input.

        Strategy:
        1. If PerceptualTuples are available, use their relation types
           as primary signal indicators
        2. Also extract signals from text patterns (causal words,
           comparative words, etc.)
        3. Score each signal for predictive value
        """
        signals: list[ExtractedSignal] = []

        # Strategy 1: Extract from PerceptualTuples
        if perceptual_tuples:
            for t in perceptual_tuples:
                # Each PerceptualTuple is already a relation = potential signal
                relation_type = getattr(t, 'relation_type', None)
                if relation_type is not None:
                    rt_value = relation_type.value if hasattr(relation_type, 'value') else str(relation_type)
                    signal_type = self._map_relation_to_signal(rt_value)
                    confidence = getattr(t, 'confidence', 0.5)
                    subject = getattr(t, 'subject', '')
                    predicate = getattr(t, 'predicate', '')
                    content = f"{subject} {rt_value} {predicate}" if subject and predicate else str(t)

                    # Causal and temporal signals have higher predictive value
                    pv = self._estimate_predictive_value(signal_type, confidence)

                    signals.append(ExtractedSignal(
                        content=content,
                        signal_type=signal_type,
                        confidence=confidence,
                        predictive_value=pv,
                    ))

        # Strategy 2: Extract from text patterns
        text_signals = self._extract_text_signals(raw_input)
        for ts in text_signals:
            # Avoid duplicates from tuple extraction
            if not any(s.content == ts.content for s in signals):
                signals.append(ts)

        return signals

    def _extract_text_signals(self, text: str) -> list[ExtractedSignal]:
        """Extract signals from text patterns.

        Looks for:
        - Causal indicators ("causes", "leads to", "results in")
        - Comparative indicators ("more than", "less than", "higher")
        - Temporal indicators ("before", "after", "during", "when")
        - Categorical indicators ("is a", "are", "classified as")
        - Quantitative indicators (numbers, percentages, ratios)
        """
        signals: list[ExtractedSignal] = []
        text_lower = text.lower()

        # Causal signals — highest predictive value
        causal_patterns = [
            "causes", "leads to", "results in", "produces",
            "triggers", "creates", "generates", "enables",
            "prevents", "blocks", "inhibits", "reduces",
        ]
        for pattern in causal_patterns:
            if pattern in text_lower:
                # Extract the clause containing the causal relation
                idx = text_lower.index(pattern)
                start = max(0, idx - 40)
                end = min(len(text), idx + len(pattern) + 40)
                clause = text[start:end].strip()

                signals.append(ExtractedSignal(
                    content=clause,
                    signal_type=SignalType.CAUSAL,
                    confidence=0.7,
                    predictive_value=0.8,  # Causal = high predictive value
                ))
                break  # One causal signal per pattern group

        # Comparative signals
        comparative_patterns = [
            "more than", "less than", "higher", "lower",
            "greater", "smaller", "faster", "slower",
            "stronger", "weaker", "compared to", "versus",
        ]
        for pattern in comparative_patterns:
            if pattern in text_lower:
                idx = text_lower.index(pattern)
                start = max(0, idx - 30)
                end = min(len(text), idx + len(pattern) + 30)
                clause = text[start:end].strip()

                signals.append(ExtractedSignal(
                    content=clause,
                    signal_type=SignalType.COMPARATIVE,
                    confidence=0.6,
                    predictive_value=0.6,
                ))
                break

        # Temporal signals
        temporal_patterns = [
            "before", "after", "during", "when", "while",
            "since", "until", "followed by", "preceded by",
        ]
        for pattern in temporal_patterns:
            if pattern in text_lower:
                idx = text_lower.index(pattern)
                start = max(0, idx - 30)
                end = min(len(text), idx + len(pattern) + 30)
                clause = text[start:end].strip()

                signals.append(ExtractedSignal(
                    content=clause,
                    signal_type=SignalType.TEMPORAL,
                    confidence=0.6,
                    predictive_value=0.7,  # Temporal = good for prediction
                ))
                break

        # Categorical signals — lower predictive value but still valid
        categorical_patterns = ["is a", "are", "classified as", "type of", "kind of"]
        for pattern in categorical_patterns:
            if pattern in text_lower:
                idx = text_lower.index(pattern)
                start = max(0, idx - 30)
                end = min(len(text), idx + len(pattern) + 30)
                clause = text[start:end].strip()

                signals.append(ExtractedSignal(
                    content=clause,
                    signal_type=SignalType.CATEGORICAL,
                    confidence=0.5,
                    predictive_value=0.4,
                ))
                break

        # Quantitative signals — numbers, percentages
        import re
        numbers = re.findall(r'\d+\.?\d*%?', text)
        if numbers:
            signals.append(ExtractedSignal(
                content=f"Quantitative data: {', '.join(numbers[:5])}",
                signal_type=SignalType.FACTUAL,
                confidence=0.7,
                predictive_value=0.6,
            ))

        # If no specific signals found, check for general content
        if not signals and len(text.strip()) > 10:
            # There's content but no clear signal pattern
            # This is likely factual/declarative — low predictive value
            signals.append(ExtractedSignal(
                content=text[:100],
                signal_type=SignalType.FACTUAL,
                confidence=0.3,
                predictive_value=0.2,
            ))

        return signals

    # ------------------------------------------------------------------
    # Quality computations
    # ------------------------------------------------------------------

    def _compute_noise_ratio(
        self, raw_input: str, signals: list[ExtractedSignal],
    ) -> float:
        """Compute noise ratio — what fraction of input is noise.

        High noise ratio = most of the input doesn't carry meaningful signal.
        Low noise ratio = most of the input contributes to signal extraction.

        Calculation:
        - signal_coverage = total characters covered by signals / total input length
        - noise_ratio = 1.0 - signal_coverage
        """
        if not raw_input or not signals:
            return 1.0

        total_len = len(raw_input)
        if total_len == 0:
            return 1.0

        # Estimate signal coverage
        covered_chars = 0
        for signal in signals:
            # Each signal covers approximately its content length
            covered_chars += len(signal.content)

        # Avoid double-counting overlapping signals
        signal_coverage = min(1.0, covered_chars / total_len)
        noise_ratio = 1.0 - signal_coverage

        return max(0.0, min(1.0, noise_ratio))

    def _compute_signal_quality(
        self, signals: list[ExtractedSignal], raw_input: str,
    ) -> float:
        """Compute overall signal quality.

        Signal quality depends on:
        1. Number of signals (more = better, up to a point)
        2. Average predictive value of signals
        3. Average confidence of signals
        4. Signal diversity (different types = better)

        Returns a score between 0.0 and 1.0.
        """
        if not signals:
            return 0.0

        # Factor 1: Signal count (diminishing returns)
        count_score = min(1.0, len(signals) / 5.0)

        # Factor 2: Average predictive value
        avg_pv = sum(s.predictive_value for s in signals) / len(signals)

        # Factor 3: Average confidence
        avg_conf = sum(s.confidence for s in signals) / len(signals)

        # Factor 4: Signal diversity
        signal_types = set(s.signal_type for s in signals)
        diversity_score = min(1.0, len(signal_types) / 3.0)

        # Weighted combination
        quality = (
            0.15 * count_score +
            0.35 * avg_pv +
            0.30 * avg_conf +
            0.20 * diversity_score
        )

        return max(0.0, min(1.0, quality))

    def _compute_convergence(self, signals: list[ExtractedSignal]) -> float:
        """Compute signal convergence — do signals confirm each other?

        Convergence happens when multiple signals point to the same
        conclusion or involve overlapping entities.

        High convergence = strong signal (multiple indicators agree).
        Low convergence = potentially conflicting or unrelated signals.

        Analogi:
          BTC breakout + volume 4x + funding neutral = HIGH convergence.
          Random price move + random news + random whale = LOW convergence.
        """
        if len(signals) <= 1:
            return 0.0

        # Check for shared entities between signals
        convergence_count = 0
        total_pairs = 0

        for i, sig_a in enumerate(signals):
            for sig_b in signals[i + 1:]:
                total_pairs += 1
                # Check content overlap (shared words beyond stop words)
                words_a = set(sig_a.content.lower().split())
                words_b = set(sig_b.content.lower().split())
                shared = words_a & words_b - {
                    "the", "a", "an", "is", "are", "was", "were",
                    "and", "or", "but", "in", "on", "at", "to",
                    "of", "for", "with", "from", "by", "as",
                }
                if shared:
                    convergence_count += 1

                # Same signal type = partial convergence
                if sig_a.signal_type == sig_b.signal_type and sig_a.signal_type != SignalType.FACTUAL:
                    convergence_count += 0.5

        if total_pairs == 0:
            return 0.0

        return min(1.0, convergence_count / total_pairs)

    # ------------------------------------------------------------------
    # Verdict determination
    # ------------------------------------------------------------------

    def _determine_verdict(
        self,
        signals: list[ExtractedSignal],
        noise_ratio: float,
        signal_quality: float,
        convergence_score: float,
    ) -> tuple[SignalVerdict, float, str]:
        """Determine the gate verdict.

        Returns:
            Tuple of (verdict, confidence_modifier, reason).
        """
        if not signals:
            return (
                SignalVerdict.REJECT,
                0.0,
                "No meaningful signals extracted from input — pure noise.",
            )

        # Compute composite score
        composite = (
            0.4 * signal_quality +
            0.3 * (1.0 - noise_ratio) +
            0.3 * convergence_score
        )

        if composite >= self._min_quality_pass:
            verdict = SignalVerdict.PASS
            confidence_modifier = min(1.0, 0.5 + composite * 0.5)
            reason = (
                f"Strong signal detected: quality={signal_quality:.2f}, "
                f"noise={noise_ratio:.2f}, convergence={convergence_score:.2f}. "
                f"{len(signals)} signal(s) extracted with high predictive value."
            )
        elif composite >= self._min_quality_weak:
            verdict = SignalVerdict.WEAK
            confidence_modifier = max(0.1, composite * 0.7)
            reason = (
                f"Weak signal: quality={signal_quality:.2f}, "
                f"noise={noise_ratio:.2f}, convergence={convergence_score:.2f}. "
                f"Proceeding with reduced confidence."
            )
        else:
            verdict = SignalVerdict.REJECT
            confidence_modifier = 0.0
            reason = (
                f"Insufficient signal: quality={signal_quality:.2f}, "
                f"noise={noise_ratio:.2f}, convergence={convergence_score:.2f}. "
                f"Input is primarily noise — rejected."
            )

        return verdict, confidence_modifier, reason

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _map_relation_to_signal(relation_type: str) -> SignalType:
        """Map PerceptualTuple RelationType to SignalType."""
        mapping = {
            "categorical": SignalType.CATEGORICAL,
            "differential": SignalType.COMPARATIVE,
            "functional": SignalType.FACTUAL,
            "spatial": SignalType.FACTUAL,
            "temporal": SignalType.TEMPORAL,
            "causal": SignalType.CAUSAL,
        }
        return mapping.get(relation_type, SignalType.FACTUAL)

    @staticmethod
    def _estimate_predictive_value(signal_type: SignalType, confidence: float) -> float:
        """Estimate predictive value based on signal type and confidence.

        Causal and temporal signals have inherently higher predictive value
        because they describe cause-effect relationships and temporal
        sequences that enable forecasting.

        Categorical and factual signals have lower predictive value
        because they describe static properties that don't imply
        future outcomes.
        """
        base_pv = {
            SignalType.CAUSAL: 0.8,
            SignalType.TEMPORAL: 0.7,
            SignalType.COMPARATIVE: 0.6,
            SignalType.RELATIONAL: 0.5,
            SignalType.CATEGORICAL: 0.4,
            SignalType.FACTUAL: 0.3,
        }
        return base_pv.get(signal_type, 0.3) * confidence

    def get_stats(self) -> dict:
        """Get gate statistics."""
        total = len(self._history)
        if total == 0:
            return {"total_evaluations": 0}

        pass_count = sum(1 for r in self._history if r.verdict == SignalVerdict.PASS)
        weak_count = sum(1 for r in self._history if r.verdict == SignalVerdict.WEAK)
        reject_count = sum(1 for r in self._history if r.verdict == SignalVerdict.REJECT)

        return {
            "total_evaluations": total,
            "pass_count": pass_count,
            "weak_count": weak_count,
            "reject_count": reject_count,
            "acceptance_rate": (pass_count + weak_count) / total,
            "avg_signal_quality": sum(r.signal_quality for r in self._history) / total,
            "avg_noise_ratio": sum(r.noise_ratio for r in self._history) / total,
        }
