"""
Predictive Coding Engine — Prediction + Belief Update + Anomaly Detection

Analogi: Jin Soun memprediksi "Ju Jangmok adalah pencuri" →
observasi menunjukkan "tidak ada yang mengonsumsi pil" →
PREDICTION ERROR → update belief → "Ju Jangmok bukan pencuri, dia kambing hitam."

Flow:
1. Prediction: Based on context, predict what compositions a concept should have
2. Observation: Ingest reality → compare with prediction
3. Belief Update: Confirm/Revise based on prediction error
4. Anomaly Detection: When |predicted - observed| > threshold → flag anomaly

Formula: belief_new = belief_old + η × (observed - predicted)
This IS Friston's free energy minimization.

The PredictiveEngine maintains an internal ledger of predictions and their
outcomes. When observations contradict predictions, beliefs are revised
according to the Rescorla-Wagner / Friston update rule, and anomalies
are flagged for downstream pattern completion.

Analogi: Jin Soun di Simhyeon Pavilion — setiap prediksi dicatat,
setiap observasi dibandingkan, dan setiap anomali ditandai dengan
benang merah. Bukan sekadar mengingat, tapi BELAJAR dari kesalahan prediksi.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import AbstractionBridge, RsvsBridge, get_bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

# Learning rate — same default as RSVS core
_DEFAULT_ETA = 0.1

# How much prediction error before we flag an anomaly
_DEFAULT_ANOMALY_THRESHOLD = 0.3

# Maximum age (seconds) for an active prediction before it expires
_PREDICTION_STALENESS_LIMIT = 600.0  # 10 minutes

# Stop words for fallback keyword extraction
_STOP_WORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "they",
    "their", "which", "would", "there", "could", "about",
    "other", "into", "more", "than", "then", "some", "very",
    "also", "just", "like", "only", "over", "such", "after",
    "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
    "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """A prediction about what compositions a concept should have.

    Created by PredictiveEngine.predict(), stored in the internal
    prediction log, and later compared against observations.

    Analogi: Jin Soun mencatat prediksi "Ju Jangmok = pencuri"
    di buku catatannya, lengkap dengan alasan dan tingkat keyakinan.
    Ketika bukti baru datang, dia membandingkan dengan catatan ini.

    Attributes:
        concept: The concept being predicted.
        expected_compositions: What compositions we expect this concept to have.
        confidence: How confident we are in this prediction (0.0 - 1.0).
        context: What context atoms triggered this prediction.
        timestamp: When this prediction was made (ISO-ish string).
    """

    concept: str
    expected_compositions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    context: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "concept": self.concept,
            "expected_compositions": list(self.expected_compositions),
            "confidence": self.confidence,
            "context": list(self.context),
            "timestamp": self.timestamp,
        }


@dataclass
class Anomaly:
    """An anomaly detected when prediction diverges from observation.

    Created by PredictiveEngine.detect_anomalies() when the gap
    between expected and observed exceeds the anomaly threshold.

    Analogi: Jin Soun menemukan bahwa "tidak ada yang mengonsumsi pil"
    padahal prediksinya "pencuri pasti mengonsumsi pil" → ANOMALI.
    Benang merah ditarik di catatan, siap untuk pattern completion.

    Attributes:
        concept: Where the anomaly was found.
        expected: What we expected to see.
        observed: What we actually saw.
        delta: Magnitude of the prediction error.
        description: Human-readable explanation of the anomaly.
    """

    concept: str
    expected: list[str] = field(default_factory=list)
    observed: list[str] = field(default_factory=list)
    delta: float = 0.0
    description: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "concept": self.concept,
            "expected": list(self.expected),
            "observed": list(self.observed),
            "delta": self.delta,
            "description": self.description,
        }


@dataclass
class BeliefUpdate:
    """A belief update resulting from comparing prediction with observation.

    Created by PredictiveEngine.observe_and_update() when new evidence
    is ingested and compared against active predictions.

    The update follows the Rescorla-Wagner rule:
        belief_new = belief_old + η × (observed - predicted)

    This IS Friston's free energy minimization — we reduce surprise
    by updating our internal model to match observations.

    Analogi: Jin Soun awalnya yakin 80% bahwa Ju Jangmok = pencuri.
    Setelah melihat bukti "tidak ada konsumsi pil", keyakinan turun ke 30%.
    Dia mengubah catatannya: "mungkin bukan pencuri, mungkin kambing hitam."

    Attributes:
        concept: What concept was updated.
        old_confidence: Confidence before the update.
        new_confidence: Confidence after the update.
        direction: "confirm" if confidence increased, "revise" if decreased.
        reason: Why this update happened.
        evidence: What evidence triggered this update.
    """

    concept: str
    old_confidence: float = 0.5
    new_confidence: float = 0.5
    direction: str = "confirm"  # "confirm" or "revise"
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "concept": self.concept,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "direction": self.direction,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# PredictiveEngine
# ---------------------------------------------------------------------------

class PredictiveEngine:
    """Predictive Coding Engine — Prediction + Belief Update + Anomaly Detection.

    Implements a predictive coding loop on top of the RSVS semantic graph:
    1. Predict what compositions a concept should have
    2. Observe reality by ingesting new text
    3. Update beliefs based on prediction error
    4. Detect anomalies when prediction error exceeds threshold

    This IS Friston's free energy minimization, implemented as a
    cognitive layer on top of RSVS. The RSVS graph provides the
    generative model; this engine provides the predictive coding loop.

    Analogi: Jin Soun tidak hanya mengingat — dia MEMPREDIKSI.
    "Jika Ju Jangmok pencuri, dia pasti mengonsumsi pil."
    Ketika prediksi salah, dia belajar. Ini bukan sekadar memory,
    ini active inference — belajar dari kesalahan prediksi.

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core is being used (via bridge).
        eta: Learning rate for belief updates.
        anomaly_threshold: Prediction error threshold for anomaly detection.
    """

    def __init__(
        self,
        rsvs_instance: Any | None = None,
        bridge: Optional[RsvsBridge] = None,
        eta: float = _DEFAULT_ETA,
        anomaly_threshold: float = _DEFAULT_ANOMALY_THRESHOLD,
    ) -> None:
        """Initialize the Predictive Coding Engine.

        Args:
            rsvs_instance: Optional pre-built RSVS instance. If None,
                the engine will try to obtain one via the RsvsBridge.
            bridge: Optional pre-built RsvsBridge instance. If provided,
                takes precedence over rsvs_instance.
            eta: Learning rate for belief updates (default 0.1, same as RSVS).
                Higher values = faster learning but less stable.
            anomaly_threshold: How much prediction error before flagging
                anomaly (default 0.3). Lower values = more sensitive.
        """
        if bridge is not None:
            self._bridge = bridge
        elif rsvs_instance is not None:
            self._bridge = RsvsBridge(rsvs_instance=rsvs_instance)
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core

        self.eta = eta
        self.anomaly_threshold = anomaly_threshold

        # Internal state
        # Analogi: Buku catatan Jin Soun — semua prediksi, pembaruan
        # keyakinan, dan anomali dicatat dengan tanggal dan alasan.
        self._predictions: list[Prediction] = []
        self._belief_updates: list[BeliefUpdate] = []
        self._anomalies: list[Anomaly] = []

        # Observed compositions cache — maps concept → list of observed compositions
        # Populated during observe_and_update()
        self._observed: dict[str, list[str]] = {}

        # P2-7: Removed self._fallback_graph — now delegates to self._bridge
        # which always has a _FallbackGraph when Rust core is unavailable.

        if self.rsvs_available:
            logger.info(
                "PredictiveEngine initialized with RSVS bridge "
                "(rust_core=%s, eta=%.3f, threshold=%.3f)",
                self.is_rust_core, eta, anomaly_threshold
            )
        else:
            logger.info(
                "PredictiveEngine initialized WITHOUT RSVS core "
                "(eta=%.3f, threshold=%.3f, fallback mode)", eta, anomaly_threshold
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, concept: str, context: list[str] | None = None
    ) -> Prediction:
        """Predict what compositions a concept should have, given context.

        Uses the RSVS graph as a generative model:
        - senses() gives current compositions → used as prediction
        - context_query() refines prediction based on active context
        - Confidence derived from RSVS confidence_map()

        Without RSVS, uses fallback keyword matching from the internal
        fallback graph.

        Analogi: Jin Soun mendengar "Snow Plum Pill" dan konteks "pencurian".
        Otomatis dia memprediksi: komposisi pil ini berkaitan dengan
        pencurian, dan pencurinya pasti ada motif konsumsi.

        Args:
            concept: The concept to predict compositions for.
            context: Optional list of context atoms that should influence
                the prediction. If provided, context_query() is used
                to refine the prediction.

        Returns:
            A Prediction object with expected compositions, confidence,
            and context.
        """
        context = context or []
        expected: list[str] = []
        confidence = 0.5  # Start neutral

        if self.rsvs_available:
            try:
                # Strategy 1: Use context_query() if context provided
                if context:
                    try:
                        cq_result = self._bridge.context_query(concept, context)
                        if cq_result:
                            expected = self._parse_compositions(cq_result)
                            confidence = 0.7  # Context-refined = more confident
                    except Exception as exc:
                        logger.debug("context_query() failed, falling back: %s", exc)

                # Strategy 2: Use senses() as the generative model prediction
                if not expected:
                    try:
                        senses = self._bridge.senses(concept)
                        if senses:
                            expected = self._parse_compositions(senses)
                            confidence = 0.6
                    except Exception as exc:
                        logger.debug("senses() failed: %s", exc)

                # Strategy 3: Use relate() to find connected compositions
                if not expected:
                    try:
                        relate_result = self._bridge.relate(concept)
                        if relate_result:
                            expected = self._parse_compositions(relate_result)
                            confidence = 0.4  # Less direct = less confident
                    except Exception as exc:
                        logger.debug("relate() failed: %s", exc)

                # Strategy 4 (L2-02): Use mcts_query() for complex prediction paths
                # MCTS explores deeper reasoning paths that simple query/relate miss
                if not expected or confidence < 0.5:
                    try:
                        mcts_result = self._bridge.mcts_query(concept, max_depth=3, simulations=50)
                        if mcts_result:
                            mcts_atoms = mcts_result.get("scored_atoms", [])
                            if mcts_atoms:
                                for atom_entry in mcts_atoms[:10]:
                                    label = self._extract_label_from_tuple(atom_entry)
                                    if label and label not in expected:
                                        expected.append(label)
                                # Boost confidence if MCTS found results
                                confidence = max(confidence, 0.5)
                    except Exception as exc:
                        logger.debug("mcts_query() failed for prediction: %s", exc)

                # Try to get confidence from confidence_map()
                try:
                    cmap = self._bridge.confidence_map()
                    if concept in cmap:
                        confidence = max(confidence, float(cmap[concept]))
                except Exception:
                    pass

            except Exception as exc:
                logger.warning("RSVS prediction failed for '%s': %s", concept, exc)
        else:
            # Fallback mode — use internal fallback graph
            expected, confidence = self._fallback_predict(concept, context)

        # Create and store the prediction
        prediction = Prediction(
            concept=concept,
            expected_compositions=expected,
            confidence=confidence,
            context=context,
        )

        self._predictions.append(prediction)
        logger.debug(
            "Prediction: '%s' → %s (confidence=%.3f, context=%s)",
            concept, expected, confidence, context
        )

        return prediction

    # ------------------------------------------------------------------
    # Observation + Belief Update
    # ------------------------------------------------------------------

    def observe_and_update(
        self, text: str, source: str = "observation"
    ) -> list[BeliefUpdate]:
        """Ingest an observation and update beliefs based on prediction error.

        The core of the predictive coding loop:
        1. Ingest the observation text into RSVS
        2. For each active prediction, compare expected vs observed
        3. Compute prediction error: δ = observed - predicted
        4. Update belief: belief_new = belief_old + η × δ
        5. If direction is "revise" and error is large → flag for anomaly

        This IS Friston's free energy minimization. The prediction error
        (δ) drives learning — the larger the error, the more we update.

        Analogi: Jin Soun mengamati "tidak ada yang mengonsumsi pil".
        Prediksinya: "Ju Jangmok = pencuri → pasti mengonsumsi pil".
        Prediction error: BESAR. Keyakinan turun dari 0.8 ke 0.3.
        Dia mengubah catatan: "mungkin bukan pencuri."

        Args:
            text: The observation text to ingest and compare.
            source: Provenance of the observation (for tracking).

        Returns:
            A list of BeliefUpdate objects, one for each prediction
            that was compared against this observation.
        """
        updates: list[BeliefUpdate] = []

        # Step 1: Ingest the observation
        observed_atoms: list[str] = []
        if self.rsvs_available:
            try:
                self._bridge.ingest(text)
            except Exception as exc:
                logger.warning("RSVS ingest failed during observation: %s", exc)

            # Extract atoms from the observation via query/appraise
            try:
                # Try to get the most relevant atoms from the observation
                for concept_pred in self._active_predictions():
                    observed = self._observe_concept(concept_pred.concept)
                    self._observed[concept_pred.concept] = observed
                    observed_atoms.extend(observed)
            except Exception as exc:
                logger.warning("Failed to observe concepts: %s", exc)
        else:
            # Fallback — extract keywords from observation text
            observed_atoms = self._fallback_atomize(text)
            self._fallback_ingest(text, source)

        # Step 2: Compare with active predictions and update beliefs
        # Analogi: Jin Soun membandingkan bukti baru dengan setiap
        # prediksi yang masih aktif di buku catatannya.
        for prediction in self._active_predictions():
            update = self._compute_belief_update(prediction, text, observed_atoms)
            if update is not None:
                updates.append(update)
                self._belief_updates.append(update)

        # Step 3: Also check for anomalies after updating
        # (anomalies are detected separately, but we prime the pump here)
        if updates:
            new_anomalies = self.detect_anomalies()
            self._anomalies.extend(new_anomalies)

        return updates

    # ------------------------------------------------------------------
    # Anomaly Detection
    # ------------------------------------------------------------------

    def detect_anomalies(self) -> list[Anomaly]:
        """Detect anomalies by comparing predictions with observations.

        An anomaly occurs when |predicted - observed| > threshold.
        We use appraise() to check if predicted statements are consistent
        with the current graph state. If appraise returns "disagree",
        that's an anomaly.

        Analogi: Jin Soun memeriksa catatannya — "Aku prediksi X,
        tapi observasi menunjukkan Y. Selisihnya terlalu besar."
        Anomali ditandai benang merah, siap untuk pattern completion
        di layer berikutnya.

        Returns:
            A list of Anomaly objects for predictions that diverge
            from observations beyond the anomaly threshold.
        """
        anomalies: list[Anomaly] = []

        for prediction in self._active_predictions():
            concept = prediction.concept
            expected = prediction.expected_compositions

            # Get observed compositions for this concept
            observed = self._observed.get(concept, [])
            if not observed and self.rsvs_available:
                observed = self._observe_concept(concept)

            if not expected and not observed:
                continue  # Nothing to compare

            # Compute prediction error (L2-02: uses structural_similarity when available)
            delta = self._compute_prediction_error(expected, observed, concept)

            # Check with appraise() if RSVS available (L2-02: anomaly verification)
            appraise_disagree = False
            if self.rsvs_available and expected:
                try:
                    # Build a statement from the prediction
                    statement = f"{concept} has compositions: {', '.join(expected)}"
                    appraise_result = self._bridge.appraise(statement)
                    # Parse appraise result — "disagree" means anomaly
                    if self._is_appraise_negative(appraise_result):
                        appraise_disagree = True
                except Exception as exc:
                    logger.debug("appraise() failed for '%s': %s", concept, exc)

            # L2-02: Also check structural_similarity between expected and observed
            structural_anomaly = False
            if self.rsvs_available and len(expected) > 0 and len(observed) > 0:
                try:
                    # Check if the top expected and observed are structurally similar
                    top_expected = expected[0] if expected else ""
                    top_observed = observed[0] if observed else ""
                    if top_expected and top_observed and top_expected != top_observed:
                        sim = self._bridge.structural_similarity(top_expected, top_observed)
                        if sim is not None:
                            sim_val = sim.get("structural_similarity", 0.0)
                            if isinstance(sim_val, (int, float)) and float(sim_val) < 0.2:
                                structural_anomaly = True
                except Exception as exc:
                    logger.debug("structural_similarity anomaly check failed: %s", exc)

            # Determine if this is an anomaly
            is_anomaly = delta > self.anomaly_threshold or appraise_disagree or structural_anomaly

            if is_anomaly:
                # Boost delta if appraise or structural analysis also disagrees
                effective_delta = delta
                if (appraise_disagree or structural_anomaly) and delta < self.anomaly_threshold:
                    effective_delta = self.anomaly_threshold + 0.1

                anomaly = Anomaly(
                    concept=concept,
                    expected=expected,
                    observed=observed,
                    delta=effective_delta,
                    description=self._describe_anomaly(
                        concept, expected, observed, effective_delta
                    ),
                )
                anomalies.append(anomaly)
                logger.info(
                    "Anomaly detected: '%s' — expected=%s, observed=%s, δ=%.3f",
                    concept, expected, observed, effective_delta
                )

        return anomalies

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_predictions(self) -> list[Prediction]:
        """Return all active (non-expired) predictions.

        Active predictions are those within the staleness limit.
        Expired predictions are automatically pruned.

        Returns:
            A list of Prediction objects that are still active.
        """
        return self._active_predictions()

    def get_belief_history(self, concept: str | None = None) -> list[BeliefUpdate]:
        """Return belief update history, optionally filtered by concept.

        Analogi: Jin Soun membuka buku catatan dan melihat riwayat
        semua pembaruan keyakinan — kapan dia mulai curiga, kapan
        dia mengubah pikiran, dan apa buktinya.

        Args:
            concept: If provided, only return updates for this concept.
                If None, return all updates.

        Returns:
            A list of BeliefUpdate objects matching the filter.
        """
        if concept is None:
            return list(self._belief_updates)
        return [u for u in self._belief_updates if u.concept == concept]

    def get_anomalies(self) -> list[Anomaly]:
        """Return all detected anomalies.

        Returns:
            A list of Anomaly objects.
        """
        return list(self._anomalies)

    def get_current_beliefs(self) -> dict[str, float]:
        """Return current belief (confidence) levels for all tracked concepts.

        Computes the latest confidence for each concept by replaying
        the belief update history.

        Returns:
            A dict mapping concept name → current confidence (0.0 - 1.0).
        """
        beliefs: dict[str, float] = {}

        # Initialize from predictions
        for pred in self._predictions:
            if pred.concept not in beliefs:
                beliefs[pred.concept] = pred.confidence

        # Apply updates in order (most recent wins)
        for update in self._belief_updates:
            beliefs[update.concept] = update.new_confidence

        return beliefs

    # ------------------------------------------------------------------
    # Internal: Active predictions
    # ------------------------------------------------------------------

    def _active_predictions(self) -> list[Prediction]:
        """Return predictions that haven't expired.

        Predictions older than _PREDICTION_STALENESS_LIMIT are pruned.
        """
        now = time.time()
        active: list[Prediction] = []

        for pred in self._predictions:
            try:
                pred_time = time.mktime(time.strptime(pred.timestamp, "%Y-%m-%dT%H:%M:%S"))
                age = now - pred_time
                if age < _PREDICTION_STALENESS_LIMIT:
                    active.append(pred)
            except (ValueError, OverflowError):
                # If timestamp parsing fails, keep the prediction
                active.append(pred)

        # Prune expired predictions from the main list
        self._predictions = active
        return active

    # ------------------------------------------------------------------
    # Internal: Belief update computation
    # ------------------------------------------------------------------

    def _compute_belief_update(
        self,
        prediction: Prediction,
        observation_text: str,
        observed_atoms: list[str],
    ) -> BeliefUpdate | None:
        """Compute a belief update for a single prediction.

        Implements the Rescorla-Wagner / Friston update:
            belief_new = belief_old + η × (observed - predicted)

        The "observed" signal is derived from how well the observation
        matches the prediction's expected compositions.

        Analogi: Jin Soun membandingkan bukti baru dengan prediksi.
        Jika bukti cocok → konfirmasi (keyakinan naik).
        Jika bukti bertentangan → revisi (keyakinan turun).

        Args:
            prediction: The prediction to compare against.
            observation_text: The raw observation text.
            observed_atoms: Atoms extracted from the observation.

        Returns:
            A BeliefUpdate if the observation is relevant, or None.
        """
        concept = prediction.concept
        expected = set(prediction.expected_compositions)
        observed = set(observed_atoms)

        # Check if the observation is relevant to this prediction
        # If the concept isn't mentioned at all, skip
        concept_mentioned = (
            concept.lower() in observation_text.lower()
            or concept in observed
            or any(c in observed for c in prediction.context)
        )

        if not concept_mentioned and not expected:
            return None

        # Compute observed signal — how much of the expected was confirmed
        if expected:
            overlap = expected & observed
            observed_signal = len(overlap) / max(len(expected), 1)
        else:
            # No specific expectations — use presence/absence
            observed_signal = 1.0 if concept_mentioned else 0.0

        # Predicted signal = current confidence
        predicted_signal = prediction.confidence

        # Prediction error: δ = observed - predicted
        delta = observed_signal - predicted_signal

        # Rescorla-Wagner update: belief_new = belief_old + η × δ
        old_confidence = predicted_signal
        new_confidence = max(0.0, min(1.0, old_confidence + self.eta * delta))

        # Determine direction
        if new_confidence >= old_confidence:
            direction = "confirm"
        else:
            direction = "revise"

        # Generate reason
        if direction == "confirm":
            reason = (
                f"Observation confirms prediction for '{concept}': "
                f"{len(overlap) if expected else 'concept mentioned'} "
                f"of {len(expected)} expected compositions matched."
            )
        else:
            reason = (
                f"Observation contradicts prediction for '{concept}': "
                f"prediction error δ={delta:.3f} exceeds threshold. "
                f"Expected {list(expected)}, observed {list(observed)}."
            )

        # Collect evidence
        evidence = []
        if expected:
            matched = expected & observed
            missed = expected - observed
            unexpected = observed - expected
            if matched:
                evidence.append(f"Confirmed: {', '.join(sorted(matched))}")
            if missed:
                evidence.append(f"Missing: {', '.join(sorted(missed))}")
            if unexpected:
                evidence.append(f"Unexpected: {', '.join(sorted(unexpected)[:5])}")
        else:
            evidence.append(f"Concept '{concept}' mentioned in observation")

        update = BeliefUpdate(
            concept=concept,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            direction=direction,
            reason=reason,
            evidence=evidence,
        )

        # Update the prediction's confidence to reflect the new belief
        prediction.confidence = new_confidence

        return update

    # ------------------------------------------------------------------
    # Internal: Observation helpers
    # ------------------------------------------------------------------

    def _observe_concept(self, concept: str) -> list[str]:
        """Get current observed compositions for a concept via the bridge.

        Uses senses() and relate() to determine what the graph
        currently knows about this concept.

        Args:
            concept: The concept to observe.

        Returns:
            A list of observed composition labels.
        """
        if not self.rsvs_available:
            return self._observed.get(concept, [])

        observed: list[str] = []

        try:
            senses = self._bridge.senses(concept)
            if senses:
                observed.extend(self._parse_compositions(senses))
        except Exception:
            pass

        try:
            relate_result = self._bridge.relate(concept)
            if relate_result:
                extra = self._parse_compositions(relate_result)
                for comp in extra:
                    if comp not in observed:
                        observed.append(comp)
        except Exception:
            pass

        return observed

    # ------------------------------------------------------------------
    # Internal: Prediction error computation
    # ------------------------------------------------------------------

    def _compute_prediction_error(
        self,
        expected: list[str],
        observed: list[str],
        concept: str = "",
    ) -> float:
        """Compute the magnitude of prediction error.

        When RSVS is available, uses structural_similarity() for
        a more accurate error measurement that considers the graph
        structure, not just label overlap.

        Falls back to Jaccard distance when RSVS is unavailable.

        Args:
            expected: Expected compositions.
            observed: Observed compositions.
            concept: The concept being evaluated (used for structural_similarity).

        Returns:
            A float between 0.0 (perfect match) and 1.0 (complete mismatch).
        """
        if not expected and not observed:
            return 0.0
        if not expected or not observed:
            return 1.0

        # L2-02: Try structural_similarity() first for more accurate error
        if self.rsvs_available and concept:
            try:
                # Compare each expected composition with observed ones
                # using structural similarity
                total_sim = 0.0
                comparisons = 0
                for exp in expected[:5]:
                    for obs in observed[:5]:
                        sim_result = self._bridge.structural_similarity(exp, obs)
                        if sim_result is not None:
                            sim_val = sim_result.get("structural_similarity", 0.0)
                            if isinstance(sim_val, (int, float)):
                                total_sim += float(sim_val)
                                comparisons += 1

                if comparisons > 0:
                    # Average structural similarity → convert to error
                    avg_sim = total_sim / comparisons
                    return 1.0 - avg_sim
            except Exception as exc:
                logger.debug("structural_similarity() for error computation failed: %s", exc)

        # Fallback: Jaccard distance
        set_a = set(expected)
        set_b = set(observed)
        intersection = set_a & set_b
        union = set_a | set_b

        if not union:
            return 0.0

        # Jaccard distance = 1 - Jaccard similarity
        return 1.0 - (len(intersection) / len(union))

    # ------------------------------------------------------------------
    # Internal: Anomaly description
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_anomaly(
        concept: str,
        expected: list[str],
        observed: list[str],
        delta: float,
    ) -> str:
        """Generate a human-readable description of an anomaly.

        Analogi: Jin Soun menulis di buku catatannya:
        "Anomali terdeteksi untuk Ju Jangmok — aku prediksi dia
        mengonsumsi pil, tapi observasi menunjukkan tidak ada konsumsi.
        Selisih: 0.7. Perlu investigasi lebih lanjut."

        Args:
            concept: The anomalous concept.
            expected: What was expected.
            observed: What was actually observed.
            delta: The prediction error magnitude.

        Returns:
            A human-readable string describing the anomaly.
        """
        expected_set = set(expected)
        observed_set = set(observed)
        missing = expected_set - observed_set
        unexpected = observed_set - expected_set

        parts = [f"Anomaly for '{concept}' (δ={delta:.3f}):"]

        if missing:
            missing_str = ", ".join(sorted(missing)[:5])
            parts.append(f"  Expected but not observed: {missing_str}")
        if unexpected:
            unexpected_str = ", ".join(sorted(unexpected)[:5])
            parts.append(f"  Observed but not expected: {unexpected_str}")
        if not missing and not unexpected:
            parts.append("  Complete mismatch between prediction and observation.")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal: Appraise result parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _is_appraise_negative(result: Any) -> bool:
        """Check if an appraise() result indicates disagreement.

        Handles various result formats from the bridge:
        - String: "disagree", "contradiction", "false"
        - Dict with "verdict" key (typical: "agree", "neutral", "disagree")
        - Dict with "disagree_pct" and "agree_pct" keys — uses percentage
          comparison as a more robust negative detection

        Args:
            result: The result from RsvsBridge.appraise().

        Returns:
            True if the result indicates disagreement.
        """
        negative_indicators = {"disagree", "contradiction", "false", "negative", "reject"}

        if isinstance(result, str):
            return result.lower().strip() in negative_indicators

        if isinstance(result, dict):
            # Robust check: compare disagree_pct vs agree_pct
            disagree_pct = result.get("disagree_pct", 0)
            agree_pct = result.get("agree_pct", 0)
            if isinstance(disagree_pct, (int, float)) and isinstance(agree_pct, (int, float)):
                if float(disagree_pct) > float(agree_pct):
                    return True

            # Verdict-based check
            verdict = result.get("verdict", result.get("result", ""))
            if isinstance(verdict, str):
                return verdict.lower().strip() in negative_indicators
            if isinstance(verdict, bool):
                return not verdict
            if isinstance(verdict, (int, float)):
                return float(verdict) < 0.0

        return False

    # ------------------------------------------------------------------
    # Internal: Composition parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_compositions(result: Any) -> list[str]:
        """Parse compositions from RsvsBridge result formats.

        The bridge returns plain dicts/lists (never PyO3 objects):

        - senses() result: list[dict] where each dict has
          "compositions": [(label, sense_id), ...] — extract labels
        - relate() result: {"related_nodes": [(id_or_label, score), ...], ...}
          — extract labels (may be numeric IDs in Rust core mode)
        - context_query() result: {"scored_atoms": [(label, score), ...],
          "compositions": [(label, sense_id), ...]}

        Handles both Rust core mode (numeric IDs) and fallback mode
        (string labels).

        Args:
            result: A result from bridge.senses(), bridge.relate(),
                bridge.context_query(), etc.

        Returns:
            A list of composition label strings.
        """
        compositions: list[str] = []

        if result is None:
            return compositions

        # --- list[dict] format (from senses()) ---
        if isinstance(result, list):
            for item in result:
                if isinstance(item, str):
                    compositions.append(item)
                elif isinstance(item, dict):
                    # Each sense dict has "compositions": [(label, sense_id), ...]
                    comp_list = item.get("compositions", [])
                    if comp_list:
                        for entry in comp_list:
                            label = PredictiveEngine._extract_label_from_tuple(entry)
                            if label:
                                compositions.append(str(label))
                    else:
                        # Fallback: try "label" or "composition" keys
                        label = item.get("label", item.get("composition", ""))
                        if label:
                            compositions.append(str(label))
            return compositions

        # --- dict format (from relate() or context_query()) ---
        if isinstance(result, dict):
            # context_query() result: "scored_atoms" and "compositions"
            for key in ("compositions", "scored_atoms"):
                items = result.get(key, [])
                if items:
                    for entry in items:
                        label = PredictiveEngine._extract_label_from_tuple(entry)
                        if label:
                            compositions.append(str(label))

            # relate() result: "related_nodes" contains [(id_or_label, score), ...]
            related_nodes = result.get("related_nodes", [])
            if related_nodes:
                for entry in related_nodes:
                    label = PredictiveEngine._extract_label_from_tuple(entry)
                    if label:
                        compositions.append(str(label))

            # Also try "structural_relations" from relate()
            structural = result.get("structural_relations", [])
            if structural:
                for entry in structural:
                    label = PredictiveEngine._extract_label_from_tuple(entry)
                    if label and str(label) not in compositions:
                        compositions.append(str(label))

            # Legacy key names
            if not compositions:
                for key in ("senses", "related", "atoms", "nodes"):
                    items = result.get(key, [])
                    if items:
                        return PredictiveEngine._parse_compositions(items)

            # If the dict itself looks like a single composition
            if not compositions:
                label = result.get("label", result.get("composition", ""))
                if label:
                    compositions.append(str(label))

            return compositions

        if isinstance(result, str):
            # Try JSON parse
            try:
                import json
                parsed = json.loads(result)
                return PredictiveEngine._parse_compositions(parsed)
            except (json.JSONDecodeError, ValueError):
                # Treat as a single composition label
                return [result] if result.strip() else []

        # Try iterating as a last resort
        try:
            for item in result:
                if isinstance(item, str):
                    compositions.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 1:
                    label = item[0]
                    if label:
                        compositions.append(str(label))
        except (TypeError, AttributeError):
            pass

        return compositions

    @staticmethod
    def _extract_label_from_tuple(entry: Any) -> str | None:
        """Extract a label from a tuple like (label, score) or (label, sense_id).

        Handles:
        - (str_label, score) → str_label
        - (int_id, score) → str(int_id)  (Rust core mode: numeric IDs)
        - plain str → str
        - plain int → str(int)
        - None → None

        Args:
            entry: A tuple, str, int, or other value.

        Returns:
            A string label, or None if nothing could be extracted.
        """
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry if entry.strip() else None
        if isinstance(entry, (int, float)):
            return str(entry)
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            first = entry[0]
            if isinstance(first, str):
                return first if first.strip() else None
            if isinstance(first, (int, float)):
                return str(first)
            return str(first) if first is not None else None
        return None

    # ------------------------------------------------------------------
    # Internal: Fallback helpers (when RSVS is unavailable)
    # ------------------------------------------------------------------

    def _fallback_predict(
        self, concept: str, context: list[str]
    ) -> tuple[list[str], float]:
        """Predict compositions without RSVS using the bridge's fallback graph.

        P2-7: Delegates to self._bridge which always has a _FallbackGraph
        when the Rust core is unavailable, ensuring consistent state.

        Args:
            concept: The concept to predict.
            context: Context atoms.

        Returns:
            A tuple of (expected_compositions, confidence).
        """
        # Delegate to bridge.query() for composition lookup
        try:
            query_result = self._bridge.query(concept)
            if query_result is not None:
                compositions = []
                # Extract composition labels from query result
                comp_list = query_result.get("compositions", [])
                for entry in comp_list:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                        compositions.append(str(entry[0]))
                    elif isinstance(entry, str):
                        compositions.append(entry)
                # Get confidence from the query result
                grounding = query_result.get("grounding_score", 0.5)
                confidence = max(0.5, grounding)

                # Boost confidence if context matches
                if context:
                    context_match = sum(
                        1 for c in context
                        if c.lower() in concept.lower() or concept.lower() in c.lower()
                    )
                    if context_match > 0:
                        confidence = min(1.0, confidence + 0.1 * context_match)

                return compositions, confidence
        except Exception as exc:
            logger.debug("bridge.query() in _fallback_predict failed: %s", exc)

        # Also try relate() for broader connections
        try:
            relate_result = self._bridge.relate(concept)
            if relate_result is not None:
                related = relate_result.get("related_nodes", [])
                compositions = []
                for entry in related:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                        compositions.append(str(entry[0]))
                    elif isinstance(entry, str):
                        compositions.append(entry)
                if compositions:
                    return compositions, 0.4
        except Exception as exc:
            logger.debug("bridge.relate() in _fallback_predict failed: %s", exc)

        # No prior knowledge — return empty prediction with low confidence
        return [], 0.3

    def _fallback_ingest(self, text: str, source: str = "observation") -> None:
        """Ingest text into the bridge's fallback graph when RSVS is unavailable.

        P2-7: Delegates to self._bridge.ingest() which always has a
        _FallbackGraph when the Rust core is unavailable, ensuring
        consistent state across all modules.

        Args:
            text: The text to ingest.
            source: Source provenance identifier.
        """
        try:
            self._bridge.ingest(text, source_provenance=source)
        except Exception as exc:
            logger.warning("bridge.ingest() in _fallback_ingest failed: %s", exc)

    @staticmethod
    def _fallback_atomize(text: str) -> list[str]:
        """Extract keywords from text for fallback mode.

        P2-7: Uses the bridge's keyword extraction when available,
        otherwise falls back to local stop-word filtering.

        Args:
            text: Input text to atomize.

        Returns:
            A list of word-level "atoms".
        """
        # Try to use bridge's keyword extraction
        try:
            from .bridge import _FallbackGraph
            return _FallbackGraph._extract_keywords(text)
        except Exception:
            pass

        # Local fallback
        words = text.lower().split()
        cleaned = []
        for w in words:
            w = w.strip(".,;:!?\"'()[]{}")
            if len(w) > 3 and w not in _STOP_WORDS:
                cleaned.append(w)
        return cleaned[:20]

    # ------------------------------------------------------------------
    # Persistence (P2-8: Cognitive persistence)
    # ------------------------------------------------------------------

    _PERSIST_SCHEMA_VERSION = "1.0"

    def save_to_dict(self) -> dict:
        """Serialize cognitive state to a plain dict (in-memory).

        Saves `_predictions`, `_belief_updates`, `_anomalies`, `_observed`,
        `eta`, `anomaly_threshold`, and the fallback graph.  The RSVS
        bridge / graph itself is NOT serialized — only the Layer-2
        cognitive state.

        Returns:
            A dict containing the full serializable state.
        """
        return {
            "schema_version": self._PERSIST_SCHEMA_VERSION,
            "eta": self.eta,
            "anomaly_threshold": self.anomaly_threshold,
            "predictions": [p.to_dict() for p in self._predictions],
            "belief_updates": [b.to_dict() for b in self._belief_updates],
            "anomalies": [a.to_dict() for a in self._anomalies],
            "observed": self._observed,
            "fallback_graph_size": 0,  # P2-7: state lives in bridge
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore cognitive state from a plain dict (in-memory).

        Restores `_predictions`, `_belief_updates`, `_anomalies`,
        `_observed`, `eta`, `anomaly_threshold`, and the fallback graph.
        Existing state is replaced.

        Args:
            data: A dict previously returned by `save_to_dict()`.
        """
        if not isinstance(data, dict):
            logger.warning("load_from_dict: expected dict, got %s", type(data).__name__)
            return

        # Schema compatibility check
        saved_version = data.get("schema_version", "0.0")
        if saved_version != self._PERSIST_SCHEMA_VERSION:
            logger.warning(
                "load_from_dict: schema version mismatch (saved=%s, current=%s). "
                "Proceeding with best-effort restore.",
                saved_version, self._PERSIST_SCHEMA_VERSION,
            )

        self.eta = data.get("eta", _DEFAULT_ETA)
        self.anomaly_threshold = data.get("anomaly_threshold", _DEFAULT_ANOMALY_THRESHOLD)

        # Reconstruct predictions
        self._predictions = []
        for p_dict in data.get("predictions", []):
            if isinstance(p_dict, dict):
                self._predictions.append(Prediction(
                    concept=p_dict.get("concept", ""),
                    expected_compositions=p_dict.get("expected_compositions", []),
                    confidence=p_dict.get("confidence", 0.5),
                    context=p_dict.get("context", []),
                    timestamp=p_dict.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
                ))

        # Reconstruct belief updates
        self._belief_updates = []
        for b_dict in data.get("belief_updates", []):
            if isinstance(b_dict, dict):
                self._belief_updates.append(BeliefUpdate(
                    concept=b_dict.get("concept", ""),
                    old_confidence=b_dict.get("old_confidence", 0.5),
                    new_confidence=b_dict.get("new_confidence", 0.5),
                    direction=b_dict.get("direction", "confirm"),
                    reason=b_dict.get("reason", ""),
                    evidence=b_dict.get("evidence", []),
                    timestamp=b_dict.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
                ))

        # Reconstruct anomalies
        self._anomalies = []
        for a_dict in data.get("anomalies", []):
            if isinstance(a_dict, dict):
                self._anomalies.append(Anomaly(
                    concept=a_dict.get("concept", ""),
                    expected=a_dict.get("expected", []),
                    observed=a_dict.get("observed", []),
                    delta=a_dict.get("delta", 0.0),
                    description=a_dict.get("description", ""),
                ))

        self._observed = data.get("observed", {})
        # P2-7: _fallback_graph removed — state now lives in the bridge.
        # Backward compat: if old data contains fallback_graph entries,
        # ingest them into the bridge so they aren't lost.
        old_graph = data.get("fallback_graph", {})
        if old_graph and isinstance(old_graph, dict):
            for concept, entry in old_graph.items():
                try:
                    # Re-ingest each concept to populate the bridge's fallback graph
                    comps = entry.get("compositions", [])
                    if comps:
                        self._bridge.ingest(f"{concept} {' '.join(comps)}")
                except Exception as exc:
                    logger.debug("Failed to migrate old fallback_graph entry '%s': %s", concept, exc)

        logger.info(
            "PredictiveEngine state restored: %d predictions, %d belief updates, "
            "%d anomalies, %d observed concepts",
            len(self._predictions), len(self._belief_updates),
            len(self._anomalies), len(self._observed),
        )

    def save(self, path: str) -> dict:
        """Save cognitive state to a JSON file.

        Args:
            path: Filesystem path to write the JSON file.

        Returns:
            A summary dict with stats about what was saved.
        """
        data = self.save_to_dict()
        summary: dict = {
            "path": path,
            "predictions": len(self._predictions),
            "belief_updates": len(self._belief_updates),
            "anomalies": len(self._anomalies),
            "observed_concepts": len(self._observed),
            "eta": self.eta,
            "anomaly_threshold": self.anomaly_threshold,
            "schema_version": self._PERSIST_SCHEMA_VERSION,
            "success": False,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            summary["success"] = True
            logger.info("PredictiveEngine state saved to %s", path)
        except (OSError, TypeError) as exc:
            summary["error"] = str(exc)
            logger.error("PredictiveEngine save failed: %s", exc)
        return summary

    def load(self, path: str) -> dict:
        """Load cognitive state from a JSON file.

        Args:
            path: Filesystem path to read the JSON file from.

        Returns:
            A summary dict with stats about what was loaded.
        """
        summary: dict = {
            "path": path,
            "predictions": 0,
            "belief_updates": 0,
            "anomalies": 0,
            "success": False,
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_from_dict(data)
            summary["predictions"] = len(self._predictions)
            summary["belief_updates"] = len(self._belief_updates)
            summary["anomalies"] = len(self._anomalies)
            summary["observed_concepts"] = len(self._observed)
            summary["schema_version"] = data.get("schema_version", "unknown")
            summary["success"] = True
            logger.info("PredictiveEngine state loaded from %s", path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            summary["error"] = str(exc)
            logger.error("PredictiveEngine load failed: %s", exc)
        return summary

    # ------------------------------------------------------------------
    # Reset / utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal state — predictions, beliefs, anomalies.

        Does NOT reset the RSVS graph itself.

        Analogi: Jin Soun mengosongkan buku catatan prediksi
        untuk memulai kasus baru. Simhyeon Pavilion tetap utuh.
        """
        self._predictions = []
        self._belief_updates = []
        self._anomalies = []
        self._observed = {}
        # P2-7: _fallback_graph removed — state now lives in the bridge
        logger.info("PredictiveEngine reset — all internal state cleared")

    def status(self) -> dict:
        """Return a status summary of the engine.

        Returns:
            A dict with counts of predictions, updates, anomalies,
            and current belief levels.
        """
        return {
            "rsvs_available": self.rsvs_available,
            "is_rust_core": self.is_rust_core,
            "eta": self.eta,
            "anomaly_threshold": self.anomaly_threshold,
            "active_predictions": len(self._active_predictions()),
            "total_belief_updates": len(self._belief_updates),
            "total_anomalies": len(self._anomalies),
            "current_beliefs": self.get_current_beliefs(),
            "fallback_graph_size": 0,  # P2-7: state lives in bridge
        }
