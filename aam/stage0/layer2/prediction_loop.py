"""
Explicit Prediction Loop — Full Predict/Observe/Update Lifecycle

The existing PredictiveEngine has predict() and observe_and_update(), but
they are NOT integrated as a proper cycle. This module implements the
COMPLETE lifecycle: predict → observe → update → (re-predict | escalate | resolve).

Lifecycle states:
    pending → observed → resolved | retired | escalated

    pending  : Prediction made, waiting for observation
    observed : At least one observation compared, belief updated
    resolved : Confidence stabilized (3+ consecutive confirms with delta < 0.05)
    retired  : Prediction expired (staleness limit) or superseded
    escalated: Large prediction error → flagged for deeper investigation

Analogi: Jin Soun tidak hanya mencatat prediksi — dia menjalankan
siklus lengkap. Dia memprediksi, mengamati, memperbarui, dan jika
prediksinya salah besar, dia menaikkan kasus ke "investigasi lebih
dalam". Seperti detektif yang tidak hanya mengumpulkan bukti, tapi
mengevaluasi, mengulang, dan menutup kasus atau meng eskalasi.

Architecture:
    PredictionLoop WRAPS PredictiveEngine — it does NOT replace it.
    The engine does the raw computation; the loop orchestrates the lifecycle.

Flow (run_cycle):
    1. Create prediction (or reuse existing pending one for the concept)
    2. Ingest observation into RSVS
    3. Compare expected vs observed
    4. Update belief via Rescorla-Wagner
    5. Detect anomalies
    6. Trigger RSVS feedback if needed (reflection / consolidation)
    7. Auto re-predict if confidence dropped significantly
    8. Update cycle tracker
    9. Return CycleResult with full traceability
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import RsvsBridge, get_bridge
from .predictive import Anomaly, BeliefUpdate, Prediction, PredictiveEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

# How much confidence must drop before we auto re-predict
_DEFAULT_RE_PREDICTION_THRESHOLD = 0.15

# Number of consecutive confirms with delta < convergence_delta to resolve
_DEFAULT_CONVERGENCE_COUNT = 3

# Delta threshold for considering a belief update as "converging"
_DEFAULT_CONVERGENCE_DELTA = 0.05

# Maximum age (seconds) for a cycle before it is retired
_DEFAULT_STALENESS_LIMIT = 600.0  # 10 minutes

# Default interval (seconds) for continuous loop checks
_DEFAULT_LOOP_INTERVAL = 30.0

# Confidence threshold below which we trigger RSVS reflection
_REFLECTION_CONFIDENCE_THRESHOLD = 0.3

# Number of anomalies in the same cycle that triggers consolidation
_CONSOLIDATION_ANOMALY_COUNT = 3

# Valid lifecycle states
_VALID_STATES = frozenset({"pending", "observed", "resolved", "retired", "escalated"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CycleTracker:
    """Tracks the complete predict→observe→update cycle.

    Records every observation event, cumulative prediction error, and
    lifecycle state transitions. Supports parent-child linking for
    re-predictions (a child cycle is created when auto re-prediction
    fires after a confidence drop).

    Analogi: Buku kasus Jin Soun — setiap kasus memiliki ID, tanggal
    pembuatan, daftar observasi, status (aktif/selesai/dieskalasi),
    dan tautan ke kasus induk jika ini adalah penyelidikan ulang.

    Attributes:
        cycle_id: Unique identifier for this cycle.
        concept: The concept being tracked.
        created_at: ISO timestamp when the cycle was created.
        observations: List of observation events [{text, timestamp, delta}].
        state: Current lifecycle state.
        parent_cycle_id: ID of the parent cycle (for re-predictions).
        child_cycle_ids: IDs of child cycles (re-predictions).
        resolution_reason: Why the cycle was resolved/retired/escalated.
    """

    cycle_id: str
    concept: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    observations: list[dict] = field(default_factory=list)
    state: str = "pending"
    parent_cycle_id: str | None = None
    child_cycle_ids: list[str] = field(default_factory=list)
    resolution_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate initial state."""
        if self.state not in _VALID_STATES:
            raise ValueError(f"Invalid cycle state: {self.state!r}")

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "cycle_id": self.cycle_id,
            "concept": self.concept,
            "created_at": self.created_at,
            "observations": list(self.observations),
            "state": self.state,
            "parent_cycle_id": self.parent_cycle_id,
            "child_cycle_ids": list(self.child_cycle_ids),
            "resolution_reason": self.resolution_reason,
        }


@dataclass
class CycleResult:
    """Result from one complete predict→observe→update cycle.

    Contains the full traceability chain: what was predicted, how belief
    changed, whether anomalies were detected, what RSVS feedback was
    triggered, and whether a re-prediction was spawned.

    Analogi: Laporan kasus lengkap Jin Soun — prediksi awal,
    pembaruan keyakinan, anomali yang ditemukan, tindakan yang diambil
    (refleksi/konsolidasi), prediksi ulang (jika ada), dan metrik siklus.

    Attributes:
        cycle_id: The cycle this result belongs to.
        concept: The concept that was processed.
        prediction: The prediction (original or re-predicted).
        belief_update: The belief update (if any observation was compared).
        anomaly: The anomaly detected (if any).
        state: Current lifecycle state of the cycle.
        re_prediction: Auto re-predicted prediction (if confidence dropped).
        rsvs_feedback: RSVS feedback results (reflection/consolidation).
        cycle_metrics: Computed metrics for this cycle.
    """

    cycle_id: str
    concept: str
    prediction: Prediction
    belief_update: BeliefUpdate | None = None
    anomaly: Anomaly | None = None
    state: str = "pending"
    re_prediction: Prediction | None = None
    rsvs_feedback: dict | None = None
    cycle_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "cycle_id": self.cycle_id,
            "concept": self.concept,
            "prediction": self.prediction.to_dict(),
            "belief_update": self.belief_update.to_dict() if self.belief_update else None,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None,
            "state": self.state,
            "re_prediction": self.re_prediction.to_dict() if self.re_prediction else None,
            "rsvs_feedback": self.rsvs_feedback,
            "cycle_metrics": self.cycle_metrics,
        }


# ---------------------------------------------------------------------------
# PredictionLoop
# ---------------------------------------------------------------------------

class PredictionLoop:
    """Explicit Prediction Loop — orchestrates the full predict/observe/update lifecycle.

    This class WRAPS PredictiveEngine (does NOT replace it). The engine
    handles the raw prediction and belief-update computation; the loop
    adds lifecycle management, state tracking, auto re-prediction, RSVS
    feedback integration, and continuous background processing.

    Lifecycle states:
        pending → observed → resolved | retired | escalated

        - pending  : Prediction made, waiting for observation
        - observed : At least one observation compared, belief updated
        - resolved : Confidence stabilized (3+ consecutive confirms, delta < 0.05)
        - retired  : Prediction expired or superseded
        - escalated: Large prediction error → flagged for deeper investigation

    Analogi: Jin Soun menjalankan siklus detektif yang lengkap —
    bukan hanya mencatat dan membandingkan, tapi mengelola setiap kasus
    dari awal hingga penutupan. Jika kasus terlalu rumit, dia naikkan
    ke level investigasi yang lebih dalam (eskalasi). Jika keyakinan
    turun drastis, dia buat prediksi ulang dengan konteks baru.

    Attributes:
        engine: The underlying PredictiveEngine for raw computation.
        bridge: The RsvsBridge for direct RSVS operations.
        rsvs_available: Whether a working RSVS instance is connected.
    """

    def __init__(
        self,
        engine: PredictiveEngine | None = None,
        bridge: RsvsBridge | None = None,
        *,
        re_prediction_threshold: float = _DEFAULT_RE_PREDICTION_THRESHOLD,
        convergence_count: int = _DEFAULT_CONVERGENCE_COUNT,
        convergence_delta: float = _DEFAULT_CONVERGENCE_DELTA,
        staleness_limit: float = _DEFAULT_STALENESS_LIMIT,
        loop_interval: float = _DEFAULT_LOOP_INTERVAL,
    ) -> None:
        """Initialize the Prediction Loop.

        Args:
            engine: Optional pre-built PredictiveEngine. If None, one is
                created using the bridge or a default bridge.
            bridge: Optional pre-built RsvsBridge. Used for direct RSVS
                operations (reflection, consolidation). If None, obtained
                from the engine or created via get_bridge().
            re_prediction_threshold: How much confidence must drop before
                auto re-prediction fires (default 0.15).
            convergence_count: Number of consecutive confirms needed to
                resolve a cycle (default 3).
            convergence_delta: Delta threshold for considering an update
                as "converging" (default 0.05).
            staleness_limit: Maximum age (seconds) before a cycle is
                retired (default 600.0 = 10 minutes).
            loop_interval: Interval (seconds) for the continuous loop
                background thread (default 30.0).
        """
        # Set up the engine — the workhorse for raw prediction/belief computation
        if engine is not None:
            self._engine = engine
        elif bridge is not None:
            self._engine = PredictiveEngine(bridge=bridge)
        else:
            self._engine = PredictiveEngine()

        # Bridge for direct RSVS operations (reflection, consolidation, ingest)
        if bridge is not None:
            self._bridge = bridge
        else:
            self._bridge = self._engine._bridge

        self.rsvs_available = self._bridge.is_available

        # Configuration
        self._re_prediction_threshold = re_prediction_threshold
        self._convergence_count = convergence_count
        self._convergence_delta = convergence_delta
        self._staleness_limit = staleness_limit
        self._loop_interval = loop_interval

        # Active cycle trackers — maps cycle_id → CycleTracker
        self._cycles: dict[str, CycleTracker] = {}

        # Maps concept → active cycle_id (for reusing pending cycles)
        self._concept_cycle_map: dict[str, str] = {}

        # Cycle history — completed cycles for retrospective analysis
        self._cycle_history: list[CycleTracker] = []

        # Consecutive confirm counters — maps cycle_id → count
        self._consecutive_confirms: dict[str, int] = {}

        # Cumulative prediction error per cycle — maps cycle_id → total delta
        self._cumulative_error: dict[str, float] = {}

        # Anomaly count per cycle — maps cycle_id → count
        self._anomaly_counts: dict[str, int] = {}

        # Continuous loop state
        self._continuous_running: bool = False
        self._continuous_thread: threading.Thread | None = None
        self._continuous_stop_event = threading.Event()

        logger.info(
            "PredictionLoop initialized "
            "(re_pred_thresh=%.3f, convergence=%d@%.3f, "
            "staleness=%.0fs, interval=%.0fs, rsvs=%s)",
            re_prediction_threshold, convergence_count, convergence_delta,
            staleness_limit, loop_interval, self.rsvs_available,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> PredictiveEngine:
        """Access the underlying PredictiveEngine."""
        return self._engine

    @property
    def bridge(self) -> RsvsBridge:
        """Access the RsvsBridge."""
        return self._bridge

    @property
    def active_cycles(self) -> dict[str, CycleTracker]:
        """Return currently active (non-terminal) cycle trackers."""
        return {
            cid: ct for cid, ct in self._cycles.items()
            if ct.state in ("pending", "observed")
        }

    @property
    def cycle_history(self) -> list[CycleTracker]:
        """Return completed cycle history."""
        return list(self._cycle_history)

    # ------------------------------------------------------------------
    # Main lifecycle method: run_cycle()
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        concept: str,
        observation: str,
        context: list[str] | None = None,
    ) -> CycleResult:
        """Run one complete predict→observe→update cycle.

        This is the main entry point. It:
        1. Creates a prediction (or uses existing pending one for the concept)
        2. Ingests the observation into RSVS
        3. Compares expected vs observed
        4. Updates belief via Rescorla-Wagner
        5. Detects anomalies
        6. Triggers RSVS feedback if needed
        7. Auto re-predicts if confidence dropped significantly
        8. Updates cycle tracker
        9. Returns CycleResult with full traceability

        Analogi: Jin Soun menjalankan satu putaran lengkap penyelidikan —
        dia buat prediksi, amati bukti, perbarui keyakinan, periksa
        anomali, dan putuskan apakah perlu prediksi ulang atau esklasi.

        Args:
            concept: The concept to predict and observe.
            observation: The observation text to compare against the prediction.
            context: Optional context atoms for the prediction.

        Returns:
            A CycleResult with the full traceability chain.
        """
        context = context or []

        # Step 1: Get or create cycle tracker + prediction
        cycle_id, tracker, prediction = self._ensure_cycle(concept, context)

        # Step 2: Ingest observation into RSVS
        self._ingest_observation(observation)

        # Step 3-4: Observe and update belief (delegated to PredictiveEngine)
        belief_update = self._compute_belief_update(prediction, observation)

        # Step 5: Detect anomalies
        anomaly = self._detect_cycle_anomaly(prediction)

        # Step 6: Trigger RSVS feedback if needed
        rsvs_feedback = self._maybe_trigger_rsvs_feedback(
            cycle_id, tracker, belief_update, anomaly
        )

        # Step 7: Determine new state and check for auto re-prediction
        re_prediction = None
        new_state = self._transition_state(
            cycle_id, tracker, belief_update, anomaly
        )

        # Auto re-predict if confidence dropped significantly
        if belief_update is not None and belief_update.direction == "revise":
            confidence_drop = belief_update.old_confidence - belief_update.new_confidence
            if confidence_drop > self._re_prediction_threshold:
                re_prediction = self._auto_re_predict(
                    cycle_id, concept, context, tracker
                )
                logger.info(
                    "Auto re-prediction for '%s' (confidence dropped %.3f → %.3f, "
                    "drop=%.3f > threshold=%.3f)",
                    concept, belief_update.old_confidence, belief_update.new_confidence,
                    confidence_drop, self._re_prediction_threshold,
                )

        # Step 8: Record observation in tracker
        obs_delta = 0.0
        if belief_update is not None:
            obs_delta = abs(belief_update.new_confidence - belief_update.old_confidence)
        tracker.observations.append({
            "text": observation[:200],  # Truncate for storage
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "delta": obs_delta,
        })

        # Update cumulative error
        self._cumulative_error[cycle_id] = (
            self._cumulative_error.get(cycle_id, 0.0) + obs_delta
        )

        # Update anomaly count
        if anomaly is not None:
            self._anomaly_counts[cycle_id] = self._anomaly_counts.get(cycle_id, 0) + 1

        # Step 9: Compute cycle metrics
        cycle_metrics = self._compute_cycle_metrics(cycle_id, tracker)

        # Build and return the result
        result = CycleResult(
            cycle_id=cycle_id,
            concept=concept,
            prediction=prediction,
            belief_update=belief_update,
            anomaly=anomaly,
            state=tracker.state,
            re_prediction=re_prediction,
            rsvs_feedback=rsvs_feedback,
            cycle_metrics=cycle_metrics,
        )

        logger.debug(
            "Cycle %s: concept='%s', state=%s, belief=%s, anomaly=%s, "
            "re_pred=%s, rsvs_fb=%s",
            cycle_id, concept, tracker.state,
            belief_update.direction if belief_update else "none",
            "yes" if anomaly else "no",
            "yes" if re_prediction else "no",
            "yes" if rsvs_feedback else "no",
        )

        return result

    # ------------------------------------------------------------------
    # Cycle management
    # ------------------------------------------------------------------

    def get_cycle(self, cycle_id: str) -> CycleTracker | None:
        """Get a cycle tracker by ID.

        Searches both active cycles and completed history.

        Args:
            cycle_id: The cycle ID to look up.

        Returns:
            The CycleTracker, or None if not found.
        """
        # Check active cycles first
        tracker = self._cycles.get(cycle_id)
        if tracker is not None:
            return tracker

        # Check completed history
        for hist_tracker in self._cycle_history:
            if hist_tracker.cycle_id == cycle_id:
                return hist_tracker

        return None

    def get_cycle_for_concept(self, concept: str) -> CycleTracker | None:
        """Get the active cycle tracker for a concept.

        Args:
            concept: The concept to look up.

        Returns:
            The active CycleTracker for the concept, or None.
        """
        cycle_id = self._concept_cycle_map.get(concept)
        if cycle_id is None:
            return None
        tracker = self._cycles.get(cycle_id)
        if tracker is not None and tracker.state in ("pending", "observed"):
            return tracker
        return None

    def retire_cycle(self, cycle_id: str, reason: str = "manual") -> bool:
        """Manually retire a cycle.

        Args:
            cycle_id: The cycle to retire.
            reason: Why the cycle is being retired.

        Returns:
            True if the cycle was retired, False if not found or already terminal.
        """
        tracker = self._cycles.get(cycle_id)
        if tracker is None:
            logger.warning("Cannot retire cycle %s: not found", cycle_id)
            return False

        if tracker.state in ("resolved", "retired", "escalated"):
            logger.debug("Cycle %s already in terminal state: %s", cycle_id, tracker.state)
            return False

        tracker.state = "retired"
        tracker.resolution_reason = reason
        self._finalize_cycle(cycle_id, tracker)

        logger.info("Cycle %s retired: %s", cycle_id, reason)
        return True

    def retire_stale_cycles(self) -> list[str]:
        """Retire all cycles that have exceeded the staleness limit.

        Returns:
            List of retired cycle IDs.
        """
        retired: list[str] = []
        now = time.time()

        for cycle_id, tracker in list(self._cycles.items()):
            if tracker.state in ("resolved", "retired", "escalated"):
                continue

            try:
                created = time.mktime(
                    time.strptime(tracker.created_at, "%Y-%m-%dT%H:%M:%S")
                )
                age = now - created
                if age > self._staleness_limit:
                    tracker.state = "retired"
                    tracker.resolution_reason = (
                        f"Staleness limit exceeded (age={age:.0f}s > "
                        f"limit={self._staleness_limit:.0f}s)"
                    )
                    self._finalize_cycle(cycle_id, tracker)
                    retired.append(cycle_id)
                    logger.info(
                        "Cycle %s retired (stale): age=%.0fs", cycle_id, age
                    )
            except (ValueError, OverflowError):
                # If timestamp parsing fails, skip this cycle
                pass

        return retired

    # ------------------------------------------------------------------
    # Continuous loop mode
    # ------------------------------------------------------------------

    def start_continuous(self, interval: float | None = None) -> None:
        """Start the continuous prediction loop in a background thread.

        The loop periodically checks for new observations on active
        (pending or observed) predictions. Only cycles in pending or
        observed states are processed.

        Analogi: Jin Soun memasang "alarm" — secara berkala dia cek
        apakah ada bukti baru untuk kasus-kasus yang masih aktif.

        Args:
            interval: Override the default interval (seconds).
                If None, uses the loop_interval from init.
        """
        if self._continuous_running:
            logger.warning("Continuous loop already running")
            return

        interval = interval or self._loop_interval
        self._continuous_stop_event.clear()
        self._continuous_running = True

        def _loop() -> None:
            """Background loop that periodically checks active cycles."""
            logger.info("Continuous prediction loop started (interval=%.0fs)", interval)
            while not self._continuous_stop_event.is_set():
                try:
                    self._continuous_tick()
                except Exception as exc:
                    logger.error("Error in continuous loop tick: %s", exc)

                # Wait for the interval, but allow early stop
                self._continuous_stop_event.wait(timeout=interval)

            self._continuous_running = False
            logger.info("Continuous prediction loop stopped")

        self._continuous_thread = threading.Thread(
            target=_loop,
            name="prediction-loop-continuous",
            daemon=True,
        )
        self._continuous_thread.start()

    def stop_continuous(self, timeout: float = 5.0) -> None:
        """Gracefully stop the continuous prediction loop.

        Args:
            timeout: Maximum time (seconds) to wait for the thread to stop.
        """
        if not self._continuous_running:
            logger.debug("Continuous loop not running")
            return

        logger.info("Stopping continuous prediction loop...")
        self._continuous_stop_event.set()

        if self._continuous_thread is not None:
            self._continuous_thread.join(timeout=timeout)
            if self._continuous_thread.is_alive():
                logger.warning(
                    "Continuous loop thread did not stop within %.1fs", timeout
                )
            else:
                logger.info("Continuous loop thread stopped cleanly")
            self._continuous_thread = None

        self._continuous_running = False

    @property
    def is_continuous_running(self) -> bool:
        """Check if the continuous loop is currently running."""
        return self._continuous_running

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_cycle_metrics(self, cycle_id: str) -> dict | None:
        """Get computed metrics for a specific cycle.

        Searches both active cycles and completed history.

        Args:
            cycle_id: The cycle to get metrics for.

        Returns:
            A dict with cycle metrics, or None if cycle not found.
        """
        tracker = self.get_cycle(cycle_id)
        if tracker is None:
            return None
        return self._compute_cycle_metrics(cycle_id, tracker)

    def get_all_metrics(self) -> dict[str, dict]:
        """Get metrics for all active cycles.

        Returns:
            A dict mapping cycle_id → metrics dict.
        """
        result: dict[str, dict] = {}
        for cid, ct in self._cycles.items():
            result[cid] = self._compute_cycle_metrics(cid, ct)
        # Also include recently finalized cycles from history
        for ct in self._cycle_history[-20:]:  # Last 20 completed
            result[ct.cycle_id] = self._compute_cycle_metrics(ct.cycle_id, ct)
        return result

    def get_cycle_tree(self, cycle_id: str) -> dict:
        """Get the full parent→child tree for a cycle.

        Searches both active cycles and completed history.
        Useful for tracing re-prediction chains.

        Args:
            cycle_id: The root cycle ID.

        Returns:
            A nested dict representing the cycle tree.
        """
        tracker = self.get_cycle(cycle_id)
        if tracker is None:
            return {}

        tree: dict = {
            "cycle_id": cycle_id,
            "concept": tracker.concept,
            "state": tracker.state,
            "children": [],
        }

        for child_id in tracker.child_cycle_ids:
            tree["children"].append(self.get_cycle_tree(child_id))

        return tree

    # ------------------------------------------------------------------
    # Internal: Cycle creation / reuse
    # ------------------------------------------------------------------

    def _ensure_cycle(
        self,
        concept: str,
        context: list[str],
    ) -> tuple[str, CycleTracker, Prediction]:
        """Get or create a cycle tracker and prediction for a concept.

        If there is an existing pending cycle for this concept, reuse it.
        Otherwise, create a new one.

        Args:
            concept: The concept to predict.
            context: Context atoms for the prediction.

        Returns:
            A tuple of (cycle_id, tracker, prediction).
        """
        # Check for existing pending cycle
        existing_id = self._concept_cycle_map.get(concept)
        if existing_id is not None:
            tracker = self._cycles.get(existing_id)
            if tracker is not None and tracker.state == "pending":
                # Reuse existing pending prediction
                # Find the prediction from the engine that matches
                predictions = self._engine.get_predictions()
                for pred in predictions:
                    if pred.concept == concept:
                        logger.debug(
                            "Reusing pending cycle %s for concept '%s'",
                            existing_id, concept
                        )
                        return existing_id, tracker, pred

        # Create a new cycle
        cycle_id = uuid.uuid4().hex[:8]
        prediction = self._engine.predict(concept, context)

        tracker = CycleTracker(
            cycle_id=cycle_id,
            concept=concept,
            state="pending",
        )

        self._cycles[cycle_id] = tracker
        self._concept_cycle_map[concept] = cycle_id
        self._consecutive_confirms[cycle_id] = 0
        self._cumulative_error[cycle_id] = 0.0
        self._anomaly_counts[cycle_id] = 0

        logger.debug(
            "Created new cycle %s for concept '%s' (confidence=%.3f)",
            cycle_id, concept, prediction.confidence
        )

        return cycle_id, tracker, prediction

    # ------------------------------------------------------------------
    # Internal: Observation ingestion
    # ------------------------------------------------------------------

    def _ingest_observation(self, observation: str) -> None:
        """Ingest an observation into the RSVS graph.

        Handles the case where RSVS is not available gracefully.

        Args:
            observation: The observation text to ingest.
        """
        if not observation or not observation.strip():
            logger.debug("Empty observation, skipping ingestion")
            return

        if self.rsvs_available:
            try:
                self._bridge.ingest(observation)
                logger.debug("Ingested observation into RSVS: %s", observation[:100])
            except Exception as exc:
                logger.warning("RSVS ingest failed: %s", exc)
        else:
            # Fallback: use the engine's observe_and_update which handles fallback
            logger.debug("RSVS not available, observation will be processed via engine fallback")

    # ------------------------------------------------------------------
    # Internal: Belief update computation
    # ------------------------------------------------------------------

    def _compute_belief_update(
        self,
        prediction: Prediction,
        observation: str,
    ) -> BeliefUpdate | None:
        """Compute a belief update by delegating to PredictiveEngine.

        The engine's observe_and_update() handles both RSVS and fallback
        modes. We invoke it and extract the update for our specific concept.

        Args:
            prediction: The prediction to update.
            observation: The observation text.

        Returns:
            A BeliefUpdate for this concept, or None.
        """
        # Use the engine's observe_and_update to do the heavy lifting
        updates = self._engine.observe_and_update(observation)

        # Find the update for our concept
        for update in updates:
            if update.concept == prediction.concept:
                return update

        # If no update was returned for our concept, return None
        # This can happen if the observation wasn't relevant to the concept
        logger.debug(
            "No belief update produced for concept '%s' from observation",
            prediction.concept
        )
        return None

    # ------------------------------------------------------------------
    # Internal: Anomaly detection
    # ------------------------------------------------------------------

    def _detect_cycle_anomaly(self, prediction: Prediction) -> Anomaly | None:
        """Detect anomaly for a specific prediction.

        Delegates to the engine's detect_anomalies() and filters for
        the given concept.

        Args:
            prediction: The prediction to check.

        Returns:
            An Anomaly if detected, or None.
        """
        anomalies = self._engine.detect_anomalies()

        for anomaly in anomalies:
            if anomaly.concept == prediction.concept:
                return anomaly

        return None

    # ------------------------------------------------------------------
    # Internal: RSVS feedback integration
    # ------------------------------------------------------------------

    def _maybe_trigger_rsvs_feedback(
        self,
        cycle_id: str,
        tracker: CycleTracker,
        belief_update: BeliefUpdate | None,
        anomaly: Anomaly | None,
    ) -> dict | None:
        """Trigger RSVS reflection/consolidation if conditions are met.

        Conditions:
        - bridge.run_reflection() if confidence < 0.3 after update
        - bridge.consolidate() if 3+ anomalies detected in same cycle

        Analogi: Jika keyakinan Jin Soun turun sangat rendah, dia
        "merenung" (refleksi) — meninjau ulang semua bukti. Jika
        terlalu banyak anomali dalam satu kasus, dia "konsolidasi" —
        menggabungkan dan menyederhanakan catatannya.

        Args:
            cycle_id: The current cycle ID.
            tracker: The cycle tracker.
            belief_update: The belief update (if any).
            anomaly: The anomaly (if any).

        Returns:
            A dict with feedback results, or None if no feedback was triggered.
        """
        feedback: dict = {}
        triggered = False

        # Condition 1: Reflection when confidence drops below threshold
        if belief_update is not None and belief_update.new_confidence < _REFLECTION_CONFIDENCE_THRESHOLD:
            try:
                reflection_result = self._bridge.run_reflection()
                feedback["reflection"] = reflection_result
                triggered = True
                logger.info(
                    "Cycle %s: triggered RSVS reflection (confidence=%.3f < %.3f)",
                    cycle_id, belief_update.new_confidence, _REFLECTION_CONFIDENCE_THRESHOLD
                )
            except Exception as exc:
                logger.warning("RSVS run_reflection() failed: %s", exc)
                feedback["reflection_error"] = str(exc)
                triggered = True

        # Condition 2: Consolidation when 3+ anomalies in same cycle
        current_anomaly_count = self._anomaly_counts.get(cycle_id, 0)
        if anomaly is not None:
            current_anomaly_count += 1

        if current_anomaly_count >= _CONSOLIDATION_ANOMALY_COUNT:
            try:
                consolidation_result = self._bridge.consolidate()
                feedback["consolidation"] = consolidation_result
                triggered = True
                logger.info(
                    "Cycle %s: triggered RSVS consolidation (%d anomalies >= %d)",
                    cycle_id, current_anomaly_count, _CONSOLIDATION_ANOMALY_COUNT
                )
            except Exception as exc:
                logger.warning("RSVS consolidate() failed: %s", exc)
                feedback["consolidation_error"] = str(exc)
                triggered = True

        return feedback if triggered else None

    # ------------------------------------------------------------------
    # Internal: State transitions
    # ------------------------------------------------------------------

    def _transition_state(
        self,
        cycle_id: str,
        tracker: CycleTracker,
        belief_update: BeliefUpdate | None,
        anomaly: Anomaly | None,
    ) -> str:
        """Determine and apply the next lifecycle state for a cycle.

        State transitions:
        - pending → observed: After first belief update
        - observed → resolved: 3+ consecutive confirms with delta < 0.05
        - observed → escalated: Large prediction error (anomaly detected)
        - Any → retired: Staleness limit exceeded (handled separately)

        Args:
            cycle_id: The cycle ID.
            tracker: The cycle tracker.
            belief_update: The belief update (if any).
            anomaly: The anomaly (if any).

        Returns:
            The new state.
        """
        old_state = tracker.state

        # pending → observed after first belief update
        if old_state == "pending" and belief_update is not None:
            tracker.state = "observed"
            logger.debug(
                "Cycle %s: pending → observed", cycle_id
            )
            old_state = "observed"

        # Check for escalation (large anomaly)
        if old_state == "observed" and anomaly is not None and anomaly.delta > 0.5:
            tracker.state = "escalated"
            tracker.resolution_reason = (
                f"Large prediction error escalated (delta={anomaly.delta:.3f})"
            )
            self._finalize_cycle(cycle_id, tracker)
            logger.info(
                "Cycle %s: observed → escalated (delta=%.3f)",
                cycle_id, anomaly.delta
            )
            return tracker.state

        # Check for resolution (consecutive confirms)
        if old_state == "observed" and belief_update is not None:
            if belief_update.direction == "confirm":
                delta = abs(belief_update.new_confidence - belief_update.old_confidence)
                if delta < self._convergence_delta:
                    self._consecutive_confirms[cycle_id] = (
                        self._consecutive_confirms.get(cycle_id, 0) + 1
                    )
                else:
                    # Reset counter — not converging
                    self._consecutive_confirms[cycle_id] = 0

                # Check if we've reached convergence
                if self._consecutive_confirms.get(cycle_id, 0) >= self._convergence_count:
                    tracker.state = "resolved"
                    tracker.resolution_reason = (
                        f"Converged after {self._convergence_count} consecutive "
                        f"confirms (delta < {self._convergence_delta})"
                    )
                    self._finalize_cycle(cycle_id, tracker)
                    logger.info(
                        "Cycle %s: observed → resolved (%d consecutive confirms)",
                        cycle_id, self._convergence_count
                    )
                    return tracker.state
            else:
                # Revision resets convergence counter
                self._consecutive_confirms[cycle_id] = 0

        # Also check staleness for active cycles
        self.retire_stale_cycles()

        return tracker.state

    # ------------------------------------------------------------------
    # Internal: Auto re-prediction
    # ------------------------------------------------------------------

    def _auto_re_predict(
        self,
        parent_cycle_id: str,
        concept: str,
        context: list[str],
        parent_tracker: CycleTracker,
    ) -> Prediction:
        """Create an automatic re-prediction after a significant confidence drop.

        The re-prediction uses the updated graph state (after the belief
        update modified RSVS). The new prediction is linked to the
        original as a "child" cycle.

        Analogi: Setelah keyakinan Jin Soun turun drastis, dia buat
        prediksi baru berdasarkan pemahaman yang diperbarui. Prediksi
        baru ini terhubung ke prediksi lama sebagai "anak" — jejak
        audit yang jelas.

        Args:
            parent_cycle_id: The cycle that triggered the re-prediction.
            concept: The concept to re-predict.
            context: Updated context atoms.
            parent_tracker: The parent cycle tracker.

        Returns:
            The new Prediction object.
        """
        # Create a new cycle for the re-prediction
        child_cycle_id = uuid.uuid4().hex[:8]

        # Make a fresh prediction using the updated engine state
        re_prediction = self._engine.predict(concept, context)

        # Create child tracker
        child_tracker = CycleTracker(
            cycle_id=child_cycle_id,
            concept=concept,
            state="pending",
            parent_cycle_id=parent_cycle_id,
        )

        # Register the child cycle
        self._cycles[child_cycle_id] = child_tracker
        self._concept_cycle_map[concept] = child_cycle_id
        self._consecutive_confirms[child_cycle_id] = 0
        self._cumulative_error[child_cycle_id] = 0.0
        self._anomaly_counts[child_cycle_id] = 0

        # Link parent → child
        parent_tracker.child_cycle_ids.append(child_cycle_id)

        logger.info(
            "Auto re-prediction: child cycle %s created for concept '%s' "
            "(parent=%s, new_confidence=%.3f)",
            child_cycle_id, concept, parent_cycle_id, re_prediction.confidence
        )

        return re_prediction

    # ------------------------------------------------------------------
    # Internal: Cycle metrics computation
    # ------------------------------------------------------------------

    def _compute_cycle_metrics(self, cycle_id: str, tracker: CycleTracker) -> dict:
        """Compute metrics for a cycle.

        Metrics:
        - accuracy: Ratio of confirmed observations to total observations
        - avg_prediction_error: Average delta across all observations
        - convergence_rate: How quickly the cycle is converging (if observed)
        - observation_count: Total number of observations
        - cumulative_error: Total prediction error
        - anomaly_count: Number of anomalies detected
        - consecutive_confirms: Current streak of confirming observations

        Args:
            cycle_id: The cycle ID.
            tracker: The cycle tracker.

        Returns:
            A dict with computed metrics.
        """
        observations = tracker.observations
        total_obs = len(observations)

        # Accuracy: fraction of observations with small delta (confirmed)
        confirmed = sum(
            1 for obs in observations
            if obs.get("delta", 1.0) < self._convergence_delta
        )
        accuracy = confirmed / max(total_obs, 1)

        # Average prediction error
        cumulative = self._cumulative_error.get(cycle_id, 0.0)
        avg_error = cumulative / max(total_obs, 1)

        # Convergence rate: consecutive_confirms / convergence_count
        # (1.0 means fully converged, 0.0 means no convergence yet)
        conf_streak = self._consecutive_confirms.get(cycle_id, 0)
        convergence_rate = min(1.0, conf_streak / max(self._convergence_count, 1))

        return {
            "accuracy": round(accuracy, 4),
            "avg_prediction_error": round(avg_error, 4),
            "convergence_rate": round(convergence_rate, 4),
            "observation_count": total_obs,
            "cumulative_error": round(cumulative, 4),
            "anomaly_count": self._anomaly_counts.get(cycle_id, 0),
            "consecutive_confirms": conf_streak,
        }

    # ------------------------------------------------------------------
    # Internal: Cycle finalization
    # ------------------------------------------------------------------

    def _finalize_cycle(self, cycle_id: str, tracker: CycleTracker) -> None:
        """Finalize a terminal cycle and move it to history.

        Also cleans up the concept→cycle mapping if this was the active cycle.

        Args:
            cycle_id: The cycle ID.
            tracker: The cycle tracker (now in terminal state).
        """
        # Move to history
        self._cycle_history.append(tracker)

        # Remove from active cycles
        if cycle_id in self._cycles:
            del self._cycles[cycle_id]

        # Clean up concept mapping if this was the active cycle
        concept = tracker.concept
        if self._concept_cycle_map.get(concept) == cycle_id:
            del self._concept_cycle_map[concept]

        # Clean up auxiliary tracking dicts
        self._consecutive_confirms.pop(cycle_id, None)
        self._cumulative_error.pop(cycle_id, None)
        self._anomaly_counts.pop(cycle_id, None)

        logger.debug(
            "Cycle %s finalized (state=%s, reason=%s)",
            cycle_id, tracker.state, tracker.resolution_reason
        )

    # ------------------------------------------------------------------
    # Internal: Continuous loop tick
    # ------------------------------------------------------------------

    def _continuous_tick(self) -> None:
        """One tick of the continuous loop.

        Checks for stale cycles and retires them. In a full implementation,
        this would also check for new observations from external sources.

        Analogi: Jin Soun secara berkala mengecek apakah ada kasus
        yang sudah terlalu lama tidak ada perkembangan.
        """
        # Retire stale cycles
        retired = self.retire_stale_cycles()
        if retired:
            logger.debug("Continuous tick: retired %d stale cycles", len(retired))

        # Count active cycles
        active = len(self.active_cycles)
        if active > 0:
            logger.debug("Continuous tick: %d active cycles", active)
