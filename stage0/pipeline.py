"""
AamPipeline — Wire semua layer jadi satu sistem

AAM — Aphantasic Abstraction Model

Analogi: Ini adalah Jin Soun secara keseluruhan —
mulai dari mendengar pertanyaan, mengingat semua yang relevan,
memahami relasi, mendeteksi anomali, sampai mengeluarkan
kesimpulan yang bisa diaudit.

Flow:
    User Input
      -> Context Layer (internet search jika perlu)
      -> Situation Layer (ingest chat, cari konteks relevan)
      -> AAM Core (spreading activation, structural analysis)
      -> Predictive Engine (predict, detect anomalies)
      -> Pattern Output (pattern completion + narrative)
      -> Layer 3 Deductive Reasoning (if applicable — policy/coder/reasoning)
      -> Appraise Self-Check (verify output consistency)
      -> Final Output (traceable reasoning chain + confidence)

Pipeline Integration Gaps (P-01 through P-06) solved here:
    P-01: Structural data contracts between layers
    P-02: Appraise self-check before returning output
    P-03: Streaming support for long-running operations
    P-04: Source provenance enforced at RSVS level
    P-05: Consolidation and reflection maintenance
    P-06: Consistent error handling with AamError hierarchy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Ensure stage0/ is on sys.path so internal imports work ──
_STAGE0_DIR = str(Path(__file__).resolve().parent)
if _STAGE0_DIR not in sys.path:
    sys.path.insert(0, _STAGE0_DIR)

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional

from layer2.bridge import V12PipelineBridge, get_bridge, is_rust_core_available
from layer2.llm import generate_narrative
from layer2.context import ContextLayer, SOURCE_TRUST
from layer2.situation import SituationLayer
from layer2.predictive import PredictiveEngine, Prediction, Anomaly, BeliefUpdate
from layer2.pattern import PatternOutput, ReasoningStep, PatternResult
from layer2.temporal import TemporalTracker, TemporalRecord
# Diffusion LLM removed — narrative generation uses layer2.llm directly

# Layer 0: Perceptual Front-End
from layer0 import TextAbstractor, ImageAbstractor, AudioAbstractor, VideoAbstractor
from layer0 import observation_to_ingest_data, ingest_observation
from layer0.base import PerceptualObservation as Layer0PerceptualObservation, ModalityType
from layer3.reasoning import ReasoningEngine, DeductiveChain, DeductiveStep
from layer3.policy import DeductivePolicyEngine
from layer3.coder import DeductiveCoderLayer

# 5-Pillar Validation Gates — full integration
from validation_gates import (
    SignalExtractionGate, SignalResult,
    RegimeDetectionGate, RegimeState,
    UncertaintyCalibrationGate, CalibrationRecord,
    StatisticalEdgeGate, EdgeAssessment,
    ExecutionDisciplineGate, DisciplineVerdict,
)
from validation_gates.signal_extraction import SignalVerdict
from validation_gates.regime_detection import CognitiveRegime
from validation_gates.statistical_edge import ReasoningPath

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P-06: AamError exception hierarchy
# ---------------------------------------------------------------------------


class AamError(Exception):
    """Base exception for all AAM pipeline errors."""

    def __init__(self, message: str, layer: str = "", details: Optional[dict] = None):
        super().__init__(message)
        self.layer = layer
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert to a dict for serialization."""
        return {
            "type": type(self).__name__,
            "message": str(self),
            "layer": self.layer,
            "details": self.details,
        }


class LayerError(AamError):
    """Error originating from a specific pipeline layer."""
    pass


class IngestError(AamError):
    """Error during ingestion into the knowledge graph."""
    pass


class ReasoningError(AamError):
    """Error during reasoning / pattern completion."""
    pass


class BridgeError(AamError):
    """Error communicating with the RSVS bridge / Rust core."""
    pass


class MaintenanceError(AamError):
    """Error during graph maintenance (consolidate / reflect)."""
    pass


# ---------------------------------------------------------------------------
# P-01: Structural data contracts between layers
# ---------------------------------------------------------------------------


@dataclass
class PerceptualObservation:
    """Structured output from Layer 0 (Context Layer).

    Carries the raw text plus extracted structural information
    so downstream layers don't have to re-parse strings.
    """

    text: str
    """The original input text."""

    source: str = "user_input"
    """Source provenance (e.g. 'user_input', 'web_search')."""

    trust: float = 1.0
    """Trust score for this source."""

    search_results: list[dict] = field(default_factory=list)
    """Web search results, if any."""

    ingest_stats: Optional[dict] = None
    """Stats from RSVS ingestion."""

    context_atoms: list[str] = field(default_factory=list)
    """Extracted context atoms from the observation."""

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "text": self.text,
            "source": self.source,
            "trust": self.trust,
            "search_results": self.search_results,
            "ingest_stats": self.ingest_stats,
            "context_atoms": self.context_atoms,
        }


@dataclass
class StructuralDelta:
    """Graph changes from Layer 1 (RSVS / Situation Layer).

    Represents what changed in the knowledge graph after ingestion
    and context retrieval — new nodes, new edges, sense changes,
    and confidence updates.
    """

    new_nodes: list[str] = field(default_factory=list)
    """New atom labels promoted into the graph."""

    new_edges: list[dict] = field(default_factory=list)
    """New edges created (each dict has 'from', 'to', 'weight')."""

    sense_changes: list[dict] = field(default_factory=list)
    """Sense changes: assigned, created, frozen."""

    confidence_updates: dict[str, float] = field(default_factory=dict)
    """Confidence updates: label → new confidence."""

    ingest_stats: Optional[dict] = None
    """Raw ingest stats from RSVS bridge."""

    relevant_context: list[dict] = field(default_factory=list)
    """Relevant context retrieved from the situation layer."""

    active_senses: list[dict] = field(default_factory=list)
    """Currently active senses from the situation layer."""

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "new_nodes": self.new_nodes,
            "new_edges": self.new_edges,
            "sense_changes": self.sense_changes,
            "confidence_updates": self.confidence_updates,
            "ingest_stats": self.ingest_stats,
            "relevant_context": self.relevant_context,
            "active_senses": self.active_senses,
        }


@dataclass
class ReasoningRequest:
    """Structured input for Layer 2 (Cognitive Runtime / Pattern Completion).

    Carries the trigger question plus evidence references and
    structural context so the reasoning layer doesn't have to
    re-query the graph.
    """

    trigger: str
    """The original question / trigger text."""

    context_atoms: list[str] = field(default_factory=list)
    """Context atoms for focusing the reasoning."""

    evidence_refs: list[str] = field(default_factory=list)
    """Node labels that serve as evidence references."""

    predictions: list[dict] = field(default_factory=list)
    """Active predictions to compare against."""

    anomalies: list[dict] = field(default_factory=list)
    """Known anomalies to feed into pattern completion."""

    source: str = "user_input"
    """Source provenance for trust weighting."""

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "trigger": self.trigger,
            "context_atoms": self.context_atoms,
            "evidence_refs": self.evidence_refs,
            "predictions": self.predictions,
            "anomalies": self.anomalies,
            "source": self.source,
        }


# Type aliases for clarity
Layer0Output = PerceptualObservation
Layer1Output = StructuralDelta
Layer2Output = ReasoningRequest


# ---------------------------------------------------------------------------
# P-03: Streaming support
# ---------------------------------------------------------------------------


@dataclass
class PipelineEvent:
    """Event emitted by ask_stream() after each layer completes.

    Allows callers to show progress, partial results, or cancel
    long-running operations.
    """

    layer: str
    """Which layer just completed (e.g. 'context', 'situation')."""

    status: str
    """Status: 'complete', 'partial', 'error'."""

    partial_result: Optional[dict] = None
    """Partial result data from this layer."""

    timestamp: float = field(default_factory=time.time)
    """When this event was emitted."""

    error: Optional[str] = None
    """Error message if status is 'error'."""

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "layer": self.layer,
            "status": self.status,
            "partial_result": self.partial_result,
            "timestamp": self.timestamp,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AamResponse:
    """Response dari AamPipeline — lengkap dengan traceability."""

    answer: str
    """Jawaban utama (naratif)."""

    confidence: float
    """Confidence keseluruhan (0.0-1.0)."""

    reasoning_chain: list[ReasoningStep]
    """Step-by-step reasoning yang bisa ditelusuri."""

    evidence_chain: list[dict]
    """Evidence nodes dari RSVS graph."""

    anomalies: list[dict]
    """Anomali yang terdeteksi selama proses."""

    predictions: list[dict]
    """Prediksi yang dibuat."""

    belief_updates: list[dict]
    """Belief updates yang terjadi."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata tambahan (latency, source, scope, dll)."""

    appraise_warning: Optional[str] = None
    """P-02: Warning from appraise self-check, if any."""

    errors: list[dict] = field(default_factory=list)
    """P-06: Non-fatal errors encountered during processing."""

    def to_dict(self) -> dict:
        """Convert ke dictionary untuk JSON serialization."""
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 3),
            "reasoning_chain": [
                {
                    "step": s.step_type,
                    "description": s.description,
                    "confidence": round(s.confidence, 3),
                    "evidence_nodes": s.evidence_nodes,
                    "evidence_node_ids": [
                        {"node_id": nid, "sense_id": sid}
                        for nid, sid in s.evidence_node_ids
                    ],
                    "grounding_scores": s.grounding_scores,
                }
                for s in self.reasoning_chain
            ],
            "evidence_chain": self.evidence_chain,
            "anomalies": self.anomalies,
            "predictions": self.predictions,
            "belief_updates": self.belief_updates,
            "metadata": self.metadata,
            "appraise_warning": self.appraise_warning,
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert ke JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AamPipeline:
    """Main pipeline — wires semua layer jadi satu sistem.

    Usage:
        pipeline = AamPipeline()

        # Simple query
        result = pipeline.ask("Siapa yang mencuri Snow Plum Pill?")
        print(result.answer)
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Evidence: {result.evidence_chain}")

        # With internet search
        result = pipeline.ask(
            "Apa itu quantum computing?",
            search_internet=True
        )

        # With scope filter
        pipeline.set_scope(["official_doc", "academic"])
        result = pipeline.ask("Berapa tarif pajak penghasilan?")

        # Streaming
        async for event in pipeline.ask_stream("Tell me about AI"):
            print(f"[{event.layer}] {event.status}")

        # Manual maintenance
        pipeline.force_maintenance()

    Analogi novel:
        Ini adalah Jin Soun secara keseluruhan.
        - ContextLayer = Simhyeon Pavilion + kemampuan membatasi sumber
        - SituationLayer = ingatan percakapan + active senses
        - AAM Core = structural memory (30 tahun ingatan)
        - PredictiveEngine = "aku predict X, ternyata Y, update belief"
        - PatternOutput = pattern completion + narrative
    """

    def __init__(
        self,
        rsvs_instance=None,
        bridge: Optional[V12PipelineBridge] = None,
        eta: float = 0.1,
        anomaly_threshold: float = 0.3,
        auto_search: bool = False,
        use_llm: bool = True,
        language: str = "id",
        maintenance_interval: int = 50,
    ):
        """Initialize the pipeline with all layers.

        Args:
            rsvs_instance: Optional RSVS instance. If None, try to create one.
                Deprecated: prefer passing `bridge` instead.
            bridge: Optional shared AbstractionBridge instance. If provided, all
                layers will share this bridge (recommended). If None and
                rsvs_instance is also None, a new bridge is created via
                get_bridge().
            eta: Learning rate for predictive coding (default: 0.1).
            anomaly_threshold: Threshold for anomaly detection (default: 0.3).
            auto_search: Automatically search internet when confidence is low.
            use_llm: Whether to use LLM for narrative generation (default: True).
            language: Output language for narratives ("id" or "en").
            maintenance_interval: Run auto-maintenance every N ingests
                (default: 50). Set to 0 to disable auto-maintenance.
        """
        self._eta = eta
        self._anomaly_threshold = anomaly_threshold
        self._auto_search = auto_search
        self._use_llm = use_llm
        self._language = language
        self._maintenance_interval = maintenance_interval

        # Create a shared bridge so all layers use the same RSVS instance
        if bridge is not None:
            self._bridge = bridge
        elif rsvs_instance is not None:
            self._bridge = get_bridge()
        else:
            self._bridge = get_bridge()

        # Initialize all layers with the SHARED bridge
        self.context = ContextLayer(bridge=self._bridge)
        self.situation = SituationLayer(bridge=self._bridge)
        self.predictive = PredictiveEngine(
            bridge=self._bridge,
            eta=eta,
            anomaly_threshold=anomaly_threshold,
        )
        self.temporal = TemporalTracker()
        self.pattern = PatternOutput(bridge=self._bridge, temporal_tracker=self.temporal)

        # Narrative generation now uses layer2.llm directly
        # (Diffusion LLM concept archived — not yet trainable)

        # Layer 3: Deductive Reasoning (optional — activated when needed)
        # Analogi: Jin Soun tidak hanya menarik kesimpulan dari pola,
        # tapi juga menelusuri rantai deduksi langkah demi langkah,
        # mengecek kepatuhan, dan menganalisis kode — layer tambahan
        # yang memperkaya jawaban tanpa mengganggu flow utama.
        self.reasoning = ReasoningEngine(bridge=self._bridge)
        self.deductive_policy = DeductivePolicyEngine(bridge=self._bridge)
        self.deductive_coder = DeductiveCoderLayer(bridge=self._bridge)

        # Layer 0: Perceptual Front-End
        # TextAbstractor gets the bridge for LLM-driven tuple extraction.
        # If bridge lacks generate(), TextAbstractor gracefully falls back
        # to noun-phrase extraction (no raw data stored, only relations).
        self._text_abstractor = TextAbstractor(llm_bridge=bridge)
        # Image/Audio/Video abstractors are lazy-initialized (need external bridges)
        self._image_abstractor: Optional[ImageAbstractor] = None
        self._audio_abstractor: Optional[AudioAbstractor] = None
        self._video_abstractor: Optional[VideoAbstractor] = None

        # Internal state
        self._conversation_history: list[dict] = []

        # P-05: Maintenance tracking
        self._ingest_count: int = 0
        self._last_maintenance_time: float = 0.0
        self._maintenance_log: list[dict] = []

        # 5-Pillar Validation Gates — standalone pipeline-level instances
        # These are used for pipeline-level checks and metadata collection
        # in addition to the per-layer gate instances.
        self.signal_gate = SignalExtractionGate()     # Gate 1: L0/L1 checkpoint
        self.regime_gate = RegimeDetectionGate()      # Gate 2: L2 checkpoint
        self.calibration_gate = UncertaintyCalibrationGate()  # Gate 3: L3 checkpoint
        self.edge_gate = StatisticalEdgeGate()        # Gate 4: L4 checkpoint
        self.discipline_gate = ExecutionDisciplineGate()  # Gate 5: L5 checkpoint

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def ask(
        self,
        question: str,
        context: list[str] | None = None,
        search_internet: bool = False,
        source: str = "user_input",
    ) -> AamResponse:
        """Ask a question and get a traceable, evidence-backed response.

        This is the main entry point. It runs through all layers:

        1. Context Layer: Search internet if needed, apply scope filter
        2. Situation Layer: Ingest the question, find relevant context
        3. RSVS Core: Spreading activation, structural analysis
        4. Predictive Engine: Predict, detect anomalies
        5. Pattern Output: Pattern completion, generate narrative
        6. Appraise Self-Check: Verify output consistency (P-02)

        Args:
            question: The question or trigger text.
            context: Optional context atoms to guide the query.
            search_internet: Whether to search the internet for additional info.
            source: Source type for trust scoring (default: "user_input").

        Returns:
            AamResponse with answer, reasoning chain, evidence, and confidence.
        """
        start_time = time.time()
        all_evidence: list[dict] = []
        all_anomalies: list[dict] = []
        all_predictions: list[dict] = []
        all_belief_updates: list[dict] = []
        non_fatal_errors: list[dict] = []

        # 5-Pillar Gate tracking for metadata
        gate_results: dict[str, dict] = {}

        # ══════════════════════════════════════════════════════════════
        # 5-PILLAR GATE 1: Signal Extraction — filter noise before ingest
        # ══════════════════════════════════════════════════════════════
        try:
            signal_result = self.signal_gate.evaluate(
                raw_input=question,
            )
            gate_results["gate_1_signal"] = signal_result.to_dict()
            if signal_result.verdict == SignalVerdict.REJECT:
                logger.info("Gate 1 REJECTED input as noise: %s", signal_result.reason)
                # Still process, but mark as low-signal
                all_evidence.append({
                    "type": "signal_gate_reject",
                    "reason": signal_result.reason,
                })
        except Exception as exc:
            logger.debug("Gate 1 (Signal) failed: %s", exc)

        # G2-6: Systematic chat ingest — every conversation enriches the graph
        # Analogi: Setiap percakapan yang Jin Soun dengar dicatat di Simhyeon Pavilion
        try:
            self.situation.add_message("user", question)
            # Also ingest into context layer for provenance tracking
            self.context.ingest_text(question, source=source)
        except Exception as exc:
            logger.debug("Systematic chat ingest failed: %s", exc)

        # ---- Step 1: Context Layer (P-01: produce Layer0Output) ----
        try:
            layer0_output = self._run_context_layer(
                question, source, search_internet,
            )
            # Collect evidence from internet search
            if layer0_output.search_results:
                all_evidence.append({
                    "type": "internet_search",
                    "query": question,
                    "results_count": len(layer0_output.search_results),
                    "trust": layer0_output.trust,
                })
        except AamError as exc:
            non_fatal_errors.append(exc.to_dict())
            # Fallback: create minimal layer0 output
            layer0_output = Layer0Output(text=question, source=source)
        except Exception as exc:
            err = LayerError(
                f"Context layer failed: {exc}", layer="context",
                details={"question": question[:100]},
            )
            non_fatal_errors.append(err.to_dict())
            layer0_output = Layer0Output(text=question, source=source)

        # ---- Step 2: Situation Layer (P-01: produce Layer1Output) ----
        try:
            layer1_output = self._run_situation_layer(
                question, source, layer0_output,
            )
            # Collect evidence from situation recall
            if layer1_output.relevant_context:
                all_evidence.append({
                    "type": "situation_recall",
                    "relevant_concepts": [
                        r.get("label", r.get("concept", ""))
                        for r in layer1_output.relevant_context[:5]
                    ],
                })
        except AamError as exc:
            non_fatal_errors.append(exc.to_dict())
            layer1_output = Layer1Output()
        except Exception as exc:
            err = LayerError(
                f"Situation layer failed: {exc}", layer="situation",
            )
            non_fatal_errors.append(err.to_dict())
            layer1_output = Layer1Output()

        # ══════════════════════════════════════════════════════════════
        # 5-PILLAR GATE 2: Regime Detection — detect current cognitive environment
        # ══════════════════════════════════════════════════════════════
        try:
            regime_state = self.situation.current_regime
            gate_results["gate_2_regime"] = regime_state.to_dict()
        except Exception as exc:
            logger.debug("Gate 2 (Regime) failed: %s", exc)

        # ---- Step 3: RSVS Core + Predictive Engine ----
        # Make predictions based on context
        context_atoms = context or layer1_output.active_senses and [
            s.get("label", s.get("concept", ""))
            for s in layer1_output.active_senses[:5]
            if s.get("label") or s.get("concept")
        ] or layer0_output.context_atoms

        try:
            prediction = self.predictive.predict(question, context_atoms)
            if prediction:
                all_predictions.append({
                    "concept": prediction.concept,
                    "expected": prediction.expected_compositions,
                    "confidence": prediction.confidence,
                })
        except Exception as exc:
            err = ReasoningError(
                f"Prediction failed: {exc}", layer="predictive",
            )
            non_fatal_errors.append(err.to_dict())
            prediction = None

        # Detect existing anomalies
        try:
            anomalies = self.predictive.detect_anomalies()
            for anomaly in anomalies:
                all_anomalies.append({
                    "concept": anomaly.concept,
                    "expected": anomaly.expected,
                    "observed": anomaly.observed,
                    "delta": anomaly.delta,
                    "description": anomaly.description,
                })
        except Exception as exc:
            err = ReasoningError(
                f"Anomaly detection failed: {exc}", layer="predictive",
            )
            non_fatal_errors.append(err.to_dict())

        # ---- Step 4: Pattern Completion Output ----
        # P-01: Use ReasoningRequest with evidence references
        reasoning_request = ReasoningRequest(
            trigger=question,
            context_atoms=context_atoms,
            evidence_refs=list(dict.fromkeys(
                n for s in (layer1_output.relevant_context or [])
                for n in [s.get("label", s.get("concept", ""))]
                if n
            )),
            predictions=all_predictions,
            anomalies=all_anomalies,
            source=source,
        )

        try:
            pattern_result = self.pattern.process(
                reasoning_request.trigger, reasoning_request.context_atoms,
            )
        except Exception as exc:
            err = ReasoningError(
                f"Pattern completion failed: {exc}", layer="pattern",
            )
            non_fatal_errors.append(err.to_dict())
            pattern_result = None

        # Extract from pattern result
        if pattern_result:
            try:
                pattern_anomalies = [
                    {
                        "description": a.get("description", str(a)),
                        "concept": a.get("concept", ""),
                    }
                    for a in (pattern_result.anomalies or [])
                ]
                all_anomalies.extend(pattern_anomalies)

                # Collect evidence from reasoning chain
                for step in (pattern_result.steps or []):
                    for node in (step.evidence_nodes or []):
                        all_evidence.append({
                            "type": step.step_type,
                            "node": node,
                        })
            except Exception as exc:
                err = ReasoningError(
                    f"Pattern result extraction failed: {exc}", layer="pattern",
                )
                non_fatal_errors.append(err.to_dict())

        # ---- Step 4.5: Layer 3 Deductive Reasoning (if applicable) ----
        # ══════════════════════════════════════════════════════════════
        # 5-PILLAR GATE 3: Uncertainty Calibration — calibrate confidence
        # ══════════════════════════════════════════════════════════════
        # Calibration is applied within ReasoningEngine.build_chain(),
        # but we also record the pipeline-level calibration result here.
        try:
            raw_conf = pattern_result.confidence if pattern_result else 0.0
            regime_str = self.situation.current_regime.regime.value if hasattr(self.situation.current_regime, 'regime') else ""
            cal_result = self.calibration_gate.calibrate(
                raw_confidence=raw_conf,
                regime=regime_str,
            )
            gate_results["gate_3_calibration"] = cal_result.to_dict()
        except Exception as exc:
            logger.debug("Gate 3 (Calibration) failed: %s", exc)

        # Analogi: Setelah Jin Soun menarik pola, dia menelusuri
        # rantai deduksi — apakah ada bukti yang lebih kuat? Apakah
        # jawaban ini perlu diperkuat dengan penalaran deduktif?
        # Jika gagal, pipeline tetap jalan — Layer 3 adalah optional.
        _deductive_confidence_override: float | None = None
        query_mode = self._detect_query_mode(question)
        if query_mode != "general" and pattern_result:
            try:
                deductive_chain = self.reasoning.build_chain(pattern_result)
                # Enhance the answer with deductive reasoning
                if deductive_chain.steps:
                    # Add deductive chain info to metadata
                    all_evidence.append({
                        "type": "deductive_chain",
                        "mode": query_mode,
                        "steps": len(deductive_chain.steps),
                        "aggregate_confidence": deductive_chain.aggregate_confidence,
                        "conclusion": deductive_chain.conclusion[:200] if deductive_chain.conclusion else "",
                    })
                    # If deductive confidence is higher, use it
                    if deductive_chain.aggregate_confidence > pattern_result.confidence:
                        _deductive_confidence_override = deductive_chain.aggregate_confidence
            except Exception as exc:
                err = ReasoningError(
                    f"Deductive reasoning failed: {exc}", layer="deductive",
                )
                non_fatal_errors.append(err.to_dict())

        # ---- Step 5: Belief Update ----
        # ══════════════════════════════════════════════════════════════
        # 5-PILLAR GATE 4: Statistical Edge — validate reasoning has positive EV
        # ══════════════════════════════════════════════════════════════
        # Edge assessment is applied within PredictiveEngine,
        # but we also record the pipeline-level assessment here.
        try:
            edge_path = ReasoningPath(
                path_type=query_mode if query_mode != "general" else "pattern",
                regime=self.situation.current_regime.regime.value if hasattr(self.situation.current_regime, 'regime') else "",
                step_types=[s.step_type for s in (pattern_result.steps or [])[:3]],
            )
            edge_assessment = self.edge_gate.assess(
                path=edge_path,
                current_confidence=pattern_result.confidence if pattern_result else 0.0,
            )
            gate_results["gate_4_edge"] = edge_assessment.to_dict()
        except Exception as exc:
            logger.debug("Gate 4 (Edge) failed: %s", exc)

        if prediction:
            try:
                belief_updates = self.predictive.observe_and_update(
                    question, source=source,
                )
                for bu in belief_updates:
                    all_belief_updates.append({
                        "concept": bu.concept,
                        "old_confidence": round(bu.old_confidence, 3),
                        "new_confidence": round(bu.new_confidence, 3),
                        "direction": bu.direction,
                        "reason": bu.reason,
                    })
            except Exception as exc:
                err = ReasoningError(
                    f"Belief update failed: {exc}", layer="predictive",
                )
                non_fatal_errors.append(err.to_dict())

        # ---- Build Response ----
        # Determine the main answer
        appraise_warning = None
        if pattern_result and pattern_result.steps:
            reasoning_chain_dicts = [s.to_dict() for s in pattern_result.steps]
            evidence_labels = list(dict.fromkeys(
                n for s in pattern_result.steps for n in s.evidence_nodes
            ))

            # Generate narrative from reasoning chain
            answer = generate_narrative(
                trigger=question,
                reasoning_chain=reasoning_chain_dicts,
                pattern=pattern_result.pattern,
                evidence_nodes=evidence_labels,
                confidence=pattern_result.confidence,
                anomalies=pattern_result.anomalies,
                language=self._language,
                use_llm=self._use_llm,
            )

            confidence = pattern_result.confidence
            reasoning_chain = pattern_result.steps
            pattern_evidence = pattern_result.evidence_chain or []
        else:
            # Fallback: simple answer from situation layer
            answer = self._generate_fallback_answer(
                question,
                layer1_output.relevant_context,
                layer1_output.active_senses,
            )
            confidence = self._estimate_confidence(
                layer1_output.relevant_context,
                layer1_output.active_senses,
            )
            reasoning_chain = []
            pattern_evidence = []

        all_evidence.extend(pattern_evidence)

        # Apply Layer 3 deductive confidence override if applicable
        if _deductive_confidence_override is not None:
            confidence = _deductive_confidence_override

        # G2-4: Apply scope filtering to evidence chain
        # Analogi: Jin Soun hanya mempercayai sumber dalam skup misi saat ini
        scope = self.context.get_scope()
        if scope:
            # Filter evidence to only in-scope sources
            filtered_evidence = []
            for ev in all_evidence:
                ev_source = ev.get("source", ev.get("type", "unknown"))
                # Check if this evidence comes from an in-scope source
                if self.context.is_in_scope(ev_source) or ev.get("type") in ("internet_search", "situation_recall", "deductive_chain"):
                    filtered_evidence.append(ev)
                else:
                    # Evidence from out-of-scope source — keep but mark
                    ev_copy = dict(ev)
                    ev_copy["out_of_scope"] = True
                    filtered_evidence.append(ev_copy)
            all_evidence = filtered_evidence

            # If scope is active, reduce confidence if most evidence is out-of-scope
            in_scope_count = sum(1 for e in all_evidence if not e.get("out_of_scope", False))
            total_count = max(len(all_evidence), 1)
            scope_coverage = in_scope_count / total_count
            if scope_coverage < 0.5:
                confidence *= scope_coverage  # Penalize confidence when scope coverage is low
                appraise_warning = (appraise_warning or "") + (
                    f" Only {scope_coverage:.0%} of evidence is within scope."
                ).strip()

        # G2-5: Weight confidence by source trust
        # Analogi: Jin Soun membedakan "informasi dari Simhyeon Pavilion" vs "gosip tavern"
        source_trust = self.context.trust_score(source)
        if source_trust < 1.0:
            # Adjust confidence based on source trustworthiness
            confidence = confidence * (0.5 + 0.5 * source_trust)

        # GN-5: Bounded execution — check narrative reliability
        # Analogi: Jin Soun tahu kapan tubuhnya tidak bisa mengeksekusi teknik
        # yang dia pikirkan — dan memilih strategi alternatif.
        is_reliable = True
        reliability_reason = "Narrative appears reliable"
        try:
            is_reliable, reliability_reason = self._check_narrative_reliability(answer, pattern_result)
            if not is_reliable:
                # Fallback to structural-only response
                logger.warning("Narrative unreliable: %s — falling back to structural response", reliability_reason)
                # Build a structural-only answer from the reasoning chain
                if pattern_result and pattern_result.steps:
                    structural_parts = []
                    for step in pattern_result.steps:
                        if step.evidence_nodes:
                            structural_parts.append(
                                f"[{step.step_type}] {step.description} "
                                f"(evidence: {', '.join(step.evidence_nodes[:3])}, "
                                f"confidence: {step.confidence:.1%})"
                            )
                    if structural_parts:
                        answer = "Structural analysis (narrative deemed unreliable):\n" + "\n".join(structural_parts)
                        confidence *= 0.7  # Reduce confidence for structural-only response
        except Exception as exc:
            logger.debug("Narrative reliability check failed: %s", exc)

        # ---- Step 6: Appraise Self-Check (P-02) ----
        try:
            appraise_result = self._bridge.appraise(answer)
            verdict = appraise_result.get("verdict", "neutral")
            disagree_pct = appraise_result.get("disagree_pct", 0.0)
            agree_pct = appraise_result.get("agree_pct", 0.0)

            # Flag if verdict indicates clash/disagreement, or if disagree
            # percentage is high. In fallback mode, disagree_pct is always 0
            # but verdict can be "disagree" when agree_pct is very low.
            is_negative = verdict in ("clash", "disagree") or disagree_pct > 0.3

            if is_negative:
                # Calculate penalty based on the severity of the disagreement
                if disagree_pct > 0:
                    penalty = min(0.3, disagree_pct * 0.5)
                elif agree_pct < 0.2:
                    # Very low agreement — modest penalty
                    penalty = 0.1
                else:
                    penalty = 0.05
                confidence = max(0.1, confidence - penalty)
                appraise_warning = (
                    f"Appraise self-check flagged output: verdict={verdict}, "
                    f"agree_pct={agree_pct:.2f}, disagree_pct={disagree_pct:.2f}. "
                    f"Confidence reduced by {penalty:.2f}."
                )
                logger.warning(appraise_warning)
        except Exception as exc:
            err = BridgeError(
                f"Appraise self-check failed: {exc}", layer="appraise",
            )
            non_fatal_errors.append(err.to_dict())

        # Record assistant response in conversation
        self.situation.add_message("assistant", answer)
        self._conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": time.time(),
        })

        latency = time.time() - start_time

        # P-05: Track ingest count for auto-maintenance
        self._ingest_count += 1
        if self._maintenance_interval > 0 and self._ingest_count % self._maintenance_interval == 0:
            try:
                self.maintenance()
            except Exception as exc:
                err = MaintenanceError(
                    f"Auto-maintenance failed: {exc}", layer="maintenance",
                )
                non_fatal_errors.append(err.to_dict())

        return AamResponse(
            answer=answer,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            evidence_chain=all_evidence,
            anomalies=all_anomalies,
            predictions=all_predictions,
            belief_updates=all_belief_updates,
            metadata={
                "latency_s": round(latency, 3),
                "source": source,
                "source_trust": source_trust,
                "scope_active": bool(scope),
                "internet_search": search_internet,
                "context_atoms": context_atoms[:5],
                "conversation_turn": len(self._conversation_history),
                "active_senses_count": len(layer1_output.active_senses),
                "rsvs_available": self._is_rsvs_available(),
                "use_llm": self._use_llm,
                "ingest_count": self._ingest_count,
                "query_mode": query_mode,
                "narrative_reliable": is_reliable,
                "reliability_reason": reliability_reason,
                # 5-Pillar Gate Results
                "validation_gates": gate_results,
            },
            appraise_warning=appraise_warning,
            errors=non_fatal_errors,
        )

    def ask_multimodal(
        self,
        text: str = "",
        image: Any = None,
        audio: Any = None,
        video: Any = None,
        context: list[str] | None = None,
        source: str = "user_input",
    ) -> AamResponse:
        """Ask a question with multi-modal input.

        Processes text, image, audio, and/or video inputs through
        Layer 0 abstractors, then runs the full cognitive pipeline.

        Each modality is independently abstracted into PerceptualTuples
        (no raw data stored — only relations). All tuples are combined
        before being fed into the cognitive pipeline.

        Analogi: Jin Soun tidak hanya membaca dokumen — dia juga
        mendengar langkah kaki dan mengenali teknik dari suara.
        Multi-modal = semua jalur persepsi aktif sekaligus.

        Args:
            text: Text input (optional).
            image: Image input — bytes, path, or description (optional).
            audio: Audio input — bytes, path, or description (optional).
            video: Video input — bytes, path, or description (optional).
            context: Optional context atoms to guide the query.
            source: Source provenance (default: "user_input").

        Returns:
            AamResponse with answer, reasoning chain, evidence, and confidence.
        """
        combined_tuples: list = []  # list of PerceptualTuple
        modality_labels: list[str] = []

        # --- Process each modality through Layer 0 ---

        # Text modality (always available)
        if text:
            try:
                text_obs = self._run_layer0(text, source)
                combined_tuples.extend(text_obs.tuples)
                modality_labels.append("text")
            except Exception as exc:
                logger.debug("Layer 0 text abstraction failed in ask_multimodal: %s", exc)

        # Image modality (lazy-initialized)
        if image is not None:
            try:
                if self._image_abstractor is None:
                    self._image_abstractor = ImageAbstractor(llm_bridge=self._bridge)
                image_obs = self._image_abstractor.abstract(image, context={"source": source})
                combined_tuples.extend(image_obs.tuples)
                modality_labels.append("image")
                # Ingest into RSVS
                if self._bridge.is_available and image_obs.tuples:
                    ingest_data = observation_to_ingest_data(image_obs)
                    if ingest_data:
                        self._bridge.ingest(ingest_data)
            except Exception as exc:
                logger.debug("Layer 0 image abstraction failed in ask_multimodal: %s", exc)

        # Audio modality (lazy-initialized)
        if audio is not None:
            try:
                if self._audio_abstractor is None:
                    self._audio_abstractor = AudioAbstractor(llm_bridge=self._bridge)
                audio_obs = self._audio_abstractor.abstract(audio, context={"source": source})
                combined_tuples.extend(audio_obs.tuples)
                modality_labels.append("audio")
                # Ingest into RSVS
                if self._bridge.is_available and audio_obs.tuples:
                    ingest_data = observation_to_ingest_data(audio_obs)
                    if ingest_data:
                        self._bridge.ingest(ingest_data)
            except Exception as exc:
                logger.debug("Layer 0 audio abstraction failed in ask_multimodal: %s", exc)

        # Video modality (lazy-initialized)
        if video is not None:
            try:
                if self._video_abstractor is None:
                    self._video_abstractor = VideoAbstractor(llm_bridge=self._bridge)
                video_obs = self._video_abstractor.abstract(video, context={"source": source})
                combined_tuples.extend(video_obs.tuples)
                modality_labels.append("video")
                # Ingest into RSVS
                if self._bridge.is_available and video_obs.tuples:
                    ingest_data = observation_to_ingest_data(video_obs)
                    if ingest_data:
                        self._bridge.ingest(ingest_data)
            except Exception as exc:
                logger.debug("Layer 0 video abstraction failed in ask_multimodal: %s", exc)

        # Extract context atoms from all combined tuples
        l0_context_atoms: list[str] = []
        for tuple_ in combined_tuples:
            if tuple_.subject and tuple_.subject not in l0_context_atoms:
                l0_context_atoms.append(tuple_.subject)
            if tuple_.predicate and tuple_.predicate not in l0_context_atoms:
                l0_context_atoms.append(tuple_.predicate)

        # Merge user-provided context with Layer 0 extracted atoms
        merged_context = list(context or [])
        for atom in l0_context_atoms:
            if atom not in merged_context:
                merged_context.append(atom)

        # Build a combined text for the pipeline query
        combined_text = text or ""
        if not combined_text and combined_tuples:
            # If no text provided but we have tuples, build a summary
            parts = []
            for t in combined_tuples[:5]:
                parts.append(f"{t.subject} {t.relation_type.value} {t.predicate}")
            combined_text = " ".join(parts)

        # Run the standard pipeline with the combined text and enriched context
        result = self.ask(
            question=combined_text,
            context=merged_context or None,
            search_internet=False,
            source=source,
        )

        # Add multi-modal metadata to the response
        result.metadata["multimodal"] = True
        result.metadata["modalities"] = modality_labels
        result.metadata["l0_tuple_count"] = len(combined_tuples)

        return result

    # -------------------------------------------------------------------
    # P-03: Streaming support
    # -------------------------------------------------------------------

    async def ask_stream(
        self,
        question: str,
        context: list[str] | None = None,
        search_internet: bool = False,
        source: str = "user_input",
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> AsyncGenerator[PipelineEvent, None]:
        """Async generator that yields PipelineEvents after each layer.

        This is the streaming version of ask(). It yields partial events
        as each layer completes, allowing callers to show progress,
        partial results, or cancel long-running operations.

        Args:
            question: The question or trigger text.
            context: Optional context atoms to guide the query.
            search_internet: Whether to search the internet.
            source: Source type for trust scoring.
            cancel_callback: Optional callable that returns True to
                cancel the pipeline execution.

        Yields:
            PipelineEvent after each layer completes.
        """
        def _check_cancel():
            if cancel_callback and cancel_callback():
                raise asyncio.CancelledError("Pipeline cancelled by callback")

        # ---- Layer 0: Context Layer ----
        _check_cancel()
        try:
            layer0_output = await asyncio.to_thread(
                self._run_context_layer, question, source, search_internet,
            )
            yield PipelineEvent(
                layer="context",
                status="complete",
                partial_result=layer0_output.to_dict(),
            )
        except Exception as exc:
            yield PipelineEvent(
                layer="context",
                status="error",
                error=str(exc),
            )
            layer0_output = Layer0Output(text=question, source=source)

        # ---- Layer 1: Situation Layer ----
        _check_cancel()
        try:
            layer1_output = await asyncio.to_thread(
                self._run_situation_layer, question, source, layer0_output,
            )
            yield PipelineEvent(
                layer="situation",
                status="complete",
                partial_result=layer1_output.to_dict(),
            )
        except Exception as exc:
            yield PipelineEvent(
                layer="situation",
                status="error",
                error=str(exc),
            )
            layer1_output = Layer1Output()

        # ---- Layer 2: Predictive Engine ----
        _check_cancel()
        context_atoms = context or [
            s.get("label", s.get("concept", ""))
            for s in layer1_output.active_senses[:5]
            if s.get("label") or s.get("concept")
        ] or layer0_output.context_atoms

        prediction = None
        try:
            prediction = await asyncio.to_thread(
                self.predictive.predict, question, context_atoms,
            )
            yield PipelineEvent(
                layer="predictive",
                status="complete",
                partial_result={
                    "concept": prediction.concept,
                    "confidence": prediction.confidence,
                } if prediction else {},
            )
        except Exception as exc:
            yield PipelineEvent(
                layer="predictive",
                status="error",
                error=str(exc),
            )

        # ---- Layer 3: Pattern Output ----
        _check_cancel()
        pattern_result = None
        try:
            pattern_result = await asyncio.to_thread(
                self.pattern.process, question, context_atoms,
            )
            yield PipelineEvent(
                layer="pattern",
                status="complete",
                partial_result={
                    "confidence": pattern_result.confidence,
                    "pattern": pattern_result.pattern[:100] if pattern_result.pattern else "",
                    "steps": len(pattern_result.steps),
                } if pattern_result else {},
            )
        except Exception as exc:
            yield PipelineEvent(
                layer="pattern",
                status="error",
                error=str(exc),
            )

        # ---- Layer 4: Appraise Self-Check ----
        _check_cancel()
        if pattern_result and pattern_result.steps:
            answer = generate_narrative(
                trigger=question,
                reasoning_chain=[s.to_dict() for s in pattern_result.steps],
                pattern=pattern_result.pattern,
                evidence_nodes=list(dict.fromkeys(
                    n for s in pattern_result.steps for n in s.evidence_nodes
                )),
                confidence=pattern_result.confidence,
                anomalies=pattern_result.anomalies,
                language=self._language,
                use_llm=self._use_llm,
            )
        else:
            answer = await asyncio.to_thread(
                self._generate_fallback_answer, question,
                layer1_output.relevant_context, layer1_output.active_senses,
            )

        try:
            appraise_result = await asyncio.to_thread(
                self._bridge.appraise, answer,
            )
            yield PipelineEvent(
                layer="appraise",
                status="complete",
                partial_result={
                    "verdict": appraise_result.get("verdict", "neutral"),
                    "agree_pct": appraise_result.get("agree_pct", 0.0),
                    "disagree_pct": appraise_result.get("disagree_pct", 0.0),
                },
            )
        except Exception as exc:
            yield PipelineEvent(
                layer="appraise",
                status="error",
                error=str(exc),
            )

        # ---- Final ----
        yield PipelineEvent(
            layer="final",
            status="complete",
            partial_result={"answer": answer[:200]},
        )

    # -------------------------------------------------------------------
    # P-05: Maintenance
    # -------------------------------------------------------------------

    def maintenance(self) -> dict:
        """Run graph maintenance: consolidate() + run_reflection().

        Consolidation merges duplicate senses, prunes dead nodes,
        and compacts the graph. Reflection reviews the graph's state
        and proposes REVISE and RETIRE actions.

        Returns:
            A dict with consolidation and reflection results.
        """
        start_time = time.time()
        result: dict = {
            "consolidation": None,
            "reflection": None,
            "duration_s": 0.0,
        }

        try:
            consolidation = self._bridge.consolidate()
            result["consolidation"] = consolidation
            logger.info(
                "Maintenance consolidation: pruned_nodes=%s, merged_senses=%s",
                consolidation.get("pruned_nodes", 0),
                consolidation.get("merged_senses", 0),
            )
        except Exception as exc:
            result["consolidation"] = {"success": False, "error": str(exc)}
            logger.warning("Consolidation failed: %s", exc)

        try:
            reflection = self._bridge.run_reflection()
            result["reflection"] = reflection
            logger.info(
                "Maintenance reflection: confirm=%s, review=%s, retire=%s",
                reflection.get("confirm", 0),
                reflection.get("review", 0),
                reflection.get("retire", 0),
            )
        except Exception as exc:
            result["reflection"] = {"success": False, "error": str(exc)}
            logger.warning("Reflection failed: %s", exc)

        result["duration_s"] = round(time.time() - start_time, 3)
        self._last_maintenance_time = time.time()
        self._maintenance_log.append(result)
        return result

    def force_maintenance(self) -> dict:
        """Force maintenance regardless of ingest count.

        Returns:
            Same as maintenance().
        """
        logger.info("Force maintenance triggered")
        return self.maintenance()

    def get_maintenance_log(self) -> list[dict]:
        """Return the maintenance history."""
        return list(self._maintenance_log)

    # -------------------------------------------------------------------
    # Ingest + Scope
    # -------------------------------------------------------------------

    def ingest(self, text: str, source: str = "user_input") -> dict:
        """Ingest text into the knowledge graph without asking a question.

        Useful for pre-loading knowledge before asking questions.

        Args:
            text: Text to ingest.
            source: Source type for trust scoring.

        Returns:
            Stats from the ingestion process.
        """
        # P-04: Pass source_provenance through pipeline
        context_stats = self.context.ingest_text(text, source=source)
        situation_stats = self.situation.add_message("system", text)

        # Track ingest count for auto-maintenance
        self._ingest_count += 1
        if self._maintenance_interval > 0 and self._ingest_count % self._maintenance_interval == 0:
            try:
                self.maintenance()
            except Exception:
                pass

        return {
            "context": context_stats,
            "situation": situation_stats,
        }

    def set_scope(self, allowed_sources: list[str]) -> None:
        """Set scope filter — only use these sources for answers.

        Analogi: Jin Soun memilih hanya mengakses laporan Hefei,
        catatan masuk-keluar, dan laporan misi — bukan seluruh perpustakaan.

        Args:
            allowed_sources: List of allowed source types.
        """
        self.context.set_scope(allowed_sources)

    def clear_scope(self) -> None:
        """Clear scope filter — accept all sources."""
        self.context.clear_scope()

    def get_status(self) -> dict:
        """Get current pipeline status."""
        # Determine which modalities are available via Layer 0
        layer0_modalities = ["text"]  # text is always available
        if self._image_abstractor is not None:
            layer0_modalities.append("image")
        if self._audio_abstractor is not None:
            layer0_modalities.append("audio")
        if self._video_abstractor is not None:
            layer0_modalities.append("video")

        return {
            "version": "8.6.0",
            "rsvs_available": self._bridge.is_available,
            "is_rust_core": self._bridge.is_rust_core,
            "scope": self.context.get_scope(),
            "conversation_turns": len(self._conversation_history),
            "active_senses": len(self.situation.get_active_senses()),
            "active_predictions": len(self.predictive.get_predictions()),
            "use_llm": self._use_llm,
            "language": self._language,
            "ingest_count": self._ingest_count,
            "maintenance_interval": self._maintenance_interval,
            "last_maintenance": self._last_maintenance_time,
            "layer3_available": True,  # Layer 3 is always available (works in fallback too)
            "layer0_available": True,
            "layer0_modalities": layer0_modalities,
            "diffusion_llm": self.diffusion_llm.get_status(),
            "temporal_tracker": self.temporal.get_stats(),
        }

    # -------------------------------------------------------------------
    # Layer 3: Direct Access Methods
    # -------------------------------------------------------------------

    def ask_deductive(
        self,
        question: str,
        context: list[str] | None = None,
    ) -> DeductiveChain:
        """Ask a question using the full deductive reasoning chain.

        Runs the complete pipeline (context → situation → pattern →
        deductive reasoning) and returns a DeductiveChain with full
        traceability — every claim maps to evidence nodes in the RSVS
        graph.

        Analogi: Jin Soun tidak hanya menyimpulkan, tapi menuliskan
        setiap langkah penalarannya dari bukti pertama hingga
        kesimpulan akhir — rantai yang bisa diaudit siapa saja.

        Args:
            question: The question or trigger text.
            context: Optional context atoms to guide the query.

        Returns:
            A DeductiveChain with auditable steps and aggregate confidence.

        Raises:
            ReasoningError: If the deductive chain cannot be built.
        """
        # Run context + situation layers first
        layer0_output = self._run_context_layer(
            question, "user_input", False,
        )
        layer1_output = self._run_situation_layer(
            question, "user_input", layer0_output,
        )

        # Determine context atoms
        context_atoms = context or [
            s.get("label", s.get("concept", ""))
            for s in layer1_output.active_senses[:5]
            if s.get("label") or s.get("concept")
        ] or layer0_output.context_atoms

        # Run pattern completion
        pattern_result = self.pattern.process(question, context_atoms)

        # Build the deductive chain
        if pattern_result is None:
            raise ReasoningError(
                "Pattern completion returned no result — cannot build "
                "deductive chain without a pattern result.",
                layer="deductive",
            )

        deductive_chain = self.reasoning.build_chain(pattern_result)

        logger.info(
            "ask_deductive(): question='%s', steps=%d, confidence=%.3f",
            question[:60], len(deductive_chain.steps),
            deductive_chain.aggregate_confidence,
        )

        return deductive_chain

    def check_policy(
        self,
        text: str,
        domain: str = "",
    ) -> list[dict]:
        """Check text against policy rules.

        Runs the DeductivePolicyEngine's compliance checking, which
        combines deterministic rule evaluation with RSVS PolicyMeta
        (governance_score, status_flip_count, seen_fingerprints)
        for trust-weighted, auditable compliance checking.

        Analogi: Jin Soun mengecek buku hukum DAN catatan
        pengawasan — bukan hanya "apa aturannya?", tapi juga
        "seberapa bisa dipercaya entitas yang dilaporkan?"

        Args:
            text: The text or entity label to check for compliance.
            domain: Optional domain hint (e.g., "tax", "regulation").

        Returns:
            A list of dicts, each with compliance results and
            adjusted confidence from RSVS PolicyMeta.
        """
        results: list[dict] = []

        # If a domain hint is provided, try domain-specific checks
        if domain:
            result = self.deductive_policy.check_with_rsvs_policy(
                entity_label=f"{domain}:{text}",
            )
            results.append(result)
        else:
            # Standard compliance check with RSVS policy metadata
            result = self.deductive_policy.check_with_rsvs_policy(
                entity_label=text,
            )
            results.append(result)

        # Also run a basic check_compliance for any embedded text
        try:
            compliance = self.deductive_policy.check_compliance(text)
            if compliance.get("violations") or compliance.get("warnings"):
                results.append({
                    "type": "text_compliance",
                    "compliance": compliance,
                })
        except Exception as exc:
            logger.debug("check_compliance() failed for text: %s", exc)

        return results

    def check_policy_with_trace(
        self,
        entity: str,
        context: dict | None = None,
        domain: str = "",
    ) -> dict:
        """Check policy compliance with a full audit trail.

        This is the key method for the tax/classification use case.
        It combines:
        1. Standard compliance checking (PolicyEngine)
        2. RSVS-enhanced trust weighting (DeductivePolicyEngine)
        3. Deductive reasoning chain (ReasoningEngine) for traceability

        The result includes not just "compliant or not" but also:
        - WHY each violation was triggered (with evidence node IDs)
        - The deductive chain showing how the conclusion was reached
        - Gaps and contradictions detected by the RSVS pipeline

        Analogi: Jin Soun tidak hanya bilang "ini melanggar aturan",
        tapi menunjukkan buku hukum yang mana, halaman berapa, dan
        catatan pengawasan mana yang mendukung kesimpulan itu.

        Args:
            entity: The entity label to check (e.g., "PT_Test_Company").
            context: Optional context dict with values for rule evaluation
                (e.g., {"value": 600_000_000, "rate": 0.15}).
            domain: Optional domain hint (e.g., "tax").

        Returns:
            A dict with:
            - compliant: bool
            - violations: list of violation dicts with messages
            - confidence: float (trust-weighted)
            - trace: list of deductive steps (fully auditable)
            - gaps: list of detected knowledge gaps
            - contradictions: list of detected contradictions
        """
        # 1. Run standard compliance check
        compliance_result = self.deductive_policy.check_compliance(
            entity, context=context or {},
        )

        # 2. Run RSVS-enhanced check for trust weighting
        rsvs_result = self.deductive_policy.check_with_rsvs_policy(entity)

        # 3. Build deductive trace for each violation
        trace_steps = []
        for violation in compliance_result.get("violations", []):
            v_dict = violation.to_dict() if hasattr(violation, "to_dict") else violation
            step = {
                "step_type": "violation",
                "rule_id": v_dict.get("rule_id", "unknown"),
                "description": v_dict.get("rule_description", ""),
                "severity": v_dict.get("severity", "warning"),
                "message": v_dict.get("message", ""),
                "evidence": [],
            }
            trace_steps.append(step)

        # 4. Add compliance-passing rules as evidence
        for rule in self.deductive_policy.get_rules():
            if rule.enabled:
                try:
                    value = (context or {}).get("value", 0)
                    passed = rule.evaluate(value)
                    if passed:
                        trace_steps.append({
                            "step_type": "evidence",
                            "rule_id": rule.rule_id,
                            "description": rule.description,
                            "condition": rule.condition,
                            "passed": True,
                        })
                except Exception:
                    pass

        # 5. Detect gaps and contradictions from the RSVS pipeline
        gaps = []
        contradictions = []
        try:
            raw_gaps = self._bridge.detect_gaps()
            for g in raw_gaps:
                gap_dict = g if isinstance(g, dict) else {}
                gaps.append({
                    "gap_id": gap_dict.get("gap_id", "unknown"),
                    "gap_type": gap_dict.get("gap_type", "unknown"),
                    "description": gap_dict.get("description", ""),
                })
        except Exception:
            pass

        # Check for contradicted compositions
        try:
            for comp in self._bridge.compositions():
                if comp.get("epistemic", "").lower() == "contradicted":
                    contradictions.append({
                        "composition_id": comp.get("id", "unknown"),
                        "type": comp.get("composition_type", "unknown"),
                    })
        except Exception:
            pass

        # 6. Compute final confidence
        adjusted_confidence = rsvs_result.get("adjusted_confidence", 0.5)
        if compliance_result.get("compliant", True):
            confidence = adjusted_confidence
        else:
            confidence = adjusted_confidence * 0.5  # Violations reduce confidence

        return {
            "compliant": compliance_result.get("compliant", True),
            "violations": [
                v.to_dict() if hasattr(v, "to_dict") else v
                for v in compliance_result.get("violations", [])
            ],
            "confidence": round(confidence, 3),
            "trust_weight": rsvs_result.get("trust_weight", 0.5),
            "trace": trace_steps,
            "gaps": gaps,
            "contradictions": contradictions,
        }

    def analyze_code(
        self,
        code: str,
        language: str = "python",
    ) -> dict:
        """Analyze code using the deductive coder layer.

        Uses the DeductiveCoderLayer's analyze_with_rsvs() method,
        which creates a full RSVS-represented code graph using
        compositional semantics for deeper structural analysis.

        Analogi: Jin Soun tidak hanya membaca satu manual teknik —
        dia menghubungkan teknik dari berbagai manual, membandingkan
        strukturnya, dan menemukan pola lintas-sumber.

        Args:
            code: Source code string to analyze.
            language: Programming language (default: "python").

        Returns:
            A dict with CodeAnalysisResult data including elements,
            similar code pairs, anomalies, patterns, and suggestions.
        """
        result = self.deductive_coder.analyze_with_rsvs(
            code=code,
            language=language,
        )
        return result.to_dict()

    # -------------------------------------------------------------------
    # P-01: Layer runner methods (produce structured outputs)
    # -------------------------------------------------------------------

    def _run_layer0(self, text: str, source: str) -> Layer0PerceptualObservation:
        """Run Layer 0 perceptual abstraction on input.

        Converts raw input into structured PerceptualTuples that capture
        categorical, differential, functional, spatial, temporal, and causal
        relations. This is the "perception" step before cognition.

        Analogi: Jin Soun mendengar langkah kaki → bukan hanya "suara",
        tapi "langkah kaki berat, dari arah timur, sekitar 3 orang,
        berlari". Layer 0 = kemampuan memecah input menjadi relasi.

        Args:
            text: The raw text input to abstract.
            source: Source provenance for the input.

        Returns:
            A Layer0PerceptualObservation containing PerceptualTuples,
            or a minimal fallback observation if abstraction fails.
        """
        try:
            obs = self._text_abstractor.abstract(text, context={"source": source})
            # Ingest the structured observation into RSVS
            if self._bridge.is_available and obs.tuples:
                ingest_data = observation_to_ingest_data(obs)
                if ingest_data:
                    self._bridge.ingest(ingest_data)
            return obs
        except Exception as exc:
            logger.debug("Layer 0 abstraction failed, using raw text: %s", exc)
            # Fallback: return a minimal observation with just the raw text
            return Layer0PerceptualObservation(
                modality=ModalityType.TEXT,
                raw_input_ref=text[:200],
                tuples=[],
                context={"source": source},
            )

    def _run_context_layer(
        self,
        question: str,
        source: str,
        search_internet: bool,
    ) -> Layer0Output:
        """Run the Context Layer and produce a Layer0Output.

        P-01: Instead of just passing strings, we now produce
        a PerceptualObservation with structural information.
        P-04: Source provenance is passed through to RSVS.

        Layer 0 integration: Before the existing context layer logic,
        we run the perceptual front-end to extract structured tuples
        from the input text. These tuples enrich the context_atoms
        that guide downstream reasoning.
        """
        # Layer 0: Perceptual abstraction — extract structured tuples
        # before the context layer processes the raw text.
        try:
            layer0_obs = self._run_layer0(question, source)
        except Exception as exc:
            logger.debug("Layer 0 integration in context layer failed: %s", exc)
            layer0_obs = None

        trust = self.context.trust_score(source)

        # P-04: Ingest with source provenance
        ingest_stats = self.context.ingest_text(question, source=source)

        # Search internet if requested or auto_search triggers
        search_results = []
        if search_internet or (self._auto_search and self._should_search(question)):
            search_result = self.context.search_and_ingest(question)
            search_results = search_result.get("results", [])

        # Extract context atoms from active senses after ingestion
        context_atoms = []
        if ingest_stats.get("success"):
            # Get updated context atoms from the situation layer
            active_senses = self.situation.get_active_senses()
            context_atoms = [
                s.get("label", "")
                for s in active_senses[:5]
                if s.get("label")
            ]

        # Enrich context_atoms with perceptual tuples from Layer 0
        if layer0_obs is not None:
            for tuple_ in layer0_obs.tuples:
                if tuple_.subject and tuple_.subject not in context_atoms:
                    context_atoms.append(tuple_.subject)
                if tuple_.predicate and tuple_.predicate not in context_atoms:
                    context_atoms.append(tuple_.predicate)

        return Layer0Output(
            text=question,
            source=source,
            trust=trust,
            search_results=search_results,
            ingest_stats=ingest_stats,
            context_atoms=context_atoms,
        )

    def _run_situation_layer(
        self,
        question: str,
        source: str,
        layer0_output: Layer0Output,
    ) -> Layer1Output:
        """Run the Situation Layer and produce a Layer1Output.

        P-01: Instead of just passing strings, we now produce
        a StructuralDelta with graph change information.
        P-04: Source provenance is passed through to RSVS.
        """
        # Record the question in conversation history
        # P-04: The situation layer prefixes with role for context
        msg_stats = self.situation.add_message("user", question)
        self._conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": time.time(),
        })

        # Find relevant context from chat history / graph
        relevant = self.situation.get_relevant_context(question, top_k=10)

        # Get active senses for context
        active_senses = self.situation.get_active_senses()

        # Extract structural changes from ingest stats
        new_nodes: list[str] = []
        sense_changes: list[dict] = []
        confidence_updates: dict[str, float] = {}

        if msg_stats.get("success") and msg_stats.get("stats"):
            stats = msg_stats["stats"]
            if isinstance(stats, dict):
                # Extract new atom information
                active_atoms = msg_stats.get("active_atoms", [])
                new_nodes.extend(active_atoms)
                # Track sense changes
                for key in ("sense_assigned", "sense_created", "sense_updated"):
                    if stats.get(key, 0) > 0:
                        sense_changes.append({
                            "type": key,
                            "count": stats[key],
                        })

        # Get confidence updates from the confidence map
        try:
            cmap = self._bridge.confidence_map()
            for label, conf in cmap.items():
                if label in new_nodes or any(
                    r.get("label") == label for r in relevant
                ):
                    confidence_updates[label] = conf
        except Exception:
            pass

        return Layer1Output(
            new_nodes=new_nodes,
            sense_changes=sense_changes,
            confidence_updates=confidence_updates,
            ingest_stats=msg_stats.get("stats"),
            relevant_context=relevant,
            active_senses=active_senses,
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _detect_query_mode(self, question: str) -> str:
        """Detect which Layer 3 sub-system should handle the query.

        Routes the question to the appropriate deductive reasoning mode
        based on keyword detection.  Returns "general" for the default
        pattern completion flow — meaning Layer 3 is not needed.

        Analogi: Jin Soun mendengar pertanyaan dan langsung tahu
        apakah ini soal regulasi, kode, analisis mendalam, atau
        pertanyaan biasa — sebelum memulai penalaran.

        Args:
            question: The user's question / trigger text.

        Returns:
            One of: "policy", "coder", "reasoning", "general".
        """
        q = question.lower()

        # Policy: regulation / compliance / tax questions
        policy_keywords = [
            "regulasi", "aturan", "compliance", "kebijakan",
            "tax", "pajak", "peraturan", "undang-undang", "uu",
            "violat", "l商务", "audit",
        ]
        if any(kw in q for kw in policy_keywords):
            return "policy"

        # Coder: code-related questions
        coder_keywords = [
            "code", "fungsi", "function", "class", "bug", "error",
            "debug", "refactor", "implementasi", "kode", "script",
            "method", "module", "api",
        ]
        if any(kw in q for kw in coder_keywords):
            return "coder"

        # Reasoning: analytical / deductive questions
        reasoning_keywords = [
            "mengapa", "kenapa", "why", "how", "bukti", "evidence",
            "analisis", "analysis", "deduce", "deduksi", "sebab",
            "akibat", "cause", "effect", "explain", "jelaskan",
            "prove", "buktikan",
        ]
        if any(kw in q for kw in reasoning_keywords):
            return "reasoning"

        return "general"

    def _should_search(self, question: str) -> bool:
        """Determine if we should search the internet for this question.

        Heuristic: search if the question contains question words
        or if we don't have enough knowledge in the graph.
        """
        question_indicators = [
            "apa", "apa itu", "siapa", "kapan", "dimana", "mengapa",
            "bagaimana", "berapa", "kenapa",
            "what", "who", "when", "where", "why", "how", "how much",
            "?",
        ]
        question_lower = question.lower()
        return any(qi in question_lower for qi in question_indicators)

    def _check_narrative_reliability(
        self,
        answer: str,
        pattern_result: PatternResult | None,
    ) -> tuple[bool, str]:
        """Check if the LLM-generated narrative is reliable.

        Bounded execution: detect when the LLM "body" produces unreliable output.
        Analogi: Jin Soun tahu kapan tubuhnya tidak bisa mengeksekusi teknik
        yang dia pikirkan — dan memilih strategi alternatif.

        Returns:
            Tuple of (is_reliable: bool, reason: str)
        """
        if not answer or len(answer.strip()) < 10:
            return False, "Answer too short or empty"

        # Check 1: Does the answer reference the evidence nodes?
        if pattern_result and pattern_result.steps:
            evidence_labels = set()
            for step in pattern_result.steps:
                for node in step.evidence_nodes:
                    evidence_labels.add(node.lower())

            # Check if any evidence label appears in the answer
            answer_lower = answer.lower()
            evidence_mentioned = any(label in answer_lower for label in evidence_labels if len(label) > 3)

            if not evidence_mentioned and evidence_labels:
                return False, "Answer doesn't reference any evidence from the graph"

        # Check 2: Does the appraise self-check flag issues?
        # (This is already handled in the main flow, but we double-check)

        # Check 3: Is the answer internally consistent?
        # Simple heuristic: check for contradiction markers
        contradiction_markers = ["namun sebenarnya", "tetapi sebaliknya", "contradict", "namun bertentangan"]
        answer_lower = answer.lower()
        has_contradiction = any(marker in answer_lower for marker in contradiction_markers)
        if has_contradiction:
            return True, "Answer contains contradiction markers — flagged for review"

        return True, "Narrative appears reliable"

    def _generate_fallback_answer(
        self,
        question: str,
        relevant: list[dict],
        active_senses: list[dict],
    ) -> str:
        """Generate a simple fallback answer when pattern completion doesn't work."""
        if relevant:
            concepts = [
                r.get("label", r.get("concept", "unknown"))
                for r in relevant[:3]
            ]
            return (
                f"Berdasarkan pengetahuan yang tersedia, konsep yang relevan "
                f"dengan pertanyaan Anda adalah: {', '.join(concepts)}. "
                f"Untuk analisis yang lebih mendalam, diperlukan lebih banyak "
                f"data dalam knowledge graph."
            )
        return (
            "Saya belum memiliki cukup pengetahuan untuk menjawab pertanyaan "
            "ini secara struktural. Coba tambahkan konteks atau nyalakan "
            "internet search untuk memperkaya knowledge graph."
        )

    def _estimate_confidence(
        self,
        relevant: list[dict],
        active_senses: list[dict],
    ) -> float:
        """Estimate confidence based on available evidence."""
        if not relevant and not active_senses:
            return 0.1
        score = min(1.0, len(relevant) * 0.1 + len(active_senses) * 0.05)
        return round(score, 3)

    def _is_rsvs_available(self) -> bool:
        """Check if RSVS (Rust core or fallback) is available."""
        return self._bridge.is_available


# Backward compat aliases
GeniusPipeline = AamPipeline
GeniusResponse = AamResponse
