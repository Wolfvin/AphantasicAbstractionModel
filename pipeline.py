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

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional

from layer2.bridge import AbstractionBridge, RsvsBridge, get_bridge, is_rust_core_available
from layer2.llm import generate_narrative
from layer2.context import ContextLayer, SOURCE_TRUST
from layer2.situation import SituationLayer
from layer2.predictive import PredictiveEngine, Prediction, Anomaly, BeliefUpdate
from layer2.pattern import PatternOutput, ReasoningStep, PatternResult

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
        bridge: Optional[RsvsBridge] = None,
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
            self._bridge = RsvsBridge(rsvs_instance=rsvs_instance)
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
        self.pattern = PatternOutput(bridge=self._bridge)

        # Internal state
        self._conversation_history: list[dict] = []

        # P-05: Maintenance tracking
        self._ingest_count: int = 0
        self._last_maintenance_time: float = 0.0
        self._maintenance_log: list[dict] = []

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

        # ---- Step 5: Belief Update ----
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
                "internet_search": search_internet,
                "context_atoms": context_atoms[:5],
                "conversation_turn": len(self._conversation_history),
                "active_senses_count": len(layer1_output.active_senses),
                "rsvs_available": self._is_rsvs_available(),
                "use_llm": self._use_llm,
                "ingest_count": self._ingest_count,
            },
            appraise_warning=appraise_warning,
            errors=non_fatal_errors,
        )

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
        return {
            "version": "8.5.0",
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
        }

    # -------------------------------------------------------------------
    # P-01: Layer runner methods (produce structured outputs)
    # -------------------------------------------------------------------

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
        """
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
