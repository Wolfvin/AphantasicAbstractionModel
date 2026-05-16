"""
GeniusPipeline — Wire semua layer jadi satu sistem

"The Genius Who Remembers Everything"

Analogi: Ini adalah Jin Soun secara keseluruhan —
mulai dari mendengar pertanyaan, mengingat semua yang relevan,
memahami relasi, mendeteksi anomali, sampai mengeluarkan
kesimpulan yang bisa diaudit.

Flow:
    User Input
      -> Context Layer (internet search jika perlu)
      -> Scope Control (hierarchical scope management)
      -> Semantic Chat Index (index conversation as graph of meaning)
      -> Situation Layer (ingest chat, cari konteks relevan)
      -> RSVS Core (spreading activation, structural analysis)
      -> Prediction Loop (predict/observe/update lifecycle)
      -> Pattern Output (pattern completion + narrative)
      -> Final Output (traceable reasoning chain + confidence)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import V12PipelineBridge, get_bridge, is_rust_core_available
from .llm import generate_narrative
from .context import ContextLayer
from .situation import SituationLayer
from .predictive import PredictiveEngine, Prediction, Anomaly, BeliefUpdate
from .prediction_loop import PredictionLoop, CycleResult, CycleTracker
from .scope_control import ScopeControl, ScopeConfig
from .chat_index import SemanticChatIndex, ChatNode
from .pattern import PatternOutput, ReasoningStep, PatternResult
from .temporal import TemporalTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GeniusResponse:
    """Response dari GeniusPipeline — lengkap dengan traceability."""

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
                }
                for s in self.reasoning_chain
            ],
            "evidence_chain": self.evidence_chain,
            "anomalies": self.anomalies,
            "predictions": self.predictions,
            "belief_updates": self.belief_updates,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert ke JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class GeniusPipeline:
    """Main pipeline — wires semua layer jadi satu sistem.

    Usage:
        pipeline = GeniusPipeline()

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

        # With hierarchical scope control
        scope_id = pipeline.define_scope(ScopeConfig(
            domain="finance",
            subdomains=["tax"],
            topics=["pajak penghasilan"],
            min_confidence=0.5,
            boundary_mode="soft",
        ))
        pipeline.activate_scope(scope_id)
        result = pipeline.ask("Berapa tarif pajak penghasilan?")
        pipeline.deactivate_scope()

    v12 integration:
        The GeniusPipeline can optionally use V12PipelineBridge for
        executive-controlled ingestion when the v12 PipelineEngine is
        available (Rust core built with ``--features v12,python``).

        The v12 pipeline provides a DAG-based execution model with three
        cognitive modes (Reactive, Analytical, Reflective) and built-in
        gap detection.  When available, the pipeline can route ingestion
        through the v12 path for richer structural analysis, while
        falling back to the standard AbstractionBridge path otherwise.

        To enable v12 ingestion:
            pipeline = GeniusPipeline()

        The v12 bridge is also accessible directly:
            pipeline.v12  # V12PipelineBridge instance

    Analogi novel:
        Ini adalah Jin Soun secara keseluruhan.
        - ContextLayer = Simhyeon Pavilion + kemampuan membatasi sumber
        - ScopeControl = batasan hierarkis (domain->subdomain->topic)
        - SemanticChatIndex = percakapan sebagai graph of meaning
        - SituationLayer = ingatan percakapan + active senses
        - RSVS Core = structural memory (30 tahun ingatan)
        - PredictionLoop = predict/observe/update lifecycle (Friston)
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

    ):
        """Initialize the pipeline with all layers.

        Args:
            rsvs_instance: Optional RSVS instance. If None, try to create one.
                Deprecated: prefer passing `bridge` instead.
            bridge: Optional shared RsvsBridge instance. If provided, all
                layers will share this bridge (recommended). If None and
                rsvs_instance is also None, a new bridge is created via
                get_bridge().
            eta: Learning rate for predictive coding (default: 0.1).
            anomaly_threshold: Threshold for anomaly detection (default: 0.3).
            auto_search: Automatically search internet when confidence is low.
            use_llm: Whether to use LLM for narrative generation (default: True).
            language: Output language for narratives ("id" or "en").

        """
        self._eta = eta
        self._anomaly_threshold = anomaly_threshold
        self._auto_search = auto_search
        self._use_llm = use_llm
        self._language = language
        self._use_v12 = True  # v12 is now the ONLY architecture

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
        self.prediction_loop = PredictionLoop(
            bridge=self._bridge,
        )
        self.scope = ScopeControl(bridge=self._bridge)
        self.chat_index = SemanticChatIndex(bridge=self._bridge)
        self.pattern = PatternOutput(bridge=self._bridge)

        # Temporal tracking layer (G1-1: temporal metadata for nodes)
        self.temporal = TemporalTracker()

        # v12 pipeline bridge — same singleton as self._bridge.
        # Exposed as self.v12 for backward compatibility with code that
        # accesses pipeline.v12 directly.
        self.v12 = self._bridge

        # Internal state
        self._conversation_history: list[dict] = []
        self._current_conversation_id: str | None = None

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def ask(
        self,
        question: str,
        context: list[str] | None = None,
        search_internet: bool = False,
        source: str = "user_input",
    ) -> GeniusResponse:
        """Ask a question and get a traceable, evidence-backed response.

        This is the main entry point. It runs through all layers:

        1. Context Layer: Search internet if needed, apply scope filter
        2. Semantic Chat Index: Index user message, retrieve by meaning
        3. Situation Layer: Ingest the question, find relevant context
        4. RSVS Core + Prediction Loop: predict/observe/update cycle
        5. Pattern Output: Pattern completion, generate narrative

        Args:
            question: The question or trigger text.
            context: Optional context atoms to guide the query.
            search_internet: Whether to search the internet for additional info.
            source: Source type for trust scoring (default: "user_input").

        Returns:
            GeniusResponse with answer, reasoning chain, evidence, and confidence.
        """
        start_time = time.time()
        all_evidence: list[dict] = []
        all_anomalies: list[dict] = []
        all_predictions: list[dict] = []
        all_belief_updates: list[dict] = []

        # ---- Step 1: Context Layer ----
        # Ingest the question into context
        self.context.ingest_text(question, source=source)

        # G1-1: Record temporal observation for the question
        self.temporal.record_observation(question, source=source)

        # Search internet if requested or auto_search triggers
        search_results = {}
        if search_internet or (self._auto_search and self._should_search(question)):
            search_results = self.context.search_and_ingest(question)
            if search_results.get("results"):
                all_evidence.append({
                    "type": "internet_search",
                    "query": question,
                    "results_count": len(search_results.get("results", [])),
                    "trust": self.context.trust_score("web_search"),
                })

        # ---- Step 2: Semantic Chat Index + Situation Layer ----
        # Index the user message in the semantic chat index
        chat_node = self.chat_index.index_message(
            "user", question, conversation_id=self._current_conversation_id
        )
        self._current_conversation_id = chat_node.conversation_id

        # Record the question in conversation history (legacy)
        msg_stats = self.situation.add_message("user", question)
        self._conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": time.time(),
        })

        # Find relevant context: use SemanticChatIndex for meaning-based retrieval
        chat_relevant = self.chat_index.retrieve_by_meaning(question, top_k=5)
        chat_relevant_labels = [
            cn.semantic_atoms for cn in chat_relevant if cn.semantic_atoms
        ]

        # Use SituationLayer for RSVS-based retrieval
        relevant = self.situation.get_relevant_context(question, top_k=10)
        if relevant:
            all_evidence.append({
                "type": "situation_recall",
                "relevant_concepts": [
                    r.get("label", r.get("concept", ""))
                    for r in relevant[:5]
                ],
            })

        # Get active senses for context
        active_senses = self.situation.get_active_senses()

        # G1-1: Record temporal observations for active senses
        for sense in active_senses:
            label = sense.get("label", "")
            if label:
                self.temporal.record_observation(label, source="situation")

        # ---- Step 3: RSVS Core + Prediction Loop ----
        # Make predictions based on context
        context_atoms = context or [
            s.get("label", s.get("concept", ""))
            for s in active_senses[:5]
            if s.get("label") or s.get("concept")
        ]
        # Enrich context_atoms with semantic chat index results
        for atoms_list in chat_relevant_labels[:3]:
            for atom in atoms_list[:3]:
                if atom not in context_atoms:
                    context_atoms.append(atom)

        # Run full prediction cycle using PredictionLoop (predict/observe/update)
        cycle_result = self.prediction_loop.run_cycle(question, question, context_atoms)
        if cycle_result.prediction:
            all_predictions.append({
                "concept": cycle_result.prediction.concept,
                "expected": cycle_result.prediction.expected_compositions,
                "confidence": cycle_result.prediction.confidence,
                "cycle_id": cycle_result.cycle_id,
                "state": cycle_result.state,
            })

        # Collect belief updates from cycle
        if cycle_result.belief_update:
            all_belief_updates.append({
                "concept": cycle_result.belief_update.concept,
                "old_confidence": round(cycle_result.belief_update.old_confidence, 3),
                "new_confidence": round(cycle_result.belief_update.new_confidence, 3),
                "direction": cycle_result.belief_update.direction,
                "reason": cycle_result.belief_update.reason,
            })

        # Collect anomaly from cycle
        if cycle_result.anomaly:
            all_anomalies.append({
                "concept": cycle_result.anomaly.concept,
                "expected": cycle_result.anomaly.expected,
                "observed": cycle_result.anomaly.observed,
                "delta": cycle_result.anomaly.delta,
                "description": cycle_result.anomaly.description,
            })

        # Collect re-prediction if auto-triggered
        if cycle_result.re_prediction:
            all_predictions.append({
                "concept": cycle_result.re_prediction.concept,
                "expected": cycle_result.re_prediction.expected_compositions,
                "confidence": cycle_result.re_prediction.confidence,
                "is_re_prediction": True,
                "parent_cycle_id": cycle_result.cycle_id,
            })

        # Also detect anomalies from the predictive engine (legacy, for completeness)
        anomalies = self.predictive.detect_anomalies()
        for anomaly in anomalies:
            # Avoid duplicating anomalies already captured by the cycle
            if not any(a.get("concept") == anomaly.concept for a in all_anomalies):
                all_anomalies.append({
                    "concept": anomaly.concept,
                    "expected": anomaly.expected,
                    "observed": anomaly.observed,
                    "delta": anomaly.delta,
                    "description": anomaly.description,
                })

        # ---- Step 4: Pattern Completion Output ----
        # Run the full pattern completion pipeline
        pattern_result = self.pattern.process(question, context_atoms)

        # Extract from pattern result
        if pattern_result:
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

        # ---- Step 5: Belief Update ----
        # Belief updates are now handled by PredictionLoop in Step 3.
        # Legacy observe_and_update kept for backward compatibility.

        # ---- Build Response ----
        # Determine the main answer
        if pattern_result and pattern_result.steps:
            # Use LLM narrative or structured fallback
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
            answer = self._generate_fallback_answer(question, relevant, active_senses)
            confidence = self._estimate_confidence(relevant, active_senses)
            reasoning_chain = []
            pattern_evidence = []

        all_evidence.extend(pattern_evidence)

        # Record assistant response in conversation and chat index
        self.chat_index.index_message(
            "assistant", answer, conversation_id=self._current_conversation_id
        )
        self.situation.add_message("assistant", answer)
        self._conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": time.time(),
        })

        latency = time.time() - start_time

        return GeniusResponse(
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
                "active_senses_count": len(active_senses),
                "rsvs_available": self._is_rsvs_available(),
                "use_llm": self._use_llm,
                "conversation_id": self._current_conversation_id,
                "prediction_cycle_id": cycle_result.cycle_id if cycle_result else None,
                "prediction_state": cycle_result.state if cycle_result else None,
                "scope_active": (self.scope.get_active_scope() is not None),
            },
        )

    def ingest(self, text: str, source: str = "user_input") -> dict:
        """Ingest text into the knowledge graph without asking a question.

        Useful for pre-loading knowledge before asking questions.

        When ``use_v12=True`` was passed at construction and the v12
        pipeline is available, ingestion is routed through
        V12PipelineBridge which runs the DAG-based pipeline
        (ExtractFrame, ReasonFrame, GovernBeliefs) and returns
        cognitive mode and gap detection info in addition to the
        standard layer stats.

        Args:
            text: Text to ingest.
            source: Source type for trust scoring.

        Returns:
            Stats from the ingestion process.  When v12 is active,
            includes a ``"v12"`` key with the V12PipelineBridge.ingest()
            result dict.
        """
        # Ingest through all layers
        context_stats = self.context.ingest_text(text, source=source)
        situation_stats = self.situation.add_message("system", text)

        result = {
            "context": context_stats,
            "situation": situation_stats,
        }

        # Optional v12 pipeline ingestion for executive-controlled path
        if self._use_v12 and self.v12.available:
            try:
                v12_stats = self.v12.ingest(text)
                result["v12"] = v12_stats
            except Exception as exc:
                logger.warning("v12 ingestion failed, continuing with standard path: %s", exc)

        return result

    def set_scope(self, allowed_sources: list[str]) -> None:
        """Set scope filter — only use these sources for answers.

        Analogi: Jin Soun memilih hanya mengakses laporan Hefei,
        catatan masuk-keluar, dan laporan misi — bukan seluruh perpustakaan.

        Args:
            allowed_sources: List of allowed source types.
        """
        self.context.set_scope(allowed_sources)

    def define_scope(self, config: ScopeConfig) -> str:
        """Define a hierarchical scope with domain/subdomain/topic filtering.

        Analogi: Jin Soun membatasi investigasi ke domain "kriminal"
        dengan subdomain "pencurian" dan topik "Snow Plum Pill".

        Args:
            config: ScopeConfig with domain, subdomains, topics, etc.

        Returns:
            scope_id for the defined scope.
        """
        return self.scope.define_scope(config)

    def activate_scope(self, scope_id: str) -> None:
        """Activate a previously defined scope."""
        self.scope.activate_scope(scope_id)

    def deactivate_scope(self) -> None:
        """Deactivate the current scope."""
        self.scope.deactivate_scope()

    def clear_scope(self) -> None:
        """Clear scope filter — accept all sources."""
        self.context.clear_scope()
        self.scope.deactivate_scope()

    def get_status(self) -> dict:
        """Get current pipeline status."""
        active_scope = self.scope.get_active_scope()
        return {
            "version": "0.6.0",
            "rsvs_available": self._bridge.is_available,
            "is_rust_core": self._bridge.is_rust_core,
            "scope": self.context.get_scope(),
            "active_scope_domain": active_scope.domain if active_scope else None,
            "conversation_turns": len(self._conversation_history),
            "active_senses": len(self.situation.get_active_senses()),
            "active_predictions": len(self.predictive.get_predictions()),
            "active_cycles": len(self.prediction_loop.get_active_cycles()) if hasattr(self.prediction_loop, 'get_active_cycles') else 0,
            "chat_index_stats": self.chat_index.get_statistics() if hasattr(self, 'chat_index') else {},
            "use_llm": self._use_llm,
            "language": self._language,
            "temporal_stats": self.temporal.get_stats(),
            "v12_available": self.v12.available,
        }

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
