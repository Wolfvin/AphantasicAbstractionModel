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
      -> Final Output (traceable reasoning chain + confidence)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from layer2.bridge import AbstractionBridge, RsvsBridge, get_bridge, is_rust_core_available
from layer2.llm import generate_narrative
from layer2.context import ContextLayer
from layer2.situation import SituationLayer
from layer2.predictive import PredictiveEngine, Prediction, Anomaly, BeliefUpdate
from layer2.pattern import PatternOutput, ReasoningStep, PatternResult


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
        """
        self._eta = eta
        self._anomaly_threshold = anomaly_threshold
        self._auto_search = auto_search
        self._use_llm = use_llm
        self._language = language

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

        This is the main entry point. It runs through all 5 layers:

        1. Context Layer: Search internet if needed, apply scope filter
        2. Situation Layer: Ingest the question, find relevant context
        3. RSVS Core: Spreading activation, structural analysis
        4. Predictive Engine: Predict, detect anomalies
        5. Pattern Output: Pattern completion, generate narrative

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

        # ---- Step 1: Context Layer ----
        # Ingest the question into context
        self.context.ingest_text(question, source=source)

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

        # ---- Step 2: Situation Layer ----
        # Record the question in conversation history
        msg_stats = self.situation.add_message("user", question)
        self._conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": time.time(),
        })

        # Find relevant context from chat history / graph
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

        # ---- Step 3: RSVS Core + Predictive Engine ----
        # Make predictions based on context
        context_atoms = context or [
            s.get("label", s.get("concept", ""))
            for s in active_senses[:5]
            if s.get("label") or s.get("concept")
        ]

        # Predict for the main concept in the question
        prediction = self.predictive.predict(question, context_atoms)
        if prediction:
            all_predictions.append({
                "concept": prediction.concept,
                "expected": prediction.expected_compositions,
                "confidence": prediction.confidence,
            })

        # Detect existing anomalies
        anomalies = self.predictive.detect_anomalies()
        for anomaly in anomalies:
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
        # If we made predictions, check if observation updates beliefs
        if prediction:
            belief_updates = self.predictive.observe_and_update(question, source=source)
            for bu in belief_updates:
                all_belief_updates.append({
                    "concept": bu.concept,
                    "old_confidence": round(bu.old_confidence, 3),
                    "new_confidence": round(bu.new_confidence, 3),
                    "direction": bu.direction,
                    "reason": bu.reason,
                })

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

        # Record assistant response in conversation
        self.situation.add_message("assistant", answer)
        self._conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": time.time(),
        })

        latency = time.time() - start_time

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
                "active_senses_count": len(active_senses),
                "rsvs_available": self._is_rsvs_available(),
                "use_llm": self._use_llm,
            },
        )

    def ingest(self, text: str, source: str = "user_input") -> dict:
        """Ingest text into the knowledge graph without asking a question.

        Useful for pre-loading knowledge before asking questions.

        Args:
            text: Text to ingest.
            source: Source type for trust scoring.

        Returns:
            Stats from the ingestion process.
        """
        # Ingest through all layers
        context_stats = self.context.ingest_text(text, source=source)
        situation_stats = self.situation.add_message("system", text)

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
            "version": "8.3.0",
            "rsvs_available": self._bridge.is_available,
            "is_rust_core": self._bridge.is_rust_core,
            "scope": self.context.get_scope(),
            "conversation_turns": len(self._conversation_history),
            "active_senses": len(self.situation.get_active_senses()),
            "active_predictions": len(self.predictive.get_predictions()),
            "use_llm": self._use_llm,
            "language": self._language,
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


# Backward compat aliases
GeniusPipeline = AamPipeline
GeniusResponse = AamResponse
