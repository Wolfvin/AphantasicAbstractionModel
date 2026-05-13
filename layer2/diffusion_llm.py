"""
AAM Diffusion LLM — The Body of Aphantasic Abstraction Model

CRITICAL CONCEPT: AAM is "1 pikiran + 1 tubuh" (1 mind + 1 body).
- Pikiran (Mind) = RSVS Knowledge Graph — structural, relational, perfect memory
- Tubuh (Body) = This Diffusion LLM — generates natural language FROM the graph

This is NOT a general-purpose LLM. This is a SPECIALIZED sentence composer
that takes structured graph data as input and produces coherent, evidence-backed
narrative output. Think of it as a "vocal cord" for the graph — it can only
say what the graph knows, but it says it fluently.

Why Diffusion?
- Diffusion models start from noise and iteratively denoise
- This mirrors how Jin Soun's thoughts form: from vague intuition →
  clearer pattern → explicit narrative
- Unlike autoregressive LLMs (GPT), diffusion models can:
  - Be conditioned on structured input (graph)
  - Revise earlier parts during generation (non-sequential)
  - Produce more coherent long-form text from structure

Architecture (Future Training Target):
  Input: Graph conditioning (evidence nodes, compositions, confidence, anomalies)
  Process: Iterative denoising from noise
  Output: Natural language narrative grounded in graph structure

Current Status: DESIGN PHASE
  - This module provides the interface and architecture specification
  - Actual model training is a separate effort
  - For now, falls back to z-ai-web-dev-sdk or template-based generation
  - The interface is designed so that once the diffusion model is trained,
    it can be swapped in seamlessly

Analogi: Jin Soun = graph (pikiran). Tubuhnya = this model (tubuh).
Tubuhnya third-rate, tapi karena pikirannya sempurna, outputnya
masih bisa mengalahkan lawan yang punya tubuh lebih kuat tapi
pikiran lebih lemah. Bedanya dengan konsep sebelumnya: tubuh
ini BUKAN LLM umum, tapi model yang KHUSUS dilatih untuk AAM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class GraphConditioning:
    """Structured input for the Diffusion LLM.

    This is the "thought" that the body (diffusion model) expresses.
    Contains everything the graph knows about a topic, structured
    so the model can generate from it.

    Analogi: Ini adalah "pikiran terstruktur" yang Jin Soun
    coba ucapkan — semua bukti, semua relasi, semua confidence.
    """
    trigger: str = ""
    evidence_nodes: list[str] = field(default_factory=list)
    compositions: list[dict] = field(default_factory=list)
    confidence_map: dict[str, float] = field(default_factory=dict)
    anomalies: list[dict] = field(default_factory=list)
    reasoning_chain: list[dict] = field(default_factory=list)
    source_trust: float = 1.0
    scope_active: bool = False
    temporal_context: list[dict] = field(default_factory=list)

    def to_conditioning_text(self) -> str:
        """Convert to a text representation that can condition generation."""
        parts = []
        if self.trigger:
            parts.append(f"TOPIC: {self.trigger}")
        if self.evidence_nodes:
            parts.append(f"EVIDENCE: {', '.join(self.evidence_nodes[:20])}")
        if self.compositions:
            comp_strs = []
            for c in self.compositions[:10]:
                comp_strs.append(f"  - {c.get('label', str(c))}")
            parts.append("COMPOSITIONS:\n" + "\n".join(comp_strs))
        if self.confidence_map:
            conf_strs = [f"  - {k}: {v:.1%}" for k, v in list(self.confidence_map.items())[:10]]
            parts.append("CONFIDENCE:\n" + "\n".join(conf_strs))
        if self.anomalies:
            parts.append(f"ANOMALIES: {len(self.anomalies)} detected")
            for a in self.anomalies[:5]:
                parts.append(f"  - {a.get('description', str(a)[:100])}")
        if self.reasoning_chain:
            parts.append(f"REASONING STEPS: {len(self.reasoning_chain)}")
            for i, step in enumerate(self.reasoning_chain[:5], 1):
                parts.append(f"  {i}. [{step.get('step_type', '?')}] {step.get('description', '')[:80]}")
        if self.source_trust < 1.0:
            parts.append(f"SOURCE TRUST: {self.source_trust:.0%}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "evidence_nodes": self.evidence_nodes,
            "compositions": self.compositions,
            "confidence_map": self.confidence_map,
            "anomalies": self.anomalies,
            "reasoning_chain": self.reasoning_chain,
            "source_trust": self.source_trust,
            "scope_active": self.scope_active,
            "temporal_context": self.temporal_context,
        }


@dataclass
class DiffusionOutput:
    """Output from the Diffusion LLM.

    Unlike raw LLM text, this includes:
    - The generated narrative
    - Which evidence nodes were referenced
    - Confidence of each sentence
    - Number of diffusion steps used
    """
    narrative: str = ""
    referenced_evidence: list[str] = field(default_factory=list)
    sentence_confidences: list[float] = field(default_factory=list)
    diffusion_steps: int = 0
    model_name: str = "fallback"
    generation_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "narrative": self.narrative,
            "referenced_evidence": self.referenced_evidence,
            "sentence_confidences": self.sentence_confidences,
            "diffusion_steps": self.diffusion_steps,
            "model_name": self.model_name,
            "generation_time_s": round(self.generation_time_s, 3),
        }


class AamDiffusionLLM:
    """AAM's dedicated sentence composition model.

    This is NOT a general LLM. This is AAM's own "tubuh" (body) —
    a specialized model trained to compose sentences FROM graph structure.

    Current Status: DESIGN PHASE
    - Interface is fully specified
    - Falls back to template-based or external LLM generation
    - When the actual diffusion model is trained, it will be loaded here
    - The key insight: 1 pikiran + 1 tubuh, not "pikiran + LLM sewa"

    Future Architecture:
    ┌──────────────────────────────────────────────────┐
    │  AAM Diffusion LLM (The Body)                    │
    │                                                   │
    │  Training Data: Graph→Narrative pairs            │
    │  Architecture: Small transformer (100M-500M)     │
    │  Process:                                         │
    │    1. Encode graph conditioning (evidence, etc.) │
    │    2. Start from learned noise prior             │
    │    3. Iteratively denoise (N steps)              │
    │    4. Output: grounded narrative                 │
    │                                                   │
    │  Key difference from autoregressive:             │
    │    - Can revise earlier tokens during generation │
    │    - Conditioned on full graph, not just prefix  │
    │    - Non-sequential = more coherent long-form   │
    └──────────────────────────────────────────────────┘

    Analogi: Jin Soun (graph) + tubuhnya (this model).
    Tubuhnya third-rate, tapi karena KHUSUS dilatih untuk
    mengeksekusi perintah dari graph-nya sendiri, outputnya
    lebih terarah daripada LLM umum yang "tidak kenal" graph.
    """

    # Model status
    MODEL_STATUS_DESIGN = "design"
    MODEL_STATUS_TRAINING = "training"
    MODEL_STATUS_READY = "ready"

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the AAM Diffusion LLM.

        Args:
            model_path: Path to a trained diffusion model.
                If None, uses fallback generation.
        """
        self._model_path = model_path
        self._model = None
        self._model_status = self.MODEL_STATUS_DESIGN
        self._diffusion_steps = 10  # Default denoising steps

        if model_path:
            self._try_load_model(model_path)

    def _try_load_model(self, path: str) -> None:
        """Try to load a trained diffusion model.

        This will be implemented once the model is trained.
        For now, it logs a warning and stays in design mode.
        """
        # TODO: Implement model loading once training is complete
        logger.info(
            "AAM Diffusion LLM: Model loading not yet implemented. "
            "Path: %s. Using fallback generation.", path
        )
        self._model_status = self.MODEL_STATUS_DESIGN

    @property
    def model_status(self) -> str:
        """Current status of the diffusion model."""
        return self._model_status

    @property
    def is_ready(self) -> bool:
        """Whether the diffusion model is ready for inference."""
        return self._model_status == self.MODEL_STATUS_READY

    def generate(
        self,
        conditioning: GraphConditioning,
        max_tokens: int = 500,
        temperature: float = 0.7,
        language: str = "id",
    ) -> DiffusionOutput:
        """Generate narrative from graph conditioning.

        This is the main method. It takes structured graph data
        and produces a natural language narrative.

        When the diffusion model is trained:
        1. Encode conditioning into latent representation
        2. Sample noise from learned prior
        3. Iteratively denoise for N steps
        4. Decode latent into text

        For now, uses fallback generation.

        Analogi: Jin Soun "berbicara" — mengubah pikiran terstruktur
        menjadi kata-kata. Tubuhnya mungkin terbatas, tapi karena
        pikirannya sempurna, apa yang diucapkan tetap akurat.

        Args:
            conditioning: Structured graph data to condition generation.
            max_tokens: Maximum output length.
            temperature: Sampling temperature (higher = more creative).
            language: Output language ("id" or "en").

        Returns:
            DiffusionOutput with the generated narrative.
        """
        start_time = time.time()

        if self.is_ready and self._model is not None:
            # TODO: Implement actual diffusion generation
            # This is where the trained model would be called
            pass

        # Fallback: Template-based generation from graph conditioning
        output = self._fallback_generate(conditioning, max_tokens, language)
        output.generation_time_s = time.time() - start_time
        return output

    def _fallback_generate(
        self,
        conditioning: GraphConditioning,
        max_tokens: int,
        language: str,
    ) -> DiffusionOutput:
        """Fallback generation when diffusion model is not available.

        Uses structured templates to compose narrative from graph data.
        This ensures AAM always has a "body" even without a trained model.

        Analogi: Bahkan tanpa diffusion model, Jin Soun tetap bisa
        "berbicara" — mungkin tidak sefasih LLM umum, tapi tetap
        akurat karena datanya dari graph yang sempurna.
        """
        parts = []
        referenced = []
        confidences = []

        # Opening — state the topic
        if conditioning.trigger:
            if language == "id":
                parts.append(f"Berdasarkan analisis terhadap \"{conditioning.trigger}\":")
            else:
                parts.append(f"Based on analysis of \"{conditioning.trigger}\":")

        # Reasoning chain — each step becomes a sentence
        for i, step in enumerate(conditioning.reasoning_chain[:6], 1):
            step_type = step.get("step_type", "analysis")
            desc = step.get("description", "")
            conf = step.get("confidence", 0.5)
            evidence = step.get("evidence_nodes", [])

            if desc:
                if language == "id":
                    parts.append(f"{i}. [{step_type.title()}] {desc}")
                else:
                    parts.append(f"{i}. [{step_type.title()}] {desc}")
                confidences.append(conf)
                referenced.extend(evidence[:3])

        # Anomalies
        if conditioning.anomalies:
            if language == "id":
                parts.append(f"\nAnomali terdeteksi: {len(conditioning.anomalies)}")
            else:
                parts.append(f"\nAnomalies detected: {len(conditioning.anomalies)}")
            for a in conditioning.anomalies[:3]:
                desc = a.get("description", str(a)[:100])
                parts.append(f"  - {desc}")
                confidences.append(0.6)

        # Evidence summary
        if conditioning.evidence_nodes:
            top_evidence = conditioning.evidence_nodes[:10]
            if language == "id":
                parts.append(f"\nBukti: {', '.join(top_evidence)}")
            else:
                parts.append(f"\nEvidence: {', '.join(top_evidence)}")
            referenced.extend(top_evidence)

        # Confidence
        if conditioning.confidence_map:
            avg_conf = sum(conditioning.confidence_map.values()) / max(len(conditioning.confidence_map), 1)
            confidences.append(avg_conf)
            if language == "id":
                parts.append(f"\nTingkat keyakinan rata-rata: {avg_conf:.0%}")
            else:
                parts.append(f"\nAverage confidence: {avg_conf:.0%}")

        # Source trust warning
        if conditioning.source_trust < 0.7:
            if language == "id":
                parts.append(f"\nPerhatian: Kepercayaan sumber hanya {conditioning.source_trust:.0%}")
            else:
                parts.append(f"\nNote: Source trust is only {conditioning.source_trust:.0%}")

        narrative = "\n".join(parts)

        # Truncate if needed
        if len(narrative) > max_tokens * 4:  # rough char estimate
            narrative = narrative[:max_tokens * 4] + "..."

        return DiffusionOutput(
            narrative=narrative,
            referenced_evidence=list(dict.fromkeys(referenced)),
            sentence_confidences=confidences or [0.5],
            diffusion_steps=0,  # No diffusion in fallback
            model_name="aam_fallback",
        )

    def get_status(self) -> dict:
        """Get diffusion model status."""
        return {
            "model_status": self._model_status,
            "model_path": self._model_path,
            "is_ready": self.is_ready,
            "diffusion_steps": self._diffusion_steps,
            "architecture": "diffusion_transformer",
            "training_status": "not_started",
            "estimated_params": "100M-500M",
            "note": "AAM's own body — specialized sentence composer, not a general LLM",
        }
