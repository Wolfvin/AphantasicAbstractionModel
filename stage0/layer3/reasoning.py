"""
AAM Layer 3 — Reasoning Engine

Deductive chain builder: takes PatternResult from Layer 2 and
produces an auditable, node-by-node reasoning chain backed by
evidence nodes in the RSVS graph.

Analogi Jin Soun:
  "Gu Ilmu + Jang Hangi mencuri Snow Plum Pill."
  Evidence: [tanggal Hefei] → [misi Diancang] → [tidak ada pil di pasar]
  Confidence: 87%

Core design:
  - Each claim maps to specific evidence NodeId/SenseId pairs
  - Each confidence score is derived from grounding_score in RSVS
  - Output is fully traceable — any conclusion can be audited
    by following the chain back to its evidence nodes
"""

from __future__ import annotations

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# P1-2: Cross-package imports use absolute style (layer2 is a sibling package)
from layer2.bridge import V12PipelineBridge, get_bridge
from layer2.pattern import PatternResult, ReasoningStep

# 5-Pillar: Gate 3 — Uncertainty Calibration
from validation_gates.uncertainty_calibration import UncertaintyCalibrationGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeductiveStep:
    """A single step in a deductive reasoning chain.

    Unlike ReasoningStep (which records *what happened* during pattern
    completion), DeductiveStep records a *logical claim* and the evidence
    that supports it.  Every claim is tied to specific NodeId / SenseId
    pairs in the RSVS graph, making the chain fully auditable.

    Attributes:
        claim: The logical claim or conclusion of this step.
        evidence_node_ids: List of (NodeId_label, SenseId) tuples that
            serve as evidence for this claim in the RSVS graph.
        confidence: Confidence score (0.0–1.0), derived from
            grounding scores of the evidence nodes.
        reasoning_type: Type of reasoning used
            ("deduction", "induction", "abduction", "analogy",
             "composition", "substitution", "anomaly_driven").
        grounding_scores: Mapping of evidence node labels to their
            grounding scores from RSVS.
        description: Human-readable explanation of this step.
    """

    claim: str
    evidence_node_ids: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.5
    reasoning_type: str = "deduction"
    grounding_scores: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "claim": self.claim,
            "evidence_node_ids": [
                {"node_id": nid, "sense_id": sid}
                for nid, sid in self.evidence_node_ids
            ],
            "confidence": round(self.confidence, 4),
            "reasoning_type": self.reasoning_type,
            "grounding_scores": {k: round(v, 4) for k, v in self.grounding_scores.items()},
            "description": self.description,
        }


@dataclass
class DeductiveChain:
    """A complete deductive reasoning chain with full traceability.

    Built by ReasoningEngine from a PatternResult (Layer 2 output).
    Each step is auditable back to evidence nodes in the RSVS graph.

    Analogi: Laporan lengkap Jin Soun tentang kasus Snow Plum Pill —
    bukan hanya kesimpulan, tapi rantai penalaran dari bukti pertama
    hingga kesimpulan akhir, dengan setiap langkah bisa ditelusuri.

    Attributes:
        trigger: What triggered this reasoning chain.
        steps: Ordered list of DeductiveSteps forming the chain.
        conclusion: The final conclusion of the chain.
        aggregate_confidence: Weighted confidence across all steps.
        evidence_summary: Summary of all evidence nodes used.
    """

    trigger: str
    steps: list[DeductiveStep] = field(default_factory=list)
    conclusion: str = ""
    aggregate_confidence: float = 0.0
    evidence_summary: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "trigger": self.trigger,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "aggregate_confidence": round(self.aggregate_confidence, 4),
            "evidence_summary": list(self.evidence_summary),
        }


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """Deductive chain builder — produces auditable reasoning from PatternResult.

    Takes a PatternResult from Layer 2 and builds a deductive chain where:
    - Each claim maps to evidence nodes in the RSVS graph
    - Each confidence is derived from grounding scores
    - The output can be audited node-by-node

    Flow:
    1. Extract activated nodes and compositions from PatternResult
    2. Build deductive steps using composition references as logical steps
    3. Connect each claim to evidence nodes in the RSVS graph
    4. Calculate aggregate confidence from grounding scores
    5. Produce DeductiveChain with full traceability

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(self, bridge: Optional[V12PipelineBridge] = None) -> None:
        """Initialize the ReasoningEngine.

        Args:
            bridge: Optional pre-built V12PipelineBridge instance. If None,
                a new bridge is created via get_bridge().
        """
        if bridge is not None:
            self._bridge = bridge
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.available
        self.is_rust_core = self._bridge.is_rust_core

        # 5-Pillar: Gate 3 — Uncertainty Calibration
        self._calibration_gate = UncertaintyCalibrationGate()

        if self.rsvs_available:
            logger.info("ReasoningEngine initialized with RSVS bridge (rust=%s)", self.is_rust_core)
        else:
            logger.info("ReasoningEngine initialized WITHOUT RSVS core (fallback mode)")

    # ==================================================================
    # MAIN METHOD — build_chain()
    # ==================================================================

    def build_chain(
        self,
        pattern_result: PatternResult,
        bridge: Optional[V12PipelineBridge] = None,
    ) -> DeductiveChain:
        """Build a deductive chain from a PatternResult.

        This is the primary entry point. It takes a PatternResult from
        Layer 2 (which contains activated nodes, anomalies, and patterns)
        and constructs a DeductiveChain where every step is backed by
        evidence nodes in the RSVS graph.

        Steps:
        1. EXTRACT — Get activated nodes and compositions from PatternResult
        2. COMPOSE — Use bridge.compose() for structural reasoning
        3. GROUND — Connect claims to evidence nodes via bridge.senses()
        4. EXPLORE — Use bridge.mcts_query() for complex chain exploration
        5. AGGREGATE — Calculate aggregate confidence from grounding scores

        Args:
            pattern_result: The PatternResult from Layer 2.
            bridge: Optional override bridge (uses instance bridge if None).

        Returns:
            A DeductiveChain with full evidence traceability.
        """
        b = bridge or self._bridge
        chain = DeductiveChain(trigger=pattern_result.trigger)

        # Collect all evidence nodes from the pattern result's reasoning steps
        all_evidence_labels: list[str] = []
        for step in pattern_result.steps:
            all_evidence_labels.extend(step.evidence_nodes)
        all_evidence_labels = list(dict.fromkeys(all_evidence_labels))  # deduplicate

        # ---- Step 1: EXTRACT — Gather activated nodes ----
        extract_step = self._build_extract_step(pattern_result, all_evidence_labels, b)
        chain.steps.append(extract_step)

        # ---- Step 2: COMPOSE — Structural reasoning via compositions ----
        compose_step = self._build_compose_step(pattern_result, all_evidence_labels, b)
        chain.steps.append(compose_step)

        # ---- Step 3: GROUND — Connect claims to evidence nodes ----
        ground_step = self._build_ground_step(all_evidence_labels, b)
        chain.steps.append(ground_step)

        # ---- Step 4: EXPLORE — MCTS for complex chain exploration ----
        explore_step = self._build_explore_step(pattern_result, b)
        chain.steps.append(explore_step)

        # ---- Step 5: CONCLUDE — Build conclusion and aggregate confidence ----
        conclusion_step = self._build_conclusion_step(pattern_result, chain.steps, b)
        chain.steps.append(conclusion_step)
        chain.conclusion = conclusion_step.claim

        # Calculate aggregate confidence
        chain.aggregate_confidence = self._calculate_aggregate_confidence(chain.steps)

        # 5-Pillar Gate 3: Calibrate aggregate confidence
        try:
            calibration_result = self._calibration_gate.calibrate(
                raw_confidence=chain.aggregate_confidence,
                regime="",  # Regime populated by pipeline
            )
            chain.aggregate_confidence = calibration_result.calibrated_confidence

            # Also calibrate individual step confidences
            for step in chain.steps:
                step_cal = self._calibration_gate.calibrate(
                    raw_confidence=step.confidence,
                )
                step.confidence = step_cal.calibrated_confidence
        except Exception as exc:
            logger.debug("Calibration gate failed: %s", exc)

        # Build evidence summary
        chain.evidence_summary = self._build_evidence_summary(chain.steps)

        logger.info(
            "ReasoningEngine.build_chain(): %d steps, confidence=%.3f, trigger='%s'",
            len(chain.steps), chain.aggregate_confidence,
            pattern_result.trigger[:60],
        )

        return chain

    # ==================================================================
    # Step builders
    # ==================================================================

    def _build_extract_step(
        self,
        pattern_result: PatternResult,
        evidence_labels: list[str],
        bridge: V12PipelineBridge,
    ) -> DeductiveStep:
        """Step 1: Extract activated nodes from PatternResult.

        Collects all evidence node labels from the pattern result and
        maps them to NodeId/SenseId pairs in the RSVS graph.
        """
        evidence_ids: list[tuple[str, str]] = []
        grounding_scores: dict[str, float] = {}

        for label in evidence_labels:
            node_id, sense_id, gs = self._get_node_grounding(label, bridge)
            evidence_ids.append((node_id, sense_id))
            grounding_scores[label] = gs

        confidence = self._mean_confidence(grounding_scores) if grounding_scores else 0.3

        return DeductiveStep(
            claim=f"Extracted {len(evidence_labels)} activated nodes from pattern result",
            evidence_node_ids=evidence_ids,
            confidence=confidence,
            reasoning_type="deduction",
            grounding_scores=grounding_scores,
            description=(
                f"Pattern result has {len(pattern_result.steps)} reasoning steps "
                f"and {len(evidence_labels)} unique evidence nodes. "
                f"Anomalies: {len(pattern_result.anomalies)}. "
                f"Pattern confidence: {pattern_result.confidence:.3f}."
            ),
        )

    def _build_compose_step(
        self,
        pattern_result: PatternResult,
        evidence_labels: list[str],
        bridge: V12PipelineBridge,
    ) -> DeductiveStep:
        """Step 2: Use bridge.compose() for structural reasoning.

        Creates compositional nodes that represent logical groupings
        of evidence. Each composition becomes a logical step in the
        deductive chain.
        """
        evidence_ids: list[tuple[str, str]] = []
        grounding_scores: dict[str, float] = {}
        composition_claims: list[str] = []

        # Group evidence by pattern step type
        step_groups: dict[str, list[str]] = {}
        for step in pattern_result.steps:
            step_groups.setdefault(step.step_type, []).extend(step.evidence_nodes)

        for step_type, nodes in step_groups.items():
            if not nodes or not bridge.available:
                continue

            # Create a composition node that represents this logical grouping
            comp_label = f"deduction_{step_type}_{len(composition_claims)}"
            comp_tuples = [(n, "0") for n in nodes[:10]]  # sense_id=0 for default

            try:
                comp_result = bridge.compose(comp_label, comp_tuples)
                if comp_result is not None:
                    node_id = str(comp_result)
                    evidence_ids.append((comp_label, "0"))
                    grounding_scores[comp_label] = 0.6  # composition has moderate grounding
                    composition_claims.append(
                        f"Composed '{step_type}' group with {len(nodes)} evidence nodes → node {node_id}"
                    )
            except Exception as exc:
                logger.debug("compose() failed for '%s': %s", comp_label, exc)

        # If no compositions created, still create a step based on available data
        if not composition_claims:
            if evidence_labels:
                composition_claims.append(
                    f"No compositions created, but {len(evidence_labels)} evidence nodes available for reasoning"
                )
                for label in evidence_labels[:5]:
                    nid, sid, gs = self._get_node_grounding(label, bridge)
                    evidence_ids.append((nid, sid))
                    grounding_scores[label] = gs
            else:
                composition_claims.append("No evidence nodes available for structural reasoning")

        confidence = self._mean_confidence(grounding_scores) if grounding_scores else 0.2
        claim = f"Structural reasoning: {len(composition_claims)} composition group(s) formed"
        description = "; ".join(composition_claims[:5])

        return DeductiveStep(
            claim=claim,
            evidence_node_ids=evidence_ids,
            confidence=confidence,
            reasoning_type="composition",
            grounding_scores=grounding_scores,
            description=description,
        )

    def _build_ground_step(
        self,
        evidence_labels: list[str],
        bridge: V12PipelineBridge,
    ) -> DeductiveStep:
        """Step 3: Connect claims to evidence nodes via bridge.senses().

        For each evidence node, retrieves its senses from the RSVS graph
        to establish the grounding connection. Each sense provides
        grounding_score, coherence, and composition references.
        """
        evidence_ids: list[tuple[str, str]] = []
        grounding_scores: dict[str, float] = {}
        grounded_count = 0

        for label in evidence_labels:
            if bridge.available:
                try:
                    senses = bridge.senses(label)
                    if senses and isinstance(senses, list):
                        for sense in senses:
                            if isinstance(sense, dict):
                                sense_idx = str(sense.get("sense_idx", 0))
                                gs = sense.get("grounding_score", 0.5)
                                evidence_ids.append((label, sense_idx))
                                grounding_scores[label] = max(
                                    grounding_scores.get(label, 0.0), gs
                                )
                                grounded_count += 1
                            else:
                                evidence_ids.append((label, "0"))
                                grounding_scores[label] = 0.4
                                grounded_count += 1
                    else:
                        # Node exists but has no senses
                        evidence_ids.append((label, "0"))
                        grounding_scores[label] = 0.3
                except Exception as exc:
                    logger.debug("senses() failed for '%s': %s", label, exc)
                    evidence_ids.append((label, "0"))
                    grounding_scores[label] = 0.2
            else:
                # Fallback mode — assign default grounding
                evidence_ids.append((label, "0"))
                grounding_scores[label] = 0.3
                grounded_count += 1

        confidence = self._mean_confidence(grounding_scores) if grounding_scores else 0.2
        ungrounded = len(evidence_labels) - grounded_count

        return DeductiveStep(
            claim=f"Grounded {grounded_count}/{len(evidence_labels)} evidence nodes to RSVS senses",
            evidence_node_ids=evidence_ids,
            confidence=confidence,
            reasoning_type="deduction",
            grounding_scores=grounding_scores,
            description=(
                f"Retrieved senses for {grounded_count} evidence nodes. "
                f"{'All nodes grounded.' if ungrounded == 0 else f'{ungrounded} node(s) could not be grounded.'}"
            ),
        )

    def _build_explore_step(
        self,
        pattern_result: PatternResult,
        bridge: V12PipelineBridge,
    ) -> DeductiveStep:
        """Step 4: Use bridge.mcts_query() for complex chain exploration.

        Uses Monte Carlo Tree Search to explore potential reasoning
        paths beyond the directly activated nodes. This discovers
        indirect evidence and alternative explanations.
        """
        evidence_ids: list[tuple[str, str]] = []
        grounding_scores: dict[str, float] = {}
        exploration_results: list[str] = []

        # Collect context atoms from pattern result
        context_atoms: list[str] = []
        for step in pattern_result.steps:
            context_atoms.extend(step.evidence_nodes[:3])
        context_atoms = list(dict.fromkeys(context_atoms))[:10]

        # Use the trigger as the concept to explore
        trigger_concept = pattern_result.trigger.strip()[:100]

        if bridge.available and context_atoms:
            try:
                mcts_result = bridge.mcts_query(
                    node_label=trigger_concept,
                    max_depth=3,
                    simulations=20,
                )
                if mcts_result:
                    if isinstance(mcts_result, dict):
                        # Parse MCTS result
                        best_path = mcts_result.get("best_path", [])
                        if best_path:
                            for node_label in best_path[:5]:
                                if isinstance(node_label, str):
                                    nid, sid, gs = self._get_node_grounding(node_label, bridge)
                                    evidence_ids.append((nid, sid))
                                    grounding_scores[node_label] = gs
                            exploration_results.append(
                                f"MCTS found best path with {len(best_path)} nodes"
                            )

                        simulations = mcts_result.get("simulations_run", 0)
                        depth_reached = mcts_result.get("max_depth_reached", 0)
                        exploration_results.append(
                            f"MCTS: {simulations} simulations, depth {depth_reached}"
                        )
                    else:
                        exploration_results.append(
                            f"MCTS returned result of type {type(mcts_result).__name__}"
                        )
            except Exception as exc:
                logger.debug("mcts_query() failed: %s", exc)
                exploration_results.append(f"MCTS exploration unavailable: {exc}")

        if not exploration_results:
            exploration_results.append(
                "MCTS exploration skipped (no context atoms or bridge unavailable)"
            )

        confidence = self._mean_confidence(grounding_scores) if grounding_scores else 0.3

        return DeductiveStep(
            claim=f"Explored {len(exploration_results)} reasoning path(s) via MCTS",
            evidence_node_ids=evidence_ids,
            confidence=confidence,
            reasoning_type="abduction",
            grounding_scores=grounding_scores,
            description="; ".join(exploration_results),
        )

    def _build_conclusion_step(
        self,
        pattern_result: PatternResult,
        previous_steps: list[DeductiveStep],
        bridge: V12PipelineBridge,
    ) -> DeductiveStep:
        """Step 5: Build final conclusion from all evidence.

        Synthesizes the deductive chain into a final conclusion,
        incorporating anomaly information and pattern confidence.
        """
        # Collect all evidence from previous steps
        all_evidence_ids: list[tuple[str, str]] = []
        all_grounding: dict[str, float] = {}

        for step in previous_steps:
            all_evidence_ids.extend(step.evidence_node_ids)
            all_grounding.update(step.grounding_scores)

        # Build conclusion based on pattern result
        anomaly_count = len(pattern_result.anomalies)
        pattern_conf = pattern_result.confidence
        has_pattern = bool(pattern_result.pattern)

        # Determine conclusion reasoning type
        if anomaly_count > 0:
            reasoning_type = "anomaly_driven"
            conclusion_prefix = "Anomaly-driven conclusion"
        elif has_pattern:
            reasoning_type = "induction"
            conclusion_prefix = "Pattern-based conclusion"
        else:
            reasoning_type = "abduction"
            conclusion_prefix = "Abductive conclusion"

        # Build the claim
        if has_pattern and anomaly_count > 0:
            claim = (
                f"{conclusion_prefix}: Pattern identified with {anomaly_count} "
                f"anomalies. Confidence: {pattern_conf:.1%}. "
                f"Chain backed by {len(all_evidence_ids)} evidence references."
            )
        elif has_pattern:
            claim = (
                f"{conclusion_prefix}: Consistent pattern identified. "
                f"Confidence: {pattern_conf:.1%}. "
                f"Chain backed by {len(all_evidence_ids)} evidence references."
            )
        else:
            claim = (
                f"{conclusion_prefix}: Insufficient pattern evidence. "
                f"Confidence: {pattern_conf:.1%}. "
                f"Chain backed by {len(all_evidence_ids)} evidence references."
            )

        # Aggregate confidence
        avg_grounding = self._mean_confidence(all_grounding) if all_grounding else 0.3
        # Blend pattern confidence with grounding confidence
        confidence = 0.6 * pattern_conf + 0.4 * avg_grounding

        return DeductiveStep(
            claim=claim,
            evidence_node_ids=all_evidence_ids[:20],  # Cap for readability
            confidence=min(1.0, confidence),
            reasoning_type=reasoning_type,
            grounding_scores=all_grounding,
            description=(
                f"Final conclusion synthesized from {len(previous_steps)} reasoning steps, "
                f"{len(all_evidence_ids)} evidence references, "
                f"{anomaly_count} anomalies, and pattern confidence {pattern_conf:.3f}."
            ),
        )

    # ==================================================================
    # 5-Pillar: Gate 3 — Calibration outcome recording
    # ==================================================================

    def record_outcome(
        self,
        predicted_confidence: float,
        actual_correct: bool,
        regime: str = "",
        context: str = "",
    ) -> None:
        """Record whether a reasoning outcome was correct (5-Pillar Gate 3).

        This feeds the UncertaintyCalibrationGate with ground truth data
        so that future confidence scores are better calibrated.

        Args:
            predicted_confidence: The confidence that was assigned.
            actual_correct: Whether the reasoning was actually correct.
            regime: The regime at the time of reasoning.
            context: Context description.
        """
        self._calibration_gate.record_outcome(
            predicted_confidence=predicted_confidence,
            actual_correct=actual_correct,
            regime=regime,
            context=context,
        )

    @property
    def calibration_gate(self) -> UncertaintyCalibrationGate:
        """Access the calibration gate for external queries."""
        return self._calibration_gate

    # ==================================================================
    # Utility methods
    # ==================================================================

    def _get_node_grounding(
        self, label: str, bridge: V12PipelineBridge
    ) -> tuple[str, str, float]:
        """Get grounding info for a node label.

        Returns:
            Tuple of (node_label, sense_id, grounding_score).
        """
        if bridge.available:
            try:
                senses = bridge.senses(label)
                if senses and isinstance(senses, list) and len(senses) > 0:
                    sense = senses[0]
                    if isinstance(sense, dict):
                        sense_idx = str(sense.get("sense_idx", 0))
                        gs = sense.get("grounding_score", 0.5)
                        return (label, sense_idx, gs)
            except Exception as exc:
                logger.debug("Sense lookup failed for '%s': %s", label, exc)

        return (label, "0", 0.3)

    @staticmethod
    def _mean_confidence(scores: dict[str, float]) -> float:
        """Calculate mean confidence from a dict of scores."""
        if not scores:
            return 0.3
        values = list(scores.values())
        return sum(values) / len(values)

    def _calculate_aggregate_confidence(self, steps: list[DeductiveStep]) -> float:
        """Calculate weighted aggregate confidence across all steps.

        Later steps (especially the conclusion) get more weight since
        they build on earlier ones.
        """
        if not steps:
            return 0.0

        # Weights: extract=0.1, compose=0.15, ground=0.2, explore=0.2, conclude=0.35
        weights = [0.10, 0.15, 0.20, 0.20, 0.35]

        total_weight = 0.0
        weighted_sum = 0.0
        for i, step in enumerate(steps):
            w = weights[i] if i < len(weights) else 0.05
            weighted_sum += w * step.confidence
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _build_evidence_summary(self, steps: list[DeductiveStep]) -> list[dict]:
        """Build a summary of all evidence used in the chain."""
        seen_nodes: set[str] = set()
        summary: list[dict] = []

        for step in steps:
            for node_id, sense_id in step.evidence_node_ids:
                key = f"{node_id}:{sense_id}"
                if key not in seen_nodes:
                    seen_nodes.add(key)
                    summary.append({
                        "node_id": node_id,
                        "sense_id": sense_id,
                        "grounding_score": step.grounding_scores.get(node_id, 0.0),
                        "used_in_step": step.reasoning_type,
                    })

        return summary


# ---------------------------------------------------------------------------
# v12-aware stubs
# ---------------------------------------------------------------------------

class V12ReasoningBridge:
    """Bridge to v12 reasoning capabilities.
    
    In v12.0, reasoning is handled by the ExecutiveOrchestrator which selects
    cognitive mode (Reactive/Analytical/Reflective) and runs the appropriate
    transforms. This bridge exposes v12 reasoning to layer3.
    
    STUB:IMPROVED — now performs real analysis using bridge compositions,
    gap detection, and cognitive mode selection.
    """
    
    def __init__(self, v12_bridge=None):
        self._bridge = v12_bridge
    
    def analyze(self, text: str) -> dict:
        """Analyze text using v12 cognitive mode selection and graph state.
        
        STUB:IMPROVED — performs meaningful analysis:
        1. Ingests text to update graph
        2. Selects cognitive mode
        3. Extracts key concepts from compositions
        4. Identifies related knowledge via compositions()
        5. Detects gaps via detect_gaps()
        
        Returns a dict with: mode, related_concepts, gaps, confidence,
        ingest_summary, and analysis description.
        """
        if self._bridge is None or not self._bridge.available:
            return {
                "mode": "unavailable",
                "analysis": None,
                "related_concepts": [],
                "gaps": [],
                "confidence": 0.0,
            }
        
        # Step 1: Select cognitive mode
        mode = self._bridge.cognitive_mode(text)
        
        # Step 2: Ingest text to update graph
        ingest_result = self._bridge.ingest(text)
        
        # Step 3: Extract key concepts from the text
        # Use the bridge's query/relate to find concepts that are in the graph
        raw_words = text.replace(",", " ").replace(".", " ").split()
        candidate_concepts = [w for w in raw_words if len(w) > 3]
        
        # Step 4: Find related knowledge through compositions
        related_concepts: list[dict] = []
        seen_concept_labels: set[str] = set()
        
        for concept in candidate_concepts[:10]:
            try:
                # Find compositions containing this concept
                senses = self._bridge.senses(concept)
                if senses and isinstance(senses, list):
                    for sense in senses:
                        if isinstance(sense, dict):
                            core_atoms = sense.get("core_atoms", [])
                            coherence = sense.get("coherence", 0.5)
                            for atom in core_atoms:
                                if atom.lower() not in seen_concept_labels:
                                    seen_concept_labels.add(atom.lower())
                                    related_concepts.append({
                                        "concept": atom,
                                        "coherence": coherence,
                                        "source": "composition",
                                    })
                
                # Also check direct relations
                related = self._bridge.relate(concept)
                for rel in related[:5]:
                    if rel.lower() not in seen_concept_labels:
                        seen_concept_labels.add(rel.lower())
                        related_concepts.append({
                            "concept": rel,
                            "coherence": 0.4,
                            "source": "edge",
                        })
            except Exception:
                continue
        
        # Step 5: Detect knowledge gaps
        gaps: list[dict] = []
        try:
            raw_gaps = self._bridge.detect_gaps()
            for gap in raw_gaps[:10]:
                gaps.append({
                    "gap_id": gap.get("gap_id", "unknown"),
                    "gap_type": gap.get("gap_type", "Unknown"),
                    "description": gap.get("description", ""),
                    "severity": gap.get("severity", "low"),
                    "missing_role": gap.get("missing_role", "unknown"),
                })
        except Exception:
            pass
        
        # Step 6: Compute overall confidence
        composition_count = self._bridge.composition_count()
        node_count = self._bridge.node_count()
        
        # Confidence based on graph richness and ingest success
        confidence = 0.3
        if ingest_result.get("compositions_created", 0) > 0:
            confidence += 0.2
        if ingest_result.get("atoms_created", 0) > 0:
            confidence += 0.1
        if related_concepts:
            confidence += min(0.2, len(related_concepts) * 0.02)
        if gaps:
            confidence -= min(0.15, len(gaps) * 0.03)
        confidence = max(0.0, min(1.0, confidence))
        
        # Mode-specific adjustments
        if mode == "Analytical":
            confidence = min(1.0, confidence + 0.05)
        elif mode == "Reflective":
            confidence = min(1.0, confidence + 0.03)
        
        # Build analysis description
        analysis_parts = []
        if related_concepts:
            analysis_parts.append(
                f"Found {len(related_concepts)} related concepts"
            )
        if gaps:
            analysis_parts.append(f"Identified {len(gaps)} knowledge gaps")
        analysis_parts.append(f"Graph has {node_count} nodes, {composition_count} compositions")
        analysis_desc = "; ".join(analysis_parts) if analysis_parts else "Minimal analysis available"
        
        return {
            "mode": mode,
            "analysis": analysis_desc,
            "related_concepts": related_concepts[:20],
            "gaps": gaps,
            "confidence": round(confidence, 3),
            "ingest_summary": {
                "atoms_created": ingest_result.get("atoms_created", 0),
                "compositions_created": ingest_result.get("compositions_created", 0),
                "gaps_detected": ingest_result.get("gaps_detected", 0),
                "edges_created": ingest_result.get("edges_created", 0),
            },
            "graph_state": {
                "nodes": node_count,
                "compositions": composition_count,
                "total_gaps": len(gaps),
            },
        }
