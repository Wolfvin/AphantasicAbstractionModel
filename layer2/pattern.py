"""
Pattern Completion Output — Recall → Cross-Reference → Anomaly → Pattern → Narrative

Analogi: Scene Snow Plum Pill — Jin Soun BUKAN mulai dari hipotesis lalu menguji.
Dia mulai dari RECALL massal → cross-reference → anomaly detection → pattern completion.

Step 1: TRIGGER → relate() aktifkan semua koneksi
Step 2: RECALL MASSAL → spreading activation dari trigger node
Step 3: CROSS-REFERENCE → bandingkan komposisi node aktif
Step 4: ANOMALY DETECTION → appraise() expected ≠ observed
Step 5: PATTERN COMPLETION → compose() fragmen → pola utuh
Step 6: NARRATIVE OUTPUT → generate teks berdasarkan reasoning chain

KEY INSIGHT: LLM generates text FROM graph, not from nothing.
Graph = structural memory, LLM = narrative voice.
Jin Soun = graph, his body = limited LLM.

The PatternOutput layer takes a trigger (a question, an observation, a clue)
and runs the full 6-step pipeline to produce a traceable, evidence-backed
narrative. Each step produces a ReasoningStep that records what happened,
what evidence was used, and how confident we are.

Analogi: Jin Soun melihat satu petunjuk → otaknya secara otomatis
mengaktifkan semua kenangan terkait → membandingkan → menemukan
kontradiksi → menyusun pola → menghasilkan narasi. Bukan linear,
tapi paralel dan simultan — seperti bagaimana otak manusia sebenarnya bekerja.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum depth for spreading activation in recall step
_MAX_RECALL_DEPTH = 3

# Maximum number of nodes to compare in cross-reference
_MAX_CROSS_REF_PAIRS = 20

# Minimum structural similarity to consider two nodes "related"
_MIN_SIMILARITY_FOR_XREF = 0.2

# Stop words for fallback text processing
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
class ReasoningStep:
    """A single step in the reasoning chain.

    Each step records what type of reasoning was performed, what data
    was produced, which graph nodes were used as evidence, and how
    confident we are in this step's conclusions.

    Analogi: Setiap langkah pemikiran Jin Soun dicatat terpisah:
    "Pada langkah RECALL, aku mengingat X, Y, Z. Keyakinan: 0.7.
    Buktinya: node Snow_Plum_Pill, node Hefei, node racun."

    Attributes:
        step_type: The type of reasoning step
            ("trigger", "recall", "cross_reference", "anomaly",
             "pattern", "narrative").
        description: What happened in this step.
        data: Structured data produced by this step.
        evidence_nodes: Which graph nodes were used as evidence.
        confidence: Confidence of this step's output (0.0 - 1.0).
    """

    step_type: str
    description: str = ""
    data: dict = field(default_factory=dict)
    evidence_nodes: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "step_type": self.step_type,
            "description": self.description,
            "data": dict(self.data),
            "evidence_nodes": list(self.evidence_nodes),
            "confidence": self.confidence,
        }


@dataclass
class PatternResult:
    """The complete result of a pattern completion process.

    Contains the full reasoning chain from trigger to narrative,
    with traceable evidence at every step.

    Analogi: Laporan lengkap Jin Soun tentang kasus Snow Plum Pill.
    Bukan hanya kesimpulan — tapi seluruh rantai penalaran, dari
    petunjuk pertama hingga narasi akhir, dengan bukti di setiap langkah.

    Attributes:
        trigger: What triggered this pattern completion.
        steps: The reasoning chain (list of ReasoningSteps).
        pattern: The completed pattern description.
        narrative: The final narrative output text.
        evidence_chain: Traceable evidence nodes with metadata.
        confidence: Overall confidence of the result.
        anomalies: Any anomalies found during the process.
    """

    trigger: str
    steps: list[ReasoningStep] = field(default_factory=list)
    pattern: str = ""
    narrative: str = ""
    evidence_chain: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    anomalies: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "trigger": self.trigger,
            "steps": [s.to_dict() for s in self.steps],
            "pattern": self.pattern,
            "narrative": self.narrative,
            "evidence_chain": list(self.evidence_chain),
            "confidence": self.confidence,
            "anomalies": list(self.anomalies),
        }


# ---------------------------------------------------------------------------
# PatternOutput
# ---------------------------------------------------------------------------

class PatternOutput:
    """Pattern Completion Output — Recall → Cross-Reference → Anomaly → Pattern → Narrative.

    The key innovation layer. Takes a trigger and produces a complete,
    traceable, evidence-backed narrative through a 6-step pipeline:

    1. TRIGGER — Parse the trigger, identify key concepts
    2. RECALL MASSAL — Spreading activation from trigger nodes
    3. CROSS-REFERENCE — Compare activated nodes for shared structure
    4. ANOMALY DETECTION — Check for contradictions between expected and observed
    5. PATTERN COMPLETION — Connect fragments into a coherent pattern
    6. NARRATIVE OUTPUT — Generate traceable text from the reasoning chain

    KEY INSIGHT: The graph provides structure, the narrative provides voice.
    Like Jin Soun's perfect memory (graph) expressing through his limited
    body (narrative). The LLM doesn't generate from nothing — it generates
    FROM the graph.

    Analogi: Jin Soun melihat satu petunjuk → otaknya mengaktifkan
    semua kenangan terkait → membandingkan → menemukan anomali →
    menyusun pola → menghasilkan narasi yang bisa ditelusuri.

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(self, rsvs_instance: Any | None = None, bridge: Optional[RsvsBridge] = None) -> None:
        """Initialize the Pattern Output layer.

        Args:
            rsvs_instance: Optional pre-built RSVS instance. If None,
                the layer will obtain a bridge via get_bridge().
            bridge: Optional pre-built RsvsBridge instance. If provided,
                takes precedence over rsvs_instance.
        """
        if bridge is not None:
            self._bridge = bridge
        elif rsvs_instance is not None:
            self._bridge = RsvsBridge(rsvs_instance=rsvs_instance)
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core

        # Fallback graph for when RSVS is unavailable
        # Maps concept → {compositions: [...], relations: [...], confidence: float}
        self._fallback_graph: dict[str, dict] = {}

        # History of processed triggers
        self._history: list[PatternResult] = []

        if self.rsvs_available:
            logger.info("PatternOutput initialized with RSVS core (rust=%s)", self.is_rust_core)
        else:
            logger.info("PatternOutput initialized WITHOUT RSVS core (fallback mode)")

    # ==================================================================
    # MAIN METHOD — process()
    # ==================================================================

    def process(
        self, trigger: str, context: list[str] | None = None
    ) -> PatternResult:
        """The MAIN method. Takes a trigger string and returns a complete PatternResult.

        Runs the full 6-step pipeline:
        1. TRIGGER → parse the trigger, identify key concepts
        2. RECALL MASSAL → spreading activation from trigger nodes
        3. CROSS-REFERENCE → compare activated nodes for shared structure
        4. ANOMALY DETECTION → check for contradictions
        5. PATTERN COMPLETION → connect fragments into a pattern
        6. NARRATIVE OUTPUT → generate traceable text

        Each step produces a ReasoningStep that records what happened,
        what evidence was used, and the confidence level. The final
        PatternResult contains the complete reasoning chain.

        Analogi: Jin Soun mendengar "Snow Plum Pill" →
        Langkah 1: TRIGGER — identifikasi "Snow Plum Pill" sebagai konsep kunci
        Langkah 2: RECALL — aktifkan semua kenangan tentang pil, racun, Hefei
        Langkah 3: CROSS-REF — bandingkan Snow Plum Pill dengan racun lain
        Langkah 4: ANOMALY — "kok pil ini tidak mempan?" → anomali
        Langkah 5: PATTERN — pil palsu → motif → kambing hitam
        Langkah 6: NARRATIVE — "Ju Jangmok bukan pencuri, dia kambing hitam..."

        Args:
            trigger: The trigger text — a question, observation, or clue
                that starts the pattern completion process.
            context: Optional list of context atoms to focus the recall
                and cross-reference steps.

        Returns:
            A PatternResult containing the complete reasoning chain,
            pattern description, narrative, and evidence traceability.
        """
        context = context or []
        result = PatternResult(trigger=trigger)
        step_data: dict[str, Any] = {
            "context": context,
        }

        # ---- Step 1: TRIGGER ----
        logger.debug("PatternOutput.process() Step 1: TRIGGER — '%s'", trigger)
        trigger_step = self._trigger(trigger)
        result.steps.append(trigger_step)
        step_data["trigger_concepts"] = trigger_step.data.get("concepts", [])
        step_data["trigger_nodes"] = trigger_step.evidence_nodes

        # ---- Step 2: RECALL MASSAL ----
        logger.debug(
            "PatternOutput.process() Step 2: RECALL — from %d concepts",
            len(step_data["trigger_concepts"])
        )
        recall_step = self._recall(step_data["trigger_concepts"])
        result.steps.append(recall_step)
        step_data["activated_nodes"] = recall_step.data.get("activated_nodes", [])
        step_data["activated_edges"] = recall_step.data.get("activated_edges", [])

        # ---- Step 3: CROSS-REFERENCE ----
        logger.debug(
            "PatternOutput.process() Step 3: CROSS-REFERENCE — %d activated nodes",
            len(step_data["activated_nodes"])
        )
        xref_step = self._cross_reference(step_data["activated_nodes"])
        result.steps.append(xref_step)
        step_data["shared_compositions"] = xref_step.data.get("shared_compositions", [])
        step_data["similar_pairs"] = xref_step.data.get("similar_pairs", [])

        # ---- Step 4: ANOMALY DETECTION ----
        logger.debug("PatternOutput.process() Step 4: ANOMALY DETECTION")
        anomaly_step = self._detect_anomalies(xref_step.data)
        result.steps.append(anomaly_step)
        step_data["anomalies"] = anomaly_step.data.get("anomalies", [])

        # Record anomalies in the result
        for anomaly_dict in step_data["anomalies"]:
            result.anomalies.append(anomaly_dict)

        # ---- Step 5: PATTERN COMPLETION ----
        logger.debug("PatternOutput.process() Step 5: PATTERN COMPLETION")
        pattern_step = self._complete_pattern(
            recall_data=recall_step.data,
            anomalies=step_data["anomalies"],
            cross_ref=xref_step.data,
        )
        result.steps.append(pattern_step)
        result.pattern = pattern_step.data.get("pattern", "")
        step_data["pattern"] = result.pattern
        step_data["substitutions"] = pattern_step.data.get("substitutions", [])

        # ---- Step 6: NARRATIVE OUTPUT ----
        logger.debug("PatternOutput.process() Step 6: NARRATIVE OUTPUT")
        evidence_nodes = []
        for step in result.steps:
            evidence_nodes.extend(step.evidence_nodes)
        evidence_nodes = list(dict.fromkeys(evidence_nodes))  # Deduplicate, preserve order

        narrative_step = self._generate_narrative(
            pattern=result.pattern, evidence=evidence_nodes,
            trigger=trigger, steps=result.steps,
        )
        result.steps.append(narrative_step)
        result.narrative = narrative_step.data.get("narrative", "")

        # Build evidence chain — traceable from narrative back to graph
        result.evidence_chain = self._build_evidence_chain(result.steps)

        # Compute overall confidence — weighted average of step confidences
        if result.steps:
            # Weight later steps more heavily (they build on earlier ones)
            weights = [0.1, 0.15, 0.15, 0.2, 0.2, 0.2]
            total_weight = 0.0
            weighted_sum = 0.0
            for i, step in enumerate(result.steps):
                w = weights[i] if i < len(weights) else 0.1
                weighted_sum += w * step.confidence
                total_weight += w
            result.confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            result.confidence = 0.0

        # Store in history
        self._history.append(result)

        logger.info(
            "PatternOutput.process() complete — confidence=%.3f, "
            "%d steps, %d anomalies, pattern='%s'",
            result.confidence, len(result.steps),
            len(result.anomalies),
            result.pattern[:80] if result.pattern else "(none)"
        )

        return result

    # ==================================================================
    # Step 1: TRIGGER
    # ==================================================================

    def _trigger(self, text: str) -> ReasoningStep:
        """Step 1: Parse the trigger text and identify key concepts.

        Extracts concepts from the trigger using RSVS relate() to find
        connected nodes. If RSVS is unavailable, falls back to keyword
        extraction.

        Analogi: Jin Soun mendengar "Snow Plum Pill dicuri" dan
        otomatis mengidentifikasi konsep kunci: Snow_Plum_Pill, pencurian,
        Hefei. Ini adalah titik awal — dari sini, semua kenangan
        yang terkait akan diaktifkan.

        Args:
            text: The trigger text to parse.

        Returns:
            A ReasoningStep with identified concepts and their nodes.
        """
        concepts: list[str] = []
        evidence_nodes: list[str] = []
        confidence = 0.5

        if self.rsvs_available:
            # Use relate() to find concepts connected to the trigger
            try:
                relate_result = self._bridge.relate(text)
                if relate_result:
                    parsed = self._parse_concept_list(relate_result)
                    concepts.extend(parsed["labels"])
                    evidence_nodes.extend(parsed["labels"])
                    confidence = 0.7
            except Exception as exc:
                logger.debug("relate() failed in trigger step: %s", exc)

            # Also try query() for direct concept lookup
            try:
                query_result = self._bridge.query(text, context="trigger")
                if query_result:
                    parsed = self._parse_concept_list(query_result)
                    for label in parsed["labels"]:
                        if label not in concepts:
                            concepts.append(label)
                            evidence_nodes.append(label)
                    confidence = min(1.0, confidence + 0.1)
            except Exception as exc:
                logger.debug("query() failed in trigger step: %s", exc)

        # Fallback — keyword extraction
        if not concepts:
            concepts = self._fallback_extract_concepts(text)
            evidence_nodes = list(concepts)
            confidence = 0.4

        # Always include the trigger text itself as a concept
        trigger_concept = text.strip()[:50]  # Truncate long triggers
        if trigger_concept not in concepts:
            concepts.insert(0, trigger_concept)

        return ReasoningStep(
            step_type="trigger",
            description=(
                f"Parsed trigger '{text[:80]}' and identified "
                f"{len(concepts)} key concept(s): {', '.join(concepts[:5])}"
            ),
            data={
                "concepts": concepts,
                "trigger_text": text,
            },
            evidence_nodes=evidence_nodes,
            confidence=confidence,
        )

    # ==================================================================
    # Step 2: RECALL MASSAL
    # ==================================================================

    def _recall(self, concepts: list[str]) -> ReasoningStep:
        """Step 2: Spreading activation from trigger concepts.

        For each concept, call relate() to find all connected nodes.
        Then do a second-order expansion: relate() on the most relevant
        results. This creates a "wave" of activation spreading outward
        from the trigger — like how Jin Soun's memory doesn't just
        recall one fact, but a whole web of related knowledge.

        Analogi: Jin Soun mendengar "racun" → mengingat Snow Plum Pill →
        mengingat Hefei → mengingat merchant guild → mengingat auction.
        Setiap kenangan mengaktifkan kenangan lain — spreading activation,
        bukan linear search.

        Args:
            concepts: The trigger concepts from Step 1.

        Returns:
            A ReasoningStep with all activated nodes and edges.
        """
        activated_nodes: list[str] = []
        activated_edges: list[dict] = []
        evidence_nodes: list[str] = []
        seen: set[str] = set()
        confidence = 0.5

        # First-order expansion: relate() from each trigger concept
        for concept in concepts:
            if concept in seen:
                continue
            seen.add(concept)
            activated_nodes.append(concept)
            evidence_nodes.append(concept)

            if self.rsvs_available:
                try:
                    relate_result = self._bridge.relate(concept)
                    if relate_result:
                        parsed = self._parse_relate_result(relate_result)
                        for node_label in parsed["labels"]:
                            if node_label not in seen:
                                seen.add(node_label)
                                activated_nodes.append(node_label)
                                evidence_nodes.append(node_label)
                        for edge in parsed["edges"]:
                            activated_edges.append(edge)
                        confidence = max(confidence, 0.6)
                except Exception as exc:
                    logger.debug("relate() failed for '%s': %s", concept, exc)
            else:
                # Fallback — use internal graph
                related = self._fallback_relate(concept)
                for node_label in related:
                    if node_label not in seen:
                        seen.add(node_label)
                        activated_nodes.append(node_label)
                        evidence_nodes.append(node_label)
                confidence = 0.4

        # Second-order expansion: relate() from the most relevant first-order nodes
        # Analogi: Jin Soun tidak berhenti di kenangan pertama —
        # dia menggali lebih dalam ke kenangan yang paling relevan.
        second_order_candidates = activated_nodes[len(concepts):_MAX_RECALL_DEPTH * 5]
        for node in second_order_candidates:
            if self.rsvs_available:
                try:
                    relate_result = self._bridge.relate(node)
                    if relate_result:
                        parsed = self._parse_relate_result(relate_result)
                        for node_label in parsed["labels"][:5]:  # Limit expansion
                            if node_label not in seen:
                                seen.add(node_label)
                                activated_nodes.append(node_label)
                                evidence_nodes.append(node_label)
                except Exception:
                    pass
            else:
                related = self._fallback_relate(node)
                for node_label in related[:5]:
                    if node_label not in seen:
                        seen.add(node_label)
                        activated_nodes.append(node_label)
                        evidence_nodes.append(node_label)

        return ReasoningStep(
            step_type="recall",
            description=(
                f"Spreading activation from {len(concepts)} trigger concept(s) "
                f"activated {len(activated_nodes)} nodes and "
                f"{len(activated_edges)} edges."
            ),
            data={
                "activated_nodes": activated_nodes,
                "activated_edges": activated_edges,
                "depth": 2,
            },
            evidence_nodes=evidence_nodes,
            confidence=confidence,
        )

    # ==================================================================
    # Step 3: CROSS-REFERENCE
    # ==================================================================

    def _cross_reference(self, activated_nodes: list) -> ReasoningStep:
        """Step 3: Compare activated nodes for shared structure.

        For pairs of activated nodes, compute structural_similarity().
        Find shared compositions across seemingly unrelated nodes.
        Identify temporal/spatial/actor overlaps.

        This is where the "pattern" starts to emerge — by finding
        hidden connections between nodes that seem unrelated on the surface.

        Analogi: Jin Soun membandingkan "Snow Plum Pill" dengan
        "racun biasa" — dan menemukan bahwa komposisinya TIDAK sama.
        Ini bukan kecurangan biasa, ini pola baru. Cross-reference
        mengungkap apa yang tersembunyi di balik permukaan.

        Args:
            activated_nodes: The nodes activated by the recall step.

        Returns:
            A ReasoningStep with shared compositions and similar pairs.
        """
        shared_compositions: list[dict] = []
        similar_pairs: list[dict] = []
        evidence_nodes: list[str] = []
        confidence = 0.5

        # Limit the number of pairs to compare (O(n²) could be expensive)
        nodes_to_compare = activated_nodes[:_MAX_CROSS_REF_PAIRS]

        if self.rsvs_available:
            for i, node_a in enumerate(nodes_to_compare):
                for node_b in nodes_to_compare[i + 1:]:
                    # Compute structural similarity
                    try:
                        sim = self._bridge.structural_similarity(node_a, node_b)
                        sim_value = self._parse_similarity(sim)

                        if sim_value >= _MIN_SIMILARITY_FOR_XREF:
                            similar_pairs.append({
                                "node_a": node_a,
                                "node_b": node_b,
                                "similarity": sim_value,
                            })
                            evidence_nodes.extend([node_a, node_b])
                    except Exception as exc:
                        logger.debug(
                            "structural_similarity('%s', '%s') failed: %s",
                            node_a, node_b, exc
                        )

                    # Also check for shared compositions via senses()
                    try:
                        senses_a = self._bridge.senses(node_a)
                        senses_b = self._bridge.senses(node_b)
                        comps_a = self._parse_composition_list(senses_a)
                        comps_b = self._parse_composition_list(senses_b)
                        shared = set(comps_a) & set(comps_b)
                        if shared:
                            shared_compositions.append({
                                "node_a": node_a,
                                "node_b": node_b,
                                "shared": list(shared),
                            })
                    except Exception:
                        pass

            if similar_pairs or shared_compositions:
                confidence = 0.7
        else:
            # Fallback — simple string similarity between node labels
            for i, node_a in enumerate(nodes_to_compare):
                for node_b in nodes_to_compare[i + 1:]:
                    sim_value = self._fallback_similarity(node_a, node_b)
                    if sim_value >= _MIN_SIMILARITY_FOR_XREF:
                        similar_pairs.append({
                            "node_a": node_a,
                            "node_b": node_b,
                            "similarity": sim_value,
                        })

                    # Check fallback graph for shared compositions
                    comps_a = self._fallback_get_compositions(node_a)
                    comps_b = self._fallback_get_compositions(node_b)
                    shared = set(comps_a) & set(comps_b)
                    if shared:
                        shared_compositions.append({
                            "node_a": node_a,
                            "node_b": node_b,
                            "shared": list(shared),
                        })

            if similar_pairs or shared_compositions:
                confidence = 0.4

        return ReasoningStep(
            step_type="cross_reference",
            description=(
                f"Compared {len(nodes_to_compare)} nodes: found "
                f"{len(similar_pairs)} similar pairs and "
                f"{len(shared_compositions)} shared composition groups."
            ),
            data={
                "shared_compositions": shared_compositions,
                "similar_pairs": similar_pairs,
            },
            evidence_nodes=list(dict.fromkeys(evidence_nodes)),
            confidence=confidence,
        )

    # ==================================================================
    # Step 4: ANOMALY DETECTION
    # ==================================================================

    def _detect_anomalies(self, cross_ref_data: dict) -> ReasoningStep:
        """Step 4: Check for contradictions between expected and observed.

        Uses appraise() on statements about activated nodes.
        If appraise returns "disagree" → anomaly found.
        Also checks for prediction errors from cross-reference results.

        Analogi: Jin Soun memeriksa — "Jika ini pil biasa, seharusnya
        komposisinya A, B, C. Tapi komposisinya X, Y, Z. TIDAK COCOK."
        appraise() = pengecekan konsistensi internal. Anomali = benang merah.

        Args:
            cross_ref_data: The data dict from the cross-reference step.

        Returns:
            A ReasoningStep with detected anomalies.
        """
        anomalies: list[dict] = []
        evidence_nodes: list[str] = []
        confidence = 0.5

        # Check similar pairs for contradictions
        similar_pairs = cross_ref_data.get("similar_pairs", [])
        for pair in similar_pairs[:10]:
            node_a = pair.get("node_a", "")
            node_b = pair.get("node_b", "")
            similarity = pair.get("similarity", 0.0)

            # If two nodes are very similar but have contradictory properties
            if similarity > 0.5 and self.rsvs_available:
                # Build a statement and check with appraise()
                statement = (
                    f"{node_a} and {node_b} are the same type of thing "
                    f"(similarity: {similarity:.2f})"
                )
                try:
                    appraise_result = self._bridge.appraise(statement)
                    if self._is_appraise_negative(appraise_result):
                        anomalies.append({
                            "type": "contradiction",
                            "nodes": [node_a, node_b],
                            "similarity": similarity,
                            "statement": statement,
                            "description": (
                                f"Nodes '{node_a}' and '{node_b}' are structurally "
                                f"similar ({similarity:.2f}) but contradict each other."
                            ),
                        })
                        evidence_nodes.extend([node_a, node_b])
                except Exception as exc:
                    logger.debug("appraise() failed for '%s': %s", statement, exc)

        # Check shared compositions for anomalies
        shared_compositions = cross_ref_data.get("shared_compositions", [])
        for group in shared_compositions[:10]:
            node_a = group.get("node_a", "")
            node_b = group.get("node_b", "")
            shared = group.get("shared", [])

            # If nodes share compositions but shouldn't (based on context)
            if shared and self.rsvs_available:
                statement = (
                    f"{node_a} and {node_b} should not share compositions: "
                    f"{', '.join(shared[:3])}"
                )
                try:
                    appraise_result = self._bridge.appraise(statement)
                    if self._is_appraise_negative(appraise_result):
                        anomalies.append({
                            "type": "unexpected_overlap",
                            "nodes": [node_a, node_b],
                            "shared_compositions": shared,
                            "statement": statement,
                            "description": (
                                f"Unexpected overlap: '{node_a}' and '{node_b}' "
                                f"share {', '.join(shared[:3])} but shouldn't."
                            ),
                        })
                        evidence_nodes.extend([node_a, node_b])
                except Exception:
                    pass

        # Fallback: if no anomalies found via appraise, check for structural gaps
        if not anomalies and self.rsvs_available:
            try:
                # Use substitution_analysis to find structural differences
                for pair in similar_pairs[:5]:
                    node_a = pair.get("node_a", "")
                    node_b = pair.get("node_b", "")
                    if node_a and node_b:
                        try:
                            sub_result = self._bridge.substitution_analysis(node_a, node_b)
                            if sub_result:
                                parsed = self._parse_substitution(sub_result)
                                if parsed.get("has_substitution", False):
                                    anomalies.append({
                                        "type": "substitution",
                                        "nodes": [node_a, node_b],
                                        "substitution": parsed,
                                        "description": (
                                            f"Substitution detected between "
                                            f"'{node_a}' and '{node_b}': "
                                            f"{parsed.get('description', 'unknown')}"
                                        ),
                                    })
                                    evidence_nodes.extend([node_a, node_b])
                        except Exception as exc:
                            logger.debug(
                                "substitution_analysis() failed: %s", exc
                            )
            except Exception as exc:
                logger.warning("Anomaly detection fallback failed: %s", exc)

        if anomalies:
            confidence = 0.7
        else:
            confidence = 0.4  # No anomalies = low confidence in detection

        return ReasoningStep(
            step_type="anomaly",
            description=(
                f"Anomaly detection found {len(anomalies)} anomaly(ies)."
                + (" " + "; ".join(a["description"][:60] for a in anomalies[:3]) if anomalies else "")
            ),
            data={
                "anomalies": anomalies,
                "checked_pairs": len(similar_pairs),
            },
            evidence_nodes=list(dict.fromkeys(evidence_nodes)),
            confidence=confidence,
        )

    # ==================================================================
    # Step 5: PATTERN COMPLETION
    # ==================================================================

    def _complete_pattern(
        self,
        recall_data: dict,
        anomalies: list[dict],
        cross_ref: dict,
    ) -> ReasoningStep:
        """Step 5: Connect the fragments into a coherent pattern.

        From anomalies and cross-references, compose a pattern.
        Uses substitution_analysis() to find what transforms A → B.
        Builds the pattern as a structured claim.

        Analogi: Jin Soun menyatukan semua petunjuk —
        "Pil palsu, motif uang, kambing hitam" → POLA:
        "Seseorang membuat versi palsu dari Snow Plum Pill,
        menggunakan Ju Jangmok sebagai kambing hitam untuk
        menutupi operasi pemalsuan."

        Args:
            recall_data: Data from the recall step.
            anomalies: Anomalies from the anomaly detection step.
            cross_ref: Data from the cross-reference step.

        Returns:
            A ReasoningStep with the completed pattern.
        """
        pattern_parts: list[str] = []
        evidence_nodes: list[str] = []
        substitutions: list[dict] = []
        confidence = 0.4

        activated_nodes = recall_data.get("activated_nodes", [])

        # Strategy 1: Build pattern from anomalies
        # Analogi: Setiap anomali adalah potongan puzzle.
        # Pattern completion = menyusun potongan-potongan itu.
        if anomalies:
            for anomaly in anomalies:
                anomaly_type = anomaly.get("type", "unknown")
                nodes = anomaly.get("nodes", [])
                evidence_nodes.extend(nodes)

                if anomaly_type == "contradiction":
                    pattern_parts.append(
                        f"Contradiction found between {' and '.join(nodes)}: "
                        f"they appear similar but have conflicting properties. "
                        f"This suggests a hidden transformation or substitution."
                    )
                elif anomaly_type == "unexpected_overlap":
                    shared = anomaly.get("shared_compositions", [])
                    pattern_parts.append(
                        f"Unexpected shared elements between {' and '.join(nodes)}: "
                        f"{', '.join(shared[:3])}. This overlap may indicate "
                        f"a common origin or deliberate manipulation."
                    )
                elif anomaly_type == "substitution":
                    sub = anomaly.get("substitution", {})
                    pattern_parts.append(
                        f"Substitution detected between {' and '.join(nodes)}: "
                        f"one can be transformed into the other, suggesting "
                        f"a deliberate replacement or forgery."
                    )
                    substitutions.append(sub)

        # Strategy 2: Use substitution_analysis() for deeper patterns
        if self.rsvs_available:
            similar_pairs = cross_ref.get("similar_pairs", [])
            for pair in similar_pairs[:5]:
                node_a = pair.get("node_a", "")
                node_b = pair.get("node_b", "")
                try:
                    sub_result = self._bridge.substitution_analysis(node_a, node_b)
                    if sub_result:
                        parsed = self._parse_substitution(sub_result)
                        if parsed.get("has_substitution", False):
                            substitutions.append(parsed)
                            evidence_nodes.extend([node_a, node_b])
                            pattern_parts.append(
                                f"Substitution path from '{node_a}' to '{node_b}': "
                                f"{parsed.get('description', 'transformation detected')}"
                            )
                except Exception as exc:
                    logger.debug("substitution_analysis() failed: %s", exc)

        # Strategy 3: Use compose() to formalize the pattern
        if self.rsvs_available and pattern_parts:
            try:
                # Compose a pattern label from the evidence
                pattern_label = "pattern_" + "_".join(
                    activated_nodes[:3] if activated_nodes else ["unknown"]
                ).replace(" ", "_")[:50]

                composition_texts = [p[:200] for p in pattern_parts[:5]]
                # BUG FIX: RSVS compose() expects compositions: Vec<(String, u32)>
                # — a list of (node_label, sense_id) tuples, NOT a list of strings.
                comp_tuples = [(text, 0) for text in composition_texts]  # sense_id=0 for default
                compose_result = self._bridge.compose(pattern_label, comp_tuples, lang="en")
                if compose_result:
                    # The compose result may give us a formal representation
                    composed = self._parse_compose_result(compose_result)
                    if composed:
                        pattern_parts.insert(0, composed)
                        confidence = 0.8
            except Exception as exc:
                logger.debug("compose() failed: %s", exc)

        # Build the final pattern description
        if pattern_parts:
            pattern = "\n\n".join(f"• {part}" for part in pattern_parts)
            # Add a summary line
            anomaly_count = len(anomalies)
            node_count = len(activated_nodes)
            pattern = (
                f"[Pattern from {node_count} nodes, {anomaly_count} anomalies]\n"
                + pattern
            )
            confidence = min(0.9, 0.5 + 0.1 * len(anomalies))
        else:
            # No anomalies — the pattern is just "everything is consistent"
            if activated_nodes:
                pattern = (
                    f"[No anomalies found among {len(activated_nodes)} nodes. "
                    f"The activated concepts appear consistent.]"
                )
            else:
                pattern = "[No pattern could be formed — insufficient data.]"
            confidence = 0.3

        return ReasoningStep(
            step_type="pattern",
            description=(
                f"Pattern completion: {'anomaly-driven' if anomalies else 'consistent'} "
                f"pattern from {len(activated_nodes)} nodes and "
                f"{len(anomalies)} anomalies."
            ),
            data={
                "pattern": pattern,
                "substitutions": substitutions,
                "pattern_parts": pattern_parts,
            },
            evidence_nodes=list(dict.fromkeys(evidence_nodes)),
            confidence=confidence,
        )

    # ==================================================================
    # Step 6: NARRATIVE OUTPUT
    # ==================================================================

    def _generate_narrative(
        self, pattern: str, evidence: list[str],
        trigger: str = "", steps: list[ReasoningStep] | None = None,
    ) -> ReasoningStep:
        """Step 6: Generate traceable narrative text from the reasoning chain.

        Builds the narrative from the reasoning chain. Each claim has
        evidence_nodes that point back to the graph. Confidence =
        average grounding of referenced nodes.

        KEY INSIGHT: The LLM doesn't generate from nothing. It generates
        FROM the graph. Graph = structural memory, LLM = narrative voice.

        In the future, an LLM will generate the narrative FROM the
        structured reasoning chain. For now, we build a structured
        narrative that reads like a Jin Soun investigation report —
        each claim traced to evidence, each step documented.

        Analogi: Jin Soun mengungkapkan kesimpulannya — bukan dengan
        mengarang, tapi dengan menelusuri rantai bukti. Setiap klaim
        bisa ditelusuri kembali ke node di graf kenangannya.
        "Aku menyimpulkan X karena A → B → C → X."

        Args:
            pattern: The completed pattern description from Step 5.
            evidence: List of evidence node labels.
            trigger: The original trigger text.
            steps: The full reasoning chain (for structured narrative).

        Returns:
            A ReasoningStep with the generated narrative.
        """
        steps = steps or []

        # Compute confidence from evidence grounding
        evidence_confidence = 0.5
        if self.rsvs_available and evidence:
            try:
                cmap = self._bridge.confidence_map()
                confidences = [cmap.get(n, 0.3) for n in evidence if n in cmap]
                if confidences:
                    evidence_confidence = sum(confidences) / len(confidences)
            except Exception:
                pass

        # Build structured narrative — like a Jin Soun investigation report
        # Analogi: Jin Soun menyusun laporan investigasi yang bisa diaudit.
        # Setiap langkah punya footnote yang merujuk ke bukti spesifik.
        narrative_parts: list[str] = []

        # === Section 1: Trigger Summary ===
        if trigger:
            narrative_parts.append(
                f"## Trigger\n\nInput: \"{trigger}\"\n\n"
                f"This triggered a pattern completion analysis across the knowledge graph."
            )

        # === Section 2: Reasoning Chain ===
        if steps:
            chain_lines: list[str] = ["## Reasoning Chain"]
            for i, step in enumerate(steps[:-1], 1):  # Exclude this narrative step
                step_emoji = {
                    "trigger": "🎯",
                    "recall": "🔍",
                    "cross_reference": "🔗",
                    "anomaly": "⚠️",
                    "pattern": "🧩",
                }.get(step.step_type, "📋")

                chain_lines.append(
                    f"\n**Step {i}: {step.step_type.upper()}** {step_emoji}\n"
                    f"{step.description}\n"
                    f"Confidence: {step.confidence:.0%}"
                )
                if step.evidence_nodes:
                    chain_lines.append(
                        f"Evidence: {', '.join(step.evidence_nodes[:8])}"
                    )
            narrative_parts.append("\n".join(chain_lines))

        # === Section 3: Pattern Description ===
        if pattern:
            narrative_parts.append(f"## Pattern\n\n{pattern}")

        # === Section 4: Evidence Summary ===
        if evidence:
            unique_evidence = list(dict.fromkeys(evidence))
            evidence_str = ", ".join(unique_evidence[:15])
            narrative_parts.append(
                f"## Evidence\n\n"
                f"Grounded in {len(unique_evidence)} knowledge node(s): {evidence_str}."
            )

        # === Section 5: Confidence Assessment ===
        confidence_desc = "low"
        if evidence_confidence >= 0.7:
            confidence_desc = "high"
        elif evidence_confidence >= 0.4:
            confidence_desc = "moderate"

        narrative_parts.append(
            f"## Confidence\n\n"
            f"Overall: **{evidence_confidence:.0%}** ({confidence_desc}).\n\n"
            f"Each claim in this narrative can be traced back to specific nodes "
            f"in the knowledge graph. This is not probabilistic text generation — "
            f"it is structured reasoning from graph evidence."
        )

        narrative = "\n\n".join(narrative_parts)

        return ReasoningStep(
            step_type="narrative",
            description=(
                f"Generated structured narrative ({len(narrative)} chars) from "
                f"{len(evidence)} evidence nodes and {len(steps)} reasoning steps."
            ),
            data={
                "narrative": narrative,
                "evidence_count": len(evidence),
                "evidence_confidence": evidence_confidence,
            },
            evidence_nodes=evidence,
            confidence=evidence_confidence,
        )

    # ==================================================================
    # Internal: Evidence chain builder
    # ==================================================================

    def _build_evidence_chain(self, steps: list[ReasoningStep]) -> list[dict]:
        """Build a traceable evidence chain from the reasoning steps.

        Creates a list of evidence entries, each linking a claim back
        to the graph nodes that support it.

        Analogi: Footnote di laporan Jin Soun — setiap klaim punya
        referensi ke halaman dan baris di Simhyeon Pavilion.

        Args:
            steps: The reasoning steps.

        Returns:
            A list of evidence dicts with node, step, and claim info.
        """
        chain: list[dict] = []
        for step in steps:
            for node in step.evidence_nodes:
                # Get node info if available
                node_info: dict = {"label": node}
                if self.rsvs_available:
                    try:
                        info = self._bridge.node_info(node)
                        if info:
                            node_info.update(self._parse_node_info(info))
                    except Exception:
                        pass
                else:
                    node_info.update(self._fallback_get_node_info(node))

                chain.append({
                    "node": node_info,
                    "step_type": step.step_type,
                    "claim": step.description[:200],
                    "confidence": step.confidence,
                })

        return chain

    # ==================================================================
    # Internal: RSVS result parsers
    # ==================================================================

    @staticmethod
    def _parse_concept_list(result: Any) -> dict:
        """Parse a concept list from RSVS relate()/query() results.

        Handles dicts, lists, strings, and PyO3 objects.
        The bridge returns dicts with "related_nodes" key containing
        (label, score) tuples.

        Args:
            result: The result from an RSVS bridge method.

        Returns:
            A dict with "labels" (list of str).
        """
        labels: list[str] = []

        if result is None:
            return {"labels": labels}

        if isinstance(result, list):
            for item in result:
                if isinstance(item, str):
                    labels.append(item)
                elif isinstance(item, dict):
                    label = item.get("label", item.get("concept", ""))
                    if label:
                        labels.append(str(label))
                elif hasattr(item, "label"):
                    label = getattr(item, "label", None)
                    if label:
                        labels.append(str(label))
            return {"labels": labels}

        if isinstance(result, dict):
            # Bridge relate() returns "related_nodes" with (label, score) tuples
            for key in ("related_nodes", "related", "nodes", "atoms", "concepts", "results"):
                items = result.get(key, [])
                if items:
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                labels.append(item)
                            elif isinstance(item, tuple) and len(item) >= 1:
                                # (label, score) or (node_id, score) tuple
                                labels.append(str(item[0]))
                            elif isinstance(item, dict):
                                label = item.get("label", "")
                                if label:
                                    labels.append(str(label))
                            elif hasattr(item, "label"):
                                labels.append(str(getattr(item, "label", "")))
            # Single concept result
            if not labels:
                label = result.get("label", result.get("concept", ""))
                if label:
                    labels.append(str(label))
            return {"labels": labels}

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return PatternOutput._parse_concept_list(parsed)
            except (json.JSONDecodeError, ValueError):
                return {"labels": [result] if result.strip() else []}

        # PyO3 object fallback
        for attr_name in ("related", "nodes", "atoms", "concepts"):
            try:
                items = getattr(result, attr_name, None)
                if items is not None:
                    for item in items:
                        if isinstance(item, str):
                            labels.append(item)
                        elif hasattr(item, "label"):
                            labels.append(str(getattr(item, "label", "")))
                    break
            except Exception:
                pass

        return {"labels": labels}

    @staticmethod
    def _parse_relate_result(result: Any) -> dict:
        """Parse a relate() result into labels and edges.

        The bridge returns a dict with "related_nodes" key containing
        (label, score) tuples and optional "related_edges" / "structural_relations".

        Args:
            result: The result from RsvsBridge.relate().

        Returns:
            A dict with "labels" (list of str) and "edges" (list of dict).
        """
        labels: list[str] = []
        edges: list[dict] = []

        if result is None:
            return {"labels": labels, "edges": edges}

        if isinstance(result, dict):
            # Bridge returns "related_nodes" with (label, score) tuples
            for key in ("related_nodes", "related", "nodes"):
                items = result.get(key, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            labels.append(item)
                        elif isinstance(item, tuple) and len(item) >= 1:
                            # (label_or_id, score) tuple — extract label
                            labels.append(str(item[0]))
                        elif isinstance(item, dict):
                            label = item.get("label", "")
                            if label:
                                labels.append(str(label))
                            # Try to extract edge info
                            edge_type = item.get("relation", item.get("edge_type", ""))
                            target = item.get("target", item.get("to", ""))
                            if edge_type and target:
                                edges.append({
                                    "type": edge_type,
                                    "target": target,
                                    "source": item.get("source", item.get("from", "")),
                                })

            raw_edges = result.get("edges", [])
            if isinstance(raw_edges, list):
                for edge in raw_edges:
                    if isinstance(edge, dict):
                        edges.append(edge)

            # Also check "structural_relations"
            struct_rels = result.get("structural_relations", [])
            if isinstance(struct_rels, list):
                for item in struct_rels:
                    if isinstance(item, tuple) and len(item) >= 1:
                        label = str(item[0])
                        if label not in labels:
                            labels.append(label)
                    elif isinstance(item, str) and item not in labels:
                        labels.append(item)

            return {"labels": labels, "edges": edges}

        # Try PyO3 or list
        try:
            items = list(result) if not isinstance(result, str) else []
            for item in items:
                if isinstance(item, str):
                    labels.append(item)
                elif isinstance(item, tuple) and len(item) >= 1:
                    labels.append(str(item[0]))
                elif hasattr(item, "label"):
                    labels.append(str(getattr(item, "label", "")))
        except (TypeError, ValueError):
            pass

        return {"labels": labels, "edges": edges}

    @staticmethod
    def _parse_similarity(result: Any) -> float:
        """Parse a similarity value from structural_similarity().

        The bridge normalizes PyO3 StructuralSimResult to a plain dict
        with key "structural_similarity".

        Args:
            result: The result from RsvsBridge.structural_similarity().

        Returns:
            A float between 0.0 and 1.0.
        """
        if isinstance(result, dict):
            return float(result.get("structural_similarity", result.get("similarity", result.get("score", 0.0))))

        if isinstance(result, (int, float)):
            return float(result)

        if isinstance(result, str):
            try:
                return float(result)
            except ValueError:
                return 0.0

        if hasattr(result, "structural_similarity"):
            return float(result.structural_similarity)

        # Fallback — try other common attributes
        try:
            for attr in ("similarity", "score", "value"):
                val = getattr(result, attr, None)
                if val is not None:
                    return float(val)
        except Exception:
            pass

        return 0.0

    @staticmethod
    def _parse_composition_list(result: Any) -> list[str]:
        """Parse compositions from senses() or similar results.

        The bridge senses() returns a list of dicts, each with a
        "compositions" key containing (label, sense_id) tuples.

        Args:
            result: The result from RsvsBridge.senses().

        Returns:
            A list of composition label strings.
        """
        if result is None:
            return []

        if isinstance(result, list):
            compositions: list[str] = []
            for item in result:
                if isinstance(item, str):
                    compositions.append(item)
                elif isinstance(item, dict):
                    # Bridge sense dict has "compositions" key with tuples
                    for comp in item.get("compositions", []):
                        if isinstance(comp, tuple) and len(comp) >= 1:
                            compositions.append(str(comp[0]))  # (label, sense_id) → label
                        elif isinstance(comp, str):
                            compositions.append(comp)
                    # Also try direct label/composition keys
                    if not item.get("compositions"):
                        label = item.get("label", item.get("composition", ""))
                        if label:
                            compositions.append(str(label))
                elif hasattr(item, "label"):
                    compositions.append(str(getattr(item, "label", "")))
            return compositions

        if isinstance(result, dict):
            for key in ("compositions", "senses", "atoms"):
                items = result.get(key, [])
                if items:
                    return PatternOutput._parse_composition_list(items)
            return []

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return PatternOutput._parse_composition_list(parsed)
            except (json.JSONDecodeError, ValueError):
                return [result] if result.strip() else []

        # PyO3 object — try iterating
        try:
            compositions = []
            for item in result:
                if isinstance(item, str):
                    compositions.append(item)
                elif isinstance(item, dict):
                    for comp in item.get("compositions", []):
                        if isinstance(comp, tuple) and len(comp) >= 1:
                            compositions.append(str(comp[0]))
                        elif isinstance(comp, str):
                            compositions.append(comp)
                elif hasattr(item, "label"):
                    compositions.append(str(getattr(item, "label", "")))
            return compositions
        except (TypeError, AttributeError):
            return []

    @staticmethod
    def _parse_substitution(result: Any) -> dict:
        """Parse a substitution_analysis() result.

        Args:
            result: The result from RsvsBridge.substitution_analysis().

        Returns:
            A dict with at least "has_substitution" (bool) and
            optionally "description", "transformations".
        """
        if result is None:
            return {"has_substitution": False}

        if isinstance(result, dict):
            return {
                "has_substitution": bool(result.get("has_substitution", result.get("found", False))),
                "description": str(result.get("description", result.get("transform", ""))),
                "transformations": result.get("transformations", result.get("steps", result.get("substitutions", []))),
                "score": float(result.get("score", result.get("confidence", result.get("structural_similarity", 0.0)))),
            }

        if isinstance(result, bool):
            return {"has_substitution": result}

        if isinstance(result, (int, float)):
            return {
                "has_substitution": float(result) > 0.5,
                "score": float(result),
            }

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return PatternOutput._parse_substitution(parsed)
            except (json.JSONDecodeError, ValueError):
                return {
                    "has_substitution": bool(result.strip()),
                    "description": result,
                }

        # PyO3 object
        try:
            has_sub = getattr(result, "has_substitution", None)
            if has_sub is None:
                has_sub = getattr(result, "found", False)
            return {
                "has_substitution": bool(has_sub),
                "description": str(getattr(result, "description", "")),
                "score": float(getattr(result, "score", 0.0)),
            }
        except Exception:
            return {"has_substitution": False}

    @staticmethod
    def _parse_compose_result(result: Any) -> str:
        """Parse a compose() result into a string.

        Args:
            result: The result from RsvsBridge.compose().

        Returns:
            A string representation of the composed result.
        """
        if result is None:
            return ""

        if isinstance(result, str):
            return result

        if isinstance(result, int):
            # compose() returns a node ID (int) — use it as confirmation
            return f"[Composed node ID: {result}]"

        if isinstance(result, dict):
            return str(result.get("text", result.get("result", result.get("composed", ""))))

        # PyO3 object
        try:
            for attr in ("text", "result", "composed", "output"):
                val = getattr(result, attr, None)
                if val is not None:
                    return str(val)
        except Exception:
            pass

        return str(result) if result else ""

    @staticmethod
    def _parse_node_info(result: Any) -> dict:
        """Parse a node_info() result into a dict.

        Args:
            result: The result from RsvsBridge.node_info().

        Returns:
            A dict with node metadata.
        """
        if result is None:
            return {}

        if isinstance(result, dict):
            return result

        try:
            return {
                k: getattr(result, k)
                for k in dir(result)
                if not k.startswith("_") and not callable(getattr(result, k))
            }
        except Exception:
            return {"raw": str(result)}

    # ==================================================================
    # Internal: Appraise result parsing
    # ==================================================================

    @staticmethod
    def _is_appraise_negative(result: Any) -> bool:
        """Check if an appraise() result indicates disagreement.

        The bridge returns a plain dict with "verdict" key and
        "disagree_pct" / "agree_pct" keys. We check the verdict
        and also consider it negative if disagree_pct > agree_pct.

        Args:
            result: The result from RsvsBridge.appraise().

        Returns:
            True if the result indicates disagreement.
        """
        negative_indicators = {"disagree", "contradiction", "false", "negative", "reject"}

        if isinstance(result, str):
            return result.lower().strip() in negative_indicators

        if isinstance(result, dict):
            # Check verdict key first
            verdict = result.get("verdict", result.get("result", ""))
            if isinstance(verdict, str):
                if verdict.lower().strip() in negative_indicators:
                    return True
            elif isinstance(verdict, bool):
                if not verdict:
                    return True
            elif isinstance(verdict, (int, float)):
                if float(verdict) < 0.0:
                    return True

            # Also check disagree_pct > agree_pct as a secondary signal
            disagree_pct = result.get("disagree_pct", 0.0)
            agree_pct = result.get("agree_pct", 0.0)
            if isinstance(disagree_pct, (int, float)) and isinstance(agree_pct, (int, float)):
                if float(disagree_pct) > float(agree_pct):
                    return True

        # PyO3 object fallback
        try:
            verdict = getattr(result, "verdict", None)
            if verdict is None:
                verdict = getattr(result, "result", None)
            if isinstance(verdict, str):
                return verdict.lower().strip() in negative_indicators
            if isinstance(verdict, bool):
                return not verdict
            if isinstance(verdict, (int, float)):
                return float(verdict) < 0.0
        except Exception:
            pass

        return False

    # ==================================================================
    # Internal: Fallback helpers (when RSVS is unavailable)
    # ==================================================================

    def _fallback_extract_concepts(self, text: str) -> list[str]:
        """Extract concepts from text using keyword extraction.

        Fallback for when RSVS relate() is unavailable.

        Args:
            text: Input text.

        Returns:
            A list of concept labels.
        """
        words = text.lower().split()
        concepts: list[str] = []
        for w in words:
            w = w.strip(".,;:!?\"'()[]{}")
            if len(w) > 3 and w not in _STOP_WORDS:
                concepts.append(w)
        return concepts[:10]

    def _fallback_relate(self, concept: str) -> list[str]:
        """Find related concepts using the fallback graph.

        Args:
            concept: The concept to find relations for.

        Returns:
            A list of related concept labels.
        """
        related: list[str] = []

        # Direct lookup in fallback graph
        if concept in self._fallback_graph:
            entry = self._fallback_graph[concept]
            related.extend(entry.get("compositions", []))
            related.extend(entry.get("relations", []))

        # Fuzzy match — check if any graph key contains the concept
        concept_lower = concept.lower()
        for key in self._fallback_graph:
            if concept_lower in key.lower() or key.lower() in concept_lower:
                if key != concept and key not in related:
                    entry = self._fallback_graph[key]
                    related.append(key)
                    related.extend(
                        c for c in entry.get("compositions", [])
                        if c not in related
                    )

        return list(dict.fromkeys(related))[:20]  # Deduplicate, limit

    def _fallback_get_compositions(self, concept: str) -> list[str]:
        """Get compositions for a concept from the fallback graph.

        Args:
            concept: The concept to look up.

        Returns:
            A list of composition labels.
        """
        if concept in self._fallback_graph:
            return self._fallback_graph[concept].get("compositions", [])
        return []

    def _fallback_get_node_info(self, concept: str) -> dict:
        """Get node info from the fallback graph.

        Args:
            concept: The concept to look up.

        Returns:
            A dict with node metadata.
        """
        if concept in self._fallback_graph:
            return dict(self._fallback_graph[concept])
        return {"label": concept, "source": "fallback"}

    @staticmethod
    def _fallback_similarity(a: str, b: str) -> float:
        """Compute simple string similarity between two concept labels.

        Uses Jaccard similarity on word-level tokens.

        Args:
            a: First concept label.
            b: Second concept label.

        Returns:
            A similarity score between 0.0 and 1.0.
        """
        words_a = set(a.lower().replace("_", " ").split())
        words_b = set(b.lower().replace("_", " ").split())

        # Remove stop words
        words_a -= _STOP_WORDS
        words_b -= _STOP_WORDS

        if not words_a or not words_b:
            # Character-level fallback
            if a.lower() == b.lower():
                return 1.0
            shorter, longer = (a.lower(), b.lower()) if len(a) <= len(b) else (b.lower(), a.lower())
            if shorter in longer:
                return len(shorter) / max(len(longer), 1)
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    # ==================================================================
    # Public utilities
    # ==================================================================

    def ingest_for_context(self, text: str) -> None:
        """Ingest text into the fallback graph for future pattern completion.

        When RSVS is unavailable, this builds up the internal fallback
        graph so that pattern completion can still work (with reduced
        fidelity).

        When RSVS is available, this also ingests into the RSVS graph.

        Analogi: Jin Soun mencatat informasi baru di Simhyeon Pavilion
        dan di buku catatan pribadinya — dua tempat penyimpanan
        untuk jaring pengaman.

        Args:
            text: The text to ingest.
        """
        # Always ingest into fallback graph
        atoms = self._fallback_extract_concepts(text)
        for atom in atoms:
            if atom not in self._fallback_graph:
                self._fallback_graph[atom] = {
                    "compositions": [],
                    "relations": [],
                    "confidence": 0.5,
                }
            entry = self._fallback_graph[atom]
            for other in atoms:
                if other != atom:
                    if other not in entry["compositions"]:
                        entry["compositions"].append(other)
                    if other not in entry["relations"]:
                        entry["relations"].append(other)
            entry["confidence"] = min(1.0, entry.get("confidence", 0.5) + 0.05)

        # Also ingest into RSVS via bridge if available
        if self.rsvs_available:
            try:
                self._bridge.ingest(text)
            except Exception as exc:
                logger.warning("RSVS ingest failed: %s", exc)

    def get_history(self) -> list[PatternResult]:
        """Return the history of processed pattern completions.

        Returns:
            A list of PatternResult objects from previous process() calls.
        """
        return list(self._history)

    def reset(self) -> None:
        """Reset all internal state.

        Does NOT reset the RSVS graph itself.

        Analogi: Jin Soun mengosongkan meja kerja untuk kasus baru.
        Simhyeon Pavilion tetap utuh.
        """
        self._fallback_graph = {}
        self._history = []
        logger.info("PatternOutput reset — all internal state cleared")

    def status(self) -> dict:
        """Return a status summary of the pattern output layer.

        Returns:
            A dict with RSVS availability, rust core status, fallback
            graph size, and history count.
        """
        return {
            "rsvs_available": self.rsvs_available,
            "is_rust_core": self.is_rust_core,
            "fallback_graph_size": len(self._fallback_graph),
            "history_count": len(self._history),
        }
