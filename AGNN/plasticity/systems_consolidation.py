"""
SYSTEMS CONSOLIDATION: Hippocampus -> Neocortex transfer.

Biologis: Memories gradually become HPC-independent, stored in neocortex.
AI: Embeddings refine from topology, confidence -> edge weight.

Timeline: Labile (HPC-dependent) -> Stable (neocortical).

Per the AGNN architecture (ARCHITECTURE.md section 11), systems
consolidation is the slow process by which an episodic memory (Episome,
stored in the hippocampus) is gradually transferred into a semantic
memory (Semesome, stored in the neocortex) by replaying the graph and
strengthening the edges that bind related episomes.

This implementation follows the task spec:
    1. Look up the episome's neighbors in the wrapped EngramComplex.
    2. If it has at least one neighbor, pick the strongest edge
       (highest confidence) and build a Semesome from it.
    3. Strengthen the episome by +0.05 (consolidation strengthens
       memory). The mutation is in-place on the episome dataclass and
       is also mirrored onto the graph node so retrieval sees the new
       confidence.
    4. Return the Semesome (or None if the episome had no neighbors).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from engrams.episodic_engram import Episome
from engrams.semantic_engram import Semesome


# How much each consolidation pass strengthens the episome's confidence.
# Per the task spec: "Update episome.confidence += 0.05".
CONSOLIDATION_CONFIDENCE_DELTA = 0.05

# Upper bound on confidence - the architecture's "stable" phase caps at 1.0.
MAX_CONFIDENCE = 1.0


def _resolve_agnn_graph():
    """Lazy import of the AGNNGraph module (mirrors engram_complex.py)."""
    self_ai_src = os.path.join(
        os.path.dirname(__file__), "..", "..", "self-ai", "src"
    )
    self_ai_src = os.path.abspath(self_ai_src)
    if self_ai_src not in sys.path:
        sys.path.insert(0, self_ai_src)
    from agnn import graph as agnn_graph  # noqa: WPS433
    return agnn_graph


class SystemsConsolidation:
    """HPC -> NC transfer mechanism.

    Attributes:
        consolidation_count: Lifetime counter of consolidate() calls
            that actually produced a Semesome (i.e. the episome had
            at least one neighbor). Useful for replay-scheduling audits.
        transfer_log: List of (episome_id, semesome) tuples for every
            successful transfer.
    """

    def __init__(self) -> None:
        """Initialize the consolidation counter and transfer log."""
        self.consolidation_count: int = 0
        self.transfer_log: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(self, episome: Episome, graph) -> Optional[Semesome]:
        """Transfer ``episome`` from hippocampus to neocortical semantic.

        Biologis: Over days/weeks, HPC traces are replayed and NC traces
            are strengthened until HPC is no longer required.
        AI: Find the episome's strongest neighbor edge in the
            EngramComplex, build a Semesome from it, and strengthen the
            episome's confidence by +0.05.

        Args:
            episome: The Episome to consolidate. Must already be
                registered as a node in ``graph`` (the TrisynapticCircuit
                does this during encode()).
            graph: An :class:`EngramComplex` whose wrapped AGNNGraph
                contains the episome as a node plus any autoassociative
                edges CA3 created.

        Returns:
            A :class:`Semesome` representing the strongest relation, or
            None if the episome has no neighbors in the graph (nothing
            to consolidate into a semantic relation).
        """
        # 1. Strengthen the episome - consolidation always strengthens
        #    memory, regardless of whether we produced a Semesome.
        self._strengthen_episome(episome, graph)

        # 2. Find the strongest neighbor edge.
        strongest_edge = self._strongest_edge(episome, graph)
        if strongest_edge is None:
            # No neighbors - nothing to transfer to a Semesome. The
            # episome is still strengthened (above), but we return None
            # so the caller knows no semantic relation was produced.
            return None

        # 3. Build a Semesome from the strongest edge.
        semesome = self._build_semesome(strongest_edge, graph)
        if semesome is not None:
            self.consolidation_count += 1
            self.transfer_log.append((episome.id, semesome))
        return semesome

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strengthen_episome(self, episome: Episome, graph) -> None:
        """Strengthen ``episome.confidence`` by +0.05 (capped at 1.0).

        Also mirrors the new confidence onto the graph node so
        PapezCircuit.retrieve() sees the updated value.
        """
        new_conf = min(
            episome.confidence + CONSOLIDATION_CONFIDENCE_DELTA,
            MAX_CONFIDENCE,
        )
        episome.confidence = new_conf

        # Mirror onto the graph node (best-effort).
        try:
            agnn_graph = _resolve_agnn_graph()
        except ImportError:
            return
        inner = getattr(graph, "_graph", None)
        if inner is None:
            return
        node = inner.get_node(str(episome.id))
        if node is not None:
            node.confidence = new_conf

    def _strongest_edge(self, episome: Episome, graph):
        """Return the highest-confidence edge touching ``episome``.

        Considers both outgoing and incoming edges (autoassociation is
        symmetric in semantic terms). Returns None if no edges touch
        the episome.
        """
        try:
            agnn_graph = _resolve_agnn_graph()
        except ImportError:
            return None
        inner = getattr(graph, "_graph", None)
        if inner is None:
            return None
        node_id = str(episome.id)
        if inner.get_node(node_id) is None:
            # Episome not in the graph - cannot have edges.
            return None
        outgoing = inner.get_edges_from(node_id)
        incoming = inner.get_edges_to(node_id)
        candidates = list(outgoing) + list(incoming)
        if not candidates:
            return None
        # Strongest = highest confidence. Ties broken by relation type
        # weight (CATEGORICAL=1.0 > CAUSAL=0.7 > FUNCTIONAL=0.6 > ...).
        rel_weights = {
            agnn_graph.RelationType.CATEGORICAL: 1.0,
            agnn_graph.RelationType.CAUSAL: 0.7,
            agnn_graph.RelationType.FUNCTIONAL: 0.6,
            agnn_graph.RelationType.SPATIAL: 0.5,
            agnn_graph.RelationType.TEMPORAL: 0.3,
            agnn_graph.RelationType.DISCURSIVE: 0.2,
            agnn_graph.RelationType.DIFFERENTIAL: -0.8,
        }
        return max(
            candidates,
            key=lambda e: (e.confidence, rel_weights.get(e.relation_type, 0.0)),
        )

    def _build_semesome(self, edge, graph) -> Optional[Semesome]:
        """Build a Semesome from a TypedEdge.

        - type = edge.relation_type.value.upper() (e.g. "CATEGORICAL").
        - weight = edge.confidence.
        - source / target = labels of the source/target AGNNNodes.
        """
        try:
            agnn_graph = _resolve_agnn_graph()
        except ImportError:
            return None
        inner = getattr(graph, "_graph", None)
        if inner is None:
            return None
        src_node = inner.get_node(edge.source_id)
        dst_node = inner.get_node(edge.target_id)
        if src_node is None or dst_node is None:
            return None
        return Semesome(
            type=edge.relation_type.value.upper(),
            weight=float(edge.confidence),
            source=src_node.label,
            target=dst_node.label,
        )
