"""
PAPEZ CIRCUIT: Episodic memory retrieval loop.

Biologis: HC -> Mamillary Body -> Anterior Thalamus -> Cingulate Gyrus
           -> Parahippocampal Gyrus -> HC.
AI: Retrieval loop that integrates conflict detection + scene recognition.

Per ARCHITECTURE.md section 11, the Papez circuit is the episodic-memory
retrieval loop. In this implementation, retrieval is a keyword-matching
scan over the EngramComplex's nodes (every node = an Episome stored by
the TrisynapticCircuit during encode()). Each node is scored by the
fraction of query keywords that appear in its label, multiplied by the
node's confidence. The top-k scoring nodes are returned as Episome
instances, sorted by descending score (which is essentially descending
confidence when the keyword overlap is comparable).

Pure Python + numpy. No torch.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List

from engrams.episodic_engram import Episome


# Reuse the EC's stop-word logic so retrieval keyword extraction matches
# the keyword extraction used during encoding. This guarantees that a
# query "human" matches an encoded stimulus "Socrates is a human" on the
# same keyword ("human") that was indexed.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "as",
    "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "it", "its", "from", "into", "than", "so", "such", "not", "no",
})


def _resolve_agnn_graph():
    """Lazy import of AGNNGraph (mirrors engram_complex.py)."""
    self_ai_src = os.path.join(
        os.path.dirname(__file__), "..", "..", "self-ai", "src"
    )
    self_ai_src = os.path.abspath(self_ai_src)
    if self_ai_src not in sys.path:
        sys.path.insert(0, self_ai_src)
    from agnn import graph as agnn_graph  # noqa: WPS433
    return agnn_graph


def _extract_keywords(text: str) -> List[str]:
    """Extract normalized keywords from ``text`` (stop-words removed)."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class PapezCircuit:
    """Episodic memory retrieval loop.

    Attributes:
        retrieval_log: List of (query, num_returned) tuples for every
            retrieve() call. Useful for retrieval audits.
    """

    def __init__(self) -> None:
        """Initialize the retrieval log."""
        self.retrieval_log: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, graph, top_k: int = 3) -> List[Episome]:
        """Retrieve the top-k episomes matching ``query`` via the Papez loop.

        Biologis: HC -> Mamillary -> Anterior Thalamus -> Cingulate ->
            PHG -> HC. The loop seeds from hippocampal episodic traces,
            integrates conflict detection at the cingulate, and
            re-enters HC with the refined query.
        AI: Scan every node in the EngramComplex. Score each node by
            ``keyword_overlap_fraction × node.confidence``. Return the
            top-k as Episome instances, sorted by descending score.

        Args:
            query: Seed query string.
            graph: An :class:`EngramComplex` whose wrapped AGNNGraph
                contains the episomes as nodes (label = episome.text).
            top_k: Maximum number of episomes to return. Defaults to 3.

        Returns:
            List of Episome instances sorted by descending confidence.
            Only nodes with at least one query-keyword match are
            returned. Empty list if nothing matches.
        """
        if top_k <= 0:
            self.retrieval_log.append((query, 0))
            return []

        query_keywords = _extract_keywords(query or "")
        if not query_keywords:
            # No usable keywords - nothing to match.
            self.retrieval_log.append((query, 0))
            return []

        inner = getattr(graph, "_graph", None)
        if inner is None:
            self.retrieval_log.append((query, 0))
            return []

        # Score every node in the graph.
        scored = []
        for node_id in inner.all_node_ids():
            node = inner.get_node(node_id)
            if node is None:
                continue
            node_keywords = set(_extract_keywords(node.label))
            if not node_keywords:
                continue
            overlap = len(set(query_keywords) & node_keywords)
            if overlap == 0:
                continue
            # Score = keyword coverage of the query × node confidence.
            # This rewards nodes that match more of the query AND nodes
            # that are more confidently held.
            coverage = overlap / len(set(query_keywords))
            score = coverage * float(node.confidence)
            scored.append((score, node))

        if not scored:
            self.retrieval_log.append((query, 0))
            return []

        # Sort by descending score; break ties by descending confidence
        # then by node id (deterministic).
        scored.sort(key=lambda x: (-x[0], -x[1].confidence, x[1].id))

        results: List[Episome] = []
        for _, node in scored[:top_k]:
            epi = self._node_to_episome(node)
            if epi is not None:
                results.append(epi)

        self.retrieval_log.append((query, len(results)))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_to_episome(node) -> Episome:
        """Reconstruct an Episome from an AGNNNode.

        Falls back gracefully if metadata is missing (e.g. for nodes
        added by something other than the TrisynapticCircuit).
        """
        meta = getattr(node, "metadata", {}) or {}
        try:
            episome_id = int(meta.get("episome_id", node.id))
        except (TypeError, ValueError):
            # node.id is a string that isn't an int - hash it to a
            # stable integer so Episome.id stays an int.
            episome_id = abs(hash(node.id)) % (10 ** 9)
        edge_type = meta.get("edge_type", "CATEGORICAL")
        type_marker = meta.get("type", "episodic")
        epi = Episome(
            id=episome_id,
            text=node.label,
            confidence=float(node.confidence),
            edge_type=edge_type,
            type=type_marker,
        )
        return epi
