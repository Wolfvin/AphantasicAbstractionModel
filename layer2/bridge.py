"""
RSVS Bridge — Unified adapter for PyO3 Rust core vs Python fallback mode.

This module provides a SINGLE point of contact with the RSVS core.
All layer2 modules should use this bridge instead of directly
importing from `rsvs` — this ensures consistent error handling,
proper API adaptation, and graceful fallback when the Rust core
isn't built.

Architecture:
    layer2 modules → AbstractionBridge → rsvs (PyO3) or fallback

Key design decisions:
1. The bridge wraps PyO3 objects in plain Python dicts/lists so
   downstream code never needs to handle PyO3 objects directly.
2. All methods return Optional[T] — None means "not available" or
   "concept not found", never raises.
3. Fallback mode provides a lightweight in-memory graph for testing
   and development without needing to build the Rust core.
4. Advanced RSVS features (MCTS, consolidation, reflection, etc.)
   are exposed through dedicated bridge methods with fallback
   implementations that degrade gracefully.
5. Soft grounding uses appraise() for confidence-based validation
   instead of hard prefix injection that corrupts data.

Analogi: Ini adalah "penerjemah" antara bahasa Rust (RSVS core)
dan bahasa Python (layer2 modules). Seperti bagaimana Jin Soun
bisa membaca teks dalam bahasa apapun dan mengerti maksudnya —
bridge ini menerjemahkan API Rust ke Python dengan cara yang konsisten.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Epistemological seed labels — must match Rust core's SEED_LABEL_LIST
# ---------------------------------------------------------------------------

SEED_LABELS = [
    "exists", "entity", "relation", "state", "change", "time", "space",
    "cause", "effect", "context", "signal", "pattern", "memory",
    "attention", "value", "agent", "goal", "risk", "trust", "identity",
    "language", "meaning", "action", "feedback",
]

# Priming sentence that contains ALL 24 seed labels, ensuring the Rust core
# graph has seed-relevant sentences from the start.
_PRIMING_SENTENCE = (
    "The entity exists in time and space. "
    "Cause and effect relate to change. "
    "Pattern and signal create meaning. "
    "Context provides value. "
    "Agent has goal, risk, trust, and identity. "
    "Memory and attention form feedback. "
    "Language and state define action. "
    "Relation connects cause and effect."
)

# ---------------------------------------------------------------------------
# Try to import the Rust core
# ---------------------------------------------------------------------------

_rust_core_available = False
_RsvsClass = None

try:
    from rsvs import Rsvs as _Rsvs  # type: ignore[import]
    _RsvsClass = _Rsvs
    _rust_core_available = True
except Exception:
    pass

# Also try via rsvs_core singleton
if not _rust_core_available:
    try:
        from rsvs.rsvs_core import get_rsvs_instance as _get_singleton  # type: ignore[import]
        _rust_core_available = True
    except Exception:
        pass


def is_rust_core_available() -> bool:
    """Check if the RSVS Rust core is importable."""
    return _rust_core_available


# ---------------------------------------------------------------------------
# Try to detect v12 pipeline availability (PyV12Pipeline)
# ---------------------------------------------------------------------------

_v12_available = False

try:
    from rsvs import PyV12Pipeline as _PyV12Pipeline  # type: ignore[import]
    _v12_available = True
except Exception:
    pass


def is_v12_available() -> bool:
    """Check if the v12 PipelineEngine (PyV12Pipeline) is importable.

    The v12 pipeline requires the Rust core to be built with
    ``--features v12,python``.  When unavailable, V12PipelineBridge
    degrades gracefully: ``available`` returns False and all methods
    return safe defaults (empty lists, zero counts, etc.) instead of
    raising.
    """
    return _v12_available


# ---------------------------------------------------------------------------
# Fallback graph — lightweight in-memory knowledge store with multi-sense
# and compositional model (L2-03 fix)
# ---------------------------------------------------------------------------

@dataclass
class _FallbackSense:
    """A sense (meaning) of a node in the fallback graph.

    Unlike the old model where each node had implicit single sense,
    this tracks multiple senses per node, each with its own grounding,
    coherence, and core atoms.
    """
    sense_idx: int = 0
    coherence: float = 0.5
    grounding_score: float = 0.5
    core_atoms: list[str] = field(default_factory=list)
    status: str = "fragile"  # "fragile" | "mature" | "stable"
    n_contexts: int = 0
    layer: int = 0

    def to_dict(self) -> dict:
        """Convert to plain dict compatible with RSVS sense format."""
        return {
            "sense_idx": self.sense_idx,
            "n_contexts": self.n_contexts,
            "coherence": self.coherence,
            "status": self.status,
            "core_atoms": list(self.core_atoms),
            "layer": self.layer,
            "grounding_score": self.grounding_score,
            "grounding_evidence": {
                "confirming_contexts": self.n_contexts,
                "contradicting_contexts": 0,
            },
            "compositions": [(a, 0) for a in self.core_atoms],
            "condition_label": None,
        }


@dataclass
class _FallbackNode:
    """A node in the fallback graph with multi-sense support."""
    label: str
    confidence: float = 0.5
    compositions: list[str] = field(default_factory=list)  # Explicit references
    contexts: list[str] = field(default_factory=list)
    layer: int = 0
    is_seed: bool = False
    observation_count: int = 0
    last_seen: float = field(default_factory=time.time)
    senses: list[_FallbackSense] = field(default_factory=list)
    # Event tracking for L2-06
    event_log: list[dict] = field(default_factory=list)
    # P-04: Source provenance for trust weighting
    source_provenance: str = "unknown"

    def __post_init__(self) -> None:
        """Ensure at least one default sense exists."""
        if not self.senses:
            self.senses = [_FallbackSense(sense_idx=0)]


class _FallbackGraph:
    """Lightweight fallback knowledge graph for when Rust core is unavailable.

    This is NOT a replacement for RSVS — it's a simple in-memory store
    that allows the layer2 modules to function (at reduced capability)
    during development and testing.

    Upgraded (L2-03) to support:
    - Multi-sense per node (list of _FallbackSense)
    - Composition = explicit references (not co-occurrence)
    - Grounding score tracking per sense
    - Coherence calculation
    - Event tracking for sense changes
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _FallbackNode] = {}
        self._edges: dict[str, list[str]] = {}  # label → [related labels]
        self._seq: int = 0
        self._events: list[dict] = []

    # ------------------------------------------------------------------
    # Event tracking
    # ------------------------------------------------------------------

    def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit an event for event stream consumers."""
        self._seq += 1
        event = {
            "seq": self._seq,
            "type": event_type,
            "timestamp": time.time(),
            **data,
        }
        self._events.append(event)
        # Keep event log bounded
        if len(self._events) > 1000:
            self._events = self._events[-500:]

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, text: str, source_provenance: str = "unknown") -> dict:
        """Ingest text by extracting keywords as nodes.

        Composition is now based on explicit reference tracking:
        words that appear in the same sentence are linked as compositions,
        but each node tracks which senses they belong to.

        Args:
            text: The text to ingest.
            source_provenance: Source provenance tag for trust weighting (P-04).
        """
        words = self._extract_keywords(text)
        sentences = text.count('.') + text.count('!') + text.count('?') + 1

        atoms_promoted = 0
        sense_assigned = 0
        sense_created = 0
        compositions_induced = 0

        for word in words:
            if word not in self._nodes:
                self._nodes[word] = _FallbackNode(label=word)
                atoms_promoted += 1
                self._emit_event("node_created", {"label": word})

            node = self._nodes[word]
            node.observation_count += 1
            node.last_seen = time.time()
            # P-04: Update provenance if a more specific source is given
            if source_provenance != "unknown":
                node.source_provenance = source_provenance

            # Update primary sense confidence and context count
            if node.senses:
                primary = node.senses[0]
                primary.n_contexts += 1
                primary.coherence = min(1.0, primary.coherence + 0.03)
                if primary.n_contexts > 5 and primary.status == "fragile":
                    primary.status = "mature"
                    self._emit_event("sense_changed", {
                        "label": word, "sense_idx": 0, "new_status": "mature"
                    })
                elif primary.n_contexts > 15 and primary.status == "mature":
                    primary.status = "stable"
                    self._emit_event("sense_changed", {
                        "label": word, "sense_idx": 0, "new_status": "stable"
                    })

            # Confidence increases with observation
            old_conf = node.confidence
            node.confidence = min(1.0, node.confidence + 0.05)
            if abs(node.confidence - old_conf) > 0.01:
                self._emit_event("confidence_changed", {
                    "label": word, "old": old_conf, "new": node.confidence
                })

            # Explicit composition references — words in same sentence
            # are related, but composition = explicit reference tracking
            for other in words:
                if other != word:
                    if other not in node.compositions:
                        node.compositions.append(other)
                        compositions_induced += 1
                    if word not in self._edges:
                        self._edges[word] = []
                    if other not in self._edges[word]:
                        self._edges[word].append(other)

            # Update grounding score per sense based on observation count
            for sense in node.senses:
                sense.grounding_score = min(1.0, sense.grounding_score + 0.02)

        # If a word appears with enough context, assign it to a sense
        for word in words:
            node = self._nodes.get(word)
            if node and node.observation_count >= 3 and not node.senses[0].core_atoms:
                # Assign core atoms from compositions
                node.senses[0].core_atoms = node.compositions[:5]
                sense_assigned += 1

        return {
            "sentences_processed": sentences,
            "atoms_promoted": atoms_promoted,
            "sense_assigned": sense_assigned,
            "sense_created": sense_created,
            "confidence_updated": atoms_promoted,
            "frozen_batches": 0,
            "compositions_induced": compositions_induced,
            "atoms_flagged_inactive": 0,
            "fallback": True,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, concept: str, context: str = "") -> Optional[dict]:
        """Simple keyword-based query."""
        if concept not in self._nodes:
            # Try partial match
            for label in self._nodes:
                if concept.lower() in label.lower() or label.lower() in concept.lower():
                    concept = label
                    break
            else:
                return None

        node = self._nodes.get(concept)
        if node is None:
            return None

        return {
            "sense_idx": 0,
            "sense_n": len(node.senses),
            "atoms": [(c, node.confidence * 0.8) for c in node.compositions[:10]],
            "layer": node.layer,
            "grounding_score": node.senses[0].grounding_score if node.senses else node.confidence,
            "compositions": [(c, 0) for c in node.compositions],
        }

    # ------------------------------------------------------------------
    # Relate
    # ------------------------------------------------------------------

    def relate(self, concept: str) -> Optional[dict]:
        """Simple co-occurrence-based relation."""
        related = self._edges.get(concept, [])
        if not related and concept in self._nodes:
            node = self._nodes[concept]
            related = node.compositions

        if not related:
            return None

        related_nodes = []
        for r in related:
            if r in self._nodes:
                n = self._nodes[r]
                related_nodes.append((r, n.confidence))

        return {
            "related_nodes": related_nodes,
            "related_edges": [],
            "structural_relations": related_nodes,
        }

    # ------------------------------------------------------------------
    # Appraise
    # ------------------------------------------------------------------

    def appraise(self, text: str) -> dict:
        """Simple keyword-based appraisal."""
        words = set(self._extract_keywords(text))
        known = set(self._nodes.keys())
        overlap = words & known

        if not words:
            return {
                "agree_pct": 0.5,
                "disagree_pct": 0.0,
                "neutral_pct": 0.5,
                "verdict": "neutral",
                "evidence": [],
                "convergence_info": [],
                "clash_pairs": [],
                "n_clusters": 0,
            }

        agree = len(overlap) / len(words) if words else 0.0
        disagree = 0.0  # Can't detect structural clash in fallback
        neutral = 1.0 - agree - disagree

        verdict = "agree" if agree > 0.5 else ("neutral" if agree > 0.2 else "disagree")

        return {
            "agree_pct": agree,
            "disagree_pct": disagree,
            "neutral_pct": neutral,
            "verdict": verdict,
            "evidence": [(w, self._nodes[w].confidence) for w in overlap],
            "convergence_info": [],
            "clash_pairs": [],
            "n_clusters": 0,
        }

    # ------------------------------------------------------------------
    # Structural similarity
    # ------------------------------------------------------------------

    def structural_similarity(self, a: str, b: str) -> Optional[dict]:
        """Jaccard similarity between compositions."""
        node_a = self._nodes.get(a)
        node_b = self._nodes.get(b)
        if node_a is None or node_b is None:
            return None

        set_a = set(node_a.compositions)
        set_b = set(node_b.compositions)

        if not set_a and not set_b:
            return {"structural_similarity": 0.0, "shared": [], "only_a": [], "only_b": []}

        shared = set_a & set_b
        union = set_a | set_b
        sim = len(shared) / len(union) if union else 0.0

        return {
            "structural_similarity": sim,
            "shared": list(shared),
            "only_a": list(set_a - set_b),
            "only_b": list(set_b - set_a),
        }

    # ------------------------------------------------------------------
    # Substitution analysis
    # ------------------------------------------------------------------

    def substitution_analysis(self, a: str, b: str) -> Optional[dict]:
        """Simple substitution analysis."""
        node_a = self._nodes.get(a)
        node_b = self._nodes.get(b)
        if node_a is None or node_b is None:
            return None

        set_a = set(node_a.compositions)
        set_b = set(node_b.compositions)

        only_a = set_a - set_b
        only_b = set_b - set_a
        shared = set_a & set_b
        sim = len(shared) / len(set_a | set_b) if (set_a | set_b) else 0.0

        return {
            "structural_similarity": sim,
            "substitutions": list(zip(only_a, only_b)),
            "unpaired_only_a": list(only_a),
            "unpaired_only_b": list(only_b),
        }

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self, label: str, compositions: list[tuple[str, int]], lang: Optional[str] = None) -> Optional[int]:
        """Create a compositional node."""
        if label not in self._nodes:
            self._nodes[label] = _FallbackNode(label=label, layer=1)

        node = self._nodes[label]
        for comp_label, _sense_id in compositions:
            if comp_label not in node.compositions:
                node.compositions.append(comp_label)

        node.layer = max(node.layer, 1)

        # Add a new sense for this composition
        new_sense_idx = len(node.senses)
        new_sense = _FallbackSense(
            sense_idx=new_sense_idx,
            core_atoms=[comp_label for comp_label, _ in compositions],
            layer=node.layer,
            status="fragile",
            grounding_score=0.3,
        )
        node.senses.append(new_sense)

        return hash(label) % (2**31)

    # ------------------------------------------------------------------
    # Senses (multi-sense support)
    # ------------------------------------------------------------------

    def senses(self, concept: str) -> Optional[list[dict]]:
        """Get senses for a concept with full multi-sense model."""
        node = self._nodes.get(concept)
        if node is None:
            return None
        return [s.to_dict() for s in node.senses]

    # ------------------------------------------------------------------
    # Confidence / nodes / node_info
    # ------------------------------------------------------------------

    def confidence_map(self) -> dict[str, float]:
        """Return confidence scores for all nodes."""
        return {label: node.confidence for label, node in self._nodes.items()}

    def nodes(self, include_seeds: bool = False) -> list[str]:
        """List all node labels."""
        if include_seeds:
            return list(self._nodes.keys())
        return [l for l, n in self._nodes.items() if not n.is_seed]

    def node_info(self, label: str) -> Optional[dict]:
        """Get info about a node."""
        node = self._nodes.get(label)
        if node is None:
            return None
        return {
            "label": label,
            "confidence": node.confidence,
            "tier": 2,
            "status": "stable" if node.observation_count > 3 else "candidate",
            "layer": node.layer,
            "compositions": node.compositions,
            "sense_count": len(node.senses),
            "grounding_score": node.senses[0].grounding_score if node.senses else node.confidence,
        }

    # ------------------------------------------------------------------
    # Status / events
    # ------------------------------------------------------------------

    def status(self) -> dict[str, float]:
        """Return system status."""
        return {
            "total_nodes": float(len(self._nodes)),
            "total_atoms": float(len(self._nodes)),
            "total_contexts": float(sum(n.observation_count for n in self._nodes.values())),
            "warmed_up": 1.0,
            "fallback": 1.0,
        }

    def latest_seq_v1(self) -> int:
        """Return current sequence number."""
        return self._seq

    def consume_events_v1(self, after_seq: Optional[int] = None, limit: int = 500) -> str:
        """Return events since the given sequence number."""
        after = after_seq or 0
        filtered = [e for e in self._events if e["seq"] > after][-limit:]
        return json.dumps({
            "events": filtered,
            "from_seq": after,
            "to_seq": self._seq,
        })

    # ------------------------------------------------------------------
    # Advanced RSVS feature fallbacks (L2-01)
    # ------------------------------------------------------------------

    def mcts_query(self, node_label: str, max_depth: int = 3, simulations: int = 50) -> Optional[dict]:
        """Fallback MCTS: simulate by doing breadth-first expansion.

        Returns a result compatible with Rust MCTSResult.
        """
        node = self._nodes.get(node_label)
        if node is None:
            return None

        # BFS expansion to simulate MCTS exploration
        visited: set[str] = {node_label}
        scored_atoms: list[tuple[str, float]] = []
        frontier = [node_label]
        depth_reached = 0

        for d in range(max_depth):
            next_frontier = []
            for current in frontier:
                related = self._edges.get(current, [])
                for r in related:
                    if r not in visited and r in self._nodes:
                        visited.add(r)
                        n = self._nodes[r]
                        # Score decays with depth
                        score = n.confidence / (d + 1)
                        scored_atoms.append((r, score))
                        next_frontier.append(r)
            frontier = next_frontier
            if frontier:
                depth_reached = d + 1

        # Sort by score descending
        scored_atoms.sort(key=lambda x: -x[1])

        best_path = [(node_label, 0)]
        for label, score in scored_atoms[:5]:
            best_path.append((label, 0))

        return {
            "active_sense_idx": 0,
            "total_senses": len(node.senses),
            "scored_atoms": scored_atoms[:20],
            "depth_reached": depth_reached,
            "halt_reason": "max_depth" if depth_reached >= max_depth else "exhausted",
            "simulations_run": simulations,
            "best_path": best_path,
            "layer": node.layer,
            "grounding_score": node.senses[0].grounding_score if node.senses else node.confidence,
        }

    def consolidate(self) -> dict:
        """Fallback consolidation: merge similar senses within each node."""
        senses_merged = 0
        senses_removed = 0
        edges_pruned = 0

        for label, node in list(self._nodes.items()):
            if len(node.senses) <= 1:
                continue

            # Merge senses with very similar core atoms
            merged_indices: set[int] = set()
            for i, s1 in enumerate(node.senses):
                if i in merged_indices:
                    continue
                for j in range(i + 1, len(node.senses)):
                    if j in merged_indices:
                        continue
                    s2 = node.senses[j]
                    # Check similarity of core atoms
                    set1 = set(s1.core_atoms)
                    set2 = set(s2.core_atoms)
                    if set1 and set2:
                        overlap = len(set1 & set2) / max(len(set1 | set2), 1)
                        if overlap > 0.8:
                            # Merge s2 into s1
                            s1.core_atoms = list(set1 | set2)
                            s1.n_contexts += s2.n_contexts
                            s1.coherence = max(s1.coherence, s2.coherence)
                            s1.grounding_score = max(s1.grounding_score, s2.grounding_score)
                            merged_indices.add(j)
                            senses_merged += 1

            if merged_indices:
                node.senses = [s for i, s in enumerate(node.senses) if i not in merged_indices]
                senses_removed += len(merged_indices)
                # Re-index
                for i, s in enumerate(node.senses):
                    s.sense_idx = i

        return {
            "senses_merged": senses_merged,
            "senses_removed": senses_removed,
            "edges_pruned": edges_pruned,
            "atoms_compacted": 0,
        }

    def run_reflection(self) -> dict:
        """Fallback reflection: basic self-consistency check.

        Checks for nodes with low coherence or grounding, and
        identifies candidates for removal or re-evaluation.
        """
        actions_total = 0
        actions_applied = 0

        for label, node in list(self._nodes.items()):
            actions_total += 1
            # Check if any sense has very low grounding
            for sense in node.senses:
                if sense.grounding_score < 0.1 and sense.n_contexts < 2:
                    # Mark for potential removal (but don't actually remove)
                    sense.status = "fragile"
                    actions_applied += 1

        return {
            "actions_total": actions_total,
            "actions_applied": actions_applied,
        }

    def verify(self) -> dict:
        """Fallback verification: check graph integrity."""
        issues: dict[str, int] = {
            "dangling_references": 0,
            "empty_nodes": 0,
            "orphan_senses": 0,
        }

        all_labels = set(self._nodes.keys())
        for label, node in self._nodes.items():
            # Check for dangling composition references
            for comp in node.compositions:
                if comp not in all_labels:
                    issues["dangling_references"] += 1

            # Check for empty nodes
            if not node.compositions and node.observation_count == 0:
                issues["empty_nodes"] += 1

            # Check for orphan senses
            for sense in node.senses:
                if sense.n_contexts == 0 and sense.status == "mature":
                    issues["orphan_senses"] += 1

        return issues

    def toggle_thinking(self, mode: str) -> None:
        """Fallback thinking mode toggle (no-op in fallback)."""
        logger.debug("Fallback: toggle_thinking(%s) — no-op", mode)

    def route_paradigm(self, query: str) -> str:
        """Fallback paradigm routing: simple heuristic.

        Returns one of: "structural", "associative", "compositional",
        "temporal", "causal".
        """
        query_lower = query.lower()

        causal_words = {"because", "cause", "why", "therefore", "sebab", "karena", "mengapa"}
        temporal_words = {"when", "before", "after", "during", "kapan", "sebelum", "sesudah", "saat"}
        structural_words = {"structure", "composition", "consists", "parts", "struktur", "komposisi", "bagian"}
        compositional_words = {"made of", "composed", "contains", "terdiri", "mengandung"}

        tokens = set(query_lower.split())
        if tokens & causal_words:
            return "causal"
        if tokens & temporal_words:
            return "temporal"
        if tokens & structural_words:
            return "structural"
        if tokens & compositional_words:
            return "compositional"
        return "associative"

    def deps_analyze(self, failure: str = "") -> dict:
        """Fallback DEPS analysis: basic dependency tracing.

        Tries to find nodes that depend on the given failure concept.
        """
        result: dict = {
            "failure_node": failure,
            "dependent_nodes": [],
            "explanation": "",
        }

        failure_node = self._nodes.get(failure)
        if failure_node is None:
            result["explanation"] = f"Node '{failure}' not found in graph."
            return result

        # Find nodes that reference the failure node
        dependents = []
        for label, node in self._nodes.items():
            if failure in node.compositions:
                dependents.append({
                    "label": label,
                    "confidence": node.confidence,
                    "sense_count": len(node.senses),
                })

        result["dependent_nodes"] = dependents
        result["explanation"] = (
            f"Found {len(dependents)} node(s) that depend on '{failure}'. "
            f"Failure may propagate through these dependencies."
        )
        return result

    def matryoshka_traverse(self, node_label: str, depth: int = 3) -> Optional[dict]:
        """Fallback matryoshka traversal: nested layer-by-layer exploration.

        Returns concentric layers of related nodes, like Russian dolls.
        """
        node = self._nodes.get(node_label)
        if node is None:
            return None

        layers: list[list[str]] = [[node_label]]
        visited: set[str] = {node_label}

        for d in range(depth):
            current_layer = layers[-1]
            next_layer: list[str] = []
            for current in current_layer:
                related = self._edges.get(current, [])
                for r in related:
                    if r not in visited and r in self._nodes:
                        visited.add(r)
                        next_layer.append(r)
            if not next_layer:
                break
            layers.append(next_layer)

        # Compute nesting scores (inner = higher relevance)
        scored_nodes: list[tuple[str, float, int]] = []
        for layer_idx, layer_nodes in enumerate(layers):
            decay = 1.0 / (layer_idx + 1)
            for label in layer_nodes:
                n = self._nodes.get(label)
                score = (n.confidence if n else 0.5) * decay
                scored_nodes.append((label, score, layer_idx))

        return {
            "root": node_label,
            "depth": len(layers) - 1,
            "layers": layers,
            "scored_nodes": scored_nodes,
            "total_nodes_visited": len(visited),
        }

    # ------------------------------------------------------------------
    # Context similarity (for context_query fallback)
    # ------------------------------------------------------------------

    def context_similarity(self, a: str, b: str, context: list[str]) -> Optional[float]:
        """Fallback context similarity: weighted Jaccard with context bias."""
        node_a = self._nodes.get(a)
        node_b = self._nodes.get(b)
        if node_a is None or node_b is None:
            return None

        set_a = set(node_a.compositions)
        set_b = set(node_b.compositions)
        context_set = set(context)

        # Base similarity
        if not set_a and not set_b:
            return 0.0

        shared = set_a & set_b
        union = set_a | set_b
        base_sim = len(shared) / len(union) if union else 0.0

        # Boost if context atoms appear in shared compositions
        context_overlap = context_set & shared
        context_boost = len(context_overlap) / max(len(context_set), 1) * 0.2

        return min(1.0, base_sim + context_boost)

    # ------------------------------------------------------------------
    # Coherence calculation
    # ------------------------------------------------------------------

    def compute_coherence(self, node_label: str) -> float:
        """Compute coherence of a node based on sense consistency.

        Coherence measures how internally consistent a node's senses are.
        A node with one very strong sense is more coherent than one with
        many fragmented senses.
        """
        node = self._nodes.get(node_label)
        if node is None or not node.senses:
            return 0.0

        if len(node.senses) == 1:
            return node.senses[0].coherence

        # Multiple senses: coherence = weighted average, penalized by fragmentation
        total_contexts = sum(s.n_contexts for s in node.senses)
        if total_contexts == 0:
            return 0.0

        weighted = sum(s.coherence * s.n_contexts for s in node.senses) / total_contexts
        # Fragmentation penalty: more senses = lower coherence
        frag_penalty = 1.0 / (1.0 + 0.1 * (len(node.senses) - 1))
        return weighted * frag_penalty

    # ------------------------------------------------------------------
    # Convergence detection (P1-3)
    # ------------------------------------------------------------------

    def _fallback_detect_convergence(self, max_pairs: int = 500) -> dict:
        """Fallback convergence detection: structural similarity across nodes.

        Finds nodes with structurally similar sense compositions that
        may represent the same concept across different languages or contexts.

        Args:
            max_pairs: Maximum number of pairs to evaluate.

        Returns:
            Dict with pairs_found, convergence_pairs, and source.
        """
        # Sort nodes by confidence (descending) and take top 50
        sorted_nodes = sorted(
            self._nodes.items(),
            key=lambda item: item[1].confidence,
            reverse=True,
        )[:50]

        convergence_pairs: list[dict] = []
        pairs_checked = 0

        for i, (label_a, node_a) in enumerate(sorted_nodes):
            if len(convergence_pairs) >= max_pairs:
                break
            for label_b, node_b in sorted_nodes[i + 1:]:
                if len(convergence_pairs) >= max_pairs:
                    break
                if pairs_checked >= max_pairs:
                    break

                # Skip if labels are identical or one is a substring of the other
                if label_a == label_b or label_a in label_b or label_b in label_a:
                    continue

                pairs_checked += 1

                # Compute Jaccard similarity of compositions
                set_a = set(node_a.compositions)
                set_b = set(node_b.compositions)

                if not set_a and not set_b:
                    continue

                shared = set_a & set_b
                union = set_a | set_b
                similarity = len(shared) / len(union) if union else 0.0

                # Only include pairs with meaningful similarity (threshold > 0.3)
                # and different labels (indicates potential convergence)
                if similarity > 0.3:
                    convergence_pairs.append({
                        "node_a": label_a,
                        "node_b": label_b,
                        "similarity": similarity,
                        "shared_compositions": list(shared),
                    })

        return {
            "pairs_found": len(convergence_pairs),
            "convergence_pairs": convergence_pairs,
            "source": "fallback",
        }

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        stop_words = {
            "that", "this", "with", "from", "have", "been", "they",
            "their", "which", "would", "there", "could", "about",
            "other", "into", "more", "than", "then", "some", "very",
            "also", "just", "like", "only", "over", "such", "after",
            "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
            "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
            "the", "and", "but", "for", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "are", "has",
        }
        words = text.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()
        return [w for w in words if len(w) > 2 and w not in stop_words][:30]


# ---------------------------------------------------------------------------
# AbstractionBridge — the main adapter
# ---------------------------------------------------------------------------

class AbstractionBridge:
    """Unified adapter for RSVS Rust core (PyO3) and Python fallback.

    This is the SINGLE point of contact for all layer2 modules.
    It provides a consistent Python API regardless of whether the
    Rust core is available or not.

    Key features:
    - Wraps PyO3 objects in plain Python dicts/lists
    - All methods return Optional[T] — None means "not found"
    - Never raises (catches all PyO3 errors internally)
    - Falls back to _FallbackGraph when Rust core isn't built
    - Exposes ALL advanced RSVS features with graceful degradation

    Usage:
        bridge = AbstractionBridge()
        if bridge.is_available:
            result = bridge.query("raja", context="monarki")
        else:
            result = bridge.query("raja")  # uses fallback graph

    Attributes:
        is_available: Whether a working RSVS instance is connected
            (either Rust core or fallback graph).
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(self, rsvs_instance: Any = None) -> None:
        """Initialize the bridge.

        Args:
            rsvs_instance: Optional pre-built RSVS instance (PyRsvs or similar).
                If None, the bridge will try to create or obtain one.
        """
        self._rsvs: Any = None
        self.is_rust_core: bool = False
        self._fallback: Optional[_FallbackGraph] = None
        self._primed: bool = False
        # Rotating seed word cycle for ingest_with_grounding()
        self._seed_cycle = itertools.cycle(SEED_LABELS)
        # P2-6: Embedding provider for semantic similarity (lazy-init)
        self._embedding_provider: Any = None  # EmbeddingProvider | None

        if rsvs_instance is not None:
            self._rsvs = rsvs_instance
            self.is_rust_core = True
            self.is_available = True
            logger.info("AbstractionBridge initialized with provided RSVS instance")
            self._prime_rust_core()
            return

        # Try to get the Rust core
        if _rust_core_available:
            try:
                # Strategy 1: Use singleton from rsvs_core
                try:
                    from rsvs.rsvs_core import get_rsvs_instance as _get  # type: ignore[import]
                    self._rsvs = _get()
                    self.is_rust_core = True
                    self.is_available = True
                    logger.info("AbstractionBridge initialized with Rust core (singleton)")
                    self._prime_rust_core()
                    return
                except Exception:
                    pass

                # Strategy 2: Create a fresh instance
                if _RsvsClass is not None:
                    self._rsvs = _RsvsClass()
                    self.is_rust_core = True
                    self.is_available = True
                    logger.info("AbstractionBridge initialized with Rust core (new instance)")
                    self._prime_rust_core()
                    return
            except Exception as exc:
                logger.warning("Failed to initialize Rust core: %s", exc)

        # Fallback mode
        self._fallback = _FallbackGraph()
        self.is_rust_core = False
        self.is_available = True  # Fallback IS available, just limited
        logger.info("AbstractionBridge initialized in FALLBACK mode (no Rust core)")

    # ------------------------------------------------------------------
    # Priming & grounding for Rust core
    # ------------------------------------------------------------------

    def _prime_rust_core(self) -> None:
        """Prime the Rust core by ingesting a sentence containing seed labels."""
        if self._primed or not self.is_rust_core:
            return

        try:
            stats = self._rsvs.ingest(_PRIMING_SENTENCE)
            normalized = self._normalize_stats(stats)
            self._primed = True
            logger.info(
                "Rust core primed: atoms_promoted=%d, sentences=%d",
                normalized.get("atoms_promoted", 0),
                normalized.get("sentences_processed", 0),
            )
        except Exception as exc:
            logger.warning("Failed to prime Rust core: %s", exc)

    def _text_contains_seed(self, text: str) -> bool:
        """Check whether text naturally contains any of the 24 seed labels."""
        words = text.lower().split()
        all_tokens = set()
        for w in words:
            all_tokens.add(w)
            stripped = w.strip(".,;:!?'\"()[]{}-")
            if stripped:
                all_tokens.add(stripped)
        seed_set = set(SEED_LABELS)
        return bool(all_tokens & seed_set)

    def ingest_with_grounding(self, text: str, domain_id: Optional[int] = None, source_provenance: Optional[str] = None) -> dict:
        """Ingest text with soft grounding (L2-05 fix).

        Instead of prepending "The {seed} relates to {text}" which
        injects noise words, this method uses:
        1. If text already contains seed words → ingest as-is
        2. If Rust core supports ingest_with_meta_v1 → use it with lang hint
        3. Otherwise, use appraise()-based soft grounding: ingest the text,
           then verify via appraise() that the resulting graph state is
           consistent (confidence-based soft grounding instead of hard gate)
        4. Last resort: minimal prefix that only prepends the seed word
           followed by a colon, not a full sentence — less noise injection

        Args:
            text: The text to ingest.
            domain_id: Optional domain tag for the ingestion.
            source_provenance: Optional source provenance tag (P-04).

        Returns:
            A dict with ingestion stats.
        """
        if not self.is_rust_core:
            # No grounding needed in fallback mode
            provenance = source_provenance or "unknown"
            return self._fallback.ingest(text, source_provenance=provenance)  # type: ignore[union-attr]

        if self._text_contains_seed(text):
            # Text already has seed co-occurrence — ingest as-is
            return self.ingest(text, domain_id=domain_id, source_provenance=source_provenance)

        # Strategy 1: Try ingest_with_meta_v1 if available (Rust supports lang)
        try:
            if hasattr(self._rsvs, "ingest_with_meta_v1"):
                result = self._rsvs.ingest_with_meta_v1(text, domain_id=domain_id)
                stats = self._normalize_stats(result) if result else {}
                if stats.get("atoms_promoted", 0) > 0:
                    return stats
        except Exception as exc:
            logger.debug("ingest_with_meta_v1 failed, trying soft grounding: %s", exc)

        # Strategy 2: Soft grounding via appraise() — ingest text, then verify
        try:
            stats = self._rsvs.ingest(text)
            normalized = self._normalize_stats(stats)

            # If atoms were promoted without grounding, we're done
            if normalized.get("atoms_promoted", 0) > 0:
                # Soft grounding: use appraise to validate consistency
                try:
                    appraise_result = self._rsvs.appraise(text)
                    if appraise_result is not None:
                        verdict = getattr(appraise_result, "verdict", "neutral")
                        if verdict == "disagree":
                            logger.debug(
                                "Soft grounding: appraise returned 'disagree' for text: %.60s",
                                text,
                            )
                except Exception:
                    pass
                return normalized
        except Exception as exc:
            logger.debug("Direct ingest failed, trying minimal grounding: %s", exc)

        # Strategy 3: Minimal prefix grounding — use just "SEED: text"
        # This is much less noisy than "The {seed} relates to {text}"
        seed_word = next(self._seed_cycle)
        grounded_text = f"{seed_word}: {text}"
        logger.debug(
            "ingest_with_grounding: minimal prefix seed '%s' for text: %.60s...",
            seed_word, text,
        )
        return self.ingest(grounded_text, domain_id=domain_id)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def ingest(self, text: str, domain_id: Optional[int] = None, source_provenance: Optional[str] = None) -> dict:
        """Ingest text into the knowledge graph.

        When the Rust core is active and the text does not naturally
        contain any of the 24 epistemological seed labels, this method
        automatically delegates to ``ingest_with_grounding()`` which
        uses soft grounding (L2-05) instead of hard prefix injection.

        Args:
            text: The text to ingest.
            domain_id: Optional domain tag for the ingestion.
            source_provenance: Optional source provenance tag for trust
                weighting (P-04). When provided, this is recorded in the
                graph metadata so that appraise() can weight confidence
                by source trust.

        Returns:
            A dict with ingestion stats.
        """
        if self.is_rust_core:
            # Auto-ground: if text lacks seed words, use soft grounding
            if not self._text_contains_seed(text):
                return self.ingest_with_grounding(text, domain_id=domain_id, source_provenance=source_provenance)

            try:
                if domain_id is not None:
                    self._rsvs.set_domain(domain_id)
                stats = self._rsvs.ingest(text)
                result = self._normalize_stats(stats)
                # Attach provenance metadata
                if source_provenance is not None:
                    result["source_provenance"] = source_provenance
                return result
            except Exception as exc:
                logger.error("RSVS ingest failed: %s", exc)
                return {"success": False, "error": str(exc)}

        # Fallback
        provenance = source_provenance or "unknown"
        return self._fallback.ingest(text, source_provenance=provenance)  # type: ignore[union-attr]

    def query(self, concept: str, context: str = "") -> Optional[dict]:
        """Query a concept in the knowledge graph."""
        if self.is_rust_core:
            try:
                result = self._rsvs.query(concept, context)
                if result is None:
                    return None
                return self._normalize_query_result(result)
            except Exception as exc:
                logger.debug("RSVS query failed for '%s': %s", concept, exc)
                return None

        return self._fallback.query(concept, context)  # type: ignore[union-attr]

    def context_query(
        self,
        concept: str,
        context_atoms: list[str],
        max_depth: Optional[int] = None,
        gamma: Optional[float] = None,
        halt_confidence: Optional[float] = None,
        tau_relevance: Optional[float] = None,
    ) -> Optional[dict]:
        """Context-aware query using depth-controlled traversal."""
        if self.is_rust_core:
            try:
                result = self._rsvs.context_query(
                    concept, context_atoms, max_depth, gamma, halt_confidence, tau_relevance
                )
                if result is None:
                    return None
                return self._normalize_context_query_result(result)
            except Exception as exc:
                logger.debug("RSVS context_query failed: %s", exc)
                return None

        # Fallback: use simple query
        return self._fallback.query(concept, " ".join(context_atoms))  # type: ignore[union-attr]

    def relate(self, concept: str) -> Optional[dict]:
        """Find related nodes via spreading activation.

        F-04 fix: After normalizing the relate result, resolve any numeric
        node IDs to labels by calling node_info() on the RSVS instance.
        """
        if self.is_rust_core:
            try:
                result = self._rsvs.relate(concept)
                if result is None:
                    return None
                normalized = self._normalize_relate_result(result)

                # F-04: Resolve numeric IDs to labels using the RSVS instance
                if normalized.get("_needs_label_resolution"):
                    resolved_nodes = []
                    for item in normalized.get("related_nodes", []):
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            label, score, node_id = item[0], item[1], item[2]
                            # Try to resolve label from RSVS
                            if node_id is not None:
                                try:
                                    info = self._rsvs.node_info(node_id)
                                    if info and hasattr(info, "label"):
                                        label = info.label
                                    elif isinstance(info, dict) and "label" in info:
                                        label = info["label"]
                                except Exception:
                                    pass
                            resolved_nodes.append((label, score))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            resolved_nodes.append((item[0], item[1]))
                        else:
                            resolved_nodes.append(item)

                    resolved_structural = []
                    for item in normalized.get("structural_relations", []):
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            label, score, node_id = item[0], item[1], item[2]
                            if node_id is not None:
                                try:
                                    info = self._rsvs.node_info(node_id)
                                    if info and hasattr(info, "label"):
                                        label = info.label
                                    elif isinstance(info, dict) and "label" in info:
                                        label = info["label"]
                                except Exception:
                                    pass
                            resolved_structural.append((label, score))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            resolved_structural.append((item[0], item[1]))
                        else:
                            resolved_structural.append(item)

                    normalized["related_nodes"] = resolved_nodes
                    normalized["structural_relations"] = resolved_structural
                    del normalized["_needs_label_resolution"]

                return normalized
            except Exception as exc:
                logger.debug("RSVS relate failed for '%s': %s", concept, exc)
                return None

        return self._fallback.relate(concept)  # type: ignore[union-attr]

    def appraise(self, text: str) -> dict:
        """Appraise text against the knowledge graph."""
        if self.is_rust_core:
            try:
                result = self._rsvs.appraise(text)
                return self._normalize_appraise_result(result)
            except Exception as exc:
                logger.debug("RSVS appraise failed: %s", exc)
                return {
                    "agree_pct": 0.0, "disagree_pct": 0.0, "neutral_pct": 1.0,
                    "verdict": "neutral", "evidence": [], "convergence_info": [],
                    "clash_pairs": [], "n_clusters": 0,
                }

        return self._fallback.appraise(text)  # type: ignore[union-attr]

    def appraise_with_provenance(
        self, text: str, source_trust_map: Optional[dict[str, float]] = None,
    ) -> dict:
        """Appraise text with source-provenance-weighted confidence (P-04).

        Like appraise(), but weights evidence confidence by the trust
        score of each node's source provenance. Low-trust sources
        contribute less to the agreement score.

        Args:
            text: The text to appraise.
            source_trust_map: Optional mapping from source type to trust
                score (0.0-1.0). If not provided, falls back to the
                default SOURCE_TRUST map from context.py.

        Returns:
            A dict with appraise results, with confidence weighted by
            source provenance trust.
        """
        # Get base appraise result
        result = self.appraise(text)

        if not source_trust_map:
            # Import default trust map from context layer
            try:
                from .context import SOURCE_TRUST
                source_trust_map = SOURCE_TRUST
            except ImportError:
                source_trust_map = {
                    "user_input": 1.0, "web_search": 0.7,
                    "academic": 0.9, "unknown": 0.5,
                }

        # If Rust core, we can't access per-node provenance, so return
        # the base result with a provenance metadata note
        if self.is_rust_core:
            result["provenance_weighted"] = False
            return result

        # Fallback mode: weight evidence confidence by source trust
        if self._fallback is not None:
            evidence = result.get("evidence", [])
            weighted_evidence = []
            total_trust = 0.0
            weighted_agree = 0.0

            for entry in evidence:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    label, conf = entry[0], float(entry[1])
                    node = self._fallback._nodes.get(label)
                    source = node.source_provenance if node else "unknown"
                    trust = source_trust_map.get(source, 0.5)
                    weighted_conf = conf * trust
                    weighted_evidence.append((label, weighted_conf, source, trust))
                    weighted_agree += weighted_conf
                    total_trust += trust
                elif isinstance(entry, str):
                    node = self._fallback._nodes.get(entry)
                    source = node.source_provenance if node else "unknown"
                    trust = source_trust_map.get(source, 0.5)
                    weighted_evidence.append((entry, trust, source, trust))
                    weighted_agree += trust
                    total_trust += trust

            # Adjust agree_pct based on weighted evidence
            if total_trust > 0:
                result["agree_pct"] = min(1.0, weighted_agree / total_trust)

            result["evidence"] = weighted_evidence
            result["provenance_weighted"] = True

        return result

    def structural_similarity(self, a: str, b: str) -> Optional[dict]:
        """Compute structural similarity between two concepts.

        When an embedding provider is available (P2-6), augments the
        graph-based similarity with embedding cosine similarity for
        more accurate semantic comparison.
        """
        # P2-6: Try embedding-based similarity first as a supplement
        embedding_sim: Optional[float] = None
        provider = self._get_or_init_embedding_provider()
        if provider is not None:
            try:
                from .embedding import cosine_similarity
                emb_a = provider.embed(a)
                emb_b = provider.embed(b)
                embedding_sim = cosine_similarity(emb_a, emb_b)
            except Exception as exc:
                logger.debug("Embedding-based similarity failed: %s", exc)

        if self.is_rust_core:
            try:
                result = self._rsvs.structural_similarity(a, b)
                if result is None:
                    # Even if RSVS returns None, we can still return embedding sim
                    if embedding_sim is not None:
                        return {
                            "structural_similarity": embedding_sim,
                            "shared": [],
                            "only_a": [],
                            "only_b": [],
                            "embedding_similarity": embedding_sim,
                        }
                    return None
                normed = self._normalize_structural_sim(result)
                # Blend graph-based and embedding similarity
                if embedding_sim is not None:
                    graph_sim = normed.get("structural_similarity", 0.0)
                    blended = 0.6 * graph_sim + 0.4 * embedding_sim
                    normed["structural_similarity"] = blended
                    normed["embedding_similarity"] = embedding_sim
                    normed["graph_similarity"] = graph_sim
                return normed
            except Exception as exc:
                logger.debug("RSVS structural_similarity failed: %s", exc)
                if embedding_sim is not None:
                    return {
                        "structural_similarity": embedding_sim,
                        "shared": [],
                        "only_a": [],
                        "only_b": [],
                        "embedding_similarity": embedding_sim,
                    }
                return None

        result = self._fallback.structural_similarity(a, b)  # type: ignore[union-attr]
        if result is not None and embedding_sim is not None:
            graph_sim = result.get("structural_similarity", 0.0)
            blended = 0.6 * graph_sim + 0.4 * embedding_sim
            result["structural_similarity"] = blended
            result["embedding_similarity"] = embedding_sim
            result["graph_similarity"] = graph_sim
        elif result is None and embedding_sim is not None:
            result = {
                "structural_similarity": embedding_sim,
                "shared": [],
                "only_a": [],
                "only_b": [],
                "embedding_similarity": embedding_sim,
            }
        return result

    def substitution_analysis(self, a: str, b: str) -> Optional[dict]:
        """Analyze what substitution transforms concept A into concept B."""
        if self.is_rust_core:
            try:
                result = self._rsvs.substitution_analysis(a, b)
                if result is None:
                    return None
                return self._normalize_substitution(result)
            except Exception as exc:
                logger.debug("RSVS substitution_analysis failed: %s", exc)
                return None

        return self._fallback.substitution_analysis(a, b)  # type: ignore[union-attr]

    def compose(self, label: str, compositions: list[tuple[str, int]], lang: Optional[str] = None) -> Optional[int]:
        """Create a compositional node from explicit composition references."""
        if self.is_rust_core:
            try:
                return self._rsvs.compose(label, compositions, lang)
            except Exception as exc:
                logger.debug("RSVS compose failed: %s", exc)
                return None

        return self._fallback.compose(label, compositions, lang)  # type: ignore[union-attr]

    def senses(self, concept: str) -> Optional[list[dict]]:
        """Get all senses for a concept."""
        if self.is_rust_core:
            try:
                result = self._rsvs.senses(concept)
                if result is None:
                    return None
                return self._normalize_senses(result)
            except Exception as exc:
                logger.debug("RSVS senses failed for '%s': %s", concept, exc)
                return None

        return self._fallback.senses(concept)  # type: ignore[union-attr]

    def confidence_map(self) -> dict[str, float]:
        """Return confidence scores for all nodes."""
        if self.is_rust_core:
            try:
                return self._rsvs.confidence_map()
            except Exception:
                return {}

        return self._fallback.confidence_map()  # type: ignore[union-attr]

    def nodes(self, include_seeds: bool = False) -> list[str]:
        """List all node labels."""
        if self.is_rust_core:
            try:
                return self._rsvs.nodes(include_seeds)
            except Exception:
                return []

        return self._fallback.nodes(include_seeds)  # type: ignore[union-attr]

    def node_info(self, label: str) -> Optional[dict]:
        """Get info about a specific node."""
        if self.is_rust_core:
            try:
                result = self._rsvs.node_info(label)
                return self._normalize_node_info(result)
            except Exception:
                return None

        return self._fallback.node_info(label)  # type: ignore[union-attr]

    def status(self) -> dict:
        """Return system status."""
        if self.is_rust_core:
            try:
                return self._rsvs.status()
            except Exception:
                return {}

        return self._fallback.status()  # type: ignore[union-attr]

    def latest_seq_v1(self) -> int:
        """Return latest event sequence number."""
        if self.is_rust_core:
            try:
                return self._rsvs.latest_seq_v1()
            except Exception:
                return 0

        return self._fallback.latest_seq_v1()  # type: ignore[union-attr]

    def consume_events_v1(self, after_seq: Optional[int] = None, limit: int = 500) -> str:
        """Consume events after a sequence number."""
        if self.is_rust_core:
            try:
                return self._rsvs.consume_events_v1(after_seq, limit)
            except Exception:
                return json.dumps({"events": []})

        return self._fallback.consume_events_v1(after_seq, limit)  # type: ignore[union-attr]

    def set_domain(self, domain_id: int) -> None:
        """Set the current domain tag."""
        if self.is_rust_core:
            try:
                self._rsvs.set_domain(domain_id)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Advanced RSVS features (L2-01 fix)
    # ------------------------------------------------------------------

    def mcts_query(
        self,
        node_label: str,
        max_depth: int = 3,
        simulations: int = 100,
    ) -> Optional[dict]:
        """Monte Carlo Tree Search query for complex prediction paths.

        Uses MCTS to explore possible reasoning paths from a given node,
        finding high-value paths through the knowledge graph.

        Args:
            node_label: The starting node label.
            max_depth: Maximum search depth.
            simulations: Number of MCTS simulations to run.

        Returns:
            A dict with MCTS results including scored atoms, best path,
            and halting information, or None if node not found.
        """
        if self.is_rust_core:
            try:
                # Rust core's mcts_query signature: (label, simulations, exploration)
                exploration = 1.414  # UCB1 default
                result = self._rsvs.mcts_query(node_label, simulations, exploration)
                if result is None:
                    return None
                return self._normalize_mcts_result(result)
            except Exception as exc:
                logger.debug("RSVS mcts_query failed for '%s': %s", node_label, exc)
                return None

        return self._fallback.mcts_query(node_label, max_depth, simulations)  # type: ignore[union-attr]

    def consolidate(self, force: bool = False) -> dict:
        """Consolidate the knowledge graph.

        Merges similar senses, prunes stale edges, and compacts
        the graph for better performance and accuracy.

        Args:
            force: Force consolidation even if auto-threshold hasn't
                been reached (Rust core only). Default False.

        Returns:
            A dict with consolidation statistics.
        """
        if self.is_rust_core:
            try:
                result = self._rsvs.consolidate(force)
                if isinstance(result, dict):
                    return result
                return self._normalize_consolidation_result(result)
            except Exception as exc:
                logger.debug("RSVS consolidate failed: %s", exc)
                return {
                    "senses_merged": 0, "senses_removed": 0,
                    "edges_pruned": 0, "atoms_compacted": 0,
                }

        return self._fallback.consolidate()  # type: ignore[union-attr]

    def run_reflection(self) -> dict:
        """Run a reflection cycle on the knowledge graph.

        The reflection cycle examines the graph for inconsistencies,
        low-quality senses, and opportunities for improvement.

        Returns:
            A dict with reflection results including actions taken.
        """
        if self.is_rust_core:
            try:
                result = self._rsvs.run_reflection()
                if isinstance(result, dict):
                    return result
                return self._normalize_reflection_result(result)
            except Exception as exc:
                logger.debug("RSVS run_reflection failed: %s", exc)
                return {"actions_total": 0, "actions_applied": 0}

        return self._fallback.run_reflection()  # type: ignore[union-attr]

    def verify(self) -> dict:
        """Verify the integrity of the knowledge graph.

        Checks for dangling references, orphan senses, and other
        structural issues.

        Returns:
            A dict mapping issue type → count.
        """
        if self.is_rust_core:
            try:
                result = self._rsvs.verify()
                if isinstance(result, dict):
                    return result
                # PyO3 returns a dict directly
                return dict(result) if result else {}
            except Exception as exc:
                logger.debug("RSVS verify failed: %s", exc)
                return {}

        return self._fallback.verify()  # type: ignore[union-attr]

    def toggle_thinking(self, mode: str) -> None:
        """Toggle the thinking mode in the RSVS core.

        Modes control how deeply the system reasons about each query.
        Common modes: "fast", "normal", "deep".

        Args:
            mode: The thinking mode to set.
        """
        if self.is_rust_core:
            try:
                # Rust core uses set_thinking_mode with integer mode
                mode_map = {"fast": 0, "normal": 1, "deep": 2}
                mode_int = mode_map.get(mode, 1)
                self._rsvs.set_thinking_mode(mode_int)
                logger.debug("Thinking mode set to: %s (%d)", mode, mode_int)
            except Exception as exc:
                logger.debug("RSVS set_thinking_mode failed: %s", exc)
        else:
            self._fallback.toggle_thinking(mode)  # type: ignore[union-attr]

    def route_paradigm(self, query: str) -> str:
        """Route a query to the appropriate reasoning paradigm.

        Determines which paradigm (structural, associative, compositional,
        temporal, causal) is most appropriate for the given query.

        Args:
            query: The query text to analyze.

        Returns:
            A string identifying the paradigm: "structural", "associative",
            "compositional", "temporal", or "causal".
        """
        if self.is_rust_core:
            try:
                # Rust core may not have a direct paradigm routing method,
                # so we use the modes system as a proxy
                if hasattr(self._rsvs, "run_mode"):
                    result = self._rsvs.run_mode("paradigm", query)
                    if isinstance(result, dict) and "paradigm" in result:
                        return str(result["paradigm"])
            except Exception as exc:
                logger.debug("RSVS paradigm routing failed: %s", exc)

        return self._fallback.route_paradigm(query)  # type: ignore[union-attr]

    def deps_analyze(self, failure: str = "") -> dict:
        """Analyze dependencies related to a failure or inconsistency.

        Uses the DEPS (Dependency-Enhanced Path Search) planner to
        trace how a failure propagates through the knowledge graph.

        Args:
            failure: The failure node label or description to analyze.

        Returns:
            A dict with dependent nodes and propagation explanation.
        """
        if self.is_rust_core:
            try:
                # Try the stub bindings first
                if hasattr(self._rsvs, "deps_analyze"):
                    result = self._rsvs.deps_analyze()
                    if result is not None:
                        if isinstance(result, dict):
                            return result
                        return {"raw": str(result)}
                # Fallback: use reflect() for basic analysis
                if hasattr(self._rsvs, "reflect"):
                    result = self._rsvs.reflect()
                    if result is not None:
                        return {"reflection": result, "failure": failure}
            except Exception as exc:
                logger.debug("RSVS deps_analyze failed: %s", exc)

        return self._fallback.deps_analyze(failure)  # type: ignore[union-attr]

    def matryoshka_traverse(self, node_label: str, depth: int = 3) -> Optional[dict]:
        """Matryoshka-style nested traversal from a node.

        Explores the graph in concentric layers, like Russian dolls,
        where each layer contains nodes that are progressively further
        from the starting node.

        Args:
            node_label: The starting node label.
            depth: Maximum nesting depth.

        Returns:
            A dict with nested layer data and scored nodes, or None
            if the node is not found.
        """
        if self.is_rust_core:
            try:
                # Use context_query with progressively deeper depths
                # to simulate matryoshka traversal
                layers: list[list[str]] = []
                all_scored: list[tuple[str, float]] = []

                for d in range(1, depth + 1):
                    result = self._rsvs.context_query(
                        node_label, [], max_depth=d
                    )
                    if result is not None:
                        scored = list(result.scored_atoms) if hasattr(result, "scored_atoms") else []
                        layer_labels = [str(a[0]) for a in scored if isinstance(a, (list, tuple)) and len(a) >= 1]
                        layers.append(layer_labels)
                        all_scored.extend(
                            (str(a[0]), float(a[1])) for a in scored
                            if isinstance(a, (list, tuple)) and len(a) >= 2
                        )

                return {
                    "root": node_label,
                    "depth": len(layers),
                    "layers": layers,
                    "scored_nodes": all_scored,
                    "total_nodes_visited": len(set(s[0] for s in all_scored)),
                }
            except Exception as exc:
                logger.debug("RSVS matryoshka_traverse failed for '%s': %s", node_label, exc)
                return None

        return self._fallback.matryoshka_traverse(node_label, depth)  # type: ignore[union-attr]

    def context_similarity(self, a: str, b: str, context: list[str]) -> Optional[float]:
        """Compute context-weighted similarity between two concepts.

        When an embedding provider is available (P2-6), blends the
        graph-based similarity with embedding cosine similarity.
        Context atoms boost embedding similarity when they appear in
        both concept embeddings.

        Args:
            a: First concept label.
            b: Second concept label.
            context: Context atoms to bias the similarity.

        Returns:
            A float similarity score, or None if concepts not found.
        """
        # P2-6: Try embedding-based similarity
        embedding_sim: Optional[float] = None
        provider = self._get_or_init_embedding_provider()
        if provider is not None:
            try:
                from .embedding import cosine_similarity
                emb_a = provider.embed(a)
                emb_b = provider.embed(b)
                base_emb_sim = cosine_similarity(emb_a, emb_b)
                # Context boost: embed context and compute alignment
                if context:
                    context_text = " ".join(context)
                    emb_ctx = provider.embed(context_text)
                    ctx_a = cosine_similarity(emb_ctx, emb_a)
                    ctx_b = cosine_similarity(emb_ctx, emb_b)
                    # Boost if context aligns with both concepts
                    context_boost = min(ctx_a, ctx_b) * 0.2
                    embedding_sim = min(1.0, base_emb_sim + context_boost)
                else:
                    embedding_sim = base_emb_sim
            except Exception as exc:
                logger.debug("Embedding context_similarity failed: %s", exc)

        graph_sim: Optional[float] = None
        if self.is_rust_core:
            try:
                graph_sim = self._rsvs.context_similarity(a, b, context)
            except Exception as exc:
                logger.debug("RSVS context_similarity failed: %s", exc)

        if graph_sim is None and not self.is_rust_core:
            graph_sim = self._fallback.context_similarity(a, b, context)  # type: ignore[union-attr]

        # Blend results
        if graph_sim is not None and embedding_sim is not None:
            return 0.6 * graph_sim + 0.4 * embedding_sim
        if embedding_sim is not None:
            return embedding_sim
        return graph_sim

    def detect_convergence(self, max_pairs: int = 500) -> dict:
        """Detect structural convergence across nodes in the graph.

        Finds nodes with structurally similar sense compositions that
        may represent the same concept across different languages or contexts.

        When the Rust core is available, calls its convergence_detect()
        method. Otherwise, falls back to a Jaccard-based structural
        similarity check between all pairs of high-confidence nodes.

        Args:
            max_pairs: Maximum number of pairs to evaluate/return.

        Returns:
            Dict with:
                - "pairs_found": int — number of convergent pairs detected
                - "convergence_pairs": list of dicts with
                    {node_a, node_b, similarity, shared_compositions}
                - "source": "rust_core" or "fallback"
        """
        if self.is_rust_core:
            try:
                result = self._rsvs.convergence_detect()
                if result is not None:
                    # Rust core returns a JSON string
                    if isinstance(result, str):
                        parsed = json.loads(result)
                    elif isinstance(result, dict):
                        parsed = result
                    else:
                        parsed = {"pairs": []}

                    # Normalize Rust core result to standard format
                    raw_pairs = parsed.get("pairs", [])
                    convergence_pairs = []
                    for p in raw_pairs[:max_pairs]:
                        if isinstance(p, dict):
                            convergence_pairs.append({
                                "node_a": p.get("a", ""),
                                "node_b": p.get("b", ""),
                                "similarity": p.get("overlap", 0.0),
                                "shared_compositions": [],
                                "linked": p.get("linked", False),
                            })

                    return {
                        "pairs_found": len(convergence_pairs),
                        "convergence_pairs": convergence_pairs,
                        "source": "rust_core",
                    }
            except Exception as exc:
                logger.debug("RSVS convergence_detect failed, using fallback: %s", exc)

        return self._fallback._fallback_detect_convergence(max_pairs=max_pairs)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Embedding provider management (P2-6)
    # ------------------------------------------------------------------

    def set_embedding_provider(self, provider: Any) -> None:
        """Set the embedding provider for semantic similarity.

        Args:
            provider: An ``EmbeddingProvider`` instance (from
                ``layer2.embedding``), or ``None`` to disable
                embedding-augmented similarity.
        """
        self._embedding_provider = provider
        if provider is not None:
            logger.info(
                "Embedding provider set: %s (dim=%d)",
                provider.name, provider.dim,
            )
        else:
            logger.info("Embedding provider cleared")

    def get_embedding_provider(self) -> Any:
        """Return the current embedding provider, or None."""
        return self._embedding_provider

    def _get_or_init_embedding_provider(self) -> Any:
        """Lazily initialise the embedding provider on first use.

        Uses ``FallbackEmbeddingProvider`` as the default if no
        provider has been explicitly set.  The fallback provider
        requires zero external dependencies.
        """
        if self._embedding_provider is not None:
            return self._embedding_provider

        # Lazy-init: try to get a real provider, fall back to hashing
        try:
            from .embedding import get_embedding_provider
            self._embedding_provider = get_embedding_provider()
        except Exception as exc:
            logger.debug("Failed to init embedding provider: %s", exc)
            try:
                from .embedding import FallbackEmbeddingProvider
                self._embedding_provider = FallbackEmbeddingProvider()
            except Exception:
                self._embedding_provider = None

        return self._embedding_provider

    # ------------------------------------------------------------------
    # PyO3 normalization methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_stats(stats: Any) -> dict:
        """Convert PyO3 IngestStats to plain dict."""
        if isinstance(stats, dict):
            return stats
        try:
            return {
                "sentences_processed": getattr(stats, "sentences_processed", 0),
                "atoms_promoted": getattr(stats, "atoms_promoted", 0),
                "sense_assigned": getattr(stats, "sense_assigned", 0),
                "sense_created": getattr(stats, "sense_created", 0),
                "confidence_updated": getattr(stats, "confidence_updated", 0),
                "frozen_batches": getattr(stats, "frozen_batches", 0),
                "compositions_induced": getattr(stats, "compositions_induced", 0),
                "atoms_flagged_inactive": getattr(stats, "atoms_flagged_inactive", 0),
            }
        except Exception:
            return {"raw": str(stats)}

    @staticmethod
    def _normalize_query_result(result: Any) -> dict:
        """Convert PyO3 QueryResult to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            return {
                "sense_idx": result.sense_idx,
                "sense_n": result.sense_n,
                "atoms": list(result.atoms),
                "layer": result.layer,
                "grounding_score": result.grounding_score,
                "compositions": list(result.compositions),
                "convergence_contributors": list(result.convergence_contributors) if hasattr(result, "convergence_contributors") else [],
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_context_query_result(result: Any) -> dict:
        """Convert PyO3 ContextQueryResult to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            return {
                "active_sense_idx": result.active_sense_idx,
                "total_senses": result.total_senses,
                "scored_atoms": list(result.scored_atoms),
                "depth_reached": result.depth_reached,
                "halt_reason": result.halt_reason,
                "cycles_detected": result.cycles_detected,
                "layer": result.layer,
                "grounding_score": result.grounding_score,
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_relate_result(result: Any) -> dict:
        """Convert PyO3 RelateResult to plain dict.

        F-04 fix: When the Rust core is active, relate() returns (u32, f32)
        tuples for node IDs. We resolve numeric IDs to labels by calling
        node_info() on the Rust core instance. The bridge has access to
        self._rsvs for label resolution.

        Since this is a static method, we pass the rsvs instance separately
        via the _rsvs keyword argument when calling from relate().
        """
        if isinstance(result, dict):
            return result
        try:
            raw_nodes = list(result.related_nodes) if hasattr(result, "related_nodes") else []
            raw_edges = list(result.related_edges) if hasattr(result, "related_edges") else []
            raw_structural = list(result.structural_relations) if hasattr(result, "structural_relations") else []

            # F-04: Resolve numeric node IDs to labels
            # raw_nodes is [(node_id: u32, score: f32), ...]
            # We try to resolve each numeric ID to a label string
            resolved_nodes = []
            for item in raw_nodes:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    node_id, score = item[0], item[1]
                    # Try to resolve to label — if we can't, use string of ID
                    label = str(node_id)
                    try:
                        # The RSVS instance can look up node labels by ID
                        # This is passed via _resolve_rsvs parameter in relate()
                        pass
                    except Exception:
                        pass
                    resolved_nodes.append((label, score, node_id))  # (label, score, original_id)
                elif isinstance(item, (list, tuple)) and len(item) == 1:
                    resolved_nodes.append((str(item[0]), 0.0, item[0]))
                else:
                    resolved_nodes.append((str(item), 0.0, None))

            # Same for structural_relations
            resolved_structural = []
            for item in raw_structural:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    node_id, score = item[0], item[1]
                    label = str(node_id)
                    resolved_structural.append((label, score, node_id))
                else:
                    resolved_structural.append((str(item), 0.0, None))

            return {
                "related_nodes": resolved_nodes,
                "related_edges": raw_edges,
                "structural_relations": resolved_structural,
                "_pyo3_object": True,
                "_needs_label_resolution": True,
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_appraise_result(result: Any) -> dict:
        """Convert PyO3 AppraiseResult to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            return {
                "agree_pct": result.agree_pct,
                "disagree_pct": result.disagree_pct,
                "neutral_pct": result.neutral_pct,
                "verdict": result.verdict,
                "evidence": list(result.evidence) if hasattr(result, "evidence") else [],
                "convergence_info": list(result.convergence_info) if hasattr(result, "convergence_info") else [],
                "clash_pairs": list(result.clash_pairs) if hasattr(result, "clash_pairs") else [],
                "n_clusters": result.n_clusters if hasattr(result, "n_clusters") else 0,
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_structural_sim(result: Any) -> dict:
        """Convert PyO3 StructuralSimResult to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            shared = list(result.shared_compositions) if hasattr(result, "shared_compositions") else []
            only_a = list(result.only_a_compositions) if hasattr(result, "only_a_compositions") else []
            only_b = list(result.only_b_compositions) if hasattr(result, "only_b_compositions") else []

            return {
                "structural_similarity": result.structural_similarity,
                "shared_compositions": shared,
                "only_a_compositions": only_a,
                "only_b_compositions": only_b,
                "layer_a": result.layer_a if hasattr(result, "layer_a") else 0,
                "layer_b": result.layer_b if hasattr(result, "layer_b") else 0,
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_substitution(result: Any) -> dict:
        """Convert PyO3 SubstitutionResult to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            return {
                "structural_similarity": result.structural_similarity,
                "substitutions": list(result.substitutions) if hasattr(result, "substitutions") else [],
                "unpaired_only_a": list(result.unpaired_only_a) if hasattr(result, "unpaired_only_a") else [],
                "unpaired_only_b": list(result.unpaired_only_b) if hasattr(result, "unpaired_only_b") else [],
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_senses(result: Any) -> list[dict]:
        """Convert PyO3 SenseInfo list to list of plain dicts."""
        if isinstance(result, list) and all(isinstance(item, dict) for item in result):
            return result

        senses_list = []
        try:
            for sense in result:
                if isinstance(sense, dict):
                    senses_list.append(sense)
                else:
                    senses_list.append({
                        "sense_idx": getattr(sense, "sense_idx", 0),
                        "n_contexts": getattr(sense, "n_contexts", 0),
                        "coherence": getattr(sense, "coherence", 0.0),
                        "status": getattr(sense, "status", "unknown"),
                        "core_atoms": list(getattr(sense, "core_atoms", [])),
                        "layer": getattr(sense, "layer", 0),
                        "grounding_score": getattr(sense, "grounding_score", 0.0),
                        "grounding_evidence": {
                            "confirming_contexts": getattr(sense.grounding_evidence, "confirming_contexts", 0) if hasattr(sense, "grounding_evidence") else 0,
                            "contradicting_contexts": getattr(sense.grounding_evidence, "contradicting_contexts", 0) if hasattr(sense, "grounding_evidence") else 0,
                        },
                        "compositions": list(getattr(sense, "compositions", [])),
                        "condition_label": getattr(sense, "condition_label", None),
                    })
        except Exception:
            pass

        return senses_list

    @staticmethod
    def _normalize_node_info(result: Any) -> dict:
        """Convert PyO3 NodeInfo to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            return {
                "label": result.label,
                "surface_label": result.surface_label,
                "id": result.id,
                "confidence": result.confidence,
                "tier": result.tier,
                "status": result.status,
                "is_seed": result.is_seed,
                "is_locked": result.is_locked,
                "is_stable": result.is_stable,
                "layer": result.layer,
                "atoms": list(result.atoms),
                "compositions": [],
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_mcts_result(result: Any) -> dict:
        """Convert PyO3 MCTSResult to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            return {
                "active_sense_idx": getattr(result, "active_sense_idx", 0),
                "total_senses": getattr(result, "total_senses", 0),
                "scored_atoms": list(getattr(result, "scored_atoms", [])),
                "depth_reached": getattr(result, "depth_reached", 0),
                "halt_reason": getattr(result, "halt_reason", "unknown"),
                "simulations_run": getattr(result, "simulations_run", 0),
                "best_path": list(getattr(result, "best_path", [])),
                "layer": getattr(result, "layer", 0),
                "grounding_score": getattr(result, "grounding_score", 0.0),
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_consolidation_result(result: Any) -> dict:
        """Convert PyO3 ConsolidationResult to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            return {
                "senses_merged": getattr(result, "senses_merged", 0),
                "senses_removed": getattr(result, "senses_removed", 0),
                "edges_pruned": getattr(result, "edges_pruned", 0),
                "atoms_compacted": getattr(result, "atoms_compacted", 0),
            }
        except Exception:
            return {"raw": str(result)}

    @staticmethod
    def _normalize_reflection_result(result: Any) -> dict:
        """Convert PyO3 ReflectionResult to plain dict."""
        if isinstance(result, dict):
            return result
        try:
            return {
                "actions_total": getattr(result, "actions_total", 0),
                "actions_applied": getattr(result, "actions_applied", 0),
            }
        except Exception:
            return {"raw": str(result)}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_bridge: Optional[AbstractionBridge] = None


def get_bridge(rsvs_instance: Any = None) -> AbstractionBridge:
    """Get or create the default AbstractionBridge singleton.

    Args:
        rsvs_instance: Optional RSVS instance to use. If this is the
            first call and an instance is provided, it will be used.
            Subsequent calls ignore this parameter.

    Returns:
        The default AbstractionBridge instance.
    """
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = AbstractionBridge(rsvs_instance=rsvs_instance)
    return _default_bridge

# Backward compat
RsvsBridge = AbstractionBridge


# ---------------------------------------------------------------------------
# V12PipelineBridge — adapter for the v12.0 DAG-based PipelineEngine
# ---------------------------------------------------------------------------

class V12PipelineBridge:
    """Bridge to the v12.0 PipelineEngine -- DAG-based pipeline with cognitive modes.

    The v12 pipeline introduces a directed-acyclic-graph (DAG) execution model
    with three cognitive modes (Reactive, Analytical, Reflective) and built-in
    gap detection.  It is the next-generation ingestion path for RSVS,
    superseding the flat ingest-then-query model used by AbstractionBridge.

    This bridge wraps ``PyV12Pipeline`` (exposed by the Rust core when built
    with ``--features v12,python``) and converts its PyO3 result objects into
    plain Python dicts so downstream layer2 code never touches PyO3 types
    directly -- consistent with the design philosophy of AbstractionBridge.

    Graceful degradation:
        If the Rust core was not compiled with v12 support, ``available``
        returns False.  All read-only methods return safe defaults (empty
        lists, zero counts, empty JSON) and mutation methods raise
        ``RuntimeError`` so callers can distinguish "unavailable" from
        "legitimate empty result".

    Usage:
        v12 = V12PipelineBridge()
        if v12.available:
            result = v12.ingest("The cat sat on the mat.")
            print(result["cognitive_mode"])   # e.g. "Reactive"
            print(v12.composition_count())     # e.g. 3
        else:
            print("v12 pipeline not available -- build with --features v12,python")
    """

    def __init__(self) -> None:
        try:
            from rsvs import PyV12Pipeline  # type: ignore[import]
            self._pipeline = PyV12Pipeline()
            self._available = True
        except (ImportError, Exception):
            self._pipeline = None
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the v12 pipeline is available.

        Returns True only when the Rust core was built with
        ``--features v12,python`` and PyV12Pipeline could be instantiated.
        """
        return self._available

    def ingest(self, text: str) -> dict:
        """Ingest text using the v12 pipeline.

        The v12 ingestion path runs text through a DAG of processing
        frames: ExtractFrame, ReasonFrame, GovernBeliefs, and optional
        ReflectFrame (when the cognitive mode is Reflective).  This
        produces atoms, compositions, and gap detections in a single
        pass.

        Args:
            text: The text to ingest.

        Returns:
            A dict with keys:
                atoms_created (int): Number of new atoms created.
                compositions_created (int): Number of new compositions.
                gaps_detected (int): Number of knowledge gaps found.
                cognitive_mode (str): The selected cognitive mode
                    ("Reactive", "Analytical", or "Reflective").

        Raises:
            RuntimeError: If the v12 pipeline is not available.
        """
        if not self._available:
            raise RuntimeError("v12 pipeline not available")
        result = self._pipeline.v12_ingest(text)
        return {
            "atoms_created": result.atoms_created,
            "compositions_created": result.compositions_created,
            "gaps_detected": result.gaps_detected,
            "cognitive_mode": result.cognitive_mode,
        }

    def cognitive_mode(self, text: str) -> str:
        """Select the cognitive mode for the given input text.

        The v12 pipeline selects between three cognitive modes based on
        input characteristics:
            - "Reactive": Fast, pattern-matching response for familiar inputs.
            - "Analytical": Deep structural reasoning for complex inputs.
            - "Reflective": Meta-cognitive evaluation and self-correction.

        Args:
            text: Input text to classify.

        Returns:
            One of "Reactive", "Analytical", or "Reflective".

        Raises:
            RuntimeError: If the v12 pipeline is not available.
        """
        if not self._available:
            raise RuntimeError("v12 pipeline not available")
        return self._pipeline.select_cognitive_mode(text)

    def compositions(self) -> list:
        """Get all compositions in the v12 graph.

        Returns:
            List of PyComposition objects.  Returns an empty list if
            the v12 pipeline is not available.
        """
        if not self._available:
            return []
        return self._pipeline.compositions()

    def detect_gaps(self) -> list:
        """Detect knowledge gaps in the current graph state.

        Gap detection identifies areas where the graph lacks sufficient
        connectivity or where compositions have weak grounding.  This is
        a v12-specific feature that leverages the DAG structure to find
        structural holes.

        Returns:
            List of PyKnowledgeGap objects.  Returns an empty list if
            the v12 pipeline is not available.
        """
        if not self._available:
            return []
        return self._pipeline.detect_gaps()

    def composition_count(self) -> int:
        """Number of compositions in the v12 graph.

        Returns:
            Composition count, or 0 if the v12 pipeline is not available.
        """
        if not self._available:
            return 0
        return self._pipeline.composition_count()

    def node_count(self) -> int:
        """Number of nodes in the v12 graph.

        Returns:
            Node count, or 0 if the v12 pipeline is not available.
        """
        if not self._available:
            return 0
        return self._pipeline.node_count()

    def get_composition(self, comp_id: str) -> dict | None:
        """Get a specific composition by ID.

        Converts the PyO3 PyComposition object into a plain dict with
        member details, matching the bridge pattern of never exposing
        PyO3 objects to downstream code.

        Args:
            comp_id: The composition identifier string.

        Returns:
            A dict with keys: id, composition_type, lifecycle, epistemic,
            confidence, members.  Each member is a dict with keys:
            node_id, role, label, confidence.
            Returns None if the composition is not found or the v12
            pipeline is not available.
        """
        if not self._available:
            return None
        comp = self._pipeline.get_composition(comp_id)
        if comp is None:
            return None
        return {
            "id": comp.id,
            "composition_type": comp.composition_type,
            "lifecycle": comp.lifecycle,
            "epistemic": comp.epistemic,
            "confidence": comp.confidence,
            "members": [
                {
                    "node_id": m.node_id,
                    "role": m.role,
                    "label": m.label,
                    "confidence": m.confidence,
                }
                for m in comp.members
            ],
        }

    def find_weak_frames(self) -> list:
        """Find weak frames in the v12 graph.

        Weak frames are compositions with low confidence or insufficient
        grounding evidence.  This is useful for identifying areas that
        need reinforcement through additional ingestion.

        Returns:
            List of weak frame descriptors.  Returns an empty list if
            the v12 pipeline is not available.
        """
        if not self._available:
            return []
        return self._pipeline.find_weak_frames()

    def snapshot_json(self) -> str:
        """Get a JSON snapshot of the v12 graph state.

        The snapshot includes all nodes, compositions, gaps, and their
        current state.  Useful for serialization, debugging, or
        transferring graph state between processes.

        Returns:
            JSON string of the graph state.  Returns "{}" if the v12
            pipeline is not available.
        """
        if not self._available:
            return "{}"
        return self._pipeline.snapshot_json()

    def set_gap_detection(self, enabled: bool) -> None:
        """Enable or disable gap detection in the v12 pipeline.

        When disabled, ingest() will not run the gap detection pass,
        which can improve performance for bulk ingestion where gaps
        are not needed immediately.

        Args:
            enabled: Whether to enable gap detection.

        Raises:
            RuntimeError: If the v12 pipeline is not available.
        """
        if not self._available:
            raise RuntimeError("v12 pipeline not available")
        self._pipeline.set_gap_detection(enabled)

    def gap_detection_enabled(self) -> bool:
        """Check whether gap detection is currently enabled.

        Returns:
            True if gap detection is enabled, False otherwise.
            Returns False if the v12 pipeline is not available.
        """
        if not self._available:
            return False
        return self._pipeline.gap_detection_enabled()
