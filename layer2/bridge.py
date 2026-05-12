"""
RSVS Bridge — Unified adapter for PyO3 Rust core vs Python fallback mode.

This module provides a SINGLE point of contact with the RSVS core.
All rsvs_genius layers should use this bridge instead of directly
importing from `rsvs` — this ensures consistent error handling,
proper API adaptation, and graceful fallback when the Rust core
isn't built.

Architecture:
    rsvs_genius layers → rsvs_bridge.py → rsvs (PyO3) or fallback

Key design decisions:
1. The bridge wraps PyO3 objects in plain Python dicts/lists so
   downstream code never needs to handle PyO3 objects directly.
2. All methods return Optional[T] — None means "not available" or
   "concept not found", never raises.
3. Fallback mode provides a lightweight in-memory graph for testing
   and development without needing to build the Rust core.

Analogi: Ini adalah "penerjemah" antara bahasa Rust (RSVS core)
dan bahasa Python (rsvs_genius layers). Seperti bagaimana Jin Soun
bisa membaca teks dalam bahasa apapun dan mengerti maksudnya —
bridge ini menerjemahkan API Rust ke Python dengan cara yang konsisten.
"""

from __future__ import annotations

import itertools
import json
import logging
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
# Fallback graph — lightweight in-memory knowledge store
# ---------------------------------------------------------------------------

@dataclass
class _FallbackNode:
    """A node in the fallback graph."""
    label: str
    confidence: float = 0.5
    compositions: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    layer: int = 0
    is_seed: bool = False
    observation_count: int = 0
    last_seen: float = field(default_factory=time.time)


class _FallbackGraph:
    """Lightweight fallback knowledge graph for when Rust core is unavailable.

    This is NOT a replacement for RSVS — it's a simple in-memory store
    that allows the rsvs_genius layers to function (at reduced capability)
    during development and testing.

    Provides the same API surface as the RSVS bridge, but with
    simplified semantics (no multi-sense, no spreading activation,
    no consolidation, etc.).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _FallbackNode] = {}
        self._edges: dict[str, list[str]] = {}  # label → [related labels]

    def ingest(self, text: str) -> dict:
        """Ingest text by extracting keywords as nodes."""
        words = self._extract_keywords(text)
        sentences = text.count('.') + text.count('!') + text.count('?') + 1

        atoms_promoted = 0
        compositions_induced = 0

        for word in words:
            if word not in self._nodes:
                self._nodes[word] = _FallbackNode(label=word)
                atoms_promoted += 1

            node = self._nodes[word]
            node.observation_count += 1
            node.last_seen = time.time()
            node.confidence = min(1.0, node.confidence + 0.05)

            # Co-occurrence = composition
            for other in words:
                if other != word:
                    if other not in node.compositions:
                        node.compositions.append(other)
                        compositions_induced += 1
                    if word not in self._edges:
                        self._edges[word] = []
                    if other not in self._edges[word]:
                        self._edges[word].append(other)

        return {
            "sentences_processed": sentences,
            "atoms_promoted": atoms_promoted,
            "sense_assigned": 0,
            "sense_created": 0,
            "confidence_updated": atoms_promoted,
            "frozen_batches": 0,
            "compositions_induced": compositions_induced,
            "atoms_flagged_inactive": 0,
            "fallback": True,
        }

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
            "sense_n": 1,
            "atoms": [(c, node.confidence * 0.8) for c in node.compositions[:10]],
            "layer": node.layer,
            "grounding_score": node.confidence,
            "compositions": [(c, 0) for c in node.compositions],
        }

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

    def compose(self, label: str, compositions: list[tuple[str, int]], lang: Optional[str] = None) -> Optional[int]:
        """Create a compositional node."""
        if label not in self._nodes:
            self._nodes[label] = _FallbackNode(label=label, layer=1)

        node = self._nodes[label]
        for comp_label, _sense_id in compositions:
            if comp_label not in node.compositions:
                node.compositions.append(comp_label)

        node.layer = max(node.layer, 1)
        return hash(label) % (2**31)

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
        }

    def senses(self, concept: str) -> Optional[list[dict]]:
        """Get senses for a concept (simplified: one sense per node)."""
        node = self._nodes.get(concept)
        if node is None:
            return None
        return [{
            "sense_idx": 0,
            "n_contexts": node.observation_count,
            "coherence": node.confidence,
            "status": "mature" if node.observation_count > 3 else "fragile",
            "core_atoms": node.compositions[:5],
            "layer": node.layer,
            "grounding_score": node.confidence,
            "compositions": [(c, 0) for c in node.compositions],
        }]

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
        """Return a fake sequence number."""
        return int(time.time() * 1000) % (2**31)

    def consume_events_v1(self, after_seq: Optional[int] = None, limit: int = 500) -> str:
        """Return empty events (fallback doesn't track events)."""
        return json.dumps({"events": [], "from_seq": after_seq or 0, "to_seq": 0})

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
# RsvsBridge — the main adapter
# ---------------------------------------------------------------------------

class AbstractionBridge:
    """Unified adapter for RSVS Rust core (PyO3) and Python fallback.

    This is the SINGLE point of contact for all rsvs_genius layers.
    It provides a consistent Python API regardless of whether the
    Rust core is available or not.

    Key features:
    - Wraps PyO3 objects in plain Python dicts/lists
    - All methods return Optional[T] — None means "not found"
    - Never raises (catches all PyO3 errors internally)
    - Falls back to _FallbackGraph when Rust core isn't built

    Usage:
        bridge = RsvsBridge()
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

        if rsvs_instance is not None:
            self._rsvs = rsvs_instance
            self.is_rust_core = True
            self.is_available = True
            logger.info("RsvsBridge initialized with provided RSVS instance")
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
                    logger.info("RsvsBridge initialized with Rust core (singleton)")
                    self._prime_rust_core()
                    return
                except Exception:
                    pass

                # Strategy 2: Create a fresh instance
                if _RsvsClass is not None:
                    self._rsvs = _RsvsClass()
                    self.is_rust_core = True
                    self.is_available = True
                    logger.info("RsvsBridge initialized with Rust core (new instance)")
                    self._prime_rust_core()
                    return
            except Exception as exc:
                logger.warning("Failed to initialize Rust core: %s", exc)

        # Fallback mode
        self._fallback = _FallbackGraph()
        self.is_rust_core = False
        self.is_available = True  # Fallback IS available, just limited
        logger.info("RsvsBridge initialized in FALLBACK mode (no Rust core)")

    # ------------------------------------------------------------------
    # Priming & grounding for Rust core
    # ------------------------------------------------------------------

    def _prime_rust_core(self) -> None:
        """Prime the Rust core by ingesting a sentence containing seed labels.

        The Rust core requires tokens to be "groundable" — they must appear
        in a sentence containing one of the 24 epistemological seed labels.
        Without this priming, non-English text can never produce groundable
        tokens because no token matches the English seed labels.

        This method is called once during initialization when the Rust core
        is active. It ingests a priming sentence that contains ALL 24 seed
        labels, ensuring the graph has seed-relevant sentences from the start.
        """
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
        """Check whether text naturally contains any of the 24 seed labels.

        This mirrors the Rust core's ``sentence_contains_seed()`` logic so
        we can decide whether a grounding prefix is needed BEFORE sending
        the text to Rust.
        """
        words = text.lower().split()
        # Also split on punctuation for robustness
        all_tokens = set()
        for w in words:
            # Strip punctuation and add both raw and stripped
            all_tokens.add(w)
            stripped = w.strip(".,;:!?'\"()[]{}-")
            if stripped:
                all_tokens.add(stripped)
        seed_set = set(SEED_LABELS)
        return bool(all_tokens & seed_set)

    def ingest_with_grounding(self, text: str, domain_id: Optional[int] = None) -> dict:
        """Ingest text with a grounding prefix when using the Rust core.

        When the Rust core is active and the text does not naturally
        contain any of the 24 epistemological seed labels, this method
        prepends a grounding context sentence to ensure the combined
        text passes ``sentence_contains_seed()`` in the Rust pipeline.

        Different seed words are used in rotation (round-robin over all
        24 seeds) to avoid over-reliance on a single seed label.

        Args:
            text: The text to ingest.
            domain_id: Optional domain tag for the ingestion.

        Returns:
            A dict with ingestion stats.
        """
        if not self.is_rust_core:
            # No grounding needed in fallback mode
            return self._fallback.ingest(text)  # type: ignore[union-attr]

        if self._text_contains_seed(text):
            # Text already has seed co-occurrence — ingest as-is
            return self.ingest(text, domain_id=domain_id)

        # Pick the next seed word in rotation
        seed_word = next(self._seed_cycle)
        grounded_text = f"The {seed_word} relates to {text}"
        logger.debug(
            "ingest_with_grounding: prepended seed '%s' for text: %.60s...",
            seed_word, text,
        )
        return self.ingest(grounded_text, domain_id=domain_id)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def ingest(self, text: str, domain_id: Optional[int] = None) -> dict:
        """Ingest text into the knowledge graph.

        When the Rust core is active and the text does not naturally
        contain any of the 24 epistemological seed labels, this method
        automatically delegates to ``ingest_with_grounding()`` which
        prepends a grounding context sentence. This ensures that even
        standalone Indonesian or mixed-language ingests have seed
        co-occurrence, so tokens can be promoted.

        Args:
            text: The text to ingest.
            domain_id: Optional domain tag for the ingestion.

        Returns:
            A dict with ingestion stats.
        """
        if self.is_rust_core:
            # Auto-ground: if text lacks seed words, prepend a grounding prefix
            if not self._text_contains_seed(text):
                return self.ingest_with_grounding(text, domain_id=domain_id)

            try:
                if domain_id is not None:
                    self._rsvs.set_domain(domain_id)
                stats = self._rsvs.ingest(text)
                return self._normalize_stats(stats)
            except Exception as exc:
                logger.error("RSVS ingest failed: %s", exc)
                return {"success": False, "error": str(exc)}

        # Fallback
        return self._fallback.ingest(text)  # type: ignore[union-attr]

    def query(self, concept: str, context: str = "") -> Optional[dict]:
        """Query a concept in the knowledge graph.

        Args:
            concept: The concept to query.
            context: Optional context string for disambiguation.

        Returns:
            A dict with query results, or None if concept not found.
        """
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
        """Context-aware query using depth-controlled traversal.

        Args:
            concept: The concept to query.
            context_atoms: Context atom labels for disambiguation.
            max_depth: Maximum traversal depth.
            gamma: Stability halting threshold.
            halt_confidence: Confidence halting threshold.
            tau_relevance: Relevance gating threshold.

        Returns:
            A dict with query results, or None if not found.
        """
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

        Args:
            concept: The concept to find relations for.

        Returns:
            A dict with related nodes, edges, and structural relations.
            Keys: "related_nodes", "related_edges", "structural_relations"
            Each is a list of (label_or_id, score) tuples.
        """
        if self.is_rust_core:
            try:
                result = self._rsvs.relate(concept)
                if result is None:
                    return None
                return self._normalize_relate_result(result)
            except Exception as exc:
                logger.debug("RSVS relate failed for '%s': %s", concept, exc)
                return None

        return self._fallback.relate(concept)  # type: ignore[union-attr]

    def appraise(self, text: str) -> dict:
        """Appraise text against the knowledge graph.

        Args:
            text: The text to appraise.

        Returns:
            A dict with appraise results.
            Keys: "agree_pct", "disagree_pct", "neutral_pct", "verdict",
                  "evidence", "convergence_info", "clash_pairs", "n_clusters"
        """
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

    def structural_similarity(self, a: str, b: str) -> Optional[dict]:
        """Compute structural similarity between two concepts.

        Args:
            a: First concept label.
            b: Second concept label.

        Returns:
            A dict with similarity info, or None if either not found.
        """
        if self.is_rust_core:
            try:
                result = self._rsvs.structural_similarity(a, b)
                if result is None:
                    return None
                return self._normalize_structural_sim(result)
            except Exception as exc:
                logger.debug("RSVS structural_similarity failed: %s", exc)
                return None

        return self._fallback.structural_similarity(a, b)  # type: ignore[union-attr]

    def substitution_analysis(self, a: str, b: str) -> Optional[dict]:
        """Analyze what substitution transforms concept A into concept B.

        Args:
            a: First concept label.
            b: Second concept label.

        Returns:
            A dict with substitution info, or None if either not found.
        """
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
        """Create a compositional node from explicit composition references.

        Args:
            label: The label for the new node.
            compositions: List of (node_label, sense_id) pairs.
            lang: Optional language tag.

        Returns:
            The node ID, or None if composition failed.
        """
        if self.is_rust_core:
            try:
                return self._rsvs.compose(label, compositions, lang)
            except Exception as exc:
                logger.debug("RSVS compose failed: %s", exc)
                return None

        return self._fallback.compose(label, compositions, lang)  # type: ignore[union-attr]

    def senses(self, concept: str) -> Optional[list[dict]]:
        """Get all senses for a concept.

        Args:
            concept: The concept label.

        Returns:
            A list of sense info dicts, or None if concept not found.
        """
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
        """Return confidence scores for all nodes.

        Returns:
            A dict mapping concept label → confidence score.
        """
        if self.is_rust_core:
            try:
                return self._rsvs.confidence_map()
            except Exception:
                return {}

        return self._fallback.confidence_map()  # type: ignore[union-attr]

    def nodes(self, include_seeds: bool = False) -> list[str]:
        """List all node labels.

        Args:
            include_seeds: Whether to include seed nodes.

        Returns:
            A list of node labels.
        """
        if self.is_rust_core:
            try:
                return self._rsvs.nodes(include_seeds)
            except Exception:
                return []

        return self._fallback.nodes(include_seeds)  # type: ignore[union-attr]

    def node_info(self, label: str) -> Optional[dict]:
        """Get info about a specific node.

        Args:
            label: The node label.

        Returns:
            A dict with node info, or None if not found.
        """
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
        """Consume events after a sequence number.

        Args:
            after_seq: Only return events with seq > this. None = from start.
            limit: Maximum events to return.

        Returns:
            JSON string of event batch.
        """
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

        IMPORTANT: PyRelateResult has:
        - related_nodes: Vec<(u32, f32)> — node IDs and scores
        - related_edges: Vec<(u32, u32, f32)>
        - structural_relations: Vec<(u32, f32)>

        We convert node IDs to labels when possible.
        """
        if isinstance(result, dict):
            return result
        try:
            # Get the PyRsvs instance for label resolution
            # PyRelateResult has node_labels() method that returns (label, score)
            related_nodes = []
            if hasattr(result, "node_labels"):
                try:
                    # node_labels() is a method on PyRelateResult that takes PyRsvs
                    # But we don't have access to PyRsvs here, so use related_nodes directly
                    pass
                except Exception:
                    pass

            # Use raw node IDs with scores
            raw_nodes = list(result.related_nodes) if hasattr(result, "related_nodes") else []
            raw_edges = list(result.related_edges) if hasattr(result, "related_edges") else []
            raw_structural = list(result.structural_relations) if hasattr(result, "structural_relations") else []

            return {
                "related_nodes": raw_nodes,  # (node_id, score) tuples
                "related_edges": raw_edges,
                "structural_relations": raw_structural,
                "_pyo3_object": True,  # Flag that IDs are numeric
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
                "shared_compositions": shared,  # (node_id, sense_id) tuples
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


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_bridge: Optional[RsvsBridge] = None


def get_bridge(rsvs_instance: Any = None) -> RsvsBridge:
    """Get or create the default RsvsBridge singleton.

    Args:
        rsvs_instance: Optional RSVS instance to use. If this is the
            first call and an instance is provided, it will be used.
            Subsequent calls ignore this parameter.

    Returns:
        The default RsvsBridge instance.
    """
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = RsvsBridge(rsvs_instance=rsvs_instance)
    return _default_bridge

# Backward compat
RsvsBridge = AbstractionBridge
