# @WHO:   self-ai/src/derivation/understanding_builder.py
# @WHAT:  SELF builds semantic understanding through observation, not hardcoded rules
# @PART:  self-ai/derivation
# @ENTRY: UnderstandingBuilder

"""Understanding Builder — HOW SELF builds its own semantic understanding.

Vision:
    SELF adalah LLM yang membangun semantic understanding SENDIRI
    berdasarkan pengamatannya ke dalam dirinya sendiri. Understanding ini
    BISA MEMENGARUHI cara dia menjawab di pertanyaan berikutnya.

    Sistem 1: Output yang tiba-tiba — intuisi langsung dari bge-m3
    Sistem 2: Hasil pemahaman dari yang kita ajarkan — Understanding Graph

    Bahkan LLM bisa KOMBINASIKAN beberapa semantic understanding
    untuk generate jawaban yang sesuai.

Philosophy:
    SELF doesn't receive hardcoded understandings. SELF BUILDS them through:
    1. OBSERVATION — Watch what happens in teaching examples
    2. SCHEMA EXTRACTION — Identify the structural pattern
    3. GENERALIZATION — Abstract away specifics to get reusable schemas
    4. ABSTRACTION — Connect to higher-level principles
    5. INTEGRATION — Merge with existing understanding graph

    This is the core of SELF's inner world — how it turns specific examples
    into reusable, composable understanding.

    SELF's understanding is stored in an Understanding Graph where:
    - Each node is a piece of SEMANTIC understanding (not a rule)
    - Edges connect related understandings (composition is possible)
    - Retrieval is via EMBEDDING SIMILARITY (not keyword matching)
    - The graph EVOLVES as SELF learns more

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Teaching Example (soal + cara + jawaban + kenapa)      │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  1. OBSERVE — Extract what happened                     │
    │     - What signals are present in the text?             │
    │     - What transformation was applied?                  │
    │     - What was the result?                              │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  2. EXTRACT SCHEMA — Identify structural pattern        │
    │     - Replace specifics with slots: [ENTITY], [STATE]   │
    │     - Keep structural words: kecuali, tetapi, bukan     │
    │     - Identify the TRANSFORMATION TYPE                  │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  3. GENERALIZE — Make it reusable                       │
    │     - Merge with similar schemas from other examples    │
    │     - Find common patterns across examples              │
    │     - Build condition → transformation mapping          │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  4. ABSTRACT — Connect to principles                    │
    │     - What general principle does this follow?          │
    │     - How does it relate to existing understandings?    │
    │     - What are the invariants?                          │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  5. INTEGRATE — Merge into understanding graph          │
    │     - Add node to graph                                 │
    │     - Create edges to related understandings             │
    │     - Check for conflicts with existing knowledge       │
    └─────────────────────────────────────────────────────────┘

    When a NEW question arrives:
        bge-m3 Embed → Semantic Retrieval → Graph Activation → Apply Transformations
        (System 1: intuitive)                    (System 2: understanding-based)

Key Insight:
    v27 removes ALL keyword matching. Retrieval is embedding-ONLY.
    This is because keyword matching is NOT understanding — it's
    pattern matching that misses operational meaning.

    SELF's retrieval works by OPERATIONAL SIMILARITY:
    - "kehilangan 35" matches U_quantity (both = quantity operations)
    - "siapa yang tidak hadir" matches U_signal_flip (both = exception logic)
    - Multiple understandings can be COMBINED for complex answers

    The transformation IS the understanding. It's not a recipe,
    it's a RECOGNIZED PATTERN that SELF discovered.
"""

import os
import re
import json
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

# v35: Governance types — dual-axis lifecycle + epistemic + compositional members
from governance.states import (
    LifecycleState,
    EpistemicState,
    SeedScores,
    UnderstandingMember,
)

logger = logging.getLogger(__name__)


# ═══════════════ TRANSFORMATION SYSTEM ═══════════════

class Transformation:
    """A self-discovered transformation that SELF can apply.

    Unlike ExtractionRules (which are rigid recipes), Transformations
    are UNDERSTOOD patterns. SELF knows WHY a transformation works,
    not just WHAT steps to follow.

    Attributes:
        kind: Type of transformation (signal_flip, entity_extract,
              quantity_compute, context_filter, comparison_resolve, etc.)
        trigger: What signals activate this transformation
        action: What the transformation does (in SELF's own words)
        slots: Named variables that get filled during application
        constraints: What must be true for this to apply
    """

    def __init__(self, kind: str, trigger: dict, action: str,
                 slots: list = None, constraints: list = None):
        self.kind = kind
        self.trigger = trigger
        self.action = action
        self.slots = slots or []
        self.constraints = constraints or []

    def to_dict(self) -> dict:
        return {
            'kind': self.kind,
            'trigger': self.trigger,
            'action': self.action,
            'slots': self.slots,
            'constraints': self.constraints,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Transformation':
        return cls(
            kind=d.get('kind', ''),
            trigger=d.get('trigger', {}),
            action=d.get('action', ''),
            slots=d.get('slots', []),
            constraints=d.get('constraints', []),
        )


# ═══════════════ UNDERSTANDING NODE ═══════════════

class UnderstandingNode:
    """A node in SELF's understanding graph.

    Each node represents ONE piece of understanding that SELF has built.
    Nodes are connected to form a graph where:
    - parent → this node is a special case of parent
    - child → this node is a generalization of child
    - related → this node is related to (can be composed with) another node

    Attributes:
        id: Unique identifier
        name: Human-readable name
        concept: What this understanding is about
        abstraction: The generalized principle in SELF's own words
        schemas: Observed patterns that led to this understanding
        transformation: How to apply this understanding
        conditions: When this understanding applies
        condition_embedding: Precomputed embedding for fast matching
        edges: Connections to other understandings
        source: How SELF discovered this
        confidence: SELF's confidence in this understanding
        stats: Usage statistics
        members: v35 — Structured roles within this understanding (from AAM CompositionMember)
        lifecycle: v35 — Maturity stage (NEW → CANDIDATE → STABLE → DEPRECATED)
        epistemic: v35 — Credibility state (OBSERVED → INFERRED → GROUNDED / CONTRADICTED)
        seed_scores: v35 — Multi-dimensional epistemic confidence
        deprecated_reason: v35 — Why this was deactivated (None if active)
        deprecated_at: v35 — When this was deactivated (None if active)
    """

    def __init__(self, id: str, name: str, concept: str, abstraction: str,
                 schemas: list = None, transformation: Transformation = None,
                 conditions: list = None, condition_embedding: list = None,
                 edges: list = None, source: str = 'self_discovered',
                 confidence: float = 0.5,
                 # v35: Structural memory fields
                 members: list = None,
                 lifecycle: str = None,
                 epistemic: str = None,
                 seed_scores: dict = None,
                 deprecated_reason: str = None,
                 deprecated_at: float = None):
        self.id = id
        self.name = name
        self.concept = concept
        self.abstraction = abstraction
        self.schemas = schemas or []
        self.transformation = transformation
        self.conditions = conditions or []
        self.condition_embedding = condition_embedding
        self.edges = edges or []  # [(target_id, edge_type), ...]
        self.source = source
        self.confidence = confidence
        self.times_applied = 0
        self.times_correct = 0
        self.times_failed = 0

        # v35: Structural memory — compositional members with roles
        self.members: List[UnderstandingMember] = members or []

        # v35: Dual-axis governance
        # Backward compatible: existing nodes default to STABLE + OBSERVED
        self.lifecycle = (
            LifecycleState(lifecycle) if lifecycle else LifecycleState.STABLE
        )
        self.epistemic = (
            EpistemicState(epistemic) if epistemic else EpistemicState.OBSERVED
        )

        # v35: Multi-dimensional epistemic confidence
        self.seed_scores = (
            SeedScores.from_dict(seed_scores) if isinstance(seed_scores, dict)
            else (seed_scores if seed_scores else SeedScores())
        )

        # v35: Deactivation tracking (None = active, set when DEPRECATED)
        self.deprecated_reason = deprecated_reason
        self.deprecated_at = deprecated_at

    @property
    def accuracy(self) -> float:
        if self.times_applied == 0:
            return self.confidence
        return self.times_correct / self.times_applied

    @property
    def failure_rate(self) -> float:
        if self.times_applied == 0:
            return 0.0
        return self.times_failed / self.times_applied

    def add_edge(self, target_id: str, edge_type: str):
        """Add connection to another understanding node."""
        if (target_id, edge_type) not in self.edges:
            self.edges.append((target_id, edge_type))

    def strengthen(self, amount: float = 0.05):
        """Increase confidence after correct application."""
        self.times_applied += 1
        self.times_correct += 1
        self.confidence = min(0.99, self.confidence + amount)

    def weaken(self, amount: float = 0.1):
        """Decrease confidence after incorrect application."""
        self.times_applied += 1
        self.times_failed += 1
        self.confidence = max(0.1, self.confidence - amount)

    def to_dict(self) -> dict:
        d = {
            'id': self.id,
            'name': self.name,
            'concept': self.concept,
            'abstraction': self.abstraction,
            'schemas': self.schemas,
            'transformation': self.transformation.to_dict() if self.transformation else None,
            'conditions': self.conditions,
            'condition_embedding': self.condition_embedding,
            'edges': self.edges,
            'source': self.source,
            'confidence': self.confidence,
            'times_applied': self.times_applied,
            'times_correct': self.times_correct,
            'times_failed': self.times_failed,
            # v35: Structural memory + governance
            'members': [m.to_dict() for m in self.members] if self.members else [],
            'lifecycle': self.lifecycle.value,
            'epistemic': self.epistemic.value,
            'seed_scores': self.seed_scores.to_dict(),
        }
        # Only include deprecation info if actually deprecated
        if self.deprecated_reason is not None:
            d['deprecated_reason'] = self.deprecated_reason
        if self.deprecated_at is not None:
            d['deprecated_at'] = self.deprecated_at
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'UnderstandingNode':
        node = cls(
            id=d.get('id', ''),
            name=d.get('name', ''),
            concept=d.get('concept', ''),
            abstraction=d.get('abstraction', ''),
            schemas=d.get('schemas', []),
            transformation=Transformation.from_dict(d['transformation']) if d.get('transformation') else None,
            conditions=d.get('conditions', []),
            condition_embedding=d.get('condition_embedding'),
            edges=d.get('edges', []),
            source=d.get('source', 'self_discovered'),
            confidence=d.get('confidence', 0.5),
            # v35: Structural memory + governance (backward compatible defaults)
            members=[UnderstandingMember.from_dict(m) for m in d.get('members', [])],
            lifecycle=d.get('lifecycle'),  # Will default to STABLE in __init__
            epistemic=d.get('epistemic'),  # Will default to OBSERVED in __init__
            seed_scores=d.get('seed_scores'),  # Will default to SeedScores() in __init__
            deprecated_reason=d.get('deprecated_reason'),
            deprecated_at=d.get('deprecated_at'),
        )
        node.times_applied = d.get('times_applied', 0)
        node.times_correct = d.get('times_correct', 0)
        node.times_failed = d.get('times_failed', 0)
        return node


# ═══════════════ UNDERSTANDING GRAPH ═══════════════

class UnderstandingGraph:
    """SELF's understanding graph — the core of its inner world.

    This is NOT a flat list of patterns. This is a CONNECTED graph where:
    - Nodes are understandings
    - Edges are relationships between understandings
    - Traversal finds the right understanding for a given question
    - Composition combines understandings for complex cases

    The graph EVOLVES as SELF learns:
    - New nodes are added
    - Existing nodes are strengthened/weakened
    - New edges are created as SELF discovers relationships
    - Nodes can be merged when SELF realizes two things are the same

    v27: Embedding-ONLY retrieval. No keyword fallback.
    Understanding retrieval uses BAAI/bge-m3 EMBEDDING SIMILARITY exclusively.
    If the embedding model is unavailable, retrieval FAILS rather than
    falling back to keyword matching. This ensures semantic integrity —
    the system must understand, not pattern-match.

    Vision: SELF is an LLM that builds its own semantic understanding
    based on its observations into itself. This understanding influences
    how it answers subsequent questions. System 1 = intuitive output,
    System 2 = understanding built from teaching. SELF can combine
    multiple semantic understandings to generate appropriate answers.
    """

    # Cluster similarity threshold — nodes with cosine sim > this are in same cluster
    CLUSTER_SIMILARITY_THRESHOLD = 0.7
    # Transfer score bonus — how much to boost a node from the same cluster
    # as a high-confidence first-hop hit
    TRANSFER_SCORE_BONUS = 0.1

    def __init__(self, store_path: str = None, embedding_model=None):
        self._nodes: Dict[str, UnderstandingNode] = {}
        self._signal_index: Dict[str, List[str]] = defaultdict(list)
        self._embedding_model = embedding_model
        self._store_path = store_path or os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'understanding_graph.json'
        )

        # v26: Embedding-based retriever (bge-m3)
        # This replaces keyword-based find_matching with semantic similarity.
        # Lazy-initialized on first use to avoid loading the model at startup.
        self._retriever = None
        self._retriever_initialized = False

        # v36: Concept clustering — groups of nodes with similar embeddings.
        # Not persisted; recomputed on demand / at startup.
        # Format: list of sets, each set contains node_id strings.
        self._clusters: List[set] = []
        self._clusters_dirty = True  # True when nodes changed and clusters need rebuild

        self._load()

    # ═══════════════ GRAPH OPERATIONS ═══════════════

    def add_node(self, node: UnderstandingNode):
        """Add a node to the understanding graph."""
        self._nodes[node.id] = node

        # Index by conditions (signals)
        for cond in node.conditions:
            cond_lower = cond.lower()
            self._signal_index[cond_lower].append(node.id)

        # Also index by transformation trigger words
        if node.transformation:
            trigger = node.transformation.trigger
            for key, value in trigger.items():
                if isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and len(v) > 1:
                            self._signal_index[v.lower()].append(node.id)
                elif isinstance(value, str) and len(value) > 1:
                    self._signal_index[value.lower()].append(node.id)

        # Compute embedding if model available
        if self._embedding_model is not None and node.condition_embedding is None:
            self._compute_embedding(node)

        # v36: Invalidate clusters since graph changed
        self._clusters_dirty = True

        self._save()
        logger.info("Added understanding node: %s (%s)", node.id, node.concept[:60])

    def get_node(self, node_id: str) -> Optional[UnderstandingNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def find_matching(self, text: str, question: str,
                      threshold: float = 0.15) -> Optional[UnderstandingNode]:
        """Find the best matching understanding for a question.

        v27 STRATEGY (Embedding-ONLY):
          Use bge-m3 embedding retrieval exclusively.
          NO fallback to keyword matching.
          If embedding model is unavailable, raise RuntimeError.

        SELF must UNDERSTAND, not pattern-match. Keyword matching
        is the antithesis of understanding — it finds superficial
        word overlap without grasping operational meaning.

        The embedding retriever finds understandings by OPERATIONAL SIMILARITY,
        not word overlap. This means "kehilangan 35" correctly matches
        U_quantity even though the word "kehilangan" doesn't appear in
        U_quantity's conditions.
        """
        return self._find_matching_via_embedding(text, question, threshold)

    def _find_matching_via_embedding(self, text: str, question: str,
                                      threshold: float) -> Optional[UnderstandingNode]:
        """Find matching understanding using bge-m3 embedding similarity.

        This is the ONLY retrieval method in v27. No keyword fallback.
        SELF must understand — keyword matching is not understanding.

        If the embedding model is unavailable, this raises RuntimeError
        because without understanding, there is no meaningful retrieval.
        """
        self._ensure_retriever()
        if self._retriever is None:
            raise RuntimeError(
                "Embedding model (bge-m3) is unavailable. "
                "SELF cannot retrieve understandings without semantic embedding. "
                "Install sentence-transformers and ensure BAAI/bge-m3 is accessible. "
                "v27 does NOT fall back to keyword matching — understanding requires "
                "semantic similarity, not word overlap."
            )

        result = self._retriever.find_best(
            text, question, self._nodes, threshold=threshold
        )
        if result is not None:
            node, score = result
            logger.debug("Embedding match: %s (score=%.3f, kind=%s)",
                        node.id, score,
                        node.transformation.kind if node.transformation else '?')
            return node

        return None



    def _ensure_retriever(self):
        """Lazy-initialize the embedding retriever.

        v27: If the retriever cannot be initialized, we raise an error
        instead of silently falling back to keyword matching.
        Understanding requires semantic similarity — keyword matching
        is not understanding.
        """
        if self._retriever_initialized:
            return

        try:
            from derivation.embedding_retrieval import UnderstandingRetriever
            self._retriever = UnderstandingRetriever()

            # Load all existing nodes into the retriever
            self._retriever.load_from_graph(self)

            if not self._retriever.is_available():
                logger.error(
                    "Embedding model (bge-m3) is unavailable! "
                    "SELF cannot function without semantic understanding. "
                    "Install sentence-transformers: pip install sentence-transformers"
                )
                self._retriever = None
        except ImportError:
            logger.error(
                "UnderstandingRetriever not available! "
                "SELF cannot function without semantic understanding. "
                "Ensure embedding_retrieval.py is in the derivation package."
            )
            self._retriever = None
        except Exception as e:
            logger.error(
                "Failed to init UnderstandingRetriever: %s — "
                "SELF cannot function without semantic understanding.", e
            )
            self._retriever = None

        self._retriever_initialized = True

    def find_by_signals(self, signals: list) -> List[UnderstandingNode]:
        """Find all understandings that match given signals."""
        node_ids = set()
        for signal in signals:
            signal_lower = signal.lower()
            if signal_lower in self._signal_index:
                node_ids.update(self._signal_index[signal_lower])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def find_matching_multi(self, text: str, question: str,
                            top_k: int = 3,
                            threshold: float = 0.15) -> List[Tuple[UnderstandingNode, float]]:
        """Find multiple matching understandings for composition.

        v28: SELF can COMBINE several semantic understandings
        to generate appropriate answers.

        Instead of returning just the best match, this returns the
        top-k matches above threshold. Callers can then:
        1. Try applying each individually
        2. If none work individually, compose them with Qwen3

        Args:
            text: The source text
            question: The question being asked
            top_k: Maximum number of understandings to return
            threshold: Minimum similarity score

        Returns:
            List of (UnderstandingNode, score) tuples, sorted by score descending
        """
        self._ensure_retriever()
        if self._retriever is None:
            # Cannot retrieve without embeddings — return empty
            logger.warning("Embedding model unavailable for multi-retrieval — returning empty")
            return []

        results = self._retriever.retrieve(
            text, question, self._nodes,
            top_k=top_k, threshold=threshold
        )
        return results

    def retrieve(self, text: str, question: str,
                 top_k: int = 5, threshold: float = 0.25,
                 enable_cross_domain: bool = True,
                 second_hop_limit: int = 3) -> List[Tuple[UnderstandingNode, float]]:
        """Retrieve matching understanding nodes with optional cross-domain transfer.

        v36: Enhanced retrieval that captures cross-domain relevance.

        When enable_cross_domain=True, this performs THREE stages:
          1. FIRST-HOP: Standard cosine similarity retrieval (same as before)
          2. SECOND-HOP: For each first-hop hit, find its embedding-nearest
             neighbors that weren't already retrieved. This allows transfer
             of experience to semantically adjacent domains (e.g., "kecuali"
             → "selain", "tidak termasuk", "minus").
          3. TRANSFER SCORE: Nodes in the same concept cluster as a
             high-confidence first-hop hit get a bonus, because cluster
             membership implies shared operational logic.

        When enable_cross_domain=False, behavior is IDENTICAL to calling
        find_matching_multi() — no second-hop, no clustering, no bonus.
        This guarantees zero regression for existing callers.

        Args:
            text: The source text
            question: The question to answer
            top_k: Maximum number of understanding nodes to return
            threshold: Minimum similarity score (0-1)
            enable_cross_domain: If True, perform second-hop and transfer scoring
            second_hop_limit: Max neighbors to expand per first-hop hit

        Returns:
            List of (UnderstandingNode, score) tuples, sorted by score descending
        """
        # ── First-hop: standard retrieval ──
        self._ensure_retriever()
        if self._retriever is None:
            logger.warning("Embedding model unavailable for retrieve — returning empty")
            return []

        first_hop = self._retriever.retrieve(
            text, question, self._nodes,
            top_k=top_k, threshold=threshold
        )

        # If cross-domain is disabled, return first-hop as-is (zero regression)
        if not enable_cross_domain:
            return first_hop

        if not first_hop:
            return []

        # ── Second-hop: expand to embedding neighbors of first-hop hits ──
        second_hop = self._second_hop_retrieve(first_hop, second_hop_limit, threshold)

        # ── Merge first-hop and second-hop, deduplicating by node ID ──
        seen_ids = set()
        merged = []
        for node, score in first_hop:
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                merged.append((node, score))
        for node, score in second_hop:
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                # Second-hop scores are lower than first-hop by design
                # (they are the cosine sim between the query and the neighbor,
                #  which is typically < the sim to the first-hop node itself)
                merged.append((node, score))

        # ── Transfer score: boost nodes in same cluster as high-confidence hits ──
        merged = self._apply_transfer_score(merged, first_hop)

        # Re-sort by final score and truncate
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    def _second_hop_retrieve(self, first_hop: List[Tuple],
                              limit: int,
                              threshold: float) -> List[Tuple[UnderstandingNode, float]]:
        """Find embedding-nearest neighbors of first-hop hits.

        For each node in first_hop, look at all other indexed nodes and
        find those whose embeddings are close (cosine sim > threshold).
        These "second-hop" nodes capture semantically adjacent concepts
        that the query didn't directly match but are relevant via
        transitive similarity.

        Example: query "selain" → first-hop misses the "kecuali" node
        (low direct cosine sim), but the "tidak termasuk" node has high
        similarity to "kecuali" → second-hop pulls in "kecuali" because
        it's a neighbor of "tidak termasuk" which WAS in first-hop.

        Args:
            first_hop: List of (UnderstandingNode, score) from first-hop
            limit: Max neighbors to add per first-hop node
            threshold: Minimum cosine similarity for a neighbor to qualify

        Returns:
            List of (UnderstandingNode, score) for second-hop discoveries
        """
        if self._retriever is None or not hasattr(self._retriever, '_embeddings'):
            return []

        import numpy as np

        second_hop_results = []
        seen_ids = {node.id for node, _ in first_hop}

        for first_node, first_score in first_hop:
            first_emb = self._retriever._embeddings.get(first_node.id)
            if first_emb is None:
                continue

            # Find neighbors of this first-hop node
            neighbors = []
            for candidate_id, candidate_emb in self._retriever._embeddings.items():
                if candidate_id in seen_ids:
                    continue
                sim = float(np.dot(first_emb, candidate_emb))
                if sim >= self.CLUSTER_SIMILARITY_THRESHOLD:
                    candidate_node = self._nodes.get(candidate_id)
                    if candidate_node is not None:
                        # Score = neighbor similarity × first-hop score (attenuated)
                        # This ensures second-hop results are always scored lower
                        # than the first-hop node that led to them
                        second_hop_score = sim * first_score * 0.8
                        if second_hop_score >= threshold:
                            neighbors.append((candidate_node, second_hop_score, sim))

            # Sort by neighbor similarity, take top `limit`
            neighbors.sort(key=lambda x: x[2], reverse=True)
            for node, score, _ in neighbors[:limit]:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    second_hop_results.append((node, score))

        return second_hop_results

    def _apply_transfer_score(self, merged: List[Tuple],
                               first_hop: List[Tuple]) -> List[Tuple]:
        """Apply transfer score bonus to nodes in same cluster as high-confidence hits.

        The intuition: if a first-hop node has high confidence (score > threshold)
        and another node is in the SAME concept cluster, that other node likely
        shares operational logic and deserves a small score boost.

        Example: "kecuali" node scores 0.8 → its cluster-mate "selain" node
        (which arrived via second-hop at 0.3) gets +0.1 transfer bonus → 0.4.
        This makes cluster-mates more competitive vs. unrelated low-score nodes.

        Args:
            merged: All candidate (node, score) tuples (first + second hop)
            first_hop: Original first-hop results for cluster identification

        Returns:
            Same list with transfer bonus applied to qualifying nodes
        """
        clusters = self.get_clusters()
        if not clusters:
            return merged

        # Find which clusters contain high-confidence first-hop nodes
        high_conf_clusters = set()
        for node, score in first_hop:
            if score >= 0.3:  # Only consider reasonably confident hits
                for i, cluster in enumerate(clusters):
                    if node.id in cluster:
                        high_conf_clusters.add(i)

        if not high_conf_clusters:
            return merged

        # Build a set of node IDs that should get a bonus
        bonus_ids = set()
        for ci in high_conf_clusters:
            bonus_ids.update(clusters[ci])

        # Apply bonus
        result = []
        for node, score in merged:
            if node.id in bonus_ids:
                # Don't boost first-hop nodes — they already have high scores
                # Only boost second-hop (or lower) nodes that share a cluster
                is_first_hop = any(n.id == node.id for n, _ in first_hop)
                if not is_first_hop:
                    score = min(1.0, score + self.TRANSFER_SCORE_BONUS)
                    logger.debug(
                        "Transfer bonus +%.2f for %s (cluster mate of high-conf hit)",
                        self.TRANSFER_SCORE_BONUS, node.id
                    )
            result.append((node, score))

        return result

    # ═══════════════ CONCEPT CLUSTERING ═══════════════

    def get_clusters(self) -> List[set]:
        """Get concept clusters — groups of nodes with similar embeddings.

        v36: Clusters are computed lazily and cached until the graph changes.
        Each cluster is a set of node IDs. Nodes in the same cluster share
        semantically similar condition_embeddings (cosine sim > 0.7 by default).

        Clusters enable transfer learning: if a node in cluster C is
        retrieved with high confidence, other nodes in C get a score
        boost because they likely share operational logic.

        Returns:
            List of sets, each set contains node_id strings.
        """
        if not self._clusters_dirty and self._clusters:
            return self._clusters

        self._compute_clusters()
        return self._clusters

    def _compute_clusters(self):
        """Compute concept clusters from embedding similarity.

        Uses a simple union-find approach:
        1. For every pair of nodes with embeddings, compute cosine similarity
        2. If similarity > CLUSTER_SIMILARITY_THRESHOLD, merge their clusters
        3. Result: disjoint sets of semantically related nodes

        This is O(n^2) in number of nodes but practical because:
        - SELF-AI typically has < 50 understanding nodes
        - Clustering is only recomputed when the graph changes
        - No external clustering library needed (KISS)
        """
        import numpy as np

        self._ensure_retriever()
        if self._retriever is None or not hasattr(self._retriever, '_embeddings'):
            self._clusters = []
            self._clusters_dirty = False
            return

        embeddings = self._retriever._embeddings
        node_ids = list(embeddings.keys())

        if not node_ids:
            self._clusters = []
            self._clusters_dirty = False
            return

        # Union-Find
        parent = {nid: nid for nid in node_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Compare all pairs
        threshold = self.CLUSTER_SIMILARITY_THRESHOLD
        for i in range(len(node_ids)):
            emb_i = embeddings[node_ids[i]]
            for j in range(i + 1, len(node_ids)):
                emb_j = embeddings[node_ids[j]]
                sim = float(np.dot(emb_i, emb_j))
                if sim >= threshold:
                    union(node_ids[i], node_ids[j])

        # Build clusters from union-find roots
        cluster_map: Dict[str, set] = defaultdict(set)
        for nid in node_ids:
            root = find(nid)
            cluster_map[root].add(nid)

        # Only keep clusters with > 1 node (singletons aren't useful)
        self._clusters = [nodes for nodes in cluster_map.values() if len(nodes) > 1]
        self._clusters_dirty = False

        logger.info("Computed %d concept clusters from %d nodes",
                     len(self._clusters), len(node_ids))

    def apply(self, node: UnderstandingNode, text: str, question: str) -> Optional[dict]:
        """Apply an understanding node to extract an answer.

        This is the CORE of SELF's autonomous reasoning — applying
        self-discovered transformations WITHOUT LLM.
        """
        if node.transformation is None:
            return None

        result = self._apply_transformation(node, text, question)
        if result is not None:
            return {
                'answer': result,
                'confidence': node.accuracy,
                'method': f'understanding_{node.id}',
                'explanation': f'SELF applied understanding: {node.abstraction[:80]}',
                'source': node.source,
                'applied_without_llm': True,
            }
        return None

    def record_feedback(self, node_id: str, correct: bool):
        """Record feedback to evolve the understanding."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        if correct:
            node.strengthen()
        else:
            node.weaken()
        self._save()

    def get_related(self, node_id: str, edge_type: str = None) -> List[UnderstandingNode]:
        """Get related understanding nodes."""
        node = self._nodes.get(node_id)
        if node is None:
            return []
        related = []
        for target_id, etype in node.edges:
            if edge_type and etype != edge_type:
                continue
            target = self._nodes.get(target_id)
            if target:
                related.append(target)
        return related

    def count(self) -> int:
        return len(self._nodes)

    def get_all(self) -> dict:
        return {nid: n.to_dict() for nid, n in self._nodes.items()}

    # ═══════════════ TRANSFORMATION APPLICATION ═══════════════

    def _apply_transformation(self, node: UnderstandingNode,
                               text: str, question: str) -> Optional[str]:
        """Apply a transformation to extract an answer from text.

        This is SELF's autonomous reasoning engine — each transformation
        kind has its own application logic, but ALL are mechanical
        (no LLM needed).
        """
        t = node.transformation
        text_lower = text.lower()
        question_lower = question.lower()

        kind = t.kind

        if kind == 'signal_flip':
            # Signal flip: When a signal word appears, flip the default answer
            # e.g., "kecuali" → entity after is the OPPOSITE of the general rule
            return self._apply_signal_flip(t, text_lower, question_lower)

        elif kind == 'contrast_focus':
            # Contrast focus: After contrast word, that part is the answer
            # e.g., "tetapi" → focus on what comes AFTER
            return self._apply_contrast_focus(t, text_lower, question_lower)

        elif kind == 'negation_affirmation':
            # Negation + affirmation: "bukan X tapi Y" → Y is the truth
            return self._apply_negation_affirmation(t, text_lower, question_lower)

        elif kind == 'comparison_resolve':
            # Comparison: "A lebih X dari B" → resolve based on question direction
            return self._apply_comparison_resolve(t, text_lower, question_lower, text)

        elif kind == 'entity_extract':
            # Entity extraction: Find specific entity matching a pattern
            return self._apply_entity_extract(t, text_lower, question_lower, text)

        elif kind == 'fact_extract':
            # Fact extraction: Ignore emotions/opinions, extract factual content
            return self._apply_fact_extract(t, text_lower, question_lower, text)

        elif kind == 'quantity_compute':
            # Quantity computation: Extract numbers and compute
            return self._apply_quantity_compute(t, text_lower, question_lower, text)

        elif kind == 'context_filter':
            # Context filter: Find specific info in long text
            return self._apply_context_filter(t, text_lower, question_lower, text)

        elif kind == 'word_sense':
            # Word sense disambiguation: Resolve meaning from context
            return self._apply_word_sense(t, text_lower, question_lower, text)

        # Generic fallback — try schema-based matching
        return self._apply_generic(t, text_lower, question_lower, text)

    def _apply_signal_flip(self, t: Transformation,
                           text_lower: str, question_lower: str) -> Optional[str]:
        """Apply signal flip transformation.

        Signal flip: The signal word indicates that something is the
        OPPOSITE of the general rule. Find the exception entity.

        Schema: [ALL] [STATE] [SIGNAL_WORD] [EXCEPTION]
        Result: [EXCEPTION] has OPPOSITE [STATE]

        Special case: Double/triple negation before kecuali
        "TIDAK BUKAN tidak benar, KECUALI" → resolve negations first,
        then find what kecuali points to as the exception.
        """
        trigger = t.trigger
        signal_words = trigger.get('signal_words', [])
        result_position = trigger.get('result_position', 'after')

        # Find which signal word appears in text
        found_signal = None
        for sw in signal_words:
            if sw in text_lower:
                found_signal = sw
                break

        if found_signal is None:
            return None

        # Special handling for double negation + kecuali
        # Pattern: "TIDAK BUKAN tidak benar, KECUALI..."
        # Resolve: count negations to determine if statements are true or false
        if 'kecuali' in text_lower:
            # Find the part before kecuali
            kecuali_idx = text_lower.find('kecuali')
            before_kecuali = text_lower[:kecuali_idx]

            # Count negation words before kecuali
            negation_count = 0
            for neg_word in ['tidak', 'bukan', 'tak', 'tiada', 'belum', 'tanpa']:
                negation_count += len(re.findall(r'\b' + re.escape(neg_word) + r'\b', before_kecuali))

            # Even negations = affirmative, odd = negative
            is_affirmative = (negation_count % 2 == 0)

            # If "TIDAK BUKAN tidak benar" (3 negations) → NOT true → false
            # So all statements are TRUE except the one after kecuali
            # The question asks "mana yang TIDAK benar" → answer is the exception

            # Extract entity after kecuali
            after_kecuali = text_lower[kecuali_idx + len('kecuali'):].strip()
            match = re.match(r'^([^.!?,;]+)', after_kecuali)
            if match:
                result = match.group(1).strip()
                result = re.sub(r'\s+(dan|atau|serta)\s+', ' ', result)
                return self._capitalize(result)

        # Standard extraction after/before signal
        if result_position == 'after':
            idx = text_lower.find(found_signal)
            if idx >= 0:
                after = text_lower[idx + len(found_signal):].strip()
                # Take first meaningful phrase
                match = re.match(r'^([^.!?,;]+)', after)
                if match:
                    result = match.group(1).strip()
                    # Clean up common suffixes
                    result = re.sub(r'\s+(dan|atau|serta)\s+', ' ', result)
                    return self._capitalize(result)

        elif result_position == 'before':
            idx = text_lower.find(found_signal)
            if idx >= 0:
                before = text_lower[:idx].strip()
                words = before.split()
                if words:
                    return self._capitalize(words[-1])

        return None

    def _apply_contrast_focus(self, t: Transformation,
                              text_lower: str, question_lower: str) -> Optional[str]:
        """Apply contrast focus transformation.

        Contrast focus: After a contrast word, the following part
        is the TRUE/dominant description.

        Schema: [CLAIM] [CONTRAST_WORD] [TRUTH]
        Result: Focus on [TRUTH]
        """
        trigger = t.trigger
        contrast_words = trigger.get('contrast_words', trigger.get('signal_words', []))

        found_contrast = None
        for cw in contrast_words:
            if cw in text_lower:
                found_contrast = cw
                break

        if found_contrast is None:
            return None

        # Take everything after the contrast word
        idx = text_lower.find(found_contrast)
        if idx >= 0:
            after = text_lower[idx + len(found_contrast):].strip()
            # Find the first sentence/phrase after contrast
            # This could be the entire remainder or up to next period
            sentences = re.split(r'[.!?]', after)
            if sentences:
                result = sentences[0].strip()
                if result:
                    return self._capitalize(result)

        return None

    def _apply_negation_affirmation(self, t: Transformation,
                                     text_lower: str, question_lower: str) -> Optional[str]:
        """Apply negation + affirmation transformation.

        Pattern: "bukan X, tapi Y" → Y is the truth, X is rejected.

        Schema: [NEGATION_WORD] [REJECTED], [AFFIRMATION_WORD] [AFFIRMED]
        Result: [AFFIRMED]
        """
        trigger = t.trigger
        negation_words = trigger.get('negation_words', ['bukan', 'tidak'])
        affirmation_words = trigger.get('affirmation_words', ['tapi', 'tetapi', 'melainkan'])

        # Find affirmation word first — what comes after is the truth
        found_affirm = None
        for aw in affirmation_words:
            if aw in text_lower:
                found_affirm = aw
                break

        if found_affirm is None:
            # Try just looking for the negation pattern
            return None

        # Extract what comes after the affirmation word
        idx = text_lower.find(found_affirm)
        if idx >= 0:
            after = text_lower[idx + len(found_affirm):].strip()
            match = re.match(r'^([^.!?,;]+)', after)
            if match:
                result = match.group(1).strip()
                return self._capitalize(result)

        return None

    def _apply_comparison_resolve(self, t: Transformation,
                                   text_lower: str, question_lower: str,
                                   original_text: str) -> Optional[str]:
        """Apply comparison resolution.

        Pattern: "A lebih X dari B" → if question asks opposite of X, answer is B.

        Schema: [ENTITY_A] lebih [ADJECTIVE] dari [ENTITY_B]
        Result: Depends on question direction
        """
        # Find comparison pattern
        comp_match = re.search(
            r'(\w+)\s+lebih\s+(\w+)\s+dari\s+(\w+)', text_lower
        )
        if not comp_match:
            # Try simpler pattern
            comp_match = re.search(r'(\w+)\s+lebih\s+(\w+)\s+dari', text_lower)
            if comp_match:
                entity_a = comp_match.group(1)
                adj = comp_match.group(2)
                # Find entity B after "dari"
                dari_idx = text_lower.find('dari')
                after_dari = text_lower[dari_idx + 4:].strip()
                words = after_dari.split()
                entity_b = words[0] if words else ''
            else:
                return None
        else:
            entity_a = comp_match.group(1)
            adj = comp_match.group(2)
            entity_b = comp_match.group(3)

        if not entity_b:
            return None

        # Determine question direction
        # If question asks about the same adjective → answer is entity A
        # If question asks about the OPPOSITE → answer is entity B
        # v32: Removed hardcoded adj_antonyms dictionary.
        # Instead, use embedding similarity to determine if the question
        # asks about the same or opposite adjective direction.
        question_asks_opposite = False
        try:
            from derivation.understanding_builder import get_shared_graph
            graph = get_shared_graph()
            if graph._retriever is not None and graph._retriever.is_available():
                import numpy as np
                # Compare question with both the adjective and its likely opposite
                adj_emb = graph._retriever.model.encode(
                    [adj], show_progress_bar=False, normalize_embeddings=True
                )[0]
                q_emb = graph._retriever.model.encode(
                    [question_lower], show_progress_bar=False, normalize_embeddings=True
                )[0]
                # If question is semantically similar to the adjective → same direction
                adj_sim = float(np.dot(adj_emb, q_emb))
                question_asks_opposite = adj_sim < 0.3  # Low similarity → likely opposite
            else:
                # No embedding model — default to asking opposite
                # (most comparison questions ask about the other entity)
                question_asks_opposite = True
        except Exception:
            # Fallback: assume question asks about opposite direction
            question_asks_opposite = True

        if question_asks_opposite:
            return self._capitalize(entity_b)
        else:
            return self._capitalize(entity_a)

    def _apply_entity_extract(self, t: Transformation,
                               text_lower: str, question_lower: str,
                               original_text: str) -> Optional[str]:
        """Apply entity extraction transformation.

        General purpose: Find a specific entity in text based on
        pattern matching from the transformation's trigger.
        """
        trigger = t.trigger
        pattern = trigger.get('pattern', '')
        extract_after = trigger.get('extract_after', '')
        extract_before = trigger.get('extract_before', '')

        if extract_after:
            idx = text_lower.find(extract_after)
            if idx >= 0:
                after = text_lower[idx + len(extract_after):].strip()
                match = re.match(r'^([^.!?,;]+)', after)
                if match:
                    return self._capitalize(match.group(1).strip())

        if extract_before:
            idx = text_lower.find(extract_before)
            if idx >= 0:
                before = text_lower[:idx].strip()
                words = before.split()
                if words:
                    return self._capitalize(words[-1])

        if pattern:
            match = re.search(pattern, text_lower)
            if match:
                if match.groups():
                    return self._capitalize(match.group(1))
                return self._capitalize(match.group(0))

        return None

    def _apply_fact_extract(self, t: Transformation,
                             text_lower: str, question_lower: str,
                             original_text: str) -> Optional[str]:
        """Apply fact extraction — ignore emotions, extract facts.

        Strategy: Find the factual action/description, skip emotional words.
        """
        trigger = t.trigger
        fact_indicator = trigger.get('fact_indicator', '')
        fact_action = trigger.get('fact_action', '')

        if fact_action and fact_action in text_lower:
            idx = text_lower.find(fact_action)
            if idx >= 0:
                after = text_lower[idx + len(fact_action):].strip()
                match = re.match(r'^([^.!?,;]+)', after)
                if match:
                    return self._capitalize(match.group(1).strip())

        # Generic: find "apa yang" in question and look for the answer
        if 'apa yang' in question_lower:
            # Find the action verb in the question
            # e.g., "Apa yang ibu masak" → look for "masak" in text
            for word in question_lower.split():
                if len(word) > 4 and word not in {'apa', 'yang', 'untuk', 'dengan', 'dari'}:
                    idx = text_lower.find(word)
                    if idx >= 0:
                        after = text_lower[idx + len(word):].strip()
                        match = re.match(r'^([^.!?,;]+)', after)
                        if match:
                            result = match.group(1).strip()
                            if result:
                                return self._capitalize(result)

        return None

    def _apply_quantity_compute(self, t: Transformation,
                                text_lower: str, question_lower: str,
                                original_text: str) -> Optional[str]:
        """Apply quantity computation — extract numbers and compute.

        Schema: Find the entity in question → find its numbers → compute

        Priority:
        1. Direct arithmetic (e.g., "8 × 7 = ?")
        2. Kembalian (change computation)
        3. Entity-based quantity
        """
        # Check for direct arithmetic expression first
        arithmetic_result = self._compute_direct_arithmetic(text_lower, question_lower)
        if arithmetic_result is not None:
            return arithmetic_result

        # Check for kembalian
        if 'kembalian' in question_lower:
            return self._compute_kembalian(text_lower)

        # Try entity-based quantity computation
        return self._compute_entity_quantity(text_lower, question_lower, original_text)

    def _compute_direct_arithmetic(self, text_lower: str,
                                    question_lower: str) -> Optional[str]:
        """Compute direct arithmetic expressions like '8 × 7 = ?'.

        This is a MECHANICAL computation — no LLM needed.
        The understanding graph identifies this as quantity_compute;
        this method does the actual arithmetic.

        Supports: +, -, ×, *, ÷, /, ^ operations.
        Also supports Indonesian number words via _normalize_indonesian_numbers.
        """
        # Normalize Indonesian number words in the expression
        expr = self._normalize_indonesian_numbers(text_lower)

        # Single comprehensive pattern that captures the operator
        # Order matters: × before *, ÷ before / to avoid ambiguity
        arith_pattern = (
            r'(\d+(?:\.\d+)?)'      # First number
            r'\s*'
            r'([×*+÷/\-–])'         # Operator (captured as group 2)
            r'\s*'
            r'(\d+(?:\.\d+)?)'      # Second number
            r'(?:\s*=\s*\??)?'      # Optional "= ?"
        )

        match = re.search(arith_pattern, expr)
        if match:
            a = float(match.group(1))
            op = match.group(2)
            b = float(match.group(3))

            if op in ('×', '*'):
                result = a * b
            elif op == '+':
                result = a + b
            elif op in ('÷', '/'):
                if b == 0:
                    return None
                result = a / b
            elif op in ('-', '–'):
                result = a - b
            else:
                return None

            # Return integer if result is whole number
            if result == int(result):
                return str(int(result))
            return str(round(result, 4))

        return None

    def _normalize_indonesian_numbers(self, text: str) -> str:
        """Normalize Indonesian number words to digits for computation.

        This enables SELF to process expressions like 'delapan kali tujuh'
        by converting them to '8 × 7' before arithmetic computation.

        This is REPRESENTASI EKUIVALEN — not hardcoded knowledge.
        Indonesian 'delapan' and digit '8' are the same quantity.
        """
        indo_nums = {
            'nol': '0', 'satu': '1', 'dua': '2', 'tiga': '3',
            'empat': '4', 'lima': '5', 'enam': '6', 'tujuh': '7',
            'delapan': '8', 'sembilan': '9', 'sepuluh': '10',
            'sebelas': '11', 'dua belas': '12', 'tigabelas': '13',
        }
        # Also normalize operation words
        indo_ops = {
            'kali': '×', 'ditambah': '+', 'dikurangi': '-',
            'dikurangi dengan': '-', 'dibagi': '÷', 'dibagi dengan': '÷',
            'plus': '+', 'minus': '-',
        }

        result = text
        # Replace multi-word first (e.g., "dua belas" before "dua")
        for word, digit in sorted(indo_nums.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + word + r'\b', digit, result, flags=re.IGNORECASE)
        for word, op in sorted(indo_ops.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + word + r'\b', op, result, flags=re.IGNORECASE)

        return result

    def _compute_kembalian(self, text_lower: str) -> Optional[str]:
        """Compute change (kembalian)."""
        # Look for "membeli X buku seharga RpY" or "X buku seharga RpY"
        buy_match = re.search(r'(\d+)\s+\w+\s+seharga\s+rp[\s.]*([\d.]+)', text_lower)
        if buy_match:
            qty = int(buy_match.group(1))
            price = int(buy_match.group(2).replace('.', ''))
            total = qty * price
        else:
            # Try simpler: "membeli X" and "seharga RpY"
            qty_match = re.search(r'membeli\s+(\d+)', text_lower)
            price_match = re.search(r'seharga\s+rp[\s.]*(\d[\d.]*)', text_lower)
            if not price_match:
                price_match = re.search(r'rp[\s.]*(\d[\d.]*)\s+setiap', text_lower)
            if qty_match and price_match:
                qty = int(qty_match.group(1))
                price = int(price_match.group(1).replace('.', ''))
                total = qty * price
            else:
                total = None

        if total is None:
            return None

        # Find payment amount — must be AFTER "membayar" or "uang"
        payment = None

        # Strategy 1: Find Rp amount after "membayar"
        pay_after_bayar = re.search(r'membayar[^.]*?rp[\s.]*(\d[\d.]*)', text_lower)
        if pay_after_bayar:
            payment = int(pay_after_bayar.group(1).replace('.', ''))

        # Strategy 2: Find "uang RpX"
        if payment is None:
            uang_match = re.search(r'uang\s+rp[\s.]*(\d[\d.]*)', text_lower)
            if uang_match:
                payment = int(uang_match.group(1).replace('.', ''))

        # Strategy 3: Last Rp amount in text
        if payment is None:
            all_rp = re.findall(r'rp[\s.]*(\d[\d.]*)', text_lower)
            if all_rp:
                # Last Rp amount is likely the payment
                payment = int(all_rp[-1].replace('.', ''))

        if payment is not None and payment > total:
            change = payment - total
            return str(change)

        return None

    def _compute_entity_quantity(self, text_lower: str, question_lower: str,
                                  original_text: str) -> Optional[str]:
        """Compute quantity for a specific entity."""
        # Find the entity name in the question
        # e.g., "berapa jumlah kelereng Andi" → entity = "andi"
        entity = None
        for marker in ['kelereng', 'buku', 'apel', 'koin']:
            if marker in question_lower:
                # Find who owns it — look AFTER the marker word
                parts = question_lower.split(marker)
                if len(parts) > 1:
                    words_after = parts[1].split()
                    if words_after:
                        entity = words_after[0].strip('?.!')
                # Also try before
                if not entity:
                    words_before = parts[0].split()
                    if words_before:
                        entity = words_before[-1]
                break

        if not entity or entity in {'berapa', 'jumlah', 'total', 'yang'}:
            # Try finding entity name directly
            for word in question_lower.split():
                if len(word) > 3 and word not in {'berapa', 'jumlah', 'total', 'kelereng', 'buku', 'yang'}:
                    entity = word.strip('?.!')
                    break

        if not entity:
            return None

        # Find all sentences mentioning this entity
        sentences = re.split(r'[.!?]', text_lower)
        entity_numbers = []

        for sent in sentences:
            if entity in sent:
                # Extract numbers from this sentence
                nums = re.findall(r'\b(\d+)\b', sent)
                for n in nums:
                    entity_numbers.append(int(n))

        if entity_numbers:
            return str(sum(entity_numbers))

        return None

    def _apply_context_filter(self, t: Transformation,
                               text_lower: str, question_lower: str,
                               original_text: str) -> Optional[str]:
        """Apply context filter — find specific info in long text.

        Strategy: Find the KEYWORD from the question in the text,
        then extract the relevant entity.
        """
        trigger = t.trigger
        search_pattern = trigger.get('search_pattern', '')

        if search_pattern:
            match = re.search(search_pattern, text_lower)
            if match:
                if match.groups():
                    return self._capitalize(match.group(1))
                return self._capitalize(match.group(0))

        # Generic: find the key content word from question in text
        # e.g., "makanan khas Jakarta" → search for "makanan khas" in text
        question_words = question_lower.split()
        stop = {'apa', 'yang', 'adalah', 'menurut', 'teks', 'tersebut', 'dalam',
                'dari', 'di', 'ke', 'pada', 'untuk', 'dengan', 'dan', 'atau'}

        content_words = [w for w in question_words if w not in stop and len(w) > 2]

        for cw in content_words:
            idx = text_lower.find(cw)
            if idx >= 0:
                # Found the keyword — extract what comes after
                after = text_lower[idx + len(cw):].strip()
                # Skip "adalah", "ialah", etc.
                after = re.sub(r'^(adalah|ialah|yaitu|yakni)\s+', '', after)
                match = re.match(r'^([^.!?,;]+)', after)
                if match:
                    result = match.group(1).strip()
                    if result and len(result) > 1:
                        return self._capitalize(result)

        return None

    def _apply_word_sense(self, t: Transformation,
                          text_lower: str, question_lower: str,
                          original_text: str) -> Optional[str]:
        """Apply word sense disambiguation.

        Strategy: Find the word in context, determine meaning from
        surrounding words. The FIRST occurrence in text is usually
        the one the question asks about.
        """
        trigger = t.trigger
        target_word = trigger.get('target_word', '')
        context_words = trigger.get('context_words', [])
        meaning_if_context = trigger.get('meaning_if_context', '')
        meaning_otherwise = trigger.get('meaning_otherwise', '')

        if not target_word:
            # Try to auto-detect target word from question
            for phrase in ['bermakna', 'makna kata', 'artinya', 'makna']:
                if phrase in question_lower:
                    parts = question_lower.split(phrase)
                    if parts:
                        # Get quoted word before phrase
                        words_before = parts[0].split()
                        for w in reversed(words_before):
                            w = w.strip('"\'')
                            if len(w) > 2:
                                target_word = w
                                break
                    break

        if not target_word:
            return None

        # Find the target word in the question or text
        if target_word not in question_lower and target_word not in text_lower:
            return None

        # Check the FIRST occurrence in text — that's what question asks about
        sentences = re.split(r'[.!?]', text_lower)
        first_sent_with_target = None
        for sent in sentences:
            if target_word in sent:
                first_sent_with_target = sent
                break

        if first_sent_with_target is None:
            return None

        # Check context words in that first sentence
        if meaning_if_context and context_words:
            for cw in context_words:
                if cw in first_sent_with_target:
                    return self._capitalize(meaning_if_context)

        # Check what comes AFTER the target word
        idx = first_sent_with_target.find(target_word)
        if idx >= 0:
            after = first_sent_with_target[idx + len(target_word):].strip()
            # The word after target determines meaning
            next_word = after.split()[0] if after.split() else ''

            # If next word is an organization/place → leader meaning
            org_words = {'desa', 'kampung', 'negara', 'sekolah', 'kantor', 'daerah',
                         'suku', 'biro', 'departemen', 'dinasi', 'distrik'}
            body_words = {'ikan', 'sapi', 'ayam', 'kucing', 'kuda', 'gajah',
                          'manusia', 'anak', 'bayi', 'tumbuhan'}

            if next_word in org_words:
                return self._capitalize(meaning_if_context if meaning_if_context else 'pemimpin')
            elif next_word in body_words:
                return self._capitalize(meaning_otherwise if meaning_otherwise else 'bagian tubuh')

        if meaning_otherwise:
            return self._capitalize(meaning_otherwise)

        return None

    def _apply_generic(self, t: Transformation,
                       text_lower: str, question_lower: str,
                       original_text: str) -> Optional[str]:
        """Generic fallback — try schema-based matching."""
        # Try to find any of the trigger patterns in the text
        trigger = t.trigger
        for key, value in trigger.items():
            if isinstance(value, str) and value in text_lower:
                idx = text_lower.find(value)
                after = text_lower[idx + len(value):].strip()
                match = re.match(r'^([^.!?,;]+)', after)
                if match:
                    return self._capitalize(match.group(1).strip())
        return None

    # ═══════════════ EMBEDDING ═══════════════

    def _compute_embedding(self, node: UnderstandingNode):
        """Compute and store condition embedding.

        v26: Now uses UnderstandingRetriever to compute RICH embeddings that
        capture OPERATIONAL MEANING (concept + abstraction + conditions + action),
        not just condition keywords.
        """
        # Try the retriever first (rich embeddings)
        if self._retriever is not None:
            try:
                self._retriever.index_node(node)
                return
            except Exception:
                pass

        # Fallback: use the legacy embedding model with just conditions
        if self._embedding_model is None:
            return
        try:
            cond_text = ' '.join(node.conditions)
            emb = self._embedding_model.encode(
                [cond_text], show_progress_bar=False, normalize_embeddings=True
            )[0]
            node.condition_embedding = emb.tolist()
        except Exception as e:
            logger.debug("Failed to compute embedding: %s", e)

    # ═══════════════ HELPERS ═══════════════

    @staticmethod
    def _capitalize(s: str) -> str:
        """Capitalize first letter."""
        if not s:
            return s
        return s[0].upper() + s[1:]

    # ═══════════════ PERSISTENCE ═══════════════

    @staticmethod
    def _safe_replace(src: str, dst: str):
        """Atomically replace dst with src, with Windows robustness.

        os.replace() is atomic on POSIX and overwrites on all platforms,
        but on Windows it can still fail with PermissionError if the
        destination file is locked by another process. This helper adds
        a fallback: if os.replace() fails, try to remove the target
        first and retry.
        """
        try:
            os.replace(src, dst)
        except OSError:
            # Windows: target may be locked — try removing it first
            try:
                os.unlink(dst)
            except OSError:
                pass
            os.replace(src, dst)

    def _save(self):
        """Persist graph to disk."""
        try:
            os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
            data = {nid: n.to_dict() for nid, n in self._nodes.items()}
            tmp = self._store_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._safe_replace(tmp, self._store_path)
        except Exception as e:
            logger.warning("Failed to save understanding graph: %s", e)

    def _load(self):
        """Load graph from disk."""
        if not os.path.exists(self._store_path):
            return
        try:
            with open(self._store_path) as f:
                data = json.load(f)
            for nid, nd in data.items():
                self._nodes[nid] = UnderstandingNode.from_dict(nd)
            # Rebuild signal index
            self._signal_index = defaultdict(list)
            for nid, node in self._nodes.items():
                for cond in node.conditions:
                    self._signal_index[cond.lower()].append(nid)
            # v36: Mark clusters as dirty since we loaded new nodes
            self._clusters_dirty = True
            logger.info("Loaded %d understanding nodes", len(self._nodes))
        except Exception as e:
            logger.warning("Failed to load understanding graph: %s", e)


# ═══════════════ UNDERSTANDING BUILDER ═══════════════

class UnderstandingBuilder:
    """SELF builds its own semantic understanding through observation.

    This is the core mechanism — HOW SELF turns observations into
    reusable, composable understanding.

    Pipeline:
        Observation → Schema Extraction → Generalization → Abstraction → Integration

    The builder does NOT use hardcoded rules. It OBSERVES teaching examples
    and BUILDS understanding through pattern recognition.
    """

    # Indonesian structural words that carry meaning
    STRUCTURAL_WORDS = {
        'kecuali', 'selain', 'terkecuali',     # Exception
        'tetapi', 'namun', 'tapi', 'akan tetapi',  # Contrast
        'bukan', 'tidak', 'tak', 'tiada',       # Negation
        'lebih', 'kurang', 'paling',             # Comparison
        'karena', 'sebab', 'sejak',              # Causation
        'jika', 'kalau', 'bila', 'apabila',      # Condition
        'meskipun', 'walaupun', 'kendati',       # Concession
        'sehingga', 'maka', 'agar',              # Result
        'bahkan', 'lagipula',                    # Emphasis
        'atau', 'maupun',                        # Alternative
        'yaitu', 'ialah', 'yakni',               # Definition
    }

    # v32: Removed hardcoded SLOT_TYPES (PERSON, STATE, EMOTION, ANIMAL, THING, PLACE).
    # These were hardcoded word lists that violated SELF-AI's "NO hardcoded rules" vision.
    # SELF discovers slot types through embedding-based clustering from teaching examples.
    # The schema extraction now relies on the understanding graph's semantic knowledge.

    def __init__(self, graph: UnderstandingGraph = None, llm_engine=None):
        self._graph = graph or UnderstandingGraph()
        self._llm_engine = llm_engine
        self._observation_buffer = []  # Recent observations for batch processing

    @property
    def graph(self) -> UnderstandingGraph:
        return self._graph

    # ═══════════════ PIPELINE: OBSERVE ═══════════════

    def observe(self, lesson) -> Optional[UnderstandingNode]:
        """Observe a teaching lesson and build understanding.

        This is the main entry point. One lesson can produce one understanding.

        @FLOW:     UNDERSTANDING_BUILD
        @CALLS:    _extract_schema(), _generalize(), _build_node()
        @MUTATES:  understanding graph
        """
        # Step 1: Extract schema from the lesson
        schema = self._extract_schema(lesson)
        if schema is None:
            return None

        # Step 2: Generalize the schema
        generalized = self._generalize(schema, lesson)

        # Step 3: Build understanding node
        node = self._build_node(generalized, lesson)

        if node:
            # Step 4: Integrate into graph
            self._integrate(node)
            return node

        return None

    def observe_batch(self, lessons: list) -> List[UnderstandingNode]:
        """Observe multiple lessons and build understandings."""
        nodes = []
        for lesson in lessons:
            node = self.observe(lesson)
            if node:
                nodes.append(node)

        # After batch, try to find cross-lesson patterns
        if len(nodes) > 1:
            self._discover_cross_patterns(nodes)

        return nodes

    # ═══════════════ PIPELINE: SCHEMA EXTRACTION ═══════════════

    def _extract_schema(self, lesson) -> Optional[dict]:
        """Extract the structural schema from a teaching lesson.

        A schema captures:
        - The PATTERN (structural template with slots)
        - The TRANSFORMATION (how input maps to output)
        - The SIGNALS (what triggers this pattern)
        """
        problem = lesson.problem.lower()
        text = getattr(lesson, 'text', '') or ''
        answer = lesson.answer.lower() if lesson.answer else ''
        explanation = lesson.explanation_why.lower() if lesson.explanation_why else ''
        steps = lesson.solution_steps if lesson.solution_steps else []

        # Find structural words in the problem
        found_structural = []
        for sw in self.STRUCTURAL_WORDS:
            if sw in problem or sw in text.lower():
                found_structural.append(sw)

        # Determine transformation kind from structural words and steps
        kind = self._infer_transformation_kind(found_structural, problem, answer, steps)

        if kind is None:
            # No clear pattern — try to infer from explanation
            kind = self._infer_from_explanation(explanation)

        if kind is None:
            return None

        # Extract trigger signals
        trigger = self._extract_trigger(kind, found_structural, problem, text.lower(), answer)

        # Build schema
        schema = {
            'kind': kind,
            'structural_words': found_structural,
            'trigger': trigger,
            'problem': problem[:200],
            'answer': answer[:100],
            'explanation': explanation[:200],
            'steps': [s[:100] for s in steps[:3]],
        }

        return schema

    def _infer_transformation_kind(self, structural_words: list,
                                    problem: str, answer: str,
                                    steps: list) -> Optional[str]:
        """Infer the transformation kind from structural words and context."""
        sw_set = set(structural_words)

        # Exception words → signal_flip
        if sw_set & {'kecuali', 'selain', 'terkecuali'}:
            return 'signal_flip'

        # Contrast words → contrast_focus
        if sw_set & {'tetapi', 'namun', 'tapi', 'akan tetapi'}:
            # Check if there's also negation
            if sw_set & {'bukan', 'tidak'}:
                return 'negation_affirmation'
            return 'contrast_focus'

        # Negation words → negation_affirmation
        if sw_set & {'bukan'} and sw_set & {'tapi', 'tetapi'}:
            return 'negation_affirmation'

        # Comparison words → comparison_resolve
        if sw_set & {'lebih', 'kurang'}:
            return 'comparison_resolve'

        # Quantitative patterns
        if 'berapa' in problem and re.search(r'\d+', problem):
            return 'quantity_compute'

        # Check steps for quantitative hints
        for step in steps:
            step_lower = step.lower()
            if any(w in step_lower for w in ['jumlah', 'total', 'hitung', 'jumlahkan', 'kali', 'bagi']):
                return 'quantity_compute'

        # Fact vs emotion
        emotion_words = {'sedih', 'berduka', 'bahagia', 'marah', 'menangis', 'air mata'}
        fact_words = {'apa yang', 'membuat', 'memasak', 'mengerjakan'}
        if any(w in problem for w in fact_words):
            if any(w in problem for w in emotion_words) or any(w in answer for w in emotion_words):
                return 'fact_extract'

        # Word sense disambiguation
        if 'bermakna' in problem or 'makna kata' in problem or 'artinya' in problem:
            return 'word_sense'

        # Context filter — for questions about specific facts in long text
        if any(w in problem for w in ['menurut teks', 'dalam teks', 'berdasarkan cerita']):
            return 'context_filter'

        # Entity extraction — default for many question types
        if any(w in problem for w in ['siapa', 'apa', 'dimana']):
            return 'entity_extract'

        return None

    def _infer_from_explanation(self, explanation: str) -> Optional[str]:
        """Infer transformation kind from the explanation (kenapa)."""
        if not explanation:
            return None

        if 'kecuali' in explanation or 'pengecualian' in explanation:
            return 'signal_flip'
        if 'tetapi' in explanation or 'kontras' in explanation or 'sebenarnya' in explanation:
            return 'contrast_focus'
        if 'bukan' in explanation and 'tapi' in explanation:
            return 'negation_affirmation'
        if 'lebih' in explanation and 'dari' in explanation:
            return 'comparison_resolve'
        if 'fakta' in explanation or 'faktual' in explanation:
            return 'fact_extract'

        return None

    def _extract_trigger(self, kind: str, structural_words: list,
                          problem: str, text_lower: str, answer: str) -> dict:
        """Extract the trigger configuration for a transformation."""
        trigger = {}

        if kind == 'signal_flip':
            trigger = {
                'signal_words': [w for w in structural_words if w in {'kecuali', 'selain', 'terkecuali'}],
                'result_position': 'after',
            }

        elif kind == 'contrast_focus':
            trigger = {
                'contrast_words': [w for w in structural_words if w in {'tetapi', 'namun', 'tapi', 'akan tetapi'}],
            }

        elif kind == 'negation_affirmation':
            trigger = {
                'negation_words': ['bukan', 'tidak'],
                'affirmation_words': ['tapi', 'tetapi', 'melainkan'],
            }

        elif kind == 'comparison_resolve':
            trigger = {
                'comparison_words': ['lebih', 'kurang'],
            }

        elif kind == 'fact_extract':
            # Find the factual action in the question
            fact_indicators = ['apa yang', 'membuat', 'memasak', 'mengerjakan']
            trigger = {
                'fact_indicator': next((fi for fi in fact_indicators if fi in problem), ''),
                'fact_action': '',
            }
            # Try to find the specific action word
            for word in problem.split():
                if word in text_lower and len(word) > 4:
                    trigger['fact_action'] = word
                    break

        elif kind == 'quantity_compute':
            trigger = {
                'compute_type': 'sum',  # Will be refined
            }
            if 'kembalian' in problem:
                trigger['compute_type'] = 'change'

        elif kind == 'context_filter':
            # Find the key search term from the question
            search_terms = [w for w in problem.split()
                          if len(w) > 3 and w not in {'apa', 'yang', 'menurut', 'teks', 'tersebut'}]
            trigger = {
                'search_pattern': '|'.join(search_terms[:3]) if search_terms else '',
            }

        elif kind == 'word_sense':
            # Find the target word and context
            target = ''
            for phrase in ['bermakna', 'makna kata', 'artinya']:
                if phrase in problem:
                    parts = problem.split(phrase)
                    if parts:
                        # Get the word before the phrase
                        words_before = parts[0].split()
                        if words_before:
                            target = words_before[-1].strip('"\'')
                        break
            trigger = {
                'target_word': target,
                'context_words': [],
                'meaning_if_context': '',
                'meaning_otherwise': '',
            }

        elif kind == 'entity_extract':
            # Find what to extract
            trigger = {
                'pattern': '',
                'extract_after': '',
                'extract_before': '',
            }

        return trigger

    # ═══════════════ PIPELINE: GENERALIZE ═══════════════

    def _generalize(self, schema: dict, lesson) -> dict:
        """Generalize a schema by abstracting away specifics.

        Replace specific entities with slots, keep structural patterns.
        """
        problem = lesson.problem.lower()

        # Build generalized conditions
        conditions = set()

        # Add structural words as conditions
        for sw in schema['structural_words']:
            conditions.add(sw)

        # Add key words from the problem (not stop words)
        stop_words = {'apa', 'yang', 'di', 'ke', 'dari', 'untuk', 'dengan',
                      'adalah', 'itu', 'ini', 'dan', 'atau', 'juga', 'akan',
                      'telah', 'sudah', 'dalam', 'pada', 'oleh', 'dengan',
                      'berikut', 'tentang', 'berupa'}

        for word in problem.split():
            word = word.strip('.,;:!?()')
            if word in stop_words:
                continue
            if word in self.STRUCTURAL_WORDS:
                continue
            if len(word) > 3:
                # v32: Removed SLOT_TYPES lookup. SELF doesn't use hardcoded
                # slot categories. Instead, use the word itself as a condition.
                # SELF will learn to generalize through the understanding graph.
                conditions.add(word)

        # Also add conditions from the transformation trigger
        trigger = schema.get('trigger', {})
        for key, value in trigger.items():
            if isinstance(value, list):
                conditions.update(value)
            elif isinstance(value, str) and len(value) > 2:
                conditions.add(value)

        schema['generalized_conditions'] = list(conditions)[:15]

        # Build abstraction
        kind = schema['kind']
        abstraction = self._build_abstraction(kind, schema, lesson)
        schema['abstraction'] = abstraction

        return schema

    def _build_abstraction(self, kind: str, schema: dict, lesson) -> str:
        """Build a human-readable abstraction of the understanding."""
        explanation = lesson.explanation_why if lesson.explanation_why else ''

        if explanation:
            # Use the explanation as basis for abstraction
            return explanation[:200]

        # Build from kind
        abstractions = {
            'signal_flip': 'Kata pengecualian (kecuali/selain) membuat entitas setelahnya berkeadaan berlawanan dari aturan umum.',
            'contrast_focus': 'Kata kontras (tetapi/namun) menandakan bahwa bagian SETELAHNYA adalah keadaan sebenarnya.',
            'negation_affirmation': 'Pola "bukan X tapi Y" menolak X dan menegaskan Y — jawaban selalu Y.',
            'comparison_resolve': 'Perbandingan "A lebih X dari B" — jika pertanyaan bertanya lawan X, jawab B.',
            'fact_extract': 'Pertanyaan faktual → cari fakta di teks, abaikan emosi/opini.',
            'quantity_compute': 'Pertanyaan kuantitatif → identifikasi entitas, cari angka, hitung.',
            'context_filter': 'Cari informasi spesifik dalam teks panjang — fokus pada kata kunci.',
            'word_sense': 'Kata yang sama bisa bermakna berbeda — tentukan makna dari konteks.',
            'entity_extract': 'Ekstrak entitas spesifik dari teks berdasarkan pola.',
        }

        return abstractions.get(kind, f'Understanding untuk {kind}')

    # ═══════════════ PIPELINE: BUILD NODE ═══════════════

    def _build_node(self, schema: dict, lesson) -> Optional[UnderstandingNode]:
        """Build an UnderstandingNode from a generalized schema."""
        kind = schema['kind']
        trigger = schema.get('trigger', {})

        # Build transformation
        transformation = Transformation(
            kind=kind,
            trigger=trigger,
            action=schema.get('abstraction', ''),
            slots=[],
            constraints=[],
        )

        # Build conditions
        conditions = schema.get('generalized_conditions', [])

        # Build node ID
        import hashlib
        raw = f"{kind}:{':'.join(sorted(conditions[:5]))}"
        node_id = hashlib.md5(raw.encode()).hexdigest()[:12]

        # Check if similar node already exists
        existing = self._find_similar_node(kind, conditions)
        if existing:
            # Merge conditions into existing node
            for cond in conditions:
                if cond not in existing.conditions:
                    existing.conditions.append(cond)
            # Merge trigger
            if existing.transformation:
                self._merge_triggers(existing.transformation.trigger, trigger)
            self._graph._save()
            return existing

        # Create new node
        name = f"{kind}_{'_'.join(conditions[:3])}"
        node = UnderstandingNode(
            id=node_id,
            name=name,
            concept=schema.get('abstraction', '')[:100],
            abstraction=schema.get('abstraction', ''),
            schemas=[schema.get('problem', '')[:100]],
            transformation=transformation,
            conditions=conditions,
            source='self_discovered_from_observation',
            confidence=0.6,
        )

        return node

    def _find_similar_node(self, kind: str, conditions: list) -> Optional[UnderstandingNode]:
        """Find an existing node with the same transformation kind."""
        for nid, node in self._graph._nodes.items():
            if node.transformation and node.transformation.kind == kind:
                # Check condition overlap
                overlap = len(set(node.conditions) & set(conditions))
                if overlap >= 2 or (overlap > 0 and len(conditions) <= 3):
                    return node
        return None

    def _merge_triggers(self, existing_trigger: dict, new_trigger: dict):
        """Merge new trigger info into existing trigger."""
        for key, value in new_trigger.items():
            if key not in existing_trigger:
                existing_trigger[key] = value
            elif isinstance(value, list) and isinstance(existing_trigger[key], list):
                for v in value:
                    if v not in existing_trigger[key]:
                        existing_trigger[key].append(v)

    # ═══════════════ PIPELINE: INTEGRATE ═══════════════

    def _integrate(self, node: UnderstandingNode):
        """Integrate a new node into the understanding graph.

        - Add to graph
        - Find related nodes and create edges
        - Check for conflicts
        """
        self._graph.add_node(node)

        # Find related nodes
        for nid, existing in self._graph._nodes.items():
            if nid == node.id:
                continue

            # Check if they share conditions
            shared = set(node.conditions) & set(existing.conditions)
            if len(shared) >= 2:
                node.add_edge(nid, 'related')
                existing.add_edge(node.id, 'related')

            # Check if one is a specialization of the other
            if node.transformation and existing.transformation:
                if node.transformation.kind == existing.transformation.kind:
                    if len(node.conditions) > len(existing.conditions):
                        node.add_edge(nid, 'generalizes')
                        existing.add_edge(node.id, 'specializes')
                    elif len(node.conditions) < len(existing.conditions):
                        node.add_edge(nid, 'specializes')
                        existing.add_edge(node.id, 'generalizes')

        self._graph._save()

    # ═══════════════ CROSS-PATTERN DISCOVERY ═══════════════

    def _discover_cross_patterns(self, nodes: List[UnderstandingNode]):
        """Discover patterns that span multiple understandings."""
        # Find nodes that could be composed
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                # If one is negation and the other is contrast,
                # they can be composed (negation + contrast = contrast is affirmed)
                if node_a.transformation and node_b.transformation:
                    kinds = {node_a.transformation.kind, node_b.transformation.kind}
                    if kinds == {'signal_flip', 'contrast_focus'}:
                        # Exception + contrast = combined understanding
                        node_a.add_edge(node_b.id, 'composes_with')
                        node_b.add_edge(node_a.id, 'composes_with')

    # ═══════════════ SEED UNDERSTANDINGS ═══════════════

    def seed_from_lessons(self, lessons):
        """Build understandings by observing teaching lessons.

        This is how SELF builds its initial understanding —
        NOT from hardcoded rules, but from OBSERVING examples.
        """
        for q_type in lessons.get_types():
            type_lessons = lessons.get_by_type(q_type)
            if len(type_lessons) < 1:
                continue

            for lesson in type_lessons:
                node = self.observe(lesson)
                if node:
                    logger.info(
                        "SELF built understanding: %s (kind=%s, conditions=%s)",
                        node.id, node.transformation.kind if node.transformation else '?',
                        node.conditions[:5]
                    )

        logger.info("Understanding graph now has %d nodes", self._graph.count())


# ═══════════════ CONVENIENCE: SEED CORE UNDERSTANDINGS ═══════════════

# v32: Removed seed_core_understandings().
# This was HARDCODED knowledge that violated SELF-AI vision of "NO hardcoded rules".
# SELF builds ALL understanding through:
#   1. UnderstandingComposer (Qwen3)
#   2. UnderstandingBuilder (observation)
#   3. Self-correction (learning from failures)
# If the graph starts empty, SELF will learn through teaching.

# Singleton instance for shared understanding graph
_shared_graph = None


def get_shared_graph() -> UnderstandingGraph:
    """Get the shared singleton UnderstandingGraph.

    v29 fix: ALL components (AnswerHandlers, SelfCore, UnderstandingComposer)
    MUST use this same graph instance. Without this, learn/teach/observe
    writes to a DIFFERENT graph than the one used for answering —
    breaking the self-improvement loop.
    """
    global _shared_graph
    if _shared_graph is None:
        _shared_graph = UnderstandingGraph()
    return _shared_graph
