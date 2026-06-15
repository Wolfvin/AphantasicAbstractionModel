"""AGNN Knowledge Graph — typed nodes and typed edges.

This module defines the core data structures for AGNN's knowledge graph.
Every piece of knowledge is stored as a typed triple: (subject, predicate, object),
where each element carries a semantic type tag (e.g., ENTITY, RELATION, CONCEPT).

The graph supports:
    - Typed nodes with semantic categories (entity, concept, quantity, etc.)
    - Typed edges with predicate labels (is_a, causes, negates, etc.)
    - Confidence-weighted edges for probabilistic knowledge
    - Efficient adjacency lookup for traversal and message passing
    - Message passing (GNN neighborhood aggregation)
    - Spread activation (confidence propagation through edges)
    - Multi-hop traversal with reasoning chain output

Vision:
    Unlike the previous bge-m3 embedding approach where knowledge was stored
    as dense vectors and retrieved by cosine similarity, AGNN's graph stores
    knowledge as structured, interpretable triples. This enables:
      1. Compositional reasoning via multi-hop traversal (not just similarity)
      2. Explainable reasoning chains (follow the edges)
      3. Efficient message passing over the graph topology
      4. Type-aware aggregation (different predicates → different updates)

Design Decisions:
    - TypedEdge is richer than UnderstandingNode.edges tuples: it carries
      source_id, target_id, relation_type, confidence, role, and context.
      This mirrors AAM v12's SemanticEdge (relation, role, source) but
      adapted for Python/numpy without Rust dependencies.
    - AGNNGraph wraps UnderstandingGraph rather than extending it, so
      existing callers (add_node, retrieve, reinforce, penalize) continue
      to work unchanged. AGNNGraph adds the GNN layer on top.
    - Embeddings are numpy arrays (float32). No external model required.
      The graph starts with random embeddings and refines them via
      message passing — the topology IS the signal.
    - RelationType enum is inspired by AAM v12's RelationType but kept
      lightweight: Categorical, Causal, Differential, Functional, Temporal,
      Spatial, Discursive. This gives message passing type-aware routing
      without over-engineering.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from agnn.embeddings import ModelEmbedder, EmbeddingCache, embed_node, embed_nodes_batch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────

DEFAULT_EMBEDDING_DIM = 64
"""Default embedding dimensionality for AGNN nodes.

64-dim is chosen deliberately:
  - Small enough for fast numpy operations (dot products, norms)
  - Large enough to discriminate between ~1000 node types
  - Matches the 'aphantasic' philosophy: structure > dimensionality
  - Can be overridden via AGNNGraph(embedding_dim=N)
"""


# ──────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────

class NodeType(Enum):
    """Semantic category of a graph node.

    Inspired by AAM v12's AtomType but simplified for AGNN.
    Each type determines how message passing aggregates information
    from this node's neighbors.
    """
    ENTITY = "entity"          # A specific thing: "harimau", "Jakarta"
    CONCEPT = "concept"        # An abstract idea: "pengecualian", "penjumlahan"
    QUANTITY = "quantity"      # A numeric value: 35, 0.5
    RELATION = "relation"      # A predicate edge reified as a node
    RULE = "rule"              # A transformation/understanding (bridges to UnderstandingNode)
    CONTEXT = "context"        # A situational frame: "soal matematika kelas 4"


class RelationType(Enum):
    """Nature of the relationship between two nodes.

    Directly inspired by AAM v12's RelationType. Each type determines
    how message passing routes information along this edge:
      - Categorical: propagate identity (is_a, member_of)
      - Causal: propagate consequence (causes, prevents)
      - Differential: invert or contrast (negates, contrasts_with)
      - Functional: propagate enablement (requires, enables)
      - Temporal: propagate sequence (before, after)
      - Spatial: propagate containment (contains, near)
      - Discursive: propagate reference (topic, about)
    """
    CATEGORICAL = "categorical"    # is_a, member_of, instance_of
    CAUSAL = "causal"              # causes, prevents, results_in
    DIFFERENTIAL = "differential"  # negates, contrasts_with, excepts
    FUNCTIONAL = "functional"      # requires, enables, computes
    TEMPORAL = "temporal"          # before, after, during
    SPATIAL = "spatial"            # contains, near, inside
    DISCURSIVE = "discursive"      # topic, about, refers_to


class EdgeRole(Enum):
    """Role an edge plays within a structured composition.

    Borrowed from AAM v12's SemanticRole. Only populated for edges
    that participate in a composition (e.g., cause→effect within
    a causal chain). Free edges have role=NONE.
    """
    NONE = "none"
    AGENT = "agent"           # Who/what performs the action
    PATIENT = "patient"       # Who/what is affected
    CAUSE = "cause"           # The trigger in a causal relation
    EFFECT = "effect"         # The outcome in a causal relation
    CONDITION = "condition"   # A precondition
    EXCEPTION = "exception"   # An exclusion/exception


# ──────────────────────────────────────────────────────
#  Aggregation weights per RelationType
# ──────────────────────────────────────────────────────

_RELATION_AGGREGATION_WEIGHTS: Dict[RelationType, float] = {
    # Categorical edges propagate strongly — "X is_a Y" means X inherits Y's meaning
    RelationType.CATEGORICAL: 1.0,
    # Causal edges propagate moderately — "X causes Y" transfers some but not all info
    RelationType.CAUSAL: 0.7,
    # Differential edges INVERT — "X negates Y" should flip the signal
    # We use negative weight so the aggregation naturally inverts
    RelationType.DIFFERENTIAL: -0.8,
    # Functional edges propagate moderately — "X requires Y" links them
    RelationType.FUNCTIONAL: 0.6,
    # Temporal edges weakly propagate — sequence isn't semantic identity
    RelationType.TEMPORAL: 0.3,
    # Spatial edges moderately propagate — containment implies some shared context
    RelationType.SPATIAL: 0.5,
    # Discursive edges weakly propagate — reference isn't semantic overlap
    RelationType.DISCURSIVE: 0.2,
}

# Spread activation decay per RelationType
_SPREAD_DECAY: Dict[RelationType, float] = {
    RelationType.CATEGORICAL: 0.9,
    RelationType.CAUSAL: 0.7,
    RelationType.DIFFERENTIAL: 0.5,   # Negation spreads weakly (it blocks more than it carries)
    RelationType.FUNCTIONAL: 0.6,
    RelationType.TEMPORAL: 0.4,
    RelationType.SPATIAL: 0.5,
    RelationType.DISCURSIVE: 0.3,
}


# ──────────────────────────────────────────────────────
#  TypedEdge
# ──────────────────────────────────────────────────────

@dataclass
class TypedEdge:
    """A typed, directed edge in the AGNN knowledge graph.

    This is richer than UnderstandingNode.edges (which are just
    (target_id, edge_type) tuples). TypedEdge carries:

      - source_id: Where the edge originates (direction matters for GNN)
      - target_id: Where the edge points
      - relation_type: WHAT kind of semantic relation (Categorical, Causal, etc.)
      - confidence: How strongly this relation holds (0.0–1.0)
      - role: Optional compositional role (Agent, Patient, Cause, Effect, etc.)
      - context: Optional provenance — where this edge came from

    This mirrors AAM v12's SemanticEdge (relation + role + source) but
    adds confidence for probabilistic reasoning.
    """
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = 1.0
    role: EdgeRole = EdgeRole.NONE
    context: str = ""

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Edge confidence must be 0.0–1.0, got {self.confidence}")

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "role": self.role.value,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TypedEdge":
        return cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            relation_type=RelationType(d.get("relation_type", "categorical")),
            confidence=d.get("confidence", 1.0),
            role=EdgeRole(d.get("role", "none")),
            context=d.get("context", ""),
        )


# ──────────────────────────────────────────────────────
#  AGNNNode
# ──────────────────────────────────────────────────────

@dataclass
class AGNNNode:
    """A node in the AGNN knowledge graph.

    Each node represents a piece of knowledge with:
      - A semantic type (ENTITY, CONCEPT, QUANTITY, etc.)
      - A text label (human-readable)
      - An embedding vector (numpy float32, refined by message passing)
      - A confidence score (how well-established this knowledge is)
      - Optional metadata (links to UnderstandingNode, source info)

    The embedding starts as a random vector and gets refined through
    message passing — the topology of the graph determines the
    embedding, not an external model. This is the core insight:
    structure IS the signal.
    """
    id: str
    label: str
    node_type: NodeType
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_EMBEDDING_DIM, dtype=np.float32))
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure embedding is float32 numpy array
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)
        else:
            self.embedding = self.embedding.astype(np.float32)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Node confidence must be 0.0–1.0, got {self.confidence}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type.value,
            "embedding": self.embedding.tolist(),
            "confidence": float(self.confidence),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AGNNNode":
        return cls(
            id=d["id"],
            label=d["label"],
            node_type=NodeType(d.get("node_type", "entity")),
            embedding=np.array(d.get("embedding", []), dtype=np.float32),
            confidence=d.get("confidence", 0.5),
            metadata=d.get("metadata", {}),
        )


# ──────────────────────────────────────────────────────
#  ReasoningChain
# ──────────────────────────────────────────────────────

@dataclass
class ReasoningChain:
    """A multi-hop reasoning path through the AGNN graph.

    Produced by AGNNGraph.traverse(), this represents a chain of
    (node → edge → node → edge → ... → node) steps that form
    a logical argument from a query to an answer.

    The chain can be verbalized into natural language for feeding
    to a language model as context.
    """
    steps: List[Tuple[str, str, str]]  # [(source_label, relation, target_label), ...]
    confidence: float = 0.0
    node_ids: List[str] = field(default_factory=list)

    def verbalize(self) -> str:
        """Convert the reasoning chain to a natural-language string.

        Produces sentences like:
          "harimau IS_A karnivora, karnivora CAUSES pemakan_daging"
        which can be fed directly to a language model as context.

        The verbalization uses arrow notation (→) for readability
        and includes relation types in UPPERCASE for emphasis.
        """
        if not self.steps:
            return ""
        parts = []
        for source, relation, target in self.steps:
            rel_display = relation.upper().replace("_", " ")
            parts.append(f"{source} → [{rel_display}] → {target}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "confidence": self.confidence,
            "node_ids": self.node_ids,
            "verbalized": self.verbalize(),
        }


# ──────────────────────────────────────────────────────
#  AGNNGraph
# ──────────────────────────────────────────────────────

class AGNNGraph:
    """AGNN Knowledge Graph — the core data structure for composable knowledge memory.

    AGNNGraph is a typed, directed graph where:
      - Nodes carry semantic types and numpy embeddings
      - Edges carry relation types, confidence, and optional roles
      - Message passing aggregates neighbor information along typed edges
      - Spread activation propagates confidence through the graph
      - Multi-hop traversal produces explainable reasoning chains

    Design: WRAP, don't extend UnderstandingGraph
      UnderstandingGraph has a complex API (add_node, retrieve, reinforce,
      penalize, find_matching, etc.) that we must not break. Instead of
      modifying it, AGNNGraph wraps it:
        - UnderstandingGraph handles persistence and the existing API
        - AGNNGraph adds GNN operations on top
        - UnderstandingNodes are bridged to AGNNNodes via _bridge_node()
        - Edges in UnderstandingGraph (flat tuples) are lifted to TypedEdges

    This means existing code continues to work unchanged, while new code
    can use AGNNGraph for GNN-powered reasoning.

    Usage:
        # Standalone (no UnderstandingGraph)
        graph = AGNNGraph()
        graph.add_node(AGNNNode(id="harimau", label="harimau", node_type=NodeType.ENTITY))
        graph.add_node(AGNNNode(id="karnivora", label="karnivora", node_type=NodeType.CONCEPT))
        graph.add_edge(TypedEdge("harimau", "karnivora", RelationType.CATEGORICAL))

        # Message passing
        graph.message_pass("harimau")

        # Spread activation
        graph.spread_activation(["harimau"], steps=2)

        # Traverse
        chain = graph.traverse("harimau", max_hops=3)
        print(chain.verbalize())
    """

    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 understanding_graph=None):
        """Initialize the AGNN graph.

        Args:
            embedding_dim: Dimensionality of node embeddings.
                Default 64 — small enough for fast numpy ops, large enough
                to discriminate between nodes.
            understanding_graph: Optional UnderstandingGraph instance to wrap.
                If provided, existing UnderstandingNodes are bridged into
                AGNNNodes, and existing edge tuples are lifted to TypedEdges.
        """
        self._nodes: Dict[str, AGNNNode] = {}
        self._edges: List[TypedEdge] = []
        self._outgoing: Dict[str, List[int]] = {}  # node_id → [edge_indices]
        self._incoming: Dict[str, List[int]] = {}   # node_id → [edge_indices]
        self._embedding_dim = embedding_dim
        self._underlying = understanding_graph

        # Embedder integration — None means fallback to random init
        self._embedder: Optional[ModelEmbedder] = None
        self._embedding_cache: Optional[EmbeddingCache] = None

        # If wrapping an UnderstandingGraph, bridge all existing nodes
        if understanding_graph is not None:
            self._bridge_from_understanding_graph(understanding_graph)

    # ──────────────── Node Operations ────────────────

    def add_node(self, node: AGNNNode) -> AGNNNode:
        """Add a node to the graph.

        If the node's embedding is all zeros, initialize it with a
        small random vector (scaled by 0.1 to keep initial values
        close to zero — the signal should come from message passing,
        not from random initialization).

        Args:
            node: The AGNNNode to add.

        Returns:
            The added node (with embedding initialized if needed).
        """
        # Initialize zero embeddings with small random values
        if np.all(node.embedding == 0) or len(node.embedding) != self._embedding_dim:
            node.embedding = np.random.randn(self._embedding_dim).astype(np.float32) * 0.1
        elif len(node.embedding) == self._embedding_dim:
            node.embedding = node.embedding.astype(np.float32)
        else:
            # Resize embedding to match graph dimensionality
            padded = np.zeros(self._embedding_dim, dtype=np.float32)
            copy_len = min(len(node.embedding), self._embedding_dim)
            padded[:copy_len] = node.embedding[:copy_len]
            node.embedding = padded

        self._nodes[node.id] = node
        self._outgoing.setdefault(node.id, [])
        self._incoming.setdefault(node.id, [])
        return node

    def get_node(self, node_id: str) -> Optional[AGNNNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connected edges.

        Returns True if the node existed and was removed, False otherwise.
        """
        if node_id not in self._nodes:
            return False

        # Remove all edges connected to this node
        self._edges = [e for e in self._edges
                       if e.source_id != node_id and e.target_id != node_id]
        del self._nodes[node_id]
        self._rebuild_adjacency()
        return True

    def all_node_ids(self) -> List[str]:
        """Return all node IDs in the graph."""
        return list(self._nodes.keys())

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    # ──────────────── Edge Operations ────────────────

    def add_edge(self, edge: TypedEdge) -> TypedEdge:
        """Add a directed edge to the graph.

        Both source and target nodes must already exist in the graph.
        Auto-creates the reverse lookup edge for undirected traversal
        (but the edge itself is directed).

        Args:
            edge: The TypedEdge to add.

        Returns:
            The added edge.

        Raises:
            ValueError: If source_id or target_id not in graph.
        """
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node '{edge.source_id}' not in graph")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node '{edge.target_id}' not in graph")

        idx = len(self._edges)
        self._edges.append(edge)
        self._outgoing.setdefault(edge.source_id, []).append(idx)
        self._incoming.setdefault(edge.target_id, []).append(idx)
        return edge

    def get_edges_from(self, node_id: str) -> List[TypedEdge]:
        """Get all outgoing edges from a node."""
        return [self._edges[i] for i in self._outgoing.get(node_id, [])
                if i < len(self._edges)]

    def get_edges_to(self, node_id: str) -> List[TypedEdge]:
        """Get all incoming edges to a node."""
        return [self._edges[i] for i in self._incoming.get(node_id, [])
                if i < len(self._edges)]

    def get_neighbors(self, node_id: str, relation_type: RelationType = None) -> List[AGNNNode]:
        """Get neighbor nodes connected by outgoing edges.

        Args:
            node_id: The node to find neighbors for.
            relation_type: If provided, only follow edges of this type.

        Returns:
            List of neighbor AGNNNodes.
        """
        neighbors = []
        for edge in self.get_edges_from(node_id):
            if relation_type is not None and edge.relation_type != relation_type:
                continue
            target = self._nodes.get(edge.target_id)
            if target is not None:
                neighbors.append(target)
        return neighbors

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return len(self._edges)

    # ──────────────── Message Passing (GNN Core) ────────────────

    def message_pass(self, node_id: str, damping: float = 0.5) -> Optional[np.ndarray]:
        """Aggregate embeddings from neighbors via GNN message passing.

        This is the core GNN operation: for a given node, collect messages
        from all its neighbors along typed edges, weight each message by
        the relation type's aggregation weight and edge confidence, then
        update the node's embedding as a weighted combination of its
        current embedding and the aggregated message.

        The update rule:
            message = Σ(agg_weight[relation] × edge.confidence × neighbor.embedding)
                     / Σ(edge.confidence)                          [normalized]
            new_embedding = (1 - damping) × old_embedding
                          + damping × message                      [residual]

        For DIFFERENTIAL edges (negation, contrast), the aggregation weight
        is NEGATIVE, so those neighbors SUBTRACT from the message rather
        than adding. This means "X negates Y" causes X's embedding to
        move AWAY from Y's embedding — which is semantically correct.

        Args:
            node_id: The node to update.
            damping: How much to weight the new message vs. the old embedding.
                0.0 = keep old embedding (no update)
                1.0 = replace with message (no residual)
                0.5 = equal blend (default)

        Returns:
            The new embedding, or None if the node has no neighbors or doesn't exist.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return None

        incoming = self.get_edges_to(node_id)
        if not incoming:
            # No neighbors → no message → no update
            return node.embedding.copy()

        # Aggregate messages from all incoming edges
        message = np.zeros(self._embedding_dim, dtype=np.float32)
        total_weight = 0.0

        for edge in incoming:
            source = self._nodes.get(edge.source_id)
            if source is None:
                continue

            # Weight = relation aggregation weight × edge confidence
            rel_weight = _RELATION_AGGREGATION_WEIGHTS.get(edge.relation_type, 0.5)
            w = rel_weight * edge.confidence
            message += w * source.embedding
            total_weight += abs(w)

        # Normalize by total weight to prevent explosion
        if total_weight > 1e-8:
            message /= total_weight

        # Residual update: blend old embedding with message
        old_embedding = node.embedding.copy()
        node.embedding = (1.0 - damping) * old_embedding + damping * message

        # Re-normalize to unit length (keeps embeddings in a consistent space)
        norm = np.linalg.norm(node.embedding)
        if norm > 1e-8:
            node.embedding /= norm

        return node.embedding.copy()

    def message_pass_all(self, damping: float = 0.5, iterations: int = 1):
        """Run message passing on ALL nodes in the graph.

        Multiple iterations allow information to propagate across k-hop
        neighborhoods. Each iteration, every node aggregates from its
        (now-updated) neighbors, so after k iterations, every node's
        embedding reflects information from k hops away.

        Args:
            damping: Blend factor for message vs. old embedding (see message_pass).
            iterations: Number of full-graph message passing iterations.
        """
        for _ in range(iterations):
            # Process all nodes in a fixed order (deterministic)
            for node_id in list(self._nodes.keys()):
                self.message_pass(node_id, damping=damping)

    # ──────────────── Spread Activation ────────────────

    def spread_activation(self, seed_ids: List[str], steps: int = 2,
                          decay: float = None) -> Dict[str, float]:
        """Propagate activation (confidence) through the graph from seed nodes.

        Spread activation models how activation flows from seed nodes
        through the graph topology. Unlike message passing (which updates
        embeddings), spread activation updates a separate "activation"
        score that represents how "relevant" or "active" each node is
        given the seed set.

        This is useful for:
          - Identifying which nodes are contextually relevant to a query
          - Boosting confidence of nodes connected to confirmed-correct answers
          - Finding indirectly related knowledge for composition

        The algorithm:
          1. Seed nodes start with activation = their confidence
          2. Each step, activation flows along edges:
             target.activation += source.activation × spread_decay[relation] × edge.confidence
          3. Activation is clamped to [0, 1]
          4. After `steps` iterations, return the activation map

        Args:
            seed_ids: Node IDs to start activation from.
            steps: Number of propagation steps (default 2).
            decay: Global decay multiplier. If None, uses per-relation decay
                from _SPREAD_DECAY.

        Returns:
            Dict mapping node_id → final activation score (0.0–1.0).
        """
        # Initialize activation: seeds get their confidence, others get 0
        activation: Dict[str, float] = {}
        seed_set: Set[str] = set()
        for nid, node in self._nodes.items():
            activation[nid] = 0.0

        for seed_id in seed_ids:
            seed = self._nodes.get(seed_id)
            if seed is not None:
                activation[seed_id] = seed.confidence
                seed_set.add(seed_id)

        # Propagate for `steps` iterations
        for _ in range(steps):
            new_activation = dict(activation)  # copy current state

            for edge in self._edges:
                source_act = activation.get(edge.source_id, 0.0)
                if source_act < 1e-8:
                    continue  # No activation to spread

                # Determine decay: per-relation or global
                if decay is not None:
                    edge_decay = decay
                else:
                    edge_decay = _SPREAD_DECAY.get(edge.relation_type, 0.5)

                # For DIFFERENTIAL edges, activation weakens but doesn't go negative
                # (negative activation doesn't make semantic sense for relevance)
                spread_amount = source_act * edge_decay * edge.confidence

                # Accumulate into target
                current = new_activation.get(edge.target_id, 0.0)
                new_activation[edge.target_id] = min(1.0, current + spread_amount)

            activation = new_activation

            # Preserve seed activation: seeds should never lose their initial
            # activation level (they are the source of truth)
            for seed_id in seed_set:
                seed = self._nodes.get(seed_id)
                if seed is not None:
                    activation[seed_id] = max(activation[seed_id], seed.confidence)

        return activation

    # ──────────────── Traversal ────────────────

    def traverse(self, query: str, max_hops: int = 3,
                 relation_filter: List[RelationType] = None,
                 confidence_threshold: float = 0.1) -> Optional[ReasoningChain]:
        """Traverse the graph from a query node, producing a reasoning chain.

        Uses a modified breadth-first search that:
          1. Finds the best-matching seed node by label similarity
          2. Explores neighbors, preferring high-confidence edges
          3. Stops when a leaf node (no outgoing edges) is found or max_hops reached
          4. Returns the highest-confidence path as a ReasoningChain

        The traversal is confidence-weighted: at each step, edges with
        higher confidence × relation_weight are preferred. This means
        "harimau IS_A karnivora" (weight 1.0) is preferred over
        "harimau NEGATES herbivora" (weight -0.8) as a primary path.

        Args:
            query: Text to match against node labels (case-insensitive substring).
            max_hops: Maximum number of edges to traverse.
            relation_filter: If provided, only follow edges of these relation types.
            confidence_threshold: Minimum edge confidence to follow.

        Returns:
            A ReasoningChain, or None if no matching seed node is found.
        """
        # Find seed node(s) by label matching
        seed = self._find_seed(query)
        if seed is None:
            return None

        # BFS with priority (confidence-weighted)
        # Each state: (current_node_id, path_so_far, path_confidence)
        best_chain: Optional[ReasoningChain] = None
        best_score = -1.0

        # Priority queue: (negative_score, node_id, steps_taken, chain_steps, node_ids)
        import heapq
        # We use negative score because heapq is a min-heap
        initial_score = seed.confidence
        queue = [(-initial_score, seed.id, 0, [], [seed.id])]
        visited_paths: Set[str] = set()

        while queue:
            neg_score, current_id, hops, chain_steps, node_ids = heapq.heappop(queue)
            current_score = -neg_score

            if hops >= max_hops:
                # Check if this is the best complete chain
                if current_score > best_score and chain_steps:
                    best_score = current_score
                    best_chain = ReasoningChain(
                        steps=chain_steps,
                        confidence=current_score,
                        node_ids=node_ids,
                    )
                continue

            # Explore neighbors
            edges = self.get_edges_from(current_id)
            if not edges and hops > 0:
                # Leaf node — this is a complete chain
                if current_score > best_score and chain_steps:
                    best_score = current_score
                    best_chain = ReasoningChain(
                        steps=chain_steps,
                        confidence=current_score,
                        node_ids=node_ids,
                    )
                continue

            for edge in edges:
                if edge.confidence < confidence_threshold:
                    continue
                if relation_filter and edge.relation_type not in relation_filter:
                    continue

                target = self._nodes.get(edge.target_id)
                if target is None:
                    continue

                # Avoid cycles
                path_key = f"{current_id}->{edge.target_id}"
                if path_key in visited_paths:
                    continue
                visited_paths.add(path_key)

                # Score: current × edge confidence × relation weight
                rel_weight = abs(_RELATION_AGGREGATION_WEIGHTS.get(edge.relation_type, 0.5))
                step_score = current_score * edge.confidence * rel_weight
                # Also factor in target node's own confidence
                combined_score = step_score * target.confidence

                source_label = self._nodes[current_id].label
                new_steps = chain_steps + [(source_label, edge.relation_type.value, target.label)]
                new_ids = node_ids + [edge.target_id]

                heapq.heappush(queue, (
                    -combined_score,
                    edge.target_id,
                    hops + 1,
                    new_steps,
                    new_ids,
                ))

        # If no complete chain found at max_hops, return the best partial chain
        if best_chain is None and queue:
            # Take the deepest explored path
            _, _, _, chain_steps, node_ids = queue[0]
            if chain_steps:
                best_chain = ReasoningChain(
                    steps=chain_steps,
                    confidence=0.0,
                    node_ids=node_ids,
                )

        return best_chain

    def _find_seed(self, query: str) -> Optional[AGNNNode]:
        """Find the best-matching node for a query string.

        Uses simple substring matching on labels. This is intentionally
        simple — no embedding model needed. The real semantic matching
        happens through the graph structure itself (message passing +
        traversal), not through string similarity.

        If multiple nodes match, returns the one with highest confidence.
        """
        query_lower = query.lower().strip()
        best: Optional[AGNNNode] = None
        best_score = -1.0

        for node in self._nodes.values():
            label_lower = node.label.lower()
            # Exact match
            if label_lower == query_lower:
                return node
            # Substring match
            if query_lower in label_lower or label_lower in query_lower:
                if node.confidence > best_score:
                    best = node
                    best_score = node.confidence

        return best

    # ──────────────── Embedder Integration ────────────────

    def set_embedder(self, embedder: ModelEmbedder,
                     cache: Optional[EmbeddingCache] = None) -> None:
        """Attach a ModelEmbedder to the graph for semantic embedding initialization.

        Once an embedder is set, new nodes can be initialized with
        model-native embeddings instead of random vectors. This replaces
        the random initialization in add_node() when an embedder is available.

        The cache is optional but recommended — it avoids re-computing
        embeddings for the same text across multiple calls.

        If the embedder's hidden_size does not match the graph's
        embedding_dim, the embedding will be projected (truncated or
        zero-padded) to fit.

        Args:
            embedder: A ModelEmbedder instance (real or mock).
            cache: Optional EmbeddingCache for storing computed embeddings.
        """
        self._embedder = embedder
        self._embedding_cache = cache
        logger.info("Embedder set: model_id=%s, hidden_size=%d, cache=%s",
                    embedder.model_id, embedder.hidden_size,
                    "enabled" if cache is not None else "disabled")

    def initialize_embeddings(self, texts: Optional[Dict[str, str]] = None) -> int:
        """Batch-compute embeddings for nodes that have text labels.

        For each node in the graph, compute its embedding using the
        attached ModelEmbedder (if set). If an embedder is NOT set,
        this method is a no-op — existing random embeddings are preserved.

        The texts parameter allows overriding the text used for each node.
        If not provided, each node's label is used as the text.

        Args:
            texts: Optional dict mapping node_id → text to embed.
                If None, uses each node's label as the embedding text.
                Useful when node labels are IDs and the actual text
                is stored elsewhere.

        Returns:
            Number of nodes whose embeddings were initialized.

        Raises:
            RuntimeError: If no embedder has been set via set_embedder().
        """
        if self._embedder is None:
            logger.warning("initialize_embeddings() called without embedder — skipping")
            return 0

        # Determine which nodes to embed
        node_ids = list(self._nodes.keys())
        if not node_ids:
            return 0

        # Build text list
        text_list: List[str] = []
        for nid in node_ids:
            if texts and nid in texts:
                text_list.append(texts[nid])
            else:
                text_list.append(self._nodes[nid].label)

        # Batch-compute embeddings
        embeddings = embed_nodes_batch(text_list, self._embedder, self._embedding_cache)

        # Assign embeddings to nodes
        count = 0
        for i, nid in enumerate(node_ids):
            embedding = embeddings[i]
            # Project to graph's embedding_dim if needed
            if len(embedding) != self._embedding_dim:
                embedding = self._project_embedding(embedding)
            self._nodes[nid].embedding = embedding.astype(np.float32)
            count += 1

        logger.info("Initialized embeddings for %d nodes (model_id=%s)",
                    count, self._embedder.model_id)
        return count

    def _project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Project an embedding to the graph's embedding dimension.

        If the embedding is larger than embedding_dim, truncate it.
        If smaller, zero-pad it. This handles the mismatch between
        the model's hidden_size and the graph's embedding_dim.

        Truncation is a valid strategy because:
          - For very large hidden_size (e.g., 896 for Qwen3-0.6B),
            the graph typically uses a smaller embedding_dim (64 or 128)
          - The first N dimensions of a mean-pooled hidden state
            capture the most important information (PCA-like)
          - Zero-padding for smaller embeddings preserves the
            original values in their existing dimensions

        Args:
            embedding: The source embedding vector.

        Returns:
            Projected embedding of shape (self._embedding_dim,).
        """
        result = np.zeros(self._embedding_dim, dtype=np.float32)
        copy_len = min(len(embedding), self._embedding_dim)
        result[:copy_len] = embedding[:copy_len]

        # Re-normalize after truncation/padding
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result /= norm

        return result

    # ──────────────── Bridge from UnderstandingGraph ────────────────

    def _bridge_from_understanding_graph(self, ug):
        """Bridge all nodes and edges from an UnderstandingGraph into AGNN.

        This converts UnderstandingNodes into AGNNNodes and lifts the flat
        (target_id, edge_type) tuples into proper TypedEdges.

        Mapping:
          - UnderstandingNode.id → AGNNNode.id
          - UnderstandingNode.concept → AGNNNode.label
          - UnderstandingNode.source → mapped to NodeType:
              composed_from_teaching → RULE
              composed_from_observation → CONCEPT
              composed_from_failure → RULE
              self_discovered → CONCEPT
              default → CONCEPT
          - UnderstandingNode.condition_embedding → AGNNNode.embedding
              (if available and dimensionality matches)
          - UnderstandingNode.edges → TypedEdge with inferred RelationType
          - UnderstandingNode.confidence → AGNNNode.confidence

        The underlying UnderstandingGraph is NOT modified.
        """
        all_nodes = ug.get_all() if hasattr(ug, 'get_all') else {}
        for nid, ndata in all_nodes.items():
            # Map source to NodeType
            source = ndata.get('source', 'self_discovered')
            node_type = self._map_source_to_type(source)

            # Get embedding if available
            emb_list = ndata.get('condition_embedding')
            if emb_list and len(emb_list) == self._embedding_dim:
                embedding = np.array(emb_list, dtype=np.float32)
            else:
                embedding = np.zeros(self._embedding_dim, dtype=np.float32)

            node = AGNNNode(
                id=nid,
                label=ndata.get('concept', nid),
                node_type=node_type,
                embedding=embedding,
                confidence=ndata.get('confidence', 0.5),
                metadata={"source": "understanding_graph", "original_source": source},
            )
            self._nodes[nid] = node
            self._outgoing.setdefault(nid, [])
            self._incoming.setdefault(nid, [])

            # Bridge edges
            for target_id, edge_type in ndata.get('edges', []):
                if target_id in self._nodes or True:  # Add even if target not yet seen
                    relation = self._infer_relation_type(edge_type)
                    edge = TypedEdge(
                        source_id=nid,
                        target_id=target_id,
                        relation_type=relation,
                        confidence=ndata.get('confidence', 0.5),
                        context=edge_type,
                    )
                    idx = len(self._edges)
                    self._edges.append(edge)
                    self._outgoing.setdefault(nid, []).append(idx)
                    self._incoming.setdefault(target_id, []).append(idx)

    @staticmethod
    def _map_source_to_type(source: str) -> NodeType:
        """Map an UnderstandingNode source string to a NodeType."""
        mapping = {
            'composed_from_teaching': NodeType.RULE,
            'composed_from_observation': NodeType.CONCEPT,
            'composed_from_failure': NodeType.RULE,
            'self_discovered': NodeType.CONCEPT,
            'teaching': NodeType.RULE,
            'observation': NodeType.CONCEPT,
            'failure': NodeType.RULE,
        }
        return mapping.get(source, NodeType.CONCEPT)

    @staticmethod
    def _infer_relation_type(edge_type: str) -> RelationType:
        """Infer a RelationType from an UnderstandingGraph edge_type string.

        UnderstandingGraph edges use free-form strings like 'related',
        'parent', 'child', 'composes'. We map these to AGNN's typed
        RelationType enum.
        """
        edge_lower = edge_type.lower().strip()
        mapping = {
            'is_a': RelationType.CATEGORICAL,
            'instance_of': RelationType.CATEGORICAL,
            'member_of': RelationType.CATEGORICAL,
            'parent': RelationType.CATEGORICAL,
            'child': RelationType.CATEGORICAL,
            'causes': RelationType.CAUSAL,
            'results_in': RelationType.CAUSAL,
            'prevents': RelationType.CAUSAL,
            'negates': RelationType.DIFFERENTIAL,
            'contrasts_with': RelationType.DIFFERENTIAL,
            'excepts': RelationType.DIFFERENTIAL,
            'requires': RelationType.FUNCTIONAL,
            'enables': RelationType.FUNCTIONAL,
            'computes': RelationType.FUNCTIONAL,
            'before': RelationType.TEMPORAL,
            'after': RelationType.TEMPORAL,
            'contains': RelationType.SPATIAL,
            'related': RelationType.DISCURSIVE,
            'composes': RelationType.FUNCTIONAL,
            'generalizes': RelationType.CATEGORICAL,
        }
        return mapping.get(edge_lower, RelationType.DISCURSIVE)

    # ──────────────── Internal ────────────────

    def _rebuild_adjacency(self):
        """Rebuild the outgoing/incoming adjacency indices."""
        self._outgoing = {}
        self._incoming = {}
        for idx, edge in enumerate(self._edges):
            self._outgoing.setdefault(edge.source_id, []).append(idx)
            self._incoming.setdefault(edge.target_id, []).append(idx)

    # ──────────────── Serialization ────────────────

    def to_dict(self) -> dict:
        """Serialize the graph to a dictionary."""
        return {
            "embedding_dim": self._embedding_dim,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
            "edges": [edge.to_dict() for edge in self._edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AGNNGraph":
        """Deserialize a graph from a dictionary."""
        graph = cls(embedding_dim=d.get("embedding_dim", DEFAULT_EMBEDDING_DIM))
        for nid, ndata in d.get("nodes", {}).items():
            node = AGNNNode.from_dict(ndata)
            graph._nodes[nid] = node
            graph._outgoing.setdefault(nid, [])
            graph._incoming.setdefault(nid, [])
        for edata in d.get("edges", []):
            edge = TypedEdge.from_dict(edata)
            idx = len(graph._edges)
            graph._edges.append(edge)
            graph._outgoing.setdefault(edge.source_id, []).append(idx)
            graph._incoming.setdefault(edge.target_id, []).append(idx)
        return graph

    def __repr__(self) -> str:
        return f"AGNNGraph(nodes={self.node_count()}, edges={self.edge_count()}, dim={self._embedding_dim})"
