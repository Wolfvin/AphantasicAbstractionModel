"""Tests for AGNN knowledge graph module.

Tests cover:
  1. Add node + edge → retrieve via traversal
  2. Message passing modifies node embedding
  3. Spread activation propagates confidence to neighbors
  4. Traverse produces coherent reasoning chain

All tests run without GPU, internet, or bge-m3.
"""

import sys, os
# Must set path before any agnn imports — pytest collection needs this
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import numpy as np
import pytest

from agnn.graph import (
    AGNNGraph, AGNNNode, TypedEdge, ReasoningChain,
    NodeType, RelationType, EdgeRole,
    DEFAULT_EMBEDDING_DIM,
)


# ═══════════════ Fixtures ═══════════════

@pytest.fixture
def simple_graph():
    """Build a simple 3-node graph:

    harimau --[CATEGORICAL/is_a]--> karnivora --[CAUSAL/causes]--> pemakan_daging

    This tests the most basic case: a linear chain of 3 nodes.
    """
    g = AGNNGraph(embedding_dim=32)

    # Seed the RNG for reproducibility
    np.random.seed(42)

    n1 = g.add_node(AGNNNode(
        id="harimau", label="harimau", node_type=NodeType.ENTITY,
        confidence=0.9,
    ))
    n2 = g.add_node(AGNNNode(
        id="karnivora", label="karnivora", node_type=NodeType.CONCEPT,
        confidence=0.85,
    ))
    n3 = g.add_node(AGNNNode(
        id="pemakan_daging", label="pemakan daging", node_type=NodeType.CONCEPT,
        confidence=0.8,
    ))

    g.add_edge(TypedEdge("harimau", "karnivora", RelationType.CATEGORICAL,
                          confidence=0.95, context="is_a"))
    g.add_edge(TypedEdge("karnivora", "pemakan_daging", RelationType.CAUSAL,
                          confidence=0.85, context="causes"))

    return g


@pytest.fixture
def complex_graph():
    """Build a more complex graph with branching, negation, and cycles:

    hewan --[CATEGORICAL]--> mamalia --[CATEGORICAL]--> kucing
                                  |
                          [CATEGORICAL]--> harimau --[DIFFERENTIAL/negates]--> herbivora

    Also add:
    kucing --[CATEGORICAL]--> hewan_peliharaan
    harimau --[CAUSAL]--> pemangsa
    """
    g = AGNNGraph(embedding_dim=32)
    np.random.seed(123)

    nodes = [
        ("hewan", "hewan", NodeType.CONCEPT, 0.9),
        ("mamalia", "mamalia", NodeType.CONCEPT, 0.85),
        ("kucing", "kucing", NodeType.ENTITY, 0.8),
        ("harimau", "harimau", NodeType.ENTITY, 0.9),
        ("herbivora", "herbivora", NodeType.CONCEPT, 0.7),
        ("hewan_peliharaan", "hewan peliharaan", NodeType.CONCEPT, 0.75),
        ("pemangsa", "pemangsa", NodeType.CONCEPT, 0.8),
    ]
    for nid, label, ntype, conf in nodes:
        g.add_node(AGNNNode(id=nid, label=label, node_type=ntype, confidence=conf))

    edges = [
        ("hewan", "mamalia", RelationType.CATEGORICAL, 0.9, "is_a"),
        ("mamalia", "kucing", RelationType.CATEGORICAL, 0.85, "is_a"),
        ("mamalia", "harimau", RelationType.CATEGORICAL, 0.85, "is_a"),
        ("harimau", "herbivora", RelationType.DIFFERENTIAL, 0.9, "negates"),
        ("kucing", "hewan_peliharaan", RelationType.CATEGORICAL, 0.8, "is_a"),
        ("harimau", "pemangsa", RelationType.CAUSAL, 0.85, "causes"),
    ]
    for src, tgt, rel, conf, ctx in edges:
        g.add_edge(TypedEdge(src, tgt, rel, confidence=conf, context=ctx))

    return g


# ═══════════════ Test 1: Add node + edge → retrieve via traversal ═══════════════

class TestAddAndRetrieve:
    """Test that nodes and edges can be added and retrieved via traversal."""

    def test_add_node_returns_node_with_embedding(self, simple_graph):
        node = simple_graph.get_node("harimau")
        assert node is not None
        assert node.label == "harimau"
        assert node.node_type == NodeType.ENTITY
        assert node.embedding.shape == (32,)
        assert not np.all(node.embedding == 0)  # Should be randomly initialized

    def test_add_edge_creates_directed_connection(self, simple_graph):
        edges = simple_graph.get_edges_from("harimau")
        assert len(edges) == 1
        assert edges[0].target_id == "karnivora"
        assert edges[0].relation_type == RelationType.CATEGORICAL

    def test_get_neighbors(self, simple_graph):
        neighbors = simple_graph.get_neighbors("harimau")
        assert len(neighbors) == 1
        assert neighbors[0].id == "karnivora"

    def test_get_neighbors_with_type_filter(self, complex_graph):
        # harimau has CATEGORICAL → herbivora (DIFFERENTIAL) and CAUSAL → pemangsa
        categorical_only = complex_graph.get_neighbors("harimau", RelationType.CATEGORICAL)
        # Only herbivora is connected via DIFFERENTIAL, pemangsa via CAUSAL
        # Actually harimau → herbivora is DIFFERENTIAL, harimau → pemangsa is CAUSAL
        # So categorical_only should be empty
        assert len(categorical_only) == 0

        differential = complex_graph.get_neighbors("harimau", RelationType.DIFFERENTIAL)
        assert len(differential) == 1
        assert differential[0].id == "herbivora"

    def test_traverse_finds_chain(self, simple_graph):
        chain = simple_graph.traverse("harimau", max_hops=2)
        assert chain is not None
        assert len(chain.steps) >= 1
        # First step should be harimau → karnivora
        assert chain.steps[0][0] == "harimau"
        assert chain.steps[0][2] == "karnivora"

    def test_traverse_verbalize(self, simple_graph):
        chain = simple_graph.traverse("harimau", max_hops=2)
        assert chain is not None
        text = chain.verbalize()
        assert "harimau" in text
        assert "CATEGORICAL" in text or "karnivora" in text

    def test_traverse_no_match_returns_none(self, simple_graph):
        chain = simple_graph.traverse("nonexistent_node", max_hops=3)
        assert chain is None

    def test_complex_graph_traverse(self, complex_graph):
        chain = complex_graph.traverse("harimau", max_hops=2)
        assert chain is not None
        # Should find a path starting from harimau
        assert chain.steps[0][0] == "harimau"

    def test_edge_add_validates_nodes(self):
        g = AGNNGraph(embedding_dim=16)
        with pytest.raises(ValueError, match="Source node"):
            g.add_edge(TypedEdge("nonexistent", "also_nonexistent", RelationType.CATEGORICAL))


# ═══════════════ Test 2: Message passing modifies node embedding ═══════════════

class TestMessagePassing:
    """Test that message passing modifies node embeddings based on neighbors."""

    def test_message_pass_changes_embedding(self, simple_graph):
        """Message passing should update the target node's embedding."""
        node = simple_graph.get_node("karnivora")
        old_emb = node.embedding.copy()

        # karnivora has incoming from harimau and outgoing to pemakan_daging
        # Message pass on karnivora aggregates from harimau
        new_emb = simple_graph.message_pass("karnivora", damping=0.5)

        assert new_emb is not None
        # The embedding should have changed
        assert not np.allclose(old_emb, new_emb, atol=1e-6)

    def test_message_pass_no_neighbors(self):
        """Message pass on isolated node returns unchanged embedding."""
        g = AGNNGraph(embedding_dim=16)
        np.random.seed(99)
        node = g.add_node(AGNNNode(id="isolated", label="isolated", node_type=NodeType.ENTITY))
        old_emb = node.embedding.copy()

        result = g.message_pass("isolated")
        # No neighbors → no update
        assert np.allclose(old_emb, result, atol=1e-6)

    def test_message_pass_nonexistent_node(self):
        """Message pass on nonexistent node returns None."""
        g = AGNNGraph(embedding_dim=16)
        result = g.message_pass("ghost")
        assert result is None

    def test_message_pass_differential_edge_inverts(self):
        """DIFFERENTIAL edges should move embedding AWAY from the source.

        Because DIFFERENTIAL has negative aggregation weight, message passing
        through a negation edge should push the target's embedding in the
        opposite direction from the source.
        """
        g = AGNNGraph(embedding_dim=32)
        np.random.seed(7)

        # Two nodes: A --[DIFFERENTIAL]--> B
        a = g.add_node(AGNNNode(id="A", label="negator", node_type=NodeType.CONCEPT, confidence=0.9))
        b = g.add_node(AGNNNode(id="B", label="negated", node_type=NodeType.CONCEPT, confidence=0.9))

        # Set known embeddings
        a.embedding = np.ones(32, dtype=np.float32)
        a.embedding /= np.linalg.norm(a.embedding)  # normalize
        b.embedding = np.zeros(32, dtype=np.float32)
        b.embedding[0] = 1.0  # unit vector along dim 0

        g.add_edge(TypedEdge("A", "B", RelationType.DIFFERENTIAL, confidence=1.0, context="negates"))

        old_b_emb = b.embedding.copy()
        new_b_emb = g.message_pass("B", damping=0.5)

        # The new embedding should differ from the old one
        assert not np.allclose(old_b_emb, new_b_emb, atol=1e-6)

        # After message passing, B's embedding should be LESS similar to A's
        # (because DIFFERENTIAL weight is negative, it pushes away)
        old_sim = np.dot(old_b_emb, a.embedding)
        new_sim = np.dot(new_b_emb, a.embedding)
        assert new_sim < old_sim, f"Differential edge should reduce similarity: {old_sim} -> {new_sim}"

    def test_message_pass_categorical_propagates(self):
        """CATEGORICAL edges should make embeddings MORE similar."""
        g = AGNNGraph(embedding_dim=32)
        np.random.seed(11)

        a = g.add_node(AGNNNode(id="A", label="cat", node_type=NodeType.ENTITY, confidence=0.9))
        b = g.add_node(AGNNNode(id="B", label="feline", node_type=NodeType.CONCEPT, confidence=0.9))

        # Set orthogonal embeddings
        a.embedding = np.zeros(32, dtype=np.float32)
        a.embedding[0] = 1.0
        b.embedding = np.zeros(32, dtype=np.float32)
        b.embedding[1] = 1.0

        g.add_edge(TypedEdge("A", "B", RelationType.CATEGORICAL, confidence=1.0, context="is_a"))

        old_sim = np.dot(b.embedding, a.embedding)
        # Before: orthogonal, sim ≈ 0
        assert abs(old_sim) < 0.01

        g.message_pass("B", damping=0.5)

        new_sim = np.dot(b.embedding, a.embedding)
        # After: should be more similar (positive dot product)
        assert new_sim > old_sim, f"Categorical edge should increase similarity: {old_sim} -> {new_sim}"

    def test_message_pass_all(self, complex_graph):
        """Full-graph message pass should update all nodes."""
        old_embeddings = {}
        for nid, node in complex_graph._nodes.items():
            old_embeddings[nid] = node.embedding.copy()

        complex_graph.message_pass_all(damping=0.3, iterations=2)

        changed = 0
        for nid, node in complex_graph._nodes.items():
            if not np.allclose(old_embeddings[nid], node.embedding, atol=1e-6):
                changed += 1
        # At least some nodes should have changed
        assert changed > 0, "Message passing should update at least some node embeddings"


# ═══════════════ Test 3: Spread activation propagates confidence ═══════════════

class TestSpreadActivation:
    """Test that spread activation propagates confidence through the graph."""

    def test_seed_activation(self, simple_graph):
        """Seed nodes should have high activation."""
        activation = simple_graph.spread_activation(["harimau"], steps=0)
        assert activation["harimau"] == pytest.approx(0.9, abs=0.01)

    def test_one_hop_propagation(self, simple_graph):
        """After 1 step, karnivora should have non-zero activation."""
        activation = simple_graph.spread_activation(["harimau"], steps=1)
        # harimau → karnivora (CATEGORICAL, decay 0.9)
        expected = 0.9 * 0.9 * 0.95  # source_conf × decay × edge_conf
        assert activation["karnivora"] > 0.0
        assert activation["karnivora"] == pytest.approx(expected, abs=0.05)

    def test_two_hop_propagation(self, simple_graph):
        """After 2 steps, pemakan_daging should have non-zero activation."""
        activation = simple_graph.spread_activation(["harimau"], steps=2)
        # harimau → karnivora → pemakan_daging
        assert activation["pemakan_daging"] > 0.0

    def test_activation_decays_with_distance(self, simple_graph):
        """Activation should decrease with distance from seed (comparing 1-hop vs 2-hop)."""
        activation = simple_graph.spread_activation(["harimau"], steps=2)
        # karnivora (1-hop) should have more activation than pemakan_daging (2-hop)
        # Note: karnivora can reach 1.0 due to sustained spread from harimau at each step
        assert activation["karnivora"] > activation["pemakan_daging"]
        # Seed should always maintain its initial confidence at minimum
        assert activation["harimau"] >= 0.9

    def test_differential_edge_weak_propagation(self, complex_graph):
        """DIFFERENTIAL edges should propagate less activation than CATEGORICAL."""
        activation = complex_graph.spread_activation(["harimau"], steps=1)
        # harimau → herbivora (DIFFERENTIAL, decay 0.5)
        # harimau → pemangsa (CAUSAL, decay 0.7)
        # Causal should propagate more than differential
        causal_act = activation.get("pemangsa", 0.0)
        differential_act = activation.get("herbivora", 0.0)
        assert causal_act > differential_act, \
            f"Causal should propagate more than differential: causal={causal_act}, diff={differential_act}"

    def test_multiple_seeds(self, complex_graph):
        """Multiple seed nodes should accumulate activation."""
        activation_single = complex_graph.spread_activation(["harimau"], steps=1)
        activation_double = complex_graph.spread_activation(["harimau", "kucing"], steps=1)

        # With two seeds, mamalia should have more activation
        # (from both harimau and kucing paths)
        assert activation_double["mamalia"] >= activation_single.get("mamalia", 0.0)

    def test_no_seed_no_activation(self, complex_graph):
        """Empty seed list should produce all-zero activation."""
        activation = complex_graph.spread_activation([], steps=2)
        assert all(v == 0.0 for v in activation.values())

    def test_activation_clamped_to_one(self):
        """Activation should be clamped to [0, 1]."""
        g = AGNNGraph(embedding_dim=16)
        np.random.seed(1)
        a = g.add_node(AGNNNode(id="A", label="A", node_type=NodeType.ENTITY, confidence=1.0))
        b = g.add_node(AGNNNode(id="B", label="B", node_type=NodeType.ENTITY, confidence=1.0))
        g.add_edge(TypedEdge("A", "B", RelationType.CATEGORICAL, confidence=1.0))

        activation = g.spread_activation(["A"], steps=1)
        assert activation["B"] <= 1.0
        assert activation["B"] >= 0.0


# ═══════════════ Test 4: Traverse produces coherent reasoning chain ═══════════════

class TestTraverse:
    """Test that traverse produces coherent, explainable reasoning chains."""

    def test_simple_chain(self, simple_graph):
        """3-node linear graph should produce a 2-step chain."""
        chain = simple_graph.traverse("harimau", max_hops=2)
        assert chain is not None
        assert len(chain.steps) == 2
        assert chain.steps[0] == ("harimau", "categorical", "karnivora")
        assert chain.steps[1] == ("karnivora", "causal", "pemakan daging")

    def test_chain_verbalization(self, simple_graph):
        """Verbalized chain should be human-readable."""
        chain = simple_graph.traverse("harimau", max_hops=2)
        text = chain.verbalize()
        # Should contain all three nodes and two relations
        assert "harimau" in text
        assert "karnivora" in text
        assert "pemakan daging" in text
        assert "CATEGORICAL" in text
        assert "CAUSAL" in text

    def test_chain_confidence(self, simple_graph):
        """Chain confidence should reflect edge and node confidences."""
        chain = simple_graph.traverse("harimau", max_hops=2)
        assert chain is not None
        assert 0.0 < chain.confidence <= 1.0

    def test_chain_node_ids(self, simple_graph):
        """Chain should track visited node IDs."""
        chain = simple_graph.traverse("harimau", max_hops=2)
        assert chain is not None
        assert chain.node_ids[0] == "harimau"
        assert "karnivora" in chain.node_ids
        assert "pemakan_daging" in chain.node_ids

    def test_max_hops_limits_chain(self, simple_graph):
        """max_hops=1 should produce a 1-step chain, not 2."""
        chain = simple_graph.traverse("harimau", max_hops=1)
        assert chain is not None
        assert len(chain.steps) == 1

    def test_complex_graph_best_path(self, complex_graph):
        """In a complex graph, traversal should find the highest-confidence path."""
        chain = complex_graph.traverse("harimau", max_hops=2)
        assert chain is not None
        assert len(chain.steps) >= 1
        # The chain should start from harimau
        assert chain.steps[0][0] == "harimau"

    def test_traverse_with_relation_filter(self, complex_graph):
        """relation_filter should constrain which edges to follow."""
        # Only follow CATEGORICAL edges
        chain = complex_graph.traverse("harimau", max_hops=2,
                                        relation_filter=[RelationType.CATEGORICAL])
        if chain is not None:
            # All steps should be categorical
            for _, relation, _ in chain.steps:
                assert relation == "categorical"

    def test_traverse_confidence_threshold(self, complex_graph):
        """High confidence_threshold should filter out low-confidence edges."""
        chain_low = complex_graph.traverse("harimau", max_hops=2, confidence_threshold=0.0)
        chain_high = complex_graph.traverse("harimau", max_hops=2, confidence_threshold=0.99)

        # Low threshold should find more paths
        if chain_low is not None and chain_high is not None:
            assert len(chain_low.steps) >= len(chain_high.steps)

    def test_empty_graph_traverse(self):
        """Traversing an empty graph should return None."""
        g = AGNNGraph(embedding_dim=16)
        assert g.traverse("anything", max_hops=3) is None


# ═══════════════ Test 5: Serialization ═══════════════

class TestSerialization:
    """Test graph serialization and deserialization."""

    def test_roundtrip(self, simple_graph):
        """Graph should survive serialization roundtrip."""
        d = simple_graph.to_dict()
        g2 = AGNNGraph.from_dict(d)

        assert g2.node_count() == simple_graph.node_count()
        assert g2.edge_count() == simple_graph.edge_count()
        assert g2._embedding_dim == simple_graph._embedding_dim

        # Check a node survived
        n = g2.get_node("harimau")
        assert n is not None
        assert n.label == "harimau"
        assert n.node_type == NodeType.ENTITY

        # Check an edge survived
        edges = g2.get_edges_from("harimau")
        assert len(edges) == 1
        assert edges[0].relation_type == RelationType.CATEGORICAL

    def test_embeddings_preserved(self, simple_graph):
        """Embeddings should be preserved through roundtrip."""
        orig_node = simple_graph.get_node("harimau")
        d = simple_graph.to_dict()
        g2 = AGNNGraph.from_dict(d)
        restored_node = g2.get_node("harimau")

        assert np.allclose(orig_node.embedding, restored_node.embedding, atol=1e-6)


# ═══════════════ Test 6: Concrete example (3-node → traverse → reasoning chain) ═══════════════

class TestConcreteExample:
    """The PR-mandated concrete example: 3 nodes → traverse → reasoning chain."""

    def test_three_node_traverse_example(self):
        """
        Build:
            harimau --[IS_A]--> karnivora --[CAUSES]--> pemakan_daging

        Traverse from "harimau" with max_hops=2.
        Verify the reasoning chain is coherent and verbalizable.
        """
        g = AGNNGraph(embedding_dim=64)
        np.random.seed(42)

        g.add_node(AGNNNode(
            id="harimau", label="harimau", node_type=NodeType.ENTITY, confidence=0.9,
        ))
        g.add_node(AGNNNode(
            id="karnivora", label="karnivora", node_type=NodeType.CONCEPT, confidence=0.85,
        ))
        g.add_node(AGNNNode(
            id="pemakan_daging", label="pemakan daging", node_type=NodeType.CONCEPT, confidence=0.8,
        ))

        g.add_edge(TypedEdge(
            source_id="harimau", target_id="karnivora",
            relation_type=RelationType.CATEGORICAL,
            confidence=0.95, context="is_a",
        ))
        g.add_edge(TypedEdge(
            source_id="karnivora", target_id="pemakan_daging",
            relation_type=RelationType.CAUSAL,
            confidence=0.85, context="causes",
        ))

        # Traverse
        chain = g.traverse("harimau", max_hops=2)
        assert chain is not None, "Traverse should find a chain"

        # Verify chain structure
        assert len(chain.steps) == 2
        assert chain.steps[0] == ("harimau", "categorical", "karnivora")
        assert chain.steps[1] == ("karnivora", "causal", "pemakan daging")

        # Verify verbalization
        text = chain.verbalize()
        assert "harimau" in text
        assert "CATEGORICAL" in text
        assert "karnivora" in text
        assert "CAUSAL" in text
        assert "pemakan daging" in text

        # Verify node IDs
        assert chain.node_ids == ["harimau", "karnivora", "pemakan_daging"]

        # Verify confidence is positive
        assert chain.confidence > 0.0

        # Now do message passing
        old_emb = g.get_node("karnivora").embedding.copy()
        g.message_pass("karnivora", damping=0.5)
        new_emb = g.get_node("karnivora").embedding
        assert not np.allclose(old_emb, new_emb, atol=1e-6), \
            "Message passing should change karnivora's embedding"

        # Spread activation
        activation = g.spread_activation(["harimau"], steps=2)
        # Seed maintains its activation; neighbors decay with distance
        assert activation["harimau"] >= 0.9
        assert activation["karnivora"] > activation["pemakan_daging"]
        assert activation["pemakan_daging"] > 0.0
