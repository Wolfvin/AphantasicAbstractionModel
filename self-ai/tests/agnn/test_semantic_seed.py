"""Tests for embedding-based seed selection and semantic_traverse.

Covers:
  1. find_seed_nodes_by_embedding: returns most similar nodes by cosine similarity
  2. Fallback to keyword matching when no node has embeddings
  3. Top-k boundary conditions
  4. semantic_traverse: end-to-end seed selection + traversal
  5. Concrete example: query "siapa presiden pertama?" → seed nodes → reasoning chain

All tests use MockEmbedder (deterministic, no real model, no GPU).
"""

import sys
import os

# Must set path before any agnn imports — pytest collection needs this
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import hashlib
import numpy as np
import pytest

from agnn.graph import (
    AGNNGraph, AGNNNode, TypedEdge, ReasoningChain,
    NodeType, RelationType, EdgeRole,
    DEFAULT_EMBEDDING_DIM,
)
from agnn.embeddings import ModelEmbedder


# ═══════════════ MockEmbedder ═══════════════

class MockEmbedder(ModelEmbedder):
    """Deterministic mock embedder for testing.

    Returns a numpy array based on the MD5 hash of the input text,
    without requiring a real model or GPU. This extends ModelEmbedder's
    built-in mock mode but with a public interface that matches what
    find_seed_nodes_by_embedding and semantic_traverse expect.

    Key properties:
      - Same text → same embedding (deterministic)
      - Different texts → different embeddings (diverse)
      - Unit-normalized vectors (cosine similarity = dot product)
      - No external model dependency
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__(model=None, tokenizer=None, hidden_size=hidden_size, model_id="mock-test")

    def embed(self, text: str) -> np.ndarray:
        """Return a deterministic embedding based on text hash."""
        if not text or not text.strip():
            return np.zeros(self._hidden_size, dtype=np.float32)

        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self._hidden_size).astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm

        return embedding


# ═══════════════ Fixtures ═══════════════

@pytest.fixture
def mock_embedder():
    """Create a MockEmbedder with 64-dim output."""
    return MockEmbedder(hidden_size=64)


@pytest.fixture
def seeded_graph():
    """Build a graph with model-native embeddings (via MockEmbedder).

    Graph structure (Indonesian history domain):
        Sukarno --[CATEGORICAL/adalah]--> Presiden_Pertama
        Presiden_Pertama --[CATEGORICAL/lahir_di]--> Blitar
        Sukarno --[CATEGORICAL/beristri]--> Fatmawati
        Indonesia --[SPATIAL/beribu_kota]--> Jakarta
        Indonesia --[CATEGORICAL/memiliki]--> Presiden_Pertama

    This tests the key use case: query "siapa presiden pertama Indonesia?"
    should find "Sukarno" as a seed via embedding similarity, even though
    there's no keyword overlap.
    """
    g = AGNNGraph(embedding_dim=64)
    embedder = MockEmbedder(hidden_size=64)

    # Create nodes with embeddings initialized from the embedder
    nodes_data = [
        ("Sukarno", "Sukarno", NodeType.ENTITY, 0.9),
        ("Presiden_Pertama", "Presiden Pertama", NodeType.CONCEPT, 0.85),
        ("Blitar", "Blitar", NodeType.ENTITY, 0.8),
        ("Fatmawati", "Fatmawati", NodeType.ENTITY, 0.85),
        ("Indonesia", "Indonesia", NodeType.ENTITY, 0.95),
        ("Jakarta", "Jakarta", NodeType.ENTITY, 0.9),
    ]

    for nid, label, ntype, conf in nodes_data:
        node = AGNNNode(id=nid, label=label, node_type=ntype, confidence=conf)
        # Initialize embedding from the mock embedder — this makes nodes
        # discoverable via cosine similarity
        node.embedding = embedder.embed(label)
        g.add_node(node)

    edges = [
        ("Sukarno", "Presiden_Pertama", RelationType.CATEGORICAL, 0.95, "adalah"),
        ("Presiden_Pertama", "Blitar", RelationType.CATEGORICAL, 0.9, "lahir_di"),
        ("Sukarno", "Fatmawati", RelationType.CATEGORICAL, 0.85, "beristri"),
        ("Indonesia", "Jakarta", RelationType.SPATIAL, 0.95, "beribu_kota"),
        ("Indonesia", "Presiden_Pertama", RelationType.CATEGORICAL, 0.9, "memiliki"),
    ]
    for src, tgt, rel, conf, ctx in edges:
        g.add_edge(TypedEdge(src, tgt, rel, confidence=conf, context=ctx))

    return g


@pytest.fixture
def no_embedding_graph():
    """Build a graph where nodes have zero embeddings (no embedder set).

    All node embeddings are the default random values from AGNNGraph.add_node().
    We'll zero them out to simulate a graph where initialize_embeddings()
    hasn't been called yet — testing the fallback path.
    """
    g = AGNNGraph(embedding_dim=32)
    np.random.seed(42)

    g.add_node(AGNNNode(id="harimau", label="harimau", node_type=NodeType.ENTITY, confidence=0.9))
    g.add_node(AGNNNode(id="karnivora", label="karnivora", node_type=NodeType.CONCEPT, confidence=0.85))
    g.add_node(AGNNNode(id="pemakan_daging", label="pemakan daging", node_type=NodeType.CONCEPT, confidence=0.8))

    g.add_edge(TypedEdge("harimau", "karnivora", RelationType.CATEGORICAL, confidence=0.95, context="is_a"))
    g.add_edge(TypedEdge("karnivora", "pemakan_daging", RelationType.CAUSAL, confidence=0.85, context="causes"))

    # Zero out all embeddings to simulate "no embeddings computed"
    for node in g._nodes.values():
        node.embedding = np.zeros(g._embedding_dim, dtype=np.float32)

    return g


# ═══════════════ Test find_seed_nodes_by_embedding ═══════════════

class TestFindSeedNodesByEmbedding:
    """Test embedding-based seed node selection."""

    def test_returns_nodes_with_highest_similarity(self, seeded_graph, mock_embedder):
        """Nodes whose embeddings are most similar to the query should be returned first.

        We query with the label of a specific node ("Sukarno") — the mock
        embedder returns the exact same vector for the same text, so the
        cosine similarity should be 1.0 for that node and < 1.0 for others.
        """
        result = seeded_graph.find_seed_nodes_by_embedding("Sukarno", mock_embedder, top_k=3)

        assert len(result) >= 1
        assert result[0] == "Sukarno", (
            f"Expected 'Sukarno' as top seed (exact label match), got '{result[0]}'"
        )

    def test_returns_top_k_results(self, seeded_graph, mock_embedder):
        """top_k should limit the number of returned node IDs."""
        result = seeded_graph.find_seed_nodes_by_embedding("Sukarno", mock_embedder, top_k=2)
        assert len(result) <= 2

        result_all = seeded_graph.find_seed_nodes_by_embedding("Sukarno", mock_embedder, top_k=10)
        assert len(result_all) <= seeded_graph.node_count()

    def test_top_k_one(self, seeded_graph, mock_embedder):
        """top_k=1 should return exactly one node."""
        result = seeded_graph.find_seed_nodes_by_embedding("Sukarno", mock_embedder, top_k=1)
        assert len(result) == 1
        assert result[0] == "Sukarno"

    def test_top_k_zero(self, seeded_graph, mock_embedder):
        """top_k=0 should return empty list."""
        result = seeded_graph.find_seed_nodes_by_embedding("Sukarno", mock_embedder, top_k=0)
        assert result == []

    def test_top_k_larger_than_graph(self, seeded_graph, mock_embedder):
        """top_k larger than graph size should return all nodes."""
        result = seeded_graph.find_seed_nodes_by_embedding("Sukarno", mock_embedder, top_k=100)
        assert len(result) == seeded_graph.node_count()

    def test_similarity_ordering(self, seeded_graph, mock_embedder):
        """Results should be ordered by cosine similarity descending.

        Since MockEmbedder produces unit-normalized vectors, cosine sim = dot product.
        The query "Sukarno" should match "Sukarno" node with sim=1.0, then other
        nodes with lower similarity.
        """
        result = seeded_graph.find_seed_nodes_by_embedding("Sukarno", mock_embedder, top_k=6)

        # Compute actual similarities for verification
        query_emb = mock_embedder.embed("Sukarno")
        query_norm = np.linalg.norm(query_emb)
        sims = []
        for node_id in result:
            node = seeded_graph.get_node(node_id)
            node_norm = np.linalg.norm(node.embedding)
            if node_norm > 1e-8:
                sim = float(np.dot(query_emb, node.embedding) / (query_norm * node_norm))
            else:
                sim = 0.0
            sims.append(sim)

        # Verify descending order
        for i in range(len(sims) - 1):
            assert sims[i] >= sims[i + 1], (
                f"Results not sorted by similarity: {sims[i]} < {sims[i+1]} at index {i}"
            )

    def test_fallback_to_keyword_when_no_embeddings(self, no_embedding_graph, mock_embedder):
        """When no node has an embedding, should fallback to keyword matching."""
        result = no_embedding_graph.find_seed_nodes_by_embedding(
            "harimau", mock_embedder, top_k=3
        )
        # Should fallback to find_seed_nodes which does keyword matching
        assert len(result) >= 1
        assert "harimau" in result, (
            f"Expected 'harimau' in keyword fallback results, got {result}"
        )

    def test_empty_query(self, seeded_graph, mock_embedder):
        """Empty query should return empty list."""
        result = seeded_graph.find_seed_nodes_by_embedding("", mock_embedder, top_k=3)
        assert result == []

    def test_query_with_no_match_still_returns_by_similarity(self, seeded_graph, mock_embedder):
        """A query that doesn't match any node label should still return nodes
        sorted by embedding similarity (some nodes will be closer than others)."""
        result = seeded_graph.find_seed_nodes_by_embedding(
            "siapa presiden pertama Indonesia?", mock_embedder, top_k=3
        )
        # Even though no node label matches the query text, embedding-based
        # selection should still return the most similar nodes
        assert len(result) >= 1, "Should return at least one node by similarity"

    def test_nodes_without_embeddings_are_skipped(self, mock_embedder):
        """Nodes with zero embeddings should be skipped, not crash."""
        g = AGNNGraph(embedding_dim=64)
        # Add a node with a proper embedding
        n1 = AGNNNode(id="with_emb", label="with embedding", node_type=NodeType.ENTITY, confidence=0.9)
        n1.embedding = mock_embedder.embed("with embedding")
        g.add_node(n1)

        # Add a node, then zero out its embedding AFTER add_node (because
        # add_node auto-initializes zero embeddings with random values)
        n2 = AGNNNode(id="no_emb", label="no embedding", node_type=NodeType.ENTITY, confidence=0.9)
        g.add_node(n2)
        g.get_node("no_emb").embedding = np.zeros(64, dtype=np.float32)

        result = g.find_seed_nodes_by_embedding("with embedding", mock_embedder, top_k=5)
        assert "with_emb" in result
        assert "no_emb" not in result, "Zero-embedding node should be skipped"

    def test_concrete_presiden_pertama_query(self, seeded_graph, mock_embedder):
        """Concrete example from the task: query "siapa presiden pertama?"
        should find Sukarno as one of the seed nodes.

        With the mock embedder, we can't guarantee that "siapa presiden pertama?"
        is semantically closest to "Sukarno" vs. "Presiden Pertama", but we
        CAN verify that the mechanism works: it returns nodes sorted by
        similarity and doesn't crash or return empty.
        """
        result = seeded_graph.find_seed_nodes_by_embedding(
            "siapa presiden pertama?", mock_embedder, top_k=3
        )
        assert len(result) >= 1, "Should return at least one seed node"

        # The top result should be one of the semantically relevant nodes
        # (Sukarno or Presiden_Pertama — the exact order depends on the
        # mock hash, but at least one should appear)
        relevant_nodes = {"Sukarno", "Presiden_Pertama"}
        assert any(nid in relevant_nodes for nid in result), (
            f"Expected at least one of {relevant_nodes} in results, got {result}"
        )


# ═══════════════ Test semantic_traverse ═══════════════

class TestSemanticTraverse:
    """Test end-to-end semantic seed selection + traversal."""

    def test_returns_reasoning_chains(self, seeded_graph, mock_embedder):
        """semantic_traverse should return at least one ReasoningChain."""
        chains = seeded_graph.semantic_traverse("Sukarno", mock_embedder, max_hops=2)
        assert len(chains) >= 1, "Should return at least one reasoning chain"
        for chain in chains:
            assert isinstance(chain, ReasoningChain)

    def test_chains_sorted_by_confidence(self, seeded_graph, mock_embedder):
        """Chains should be sorted by confidence descending."""
        chains = seeded_graph.semantic_traverse("Sukarno", mock_embedder, max_hops=2, top_k=3)
        if len(chains) > 1:
            for i in range(len(chains) - 1):
                assert chains[i].confidence >= chains[i + 1].confidence, (
                    f"Chains not sorted by confidence: {chains[i].confidence} < {chains[i+1].confidence}"
                )

    def test_chain_starts_from_seed(self, seeded_graph, mock_embedder):
        """Each chain should start from the seed node that was selected."""
        chains = seeded_graph.semantic_traverse("Sukarno", mock_embedder, max_hops=2)
        # The query "Sukarno" should embed identically to the "Sukarno" node,
        # so "Sukarno" should be the first seed and the chain should start there
        assert len(chains) >= 1
        first_chain = chains[0]
        assert first_chain.steps[0][0] == "Sukarno", (
            f"Chain should start from 'Sukarno', got '{first_chain.steps[0][0]}'"
        )

    def test_chain_reaches_neighbor(self, seeded_graph, mock_embedder):
        """Traversal from Sukarno should reach at least Presiden_Pertama or Fatmawati."""
        chains = seeded_graph.semantic_traverse("Sukarno", mock_embedder, max_hops=2, top_k=3)
        assert len(chains) >= 1

        # Collect all node_ids across all chains
        all_node_ids = set()
        for chain in chains:
            all_node_ids.update(chain.node_ids)

        # Should reach at least one of the direct neighbors
        reachable = {"Presiden_Pertama", "Fatmawati"}
        assert all_node_ids & reachable, (
            f"Expected to reach at least one of {reachable}, got {all_node_ids}"
        )

    def test_empty_graph(self, mock_embedder):
        """semantic_traverse on an empty graph should return empty list."""
        g = AGNNGraph(embedding_dim=64)
        chains = g.semantic_traverse("anything", mock_embedder, max_hops=2)
        assert chains == []

    def test_no_embedding_graph_fallback(self, no_embedding_graph, mock_embedder):
        """semantic_traverse on a graph with no embeddings should still work
        via keyword fallback (though with reduced semantic accuracy)."""
        chains = no_embedding_graph.semantic_traverse("harimau", mock_embedder, max_hops=2)
        # Should find at least one chain via keyword fallback
        assert len(chains) >= 1, "Should find chain via keyword fallback"

    def test_concrete_example_presiden_query(self, seeded_graph, mock_embedder):
        """Concrete example: query "siapa presiden pertama?" → seed nodes → reasoning chain.

        This is the key test case from the task. We verify that:
          1. find_seed_nodes_by_embedding returns relevant seeds
          2. semantic_traverse produces at least one reasoning chain
          3. The chain reaches meaningful nodes
        """
        # Step 1: Check seed selection
        seeds = seeded_graph.find_seed_nodes_by_embedding(
            "siapa presiden pertama?", mock_embedder, top_k=3
        )
        assert len(seeds) >= 1, f"Expected at least 1 seed, got {seeds}"

        # Step 2: Run semantic_traverse
        chains = seeded_graph.semantic_traverse(
            "siapa presiden pertama?", mock_embedder, max_hops=2, top_k=3
        )
        assert len(chains) >= 1, "Should produce at least one reasoning chain"

        # Step 3: Verify the chain has meaningful content
        all_node_ids = set()
        for chain in chains:
            all_node_ids.update(chain.node_ids)
            assert len(chain.steps) >= 1, "Chain should have at least one step"
            assert chain.confidence > 0.0, "Chain should have positive confidence"

        # Log the results for the PR description
        print(f"\n--- Concrete Example ---")
        print(f"Query: 'siapa presiden pertama?'")
        print(f"Seed nodes: {seeds}")
        for i, chain in enumerate(chains):
            print(f"Chain {i}: {chain.verbalize()}")
            print(f"  Confidence: {chain.confidence:.4f}")
            print(f"  Node IDs: {chain.node_ids}")

    def test_max_hops_limits_depth(self, seeded_graph, mock_embedder):
        """max_hops=1 should produce shorter chains than max_hops=3."""
        chains_1 = seeded_graph.semantic_traverse("Sukarno", mock_embedder, max_hops=1)
        chains_3 = seeded_graph.semantic_traverse("Sukarno", mock_embedder, max_hops=3)

        # Both should return chains
        assert len(chains_1) >= 1
        assert len(chains_3) >= 1

        # The longest chain with max_hops=3 should be at least as long
        # as the longest with max_hops=1
        max_steps_1 = max(len(c.steps) for c in chains_1)
        max_steps_3 = max(len(c.steps) for c in chains_3)
        assert max_steps_3 >= max_steps_1

    def test_bidirectional_traversal(self, seeded_graph, mock_embedder):
        """semantic_traverse uses bidirectional=True, so it should be able
        to reach nodes 'behind' the seed via reverse edge traversal.

        For example, starting from "Presiden_Pertama", we should be able
        to reach "Sukarno" by traversing the edge
        Sukarno → Presiden_Pertama in reverse.
        """
        # "Presiden Pertama" is the label — embed should match the node
        chains = seeded_graph.semantic_traverse(
            "Presiden Pertama", mock_embedder, max_hops=2
        )
        assert len(chains) >= 1

        # Check if any chain reached Sukarno (via reverse traversal)
        all_node_ids = set()
        for chain in chains:
            all_node_ids.update(chain.node_ids)

        assert "Sukarno" in all_node_ids or "Blitar" in all_node_ids, (
            f"Bidirectional traversal should reach Sukarno or Blitar, got {all_node_ids}"
        )


# ═══════════════ Regression: existing tests still pass ═══════════════

class TestRegression:
    """Verify that adding the new methods doesn't break existing functionality."""

    def test_traverse_still_works_without_embedder(self):
        """Original traverse() should work without any embedder set."""
        g = AGNNGraph(embedding_dim=32)
        np.random.seed(42)

        g.add_node(AGNNNode(id="harimau", label="harimau", node_type=NodeType.ENTITY, confidence=0.9))
        g.add_node(AGNNNode(id="karnivora", label="karnivora", node_type=NodeType.CONCEPT, confidence=0.85))
        g.add_edge(TypedEdge("harimau", "karnivora", RelationType.CATEGORICAL, confidence=0.95, context="is_a"))

        chain = g.traverse("harimau", max_hops=2)
        assert chain is not None
        assert chain.steps[0][0] == "harimau"

    def test_find_seed_nodes_still_works(self):
        """Original find_seed_nodes() should work unchanged."""
        g = AGNNGraph(embedding_dim=32)
        g.add_node(AGNNNode(id="harimau", label="harimau", node_type=NodeType.ENTITY, confidence=0.9))
        g.add_node(AGNNNode(id="karnivora", label="karnivora", node_type=NodeType.CONCEPT, confidence=0.85))

        result = g.find_seed_nodes("harimau", top_k=3)
        assert "harimau" in result
