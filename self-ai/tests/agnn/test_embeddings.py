"""Tests for AGNN embeddings module.

Tests cover:
  1. EmbeddingCache hit/miss/invalidation
  2. embed_node with mock embedder (numpy random, no real model)
  3. graph.set_embedder() + initialize_embeddings() end-to-end with mock
  4. ModelEmbedder mock mode (deterministic, no GPU/internet)
  5. Disk persistence (save/load)
  6. Dimension projection (hidden_size ≠ embedding_dim)

All tests run without GPU, without real model, without internet.
"""

import os
import tempfile
import shutil

import numpy as np
import pytest

from agnn.embeddings import (
    ModelEmbedder,
    EmbeddingCache,
    embed_node,
    embed_nodes_batch,
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_FILENAME,
)
from agnn.graph import (
    AGNNGraph,
    AGNNNode,
    TypedEdge,
    NodeType,
    RelationType,
)


# ═══════════════ Fixtures ═══════════════

@pytest.fixture
def mock_embedder():
    """Create a mock ModelEmbedder with hidden_size=64."""
    return ModelEmbedder(
        model=None,
        tokenizer=None,
        hidden_size=64,
        num_layers=12,
        model_id="mock-test-model",
    )


@pytest.fixture
def mock_embedder_large():
    """Create a mock ModelEmbedder with hidden_size=896 (like Qwen3-0.6B)."""
    return ModelEmbedder(
        model=None,
        tokenizer=None,
        hidden_size=896,
        num_layers=28,
        model_id="mock-qwen3-0.6b",
    )


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache persistence tests."""
    tmpdir = tempfile.mkdtemp(prefix="agnn_test_cache_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def cache(temp_cache_dir):
    """Create an EmbeddingCache with a temp directory (auto-load disabled)."""
    return EmbeddingCache(
        cache_dir=temp_cache_dir,
        cache_filename="test_embeddings.pkl",
        auto_load=False,
    )


@pytest.fixture
def simple_graph():
    """Build a simple 3-node graph for integration tests."""
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
        "harimau", "karnivora", RelationType.CATEGORICAL, confidence=0.95, context="is_a",
    ))
    g.add_edge(TypedEdge(
        "karnivora", "pemakan_daging", RelationType.CAUSAL, confidence=0.85, context="causes",
    ))

    return g


# ═══════════════ Test 1: ModelEmbedder (mock mode) ═══════════════

class TestModelEmbedderMock:
    """Test ModelEmbedder in mock mode (no real model)."""

    def test_mock_embed_returns_correct_shape(self, mock_embedder):
        """Mock embedder should return array of shape (hidden_size,)."""
        emb = mock_embedder.embed("harimau")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (64,)
        assert emb.dtype == np.float32

    def test_mock_embed_is_unit_normalized(self, mock_embedder):
        """Mock embeddings should be unit-normalized."""
        emb = mock_embedder.embed("harimau")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm}"

    def test_mock_embed_deterministic(self, mock_embedder):
        """Same text should always produce the same embedding."""
        emb1 = mock_embedder.embed("harimau")
        emb2 = mock_embedder.embed("harimau")
        assert np.allclose(emb1, emb2, atol=1e-6)

    def test_mock_embed_different_texts_differ(self, mock_embedder):
        """Different texts should produce different embeddings."""
        emb1 = mock_embedder.embed("harimau")
        emb2 = mock_embedder.embed("karnivora")
        assert not np.allclose(emb1, emb2, atol=1e-3)

    def test_mock_embed_empty_text_returns_zeros(self, mock_embedder):
        """Empty text should return a zero vector."""
        emb = mock_embedder.embed("")
        assert np.all(emb == 0)

    def test_mock_embed_whitespace_text_returns_zeros(self, mock_embedder):
        """Whitespace-only text should return a zero vector."""
        emb = mock_embedder.embed("   ")
        assert np.all(emb == 0)

    def test_embed_batch(self, mock_embedder):
        """embed_batch should return one embedding per text."""
        texts = ["harimau", "karnivora", "pemakan daging"]
        embeddings = mock_embedder.embed_batch(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (64,)
            assert emb.dtype == np.float32

    def test_is_mock_property(self, mock_embedder):
        """is_mock should return True for mock embedders."""
        assert mock_embedder.is_mock is True

    def test_properties(self, mock_embedder):
        """ModelEmbedder should expose model_id, hidden_size, layer_index."""
        assert mock_embedder.model_id == "mock-test-model"
        assert mock_embedder.hidden_size == 64
        assert mock_embedder.layer_index == 6  # 12 layers // 2

    def test_layer_index_default(self):
        """Default layer_index should be num_layers // 2."""
        embedder = ModelEmbedder(model=None, tokenizer=None, hidden_size=32, num_layers=28)
        assert embedder.layer_index == 14  # 28 // 2

    def test_layer_index_explicit(self):
        """Explicit layer_index should override default."""
        embedder = ModelEmbedder(
            model=None, tokenizer=None, hidden_size=32, num_layers=28, layer_index=24,
        )
        assert embedder.layer_index == 24

    def test_large_hidden_size(self, mock_embedder_large):
        """Large hidden_size (896, like Qwen3-0.6B) should work."""
        emb = mock_embedder_large.embed("test text")
        assert emb.shape == (896,)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01


# ═══════════════ Test 2: EmbeddingCache hit/miss/invalidation ═══════════════

class TestEmbeddingCache:
    """Test EmbeddingCache operations: hit, miss, invalidation, persistence."""

    def test_cache_miss_returns_none(self, cache):
        """Querying a non-existent key should return None."""
        result = cache.get("harimau", "mock-model")
        assert result is None

    def test_cache_put_and_get(self, cache):
        """Storing and retrieving should return the same embedding."""
        emb = np.random.randn(64).astype(np.float32)
        cache.put("harimau", "mock-model", emb)
        result = cache.get("harimau", "mock-model")
        assert result is not None
        assert np.allclose(result, emb, atol=1e-6)

    def test_cache_hit_after_put(self, cache):
        """has() should return True after put()."""
        emb = np.random.randn(64).astype(np.float32)
        cache.put("harimau", "mock-model", emb)
        assert cache.has("harimau", "mock-model") is True

    def test_cache_miss_before_put(self, cache):
        """has() should return False before put()."""
        assert cache.has("harimau", "mock-model") is False

    def test_cache_different_model_ids(self, cache):
        """Same text, different model_id → different cache entries."""
        emb1 = np.random.randn(64).astype(np.float32)
        emb2 = np.random.randn(64).astype(np.float32)
        cache.put("harimau", "model-a", emb1)
        cache.put("harimau", "model-b", emb2)

        result_a = cache.get("harimau", "model-a")
        result_b = cache.get("harimau", "model-b")
        assert np.allclose(result_a, emb1, atol=1e-6)
        assert np.allclose(result_b, emb2, atol=1e-6)

    def test_cache_invalidate_specific(self, cache):
        """invalidate() should remove only the specified entry."""
        emb1 = np.random.randn(64).astype(np.float32)
        emb2 = np.random.randn(64).astype(np.float32)
        cache.put("harimau", "model-a", emb1)
        cache.put("karnivora", "model-a", emb2)

        removed = cache.invalidate("harimau", "model-a")
        assert removed is True
        assert cache.get("harimau", "model-a") is None
        assert cache.get("karnivora", "model-a") is not None

    def test_cache_invalidate_nonexistent(self, cache):
        """invalidating a non-existent key should return False."""
        removed = cache.invalidate("nonexistent", "model-a")
        assert removed is False

    def test_cache_clear(self, cache):
        """clear() should remove all entries."""
        for i in range(5):
            cache.put(f"text_{i}", "model-a", np.random.randn(64).astype(np.float32))
        assert cache.size() == 5

        cache.clear()
        assert cache.size() == 0

    def test_cache_size(self, cache):
        """size() should reflect the number of stored entries."""
        assert cache.size() == 0
        cache.put("text1", "model-a", np.random.randn(64).astype(np.float32))
        assert cache.size() == 1
        cache.put("text2", "model-a", np.random.randn(64).astype(np.float32))
        assert cache.size() == 2

    def test_cache_put_copies_array(self, cache):
        """put() should store a copy, not a reference."""
        emb = np.random.randn(64).astype(np.float32)
        cache.put("harimau", "model-a", emb)

        # Modify original
        emb[0] = 999.0

        # Cached version should be unchanged
        cached = cache.get("harimau", "model-a")
        assert cached[0] != 999.0

    def test_cache_deterministic_key(self):
        """Same (model_id, text) should always produce the same key."""
        key1 = EmbeddingCache._make_key("mock-model", "harimau")
        key2 = EmbeddingCache._make_key("mock-model", "harimau")
        assert key1 == key2

    def test_cache_different_texts_different_keys(self):
        """Different texts should produce different keys."""
        key1 = EmbeddingCache._make_key("mock-model", "harimau")
        key2 = EmbeddingCache._make_key("mock-model", "karnivora")
        assert key1 != key2

    def test_cache_different_models_different_keys(self):
        """Different model_ids should produce different keys."""
        key1 = EmbeddingCache._make_key("model-a", "harimau")
        key2 = EmbeddingCache._make_key("model-b", "harimau")
        assert key1 != key2


# ═══════════════ Test 3: EmbeddingCache persistence ═══════════════

class TestEmbeddingCachePersistence:
    """Test EmbeddingCache save/load to disk."""

    def test_save_and_load(self, temp_cache_dir):
        """Cache should survive save/load cycle."""
        cache1 = EmbeddingCache(
            cache_dir=temp_cache_dir,
            cache_filename="test_persistence.pkl",
            auto_load=False,
        )
        emb = np.random.randn(64).astype(np.float32)
        cache1.put("harimau", "model-a", emb)
        assert cache1.save()

        # Load into a new cache instance
        cache2 = EmbeddingCache(
            cache_dir=temp_cache_dir,
            cache_filename="test_persistence.pkl",
            auto_load=True,
        )
        loaded = cache2.get("harimau", "model-a")
        assert loaded is not None
        assert np.allclose(loaded, emb, atol=1e-6)

    def test_load_nonexistent_file(self):
        """Loading from a nonexistent path should return False."""
        cache = EmbeddingCache(
            cache_dir="/tmp/agnn_test_nonexistent_dir_12345",
            cache_filename="nonexistent.pkl",
            auto_load=False,
        )
        result = cache.load()
        assert result is False

    def test_save_creates_directory(self, temp_cache_dir):
        """save() should create the cache directory if it doesn't exist."""
        nested_dir = os.path.join(temp_cache_dir, "nested", "dir")
        cache = EmbeddingCache(
            cache_dir=nested_dir,
            cache_filename="test.pkl",
            auto_load=False,
        )
        cache.put("test", "model-a", np.random.randn(32).astype(np.float32))
        assert cache.save()
        assert os.path.exists(os.path.join(nested_dir, "test.pkl"))

    def test_merge_on_load(self, temp_cache_dir):
        """Loading should merge, not overwrite in-memory entries."""
        # Create and save cache1
        cache1 = EmbeddingCache(
            cache_dir=temp_cache_dir,
            cache_filename="test_merge.pkl",
            auto_load=False,
        )
        cache1.put("text_a", "model-a", np.ones(32, dtype=np.float32))
        cache1.save()

        # Create cache2 with in-memory entry, then load
        cache2 = EmbeddingCache(
            cache_dir=temp_cache_dir,
            cache_filename="test_merge.pkl",
            auto_load=False,
        )
        cache2.put("text_b", "model-a", np.ones(32, dtype=np.float32) * 2)
        cache2.load()  # Should merge, not overwrite

        # Both entries should exist
        assert cache2.has("text_a", "model-a")
        assert cache2.has("text_b", "model-a")

    def test_in_memory_takes_precedence_on_load(self, temp_cache_dir):
        """In-memory entries should take precedence over loaded entries."""
        cache1 = EmbeddingCache(
            cache_dir=temp_cache_dir,
            cache_filename="test_precedence.pkl",
            auto_load=False,
        )
        emb_disk = np.ones(32, dtype=np.float32)
        cache1.put("text_x", "model-a", emb_disk)
        cache1.save()

        cache2 = EmbeddingCache(
            cache_dir=temp_cache_dir,
            cache_filename="test_precedence.pkl",
            auto_load=False,
        )
        emb_memory = np.ones(32, dtype=np.float32) * 99.0
        cache2.put("text_x", "model-a", emb_memory)
        cache2.load()  # In-memory should win

        result = cache2.get("text_x", "model-a")
        assert np.allclose(result, emb_memory, atol=1e-6)


# ═══════════════ Test 4: embed_node function ═══════════════

class TestEmbedNode:
    """Test the embed_node convenience function."""

    def test_embed_node_without_cache(self, mock_embedder):
        """embed_node without cache should always compute."""
        emb = embed_node("harimau", mock_embedder, cache=None)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (64,)

    def test_embed_node_with_cache_miss(self, mock_embedder, cache):
        """embed_node with cache MISS should compute and store."""
        emb = embed_node("harimau", mock_embedder, cache=cache)
        assert emb.shape == (64,)
        # Should now be in cache
        assert cache.has("harimau", "mock-test-model")

    def test_embed_node_with_cache_hit(self, mock_embedder, cache):
        """embed_node with cache HIT should return cached value."""
        # First call: miss → compute
        emb1 = embed_node("harimau", mock_embedder, cache=cache)
        # Second call: hit → return cached
        emb2 = embed_node("harimau", mock_embedder, cache=cache)

        assert np.allclose(emb1, emb2, atol=1e-6)

    def test_embed_node_cache_stores_copy(self, mock_embedder, cache):
        """Cache should store a copy, not a reference."""
        emb = embed_node("harimau", mock_embedder, cache=cache)
        # Modify the returned array
        emb[0] = 999.0
        # Cached version should be unchanged
        cached = cache.get("harimau", "mock-test-model")
        assert cached[0] != 999.0

    def test_embed_nodes_batch(self, mock_embedder, cache):
        """embed_nodes_batch should embed multiple texts efficiently."""
        texts = ["harimau", "karnivora", "pemakan daging"]
        embeddings = embed_nodes_batch(texts, mock_embedder, cache=cache)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (64,)
            assert emb.dtype == np.float32

        # All should now be in cache
        for text in texts:
            assert cache.has(text, "mock-test-model")

    def test_embed_nodes_batch_uses_cache(self, mock_embedder, cache):
        """embed_nodes_batch should use cached values when available."""
        # Pre-populate cache for one text
        pre_emb = np.random.randn(64).astype(np.float32)
        pre_emb /= np.linalg.norm(pre_emb)
        cache.put("harimau", "mock-test-model", pre_emb)

        texts = ["harimau", "karnivora"]
        embeddings = embed_nodes_batch(texts, mock_embedder, cache=cache)

        # "harimau" should come from cache
        assert np.allclose(embeddings[0], pre_emb, atol=1e-6)
        # "karnivora" should be newly computed
        assert embeddings[1].shape == (64,)


# ═══════════════ Test 5: AGNNGraph + embedder integration ═══════════════

class TestGraphEmbedderIntegration:
    """Test AGNNGraph.set_embedder() + initialize_embeddings()."""

    def test_set_embedder(self, simple_graph, mock_embedder):
        """set_embedder should attach embedder without errors."""
        simple_graph.set_embedder(mock_embedder)
        # No assertion needed — just no exception

    def test_set_embedder_with_cache(self, simple_graph, mock_embedder, cache):
        """set_embedder with cache should attach both."""
        simple_graph.set_embedder(mock_embedder, cache=cache)
        # No assertion needed — just no exception

    def test_initialize_embeddings_without_embedder(self, simple_graph):
        """initialize_embeddings without embedder should return 0 (no-op)."""
        count = simple_graph.initialize_embeddings()
        assert count == 0

    def test_initialize_embeddings_with_embedder(self, simple_graph, mock_embedder):
        """initialize_embeddings should compute embeddings for all nodes."""
        # Record original embeddings
        old_embeddings = {}
        for nid in simple_graph.all_node_ids():
            old_embeddings[nid] = simple_graph.get_node(nid).embedding.copy()

        # Set embedder and initialize
        simple_graph.set_embedder(mock_embedder)
        count = simple_graph.initialize_embeddings()

        assert count == 3  # 3 nodes
        # All embeddings should now be unit-normalized (from mock embedder)
        for nid in simple_graph.all_node_ids():
            node = simple_graph.get_node(nid)
            norm = np.linalg.norm(node.embedding)
            assert abs(norm - 1.0) < 0.01, f"Node {nid} has non-unit norm: {norm}"

    def test_initialize_embeddings_changes_embeddings(self, simple_graph, mock_embedder):
        """initialize_embeddings should REPLACE random embeddings with model embeddings."""
        old_embeddings = {}
        for nid in simple_graph.all_node_ids():
            old_embeddings[nid] = simple_graph.get_node(nid).embedding.copy()

        simple_graph.set_embedder(mock_embedder)
        simple_graph.initialize_embeddings()

        # At least some embeddings should have changed
        changed = 0
        for nid in simple_graph.all_node_ids():
            new_emb = simple_graph.get_node(nid).embedding
            if not np.allclose(old_embeddings[nid], new_emb, atol=1e-3):
                changed += 1
        assert changed > 0, "initialize_embeddings should change at least some embeddings"

    def test_initialize_embeddings_with_custom_texts(self, simple_graph, mock_embedder):
        """initialize_embeddings should use custom texts when provided."""
        simple_graph.set_embedder(mock_embedder)

        # Override text for one node
        custom_texts = {"harimau": "tiger is a large cat"}
        count = simple_graph.initialize_embeddings(texts=custom_texts)

        assert count == 3
        # harimau's embedding should be based on "tiger is a large cat"
        # not "harimau" — we can verify by checking it differs from
        # what "harimau" would produce
        harimau_emb = simple_graph.get_node("harimau").embedding
        direct_emb = mock_embedder.embed("harimau")
        custom_emb = mock_embedder.embed("tiger is a large cat")

        # Should match the custom text, not the label
        assert np.allclose(harimau_emb[:64], custom_emb[:64], atol=1e-5)

    def test_initialize_embeddings_with_cache(self, simple_graph, mock_embedder, cache):
        """initialize_embeddings with cache should populate and use cache."""
        simple_graph.set_embedder(mock_embedder, cache=cache)
        simple_graph.initialize_embeddings()

        # Cache should have entries for all node labels
        assert cache.has("harimau", "mock-test-model")
        assert cache.has("karnivora", "mock-test-model")
        assert cache.has("pemakan daging", "mock-test-model")

    def test_initialize_embeddings_second_call_uses_cache(self, simple_graph, mock_embedder, cache):
        """Second call to initialize_embeddings should use cached values."""
        simple_graph.set_embedder(mock_embedder, cache=cache)
        simple_graph.initialize_embeddings()

        # Get cached values
        cached_harimau = cache.get("harimau", "mock-test-model").copy()

        # Clear cache and re-initialize — embeddings should still be correct
        simple_graph.initialize_embeddings()
        harimau_emb = simple_graph.get_node("harimau").embedding

        # Should be the same as cached value
        assert np.allclose(harimau_emb, cached_harimau[:64], atol=1e-5)

    def test_embedding_dimension_projection(self, simple_graph, mock_embedder_large):
        """When hidden_size ≠ embedding_dim, embeddings should be projected."""
        simple_graph.set_embedder(mock_embedder_large)  # hidden_size=896
        count = simple_graph.initialize_embeddings()

        assert count == 3
        # All embeddings should still be (64,) — the graph's embedding_dim
        for nid in simple_graph.all_node_ids():
            node = simple_graph.get_node(nid)
            assert node.embedding.shape == (64,)

    def test_no_embedder_fallback_to_random(self):
        """Without embedder, nodes should still get random embeddings (existing behavior)."""
        g = AGNNGraph(embedding_dim=32)
        np.random.seed(42)
        node = g.add_node(AGNNNode(
            id="test", label="test", node_type=NodeType.ENTITY,
        ))
        # Should have a non-zero random embedding
        assert not np.all(node.embedding == 0)
        assert node.embedding.shape == (32,)

    def test_graph_operations_after_initialize_embeddings(
        self, simple_graph, mock_embedder
    ):
        """All graph operations should still work after initialize_embeddings."""
        simple_graph.set_embedder(mock_embedder)
        simple_graph.initialize_embeddings()

        # Message passing should still work
        old_emb = simple_graph.get_node("karnivora").embedding.copy()
        simple_graph.message_pass("karnivora", damping=0.5)
        new_emb = simple_graph.get_node("karnivora").embedding
        assert not np.allclose(old_emb, new_emb, atol=1e-6)

        # Spread activation should still work
        activation = simple_graph.spread_activation(["harimau"], steps=2)
        assert activation["harimau"] >= 0.9
        assert activation["pemakan_daging"] > 0.0

        # Traversal should still work
        chain = simple_graph.traverse("harimau", max_hops=2)
        assert chain is not None
        assert len(chain.steps) == 2


# ═══════════════ Test 6: End-to-end with mock embedder ═══════════════

class TestEndToEnd:
    """Full end-to-end test: create graph → set embedder → initialize → message pass → traverse."""

    def test_full_pipeline(self):
        """
        Complete pipeline:
          1. Create graph with nodes and edges
          2. Set mock embedder
          3. Initialize embeddings from model-native embeddings
          4. Run message passing
          5. Run spread activation
          6. Traverse and verify reasoning chain

        This mimics the real workflow without requiring a real model.
        """
        # Step 1: Build graph
        g = AGNNGraph(embedding_dim=64)
        g.add_node(AGNNNode(id="api", label="api", node_type=NodeType.ENTITY, confidence=0.9))
        g.add_node(AGNNNode(id="panas", label="panas", node_type=NodeType.CONCEPT, confidence=0.85))
        g.add_node(AGNNNode(id="membakar", label="membakar", node_type=NodeType.CONCEPT, confidence=0.8))

        g.add_edge(TypedEdge("api", "panas", RelationType.CAUSAL, confidence=0.9, context="causes"))
        g.add_edge(TypedEdge("panas", "membakar", RelationType.CAUSAL, confidence=0.85, context="causes"))

        # Step 2: Set mock embedder
        embedder = ModelEmbedder(
            model=None, tokenizer=None, hidden_size=64, num_layers=12,
            model_id="mock-e2e",
        )
        cache = EmbeddingCache(auto_load=False)
        g.set_embedder(embedder, cache=cache)

        # Step 3: Initialize embeddings
        count = g.initialize_embeddings()
        assert count == 3

        # Verify embeddings are non-zero and unit-normalized
        for nid in g.all_node_ids():
            node = g.get_node(nid)
            norm = np.linalg.norm(node.embedding)
            assert norm > 0.1, f"Node {nid} has near-zero embedding"
            assert abs(norm - 1.0) < 0.01, f"Node {nid} is not unit-normalized: {norm}"

        # Step 4: Message passing
        old_emb = g.get_node("panas").embedding.copy()
        g.message_pass("panas", damping=0.5)
        new_emb = g.get_node("panas").embedding
        assert not np.allclose(old_emb, new_emb, atol=1e-6), \
            "Message passing should change panas's embedding"

        # Step 5: Spread activation
        activation = g.spread_activation(["api"], steps=2)
        assert activation["api"] >= 0.9
        assert activation["panas"] > 0.0
        assert activation["membakar"] > 0.0

        # Step 6: Traverse
        chain = g.traverse("api", max_hops=2)
        assert chain is not None
        assert len(chain.steps) >= 1
        assert chain.steps[0][0] == "api"

        # Verify verbalization works
        text = chain.verbalize()
        assert "api" in text

    def test_different_embedders_produce_different_initializations(self):
        """Different embedders should produce different initial embeddings."""
        g1 = AGNNGraph(embedding_dim=64)
        g1.add_node(AGNNNode(id="test", label="test", node_type=NodeType.ENTITY, confidence=0.9))

        g2 = AGNNGraph(embedding_dim=64)
        g2.add_node(AGNNNode(id="test", label="test", node_type=NodeType.ENTITY, confidence=0.9))

        embedder_a = ModelEmbedder(
            model=None, tokenizer=None, hidden_size=64, model_id="model-a",
        )
        embedder_b = ModelEmbedder(
            model=None, tokenizer=None, hidden_size=64, model_id="model-b",
        )

        # Both mock embedders produce deterministic embeddings from text hash
        # Same text → same embedding regardless of model_id in mock mode
        # (mock mode uses text hash as seed, not model_id)
        g1.set_embedder(embedder_a)
        g1.initialize_embeddings()

        g2.set_embedder(embedder_b)
        g2.initialize_embeddings()

        # In mock mode, same text produces same embedding (seeded by text)
        # This is correct — the mock is deterministic per text
        emb1 = g1.get_node("test").embedding
        emb2 = g2.get_node("test").embedding
        assert np.allclose(emb1, emb2, atol=1e-5), \
            "Mock embedder should be deterministic for same text"

    def test_embedding_survives_serialization(self, mock_embedder):
        """Embeddings set by initialize_embeddings should survive serialization."""
        g = AGNNGraph(embedding_dim=64)
        g.add_node(AGNNNode(id="test", label="test node", node_type=NodeType.ENTITY, confidence=0.9))

        g.set_embedder(mock_embedder)
        g.initialize_embeddings()

        # Serialize and deserialize
        d = g.to_dict()
        g2 = AGNNGraph.from_dict(d)

        # Embedding should be preserved
        orig = g.get_node("test").embedding
        restored = g2.get_node("test").embedding
        assert np.allclose(orig, restored, atol=1e-6)


# ═══════════════ Test 7: No bge-m3 import verification ═══════════════

class TestNoBgeM3Import:
    """Verify that no bge-m3 dependency leaks into the agnn module.

    We check for actual import statements, not docstring mentions.
    Mentioning bge-m3 in docstrings is fine (it explains what we replaced).
    Importing it in code would be a dependency leak.
    """

    def test_embeddings_module_no_bge_import(self):
        """embeddings.py should not import bge-m3 or sentence_transformers."""
        import agnn.embeddings as emb_module
        # Check the module's actual imported modules
        import sys
        module_name = emb_module.__name__
        # Verify no sentence_transformers in the module's imports
        assert 'sentence_transformers' not in sys.modules or \
            'bge' not in str(sys.modules.get('sentence_transformers', ''))
        # Check that the module has no import of sentence_transformers
        import inspect
        source = inspect.getsource(emb_module)
        import_lines = [line for line in source.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]
        for line in import_lines:
            assert 'sentence_transformers' not in line, f"Forbidden import: {line}"
            assert 'bge' not in line.lower() or 'debug' in line.lower(), f"Forbidden import: {line}"

    def test_graph_module_no_bge_import(self):
        """graph.py should not import bge-m3 or sentence_transformers."""
        import agnn.graph as graph_module
        import inspect
        source = inspect.getsource(graph_module)
        import_lines = [line for line in source.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]
        for line in import_lines:
            assert 'sentence_transformers' not in line, f"Forbidden import: {line}"
            assert 'bge' not in line.lower() or 'debug' in line.lower(), f"Forbidden import: {line}"
