#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_pipeline_integration.py
# @WHAT:  Integration test — full pipeline (learn → reinforce → penalize → introspect)
# @PART:  self-ai/tests
# @ENTRY: python -m pytest self-ai/tests/test_pipeline_integration.py -v

"""Integration test: full SELF pipeline without model loading.

All 6 tests exercise the complete pipeline end-to-end:
  learn() → UnderstandingGraph → reinforce()/penalize() → introspect()

No Qwen3, no bge-m3, no GPU, no internet. All embedding calls are mocked
with deterministic numpy arrays so retrieval still works via cosine similarity.

Each test is fully independent — no shared state, fresh graph per test.

Run:
  cd self-ai/src
  python -m pytest ../tests/test_pipeline_integration.py -v
"""

import os
import sys
import uuid
import logging
from unittest.mock import patch, MagicMock

import numpy as np

# ─── PATH SETUP ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

EMBEDDING_DIM = 64  # Small dimension for test speed (bge-m3 uses 1024)


# ═══════════════════════════════════════════════════════════════════
#  Deterministic mock embedding function
# ═══════════════════════════════════════════════════════════════════

def _text_to_embedding(text: str) -> np.ndarray:
    """Deterministic mock embedding with high inter-text similarity.

    In the real system, bge-m3 produces semantically similar vectors
    for related texts (e.g., a question and its correction experience).
    Our mock can't do that, so we use a shared base vector with a small
    text-specific perturbation. This ensures:

      - Same text → identical embedding (deterministic)
      - Different texts → high cosine similarity (>0.9) so retrieval works
      - Small differences preserve ordering for multi-node scenarios

    This is intentional: we're testing the PIPELINE wiring, not the
    quality of semantic similarity (that's bge-m3's job).
    """
    # Shared base vector (unit) — gives all embeddings high mutual similarity
    base = np.ones(EMBEDDING_DIM, dtype=np.float32)
    base = base / np.linalg.norm(base)

    # Small text-specific perturbation — makes embeddings deterministic
    # but different enough to test multi-node ranking
    seed = abs(hash(text)) % (2**31)
    rng = np.random.RandomState(seed)
    perturbation = rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.05

    emb = base + perturbation
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb


def _mock_embed_function(text: str) -> list:
    """Generate a mock embedding as a Python list (for UnderstandingNode.condition_embedding)."""
    return _text_to_embedding(text).tolist()


# ═══════════════════════════════════════════════════════════════════
#  MockRetriever — drop-in replacement for UnderstandingRetriever
# ═══════════════════════════════════════════════════════════════════

class MockRetriever:
    """Deterministic retriever that uses mock numpy embeddings.

    Implements the same interface as UnderstandingRetriever so that
    graph.retrieve() can call it end-to-end without patching.

    Key attributes that graph.retrieve() / _second_hop_retrieve() rely on:
      - self._embeddings: dict[node_id, np.ndarray] — used by retrieve()
        for cosine similarity AND by _second_hop_retrieve() for neighbor lookup
      - self.retrieve(text, question, nodes, top_k, threshold)
      - self.find_best(text, question, nodes, threshold)
    """

    def __init__(self):
        self._embeddings = {}  # node_id → np.ndarray (normalized)

    def index_node(self, node):
        """Index a node's embedding for retrieval.

        Called manually after learn() to keep _embeddings in sync.
        """
        if node.condition_embedding is not None:
            self._embeddings[node.id] = np.array(
                node.condition_embedding, dtype=np.float32
            )

    def load_from_graph(self, graph):
        """Load all existing nodes from a graph (mirrors UnderstandingRetriever.load_from_graph)."""
        for nid, node in graph._nodes.items():
            self.index_node(node)

    def is_available(self) -> bool:
        """Always available — no real model needed."""
        return True

    def retrieve(self, text: str, question: str,
                 nodes: dict, top_k: int = 5,
                 threshold: float = 0.25) -> list:
        """Find matching nodes by cosine similarity on mock embeddings.

        This mirrors UnderstandingRetriever.retrieve() but uses
        deterministic hash-based embeddings instead of bge-m3.
        """
        if not self._embeddings:
            return []

        # Encode query using the same deterministic embedding function
        query_text = question  # Simplified: use question as query text
        query_emb = _text_to_embedding(query_text)

        scored = []
        for node_id, node_emb in self._embeddings.items():
            # Both vectors are already normalized
            sim = float(np.dot(query_emb, node_emb))
            if sim >= threshold:
                node = nodes.get(node_id)
                if node is not None:
                    scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def find_best(self, text: str, question: str,
                  nodes: dict, threshold: float = 0.25):
        """Find the single best matching node."""
        results = self.retrieve(text, question, nodes, top_k=1, threshold=threshold)
        if results:
            return results[0]
        return None


# ═══════════════════════════════════════════════════════════════════
#  Fixtures — isolated UnderstandingGraph with mock embeddings
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_shared_graph():
    """Reset the shared graph singleton between tests."""
    import derivation.understanding_builder as ub_module
    ub_module._shared_graph = None
    yield
    ub_module._shared_graph = None


def _make_graph():
    """Create a fresh UnderstandingGraph with an isolated store path."""
    from derivation.understanding_builder import UnderstandingGraph
    store_path = f"/tmp/test_pipeline_{uuid.uuid4().hex}.json"
    graph = UnderstandingGraph(store_path=store_path)
    return graph


def _inject_mock_retriever(graph):
    """Inject a MockRetriever into the graph.

    This replaces the real UnderstandingRetriever so that graph.retrieve()
    works end-to-end using deterministic hash-based embeddings instead of
    bge-m3. This is what makes this an *integration* test: the full
    retrieval pipeline is exercised, just with fake embeddings.
    """
    mock_retriever = MockRetriever()
    # Load any nodes that were already in the graph
    mock_retriever.load_from_graph(graph)
    graph._retriever = mock_retriever
    graph._retriever_initialized = True
    return mock_retriever


def _setup_pipeline():
    """Set up the full pipeline: fresh graph + mock retriever + SelfCore.

    Returns:
        (SelfCore instance, UnderstandingGraph instance, MockRetriever)
    """
    from core.self import SelfCore
    import derivation.understanding_builder as ub_module

    graph = _make_graph()
    ub_module._shared_graph = graph
    mock_retriever = _inject_mock_retriever(graph)

    self_instance = SelfCore()
    # Cache the graph on the instance so introspect() finds it directly
    self_instance._graph = graph

    return self_instance, graph, mock_retriever


def _learn_with_mock_embedding(self_instance, graph, mock_retriever,
                                question, wrong_answer, correction):
    """Call learn() with mock embedding so the node gets a retrievable embedding.

    learn() internally tries to call get_shared_embedding_model().encode(),
    which we mock to return a deterministic numpy array. After learn(),
    we also index the node in the MockRetriever so retrieve() can find it.
    """
    mock_model = MagicMock()
    mock_model.encode = lambda texts, show_progress_bar=False, normalize_embeddings=True: \
        np.array([_mock_embed_function(t) for t in texts])

    with patch('derivation.model_registry.get_shared_embedding_model', return_value=mock_model):
        result = self_instance.learn(
            question=question, wrong_answer=wrong_answer, correction=correction
        )

    # Keep the MockRetriever's _embeddings in sync with the graph
    if result['node_id'] is not None and not result.get('duplicate', False):
        node = graph.get_node(result['node_id'])
        if node is not None:
            mock_retriever.index_node(node)

    return result


# ═══════════════════════════════════════════════════════════════════
#  Test 1: learn() → node stored → graph.retrieve() returns it
# ═══════════════════════════════════════════════════════════════════

def test_learn_stores_node_retrievable():
    """learn() → node masuk graph → graph.retrieve() dengan mock embeddings returns node itu.

    This is the foundational integration test: after learn(), the node
    must be stored in the graph AND retrievable via the full retrieval
    pipeline (not just by ID). We mock embeddings so cosine similarity
    retrieval works end-to-end.
    """
    self_instance, graph, mock_retriever = _setup_pipeline()

    # Step 1: learn a correction (with mock embedding)
    result = _learn_with_mock_embedding(
        self_instance, graph, mock_retriever,
        question="Siapa presiden Indonesia pertama?",
        wrong_answer="Jokowi",
        correction="Sukarno",
    )

    assert result['node_id'] is not None
    assert result['duplicate'] is False
    node_id = result['node_id']

    # Step 2: Verify the node is in the graph
    node = graph.get_node(node_id)
    assert node is not None
    assert 'Sukarno' in node.abstraction
    assert node.condition_embedding is not None  # Mock embedding was stored

    # Step 3: Retrieve via the full retrieval pipeline (cosine similarity)
    retrieved = graph.retrieve(
        "Siapa presiden Indonesia pertama?",
        "Siapa presiden Indonesia pertama?",
        top_k=5,
        threshold=0.1,
        enable_cross_domain=False,
    )

    # The learned node must be among the retrieved results
    retrieved_ids = [n.id for n, score in retrieved]
    assert node_id in retrieved_ids, (
        f"Learned node {node_id} not found in retrieve() results: {retrieved_ids}"
    )


# ═══════════════════════════════════════════════════════════════════
#  Test 2: learn() → reinforce() increases confidence
# ═══════════════════════════════════════════════════════════════════

def test_reinforce_after_learn_increases_confidence():
    """learn() → confidence=0.6 → reinforce() → confidence=0.68.

    After learning a correction, reinforcing it should increase the
    node's confidence by 0.08 (from 0.6 to 0.68).
    """
    self_instance, graph, mock_retriever = _setup_pipeline()

    # Step 1: learn
    learn_result = _learn_with_mock_embedding(
        self_instance, graph, mock_retriever,
        question="Apa ibukota Jepang?",
        wrong_answer="Osaka",
        correction="Tokyo",
    )
    node_id = learn_result['node_id']
    assert learn_result['confidence'] == 0.6

    # Step 2: reinforce — retrieve will find the node via mock embeddings
    reinforce_result = self_instance.reinforce(
        question="Apa ibukota Jepang?",
        confirmed_answer="Tokyo",
    )

    # Step 3: Verify confidence increased
    assert reinforce_result['reinforced_count'] >= 1
    assert node_id in reinforce_result['node_ids']
    new_confidence = reinforce_result['new_confidences'][node_id]
    assert new_confidence == pytest.approx(0.68, abs=0.01), (
        f"Expected confidence 0.68, got {new_confidence}"
    )

    # Step 4: Double-check in the graph itself
    node = graph.get_node(node_id)
    assert node.confidence == pytest.approx(0.68, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
#  Test 3: learn() → penalize() decreases confidence
# ═══════════════════════════════════════════════════════════════════

def test_penalize_after_learn_decreases_confidence():
    """learn() → confidence=0.6 → penalize() → confidence=0.5.

    After learning a correction, penalizing it should decrease the
    node's confidence by 0.1 (from 0.6 to 0.5).
    """
    self_instance, graph, mock_retriever = _setup_pipeline()

    # Step 1: learn
    learn_result = _learn_with_mock_embedding(
        self_instance, graph, mock_retriever,
        question="Berapa hasil 5 + 3?",
        wrong_answer="7",
        correction="8",
    )
    node_id = learn_result['node_id']
    assert learn_result['confidence'] == 0.6

    # Step 2: penalize — the wrong_answer "7" appears in the node's abstraction
    penalize_result = self_instance.penalize(
        question="Berapa hasil 5 + 3?",
        wrong_answer="7",
    )

    # Step 3: Verify confidence decreased
    assert penalize_result['penalized_count'] >= 1
    assert node_id in penalize_result['node_ids']
    new_confidence = penalize_result['new_confidences'][node_id]
    assert new_confidence == pytest.approx(0.5, abs=0.01), (
        f"Expected confidence 0.5, got {new_confidence}"
    )

    # Step 4: Double-check in the graph itself
    node = graph.get_node(node_id)
    assert node.confidence == pytest.approx(0.5, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
#  Test 4: learn() → introspect() reflects the new node
# ═══════════════════════════════════════════════════════════════════

def test_introspect_after_learn_reflects_new_node():
    """learn() → introspect() → graph_size=1, sources={"user_correction": 1}.

    After learning one correction, introspect() must reflect exactly
    one node in the graph, with the source correctly counted.
    """
    self_instance, graph, mock_retriever = _setup_pipeline()

    # Step 1: introspect before learning — should be empty
    before = self_instance.introspect()
    assert before['graph_size'] == 0
    assert before['status'] == 'empty'

    # Step 2: learn a correction
    learn_result = _learn_with_mock_embedding(
        self_instance, graph, mock_retriever,
        question="Warna langit saat cerah?",
        wrong_answer="Merah",
        correction="Biru",
    )
    assert learn_result['graph_size'] == 1

    # Step 3: introspect after learning
    after = self_instance.introspect()
    assert after['graph_size'] == 1
    assert after['sources'].get('user_correction') == 1
    assert after['status'] == 'small'  # 1 < 10

    # The learned node should appear in top_nodes
    top_ids = [n['id'] for n in after['top_nodes']]
    assert learn_result['node_id'] in top_ids


# ═══════════════════════════════════════════════════════════════════
#  Test 5: learn → reinforce → penalize chain
# ═══════════════════════════════════════════════════════════════════

def test_learn_reinforce_penalize_chain():
    """learn → reinforce → penalize → final confidence = 0.6 + 0.08 - 0.1 = 0.58.

    This tests the full feedback chain: a node is created at 0.6,
    reinforced by +0.08, then penalized by -0.1. The final
    confidence should be 0.58.
    """
    self_instance, graph, mock_retriever = _setup_pipeline()

    # Step 1: learn
    learn_result = _learn_with_mock_embedding(
        self_instance, graph, mock_retriever,
        question="Apa bahasa resmi Brasil?",
        wrong_answer="Spanyol",
        correction="Portugis",
    )
    node_id = learn_result['node_id']
    assert learn_result['confidence'] == 0.6

    # Step 2: reinforce
    reinforce_result = self_instance.reinforce(
        question="Apa bahasa resmi Brasil?",
        confirmed_answer="Portugis",
    )
    assert reinforce_result['reinforced_count'] >= 1
    assert node_id in reinforce_result['node_ids']
    after_reinforce = reinforce_result['new_confidences'][node_id]
    assert after_reinforce == pytest.approx(0.68, abs=0.01)

    # Step 3: penalize — penalize "Spanyol" which appears in the node's abstraction
    penalize_result = self_instance.penalize(
        question="Apa bahasa resmi Brasil?",
        wrong_answer="Spanyol",
    )
    assert penalize_result['penalized_count'] >= 1
    assert node_id in penalize_result['node_ids']
    after_penalize = penalize_result['new_confidences'][node_id]

    # Final: 0.6 + 0.08 - 0.1 = 0.58
    assert after_penalize == pytest.approx(0.58, abs=0.01), (
        f"Expected confidence 0.58, got {after_penalize}"
    )

    # Double-check in the graph
    node = graph.get_node(node_id)
    assert node.confidence == pytest.approx(0.58, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
#  Test 6: duplicate learn() is idempotent
# ═══════════════════════════════════════════════════════════════════

def test_duplicate_learn_idempotent():
    """learn() dua kali dengan pertanyaan sama → graph_size tetap 1, duplicate=True pada call kedua.

    Calling learn() twice with identical (question, wrong_answer, correction)
    must not create duplicate nodes. The second call returns duplicate=True
    and the graph size stays at 1.
    """
    self_instance, graph, mock_retriever = _setup_pipeline()

    # Step 1: First learn
    result1 = _learn_with_mock_embedding(
        self_instance, graph, mock_retriever,
        question="Siapa penemu lampu pijar?",
        wrong_answer="Newton",
        correction="Edison",
    )
    assert result1['duplicate'] is False
    assert result1['graph_size'] == 1
    node_id_1 = result1['node_id']

    # Step 2: Second learn with identical input
    result2 = _learn_with_mock_embedding(
        self_instance, graph, mock_retriever,
        question="Siapa penemu lampu pijar?",
        wrong_answer="Newton",
        correction="Edison",
    )
    assert result2['duplicate'] is True
    assert result2['graph_size'] == 1
    assert result2['node_id'] == node_id_1  # Same node returned

    # Step 3: Graph still has exactly 1 node
    assert graph.count() == 1
