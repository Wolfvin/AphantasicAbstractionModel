#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_selfcore_agnn.py
# @WHAT:  Test SelfCore + AGNNGraph integration (feat/selfcore-agnn-integration)
# @PART:  self-ai/tests
# @ENTRY: python -m pytest self-ai/tests/test_selfcore_agnn.py -v

"""Test: SelfCore + AGNNGraph additive integration.

All tests run without GPU and without a real model. SelfCore instances
are created with pre-populated graphs where needed. The shared graph
singleton is reset between tests.

Test matrix:
  1. learn()  → node appears in BOTH UnderstandingGraph and AGNNGraph
  2. reinforce() / penalize() → confidence changes in AGNNGraph
  3. agnn_traverse() → returns non-empty string
  4. introspect() → response has agnn_size field
  5. Edge inference → CATEGORICAL edges created on keyword overlap
  6. Graceful degradation → SelfCore works without AGNN
  7. process() regression → existing behaviour not broken

Run:
  cd self-ai/src
  python -m pytest ../tests/test_selfcore_agnn.py -v
"""

import os
import sys
import tempfile
import logging
from unittest.mock import patch, MagicMock

# ─── PATH SETUP ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_shared_graph():
    """Reset the shared graph singleton between tests."""
    import derivation.understanding_builder as ub_module
    ub_module._shared_graph = None
    yield
    ub_module._shared_graph = None


def _make_graph():
    """Create a fresh UnderstandingGraph in a temp directory."""
    from derivation.understanding_builder import UnderstandingGraph
    tmpdir = tempfile.mkdtemp()
    return UnderstandingGraph(store_path=os.path.join(tmpdir, 'test_graph.json'))


def _add_node_to_graph(graph, node_id, abstraction, concept, confidence=0.6, source='test'):
    """Helper: add an UnderstandingNode directly to the graph."""
    from derivation.understanding_builder import UnderstandingNode
    node = UnderstandingNode(
        id=node_id,
        name=f"Test node {node_id}",
        concept=concept,
        abstraction=abstraction,
        conditions=[concept.lower()],
        condition_embedding=None,
        source=source,
        confidence=confidence,
        lifecycle='new',
    )
    graph.add_node(node)
    return node


def _make_core_with_data():
    """Create a SelfCore and pre-populate it with a few nodes via learn()."""
    from core.self import SelfCore
    import derivation.understanding_builder as ub_module

    graph = _make_graph()
    ub_module._shared_graph = graph

    core = SelfCore()
    core.learn("machine learning models", "wrong", "correct models answer")
    core.learn("deep learning is a subset of machine learning", "wrong2", "correct deep answer")
    core.learn("neural networks power deep learning systems", "wrong3", "correct neural answer")
    return core


def _add_agnn_node(core, node_id, label, experience="", confidence=0.6):
    """Helper: add an AGNNNode directly to the AGNNGraph."""
    if core._agnn is None:
        pytest.skip("AGNNGraph not available")
    from agnn.graph import AGNNNode, NodeType
    node = AGNNNode(
        id=node_id,
        label=label,
        node_type=NodeType.RULE,
        confidence=confidence,
        metadata={"source": "test", "experience": experience},
    )
    core._agnn.add_node(node)
    return node


# ═══════════════════════════════════════════════════════════════════
#  Test 1: learn() mirrors to both graphs
# ═══════════════════════════════════════════════════════════════════

class TestLearnMirroring:
    """Verify that learn() adds a node to both UnderstandingGraph and AGNNGraph."""

    def test_node_in_understanding_graph(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        result = core.learn("test question", "wrong answer", "correct answer")
        node_id = result['node_id']

        ug_node = graph.get_node(node_id)
        assert ug_node is not None, "Node must exist in UnderstandingGraph"
        assert "correct answer" in ug_node.abstraction

    def test_node_in_agnn_graph(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        result = core.learn("test question", "wrong answer", "correct answer")
        node_id = result['node_id']

        agnn_node = core._agnn.get_node(node_id)
        assert agnn_node is not None, "Node must exist in AGNNGraph"
        assert agnn_node.label == "test question"

    def test_node_id_shared_between_graphs(self):
        """Both graphs must use the same node id."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        result = core.learn("shared id question", "wrong", "correct")
        node_id = result['node_id']

        assert graph.get_node(node_id) is not None
        assert core._agnn.get_node(node_id) is not None

    def test_agnn_confidence_is_06(self):
        """AGNN node should start at confidence=0.6 as per spec."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        result = core.learn("confidence test question", "wrong", "correct")
        node_id = result['node_id']
        agnn_node = core._agnn.get_node(node_id)
        assert agnn_node.confidence == pytest.approx(0.6)


# ═══════════════════════════════════════════════════════════════════
#  Test 2: reinforce() / penalize() change AGNN confidence
# ═══════════════════════════════════════════════════════════════════

class TestReinforcePenalize:
    """Verify that reinforce/penalize propagate to AGNNGraph."""

    def test_reinforce_increases_agnn_confidence(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        # Learn a node, then reinforce it
        learn_result = core.learn("reinforce question", "wrong", "confirmed answer")
        node_id = learn_result['node_id']

        # Add AGNN node manually so it exists (already created by learn())
        before = core._agnn.get_node(node_id).confidence

        # Mock retrieve to return the node, then reinforce
        ug_node = graph.get_node(node_id)
        with patch.object(graph, 'retrieve', return_value=[(ug_node, 0.9)]):
            core.reinforce("reinforce question", "confirmed answer")

        after = core._agnn.get_node(node_id).confidence
        assert after > before, "AGNN confidence should increase after reinforce()"

    def test_penalize_decreases_agnn_confidence(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        learn_result = core.learn("penalize question", "wrong answer", "correct")
        node_id = learn_result['node_id']

        before = core._agnn.get_node(node_id).confidence

        ug_node = graph.get_node(node_id)
        with patch.object(graph, 'retrieve', return_value=[(ug_node, 0.9)]):
            core.penalize("penalize question", "wrong answer")

        after = core._agnn.get_node(node_id).confidence
        assert after < before, "AGNN confidence should decrease after penalize()"

    def test_reinforce_delta_agnn(self):
        """AGNN reinforce uses delta=0.08."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        learn_result = core.learn("delta test question", "wrong", "correct delta")
        node_id = learn_result['node_id']
        before = core._agnn.get_node(node_id).confidence

        ug_node = graph.get_node(node_id)
        with patch.object(graph, 'retrieve', return_value=[(ug_node, 0.9)]):
            core.reinforce("delta test question", "correct delta")

        after = core._agnn.get_node(node_id).confidence
        assert after == pytest.approx(before + 0.08)

    def test_penalize_delta_agnn(self):
        """AGNN penalize uses delta=0.1."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        learn_result = core.learn("pen delta test question", "wrong answer", "correct")
        node_id = learn_result['node_id']
        before = core._agnn.get_node(node_id).confidence

        ug_node = graph.get_node(node_id)
        with patch.object(graph, 'retrieve', return_value=[(ug_node, 0.9)]):
            core.penalize("pen delta test question", "wrong answer")

        after = core._agnn.get_node(node_id).confidence
        assert after == pytest.approx(before - 0.1)

    def test_reinforce_also_updates_understanding_graph(self):
        """Existing UG behaviour must not break."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        learn_result = core.learn("ug reinforce test", "wrong", "correct ug")
        node_id = learn_result['node_id']
        before = graph.get_node(node_id).confidence

        ug_node = graph.get_node(node_id)
        with patch.object(graph, 'retrieve', return_value=[(ug_node, 0.9)]):
            core.reinforce("ug reinforce test", "correct ug")

        after = graph.get_node(node_id).confidence
        assert after > before

    def test_penalize_also_updates_understanding_graph(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        learn_result = core.learn("ug penalize test", "wrong", "correct ug")
        node_id = learn_result['node_id']
        before = graph.get_node(node_id).confidence

        ug_node = graph.get_node(node_id)
        with patch.object(graph, 'retrieve', return_value=[(ug_node, 0.9)]):
            core.penalize("ug penalize test", "wrong")

        after = graph.get_node(node_id).confidence
        assert after < before


# ═══════════════════════════════════════════════════════════════════
#  Test 3: agnn_traverse() returns non-empty string
# ═══════════════════════════════════════════════════════════════════

class TestAgnnTraverse:
    """Verify agnn_traverse() produces a reasoning chain."""

    def test_traverse_returns_string(self):
        core = _make_core_with_data()
        if core._agnn is None:
            pytest.skip("AGNNGraph not available")
        result = core.agnn_traverse("machine learning")
        assert isinstance(result, str), "agnn_traverse must return a string"

    def test_traverse_returns_nonempty_for_matching_query(self):
        core = _make_core_with_data()
        if core._agnn is None:
            pytest.skip("AGNNGraph not available")
        result = core.agnn_traverse("machine learning")
        # With 3 nodes that share keywords, traverse should find something
        assert len(result) > 0, "agnn_traverse should return a non-empty chain for a matching query"

    def test_traverse_empty_graph(self):
        from core.self import SelfCore
        core = SelfCore()
        result = core.agnn_traverse("anything")
        assert result == "", "Traverse on empty graph should return empty string"

    def test_traverse_no_match(self):
        core = _make_core_with_data()
        if core._agnn is None:
            pytest.skip("AGNNGraph not available")
        result = core.agnn_traverse("quantum physics entanglement")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════
#  Test 4: introspect() has agnn_size
# ═══════════════════════════════════════════════════════════════════

class TestIntrospect:
    """Verify introspect() includes AGNN fields."""

    def test_has_agnn_size(self):
        core = _make_core_with_data()
        info = core.introspect()
        assert "agnn_size" in info, "introspect() must include agnn_size"

    def test_has_agnn_avg_confidence(self):
        core = _make_core_with_data()
        info = core.introspect()
        assert "agnn_avg_confidence" in info, "introspect() must include agnn_avg_confidence"

    def test_agnn_size_matches_learn_count(self):
        core = _make_core_with_data()
        info = core.introspect()
        if core._agnn is not None:
            assert info["agnn_size"] == 3, "Three nodes learned"

    def test_existing_fields_preserved(self):
        core = _make_core_with_data()
        info = core.introspect()
        assert "graph_size" in info, "Legacy graph_size field must be preserved"
        assert "avg_confidence" in info, "Legacy avg_confidence field must be preserved"
        assert "top_nodes" in info, "Legacy top_nodes field must be preserved"
        assert "recent_nodes" in info, "Legacy recent_nodes field must be preserved"
        assert "sources" in info, "Legacy sources field must be preserved"
        assert "status" in info, "Legacy status field must be preserved"

    def test_ug_size_matches(self):
        core = _make_core_with_data()
        info = core.introspect()
        assert info["graph_size"] == 3


# ═══════════════════════════════════════════════════════════════════
#  Test 5: Edge inference on keyword overlap
# ═══════════════════════════════════════════════════════════════════

class TestEdgeInference:
    """Verify that CATEGORICAL edges are inferred when nodes share keywords."""

    def test_categorical_edge_created(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module
        from agnn.graph import RelationType

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        core.learn("machine learning algorithms", "wrong1", "correct1")
        core.learn("deep learning techniques", "wrong2", "correct2")

        edges = core._agnn._edges
        categorical_edges = [e for e in edges if e.relation_type == RelationType.CATEGORICAL]
        assert len(categorical_edges) > 0, "Should have CATEGORICAL edges for overlapping keyword 'learning'"

    def test_bidirectional_edges(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        nid1_result = core.learn("machine learning algorithms", "wrong1", "correct1")
        nid2_result = core.learn("deep learning techniques", "wrong2", "correct2")
        nid1 = nid1_result['node_id']
        nid2 = nid2_result['node_id']

        forward = any(e.source_id == nid1 and e.target_id == nid2 for e in core._agnn._edges)
        backward = any(e.source_id == nid2 and e.target_id == nid1 for e in core._agnn._edges)
        assert forward, "Forward CATEGORICAL edge should exist"
        assert backward, "Backward CATEGORICAL edge should exist"

    def test_no_edge_without_overlap(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module
        from agnn.graph import RelationType

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        # Note: learn() generates Indonesian-template experience text that
        # contains common words ("jawaban", "benar", etc.), so the edge
        # inference will still find overlap from those template words.
        # We test with AGNNNode directly to verify pure non-overlap.
        from agnn.graph import AGNNNode, NodeType

        n1 = AGNNNode(id="n1", label="alpha beta", node_type=NodeType.CONCEPT, confidence=0.5)
        n2 = AGNNNode(id="n2", label="gamma delta", node_type=NodeType.CONCEPT, confidence=0.5)
        core._agnn.add_node(n1)
        core._agnn.add_node(n2)

        # Manually call _infer_agnn_edges on the new node
        core._infer_agnn_edges("n2", "gamma delta")

        categorical_edges = [e for e in core._agnn._edges if e.relation_type == RelationType.CATEGORICAL and e.source_id in ("n1", "n2") and e.target_id in ("n1", "n2")]
        assert len(categorical_edges) == 0, "No CATEGORICAL edges expected for non-overlapping labels"

    def test_edge_confidence_is_05(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module
        from agnn.graph import RelationType

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        core.learn("machine learning algorithms", "wrong1", "correct1")
        core.learn("deep learning techniques", "wrong2", "correct2")

        categorical_edges = [e for e in core._agnn._edges if e.relation_type == RelationType.CATEGORICAL]
        for edge in categorical_edges:
            assert edge.confidence == pytest.approx(0.5), "Inferred CATEGORICAL edges should have confidence=0.5"


# ═══════════════════════════════════════════════════════════════════
#  Test 6: Graceful degradation
# ═══════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """SelfCore must work even if AGNN import fails."""

    def test_core_works_without_agnn(self):
        """Simulate AGNN unavailability — SelfCore should still function."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Force _agnn to None after init to simulate AGNN init failure
        core = SelfCore()
        core._agnn = None
        core._agnn_available = False

        result = core.learn("still works without agnn", "wrong", "correct")
        assert result['node_id'] is not None

    def test_agnn_traverse_returns_empty_without_agnn(self):
        from core.self import SelfCore
        core = SelfCore()
        core._agnn = None
        result = core.agnn_traverse("anything")
        assert result == ""

    def test_introspect_without_agnn_still_works(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()
        core._agnn = None

        info = core.introspect()
        # When _agnn is None, introspect should still return agnn_size=0
        assert info.get("agnn_size", 0) == 0
        assert "graph_size" in info, "Legacy fields must still be present"


# ═══════════════════════════════════════════════════════════════════
#  Test 7: process() still works (regression)
# ═══════════════════════════════════════════════════════════════════

class TestProcessRegression:
    """Ensure existing process() behaviour is not broken by AGNN integration."""

    def test_process_returns_result(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()
        result = core.process("test question")
        assert isinstance(result, dict), "process() should return a dict"


# ═══════════════════════════════════════════════════════════════════
#  Test 8: adapt_agnn() with SelfAdapter
# ═══════════════════════════════════════════════════════════════════

class TestAdaptAgnn:
    """Verify adapt_agnn() correctly uses SelfAdapter."""

    def test_adapt_agnn_with_adapter(self):
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph
        core = SelfCore()

        if core._agnn is None:
            pytest.skip("AGNNGraph not available")

        from agnn.adapter import SelfAdapter, ModelProfile

        profile = ModelProfile(
            model_id="test-model",
            hidden_size=256,
            num_layers=12,
            recommended_hook_layer=6,
            architecture="unknown",
        )
        adapter = SelfAdapter(profile=profile)

        original_dim = core._agnn._embedding_dim
        core.adapt_agnn(adapter)
        assert core._agnn._embedding_dim == 256, "AGNN embedding_dim should be adapted to 256"

    def test_adapt_agnn_with_none(self):
        """adapt_agnn(None) should be a no-op, not an error."""
        from core.self import SelfCore
        core = SelfCore()
        core.adapt_agnn(None)  # Should not raise

    def test_adapt_agnn_without_agnn(self):
        """adapt_agnn() with no AGNN should log and return."""
        from core.self import SelfCore
        core = SelfCore()
        core._agnn = None
        core.adapt_agnn(MagicMock())  # Should not raise
