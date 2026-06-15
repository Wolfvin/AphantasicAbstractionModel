#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_introspect.py
# @WHAT:  Test SelfCore.introspect() — SELF's self-reflection capability
# @PART:  self-ai/tests
# @ENTRY: python -m pytest self-ai/tests/test_introspect.py -v

"""Test: Does introspect() return a correct snapshot of SELF's knowledge?

This test verifies Step 5 of the SELF-AI vision:
  1. Empty graph → status="empty", graph_size=0
  2. Correct graph_size after adding nodes
  3. top_nodes sorted by confidence (highest first)
  4. sources dict counts per source correctly
  5. self._graph = None → default dict, no error

The test uses a temporary UnderstandingGraph with a temp store_path
(since bge-m3 is not available in test environments), following the
same pattern as test_reinforce.py.

Run:
  cd self-ai/src
  python -m pytest ../tests/test_introspect.py -v
"""

import os
import sys
import tempfile
import logging

# ─── PATH SETUP ───
# self-ai/src must be on sys.path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures — temporary UnderstandingGraph with mocked retrieval
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_shared_graph():
    """Reset the shared graph singleton between tests."""
    import derivation.understanding_builder as ub_module
    ub_module._shared_graph = None
    yield
    ub_module._shared_graph = None


def _make_graph():
    """Create a fresh UnderstandingGraph in a temp directory.

    Uses a temporary store_path so we don't pollute the project's
    persistent data. The graph has NO embedding model (bge-m3 unavailable).
    """
    from derivation.understanding_builder import UnderstandingGraph
    tmpdir = tempfile.mkdtemp()
    return UnderstandingGraph(store_path=os.path.join(tmpdir, 'test_graph.json'))


def _add_node_to_graph(graph, node_id, abstraction, concept,
                       confidence=0.6, source='test'):
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


# ═══════════════════════════════════════════════════════════════════
#  Test 1: introspect() with empty graph
# ═══════════════════════════════════════════════════════════════════

class TestIntrospectEmptyGraph:
    """introspect() should return status='empty' when graph has no nodes."""

    def test_introspect_empty_graph(self):
        """Empty graph → status='empty', graph_size=0, avg_confidence=0.0."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.introspect()

        assert result['graph_size'] == 0
        assert result['avg_confidence'] == 0.0
        assert result['top_nodes'] == []
        assert result['recent_nodes'] == []
        assert result['sources'] == {}
        assert result['status'] == 'empty'


# ═══════════════════════════════════════════════════════════════════
#  Test 2: introspect() returns correct graph_size
# ═══════════════════════════════════════════════════════════════════

class TestIntrospectGraphSize:
    """introspect() should report the correct number of nodes."""

    def test_introspect_returns_correct_graph_size(self):
        """After adding 3 nodes, graph_size should be 3."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Add 3 nodes with different sources and confidences
        _add_node_to_graph(
            graph, node_id="node_1",
            abstraction="ibukota Jepang adalah Tokyo",
            concept="Apa ibukota Jepang?",
            confidence=0.7, source='benchmark',
        )
        _add_node_to_graph(
            graph, node_id="node_2",
            abstraction="ibukota Prancis adalah Paris",
            concept="Apa ibukota Prancis?",
            confidence=0.8, source='user_correction',
        )
        _add_node_to_graph(
            graph, node_id="node_3",
            abstraction="2+2=4",
            concept="Berapa 2+2?",
            confidence=0.9, source='benchmark',
        )

        self_instance = SelfCore()
        result = self_instance.introspect()

        assert result['graph_size'] == 3
        assert result['status'] == 'small'  # 3 < 10
        # avg_confidence = (0.7 + 0.8 + 0.9) / 3
        assert result['avg_confidence'] == pytest.approx(0.8, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
#  Test 3: introspect() top_nodes sorted by confidence
# ═══════════════════════════════════════════════════════════════════

class TestIntrospectTopNodesByConfidence:
    """top_nodes should be sorted by confidence in descending order."""

    def test_introspect_top_nodes_sorted_by_confidence(self):
        """Node with highest confidence should appear first in top_nodes."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Add nodes with varying confidence — deliberately not in order
        _add_node_to_graph(
            graph, node_id="low_conf",
            abstraction="tebakan sembarang",
            concept="Pertanyaan sulit?",
            confidence=0.3, source='self_discovered',
        )
        _add_node_to_graph(
            graph, node_id="high_conf",
            abstraction="jawaban yang sangat yakin",
            concept="Pertanyaan mudah?",
            confidence=0.95, source='benchmark',
        )
        _add_node_to_graph(
            graph, node_id="mid_conf",
            abstraction="jawaban cukup yakin",
            concept="Pertanyaan menengah?",
            confidence=0.6, source='user_correction',
        )

        self_instance = SelfCore()
        result = self_instance.introspect()

        top = result['top_nodes']
        assert len(top) == 3

        # Highest confidence first
        assert top[0]['id'] == 'high_conf'
        assert top[0]['confidence'] == 0.95

        # Then middle
        assert top[1]['id'] == 'mid_conf'
        assert top[1]['confidence'] == 0.6

        # Then lowest
        assert top[2]['id'] == 'low_conf'
        assert top[2]['confidence'] == 0.3

        # Verify abstraction is truncated to 100 chars
        for node_info in top:
            assert len(node_info['abstraction']) <= 100


# ═══════════════════════════════════════════════════════════════════
#  Test 4: introspect() sources count
# ═══════════════════════════════════════════════════════════════════

class TestIntrospectSourcesCount:
    """sources dict should count nodes per source correctly."""

    def test_introspect_sources_count(self):
        """sources should correctly count benchmark, user_correction, etc."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Add nodes with different sources
        _add_node_to_graph(
            graph, node_id="b1",
            abstraction="benchmark node 1",
            concept="Q1?", confidence=0.7, source='benchmark',
        )
        _add_node_to_graph(
            graph, node_id="b2",
            abstraction="benchmark node 2",
            concept="Q2?", confidence=0.8, source='benchmark',
        )
        _add_node_to_graph(
            graph, node_id="b3",
            abstraction="benchmark node 3",
            concept="Q3?", confidence=0.9, source='benchmark',
        )
        _add_node_to_graph(
            graph, node_id="u1",
            abstraction="user correction node 1",
            concept="Q4?", confidence=0.6, source='user_correction',
        )
        _add_node_to_graph(
            graph, node_id="s1",
            abstraction="self discovered node 1",
            concept="Q5?", confidence=0.5, source='self_discovered',
        )

        self_instance = SelfCore()
        result = self_instance.introspect()

        sources = result['sources']
        assert sources.get('benchmark') == 3
        assert sources.get('user_correction') == 1
        assert sources.get('self_discovered') == 1


# ═══════════════════════════════════════════════════════════════════
#  Test 5: introspect() with no graph (self._graph = None)
# ═══════════════════════════════════════════════════════════════════

class TestIntrospectNoGraph:
    """introspect() should return default dict when graph is unavailable."""

    def test_introspect_no_graph(self):
        """When self._graph is None and get_shared_graph fails, return defaults."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        # Make get_shared_graph raise ImportError so introspect() can't
        # fall back to it
        original_fn = ub_module.get_shared_graph
        ub_module.get_shared_graph = lambda: (_ for _ in ()).throw(
            ImportError("test: no graph available")
        )

        try:
            self_instance = SelfCore()
            # self._graph is None by default
            assert self_instance._graph is None

            result = self_instance.introspect()

            # Should return default dict without error
            assert result['graph_size'] == 0
            assert result['avg_confidence'] == 0.0
            assert result['top_nodes'] == []
            assert result['recent_nodes'] == []
            assert result['sources'] == {}
            assert result['status'] == 'empty'
        finally:
            ub_module.get_shared_graph = original_fn
