#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_reinforce.py
# @WHAT:  Test SELF.reinforce() and SELF.penalize() — feedback-driven confidence adjustment
# @PART:  self-ai/tests
# @ENTRY: python -m pytest self-ai/tests/test_reinforce.py -v

"""Test: Does reinforce() and penalize() correctly adjust node confidence?

This test verifies the feedback loop:
  1. reinforce(question, confirmed_answer) increases confidence of matching nodes
  2. penalize(question, wrong_answer) decreases confidence of matching nodes
  3. Only nodes whose experience contains the answer are affected
  4. Both methods are graceful when the graph is empty or unavailable
  5. A learn() → reinforce() workflow strengthens nodes from 0.6

The test uses a temporary UnderstandingGraph with a mocked retrieve()
method (since bge-m3 is not available in test environments).

Run:
  cd self-ai/src
  python -m pytest ../tests/test_reinforce.py -v
"""

import os
import sys
import tempfile
import logging
from unittest.mock import patch

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
    persistent data. The graph has NO embedding model (bge-m3 unavailable)
    so retrieve() would return [] — we mock it in tests that need retrieval.
    """
    from derivation.understanding_builder import UnderstandingGraph
    tmpdir = tempfile.mkdtemp()
    return UnderstandingGraph(store_path=os.path.join(tmpdir, 'test_graph.json'))


def _add_node_to_graph(graph, node_id, abstraction, concept, confidence=0.6):
    """Helper: add an UnderstandingNode directly to the graph.

    This bypasses the need for embedding-based retrieval — we add nodes
    directly and then mock graph.retrieve() to return them.
    """
    from derivation.understanding_builder import UnderstandingNode

    node = UnderstandingNode(
        id=node_id,
        name=f"Test node {node_id}",
        concept=concept,
        abstraction=abstraction,
        conditions=[concept.lower()],
        condition_embedding=None,
        source='test',
        confidence=confidence,
        lifecycle='new',
    )
    graph.add_node(node)
    return node


# ═══════════════════════════════════════════════════════════════════
#  Test 1: reinforce() increases confidence
# ═══════════════════════════════════════════════════════════════════

class TestReinforceIncreasesConfidence:
    """reinforce() should increase confidence of matching nodes."""

    def test_reinforce_increases_confidence(self):
        """After reinforce(), the node's confidence should be higher than before."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Add a node directly
        node = _add_node_to_graph(
            graph,
            node_id="test_r1",
            abstraction="ketika ditanya 'siapa presiden', jawaban yang benar adalah 'Sukarno' bukan 'Jokowi'",
            concept="Siapa presiden Indonesia pertama?",
            confidence=0.6,
        )
        original_confidence = node.confidence

        # Mock retrieve to return our node
        with patch.object(graph, 'retrieve', return_value=[(node, 0.9)]):
            self_instance = SelfCore()
            result = self_instance.reinforce(
                question="Siapa presiden Indonesia pertama?",
                confirmed_answer="Sukarno",
            )

        assert result['reinforced_count'] == 1
        assert 'test_r1' in result['node_ids']
        # Confidence should have increased by 0.08
        new_confidence = result['new_confidences']['test_r1']
        assert new_confidence > original_confidence
        assert new_confidence == pytest.approx(0.68, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
#  Test 2: reinforce() only affects matching nodes
# ═══════════════════════════════════════════════════════════════════

class TestReinforceOnlyMatchingNodes:
    """reinforce() should only affect nodes whose experience contains the confirmed answer."""

    def test_reinforce_only_matching_nodes(self):
        """Nodes whose experience doesn't contain confirmed_answer are not reinforced."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Add two nodes: one matching, one not matching
        matching_node = _add_node_to_graph(
            graph,
            node_id="match_node",
            abstraction="jawaban yang benar adalah 'Tokyo' bukan 'Osaka'",
            concept="Apa ibukota Jepang?",
            confidence=0.6,
        )
        non_matching_node = _add_node_to_graph(
            graph,
            node_id="no_match_node",
            abstraction="jawaban yang benar adalah 'Paris' bukan 'London'",
            concept="Apa ibukota Prancis?",
            confidence=0.6,
        )

        # Mock retrieve to return BOTH nodes
        with patch.object(graph, 'retrieve', return_value=[
            (matching_node, 0.9),
            (non_matching_node, 0.7),
        ]):
            self_instance = SelfCore()
            result = self_instance.reinforce(
                question="Apa ibukota Jepang?",
                confirmed_answer="Tokyo",
            )

        # Only the matching node should be reinforced
        assert result['reinforced_count'] == 1
        assert 'match_node' in result['node_ids']
        assert 'no_match_node' not in result['node_ids']

        # Verify: matching node confidence went up, non-matching stayed same
        assert graph.get_node('match_node').confidence > 0.6
        assert graph.get_node('no_match_node').confidence == 0.6


# ═══════════════════════════════════════════════════════════════════
#  Test 3: penalize() decreases confidence
# ═══════════════════════════════════════════════════════════════════

class TestPenalizeDecreasesConfidence:
    """penalize() should decrease confidence of matching nodes."""

    def test_penalize_decreases_confidence(self):
        """After penalize(), the node's confidence should be lower than before."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Add a node directly
        node = _add_node_to_graph(
            graph,
            node_id="test_p1",
            abstraction="ketika ditanya 'berapa 2+2', jawaban yang benar adalah '5' bukan '3'",
            concept="Berapa hasil 2+2?",
            confidence=0.7,
        )
        original_confidence = node.confidence

        # Mock retrieve to return our node
        with patch.object(graph, 'retrieve', return_value=[(node, 0.9)]):
            self_instance = SelfCore()
            result = self_instance.penalize(
                question="Berapa hasil 2+2?",
                wrong_answer="5",
            )

        assert result['penalized_count'] == 1
        assert 'test_p1' in result['node_ids']
        # Confidence should have decreased by 0.1
        new_confidence = result['new_confidences']['test_p1']
        assert new_confidence < original_confidence
        assert new_confidence == pytest.approx(0.6, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
#  Test 4: penalize() only affects matching nodes
# ═══════════════════════════════════════════════════════════════════

class TestPenalizeOnlyMatchingNodes:
    """penalize() should only affect nodes whose experience contains the wrong answer."""

    def test_penalize_only_matching_nodes(self):
        """Nodes whose experience doesn't contain wrong_answer are not penalized."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Add two nodes: one matching, one not matching
        matching_node = _add_node_to_graph(
            graph,
            node_id="pen_match",
            abstraction="jawaban yang benar adalah '8' bukan '7'",
            concept="Berapa 5+3?",
            confidence=0.6,
        )
        non_matching_node = _add_node_to_graph(
            graph,
            node_id="pen_no_match",
            abstraction="jawaban yang benar adalah 'Merkurius' bukan 'Venus'",
            concept="Planet terdekat dari matahari?",
            confidence=0.6,
        )

        # Mock retrieve to return BOTH nodes
        with patch.object(graph, 'retrieve', return_value=[
            (matching_node, 0.9),
            (non_matching_node, 0.7),
        ]):
            self_instance = SelfCore()
            result = self_instance.penalize(
                question="Berapa 5+3?",
                wrong_answer="7",
            )

        # Only the matching node should be penalized
        assert result['penalized_count'] == 1
        assert 'pen_match' in result['node_ids']
        assert 'pen_no_match' not in result['node_ids']

        # Verify: matching node confidence went down, non-matching stayed same
        assert graph.get_node('pen_match').confidence < 0.6
        assert graph.get_node('pen_no_match').confidence == 0.6


# ═══════════════════════════════════════════════════════════════════
#  Test 5: reinforce() with no graph — no error
# ═══════════════════════════════════════════════════════════════════

class TestReinforceNoGraph:
    """reinforce() should not error when the graph is empty or unavailable."""

    def test_reinforce_no_graph(self):
        """When graph.retrieve() returns empty, reinforce() returns count 0."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Graph is empty — retrieve() with no bge-m3 returns []
        self_instance = SelfCore()
        result = self_instance.reinforce(
            question="Pertanyaan apapun",
            confirmed_answer="Jawaban apapun",
        )

        assert result['reinforced_count'] == 0
        assert result['node_ids'] == []
        assert result['new_confidences'] == {}

    def test_penalize_no_graph(self):
        """When graph.retrieve() returns empty, penalize() returns count 0."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        # Graph is empty — retrieve() with no bge-m3 returns []
        self_instance = SelfCore()
        result = self_instance.penalize(
            question="Pertanyaan apapun",
            wrong_answer="Jawaban salah",
        )

        assert result['penalized_count'] == 0
        assert result['node_ids'] == []
        assert result['new_confidences'] == {}

    def test_reinforce_graph_import_error(self):
        """When get_shared_graph raises ImportError, reinforce() returns count 0."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        # Make get_shared_graph raise ImportError
        original_fn = ub_module.get_shared_graph
        ub_module.get_shared_graph = lambda: (_ for _ in ()).throw(ImportError("test"))

        try:
            self_instance = SelfCore()
            result = self_instance.reinforce(
                question="test?",
                confirmed_answer="right",
            )
            assert result['reinforced_count'] == 0
            assert result['node_ids'] == []
            assert result['new_confidences'] == {}
        finally:
            ub_module.get_shared_graph = original_fn


# ═══════════════════════════════════════════════════════════════════
#  Test 6: learn() → reinforce() workflow
# ═══════════════════════════════════════════════════════════════════

class TestLearnThenReinforceWorkflow:
    """Full workflow: learn() creates node at 0.6, reinforce() strengthens it."""

    def test_learn_then_reinforce_workflow(self):
        """After learn() + reinforce(), node confidence increases from 0.6."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        self_instance = SelfCore()

        # Step 1: learn a correction
        learn_result = self_instance.learn(
            question="Siapa presiden Indonesia pertama?",
            wrong_answer="Jokowi",
            correction="Sukarno",
        )

        assert learn_result['confidence'] == 0.6
        node_id = learn_result['node_id']
        assert node_id is not None

        # Step 2: Verify the node is in the graph with confidence 0.6
        node = graph.get_node(node_id)
        assert node is not None
        assert node.confidence == 0.6

        # Step 3: Mock retrieve to return the learned node
        with patch.object(graph, 'retrieve', return_value=[(node, 0.9)]):
            reinforce_result = self_instance.reinforce(
                question="Siapa presiden Indonesia pertama?",
                confirmed_answer="Sukarno",
            )

        # Step 4: Verify reinforcement worked
        assert reinforce_result['reinforced_count'] == 1
        assert node_id in reinforce_result['node_ids']
        new_confidence = reinforce_result['new_confidences'][node_id]
        assert new_confidence > 0.6
        # 0.6 + 0.08 = 0.68
        assert new_confidence == pytest.approx(0.68, abs=0.01)

        # Step 5: Verify the node in the graph has the updated confidence
        node_after = graph.get_node(node_id)
        assert node_after.confidence == pytest.approx(0.68, abs=0.01)

    def test_reinforce_multiple_times_approaches_max(self):
        """Multiple reinforce() calls should approach but not exceed 1.0."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        graph = _make_graph()
        ub_module._shared_graph = graph

        self_instance = SelfCore()

        # Learn a correction
        learn_result = self_instance.learn(
            question="Apa ibukota Prancis?",
            wrong_answer="London",
            correction="Paris",
        )
        node_id = learn_result['node_id']
        node = graph.get_node(node_id)

        # Reinforce multiple times
        for i in range(10):
            with patch.object(graph, 'retrieve', return_value=[(node, 0.9)]):
                result = self_instance.reinforce(
                    question="Apa ibukota Prancis?",
                    confirmed_answer="Paris",
                )

        # Confidence should be capped at 1.0
        final_node = graph.get_node(node_id)
        assert final_node.confidence <= 1.0
        # After 10 reinforcements from 0.6 with step 0.08:
        # 0.6 + 10*0.08 = 1.4, capped at 1.0
        assert final_node.confidence == 1.0
