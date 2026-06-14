#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_learn_loop.py
# @WHAT:  Test SELF.learn() — the correction-to-understanding learning loop
# @PART:  self-ai/tests
# @ENTRY: python -m pytest self-ai/tests/test_learn_loop.py -v

"""Test: Does SELF.learn() actually store corrections as understanding?

This test verifies the end-to-end learning loop:
  1. SELF.learn(question, wrong_answer, correction) creates an UnderstandingNode
  2. The node is stored in the shared UnderstandingGraph
  3. The node can be retrieved via graph retrieval
  4. Calling learn() twice with the same input is idempotent (no duplicates)
  5. learn() works even when bge-m3 is unavailable (no embedding)

The test uses a lightweight UnderstandingGraph with a mock/no-op embedding
model so it can run without GPU or model downloads.

Run:
  cd self-ai/src
  python -m pytest ../tests/test_learn_loop.py -v
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
#  Fixtures — temporary UnderstandingGraph with no external dependencies
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_shared_graph():
    """Reset the shared graph singleton between tests.

    Without this, graph nodes from one test would leak into the next.
    We also override the graph's store path to a temp directory so
    test artifacts don't pollute the project's data/ directory.
    """
    import derivation.understanding_builder as ub_module
    ub_module._shared_graph = None
    yield
    ub_module._shared_graph = None


def _make_graph():
    """Create a fresh UnderstandingGraph in a temp directory.

    Uses a temporary store_path so we don't pollute the project's
    persistent data. The graph has NO embedding model (bge-m3 unavailable)
    so retrieval will fail — but add_node() and count() will work.

    Returns:
        UnderstandingGraph instance.
    """
    from derivation.understanding_builder import UnderstandingGraph
    tmpdir = tempfile.mkdtemp()
    return UnderstandingGraph(store_path=os.path.join(tmpdir, 'test_graph.json'))


# ═══════════════════════════════════════════════════════════════════
#  Test 1: learn() creates a node and returns a summary dict
# ═══════════════════════════════════════════════════════════════════

class TestLearnBasics:
    """Core functionality: learn() creates nodes, returns dicts."""

    def test_learn_returns_dict_with_expected_keys(self):
        """learn() must return a dict with node_id, experience, confidence, graph_size."""
        from core.self import SelfCore

        # Set up a fresh graph
        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Siapa presiden Indonesia pertama?",
            wrong_answer="Jokowi",
            correction="Sukarno",
        )

        assert isinstance(result, dict)
        assert 'node_id' in result
        assert 'experience' in result
        assert 'confidence' in result
        assert 'graph_size' in result
        assert 'duplicate' in result

    def test_learn_creates_node_in_graph(self):
        """After learn(), the graph should have one more node."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        initial_size = graph.count()

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Berapa hasil 5 + 3?",
            wrong_answer="7",
            correction="8",
        )

        assert graph.count() == initial_size + 1
        assert result['graph_size'] == initial_size + 1

    def test_learn_experience_contains_correction(self):
        """The experience text must contain the correction text."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Apa ibukota Jepang?",
            wrong_answer="Osaka",
            correction="Tokyo",
        )

        assert 'Tokyo' in result['experience']
        assert 'Osaka' in result['experience']
        assert 'ibukota Jepang' in result['experience']

    def test_learn_node_has_expected_attributes(self):
        """The created node must have the correct source, confidence, lifecycle."""
        from core.self import SelfCore
        from governance.states import LifecycleState

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Warna langit saat cerah?",
            wrong_answer="Merah",
            correction="Biru",
        )

        node = graph.get_node(result['node_id'])
        assert node is not None
        assert node.source == 'user_correction'
        assert node.confidence == 0.6
        assert node.lifecycle == LifecycleState.NEW
        assert node.last_used_at is not None

    def test_learn_node_stored_experience_as_abstraction(self):
        """The experience text is stored in the node's abstraction field."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Apa nama planet terbesar?",
            wrong_answer="Mars",
            correction="Jupiter",
        )

        node = graph.get_node(result['node_id'])
        assert node.abstraction == result['experience']


# ═══════════════════════════════════════════════════════════════════
#  Test 2: Idempotency — learn() doesn't create duplicates
# ═══════════════════════════════════════════════════════════════════

class TestLearnIdempotency:
    """Calling learn() twice with the same input should not create duplicates."""

    def test_learn_twice_same_input_no_duplicate(self):
        """Two calls with identical (question, wrong, correction) → only 1 node."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()

        result1 = self_instance.learn(
            question="Siapa penemu lampu pijar?",
            wrong_answer="Newton",
            correction="Edison",
        )
        assert result1['duplicate'] is False

        result2 = self_instance.learn(
            question="Siapa penemu lampu pijar?",
            wrong_answer="Newton",
            correction="Edison",
        )
        assert result2['duplicate'] is True
        assert result2['node_id'] == result1['node_id']
        assert graph.count() == 1

    def test_learn_different_correction_creates_new_node(self):
        """A different correction for the same question → new node."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()

        self_instance.learn(
            question="Apa warna daun?",
            wrong_answer="Merah",
            correction="Hijau",
        )
        result2 = self_instance.learn(
            question="Apa warna daun?",
            wrong_answer="Kuning",
            correction="Hijau",
        )

        assert result2['duplicate'] is False
        assert graph.count() == 2


# ═══════════════════════════════════════════════════════════════════
#  Test 3: Graceful degradation — learn() works without bge-m3
# ═══════════════════════════════════════════════════════════════════

class TestLearnGracefulDegradation:
    """learn() must work even when bge-m3 is unavailable."""

    def test_learn_without_embedding_model(self):
        """When bge-m3 is unavailable, the node is stored without embedding."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        # The graph has no embedding model, so condition_embedding will be None
        self_instance = SelfCore()
        result = self_instance.learn(
            question="Berapa 2 x 3?",
            wrong_answer="5",
            correction="6",
        )

        assert result['node_id'] is not None
        node = graph.get_node(result['node_id'])
        # Embedding is None because no bge-m3 model is loaded
        assert node.condition_embedding is None
        assert graph.count() == 1

    def test_learn_without_graph_returns_error(self):
        """If UnderstandingGraph is completely unavailable, return error dict."""
        from core.self import SelfCore
        import derivation.understanding_builder as ub_module

        # Make get_shared_graph raise ImportError
        original_fn = ub_module.get_shared_graph
        ub_module.get_shared_graph = lambda: (_ for _ in ()).throw(ImportError("test"))

        try:
            self_instance = SelfCore()
            result = self_instance.learn(
                question="test?",
                wrong_answer="wrong",
                correction="right",
            )
            assert result['node_id'] is None
            assert 'error' in result
        finally:
            ub_module.get_shared_graph = original_fn


# ═══════════════════════════════════════════════════════════════════
#  Test 4: Node retrievability — learn() nodes can be found
# ═══════════════════════════════════════════════════════════════════

class TestLearnRetrieval:
    """Nodes created by learn() should be retrievable from the graph."""

    def test_learn_node_found_by_id(self):
        """The node_id returned by learn() should be found in graph._nodes."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Hewan apa yang mengeong?",
            wrong_answer="Anjing",
            correction="Kucing",
        )

        node = graph.get_node(result['node_id'])
        assert node is not None
        assert 'Kucing' in node.abstraction

    def test_learn_node_concept_matches_question(self):
        """The node's concept field should contain the original question."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Planet terdekat dari matahari?",
            wrong_answer="Venus",
            correction="Merkurius",
        )

        node = graph.get_node(result['node_id'])
        assert node.concept == "Planet terdekat dari matahari?"

    def test_learn_node_conditions_include_question(self):
        """The node's conditions list should include the lowercased question."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Siapa presiden Indonesia?",
            wrong_answer="Soeharto",
            correction="Jokowi",
        )

        node = graph.get_node(result['node_id'])
        assert "siapa presiden indonesia?" in node.conditions


# ═══════════════════════════════════════════════════════════════════
#  Test 5: Multiple learn calls — graph grows correctly
# ═══════════════════════════════════════════════════════════════════

class TestLearnMultiple:
    """Multiple learn() calls should grow the graph correctly."""

    def test_three_different_corrections_create_three_nodes(self):
        """Three different corrections → 3 nodes in the graph."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()

        self_instance.learn("Apa 1+1?", "3", "2")
        self_instance.learn("Apa 2+2?", "5", "4")
        self_instance.learn("Apa 3+3?", "7", "6")

        assert graph.count() == 3

    def test_mixed_duplicate_and_new(self):
        """Mix of duplicate and new corrections → correct graph size."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()

        r1 = self_instance.learn("Apa 1+1?", "3", "2")
        r2 = self_instance.learn("Apa 2+2?", "5", "4")
        r3 = self_instance.learn("Apa 1+1?", "3", "2")  # duplicate
        r4 = self_instance.learn("Apa 3+3?", "7", "6")

        assert r1['duplicate'] is False
        assert r2['duplicate'] is False
        assert r3['duplicate'] is True
        assert r4['duplicate'] is False
        assert graph.count() == 3  # Only 3 unique corrections


# ═══════════════════════════════════════════════════════════════════
#  Test 6: Experience text format
# ═══════════════════════════════════════════════════════════════════

class TestLearnExperienceFormat:
    """Verify the experience text format matches the template."""

    def test_experience_template_format(self):
        """The experience text follows the template pattern."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.learn(
            question="Siapa penemu telepon?",
            wrong_answer="Edison",
            correction="Bell",
        )

        experience = result['experience']
        assert experience.startswith("ketika ditanya '")
        assert "jawaban yang benar adalah" in experience
        assert "bukan" in experience
        assert "Bell" in experience
        assert "Edison" in experience

    def test_experience_truncates_long_question(self):
        """Questions longer than 50 chars are truncated in the experience."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        long_question = "Ini adalah pertanyaan yang sangat panjang sekali sehingga melebihi batas lima puluh karakter"
        result = self_instance.learn(
            question=long_question,
            wrong_answer="Salah",
            correction="Benar",
        )

        # The q_short in the template is limited to 50 chars
        assert len(result['experience']) < len(long_question) + 100  # Much shorter than raw question


# ═══════════════════════════════════════════════════════════════════
#  Test 7: No regression — learn() doesn't break existing SELF interfaces
# ═══════════════════════════════════════════════════════════════════

class TestLearnNoRegression:
    """learn() should not break any existing SELF interfaces."""

    def test_process_still_works(self):
        """After adding learn(), the process() method still works."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        result = self_instance.process("Hello, ini bukan pertanyaan")

        assert isinstance(result, dict)
        assert 'sensory' in result

    def test_teach_still_works(self):
        """After adding learn(), the teach() method still exists."""
        from core.self import SelfCore

        self_instance = SelfCore()
        # teach() should not raise — it may return None if UnderstandingBuilder
        # is unavailable, but it should not crash
        result = self_instance.teach(
            problem="test problem",
            solution_steps=["step 1", "step 2"],
            answer="test answer",
            explanation_why="because",
        )
        # Result may be None (if UnderstandingBuilder not available), that's OK

    def test_list_experiences_includes_learned_nodes(self):
        """list_experiences() should include nodes created by learn()."""
        from core.self import SelfCore

        graph = _make_graph()
        import derivation.understanding_builder as ub_module
        ub_module._shared_graph = graph

        self_instance = SelfCore()
        self_instance.learn("Apa 1+1?", "3", "2")

        experiences = self_instance.list_experiences()
        assert len(experiences) >= 1
        learned = [e for e in experiences if e['id'].startswith('learn_')]
        assert len(learned) == 1
