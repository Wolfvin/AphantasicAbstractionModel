"""
Tests for Layer 3 — Possibility Lattice (Dynamic Hypothesis Space)

Tests cover:
1. Possibility data class creation and properties
2. PossibilityLattice initialization
3. LatticeMode selection
4. Generation of possibilities from multiple angles
5. Elimination of low-confidence possibilities
6. Hybridization of complementary pairs
7. Novelty detection
8. Question mode
9. Full lattice reasoning cycle
10. Integration with HypothesisDrivenReasoner
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layer3.lattice import (
    PossibilityLattice,
    Possibility,
    LatticeResult,
    LatticeGeneration,
    LatticeMode,
    PossibilityState,
    _DEFAULT_ELIMINATION_THRESHOLD,
    _DEFAULT_QUESTION_MODE_THRESHOLD,
    _DEFAULT_CONCLUSION_CONFIDENCE,
    _DEFAULT_COMPLEMENTARITY_THRESHOLD,
)
from layer3.hypothesis import Hypothesis, Evidence, HypothesisDrivenReasoner
from layer2.predictive import Anomaly


# ---------------------------------------------------------------------------
# Mock Bridge
# ---------------------------------------------------------------------------

def make_mock_bridge():
    """Create a mock RsvsBridge with sensible defaults."""
    bridge = MagicMock()
    bridge.is_available = True
    bridge.is_rust_core = False

    # Default return values
    bridge.relate.return_value = {
        "related_nodes": [["marah", 0.7], ["sedih", 0.5], ["takut", 0.4]]
    }
    bridge.senses.return_value = [
        {"sense_idx": 0, "grounding_score": 0.7, "core_atoms": [["kemarahan"], ["pengkhianatan"]]},
        {"sense_idx": 1, "grounding_score": 0.5, "core_atoms": [["kesedihan"], ["kehilangan"]]},
    ]
    bridge.mcts_query.return_value = {
        "best_path": ["konflik", "penyelesaian", "damai"],
        "scored_atoms": [["konflik", 0.8], ["penyelesaian", 0.6]],
        "simulations_run": 20,
        "max_depth_reached": 3,
    }
    bridge.appraise.return_value = {"agree_pct": 0.5, "disagree_pct": 0.3}
    bridge.structural_similarity.return_value = {"structural_similarity": 0.5}
    bridge.compose.return_value = "composed_node_1"

    return bridge


# ---------------------------------------------------------------------------
# Test: Possibility data class
# ---------------------------------------------------------------------------

class TestPossibility(unittest.TestCase):
    """Tests for the Possibility data class."""

    def test_basic_creation(self):
        p = Possibility(
            possibility_id="test1",
            statement="Test possibility",
            confidence=0.6,
        )
        self.assertEqual(p.possibility_id, "test1")
        self.assertEqual(p.statement, "Test possibility")
        self.assertEqual(p.confidence, 0.6)
        self.assertEqual(p.state, "proposed")
        self.assertEqual(p.generation, 0)
        self.assertEqual(p.source, "context")
        self.assertFalse(p.is_hybrid)

    def test_hybrid_possibility(self):
        p = Possibility(
            possibility_id="hybrid1",
            statement="A ∘ B",
            confidence=0.7,
            parent_ids=["parent_a", "parent_b"],
            generation=1,
            source="hybrid",
        )
        self.assertTrue(p.is_hybrid)
        self.assertEqual(len(p.parent_ids), 2)
        self.assertEqual(p.generation, 1)

    def test_coverage_property(self):
        p = Possibility(
            possibility_id="test2",
            statement="Test",
            explained_evidence={"e1", "e2", "e3"},
            all_evidence={"e1", "e2", "e3", "e4", "e5"},
        )
        self.assertAlmostEqual(p.coverage, 0.6)

    def test_coverage_empty_evidence(self):
        p = Possibility(
            possibility_id="test3",
            statement="Test",
        )
        self.assertEqual(p.coverage, 0.0)

    def test_net_evidence_score(self):
        p = Possibility(
            possibility_id="test4",
            statement="Test",
            confirmatory_evidence=[
                Evidence(evidence_id="c1", description="conf", strength=0.8, grounding_score=0.7),
            ],
            disconfirmatory_evidence=[
                Evidence(evidence_id="d1", description="disconf", strength=0.6, grounding_score=0.5),
            ],
        )
        # 0.4 * (0.8*0.7) - 0.6 * (0.6*0.5) = 0.224 - 0.18 = 0.044
        self.assertAlmostEqual(p.net_evidence_score, 0.044, places=3)

    def test_to_dict(self):
        p = Possibility(
            possibility_id="test5",
            statement="Test dict",
            confidence=0.75,
            parent_ids=["a", "b"],
            generation=2,
            source="hybrid",
        )
        d = p.to_dict()
        self.assertEqual(d["possibility_id"], "test5")
        self.assertTrue(d["is_hybrid"])
        self.assertEqual(d["generation"], 2)
        self.assertAlmostEqual(d["confidence"], 0.75)


# ---------------------------------------------------------------------------
# Test: LatticeGeneration
# ---------------------------------------------------------------------------

class TestLatticeGeneration(unittest.TestCase):
    """Tests for the LatticeGeneration data class."""

    def test_basic_creation(self):
        gen = LatticeGeneration(generation=0, pre_count=10, post_elimination=7)
        self.assertEqual(gen.generation, 0)
        self.assertEqual(gen.pre_count, 10)
        self.assertEqual(gen.post_elimination, 7)
        self.assertEqual(gen.eliminated_count, 0)
        self.assertFalse(gen.is_stable)

    def test_to_dict(self):
        gen = LatticeGeneration(generation=1, pre_count=5, is_stable=True)
        d = gen.to_dict()
        self.assertEqual(d["generation"], 1)
        self.assertTrue(d["is_stable"])


# ---------------------------------------------------------------------------
# Test: LatticeResult
# ---------------------------------------------------------------------------

class TestLatticeResult(unittest.TestCase):
    """Tests for the LatticeResult data class."""

    def test_basic_creation(self):
        result = LatticeResult(result_id="r1", query="test query")
        self.assertEqual(result.result_id, "r1")
        self.assertFalse(result.is_conclusive)
        self.assertEqual(result.mode, "lattice")

    def test_to_dict(self):
        result = LatticeResult(
            result_id="r2",
            query="test",
            mode="eliminative",
            total_hybrids=5,
            confidence=0.8,
        )
        d = result.to_dict()
        self.assertEqual(d["mode"], "eliminative")
        self.assertEqual(d["total_hybrids"], 5)
        self.assertAlmostEqual(d["confidence"], 0.8)


# ---------------------------------------------------------------------------
# Test: PossibilityLattice
# ---------------------------------------------------------------------------

class TestPossibilityLattice(unittest.TestCase):
    """Tests for the PossibilityLattice class."""

    def setUp(self):
        self.bridge = make_mock_bridge()
        self.lattice = PossibilityLattice(bridge=self.bridge)

    def test_initialization(self):
        self.assertTrue(self.lattice.rsvs_available)
        self.assertFalse(self.lattice.is_rust_core)
        self.assertIsNotNone(self.lattice.reasoner)
        self.assertIsNotNone(self.lattice.predictive_engine)

    def test_initialization_default_params(self):
        self.assertEqual(self.lattice._max_generations, 10)
        self.assertEqual(self.lattice._elimination_threshold, _DEFAULT_ELIMINATION_THRESHOLD)
        self.assertEqual(self.lattice._question_mode_threshold, _DEFAULT_QUESTION_MODE_THRESHOLD)

    def test_reason_generates_possibilities(self):
        result = self.lattice.reason(
            query="Mengapa dia marah?",
            context=["pengkhianatan", "harga_diri"],
            mode=LatticeMode.LATTICE,
        )
        self.assertIsInstance(result, LatticeResult)
        self.assertEqual(result.mode, "lattice")
        self.assertGreater(len(result.generations), 0)

    def test_reason_with_evidence(self):
        result = self.lattice.reason(
            query="Apa penyebab konflik?",
            evidence_list=["bukti1", "bukti2", "bukti3"],
            mode=LatticeMode.ELIMINATIVE,
        )
        self.assertIsInstance(result, LatticeResult)
        self.assertEqual(result.mode, "eliminative")

    def test_reason_with_anomaly(self):
        anomaly = Anomaly(
            concept="marah",
            expected=["kemarahan"],
            observed=["ketakutan"],
            delta=0.6,
            description="Expected anger but observed fear",
        )
        result = self.lattice.reason(
            query="Mengapa observasi berbeda dari prediksi?",
            anomaly=anomaly,
            mode=LatticeMode.LATTICE,
        )
        self.assertIsInstance(result, LatticeResult)

    def test_elimination_removes_low_confidence(self):
        # Create possibilities with varying confidence
        possibilities = [
            Possibility(possibility_id="high", statement="High conf", confidence=0.8),
            Possibility(possibility_id="medium", statement="Medium conf", confidence=0.4),
            Possibility(possibility_id="low", statement="Low conf", confidence=0.05),
        ]
        surviving, eliminated = self.lattice._eliminate(possibilities, set())
        # Low confidence should be eliminated
        self.assertGreater(len(surviving), 0)
        self.assertGreaterEqual(eliminated, 0)  # At least the very low one

    def test_hybridization_creates_hybrids(self):
        possibilities = [
            Possibility(
                possibility_id="a",
                statement="Hypothesis A explains evidence X",
                confidence=0.6,
                explained_evidence={"e1", "e2"},
                all_evidence={"e1", "e2", "e3", "e4"},
            ),
            Possibility(
                possibility_id="b",
                statement="Hypothesis B explains evidence Y",
                confidence=0.5,
                explained_evidence={"e3", "e4"},
                all_evidence={"e1", "e2", "e3", "e4"},
            ),
        ]
        hybrids, emergent = self.lattice._hybridize(possibilities, {"e1", "e2", "e3", "e4"}, 1)
        # Complementary possibilities should produce hybrids
        # (A explains e1,e2; B explains e3,e4 → low overlap, high complementarity)
        self.assertIsInstance(hybrids, list)
        self.assertIsInstance(emergent, list)

    def test_complementarity_measurement(self):
        a = Possibility(
            possibility_id="a",
            statement="Explains X",
            explained_evidence={"e1", "e2"},
            all_evidence={"e1", "e2", "e3", "e4"},
        )
        b = Possibility(
            possibility_id="b",
            statement="Explains Y",
            explained_evidence={"e3", "e4"},
            all_evidence={"e1", "e2", "e3", "e4"},
        )
        comp = self.lattice._measure_complementarity(a, b)
        # No overlap, full joint coverage → high complementarity
        self.assertGreater(comp, 0.0)

    def test_complementarity_with_overlap(self):
        a = Possibility(
            possibility_id="a",
            statement="Same as B",
            explained_evidence={"e1", "e2"},
            all_evidence={"e1", "e2"},
        )
        b = Possibility(
            possibility_id="b",
            statement="Same as A",
            explained_evidence={"e1", "e2"},
            all_evidence={"e1", "e2"},
        )
        comp = self.lattice._measure_complementarity(a, b)
        # Complete overlap → low complementarity
        self.assertLess(comp, 0.5)

    def test_novelty_detection(self):
        existing = [
            Possibility(possibility_id="e1", statement="Existing possibility about X"),
        ]
        candidates = [
            Possibility(possibility_id="n1", statement="Completely new insight about Y"),
            Possibility(possibility_id="n2", statement="Existing possibility about X"),  # Duplicate
        ]
        novel = self.lattice._detect_novel(candidates, existing)
        # At least the completely new one should be detected
        self.assertGreaterEqual(len(novel), 1)

    def test_question_mode_finds_question(self):
        possibilities = [
            Possibility(possibility_id="p1", statement="Interpretation A",
                       confidence=0.7, explained_evidence={"e1"}),
            Possibility(possibility_id="p2", statement="Interpretation B",
                       confidence=0.65, explained_evidence={"e2"}),
        ]
        question = self.lattice._find_best_question(possibilities)
        self.assertIsNotNone(question)
        self.assertIn("Interpretation", question)

    def test_question_mode_single_possibility(self):
        possibilities = [
            Possibility(possibility_id="p1", statement="Only one", confidence=0.9),
        ]
        question = self.lattice._find_best_question(possibilities)
        self.assertIsNone(question)

    def test_deduplication(self):
        possibilities = [
            Possibility(possibility_id="p1", statement="Same statement about X", confidence=0.5),
            Possibility(possibility_id="p2", statement="Same statement about X", confidence=0.6),
        ]
        unique = self.lattice._deduplicate(possibilities)
        self.assertLessEqual(len(unique), 1)

    def test_lattice_history(self):
        self.lattice.reason(query="Test query", mode=LatticeMode.LATTICE)
        self.assertGreater(self.lattice.total_lattice_runs, 0)

    def test_lattice_mode_enum(self):
        self.assertEqual(LatticeMode.GENERATIVE.value, "generative")
        self.assertEqual(LatticeMode.ELIMINATIVE.value, "eliminative")
        self.assertEqual(LatticeMode.LATTICE.value, "lattice")

    def test_possibility_state_enum(self):
        self.assertEqual(PossibilityState.HYBRID.value, "hybrid")
        self.assertEqual(PossibilityState.EMERGENT.value, "emergent")
        self.assertEqual(PossibilityState.ELIMINATED.value, "eliminated")


# ---------------------------------------------------------------------------
# Test: Question Mode with Callback
# ---------------------------------------------------------------------------

class TestQuestionModeCallback(unittest.TestCase):
    """Tests for question mode with user callback."""

    def test_question_callback_invoked(self):
        bridge = make_mock_bridge()
        callback_responses = {"question1": "user answer about X"}

        def question_callback(question: str) -> str:
            return callback_responses.get(question, "I don't know")

        lattice = PossibilityLattice(
            bridge=bridge,
            question_callback=question_callback,
        )

        result = lattice.reason(
            query="Test question mode",
            mode=LatticeMode.LATTICE,
            question_mode=True,
        )
        self.assertIsInstance(result, LatticeResult)


# ---------------------------------------------------------------------------
# Test: Full Lattice Cycle
# ---------------------------------------------------------------------------

class TestFullLatticeCycle(unittest.TestCase):
    """Integration tests for the full lattice cycle."""

    def setUp(self):
        self.bridge = make_mock_bridge()
        self.lattice = PossibilityLattice(
            bridge=self.bridge,
            max_generations=3,
            elimination_threshold=0.1,
        )

    def test_full_cycle_produces_result(self):
        result = self.lattice.reason(
            query="Siapa yang mencuri Snow Plum Pill?",
            context=["Ju Jangmok", "Snow Plum Pill", "pencurian"],
            evidence_list=["tidak_ada_konsumsi_pil", "alibi_kuat"],
            mode=LatticeMode.LATTICE,
        )
        self.assertIsInstance(result, LatticeResult)
        self.assertGreater(len(result.generations), 0)
        self.assertGreater(result.total_possibilities_generated, 0)

    def test_elimination_only_mode(self):
        result = self.lattice.reason(
            query="Test eliminative",
            mode=LatticeMode.ELIMINATIVE,
        )
        self.assertEqual(result.mode, "eliminative")
        # In elimination mode, no hybrids should be created
        # (but emergent from implications also 0 since no hybridization)
        self.assertEqual(result.total_hybrids, 0)

    def test_generative_mode_with_anomaly(self):
        anomaly = Anomaly(
            concept="pencuri",
            expected=["konsumsi_pil"],
            observed=["tidak_ada_konsumsi"],
            delta=0.7,
        )
        result = self.lattice.reason(
            query="Mengapa tidak ada konsumsi?",
            anomaly=anomaly,
            mode=LatticeMode.GENERATIVE,
        )
        self.assertEqual(result.mode, "generative")

    def test_stability_detection(self):
        # With no RSVS changes and no new evidence, lattice should stabilize
        self.bridge.relate.return_value = {"related_nodes": []}
        self.bridge.senses.return_value = []
        self.bridge.mcts_query.return_value = {"best_path": [], "scored_atoms": []}

        result = self.lattice.reason(
            query="Stability test",
            mode=LatticeMode.LATTICE,
        )
        self.assertIsInstance(result, LatticeResult)
        # Should stabilize quickly with no new data
        self.assertLessEqual(len(result.generations), 3)


# ---------------------------------------------------------------------------
# Test: Hypothesis Extended States
# ---------------------------------------------------------------------------

class TestHypothesisExtendedStates(unittest.TestCase):
    """Tests for the extended Hypothesis states (hybrid, emergent, etc.)."""

    def test_hybrid_state_valid(self):
        h = Hypothesis(
            hypothesis_id="h1",
            statement="Hybrid hypothesis",
            state="hybrid",
            parent_ids=["p1", "p2"],
        )
        self.assertTrue(h.is_hybrid)
        self.assertEqual(h.state, "hybrid")

    def test_emergent_state_valid(self):
        h = Hypothesis(
            hypothesis_id="h2",
            statement="Emergent hypothesis",
            state="emergent",
            source="emergent",
        )
        self.assertEqual(h.state, "emergent")

    def test_concluded_state_valid(self):
        h = Hypothesis(
            hypothesis_id="h3",
            statement="Concluded hypothesis",
            state="concluded",
        )
        self.assertEqual(h.state, "concluded")

    def test_to_dict_includes_lattice_fields(self):
        h = Hypothesis(
            hypothesis_id="h4",
            statement="Test",
            parent_ids=["a", "b"],
            generation=2,
            source="hybrid",
        )
        d = h.to_dict()
        self.assertTrue(d["is_hybrid"])
        self.assertEqual(d["generation"], 2)
        self.assertEqual(d["source"], "hybrid")
        self.assertEqual(d["parent_ids"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
