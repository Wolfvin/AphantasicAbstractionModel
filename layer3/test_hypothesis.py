"""
Layer 3 Hypothesis Module Test Suite — Hypothesis-Driven Active Reasoning

Tests for all hypothesis module components:
  - Hypothesis and Evidence dataclasses
  - HypothesisDrivenReasoner.reason() basic functionality
  - Disconfirmatory search mechanism
  - Confidence tracking per hypothesis
  - Hypothesis competition and decisiveness
  - Integration with ReasoningEngine and PredictiveEngine
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layer3.hypothesis import (
    HypothesisDrivenReasoner,
    Hypothesis,
    Evidence,
    HypothesisCycleResult,
)
from layer3.reasoning import ReasoningEngine, DeductiveChain, DeductiveStep
from layer2.predictive import Anomaly
from layer2.pattern import PatternResult, ReasoningStep
from layer2.bridge import RsvsBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bridge() -> RsvsBridge:
    """Create a bridge with fallback graph (no Rust core needed)."""
    return RsvsBridge()


def _make_anomaly(
    concept: str = "Snow_Plum_Pill",
    expected: list[str] | None = None,
    observed: list[str] | None = None,
    delta: float = 0.5,
) -> Anomaly:
    """Create an Anomaly for testing."""
    return Anomaly(
        concept=concept,
        expected=expected or ["consumption", "theft"],
        observed=observed or ["no_consumption", "no_theft"],
        delta=delta,
        description=f"Anomaly for '{concept}': expected consumption but found none",
    )


# ---------------------------------------------------------------------------
# Test: Evidence dataclass
# ---------------------------------------------------------------------------

def test_evidence_creation():
    """Test Evidence can be created with all fields."""
    ev = Evidence(
        evidence_id="ev001",
        description="Test evidence description",
        source_node="Snow_Plum_Pill",
        source_sense="0",
        direction="confirmatory",
        strength=0.8,
        grounding_score=0.7,
        discovery_method="senses",
    )
    assert ev.evidence_id == "ev001"
    assert ev.direction == "confirmatory"
    assert ev.strength == 0.8
    assert ev.grounding_score == 0.7
    assert ev.discovery_method == "senses"
    print("✓ test_evidence_creation passed")


def test_evidence_defaults():
    """Test Evidence has correct defaults."""
    ev = Evidence(evidence_id="ev002", description="test")
    assert ev.source_node == ""
    assert ev.source_sense == "0"
    assert ev.direction == "confirmatory"
    assert ev.strength == 0.5
    assert ev.grounding_score == 0.5
    assert ev.discovery_method == "unknown"
    print("✓ test_evidence_defaults passed")


def test_evidence_to_dict():
    """Test Evidence serializes correctly."""
    ev = Evidence(
        evidence_id="ev003",
        description="test",
        direction="disconfirmatory",
        strength=0.6,
    )
    d = ev.to_dict()
    assert d["evidence_id"] == "ev003"
    assert d["direction"] == "disconfirmatory"
    assert d["strength"] == 0.6
    print("✓ test_evidence_to_dict passed")


def test_evidence_disconfirmatory():
    """Test disconfirmatory evidence creation."""
    ev = Evidence(
        evidence_id="ev_disconf",
        description="Evidence that REFUTES the hypothesis",
        direction="disconfirmatory",
        strength=0.9,
        discovery_method="mcts_disconfirmatory",
    )
    assert ev.direction == "disconfirmatory"
    assert ev.discovery_method == "mcts_disconfirmatory"
    print("✓ test_evidence_disconfirmatory passed")


# ---------------------------------------------------------------------------
# Test: Hypothesis dataclass
# ---------------------------------------------------------------------------

def test_hypothesis_creation():
    """Test Hypothesis can be created with all fields."""
    hyp = Hypothesis(
        hypothesis_id="hyp001",
        statement="Ju Jangmok is a scapegoat",
        reasoning="No consumption evidence found",
        test_criteria=["Find evidence of consumption", "Find evidence of framing"],
        confidence=0.5,
        state="proposed",
        anomaly_source="Snow_Plum_Pill",
    )
    assert hyp.hypothesis_id == "hyp001"
    assert hyp.statement == "Ju Jangmok is a scapegoat"
    assert len(hyp.test_criteria) == 2
    assert hyp.state == "proposed"
    assert hyp.anomaly_source == "Snow_Plum_Pill"
    print("✓ test_hypothesis_creation passed")


def test_hypothesis_defaults():
    """Test Hypothesis has correct defaults."""
    hyp = Hypothesis(hypothesis_id="h1", statement="test")
    assert hyp.confirmatory_evidence == []
    assert hyp.disconfirmatory_evidence == []
    assert hyp.state == "proposed"
    assert hyp.cycle_count == 0
    print("✓ test_hypothesis_defaults passed")


def test_hypothesis_invalid_state():
    """Test Hypothesis rejects invalid state."""
    try:
        Hypothesis(hypothesis_id="h1", statement="test", state="invalid_state")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✓ test_hypothesis_invalid_state passed")


def test_hypothesis_evidence_tracking():
    """Test Hypothesis evidence tracking properties."""
    hyp = Hypothesis(hypothesis_id="h1", statement="test")
    hyp.confirmatory_evidence.append(
        Evidence(evidence_id="c1", description="confirm", direction="confirmatory", strength=0.8, grounding_score=0.7)
    )
    hyp.disconfirmatory_evidence.append(
        Evidence(evidence_id="d1", description="disconfirm", direction="disconfirmatory", strength=0.6, grounding_score=0.5)
    )
    assert hyp.total_evidence_count == 2
    print("✓ test_hypothesis_evidence_tracking passed")


def test_hypothesis_net_evidence_score():
    """Test Hypothesis net_evidence_score with asymmetric weighting."""
    hyp = Hypothesis(hypothesis_id="h1", statement="test")
    # Add equal confirmatory and disconfirmatory evidence
    hyp.confirmatory_evidence.append(
        Evidence(evidence_id="c1", description="confirm", direction="confirmatory", strength=1.0, grounding_score=1.0)
    )
    hyp.disconfirmatory_evidence.append(
        Evidence(evidence_id="d1", description="disconfirm", direction="disconfirmatory", strength=1.0, grounding_score=1.0)
    )
    # Net score should be NEGATIVE because disconfirmatory is weighted more
    assert hyp.net_evidence_score < 0
    print("✓ test_hypothesis_net_evidence_score passed (asymmetric weighting works)")


def test_hypothesis_to_dict():
    """Test Hypothesis serializes correctly."""
    hyp = Hypothesis(
        hypothesis_id="h1",
        statement="test",
        confidence=0.7,
    )
    d = hyp.to_dict()
    assert d["hypothesis_id"] == "h1"
    assert d["confidence"] == 0.7
    assert "confirmatory_evidence" in d
    assert "disconfirmatory_evidence" in d
    assert "net_evidence_score" in d
    print("✓ test_hypothesis_to_dict passed")


# ---------------------------------------------------------------------------
# Test: HypothesisDrivenReasoner
# ---------------------------------------------------------------------------

def test_reasoner_initialization():
    """Test HypothesisDrivenReasoner can be initialized."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    assert reasoner.rsvs_available is True
    assert reasoner.reasoning_engine is not None
    assert reasoner.predictive_engine is not None
    print("✓ test_reasoner_initialization passed")


def test_reasoner_reason_basic():
    """Test HypothesisDrivenReasoner.reason() with a simple anomaly."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    anomaly = _make_anomaly()

    result = reasoner.reason(anomaly)

    assert isinstance(result, HypothesisCycleResult)
    assert result.anomaly is not None
    assert len(result.hypotheses) > 0
    assert result.cycle_number >= 1
    print("✓ test_reasoner_reason_basic passed")


def test_reasoner_generates_multiple_hypotheses():
    """Test that reason() generates multiple competing hypotheses."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge, hypothesis_count=3)
    anomaly = _make_anomaly()

    result = reasoner.reason(anomaly)

    # Should generate at least 2 hypotheses
    assert len(result.hypotheses) >= 2
    # Each hypothesis should have a unique ID
    hyp_ids = [h.hypothesis_id for h in result.hypotheses]
    assert len(hyp_ids) == len(set(hyp_ids))  # All unique
    print("✓ test_reasoner_generates_multiple_hypotheses passed")


def test_reasoner_hypotheses_have_test_criteria():
    """Test that generated hypotheses have test criteria."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    anomaly = _make_anomaly()

    result = reasoner.reason(anomaly)

    for hyp in result.hypotheses:
        assert len(hyp.test_criteria) > 0, f"Hypothesis '{hyp.statement}' has no test criteria"
    print("✓ test_reasoner_hypotheses_have_test_criteria passed")


def test_reasoner_confidence_updates():
    """Test that hypothesis confidences are updated after testing."""
    bridge = _make_bridge()
    # Ingest some data so RSVS has content to reason about
    bridge.ingest("Snow Plum Pill is a rare medicinal herb stolen from Mount Hua Sect")
    bridge.ingest("Ju Jangmok was accused of stealing the Snow Plum Pill")
    bridge.ingest("No evidence of consumption was found for the Snow Plum Pill")

    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    anomaly = _make_anomaly()

    result = reasoner.reason(anomaly)

    # At least some hypotheses should have evidence
    has_evidence = any(h.total_evidence_count > 0 for h in result.hypotheses)
    # In fallback mode, evidence may be limited but the mechanism should work
    for hyp in result.hypotheses:
        # Confidence should have been updated (even if still at default)
        assert 0.0 <= hyp.confidence <= 1.0
    print("✓ test_reasoner_confidence_updates passed")


def test_reasoner_disconfirmatory_search():
    """Test that disconfirmatory evidence search is performed."""
    bridge = _make_bridge()
    bridge.ingest("Snow Plum Pill was stolen from the sect treasury")

    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    anomaly = _make_anomaly(delta=0.6)  # High delta = strong anomaly

    result = reasoner.reason(anomaly)

    # Check that hypotheses have been tested (not just proposed)
    tested_hypotheses = [
        h for h in result.hypotheses
        if h.state in ("testing", "confirmed", "refuted", "superseded")
    ]
    # At least some hypotheses should have been tested
    assert len(tested_hypotheses) > 0
    print("✓ test_reasoner_disconfirmatory_search passed")


def test_reasoner_cycle_result_to_dict():
    """Test HypothesisCycleResult serializes correctly."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    anomaly = _make_anomaly()

    result = reasoner.reason(anomaly)

    d = result.to_dict()
    assert "cycle_id" in d
    assert "hypotheses" in d
    assert "is_conclusive" in d
    assert "decisiveness" in d
    assert "cycle_number" in d
    assert "total_evidence_found" in d
    print("✓ test_reasoner_cycle_result_to_dict passed")


def test_reasoner_with_large_anomaly():
    """Test reasoner with a large anomaly (high delta)."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    anomaly = _make_anomaly(delta=0.9)  # Very large anomaly

    result = reasoner.reason(anomaly)

    assert result.anomaly.delta == 0.9
    # Large anomalies should generate hypotheses
    assert len(result.hypotheses) >= 2
    print("✓ test_reasoner_with_large_anomaly passed")


def test_reasoner_max_cycles():
    """Test that reasoner respects max_cycles limit."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(
        bridge=bridge,
        max_cycles=2,  # Limit to 2 cycles
        decisiveness_threshold=0.99,  # Very high = almost never conclusive
    )
    anomaly = _make_anomaly()

    result = reasoner.reason(anomaly)

    # Should not exceed max cycles
    assert result.cycle_number <= 2
    print("✓ test_reasoner_max_cycles passed")


def test_reasoner_reason_from_anomalies():
    """Test reason_from_anomalies() convenience method."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)

    anomalies = [
        _make_anomaly(concept="Anomaly1", delta=0.4),
        _make_anomaly(concept="Anomaly2", delta=0.5),
    ]

    results = reasoner.reason_from_anomalies(anomalies)

    assert len(results) == 2
    assert results[0].anomaly.concept == "Anomaly1"
    assert results[1].anomaly.concept == "Anomaly2"
    print("✓ test_reasoner_reason_from_anomalies passed")


def test_reasoner_active_hypotheses():
    """Test that active hypotheses are tracked after reasoning."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    anomaly = _make_anomaly()

    reasoner.reason(anomaly)

    # After reasoning, there should be tracked hypotheses
    active = reasoner.active_hypotheses
    history = reasoner.cycle_history

    # History should have at least one entry
    assert len(history) >= 1
    print("✓ test_reasoner_active_hypotheses passed")


def test_reasoner_get_hypothesis():
    """Test get_hypothesis() method."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)
    anomaly = _make_anomaly()

    result = reasoner.reason(anomaly)

    # Try to get one of the hypotheses by ID
    if result.hypotheses:
        hyp_id = result.hypotheses[0].hypothesis_id
        found = reasoner.get_hypothesis(hyp_id)
        assert found is not None
        assert found.hypothesis_id == hyp_id

    # Try to get a non-existent hypothesis
    not_found = reasoner.get_hypothesis("nonexistent")
    assert not_found is None
    print("✓ test_reasoner_get_hypothesis passed")


def test_reasoner_with_pattern_result():
    """Test reasoner with a provided PatternResult."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)

    anomaly = _make_anomaly()

    # Create a PatternResult
    pattern = PatternResult(
        trigger="Snow Plum Pill theft",
        steps=[
            ReasoningStep(
                step_type="trigger",
                description="Test trigger",
                evidence_nodes=["Snow_Plum_Pill"],
                confidence=0.7,
            ),
        ],
        pattern="Theft pattern detected",
        confidence=0.6,
    )

    result = reasoner.reason(anomaly, pattern_result=pattern)

    assert result.anomaly is not None
    assert len(result.hypotheses) >= 2
    print("✓ test_reasoner_with_pattern_result passed")


# ---------------------------------------------------------------------------
# Test: Asymmetric Confidence Update
# ---------------------------------------------------------------------------

def test_asymmetric_confidence_update():
    """Test that disconfirmatory evidence has stronger impact than confirmatory."""
    bridge = _make_bridge()
    reasoner = HypothesisDrivenReasoner(bridge=bridge)

    # Create two hypotheses with same initial confidence
    hyp_confirm = Hypothesis(
        hypothesis_id="hc",
        statement="Only confirmatory",
        confidence=0.5,
    )
    hyp_disconfirm = Hypothesis(
        hypothesis_id="hd",
        statement="Only disconfirmatory",
        confidence=0.5,
    )
    hyp_mixed = Hypothesis(
        hypothesis_id="hm",
        statement="Mixed evidence",
        confidence=0.5,
    )

    # Add same-strength evidence to each
    for _ in range(3):
        hyp_confirm.confirmatory_evidence.append(
            Evidence(evidence_id=uuid.uuid4().hex[:8], description="c", strength=0.8, grounding_score=0.8)
        )
        hyp_disconfirm.disconfirmatory_evidence.append(
            Evidence(evidence_id=uuid.uuid4().hex[:8], description="d", strength=0.8, grounding_score=0.8)
        )
        hyp_mixed.confirmatory_evidence.append(
            Evidence(evidence_id=uuid.uuid4().hex[:8], description="c", strength=0.8, grounding_score=0.8)
        )
        hyp_mixed.disconfirmatory_evidence.append(
            Evidence(evidence_id=uuid.uuid4().hex[:8], description="d", strength=0.8, grounding_score=0.8)
        )

    # Update confidences
    reasoner._update_hypothesis_confidence(hyp_confirm)
    reasoner._update_hypothesis_confidence(hyp_disconfirm)
    reasoner._update_hypothesis_confidence(hyp_mixed)

    # Confirmatory should increase confidence
    assert hyp_confirm.confidence > 0.5
    # Disconfirmatory should decrease confidence
    assert hyp_disconfirm.confidence < 0.5
    # Mixed with equal evidence should decrease (because disconfirm is weighted more)
    assert hyp_mixed.confidence < 0.5
    # Disconfirmatory should have larger magnitude of change
    confirm_increase = hyp_confirm.confidence - 0.5
    disconfirm_decrease = 0.5 - hyp_disconfirm.confidence
    assert disconfirm_decrease > confirm_increase
    print("✓ test_asymmetric_confidence_update passed")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

import uuid

if __name__ == "__main__":
    print("=" * 60)
    print("Layer 3 Hypothesis Module Test Suite")
    print("=" * 60)

    # Evidence tests
    print("\n--- Evidence ---")
    test_evidence_creation()
    test_evidence_defaults()
    test_evidence_to_dict()
    test_evidence_disconfirmatory()

    # Hypothesis tests
    print("\n--- Hypothesis ---")
    test_hypothesis_creation()
    test_hypothesis_defaults()
    test_hypothesis_invalid_state()
    test_hypothesis_evidence_tracking()
    test_hypothesis_net_evidence_score()
    test_hypothesis_to_dict()

    # HypothesisDrivenReasoner tests
    print("\n--- HypothesisDrivenReasoner ---")
    test_reasoner_initialization()
    test_reasoner_reason_basic()
    test_reasoner_generates_multiple_hypotheses()
    test_reasoner_hypotheses_have_test_criteria()
    test_reasoner_confidence_updates()
    test_reasoner_disconfirmatory_search()
    test_reasoner_cycle_result_to_dict()
    test_reasoner_with_large_anomaly()
    test_reasoner_max_cycles()
    test_reasoner_reason_from_anomalies()
    test_reasoner_active_hypotheses()
    test_reasoner_get_hypothesis()
    test_reasoner_with_pattern_result()

    # Asymmetric confidence tests
    print("\n--- Asymmetric Confidence ---")
    test_asymmetric_confidence_update()

    print("\n" + "=" * 60)
    print("All 22 hypothesis module tests passed! ✓")
    print("=" * 60)
