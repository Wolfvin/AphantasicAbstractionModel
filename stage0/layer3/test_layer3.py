"""
Layer 3 Test Suite — Reasoning, Policy, Coder, and Evidence Traceability

Tests for all Layer 3 components:
  - ReasoningEngine.build_chain() basic functionality
  - DeductiveChain and DeductiveStep dataclasses
  - PolicyEngine.check_with_rsvs_policy()
  - CoderLayer.analyze_with_rsvs()
  - Evidence traceability in ReasoningStep and AamResponse
"""

import sys
import os

# Ensure project root is on sys.path so layer2/layer3/pipeline imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# P1-1: Import DeductivePolicyEngine / DeductiveCoderLayer (Layer 3 extensions)
# P1-2: Use same-package relative where possible, absolute for cross-package
from layer3.reasoning import ReasoningEngine, DeductiveChain, DeductiveStep
from layer3.policy import DeductivePolicyEngine, PolicyEngine, PolicyRule, PolicyViolation
from layer3.coder import DeductiveCoderLayer, CoderLayer, CodeElement, CodeAnalysisResult
from layer2.pattern import PatternResult, ReasoningStep
from layer2.bridge import V12PipelineBridge, _FallbackGraph
from pipeline import AamResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bridge() -> V12PipelineBridge:
    """Create a bridge with fallback graph (no Rust core needed)."""
    return V12PipelineBridge()


def _make_pattern_result(
    trigger: str = "test trigger",
    step_count: int = 2,
    evidence_per_step: int = 2,
    anomalies: list[dict] | None = None,
    confidence: float = 0.7,
) -> PatternResult:
    """Create a PatternResult for testing ReasoningEngine."""
    result = PatternResult(
        trigger=trigger,
        confidence=confidence,
        anomalies=anomalies or [],
    )

    step_types = ["trigger", "recall", "cross_reference", "anomaly", "pattern", "narrative"]
    for i in range(min(step_count, len(step_types))):
        step = ReasoningStep(
            step_type=step_types[i],
            description=f"Step {i}: {step_types[i]}",
            data={"test_key": f"test_value_{i}"},
            evidence_nodes=[f"evidence_node_{i}_{j}" for j in range(evidence_per_step)],
            confidence=0.5 + 0.1 * i,
        )
        result.steps.append(step)

    result.pattern = "test pattern completion"
    return result


# ---------------------------------------------------------------------------
# Test: DeductiveStep and DeductiveChain dataclasses
# ---------------------------------------------------------------------------

def test_deductive_step_creation():
    """Test DeductiveStep can be created with all fields."""
    step = DeductiveStep(
        claim="The entity exists based on evidence",
        evidence_node_ids=[("node_1", "0"), ("node_2", "1")],
        confidence=0.85,
        reasoning_type="deduction",
        grounding_scores={"node_1": 0.9, "node_2": 0.8},
        description="Deductive step from node_1 and node_2",
    )
    assert step.claim == "The entity exists based on evidence"
    assert len(step.evidence_node_ids) == 2
    assert step.evidence_node_ids[0] == ("node_1", "0")
    assert step.confidence == 0.85
    assert step.reasoning_type == "deduction"
    assert step.grounding_scores["node_1"] == 0.9
    assert step.description == "Deductive step from node_1 and node_2"
    print("✓ test_deductive_step_creation passed")


def test_deductive_step_defaults():
    """Test DeductiveStep has correct defaults."""
    step = DeductiveStep(claim="test claim")
    assert step.evidence_node_ids == []
    assert step.confidence == 0.5
    assert step.reasoning_type == "deduction"
    assert step.grounding_scores == {}
    assert step.description == ""
    print("✓ test_deductive_step_defaults passed")


def test_deductive_step_to_dict():
    """Test DeductiveStep serializes correctly."""
    step = DeductiveStep(
        claim="test",
        evidence_node_ids=[("n1", "0")],
        confidence=0.75,
        grounding_scores={"n1": 0.8},
    )
    d = step.to_dict()
    assert d["claim"] == "test"
    assert d["evidence_node_ids"][0]["node_id"] == "n1"
    assert d["evidence_node_ids"][0]["sense_id"] == "0"
    assert d["confidence"] == 0.75
    assert d["grounding_scores"]["n1"] == 0.8
    print("✓ test_deductive_step_to_dict passed")


def test_deductive_chain_creation():
    """Test DeductiveChain can be created with steps."""
    chain = DeductiveChain(
        trigger="test trigger",
        steps=[
            DeductiveStep(claim="step 1", confidence=0.8),
            DeductiveStep(claim="step 2", confidence=0.7),
        ],
        conclusion="final conclusion",
        aggregate_confidence=0.75,
    )
    assert chain.trigger == "test trigger"
    assert len(chain.steps) == 2
    assert chain.conclusion == "final conclusion"
    assert chain.aggregate_confidence == 0.75
    print("✓ test_deductive_chain_creation passed")


def test_deductive_chain_to_dict():
    """Test DeductiveChain serializes correctly."""
    chain = DeductiveChain(
        trigger="t",
        steps=[DeductiveStep(claim="c1", confidence=0.6)],
        conclusion="conclusion",
        aggregate_confidence=0.6,
        evidence_summary=[{"node_id": "n1", "sense_id": "0", "grounding_score": 0.5, "used_in_step": "deduction"}],
    )
    d = chain.to_dict()
    assert d["trigger"] == "t"
    assert len(d["steps"]) == 1
    assert d["conclusion"] == "conclusion"
    assert d["aggregate_confidence"] == 0.6
    assert len(d["evidence_summary"]) == 1
    print("✓ test_deductive_chain_to_dict passed")


# ---------------------------------------------------------------------------
# Test: ReasoningEngine.build_chain()
# ---------------------------------------------------------------------------

def test_reasoning_engine_build_chain_basic():
    """Test ReasoningEngine.build_chain() with a simple PatternResult."""
    bridge = _make_bridge()
    engine = ReasoningEngine(bridge=bridge)

    pattern_result = _make_pattern_result(
        trigger="Snow Plum Pill theft",
        step_count=3,
        evidence_per_step=2,
        confidence=0.7,
    )

    chain = engine.build_chain(pattern_result)

    assert isinstance(chain, DeductiveChain)
    assert chain.trigger == "Snow Plum Pill theft"
    assert len(chain.steps) == 5  # extract, compose, ground, explore, conclude
    assert chain.conclusion != ""
    assert chain.aggregate_confidence > 0.0
    print("✓ test_reasoning_engine_build_chain_basic passed")


def test_reasoning_engine_build_chain_with_anomalies():
    """Test ReasoningEngine with anomaly-driven reasoning."""
    bridge = _make_bridge()
    engine = ReasoningEngine(bridge=bridge)

    pattern_result = _make_pattern_result(
        trigger="anomalous pattern",
        step_count=2,
        anomalies=[{"type": "contradiction", "description": "test anomaly"}],
        confidence=0.5,
    )

    chain = engine.build_chain(pattern_result)

    assert chain.trigger == "anomalous pattern"
    # The conclusion step should be anomaly-driven
    conclusion_step = chain.steps[-1]
    assert conclusion_step.reasoning_type == "anomaly_driven"
    print("✓ test_reasoning_engine_build_chain_with_anomalies passed")


def test_reasoning_engine_build_chain_evidence_traceability():
    """Test that each step in the chain has evidence_node_ids."""
    bridge = _make_bridge()
    engine = ReasoningEngine(bridge=bridge)

    pattern_result = _make_pattern_result(
        trigger="traceability test",
        step_count=2,
        evidence_per_step=3,
    )

    chain = engine.build_chain(pattern_result)

    # Every step should have evidence_node_ids (even if empty for some)
    for step in chain.steps:
        assert isinstance(step.evidence_node_ids, list)
        assert isinstance(step.grounding_scores, dict)

    # Evidence summary should be populated
    assert len(chain.evidence_summary) > 0
    print("✓ test_reasoning_engine_build_chain_evidence_traceability passed")


def test_reasoning_engine_build_chain_empty_pattern():
    """Test ReasoningEngine with an empty PatternResult."""
    bridge = _make_bridge()
    engine = ReasoningEngine(bridge=bridge)

    pattern_result = PatternResult(trigger="empty test")

    chain = engine.build_chain(pattern_result)

    assert chain.trigger == "empty test"
    assert len(chain.steps) == 5  # Still produces 5 steps
    # Confidence should be low with no evidence
    assert chain.aggregate_confidence < 0.5
    print("✓ test_reasoning_engine_build_chain_empty_pattern passed")


# ---------------------------------------------------------------------------
# Test: PolicyEngine.check_with_rsvs_policy()
# ---------------------------------------------------------------------------

def test_policy_engine_check_with_rsvs_policy_no_meta():
    """Test check_with_rsvs_policy() when no PolicyMeta is available."""
    bridge = _make_bridge()
    # P1-1: Use DeductivePolicyEngine for check_with_rsvs_policy()
    engine = DeductivePolicyEngine(bridge=bridge)

    # Add a simple rule
    engine.add_rule(PolicyRule(
        rule_id="TEST_001",
        domain="test",
        description="Test rule",
        condition=lambda ctx: ctx.get("value", 0) > 10,
        severity="warning",
    ))

    result = engine.check_with_rsvs_policy("test_entity")

    assert result["entity"] == "test_entity"
    assert result["policy_meta_available"] is False
    assert result["governance_score"] == 0.0
    assert result["status_flip_count"] == 0
    # When no metadata, trust_weight should be 0.7 (standalone mode)
    assert result["trust_weight"] == 0.7
    assert result["instability_flag"] is False
    assert "compliance" in result
    assert "adjusted_confidence" in result
    print("✓ test_policy_engine_check_with_rsvs_policy_no_meta passed")


def test_policy_engine_check_with_rsvs_policy_with_meta():
    """Test check_with_rsvs_policy() when PolicyMeta is available in node_info."""
    # Create a mock bridge that returns PolicyMeta
    class MockBridge:
        is_available = True
        is_rust_core = False

        def node_info(self, label):
            return {
                "label": label,
                "policy_meta": {
                    "governance_score": 0.8,
                    "status_flip_count": 5,
                    "seen_fingerprints": ["policy_check:test_entity"],
                },
            }

        def relate(self, text):
            return None

        def query(self, text, context=""):
            return None

        def ingest(self, text):
            return {"success": True}

    engine = DeductivePolicyEngine(bridge=MockBridge())

    engine.add_rule(PolicyRule(
        rule_id="TEST_002",
        domain="test",
        description="Value must be positive",
        condition=lambda ctx: ctx.get("value", 0) > 0,
        severity="critical",
    ))

    result = engine.check_with_rsvs_policy("test_entity")

    assert result["policy_meta_available"] is True
    assert result["governance_score"] == 0.8
    assert result["status_flip_count"] == 5
    # trust_weight = 0.5 + 0.5 * 0.8 = 0.9
    assert abs(result["trust_weight"] - 0.9) < 0.01
    # 5 flips > 3 threshold → instability
    assert result["instability_flag"] is True
    # "policy_check:test_entity" in seen_fingerprints
    assert result["is_duplicate"] is True
    print("✓ test_policy_engine_check_with_rsvs_policy_with_meta passed")


def test_policy_engine_check_with_rsvs_policy_stability():
    """Test that instability flag triggers correctly based on flip count."""
    class LowFlipBridge:
        is_available = True
        is_rust_core = False
        def node_info(self, label):
            return {"policy_meta": {"governance_score": 0.5, "status_flip_count": 2, "seen_fingerprints": []}}
        def relate(self, text): return None
        def query(self, text, context=""): return None
        def ingest(self, text): return {"success": True}

    engine = DeductivePolicyEngine(bridge=LowFlipBridge())
    result = engine.check_with_rsvs_policy("stable_entity")

    assert result["instability_flag"] is False  # 2 flips ≤ 3
    assert result["is_duplicate"] is False
    print("✓ test_policy_engine_check_with_rsvs_policy_stability passed")


# ---------------------------------------------------------------------------
# Test: CoderLayer.analyze_with_rsvs()
# ---------------------------------------------------------------------------

def test_coder_layer_analyze_with_rsvs_basic():
    """Test analyze_with_rsvs() with simple Python code."""
    bridge = _make_bridge()
    # P1-1: Use DeductiveCoderLayer for analyze_with_rsvs()
    coder = DeductiveCoderLayer(bridge=bridge)

    code = '''
def hello(name):
    """Say hello."""
    return f"Hello, {name}!"

class Greeter:
    def __init__(self, greeting):
        self.greeting = greeting

    def greet(self, name):
        return f"{self.greeting}, {name}!"
'''

    result = coder.analyze_with_rsvs(code, bridge=bridge)

    assert isinstance(result, CodeAnalysisResult)
    assert len(result.elements_found) > 0
    # Should find the function and class
    element_kinds = [e.get("kind") for e in result.elements_found]
    assert "function" in element_kinds
    assert "class" in element_kinds
    print("✓ test_coder_layer_analyze_with_rsvs_basic passed")


def test_coder_layer_analyze_with_rsvs_empty_code():
    """Test analyze_with_rsvs() with empty code."""
    bridge = _make_bridge()
    coder = DeductiveCoderLayer(bridge=bridge)

    result = coder.analyze_with_rsvs("", bridge=bridge)

    assert isinstance(result, CodeAnalysisResult)
    assert len(result.elements_found) == 0
    print("✓ test_coder_layer_analyze_with_rsvs_empty_code passed")


def test_coder_layer_analyze_with_rsvs_non_python():
    """Test analyze_with_rsvs() with non-Python code (regex fallback)."""
    bridge = _make_bridge()
    coder = DeductiveCoderLayer(bridge=bridge)

    rust_code = '''
pub fn calculate(x: i32, y: i32) -> i32 {
    x + y
}

struct Point {
    x: f64,
    y: f64,
}
'''

    result = coder.analyze_with_rsvs(rust_code, bridge=bridge, language="rust")

    assert isinstance(result, CodeAnalysisResult)
    assert len(result.elements_found) > 0
    print("✓ test_coder_layer_analyze_with_rsvs_non_python passed")


# ---------------------------------------------------------------------------
# Test: Evidence Traceability
# ---------------------------------------------------------------------------

def test_reasoning_step_evidence_node_ids():
    """Test that ReasoningStep now has evidence_node_ids and grounding_scores."""
    step = ReasoningStep(
        step_type="recall",
        description="Recalled evidence nodes",
        evidence_nodes=["node_a", "node_b"],
        confidence=0.8,
        evidence_node_ids=[("node_a", "0"), ("node_b", "1")],
        grounding_scores={"node_a": 0.9, "node_b": 0.7},
    )

    assert step.evidence_node_ids == [("node_a", "0"), ("node_b", "1")]
    assert step.grounding_scores == {"node_a": 0.9, "node_b": 0.7}

    d = step.to_dict()
    assert "evidence_node_ids" in d
    assert len(d["evidence_node_ids"]) == 2
    assert d["evidence_node_ids"][0]["node_id"] == "node_a"
    assert d["evidence_node_ids"][0]["sense_id"] == "0"
    assert "grounding_scores" in d
    print("✓ test_reasoning_step_evidence_node_ids passed")


def test_aam_response_evidence_traceability():
    """Test that AamResponse includes evidence_node_ids in reasoning chain."""
    step = ReasoningStep(
        step_type="trigger",
        description="Test step",
        evidence_nodes=["n1"],
        confidence=0.5,
        evidence_node_ids=[("n1", "0")],
        grounding_scores={"n1": 0.6},
    )

    response = AamResponse(
        answer="test answer",
        confidence=0.5,
        reasoning_chain=[step],
        evidence_chain=[{"type": "test"}],
        anomalies=[],
        predictions=[],
        belief_updates=[],
    )

    d = response.to_dict()
    chain = d["reasoning_chain"]
    assert len(chain) == 1
    assert "evidence_node_ids" in chain[0]
    assert chain[0]["evidence_node_ids"][0]["node_id"] == "n1"
    assert "grounding_scores" in chain[0]
    assert chain[0]["grounding_scores"]["n1"] == 0.6
    print("✓ test_aam_response_evidence_traceability passed")


def test_deductive_chain_full_traceability():
    """Test full traceability from DeductiveChain back to evidence nodes."""
    bridge = _make_bridge()
    engine = ReasoningEngine(bridge=bridge)

    pattern_result = _make_pattern_result(
        trigger="full trace test",
        step_count=3,
        evidence_per_step=2,
    )

    chain = engine.build_chain(pattern_result)

    # Verify traceability: each step has evidence references
    for step in chain.steps:
        assert hasattr(step, "evidence_node_ids")
        assert hasattr(step, "grounding_scores")
        assert hasattr(step, "claim")

    # Verify evidence summary traces back to steps
    for entry in chain.evidence_summary:
        assert "node_id" in entry
        assert "sense_id" in entry
        assert "grounding_score" in entry
        assert "used_in_step" in entry

    print("✓ test_deductive_chain_full_traceability passed")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Layer 3 Test Suite")
    print("=" * 60)

    # DeductiveStep tests
    print("\n--- DeductiveStep ---")
    test_deductive_step_creation()
    test_deductive_step_defaults()
    test_deductive_step_to_dict()

    # DeductiveChain tests
    print("\n--- DeductiveChain ---")
    test_deductive_chain_creation()
    test_deductive_chain_to_dict()

    # ReasoningEngine tests
    print("\n--- ReasoningEngine ---")
    test_reasoning_engine_build_chain_basic()
    test_reasoning_engine_build_chain_with_anomalies()
    test_reasoning_engine_build_chain_evidence_traceability()
    test_reasoning_engine_build_chain_empty_pattern()

    # PolicyEngine tests
    print("\n--- PolicyEngine RSVS ---")
    test_policy_engine_check_with_rsvs_policy_no_meta()
    test_policy_engine_check_with_rsvs_policy_with_meta()
    test_policy_engine_check_with_rsvs_policy_stability()

    # CoderLayer tests
    print("\n--- CoderLayer RSVS ---")
    test_coder_layer_analyze_with_rsvs_basic()
    test_coder_layer_analyze_with_rsvs_empty_code()
    test_coder_layer_analyze_with_rsvs_non_python()

    # Evidence traceability tests
    print("\n--- Evidence Traceability ---")
    test_reasoning_step_evidence_node_ids()
    test_aam_response_evidence_traceability()
    test_deductive_chain_full_traceability()

    print("\n" + "=" * 60)
    print("All 16 tests passed! ✓")
    print("=" * 60)
