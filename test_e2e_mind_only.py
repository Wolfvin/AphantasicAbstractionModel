#!/usr/bin/env python3
"""
AAM End-to-End Test — Buktikan Pikiran Dulu, Baru Latih Tubuh

This test proves that the "mind" (RSVS graph + cognitive layers) works
end-to-end WITHOUT the diffusion model. The diffusion model is the "body"
and is not needed for the core reasoning pipeline.

Flow tested:
    Text Input
      → Layer 0: TextAbstractor (abstraction into PerceptualTuples)
      → Adapter: observation_to_ingest_data (bridge to RSVS)
      → Layer 1: RSVS graph (ingest + structure)
      → Layer 2: SituationLayer + PatternOutput (context + reasoning)
      → Layer 3: ReasoningEngine (deductive chain)
      → Output: AamResponse with evidence, confidence, reasoning chain

This is the "proof of mind" — if this works, the graph can produce
traceable, evidence-backed conclusions. Only then does it make sense
to train a "body" (diffusion LLM) to narrate those conclusions fluently.

Analogi: Ini adalah Jin Soun yang SUDAH BISA menarik kesimpulan
dari ingatannya — bahkan sebelum tubuhnya bisa berbicara fasih.
"""

from __future__ import annotations

import sys
import time
import traceback

# Ensure the repo root is in the path
sys.path.insert(0, ".")


def test_layer0_abstraction():
    """Test 1: Layer 0 — Text abstraction into PerceptualTuples."""
    from layer0 import TextAbstractor
    from layer0.base import ModalityType

    abstractor = TextAbstractor()
    text = "Rain causes flood. Flood damages crops. Crops are food."

    obs = abstractor.abstract(text)

    assert obs is not None, "Observation should not be None"
    assert obs.modality == ModalityType.TEXT, f"Expected TEXT modality, got {obs.modality}"
    assert len(obs.tuples) > 0, f"Expected at least 1 tuple, got {len(obs.tuples)}"

    print(f"  ✅ Layer 0: Extracted {len(obs.tuples)} PerceptualTuples from text")
    for t in obs.tuples:
        print(f"     - {t.relation_type.value}({t.subject}, {t.predicate})")

    return obs


def test_adapter(obs=None):
    """Test 2: Adapter — PerceptualObservation → RSVS ingest data."""
    from layer0.adapter import observation_to_ingest_data, observation_to_ingest_dicts

    # If no observation passed, create one
    if obs is None:
        from layer0 import TextAbstractor
        abstractor = TextAbstractor()
        obs = abstractor.abstract("Rain causes flood. Flood damages crops.")

    ingest_text = observation_to_ingest_data(obs)
    assert ingest_text, "Ingest text should not be empty"
    assert "causes" in ingest_text or "is" in ingest_text, \
        f"Ingest text should contain relational info: {ingest_text[:100]}"

    ingest_dicts = observation_to_ingest_dicts(obs)
    assert len(ingest_dicts) > 0, "Ingest dicts should not be empty"

    print(f"  ✅ Adapter: Converted to RSVS ingest format")
    print(f"     Text preview: {ingest_text[:120]}...")
    print(f"     Structured: {len(ingest_dicts)} dicts")

    return ingest_text


def test_layer2_coder_layer():
    """Test 3: Layer 2 CoderLayer — Code parsing and analysis."""
    from layer2.coder_layer import (
        CoderLayer, parse_python_code, detect_language, CodeElement,
    )

    python_code = '''
class Calculator:
    """A simple calculator."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

def main():
    calc = Calculator()
    result = calc.add(1, 2)
'''

    # Test language detection
    lang = detect_language(python_code)
    assert lang == "python", f"Expected 'python', got '{lang}'"

    # Test parsing
    elements = parse_python_code(python_code, source="test_snippet")
    assert len(elements) > 0, "Should extract elements from Python code"

    # Check specific elements
    classes = [e for e in elements if e.kind == "class"]
    methods = [e for e in elements if e.kind == "method"]
    functions = [e for e in elements if e.kind == "function"]

    print(f"  ✅ CoderLayer: Parsed Python code into {len(elements)} elements")
    print(f"     Classes: {len(classes)}, Methods: {len(methods)}, Functions: {len(functions)}")

    # Test full analysis
    coder = CoderLayer()
    result = coder.analyze_code(python_code, language="python")
    assert result.confidence > 0, "Analysis should have some confidence"
    print(f"     Analysis confidence: {result.confidence:.3f}")

    return result


def test_layer2_policy_engine():
    """Test 4: Layer 2 PolicyEngine — Compliance checking."""
    from layer2.policy_engine import PolicyEngine, PolicyRule

    engine = PolicyEngine()
    engine.load_tax_rules_indonesia()

    # Check compliance for a test entity
    result = engine.check_compliance(
        "PT_Test_Company",
        context={"rate": 0.15, "income": 100_000_000, "npwp": "1234567890", "month": 3},
    )

    print(f"  ✅ PolicyEngine: {result['rules_evaluated']} rules evaluated")
    print(f"     Compliant: {result['compliant']}")
    print(f"     Violations: {len(result['violations'])}")

    # Test adding a custom rule
    custom_rule = PolicyRule(
        rule_id="TEST_RULE",
        description="Test value must be positive",
        condition="value > 0",
        severity="warning",
        category="test",
    )
    engine.add_rule(custom_rule)
    assert "TEST_RULE" in [r.rule_id for r in engine.get_rules()], "Custom rule should be added"

    print(f"     Custom rule added successfully")

    return result


def test_layer2_temporal():
    """Test 5: Layer 2 TemporalTracker — Temporal tracking."""
    from layer2.temporal import TemporalTracker, TemporalRecord

    tracker = TemporalTracker()

    # Add temporal records
    # Use record_observation method (the actual API)
    tracker.record_observation(label="rain", source="weather_report", domain="weather")
    tracker.record_observation(label="flood", source="news", domain="disaster")
    tracker.record_observation(label="flood_relief", source="government", domain="disaster")

    # Query by label
    flood_record = tracker.get_record("flood")
    assert flood_record is not None, "Should find flood record"

    # Query active
    active = tracker.query_active()
    assert len(active) > 0, "Should find active records"

    print(f"  ✅ TemporalTracker: 3 records added")
    print(f"     Flood record found: {flood_record is not None}")
    print(f"     Active records: {len(active)}")

    return tracker


def test_layer3_reasoning():
    """Test 6: Layer 3 ReasoningEngine — Deductive chain building."""
    from layer2.pattern import PatternResult, ReasoningStep
    from layer3.reasoning import ReasoningEngine

    # Create a mock PatternResult (Layer 2 output)
    pattern = PatternResult(
        trigger="Who stole the Snow Plum Pill?",
        steps=[
            ReasoningStep(
                step_type="activation",
                description="Activated nodes: Gu Ilmu, Jang Hangi, Snow Plum Pill, Hefei",
                confidence=0.7,
                evidence_nodes=["Gu_Ilmu", "Jang_Hangi", "Snow_Plum_Pill", "Hefei"],
            ),
            ReasoningStep(
                step_type="composition",
                description="Gu Ilmu and Jang Hangi were both in Hefei",
                confidence=0.65,
                evidence_nodes=["Gu_Ilmu", "Jang_Hangi", "Hefei"],
            ),
            ReasoningStep(
                step_type="anomaly",
                description="No Snow Plum Pill found in market after their visit",
                confidence=0.8,
                evidence_nodes=["Snow_Plum_Pill", "market"],
            ),
        ],
        pattern="Gu Ilmu + Jang Hangi → Hefei → missing pill",
        confidence=0.72,
    )

    # Build deductive chain
    engine = ReasoningEngine()
    chain = engine.build_chain(pattern)

    assert chain is not None, "Chain should not be None"
    assert len(chain.steps) == 5, f"Expected 5 steps, got {len(chain.steps)}"
    assert chain.conclusion, "Chain should have a conclusion"
    assert chain.aggregate_confidence > 0, "Chain should have positive confidence"

    print(f"  ✅ ReasoningEngine: Built chain with {len(chain.steps)} steps")
    print(f"     Conclusion: {chain.conclusion[:100]}...")
    print(f"     Aggregate confidence: {chain.aggregate_confidence:.3f}")
    for i, step in enumerate(chain.steps):
        print(f"     Step {i+1}: [{step.reasoning_type}] conf={step.confidence:.3f}")

    return chain


def test_layer3_deductive_coder():
    """Test 7: Layer 3 DeductiveCoderLayer — RSVS-enhanced code analysis."""
    from layer2.coder_layer import CoderLayer
    from layer3.coder import DeductiveCoderLayer

    code = '''
class DataProcessor:
    def validate(self, data):
        return len(data) > 0

    def transform(self, data):
        return [d.upper() for d in data]

    def save(self, data, path):
        with open(path, 'w') as f:
            f.write(str(data))
'''

    # Layer 2 base analysis
    base_coder = CoderLayer()
    base_result = base_coder.analyze_code(code, language="python")

    # Layer 3 enhanced analysis
    deductive_coder = DeductiveCoderLayer()
    enhanced_result = deductive_coder.analyze_with_rsvs(code, language="python")

    assert enhanced_result is not None, "Enhanced result should not be None"
    assert len(enhanced_result.elements_found) > 0, "Should find elements"

    print(f"  ✅ DeductiveCoderLayer: {len(enhanced_result.elements_found)} elements")
    print(f"     Base confidence: {base_result.confidence:.3f}")
    print(f"     Enhanced confidence: {enhanced_result.confidence:.3f}")

    return enhanced_result


def test_layer3_deductive_policy():
    """Test 8: Layer 3 DeductivePolicyEngine — RSVS-enhanced compliance."""
    from layer2.policy_engine import PolicyEngine
    from layer3.policy import DeductivePolicyEngine

    engine = DeductivePolicyEngine()
    engine.load_tax_rules_indonesia()

    # Standard compliance check
    standard = engine.check_compliance("PT_Test", context={"rate": 0.15})
    print(f"     Standard compliance: {standard['compliant']}")

    # RSVS-enhanced check
    enhanced = engine.check_with_rsvs_policy("PT_Test")
    assert "adjusted_confidence" in enhanced, "Should have adjusted_confidence"
    assert "trust_weight" in enhanced, "Should have trust_weight"

    print(f"  ✅ DeductivePolicyEngine: RSVS-enhanced compliance")
    print(f"     Trust weight: {enhanced['trust_weight']:.2f}")
    print(f"     Adjusted confidence: {enhanced['adjusted_confidence']:.3f}")

    return enhanced


def test_full_pipeline():
    """Test 9: Full AamPipeline — End-to-end without diffusion model."""
    from pipeline import AamPipeline

    pipeline = AamPipeline(use_llm=False, language="en")

    # Simple factual question
    result = pipeline.ask("What is rain?")

    assert result is not None, "Pipeline should return a result"
    assert result.answer, "Pipeline should produce an answer"
    assert result.confidence >= 0, "Confidence should be non-negative"

    print(f"  ✅ Full Pipeline: Got answer (confidence={result.confidence:.3f})")
    print(f"     Answer preview: {result.answer[:150]}...")
    print(f"     Evidence items: {len(result.evidence_chain)}")
    print(f"     Reasoning steps: {len(result.reasoning_chain)}")
    print(f"     Errors: {len(result.errors)}")
    if result.appraise_warning:
        print(f"     Appraise warning: {result.appraise_warning[:80]}")

    return result


def test_pipeline_with_ingestion():
    """Test 10: Pipeline with knowledge ingestion — the real test."""
    from pipeline import AamPipeline

    pipeline = AamPipeline(use_llm=False, language="en")

    # Ingest some knowledge first
    pipeline.ask("Rain causes floods. Floods damage crops.")
    pipeline.ask("Crops are food. Food is important for survival.")

    # Now ask a reasoning question
    result = pipeline.ask("What happens when it rains heavily?")

    assert result is not None, "Should return a result"
    assert result.answer, "Should produce an answer"

    print(f"  ✅ Pipeline with ingestion: Got answer (confidence={result.confidence:.3f})")
    print(f"     Answer preview: {result.answer[:200]}...")

    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    """Run all end-to-end tests."""
    print("=" * 70)
    print("AAM END-TO-END TEST — Buktikan Pikiran Dulu, Baru Latih Tubuh")
    print("=" * 70)
    print()

    tests = [
        ("Layer 0: Text Abstraction", test_layer0_abstraction),
        ("Adapter: PerceptualObservation → RSVS", test_adapter),
        ("Layer 2: Coder Layer", test_layer2_coder_layer),
        ("Layer 2: Policy Engine", test_layer2_policy_engine),
        ("Layer 2: Temporal Tracker", test_layer2_temporal),
        ("Layer 3: Reasoning Engine", test_layer3_reasoning),
        ("Layer 3: Deductive Coder", test_layer3_deductive_coder),
        ("Layer 3: Deductive Policy", test_layer3_deductive_policy),
        ("Full Pipeline: Ask", test_full_pipeline),
        ("Full Pipeline: With Ingestion", test_pipeline_with_ingestion),
    ]

    passed = 0
    failed = 0
    results = []

    for name, test_fn in tests:
        print(f"▶ Test: {name}")
        start = time.time()
        try:
            result = test_fn()
            elapsed = time.time() - start
            passed += 1
            results.append((name, "PASS", elapsed, None))
            print(f"  ⏱ {elapsed:.2f}s")
        except Exception as exc:
            elapsed = time.time() - start
            failed += 1
            results.append((name, "FAIL", elapsed, str(exc)))
            print(f"  ❌ FAILED: {exc}")
            traceback.print_exc()
        print()

    # Summary
    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 70)
    print()

    for name, status, elapsed, error in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name} ({elapsed:.2f}s)")
        if error:
            print(f"     Error: {error[:80]}")

    print()

    if failed == 0:
        print("🎉 ALL TESTS PASSED — Pikiran terbukti bekerja end-to-end!")
        print("   Sekarang latih tubuh (Diffusion LLM) ada fondasi yang solid.")
    else:
        print(f"⚠️  {failed} test(s) failed — ada gap yang perlu diperbaiki.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
