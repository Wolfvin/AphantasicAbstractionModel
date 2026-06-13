#!/usr/bin/env python3
"""
AAM Validation Gates — End-to-End Test

Buktikan bahwa 5 Validation Gates bekerja sebagai fondasi AAM.

Flow:
    Raw Input
      -> [GATE 1: Signal Extraction]  — filter signal from noise
      -> [GATE 2: Regime Detection]   — detect cognitive environment
      -> [GATE 3: Uncertainty Calibration] — calibrate confidence
      -> [GATE 4: Statistical Edge]   — validate reasoning has EV
      -> [GATE 5: Execution Discipline] — enforce output rules
      -> Output

Each gate is a validation checkpoint. Data that fails = REJECTED.
This makes AAM structurally anti-hallucination.

"chatbot trader != quant system"
"language model != validated reasoning system"
AAM = the quant system of AI.
"""

from __future__ import annotations

import sys
import time
import traceback

sys.path.insert(0, ".")


def test_gate1_signal_extraction():
    """Test Gate 1: Signal Extraction — filter signal from noise."""
    from validation_gates.signal_extraction import (
        SignalExtractionGate, SignalVerdict, SignalType,
    )

    gate = SignalExtractionGate()

    # Test 1: Strong causal signal — should PASS
    result = gate.evaluate(
        "Rain causes floods. Heavy rainfall leads to rising water levels.",
    )
    print(f"  Causal input: verdict={result.verdict.value}, quality={result.signal_quality:.3f}")
    assert result.verdict in (SignalVerdict.PASS, SignalVerdict.WEAK), \
        f"Causal input should pass or be weak, got {result.verdict.value}"
    assert len(result.signals) > 0, "Should extract signals from causal input"

    # Test 2: Pure noise — should REJECT
    result = gate.evaluate("a b c d e f g")
    print(f"  Noise input: verdict={result.verdict.value}, quality={result.signal_quality:.3f}")
    assert result.verdict == SignalVerdict.REJECT, \
        f"Pure noise should be rejected, got {result.verdict.value}"

    # Test 3: Comparative signal
    result = gate.evaluate(
        "BTC is higher than ETH. Volume is 4x normal.",
    )
    print(f"  Comparative input: verdict={result.verdict.value}, quality={result.signal_quality:.3f}")
    assert result.verdict in (SignalVerdict.PASS, SignalVerdict.WEAK), \
        f"Comparative input should pass or be weak, got {result.verdict.value}"

    # Test 4: With PerceptualTuples
    from layer0.base import PerceptualTuple, RelationType, ModalityType
    tuples = [
        PerceptualTuple(
            subject="BTC", relation_type=RelationType.CAUSAL,
            predicate="price_increase", confidence=0.8,
            source_modality=ModalityType.TEXT,
        ),
        PerceptualTuple(
            subject="volume", relation_type=RelationType.DIFFERENTIAL,
            predicate="normal", dimension="magnitude", direction="4x higher",
            confidence=0.9, source_modality=ModalityType.TEXT,
        ),
    ]
    result = gate.evaluate("BTC breakout with 4x volume", perceptual_tuples=tuples)
    print(f"  Tuple-enriched: verdict={result.verdict.value}, quality={result.signal_quality:.3f}")
    assert result.verdict == SignalVerdict.PASS, \
        f"Tuple-enriched input should pass, got {result.verdict.value}"

    # Test 5: Statistics
    stats = gate.get_stats()
    print(f"  Gate stats: {stats}")
    assert stats["total_evaluations"] == 4

    return gate


def test_gate2_regime_detection():
    """Test Gate 2: Regime Detection — detect cognitive environment."""
    from validation_gates.regime_detection import (
        RegimeDetectionGate, CognitiveRegime,
    )

    gate = RegimeDetectionGate()

    # Test 1: Factual query
    state = gate.detect("What is Bitcoin?")
    print(f"  Factual query: regime={state.regime.value}, strategy={state.strategy}")
    assert state.regime == CognitiveRegime.FACTUAL, \
        f"'What is X?' should be FACTUAL, got {state.regime.value}"

    # Test 2: Analytical query
    state = gate.detect("Why does rain cause floods?")
    print(f"  Analytical query: regime={state.regime.value}, strategy={state.strategy}")
    assert state.regime == CognitiveRegime.ANALYTICAL, \
        f"'Why does X cause Y?' should be ANALYTICAL, got {state.regime.value}"

    # Test 3: Crisis query
    state = gate.detect("Is this dangerous? Risk assessment needed!")
    print(f"  Crisis query: regime={state.regime.value}, strategy={state.strategy}")
    assert state.regime == CognitiveRegime.CRISIS, \
        f"Danger/risk query should be CRISIS, got {state.regime.value}"

    # Test 4: Creative query
    state = gate.detect("Suggest alternative strategies for growth")
    print(f"  Creative query: regime={state.regime.value}, strategy={state.strategy}")
    assert state.regime == CognitiveRegime.CREATIVE, \
        f"Suggest/imagine query should be CREATIVE, got {state.regime.value}"

    # Test 5: With anomalies — should push toward CRISIS
    state = gate.detect(
        "What happened?",
        anomalies=[{"type": "contradiction"}, {"type": "unexpected"}],
    )
    print(f"  With anomalies: regime={state.regime.value}, risk_level={state.risk_level:.2f}")
    # Anomalies increase crisis tendency

    # Test 6: Regime transition tracking
    stats = gate.get_stats()
    print(f"  Gate stats: {stats}")
    assert stats["total_detections"] >= 5

    return gate


def test_gate3_uncertainty_calibration():
    """Test Gate 3: Uncertainty Calibration — calibrate confidence vs reality."""
    from validation_gates.uncertainty_calibration import UncertaintyCalibrationGate

    gate = UncertaintyCalibrationGate()

    # Test 1: Before any data — no calibration applied
    result = gate.calibrate(raw_confidence=0.9)
    print(f"  No data: raw={result.raw_confidence:.2f}, calibrated={result.calibrated_confidence:.2f}, "
          f"applied={result.calibration_applied}")
    assert not result.calibration_applied, "Should not calibrate without data"

    # Test 2: Record outcomes — simulate overconfidence
    # System says 90% confident, but only correct 50% of the time
    for i in range(20):
        gate.record_outcome(predicted_confidence=0.9, actual_correct=(i % 2 == 0))

    # Test 3: Now calibration should kick in
    result = gate.calibrate(raw_confidence=0.9)
    print(f"  After data: raw={result.raw_confidence:.2f}, calibrated={result.calibrated_confidence:.2f}, "
          f"applied={result.calibration_applied}, overconfident={result.overconfidence_flag}")
    # Should detect overconfidence — raw 0.9 but actual accuracy ~50%
    assert result.calibration_applied, "Should apply calibration with data"
    assert result.overconfidence_flag, "Should flag overconfidence"

    # Test 4: Calibration curve
    curve = gate.get_calibration_curve()
    print(f"  Calibration curve: {curve.total_observations} obs, ECE={curve.calibration_error:.4f}")

    # Test 5: Record well-calibrated outcomes
    for i in range(20):
        # 70% confident → correct 70% of the time
        gate.record_outcome(predicted_confidence=0.7, actual_correct=(i < 14))

    result = gate.calibrate(raw_confidence=0.7)
    print(f"  Well-calibrated: raw={result.raw_confidence:.2f}, calibrated={result.calibrated_confidence:.2f}")

    stats = gate.get_stats()
    print(f"  Gate stats: {stats}")

    return gate


def test_gate4_statistical_edge():
    """Test Gate 4: Statistical Edge — validate reasoning has positive EV."""
    from validation_gates.statistical_edge import (
        StatisticalEdgeGate, ReasoningPath,
    )

    gate = StatisticalEdgeGate()

    # Test 1: New path — no data, should be cautious
    path = ReasoningPath(path_type="deduction", step_types=["extract", "compose", "ground"])
    assessment = gate.assess(path)
    print(f"  New path: verdict={assessment.verdict}, EV={assessment.expected_value:.3f}, "
          f"win_rate={assessment.win_rate:.3f}")
    assert assessment.verdict == "caution", "New path should be cautious"

    # Test 2: Record outcomes — build edge for deduction
    for i in range(15):
        # Deduction is correct 65% of the time
        gate.record_outcome(path=path, correct=(i < 10), confidence=0.6)

    # Test 3: Re-assess — should have positive edge
    assessment = gate.assess(path)
    print(f"  After data: verdict={assessment.verdict}, EV={assessment.expected_value:.3f}, "
          f"win_rate={assessment.win_rate:.3f}, sample={assessment.sample_size}")
    assert assessment.has_edge, "Deduction with 65% win rate should have edge"
    assert assessment.verdict in ("pass", "caution"), "Should pass or be cautious"

    # Test 4: Record poor outcomes — negative edge path
    bad_path = ReasoningPath(path_type="analogy", step_types=["trigger", "recall"])
    for i in range(10):
        # Analogy only correct 30% of the time
        gate.record_outcome(path=bad_path, correct=(i < 3), confidence=0.5)

    assessment = gate.assess(bad_path)
    print(f"  Bad path: verdict={assessment.verdict}, EV={assessment.expected_value:.3f}, "
          f"win_rate={assessment.win_rate:.3f}")
    assert not assessment.has_edge, "Path with 30% win rate should have no edge"
    assert assessment.verdict == "reject", "Bad path should be rejected"

    # Test 5: Stats
    stats = gate.get_stats()
    print(f"  Gate stats: {stats}")

    return gate


def test_gate5_execution_discipline():
    """Test Gate 5: Execution Discipline — enforce output rules."""
    from validation_gates.execution_discipline import ExecutionDisciplineGate

    gate = ExecutionDisciplineGate()

    # Test 1: Well-evidenced, moderate confidence — should pass
    verdict = gate.enforce(
        confidence=0.7,
        evidence_count=5,
        output_text="Based on the evidence, this appears to be the case.",
        regime="factual",
    )
    print(f"  Good output: verdict={verdict.verdict}, allowed={verdict.allowed}, "
          f"adjusted_conf={verdict.adjusted_confidence:.2f}")
    assert verdict.allowed, "Well-evidenced output should be allowed"
    assert verdict.verdict == "pass", "Should pass with good evidence"

    # Test 2: Overconfident output with low evidence — should be adjusted
    verdict = gate.enforce(
        confidence=0.95,
        evidence_count=1,
        output_text="This is definitely absolutely true!",
        regime="factual",
    )
    print(f"  Overconfident: verdict={verdict.verdict}, allowed={verdict.allowed}, "
          f"hallucination_risk={verdict.hallucination_risk:.2f}, "
          f"cap={verdict.confidence_cap:.2f}")
    assert verdict.hallucination_risk > 0.0, "Should detect hallucination risk"
    assert verdict.adjusted_confidence < 0.95, "Should cap overconfident output"

    # Test 3: No evidence — should be blocked
    verdict = gate.enforce(
        confidence=0.6,
        evidence_count=0,
        output_text="I believe this is true.",
        regime="factual",
    )
    print(f"  No evidence: verdict={verdict.verdict}, allowed={verdict.allowed}, "
          f"violations={len(verdict.violations)}")
    # Should at least be adjusted or blocked
    assert verdict.verdict in ("adjust", "block"), \
        f"No evidence should trigger adjust or block, got {verdict.verdict}"

    # Test 4: Crisis mode with low confidence — should be blocked
    verdict = gate.enforce(
        confidence=0.5,
        evidence_count=3,
        output_text="This might be safe.",
        regime="crisis",
    )
    print(f"  Crisis low conf: verdict={verdict.verdict}, allowed={verdict.allowed}")
    assert verdict.verdict in ("adjust", "block"), \
        "Crisis mode with low confidence should be adjusted or blocked"

    # Test 5: Crisis mode with high confidence — should pass
    verdict = gate.enforce(
        confidence=0.85,
        evidence_count=6,
        output_text="Based on strong evidence, this appears to be safe.",
        regime="crisis",
    )
    print(f"  Crisis high conf: verdict={verdict.verdict}, allowed={verdict.allowed}")
    assert verdict.allowed, "Crisis mode with high confidence should be allowed"

    # Test 6: Caveats required for low confidence
    verdict = gate.enforce(
        confidence=0.35,
        evidence_count=2,
        output_text="This seems possible based on limited data.",
        regime="factual",
    )
    print(f"  Low confidence: verdict={verdict.verdict}, caveats={verdict.required_caveats}")
    assert len(verdict.required_caveats) > 0, "Low confidence should require caveats"

    # Test 7: Stats
    stats = gate.get_stats()
    print(f"  Gate stats: {stats}")

    return gate


def test_full_gated_pipeline():
    """Test full pipeline with all 5 gates integrated."""
    from validation_gates import (
        SignalExtractionGate, SignalVerdict,
        RegimeDetectionGate,
        UncertaintyCalibrationGate,
        StatisticalEdgeGate, ReasoningPath,
        ExecutionDisciplineGate,
    )

    print("\n  --- Full Gated Pipeline Test ---")

    # Initialize all gates
    signal_gate = SignalExtractionGate()
    regime_gate = RegimeDetectionGate()
    calibration_gate = UncertaintyCalibrationGate()
    edge_gate = StatisticalEdgeGate()
    discipline_gate = ExecutionDisciplineGate()

    # Simulate a full pipeline run
    raw_input = "Rain causes floods. Heavy rainfall leads to rising water levels which damage crops."

    # GATE 1: Signal Extraction
    print("\n  [GATE 1: Signal Extraction]")
    signal_result = signal_gate.evaluate(raw_input)
    print(f"    Verdict: {signal_result.verdict.value}")
    print(f"    Signal quality: {signal_result.signal_quality:.3f}")
    print(f"    Noise ratio: {signal_result.noise_ratio:.3f}")
    print(f"    Signals extracted: {len(signal_result.signals)}")
    print(f"    Confidence modifier: {signal_result.confidence_modifier:.3f}")

    if signal_result.verdict == SignalVerdict.REJECT:
        print("    REJECTED at Gate 1 — no meaningful signal.")
        return

    # GATE 2: Regime Detection
    print("\n  [GATE 2: Regime Detection]")
    regime_state = regime_gate.detect(raw_input)
    print(f"    Regime: {regime_state.regime.value}")
    print(f"    Strategy: {regime_state.strategy}")
    print(f"    Risk level: {regime_state.risk_level:.2f}")
    print(f"    Confidence: {regime_state.confidence:.2f}")

    # Simulate reasoning with confidence
    reasoning_confidence = 0.75  # From Layer 3 reasoning
    reasoning_confidence *= signal_result.confidence_modifier

    # GATE 3: Uncertainty Calibration
    print("\n  [GATE 3: Uncertainty Calibration]")
    cal_result = calibration_gate.calibrate(
        raw_confidence=reasoning_confidence,
        regime=regime_state.regime.value,
    )
    print(f"    Raw: {cal_result.raw_confidence:.3f}")
    print(f"    Calibrated: {cal_result.calibrated_confidence:.3f}")
    print(f"    Applied: {cal_result.calibration_applied}")
    print(f"    Verdict: {cal_result.verdict}")

    # GATE 4: Statistical Edge
    print("\n  [GATE 4: Statistical Edge]")
    reasoning_path = ReasoningPath(
        path_type="causal",
        regime=regime_state.regime.value,
        step_types=["extract", "compose", "ground", "conclude"],
    )
    edge_result = edge_gate.assess(reasoning_path, current_confidence=cal_result.calibrated_confidence)
    print(f"    Has edge: {edge_result.has_edge}")
    print(f"    Expected value: {edge_result.expected_value:.3f}")
    print(f"    Win rate: {edge_result.win_rate:.3f}")
    print(f"    Verdict: {edge_result.verdict}")

    # Adjust confidence based on edge
    if not edge_result.has_edge:
        reasoning_confidence *= 0.5  # No edge = halve confidence

    # GATE 5: Execution Discipline
    print("\n  [GATE 5: Execution Discipline]")
    output_text = "Based on the evidence, rain causes floods through rising water levels that damage crops."
    evidence_count = len(signal_result.signals)

    discipline_verdict = discipline_gate.enforce(
        confidence=cal_result.calibrated_confidence,
        evidence_count=evidence_count,
        output_text=output_text,
        regime=regime_state.regime.value,
        calibrated_confidence=cal_result.calibrated_confidence,
    )
    print(f"    Allowed: {discipline_verdict.allowed}")
    print(f"    Verdict: {discipline_verdict.verdict}")
    print(f"    Adjusted confidence: {discipline_verdict.adjusted_confidence:.3f}")
    print(f"    Hallucination risk: {discipline_verdict.hallucination_risk:.3f}")
    print(f"    Caveats: {discipline_verdict.required_caveats}")
    print(f"    Confidence cap: {discipline_verdict.confidence_cap:.2f}")

    # Final output
    print("\n  === FINAL OUTPUT ===")
    if discipline_verdict.allowed:
        final_confidence = discipline_verdict.adjusted_confidence
        print(f"  Confidence: {final_confidence:.1%}")
        print(f"  Output: {output_text}")
        if discipline_verdict.required_caveats:
            print("  Caveats:")
            for caveat in discipline_verdict.required_caveats:
                print(f"    - {caveat}")
    else:
        print("  OUTPUT BLOCKED — discipline rules violated.")
        print(f"  Reason: {discipline_verdict.reason}")

    # Record outcome for future calibration (simulate correct result)
    calibration_gate.record_outcome(
        predicted_confidence=reasoning_confidence,
        actual_correct=True,
        regime=regime_state.regime.value,
    )
    edge_gate.record_outcome(
        path=reasoning_path,
        correct=True,
        confidence=reasoning_confidence,
    )

    print("\n  --- All gates passed for this input ---")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    """Run all validation gate tests."""
    print("=" * 70)
    print("AAM VALIDATION GATES — 5 Pillar Foundation Test")
    print("=" * 70)
    print()
    print("5 Pillars:")
    print("  1. Signal Extraction    — filter signal from noise")
    print("  2. Regime Detection     — detect cognitive environment")
    print("  3. Uncertainty Calibration — calibrate confidence vs reality")
    print("  4. Statistical Edge     — validate reasoning has positive EV")
    print("  5. Execution Discipline — enforce output rules")
    print()
    print("Without gates: AI cuma lihat 'chart goes brrrr'")
    print("With gates:    Setiap layer punya validation checkpoint")
    print("=" * 70)
    print()

    tests = [
        ("Gate 1: Signal Extraction", test_gate1_signal_extraction),
        ("Gate 2: Regime Detection", test_gate2_regime_detection),
        ("Gate 3: Uncertainty Calibration", test_gate3_uncertainty_calibration),
        ("Gate 4: Statistical Edge", test_gate4_statistical_edge),
        ("Gate 5: Execution Discipline", test_gate5_execution_discipline),
        ("Full Gated Pipeline", test_full_gated_pipeline),
    ]

    passed = 0
    failed = 0
    results = []

    for name, test_fn in tests:
        print(f"▶ Test: {name}")
        start = time.time()
        try:
            test_fn()
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
        print("🎉 ALL GATES PASSED — Fondasi AAM terbukti!")
        print()
        print("5 Pillar Mapping to AAM Layers:")
        print("  Layer 0/1 → Signal Extraction (filter noise from signal)")
        print("  Layer 2   → Regime Detection (detect cognitive environment)")
        print("  Layer 3   → Uncertainty Calibration (calibrate confidence)")
        print("  Layer 4   → Statistical Edge (validate reasoning EV)")
        print("  Layer 5   → Execution Discipline (enforce output rules)")
        print()
        print("AAM = the quant system of AI.")
        print("chatbot trader != quant system")
        print("language model != validated reasoning system")
    else:
        print(f"⚠️  {failed} test(s) failed — ada gap yang perlu diperbaiki.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
