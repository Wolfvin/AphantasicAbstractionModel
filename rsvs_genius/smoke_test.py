#!/usr/bin/env python3
"""
Smoke test for rsvs_genius pipeline — runs in fallback mode (no Rust core needed).

Tests:
1. RsvsBridge fallback mode
2. ContextLayer ingestion and scope filtering
3. SituationLayer chat history and active senses
4. PredictiveEngine prediction and anomaly detection
5. PatternOutput full pipeline
6. GeniusPipeline end-to-end

Run: python -m rsvs_genius.smoke_test
"""

from __future__ import annotations

import sys
import json

# Ensure we can import from the parent directory
sys.path.insert(0, "/home/z/my-project/RSVS")


def test_bridge():
    """Test RsvsBridge — works in both Rust core and fallback mode."""
    from rsvs_genius.rsvs_bridge import RsvsBridge, is_rust_core_available

    bridge = RsvsBridge()
    assert bridge.is_available, "Bridge should be available"
    mode = "Rust core" if bridge.is_rust_core else "fallback"
    print(f"  Bridge mode: {mode}")

    # Test ingest
    stats = bridge.ingest("Snow Plum Pill adalah obat langka dari Mount Hua")
    assert isinstance(stats, dict), f"ingest should return dict, got {type(stats)}"
    print(f"  ✓ Bridge ingest: {stats}")

    # Test query
    result = bridge.query("snow")
    print(f"  ✓ Bridge query: {result}")

    # Test relate
    related = bridge.relate("snow")
    print(f"  ✓ Bridge relate: {related}")

    # Test appraise
    appraisal = bridge.appraise("Snow Plum Pill adalah obat")
    assert isinstance(appraisal, dict), "appraise should return dict"
    print(f"  ✓ Bridge appraise: verdict={appraisal.get('verdict')}")

    # Test structural_similarity
    bridge.ingest("raja adalah laki-laki penguasa kerajaan")
    bridge.ingest("ratu adalah perempuan penguasa kerajaan")
    sim = bridge.structural_similarity("raja", "ratu")
    print(f"  ✓ Bridge structural_similarity: {sim}")

    # Test compose
    node_id = bridge.compose("monarki", [("raja", 0), ("ratu", 0)])
    print(f"  ✓ Bridge compose: node_id={node_id}")

    # Test confidence_map
    cmap = bridge.confidence_map()
    assert isinstance(cmap, dict), "confidence_map should return dict"
    print(f"  ✓ Bridge confidence_map: {len(cmap)} nodes")

    # Test senses
    senses = bridge.senses("raja")
    print(f"  ✓ Bridge senses: {senses}")

    print(f"  ✅ RsvsBridge ({mode} mode) — ALL TESTS PASSED\n")


def test_context_layer():
    """Test ContextLayer."""
    from rsvs_genius.context_layer import ContextLayer

    layer = ContextLayer()
    assert layer.rsvs_available, "ContextLayer should be available"

    # Test ingest
    result = layer.ingest_text("Hefei adalah kota perdagangan penting", source="user_input")
    assert result["success"], "ingest should succeed"
    print(f"  ✓ ContextLayer ingest: success={result['success']}, trust={result['trust']}")

    # Test scope filtering
    layer.set_scope(["academic", "official_doc"])
    result = layer.ingest_text("Data dari media sosial", source="social_media")
    assert not result["in_scope"], "social_media should be out of scope"
    print(f"  ✓ ContextLayer scope filter: in_scope={result['in_scope']}")

    layer.clear_scope()
    result = layer.ingest_text("Data dari media sosial", source="social_media")
    assert result["in_scope"], "should be in scope after clearing"
    print(f"  ✓ ContextLayer scope cleared: in_scope={result['in_scope']}")

    # Test trust scoring
    trust = layer.trust_score("academic")
    assert trust == 0.9, f"academic trust should be 0.9, got {trust}"
    print(f"  ✓ ContextLayer trust: academic={trust}")

    print("  ✅ ContextLayer — ALL TESTS PASSED\n")


def test_situation_layer():
    """Test SituationLayer."""
    from rsvs_genius.situation_layer import SituationLayer

    layer = SituationLayer()
    assert layer.rsvs_available, "SituationLayer should be available"

    # Test add_message
    result = layer.add_message("user", "Ceritakan tentang Snow Plum Pill")
    assert result["success"], "add_message should succeed"
    print(f"  ✓ SituationLayer add_message: success={result['success']}")

    # Test active senses
    senses = layer.get_active_senses()
    assert isinstance(senses, list), "active senses should be a list"
    print(f"  ✓ SituationLayer active_senses: {len(senses)} active")

    # Test get_relevant_context
    relevant = layer.get_relevant_context("Snow Plum Pill")
    assert isinstance(relevant, list), "relevant context should be a list"
    print(f"  ✓ SituationLayer relevant_context: {len(relevant)} results")

    # Test situation summary
    summary = layer.get_situation_summary()
    assert "active_senses" in summary, "summary should have active_senses"
    print(f"  ✓ SituationLayer summary: {summary['message_count']} messages")

    print("  ✅ SituationLayer — ALL TESTS PASSED\n")


def test_predictive_engine():
    """Test PredictiveEngine."""
    from rsvs_genius.predictive_engine import PredictiveEngine

    engine = PredictiveEngine(eta=0.1, anomaly_threshold=0.3)
    assert engine.rsvs_available, "PredictiveEngine should be available"

    # Test predict
    prediction = engine.predict("Snow Plum Pill", context=["obat", "langka"])
    assert prediction.concept == "Snow Plum Pill", "prediction concept should match"
    print(f"  ✓ PredictiveEngine predict: concept={prediction.concept}, conf={prediction.confidence:.3f}")

    # Test observe_and_update
    updates = engine.observe_and_update("Snow Plum Pill dikonsumsi oleh pencuri", source="observation")
    print(f"  ✓ PredictiveEngine observe: {len(updates)} belief updates")

    # Test anomaly detection
    anomalies = engine.detect_anomalies()
    print(f"  ✓ PredictiveEngine anomalies: {len(anomalies)} detected")

    # Test belief history
    beliefs = engine.get_current_beliefs()
    assert isinstance(beliefs, dict), "beliefs should be a dict"
    print(f"  ✓ PredictiveEngine beliefs: {len(beliefs)} tracked concepts")

    print("  ✅ PredictiveEngine — ALL TESTS PASSED\n")


def test_pattern_output():
    """Test PatternOutput."""
    from rsvs_genius.pattern_output import PatternOutput

    output = PatternOutput()
    assert output.rsvs_available, "PatternOutput should be available"

    # Test full pipeline
    result = output.process("Snow Plum Pill", context=["pencurian", "Hefei"])
    assert result.trigger == "Snow Plum Pill", "trigger should match"
    assert len(result.steps) == 6, f"should have 6 steps, got {len(result.steps)}"
    print(f"  ✓ PatternOutput process: {len(result.steps)} steps")
    print(f"    Steps: {[s.step_type for s in result.steps]}")
    print(f"    Confidence: {result.confidence:.3f}")
    print(f"    Pattern: {result.pattern[:80]}..." if len(result.pattern) > 80 else f"    Pattern: {result.pattern}")

    print("  ✅ PatternOutput — ALL TESTS PASSED\n")


def test_pipeline():
    """Test GeniusPipeline end-to-end."""
    from rsvs_genius.pipeline import GeniusPipeline

    pipeline = GeniusPipeline(eta=0.1, anomaly_threshold=0.3)

    # Pre-load some knowledge
    pipeline.ingest("Snow Plum Pill adalah obat langka dari Mount Hua Sect")
    pipeline.ingest("Gyeryong Merchant Guild di Hefei menjual obat-obatan langka")
    pipeline.ingest("Ju Jangmok menghilang pada hari yang sama dengan pencurian")
    pipeline.ingest("Diancang Five Swords punya pasangan dengan success rate tinggi")

    # Ask a question
    response = pipeline.ask("Siapa yang mencuri Snow Plum Pill?")
    assert isinstance(response.answer, str), "answer should be a string"
    assert response.confidence >= 0.0, "confidence should be >= 0"
    print(f"  ✓ Pipeline ask:")
    print(f"    Answer: {response.answer[:100]}..." if len(response.answer) > 100 else f"    Answer: {response.answer}")
    print(f"    Confidence: {response.confidence:.3f}")
    print(f"    Reasoning steps: {len(response.reasoning_chain)}")
    print(f"    Evidence: {len(response.evidence_chain)} items")
    print(f"    Anomalies: {len(response.anomalies)}")
    print(f"    Predictions: {len(response.predictions)}")
    print(f"    Belief updates: {len(response.belief_updates)}")
    print(f"    Metadata: {json.dumps(response.metadata, indent=2)}")

    # Test status
    status = pipeline.get_status()
    print(f"  ✓ Pipeline status: {json.dumps(status, indent=2)}")

    # Test scope filtering
    pipeline.set_scope(["official_doc", "academic"])
    scoped_response = pipeline.ask("Berapa tarif pajak penghasilan?", search_internet=False)
    print(f"  ✓ Pipeline scoped query: confidence={scoped_response.confidence:.3f}")
    pipeline.clear_scope()

    print("  ✅ GeniusPipeline — ALL TESTS PASSED\n")


def main():
    from rsvs_genius.rsvs_bridge import is_rust_core_available
    print("=" * 70)
    print(f"RSVS Genius — Smoke Test ({'Rust Core' if is_rust_core_available() else 'Fallback'} Mode)")
    print("=" * 70)
    print()

    tests = [
        ("1. RsvsBridge", test_bridge),
        ("2. ContextLayer", test_context_layer),
        ("3. SituationLayer", test_situation_layer),
        ("4. PredictiveEngine", test_predictive_engine),
        ("5. PatternOutput", test_pattern_output),
        ("6. GeniusPipeline", test_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"Testing: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
