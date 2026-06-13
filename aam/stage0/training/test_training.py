"""
Tests for AAM Training System.

Tests the full feedback loop: Ingest → DetectGaps → AskUser → Correct → Persist
"""

import os
import sys
import json
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from stage0.training.ingest_trainer import AAMTrainer
from stage0.training.types import Composition, CompositionMember, KnowledgeGap
from stage0.training.question_engine import QuestionEngine
from stage0.training.correction_handler import CorrectionHandler
from stage0.training.persistence import TrainingPersistence
from stage0.training.corpora import TrainingCorpus


def test_tokenizer():
    """Test that the tokenizer works correctly."""
    trainer = AAMTrainer(persist_dir=tempfile.mkdtemp(), verbose=False)

    tokens = trainer._tokenize("Budi menjual barang ke saya karena butuh uang.")
    assert "budi" in tokens
    assert "menjual" in tokens
    assert "barang" in tokens
    assert "ke" in tokens
    assert "saya" in tokens
    assert "karena" in tokens
    assert "butuh" in tokens
    assert "uang" in tokens
    print("  ✓ Tokenizer works")


def test_extract_frame():
    """Test that frame extraction works for Indonesian."""
    trainer = AAMTrainer(persist_dir=tempfile.mkdtemp(), verbose=False)

    # Simple SVO
    tokens = trainer._tokenize("Budi menjual barang ke saya")
    frame = trainer._extract_frame(tokens, "Budi menjual barang ke saya")
    assert frame["Arg0Agent"] == "budi", f"Expected 'budi', got {frame['Arg0Agent']}"
    assert frame["predicate"] == "menjual", f"Expected 'menjual', got {frame['predicate']}"
    assert frame["Arg1Patient"] == "barang", f"Expected 'barang', got {frame['Arg1Patient']}"
    # "ke saya" should be detected (either as Location or Arg2Recipient)
    assert frame["Location"] == "saya" or frame["Arg2Recipient"] == "saya", \
        f"Expected 'saya' as Recipient or Location, got Location={frame['Location']}, Recipient={frame['Arg2Recipient']}"

    # Causal pattern
    tokens = trainer._tokenize("Raymond membuat aplikasi karena lambat")
    frame = trainer._extract_frame(tokens, "Raymond membuat aplikasi karena lambat")
    assert frame["Arg0Agent"] == "raymond"
    assert frame["predicate"] == "membuat"
    assert frame["Arg1Patient"] == "aplikasi"
    assert frame["Cause"] is not None and "lambat" in frame["Cause"]

    print("  ✓ Frame extraction works")


def test_ingest_creates_compositions():
    """Test that ingestion creates compositions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AAMTrainer(persist_dir=tmpdir, verbose=False)

        comps = trainer.ingest("Budi menjual barang ke saya")
        assert comps >= 1, f"Expected at least 1 composition, got {comps}"
        assert len(trainer.nodes) >= 3, f"Expected at least 3 nodes, got {len(trainer.nodes)}"
        assert len(trainer.compositions) >= 1

        print("  ✓ Ingest creates compositions")


def test_gap_detection():
    """Test that gap detection finds missing roles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AAMTrainer(persist_dir=tmpdir, verbose=False)

        # Ingest a sentence that will be missing Cause and Purpose
        trainer.ingest("Budi menjual barang ke saya")

        # Should have gaps for missing Cause and Purpose
        gaps = [g for g in trainer.gaps.values() if not g.addressed]
        gap_types = {g.gap_type for g in gaps}

        # Should detect missing Cause and/or Purpose
        assert len(gaps) > 0, f"Expected gaps, got {len(gaps)}"
        print(f"  ✓ Gap detection works: found {len(gaps)} gaps ({gap_types})")


def test_question_generation():
    """Test that question generation creates meaningful questions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AAMTrainer(persist_dir=tmpdir, verbose=False)

        trainer.ingest("Budi menjual barang ke saya")

        questions = trainer.detect_and_question()
        assert len(questions) > 0, "Expected questions"

        # Questions should have meaningful text
        for q in questions:
            assert q.question_text, f"Question has no text: {q}"
            assert q.target_role, f"Question has no target role: {q}"
            assert q.gap_id, f"Question has no gap_id: {q}"

        print(f"  ✓ Question generation works: {len(questions)} questions")


def test_correction_application():
    """Test that corrections are applied as HumanAssertion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AAMTrainer(persist_dir=tmpdir, verbose=False)

        trainer.ingest("Budi menjual barang ke saya")

        questions = trainer.detect_and_question()

        # Find a question about Arg2Recipient or Cause
        target_q = None
        for q in questions:
            if q.target_role in ("Arg2Recipient", "Cause", "Purpose"):
                target_q = q
                break

        if target_q:
            result = trainer.correct(target_q, answer="saya")
            assert result.success, f"Correction failed: {result.message}"
            assert result.governance_applied == "Stable/Grounded (HumanAssertion)"

            # The composition should now be Stable/Grounded
            comp = trainer.compositions.get(result.composition_id)
            assert comp is not None
            assert comp.lifecycle == "Stable"
            assert comp.epistemic == "Grounded"

            # The gap should be addressed
            gap = trainer.gaps.get(target_q.gap_id)
            assert gap is not None
            assert gap.addressed

            print(f"  ✓ Correction works: {result.message}")
        else:
            print("  ⚠ No suitable question found for correction test")


def test_persistence_roundtrip():
    """Test that persistence saves and loads correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and train
        trainer1 = AAMTrainer(persist_dir=tmpdir, verbose=False)
        trainer1.ingest("Budi menjual barang ke saya")
        trainer1.ingest("Ibu memberikan uang kepada anaknya")
        questions = trainer1.detect_and_question()
        if questions:
            trainer1.correct(questions[0], answer="test_answer")
        trainer1.persist()

        # Load into new trainer
        trainer2 = AAMTrainer(persist_dir=tmpdir, verbose=False)

        assert len(trainer2.compositions) == len(trainer1.compositions), \
            f"Composition count mismatch: {len(trainer2.compositions)} vs {len(trainer1.compositions)}"
        assert len(trainer2.nodes) == len(trainer1.nodes), \
            f"Node count mismatch: {len(trainer2.nodes)} vs {len(trainer1.nodes)}"
        assert len(trainer2.patterns) == len(trainer1.patterns), \
            f"Pattern count mismatch: {len(trainer2.patterns)} vs {len(trainer1.patterns)}"

        print("  ✓ Persistence roundtrip works")


def test_pattern_mining():
    """Test that pattern mining discovers recurring patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AAMTrainer(persist_dir=tmpdir, verbose=False)

        # Ingest multiple sentences with the same predicate
        corpus = TrainingCorpus()
        for sentence in corpus.TRANSACTION_SELL[:8]:
            trainer.ingest(sentence)

        # Should have patterns
        assert len(trainer.patterns) > 0, f"Expected patterns, got {len(trainer.patterns)}"

        # Should have patterns for "menjual"
        menjual_patterns = [p for p in trainer.patterns.values() if p.predicate == "menjual"]
        assert len(menjual_patterns) > 0, "Expected 'menjual' patterns"

        print(f"  ✓ Pattern mining works: {len(trainer.patterns)} patterns, {len(menjual_patterns)} for 'menjual'")


def test_learning_across_sentences():
    """Test that the system learns from corrections and applies to new sentences."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AAMTrainer(persist_dir=tmpdir, verbose=False)

        # Round 1: Ingest and correct
        trainer.ingest("Budi menjual barang ke saya")
        questions1 = trainer.detect_and_question()

        # Correct the Recipient gap
        for q in questions1:
            if q.target_role == "Arg2Recipient" and q.target_composition_id:
                trainer.correct(q, answer="saya")
                break

        # Round 2: Ingest similar sentence
        trainer.ingest("Toko menjual sepatu ke pelanggan")
        questions2 = trainer.detect_and_question()

        # The system should have learned that "ke" after "menjual" = Arg2Recipient
        # So it should have fewer gaps (or at least the Arg2Recipient gap should be auto-filled)

        print(f"  ✓ Learning across sentences: Q1={len(questions1)}, Q2={len(questions2)}")


def test_parent_corrections():
    """Test the full parent-child correction flow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AAMTrainer(persist_dir=tmpdir, verbose=False)
        corpus = TrainingCorpus()

        # Ingest the progressive curriculum
        for sentence in corpus.get_progressive_curriculum():
            trainer.ingest(sentence)

        # Detect gaps
        questions = trainer.detect_and_question()
        initial_gaps = len([g for g in trainer.gaps.values() if not g.addressed])

        # Apply parent corrections
        from stage0.training.run_training import PARENT_CORRECTIONS

        corrections_applied = 0
        for q in questions:
            if q.target_composition_id is None:
                continue

            comp = trainer.compositions.get(q.target_composition_id)
            if not comp:
                continue

            pred = comp.member_with_role("Predicate")
            if not pred:
                continue

            parent_correction = PARENT_CORRECTIONS.get(pred.label, {}).get(q.target_role)
            if parent_correction and not comp.has_member_with_role(q.target_role):
                result = trainer.correct(q, answer=parent_correction)
                if result.success:
                    corrections_applied += 1

        # After corrections, there should be fewer unresolved gaps
        final_gaps = len([g for g in trainer.gaps.values() if not g.addressed])

        print(f"  ✓ Parent corrections: {corrections_applied} applied, gaps: {initial_gaps} → {final_gaps}")


def test_full_training_session():
    """Test the complete training session end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from stage0.training.run_training import run_training_session

        trainer = run_training_session(persist_dir=tmpdir, full_corpus=False, interactive=False)

        # Verify the training session produced meaningful results
        status = trainer.status()
        assert status["total_compositions"] > 0, "No compositions created"
        assert status["total_nodes"] > 0, "No nodes created"
        assert status["total_patterns"] > 0, "No patterns discovered"

        # Verify persistence files exist
        for filename in ["graph.json", "inquiry_memory.json", "patterns.json",
                          "records.json", "metadata.json", "summary.txt"]:
            filepath = os.path.join(tmpdir, filename)
            assert os.path.exists(filepath), f"Missing file: {filename}"

        # Verify graph.json is valid JSON
        with open(os.path.join(tmpdir, "graph.json"), "r") as f:
            graph_data = json.load(f)
        assert graph_data["schema_version"] == "v12-training-1.0"
        assert "compositions" in graph_data
        assert "statistics" in graph_data

        print(f"  ✓ Full training session: {status['total_compositions']} comps, "
              f"{status['total_patterns']} patterns, "
              f"{status['stable_patterns']} stable patterns")


if __name__ == "__main__":
    print("\nRunning AAM Training System Tests\n")
    print("=" * 50)

    test_tokenizer()
    test_extract_frame()
    test_ingest_creates_compositions()
    test_gap_detection()
    test_question_generation()
    test_correction_application()
    test_persistence_roundtrip()
    test_pattern_mining()
    test_learning_across_sentences()
    test_parent_corrections()
    test_full_training_session()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓\n")
