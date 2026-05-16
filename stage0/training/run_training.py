#!/usr/bin/env python3
"""
AAM Training Runner — Execute the full AAM ingest training session.

This script:
1. Creates an AAM Trainer
2. Ingests the progressive curriculum
3. Detects gaps and generates questions
4. Applies corrections (as the "parent")
5. Persists everything to RSVS format
6. Reports the final training status

Usage:
    python -m stage0.training.run_training
    python -m stage0.training.run_training --full-corpus
    python -m stage0.training.run_training --interactive
"""

import os
import sys
import json
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from stage0.training.ingest_trainer import AAMTrainer
from stage0.training.question_engine import QuestionEngine
from stage0.training.correction_handler import CorrectionHandler
from stage0.training.corpora import TrainingCorpus


# ────────────────────────────────────────────────────────────────────
# Pre-defined corrections — The "Parent" Knowledge
# ────────────────────────────────────────────────────────────────────

# These are corrections that a knowledgeable "parent" would provide
# when the system detects gaps or makes mistakes.
PARENT_CORRECTIONS = {
    # Recipient corrections — after "ke" or "kepada" in transaction verbs
    "menjual": {
        "Arg2Recipient": "pembeli",  # After "menjual ke X", X is the Recipient (pembeli)
    },
    "membeli": {
        "Arg2Recipient": "penjual",  # After "membeli dari X", X is the source
    },
    "memberikan": {
        "Arg2Recipient": "penerima",  # After "memberikan kepada X", X is the Recipient
    },
    "mengirim": {
        "Arg2Recipient": "penerima",  # After "mengirim ke X", X is the Recipient
    },
    "mengadakan": {
        "Purpose": "pelatihan",  # Common purpose for mengadakan
    },
    "menabung": {
        "Purpose": "masa depan",
    },
    "membangun": {
        "Purpose": "kemajuan",
    },
    # Cause corrections — common cause patterns
    "bangkrut": {
        "Cause": "kerugian finansial",
    },
    "sakit": {
        "Cause": "kesehatan menurun",
    },
    "terlambat": {
        "Cause": "kendala perjalanan",
    },
}


def run_training_session(persist_dir: str, full_corpus: bool = False,
                          interactive: bool = False):
    """
    Run the full AAM ingest training session.

    This is where the "parent" teaches the "child" AAM:
    1. Ingest the progressive curriculum
    2. Detect gaps and ask questions
    3. Apply corrections based on parent knowledge
    4. Persist to RSVS format
    5. Report status
    """
    print("=" * 70)
    print("AAM INGEST TRAINING SESSION")
    print("=" * 70)
    print()

    # Create trainer
    trainer = AAMTrainer(persist_dir=persist_dir, verbose=True)
    corpus = TrainingCorpus()

    # ── Phase 1: Ingest progressive curriculum ──────────────────────
    print("\n--- Phase 1: Ingesting Progressive Curriculum ---")

    if full_corpus:
        sentences = corpus.get_all()
        print(f"  Full corpus: {len(sentences)} sentences")
    else:
        sentences = corpus.get_progressive_curriculum()
        print(f"  Progressive curriculum: {len(sentences)} sentences")

    # Ingest in batches of 5 for readability
    batch_size = 5
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        print(f"\n  Batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}:")
        for sentence in batch:
            trainer.ingest(sentence)

    trainer.print_status()

    # ── Phase 2: Detect gaps and generate questions ─────────────────
    print("\n--- Phase 2: Detecting Gaps and Generating Questions ---")

    questions = trainer.detect_and_question()
    print(f"\n  Generated {len(questions)} questions:")

    for i, q in enumerate(questions[:30], 1):  # Show first 30
        print(f"  {i}. [{q.question_type}] {q.question_text}")
        if q.context_hint:
            print(f"     Context: {q.context_hint}")
        if q.variations:
            print(f"     Variations: {q.variations[0]}")

    if len(questions) > 30:
        print(f"  ... and {len(questions) - 30} more questions")

    # ── Phase 3: Apply corrections (Parent teaches Child) ───────────
    print("\n--- Phase 3: Parent Corrections ---")

    corrections_applied = 0
    contradictions_found = 0
    patterns_promoted = 0

    for q in questions:
        if q.target_composition_id is None:
            continue

        comp = trainer.compositions.get(q.target_composition_id)
        if not comp:
            continue

        # Get the predicate
        pred = comp.member_with_role("Predicate")
        if not pred:
            continue

        # Check if we have a parent correction for this predicate + role
        predicate_label = pred.label
        role = q.target_role

        parent_correction = PARENT_CORRECTIONS.get(predicate_label, {}).get(role)

        if parent_correction and not comp.has_member_with_role(role):
            # Apply the parent's correction
            result = trainer.correct(q, answer=parent_correction, role=role)
            if result.success:
                corrections_applied += 1
                if result.contradiction_detected:
                    contradictions_found += 1
                if result.pattern_promoted:
                    patterns_promoted += 1

    # Also apply some specific corrections to demonstrate the feedback loop
    # For example: correcting "ke" ambiguities
    specific_corrections = [
        # In "menjual ... ke X", "ke" marks Recipient not Location
        # We'll correct any Location that should be Arg2Recipient
    ]

    # Find compositions with Location that should be Arg2Recipient
    for comp in trainer.compositions.values():
        pred = comp.member_with_role("Predicate")
        location = comp.member_with_role("Location")
        recipient = comp.member_with_role("Arg2Recipient")

        if pred and location and not recipient:
            # Transaction verbs: "ke" after sell/give/send = Recipient, not Location
            transaction_predicates = {"menjual", "mengirim", "memberikan", "memberi",
                                      "menyuruh", "mengirimkan", "memberikan"}
            if pred.label in transaction_predicates:
                # This Location should be Arg2Recipient!
                result = trainer.correct_direct(
                    composition_id=comp.id,
                    role="Arg2Recipient",
                    value=location.label,
                )
                if result.success:
                    corrections_applied += 1
                    trainer._log(f"  Corrected Location→Arg2Recipient: {pred.label} ke {location.label}")

    print(f"\n  Corrections applied: {corrections_applied}")
    print(f"  Contradictions found: {contradictions_found}")
    print(f"  Patterns promoted: {patterns_promoted}")

    # ── Phase 4: Re-ingest with learned patterns ────────────────────
    print("\n--- Phase 4: Re-ingest with Learned Patterns ---")

    # Now that the system has learned some patterns, re-ingest some
    # sentences to see if it fills gaps automatically
    test_sentences = [
        "Toko menjual sepatu ke pelanggan",
        "Kantor mengirim dokumen ke cabang",
        "Ibu memberikan baju kepada adik",
        "Dia membeli mobil dari dealer karena butuh",
        "Perusahaan mengadakan training agar karyawan kompeten",
    ]

    for sentence in test_sentences:
        trainer.ingest(sentence)

    # Check if fewer gaps were detected (system is learning!)
    new_questions = trainer.detect_and_question()
    print(f"\n  After re-ingest: {len(new_questions)} remaining questions (should be fewer)")

    # ── Phase 5: Persist to RSVS format ─────────────────────────────
    print("\n--- Phase 5: Persisting to RSVS Format ---")

    trainer.persist()
    print(f"  Saved to: {persist_dir}")
    print(f"  Files created:")
    for filename in ["graph.json", "inquiry_memory.json", "patterns.json",
                      "records.json", "metadata.json", "summary.txt"]:
        filepath = os.path.join(persist_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"    {filename}: {size:,} bytes")

    # ── Phase 6: Final Status ───────────────────────────────────────
    print("\n--- Phase 6: Final Training Status ---")

    trainer.print_status()

    print("\n  Recent Compositions:")
    trainer.print_compositions(limit=15)

    print("\n  Discovered Patterns:")
    trainer.print_patterns(min_count=2)

    # ── Phase 7: Interactive Mode ────────────────────────────────────
    if interactive:
        print("\n--- Interactive Mode ---")
        print("Type 'quit' to exit, 'status' for status, 'patterns' for patterns")
        print("Or type any text to ingest it:\n")

        while True:
            try:
                user_input = input("AAM> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            elif user_input == "quit":
                break
            elif user_input == "status":
                trainer.print_status()
            elif user_input == "patterns":
                trainer.print_patterns(min_count=1)
            elif user_input == "questions":
                qs = trainer.detect_and_question()
                for i, q in enumerate(qs[:10], 1):
                    print(f"  {i}. {q.question_text}")
            elif user_input == "comps":
                trainer.print_compositions(limit=20)
            elif user_input == "save":
                trainer.persist()
                print("Saved!")
            else:
                trainer.ingest(user_input)
                # Check for questions
                qs = trainer.detect_and_question()
                if qs:
                    print(f"\n  Questions ({len(qs)}):")
                    for i, q in enumerate(qs[:5], 1):
                        print(f"    {i}. {q.question_text}")
                        answer = input(f"    Answer (or press Enter to skip): ").strip()
                        if answer:
                            result = trainer.correct(q, answer=answer)
                            print(f"    → {result.message}")

                trainer.persist()

    return trainer


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AAM Ingest Training Runner")
    parser.add_argument("--persist-dir", default="training_output",
                        help="Directory to persist training state")
    parser.add_argument("--full-corpus", action="store_true",
                        help="Use the full corpus instead of progressive curriculum")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive mode after training")
    args = parser.parse_args()

    # Resolve persist directory relative to project root
    persist_dir = os.path.join(PROJECT_ROOT, args.persist_dir)

    trainer = run_training_session(
        persist_dir=persist_dir,
        full_corpus=args.full_corpus,
        interactive=args.interactive,
    )

    # Final persist
    trainer.persist()

    print(f"\n{'=' * 70}")
    print("TRAINING SESSION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
