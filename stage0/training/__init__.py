"""
AAM Ingest Training System
===========================

This module implements the AAM training feedback loop:
    Ingest → DetectGaps → AskUser → Correction → EnrichComposition → GovernBeliefs → Persist

Unlike simple in-memory training, this system:
1. Persists the full RSVS knowledge graph to disk (not just in-memory)
2. Uses the question engine to generate varied questions from detected gaps
3. Applies human corrections as HumanAssertion (Stable/Grounded immediately)
4. Mines patterns from accumulated corrections
5. Tracks learning progress across sessions

Usage:
    from stage0.training import AAMTrainer

    trainer = AAMTrainer(persist_dir="training_output")
    trainer.ingest("Budi menjual barang ke saya")
    questions = trainer.detect_and_question()
    trainer.correct(questions[0], answer="saya", role="Arg2Recipient")
    trainer.persist()
"""

from .types import (Composition, CompositionMember, KnowledgeGap, TrainingRecord,
                    PatternObservation, GeneratedQuestion, CorrectionResult)
from .ingest_trainer import AAMTrainer
from .question_engine import QuestionEngine
from .correction_handler import CorrectionHandler
from .persistence import TrainingPersistence
from .corpora import TrainingCorpus

__all__ = [
    "AAMTrainer",
    "QuestionEngine",
    "CorrectionHandler",
    "TrainingPersistence",
    "TrainingCorpus",
]

__version__ = "1.0.0"
