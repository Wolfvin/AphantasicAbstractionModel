# @WHO:   self-ai/src/training/session.py
# @WHAT:  Track satu sesi training — semua soal, jawaban, koreksi, reasoning
# @PART:  self-ai/training
# @ENTRY: TrainingSession()

"""Training Session — data container untuk satu sesi training.

Menyimpan semua informasi yang terjadi selama satu sesi TrainingAgent:
- Questions yang dijalankan beserta jawaban SELF
- Corrections yang diterapkan beserta reasoning
- Benchmark results before/after

Session ini yang di-export ke docs/training_sessions/ via results.py.
"""

from datetime import datetime
from typing import Optional


class QuestionResult:
    """Satu hasil pertanyaan yang dijalankan ke SELF."""

    __slots__ = ('context', 'question', 'answer', 'confidence', 'method', 'timestamp')

    def __init__(self, context: str, question: str, answer, confidence: float,
                 method: str, timestamp: Optional[datetime] = None):
        self.context = context
        self.question = question
        self.answer = answer
        self.confidence = confidence
        self.method = method
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict:
        return {
            'context': self.context,
            'question': self.question,
            'answer': str(self.answer) if self.answer is not None else None,
            'confidence': round(self.confidence, 3),
            'method': self.method,
            'timestamp': self.timestamp.isoformat(),
        }


class CorrectionRecord:
    """Satu koreksi yang diterapkan ke SELF."""

    __slots__ = ('question', 'wrong_answer', 'correct_answer', 'reasoning',
                 'pattern_key', 'timestamp')

    def __init__(self, question: str, wrong_answer, correct_answer: str,
                 reasoning: str, pattern_key: str,
                 timestamp: Optional[datetime] = None):
        self.question = question
        self.wrong_answer = wrong_answer
        self.correct_answer = correct_answer
        self.reasoning = reasoning
        self.pattern_key = pattern_key
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict:
        return {
            'question': self.question,
            'wrong_answer': str(self.wrong_answer) if self.wrong_answer is not None else None,
            'correct_answer': self.correct_answer,
            'reasoning': self.reasoning,
            'pattern_key': self.pattern_key,
            'timestamp': self.timestamp.isoformat(),
        }


class TrainingSession:
    """Track satu sesi training — semua soal, jawaban, koreksi, reasoning.

    Data container murni — tidak ada logic selain tracking dan summary.
    Export logic ada di results.py.
    """

    def __init__(self):
        self.started_at = datetime.now()
        self.ended_at: Optional[datetime] = None
        self.questions: list = []
        self.corrections: list = []
        self.benchmark_before: Optional[dict] = None
        self.benchmark_after: Optional[dict] = None

    def add_question(self, context: str, question: str, answer, confidence: float,
                     method: str):
        """Catat hasil satu pertanyaan."""
        self.questions.append(
            QuestionResult(context, question, answer, confidence, method)
        )

    def add_correction(self, question: str, wrong_answer, correct_answer: str,
                       reasoning: str, pattern_key: str):
        """Catat satu koreksi yang diterapkan."""
        self.corrections.append(
            CorrectionRecord(question, wrong_answer, correct_answer,
                             reasoning, pattern_key)
        )

    def set_benchmark(self, phase: str, results: dict):
        """Set benchmark results.

        Args:
            phase: 'before' atau 'after'
            results: dict dengan keys: total, correct, accuracy, per_type
        """
        if phase == 'before':
            self.benchmark_before = results
        elif phase == 'after':
            self.benchmark_after = results

    def end(self):
        """Tandai sesi selesai."""
        self.ended_at = datetime.now()

    def summary(self) -> dict:
        """Return summary dict untuk export."""
        total_q = len(self.questions)
        total_c = len(self.corrections)

        before_acc = None
        after_acc = None
        delta = None

        if self.benchmark_before:
            before_acc = self.benchmark_before.get('accuracy')
        if self.benchmark_after:
            after_acc = self.benchmark_after.get('accuracy')
        if before_acc is not None and after_acc is not None:
            delta = after_acc - before_acc

        return {
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'total_questions': total_q,
            'total_corrections': total_c,
            'accuracy_before': before_acc,
            'accuracy_after': after_acc,
            'accuracy_delta': delta,
        }
