# @WHO:   self-ai/src/training/training_agent.py
# @WHAT:  Core training agent — run questions, accept corrections, benchmark, export
# @PART:  self-ai/training
# @ENTRY: TrainingAgent.run(), TrainingAgent.correct(), TrainingAgent.confirm_correction(), TrainingAgent.benchmark(), TrainingAgent.export_session()

"""TrainingAgent — dedicated agent untuk mengajari SELF dari interaksi nyata.

Bukan simulasi. Ini agent yang benar-benar melatih SELF, mengukur hasilnya,
dan mendokumentasikan semuanya secara otomatis.

Core responsibilities:
  1. Jalankan soal ke SELF, catat hasilnya          → run()
  2. Terima koreksi, generate reasoning, simpan      → correct() + confirm_correction()
  3. Ukur improvement                                  → benchmark()
  4. Dokumentasi otomatis                              → export_session()

Design principle: intent harus eksplisit dari user, bukan ditebak sistem.
- correct() hanya generate reasoning + tampilkan — TIDAK auto-teach
- confirm_correction() baru memanggil teach_from_correction()
- User harus eksplisit konfirmasi sebelum pattern disimpan

Bug fix v1.1:
    _last_context is stored separately from _last_result to prevent
    context overwrite when benchmark() is called between run() and
    correct(). Previously, correct() used _last_result['context']
    which got overwritten by benchmark runs — storing the question
    as context instead of the original narrative text. This caused
    Qwen3 to generate reasoning based on wrong context.
"""

import os
import sys
import logging
from typing import Optional

from training.session import TrainingSession
from training.results import export_session

logger = logging.getLogger(__name__)

# Project root — 3 levels up from this file
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)


class TrainingAgent:
    """Agent yang mengajari SELF dari interaksi nyata.

    # @FLOW: TRAINING_SESSION
    # @CALLS: DerivationEngine.derive_from_text(), TextComprehension.teach_from_correction(),
    #         TextComprehension._generate_correction_reasoning(), benchmark_empiris
    # @MUTATES: learned_patterns (via teach_from_correction), disk (via _save_learned_patterns)
    # @BEHAVIOR: Runs questions against SELF, accepts explicit corrections from user,
    #            generates reasoning via Qwen3, confirms with user, then stores pattern.
    #            Never auto-detects corrections — user must explicitly trigger correct().
    """

    def __init__(self):
        self.session = TrainingSession()
        self._last_result = None  # Untuk correct() yang merujuk run() terakhir
        self._last_context = ''  # BUG FIX v1.1: stored separately, never overwritten
        self._pending_correction = None  # Koreksi yang menunggu konfirmasi

        # Setup sys.path agar import derivation.* bisa jalan
        src_dir = os.path.join(_PROJECT_ROOT, 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Initialize engine
        self._init_engine()

    def _init_engine(self):
        """Initialize DerivationEngine + TextComprehension."""
        try:
            from derivation.engine import DerivationEngine
            from core.self import SelfCore

            self_core = SelfCore()
            self.engine = DerivationEngine(self_core)
            self.engine._init_modules()
            self.tc = self.engine.text_comprehension

            logger.info("TrainingAgent initialized — engine ready")
        except Exception as e:
            logger.error("Failed to initialize engine: %s", e)
            self.engine = None
            self.tc = None

    # ── Core Operations ──

    def run(self, question: str, context: str) -> dict:
        """Jalankan soal ke SELF, catat hasilnya.

        # @FLOW: TRAINING_RUN
        # @CALLS: DerivationEngine.derive_from_text()
        # @MUTATES: session (adds QuestionResult), _last_result, _last_context
        # @BEHAVIOR: Returns {answer, confidence, method, pattern_used} for the question.
        #            Stores result internally for correct() to reference.
        #            _last_context is stored separately from _last_result so it
        #            survives subsequent benchmark() calls (bug fix v1.1).

        Args:
            question: Pertanyaan yang diajukan ke SELF
            context: Teks konteks (narrative)

        Returns:
            Dict dengan answer, confidence, method
        """
        if self.engine is None:
            return {'error': 'Engine not initialized'}

        # Clear any pending correction
        self._pending_correction = None

        result = self.engine.derive_from_text(context, question)

        answer = result.get('answer')
        confidence = result.get('confidence', 0)
        method = result.get('method', '')

        # BUG FIX v1.1: Store context separately so benchmark() can't overwrite it
        self._last_result = {
            'context': context,
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'method': method,
        }
        self._last_context = context  # The ORIGINAL narrative text

        self.session.add_question(context, question, answer, confidence, method)

        return {
            'answer': answer,
            'confidence': confidence,
            'method': method,
        }

    def correct(self, correct_answer: str) -> dict:
        """Terima koreksi untuk run() terakhir — generate reasoning, TIDAK auto-teach.

        # @FLOW: TRAINING_CORRECT
        # @CALLS: TextComprehension._generate_correction_reasoning()
        # @MUTATES: none (session mutation only on confirm)
        # @BEHAVIOR: Generates reasoning via Qwen3 for why the corrected answer
        #            is correct. Returns reasoning for user to review.
        #            Does NOT call teach_from_correction() — user must confirm
        #            via confirm_correction() first.

        Args:
            correct_answer: Jawaban yang benar dari user

        Returns:
            Dict dengan reasoning, correct_answer, question, confirmed=False
        """
        if self.tc is None:
            return {'error': 'TextComprehension not initialized'}

        if self._last_result is None:
            return {'error': 'No question to correct. Run (q)uestion first.'}

        # BUG FIX v1.1: Use _last_context (original narrative text),
        # NOT _last_result['context'] which gets overwritten by benchmark()
        context_text = self._last_context
        question = self._last_result['question']

        # Generate reasoning via Qwen3
        reasoning = self.tc._generate_correction_reasoning(
            question=question,
            correct_answer=correct_answer,
            context_text=context_text,
        )

        # Store pending correction — menunggu konfirmasi
        self._pending_correction = {
            'context': context_text,
            'question': question,
            'wrong_answer': self._last_result['answer'],
            'correct_answer': correct_answer,
            'reasoning': reasoning or f"Koreksi pengguna: jawaban yang benar untuk pertanyaan ini adalah {correct_answer}",
        }

        return {
            'reasoning': self._pending_correction['reasoning'],
            'correct_answer': correct_answer,
            'question': question,
            'confirmed': False,
        }

    def confirm_correction(self, edited_reasoning: str = '') -> dict:
        """Konfirmasi koreksi → panggil teach_from_correction().

        # @FLOW: TRAINING_CONFIRM
        # @CALLS: TextComprehension.teach_from_correction()
        # @MUTATES: learned_patterns (via teach), disk (via _save_learned_patterns),
        #           session (adds CorrectionRecord)
        # @BEHAVIOR: Only after explicit user confirmation. Calls teach_from_correction()
        #            which generates reasoning, calls teach(), persists to disk.
        #            If no pending correction, returns error.

        Args:
            edited_reasoning: Optional edited reasoning from user (replaces generated)

        Returns:
            Dict dengan pattern_key, reasoning, confirmed=True
        """
        if self.tc is None:
            return {'error': 'TextComprehension not initialized'}

        if self._pending_correction is None:
            return {'error': 'No pending correction. Run (c)orrect first.'}

        pc = self._pending_correction
        reasoning = edited_reasoning or pc['reasoning']

        # Call teach_from_correction — this stores the pattern + persists to disk
        result = self.tc.teach_from_correction(
            text=pc['context'],
            question=pc['question'],
            correct_answer=pc['correct_answer'],
            correction_raw=f"TrainingAgent correction: {pc['correct_answer']}",
        )

        # Record in session
        self.session.add_correction(
            question=pc['question'],
            wrong_answer=pc['wrong_answer'],
            correct_answer=pc['correct_answer'],
            reasoning=reasoning,
            pattern_key=result.get('pattern_key', ''),
        )

        # Clear pending
        self._pending_correction = None

        return {
            'pattern_key': result.get('pattern_key', ''),
            'reasoning': reasoning,
            'confirmed': True,
        }

    def reject_correction(self) -> dict:
        """Tolak koreksi yang pending — tidak teach, tidak simpan.

        Returns:
            Dict dengan confirmed=False, rejected=True
        """
        self._pending_correction = None
        return {'confirmed': False, 'rejected': True}

    # ── Benchmark ──

    def benchmark(self, test_cases: list = None) -> dict:
        """Ukur accuracy against test cases.

        # @FLOW: TRAINING_BENCHMARK
        # @CALLS: DerivationEngine.derive_from_text()
        # @MUTATES: none
        # @BEHAVIOR: Runs test cases through SELF and measures accuracy.
        #            Does NOT modify _last_result or _last_context — uses local
        #            variables so the correction flow isn't disrupted.
        #            If no test_cases provided, uses TEST_SOAL from benchmark_empiris.
        #            Returns dict with total, correct, accuracy, per_type breakdown.

        Args:
            test_cases: List of test case dicts with 'text', 'question',
                        'expected_keywords', 'type'. If None, uses default.

        Returns:
            Dict dengan total, correct, accuracy, per_type
        """
        if self.engine is None:
            return {'error': 'Engine not initialized'}

        if test_cases is None:
            test_cases = self._get_default_test_cases()

        if not test_cases:
            return {'error': 'No test cases available'}

        results = []
        per_type = {}

        for soal in test_cases:
            # Use local variable — do NOT overwrite _last_result or _last_context
            result = self.engine.derive_from_text(soal['text'], soal['question'])
            answer = result.get('answer')
            is_match = self._check_answer(answer, soal.get('expected_keywords', []))
            soal_type = soal.get('type', 'unknown')

            results.append({
                'id': soal.get('id', ''),
                'type': soal_type,
                'pass': is_match,
                'answer': str(answer)[:100] if answer else None,
                'method': result.get('method', ''),
            })

            if soal_type not in per_type:
                per_type[soal_type] = {'correct': 0, 'total': 0}
            per_type[soal_type]['total'] += 1
            if is_match:
                per_type[soal_type]['correct'] += 1

        # Compute accuracy
        total = len(results)
        correct = sum(1 for r in results if r['pass'])
        accuracy = correct / total if total > 0 else 0

        # Compute per-type accuracy
        for domain in per_type:
            d = per_type[domain]
            d['accuracy'] = d['correct'] / d['total'] if d['total'] > 0 else 0

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'per_type': per_type,
            'details': results,
        }

    def _get_default_test_cases(self) -> list:
        """Load default test cases from benchmark module."""
        try:
            benchmark_dir = os.path.join(_PROJECT_ROOT, 'benchmark')
            if benchmark_dir not in sys.path:
                sys.path.insert(0, benchmark_dir)

            from benchmark_empiris import TEST_SOAL
            return TEST_SOAL
        except ImportError:
            logger.warning("Could not import TEST_SOAL from benchmark_empiris")
            return []

    def _check_answer(self, answer, expected_keywords: list) -> bool:
        """Check if answer matches any expected keyword."""
        if not answer or not expected_keywords:
            return False

        answer_lower = str(answer).lower().strip()

        for keyword in expected_keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower in answer_lower:
                return True

        return False

    # ── Export ──

    def export_session(self, output_dir: str = None) -> str:
        """Export session ke Markdown file.

        # @FLOW: TRAINING_EXPORT
        # @CALLS: results.export_session()
        # @MUTATES: filesystem (writes markdown file)
        # @BEHAVIOR: Exports full session documentation to markdown.
        #            Auto-creates output directory if needed.

        Args:
            output_dir: Directory untuk simpan. Default: docs/training_sessions/

        Returns:
            str: Path file yang dibuat
        """
        if output_dir is None:
            output_dir = os.path.join(_PROJECT_ROOT, 'docs', 'training_sessions')

        self.session.end()
        filepath = export_session(self.session, output_dir)
        logger.info("Session exported to %s", filepath)
        return filepath
