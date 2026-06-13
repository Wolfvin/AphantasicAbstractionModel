# @WHO:   self-ai/src/derivation/answer_handlers.py
# @WHAT:  Understanding-first delegation — SELF's answer pipeline
# @PART:  self-ai/derivation
# @ENTRY: AnswerHandlers (imported by text_comprehension.py)

"""Answer Handlers — Understanding-first delegation for SELF-AI.

Philosophy:
    NO hardcoded word lists. NO hardcoded semantic maps.
    NO hardcoded if-else chains. NO hardcoded knowledge.

    ALL reasoning must come from:
    1. SELF-built understanding graph (transformation-based, NO LLM needed)
    2. SELF-discovered patterns from observation (fallback)
    3. LLM reasoning (last resort)

    Priority order (most autonomous → least autonomous):
        Understanding > Pattern > LLM > Give up
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AnswerHandlers:
    """Understanding-first delegation layer for answer handling.

    Delegates ALL reasoning to:
    1. Understanding graph — SELF-built transformations (NO LLM)
    2. Multi-understanding composition (Qwen3 combines understandings)
    3. Proactive self-correction when confidence is low
    4. LLM reasoning (fallback)

    v30: Added self-correction loop — proactive correction when confidence
    is low, and post-answer correction when feedback is received.

    v32: Removed legacy PatternLearner, seed_core_understandings fallback,
    _try_self_discovered_pattern(), and _try_learned_pattern().
    These were hardcoded legacy paths that violated the "NO hardcoded rules" vision.

    NO hardcoded domain knowledge in this class.
    """

    def __init__(self, tc):
        self.tc = tc
        self._understanding_graph = None
        self._correction_loop = None  # v30: Self-correction loop
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-initialize understanding graph.

        v32: Removed seed_core_understandings fallback and PatternLearner.
        These were hardcoded legacy that violated the "NO hardcoded rules" vision.
        SELF now builds ALL understandings through:
          1. UnderstandingComposer (Qwen3) — builds understanding from teaching
          2. UnderstandingBuilder (rule-based observation) — observes teaching lessons
          3. Self-correction loop — learns from failures

        If Qwen3 is unavailable, the graph starts EMPTY and fills through
        teaching and self-correction. This is by design — SELF must EARN
        its knowledge, not be seeded with hardcoded rules.

        Initialization order:
        1. Init shared UnderstandingGraph
        2. Try UnderstandingComposer with teaching lessons
        3. Try UnderstandingBuilder (observation-based)
        4. Init SelfCorrectionLoop
        """
        if self._initialized:
            return

        from derivation.understanding_builder import UnderstandingBuilder, get_shared_graph

        # v29 fix: Use shared singleton graph so ALL components see the same understandings
        self._understanding_graph = get_shared_graph()

        # v28: Try UnderstandingComposer first (Qwen3 builds understanding from lessons)
        try:
            from derivation.understanding_composer import get_shared_composer
            from derivation.teaching_lessons import get_default_lessons

            # v31 fix (P2): Use shared composer singleton
            composer = get_shared_composer()
            lessons = get_default_lessons()
            composed_count = 0
            for q_type in lessons.get_types():
                for lesson in lessons.get_by_type(q_type):
                    node = composer.compose_from_teaching(lesson)
                    if node is not None:
                        composed_count += 1

            if composed_count > 0:
                logger.info("UnderstandingComposer built %d understanding nodes from lessons", composed_count)
            else:
                logger.info("UnderstandingComposer produced no nodes — graph starts empty, will learn through teaching")
        except ImportError:
            logger.info("UnderstandingComposer not available — graph starts empty, will learn through teaching")
        except Exception as e:
            logger.warning("UnderstandingComposer failed: %s — graph starts empty, will learn through teaching", e)

        # Try to build from teaching lessons via UnderstandingBuilder
        # (rule-based observation, adds to what composer created)
        try:
            from derivation.teaching_lessons import get_default_lessons
            lessons = get_default_lessons()
            builder = UnderstandingBuilder(graph=self._understanding_graph)
            builder.seed_from_lessons(lessons)
        except Exception as e:
            logger.warning("Failed to build from teaching lessons: %s", e)

        # v30: Self-correction loop — closes the self-improvement loop
        try:
            from derivation.self_correction import SelfCorrectionLoop
            self._correction_loop = SelfCorrectionLoop(graph=self._understanding_graph)
            logger.info("SelfCorrectionLoop initialized")
        except ImportError:
            logger.info("SelfCorrectionLoop not available")
        except Exception as e:
            logger.warning("Failed to init SelfCorrectionLoop: %s", e)

        self._initialized = True

    def _try_understanding_pipeline(self, text: str, question: str,
                                     q_type: str) -> Optional[dict]:
        """Unified understanding pipeline — retrieve multi, try individually, then compose.

        v29 fix: Merged _try_understanding and _try_understanding_composition
        to avoid double embedding computation for the same input.

        Strategy:
        1. Retrieve top-3 understandings via find_matching_multi() (ONE embedding call)
        2. Try applying the best match first (Path 1 equivalent)
        3. If that fails, try other matches individually
        4. If all fail individually, compose with Qwen3
        """
        self._ensure_initialized()

        if self._understanding_graph is None:
            return None

        # Step 1: Single retrieval for ALL paths (ONE embedding computation)
        try:
            matches = self._understanding_graph.find_matching_multi(
                text, question, top_k=3, threshold=0.15
            )
        except Exception as e:
            logger.debug("Multi-understanding retrieval failed: %s", e)
            return None

        if not matches:
            return None

        # Step 2: Try applying each understanding individually
        # (Path 1 equivalent: try best match first)
        for node, score in matches:
            if node.transformation is None:
                continue
            result = self._understanding_graph.apply(node, text, question)
            if result and result.get('answer'):
                logger.debug("Answered via understanding: %s (kind=%s, score=%.3f, no LLM)",
                            node.id, node.transformation.kind, score)
                return result

        # Step 3: Compose multiple understandings via Qwen3
        # (Path 2 equivalent: combine understandings)
        if len(matches) >= 1:
            try:
                # v31 fix (P2): Use shared composer singleton instead of creating new
                from derivation.understanding_composer import get_shared_composer
                composer = get_shared_composer()

                composed_answer = composer.compose_answer_from_understandings(
                    text, question, matches
                )

                if composed_answer:
                    return {
                        'answer': composed_answer,
                        'confidence': 0.55,
                        'method': 'understanding_composition',
                        'explanation': f'Composed from {len(matches)} understandings via Qwen3',
                        'source': 'composed',
                        'applied_without_llm': False,
                        'composed_understandings': [
                            {'id': n.id, 'kind': n.transformation.kind if n.transformation else '?', 'score': float(s)}
                            for n, s in matches
                        ],
                    }
            except ImportError:
                logger.debug("UnderstandingComposer not available for composition")
            except Exception as e:
                logger.debug("Understanding composition failed: %s", e)

        return None



    def _try_llm_reasoning(self, text: str, question: str,
                            q_type: str, previous_result: dict = None) -> Optional[dict]:
        """Fallback to LLM reasoning when SELF has no understanding."""
        try:
            if not hasattr(self.tc, '_llm_engine') or self.tc._llm_engine is None:
                from derivation.llm_reasoning import LLMReasoningEngine
                self.tc._llm_engine = LLMReasoningEngine(confidence_threshold=0.5)

            result = self.tc._llm_engine.reason(
                text, question, previous_result or {'answer': None, 'confidence': 0.0}
            )
            if result and result.get('answer'):
                return result
        except Exception as e:
            logger.warning("LLM reasoning fallback error: %s", e)

        return None



    def _delegate(self, q_type: str, propositions: list, question: str,
                   text: str) -> dict:
        """Universal delegation — tries all reasoning paths in priority order.

        For EVERY question type, the flow is the same:
        1. Try SELF-built understanding graph (NO LLM, single transformation)
        2. Try multi-understanding composition (multiple transformations)
        3. Try proactive self-correction if confidence is low (v30)
        4. Try LLM reasoning
        5. Give up

        v30: Added proactive self-correction after understanding pipeline.

        v32: Removed legacy PatternLearner and learned pattern paths.
        These were hardcoded fallbacks that violated the "NO hardcoded rules" vision.
        SELF now relies entirely on:
          - Understanding graph (self-built)
          - Self-correction (self-improving)
          - LLM reasoning (last resort)

        NO type-specific hardcoded logic.

        @FLOW:     ANSWER_DELEGATE
        @CALLS:    _try_understanding_pipeline(), _try_proactive_correction(), _try_llm_reasoning()
        @MUTATES:  none
        """
        # Path 1+2: Unified understanding pipeline (v29 — merged to avoid double embedding)
        result = self._try_understanding_pipeline(text, question, q_type)
        if result and result.get('confidence', 0) >= 0.35:
            return result

        # Path 2.5: Proactive self-correction (v30)
        # If understanding pipeline gave a low-confidence answer, try self-correction
        # BEFORE falling back to LLM. This gives SELF a chance to improve.
        if result and result.get('confidence', 0) < 0.35:
            corrected = self._try_proactive_correction(text, question, result)
            if corrected is not None:
                return corrected
            # If correction didn't help, keep the original result
            return result

        # Path 3: LLM reasoning fallback (LAST RESORT)
        result = self._try_llm_reasoning(text, question, q_type)
        if result:
            return result

        # No answer found
        return {
            'answer': None,
            'confidence': 0.0,
            'method': f'{q_type}_no_understanding',
            'explanation': f'SELF has no understanding for {q_type}',
        }

    def _try_proactive_correction(self, text: str, question: str,
                                   answer: dict) -> Optional[dict]:
        """Try proactive self-correction when confidence is low.

        v30: SELF detects that its answer has low confidence and
        proactively tries to improve it before returning.

        v31 fix (P6): Pass from_correction flag to prevent cascade.
        If the answer was already produced by a previous correction,
        we don't try to correct it again.

        If the corrected answer has higher confidence, return it.
        Otherwise, return None (keep the original answer).
        """
        if self._correction_loop is None:
            return None

        try:
            # v31 fix (P6): Check if this answer came from a previous correction
            from_correction = answer.get('from_correction', False)

            corrected = self._correction_loop.proactive_correct(
                text=text, question=question, answer=answer,
                from_correction=from_correction,
            )
            if corrected is not None:
                logger.info("Proactive self-correction improved answer: "
                           "conf %.2f → %.2f",
                           answer.get('confidence', 0),
                           corrected.get('confidence', 0))
                return corrected
        except Exception as e:
            logger.debug("Proactive self-correction failed: %s", e)

        return None

    def provide_feedback(self, text: str, question: str,
                         wrong_answer: str, correct_answer: str,
                         answer_method: str = '',
                         answer_confidence: float = 0.0) -> Optional[dict]:
        """Provide feedback for self-correction.

        v30: External feedback interface. When the user or system
        knows the correct answer, call this to trigger the full
        self-correction cycle:

        1. Detect which understanding failed
        2. Diagnose the failure
        3. Compose new understanding from the correction
        4. Verify the new understanding
        5. Prune the failed understanding
        6. Re-attempt the answer

        Returns the correction result dict, or None if correction loop
        is not available.
        """
        self._ensure_initialized()

        if self._correction_loop is None:
            logger.warning("SelfCorrectionLoop not available — cannot process feedback")
            return None

        try:
            result = self._correction_loop.correct(
                text=text,
                question=question,
                wrong_answer=wrong_answer,
                correct_answer=correct_answer,
                answer_method=answer_method,
                answer_confidence=answer_confidence,
            )

            if result.get('corrected'):
                logger.info("Self-correction SUCCEEDED: new understanding %s",
                           result.get('new_understanding_id', '?'))
            else:
                logger.info("Self-correction attempted but not verified for: %s",
                           question[:50])

            return result
        except Exception as e:
            logger.warning("Self-correction feedback failed: %s", e)
            return None

    def get_correction_stats(self) -> dict:
        """Get self-correction statistics.

        v30: Returns stats about how many corrections have been
        attempted, how many verified, and which understandings
        fail most often.
        """
        if self._correction_loop is None:
            return {'available': False}

        return {
            'available': True,
            **self._correction_loop.get_correction_stats(),
        }

    # ═══════════════ HANDLER METHODS ═══════════════
    # Each handler just delegates to _delegate() with the question type.
    # NO hardcoded logic. NO word lists. NO semantic maps.

    def _answer_ide_pokok(self, propositions, question, text):
        return self._delegate('ide_pokok', propositions, question, text)

    def _answer_peribahasa(self, propositions, question, text):
        return self._delegate('peribahasa', propositions, question, text)

    def _answer_bahasa_kiasan(self, propositions, question, text):
        return self._delegate('bahasa_kiasan', propositions, question, text)

    def _answer_teks_argumentatif(self, propositions, question, text):
        return self._delegate('teks_argumentatif', propositions, question, text)

    def _answer_perbandingan(self, propositions, question, text):
        return self._delegate('perbandingan', propositions, question, text)

    def _answer_motivasi(self, propositions, question, text):
        return self._delegate('motivasi', propositions, question, text)

    def _answer_sinonim_antonim(self, propositions, question, text):
        return self._delegate('sinonim_antonim', propositions, question, text)

    def _answer_sikap_tokoh(self, propositions, question, text):
        return self._delegate('sikap_tokoh', propositions, question, text)

    def _answer_teks_prosedur(self, propositions, question, text):
        return self._delegate('teks_prosedur', propositions, question, text)

    def _answer_teks_persuasif(self, propositions, question, text):
        return self._delegate('teks_persuasif', propositions, question, text)

    def _answer_unsur_cerita(self, propositions, question, text):
        return self._delegate('unsur_cerita', propositions, question, text)

    def _answer_benar_salah(self, propositions, question, text):
        return self._delegate('benar_salah', propositions, question, text)

    def _answer_analogi(self, propositions, question, text):
        return self._delegate('analogi', propositions, question, text)

    def _answer_kesan_pesan(self, propositions, question, text):
        return self._delegate('kesan_pesan', propositions, question, text)

    def _answer_penyebab_ganda(self, propositions, question, text):
        return self._delegate('penyebab_ganda', propositions, question, text)

    def _answer_pertanyaan_negatif(self, propositions, question, text):
        return self._delegate('pertanyaan_negatif', propositions, question, text)

    def _answer_tone_mood(self, propositions, question, text):
        return self._delegate('tone_mood', propositions, question, text)

    def _answer_teks_eksplanasi(self, propositions, question, text):
        return self._delegate('teks_eksplanasi', propositions, question, text)

    def _answer_inferensi_silang(self, propositions, question, text):
        return self._delegate('inferensi_silang', propositions, question, text)

    def _answer_konteks_makna(self, propositions, question, text):
        return self._delegate('konteks_makna', propositions, question, text)

    def _answer_eksplisit(self, propositions, question, text):
        return self._delegate('eksplisit', propositions, question, text)

    def _answer_implisit(self, propositions, question, text):
        return self._delegate('implisit', propositions, question, text)

    def _answer_interpretatif(self, propositions, question, text):
        return self._delegate('interpretatif', propositions, question, text)
