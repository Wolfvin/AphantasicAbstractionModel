# @WHO:   self-ai/src/derivation/self_correction.py
# @WHAT:  Self-Correction Loop — SELF detects, diagnoses, and fixes its own mistakes
# @PART:  self-ai/derivation
# @ENTRY: SelfCorrectionLoop

"""Self-Correction Loop — Phase 4: SELF detects, diagnoses, and fixes its own mistakes.

Vision:
    This is the CLOSING of the loop. Phases 1-3 gave SELF the ability to:
      - Build understanding from observation (Phase 1)
      - Retrieve and apply understanding (Phase 2)
      - Compose multiple understandings via Qwen3 (Phase 3)

    Phase 4 closes the loop by making SELF SELF-CORRECTING:
      - When SELF answers wrong → it LEARNS from the failure
      - When SELF is uncertain → it SEEKS more information
      - When SELF's understanding is proven wrong → it PRUNES it
      - When SELF creates new understanding → it VERIFIES before committing

    This is the mechanism that makes SELF improve over time, autonomously.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Answer produced (confidence < threshold OR feedback = wrong)    │
    └──────────────┬──────────────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. DETECT — Is there a problem?                                │
    │     - Explicit feedback: "wrong answer"                          │
    │     - Implicit signal: low confidence, high novelty              │
    │     - Self-doubt: answer contradicts existing understanding      │
    └──────────────┬──────────────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  2. DIAGNOSE — WHY is this wrong?                               │
    │     - GapIdentifier (meta_cognitive.py): what knowledge is missing│
    │     - Understanding audit: which existing understanding failed?   │
    │     - Failure pattern: has this TYPE of failure happened before?  │
    └──────────────┬──────────────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  3. LEARN — Compose new/corrected understanding                 │
    │     - UnderstandingComposer.compose_from_failure()               │
    │     - New understanding starts with PROBATION confidence (0.3)   │
    │     - Only promoted to full confidence after verification        │
    └──────────────┬──────────────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  4. VERIFY — Test the new understanding before committing       │
    │     - Re-apply to the SAME question → does it produce the        │
    │       correct answer now?                                         │
    │     - Cross-check against existing verified understandings        │
    │     - If verified → promote confidence to 0.55                   │
    │     - If fails → discard and log as a failed correction          │
    └──────────────┬──────────────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  5. PRUNE — Weaken incorrect understandings                     │
    │     - The understanding that produced the wrong answer is         │
    │       WEAKENED (confidence reduced)                               │
    │     - If an understanding's confidence drops below 0.15, it's    │
    │       marked as DEPRECATED (not deleted — it might be partially  │
    │       correct in a different context)                             │
    │     - Repeated failures of the same understanding → faster decay │
    └──────────────┬──────────────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  6. ITERATE — Re-attempt the answer with improved understanding │
    │     - Only ONE re-attempt per correction cycle                    │
    │     - Prevents infinite loops                                     │
    │     - If re-attempt also fails, give up gracefully                │
    └─────────────────────────────────────────────────────────────────┘

Key Design Decisions:
    1. PROBATION SYSTEM: New understandings from failures start at 0.3
       confidence (not 0.55 like teaching). They must EARN trust.

    2. VERIFICATION BEFORE COMMIT: A new understanding must produce
       the correct answer for the triggering question before it's
       promoted. This prevents hallucinated corrections.

    3. GRACEFUL DEGRADATION: Wrong understandings are weakened, not
       deleted. They might be partially correct in other contexts.

    4. SINGLE ITERATION: The loop runs at most once per question.
       This prevents infinite correction loops.

    5. INTEGRATION POINTS: The loop integrates with existing components:
       - MetaCognitiveMonitor for gap identification
       - UnderstandingComposer for new understanding creation
       - UnderstandingGraph for storage and retrieval
       - AnswerHandlers for re-attempting answers
"""

import time
import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class CorrectionRecord:
    """Record of a self-correction attempt — what was wrong and how it was fixed.

    This is the "memory of corrections" — SELF remembers what it got wrong,
    how it tried to fix it, and whether the fix worked.
    """

    def __init__(self, question: str, wrong_answer: str, correct_answer: str,
                 failed_node_id: str = '', new_node_id: str = '',
                 diagnosis: dict = None, verified: bool = False):
        self.question = question
        self.wrong_answer = wrong_answer
        self.correct_answer = correct_answer
        self.failed_node_id = failed_node_id
        self.new_node_id = new_node_id
        self.diagnosis = diagnosis or {}
        self.verified = verified
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            'question': self.question,
            'wrong_answer': self.wrong_answer,
            'correct_answer': self.correct_answer,
            'failed_node_id': self.failed_node_id,
            'new_node_id': self.new_node_id,
            'diagnosis': self.diagnosis,
            'verified': self.verified,
            'timestamp': self.timestamp,
        }


class SelfCorrectionLoop:
    """The orchestrator for SELF's self-correction cycle.

    This is Phase 4 — closing the loop so that SELF can:
    1. Detect when it's wrong
    2. Diagnose why
    3. Learn from the failure
    4. Verify the fix
    5. Prune incorrect understandings
    6. Re-attempt with improved knowledge

    Usage:
        loop = SelfCorrectionLoop(graph=shared_graph)

        # When feedback shows a wrong answer
        result = loop.correct(
            text=text,
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            answer_method=answer_method,
        )

        # When answer has low confidence (proactive self-correction)
        result = loop.proactive_correct(
            text=text,
            question=question,
            answer=low_confidence_answer,
        )
    """

    # Confidence thresholds for self-correction triggers
    PROACTIVE_THRESHOLD = 0.35      # Below this, try proactive correction
    PROBATION_CONFIDENCE = 0.30     # New understandings from failures start here
    VERIFIED_CONFIDENCE = 0.55     # Promoted confidence after verification
    PRUNE_THRESHOLD = 0.15         # Below this, understanding is deprecated
    MAX_CORRECTION_ATTEMPTS = 1     # Only one re-attempt per cycle

    # Weakening amounts
    FIRST_FAIL_WEAKEN = 0.10       # First failure: -0.10 confidence
    REPEAT_FAIL_WEAKEN = 0.20      # Repeat failure: -0.20 confidence
    CONFLICT_WEAKEN = 0.15         # Conflicts with verified understanding: -0.15

    def __init__(self, graph=None, composer=None):
        """Initialize the SelfCorrectionLoop.

        Args:
            graph: UnderstandingGraph (shared singleton)
            composer: UnderstandingComposer (lazy-init if None)
        """
        self._graph = graph
        self._composer = composer
        self._correction_history: List[CorrectionRecord] = []
        self._failure_counts: Dict[str, int] = defaultdict(int)  # node_id -> fail count
        self._correction_count = 0
        self._verification_count = 0
        self._successful_corrections = 0

    @property
    def graph(self):
        """Lazy-initialize understanding graph."""
        if self._graph is None:
            from derivation.understanding_builder import get_shared_graph
            self._graph = get_shared_graph()
        return self._graph

    @property
    def composer(self):
        """Lazy-initialize understanding composer (singleton)."""
        if self._composer is None:
            # v31 fix (P2): Use shared composer singleton
            from derivation.understanding_composer import get_shared_composer
            self._composer = get_shared_composer()
        return self._composer

    # ═══════════════ MAIN API ═══════════════

    def correct(self, text: str, question: str, wrong_answer: str,
                correct_answer: str, answer_method: str = '',
                answer_confidence: float = 0.0) -> dict:
        """Full self-correction cycle when feedback shows a wrong answer.

        This is the MAIN entry point for self-correction. Called when
        external feedback shows that SELF's answer was wrong.

        Flow:
          1. Identify which understanding produced the wrong answer
          2. Diagnose the gap/failure
          3. Compose new understanding from the failure
          4. Verify the new understanding against the correct answer
          5. Prune the failed understanding
          6. Return correction result

        Args:
            text: Source text
            question: The question that was answered wrong
            wrong_answer: The answer SELF gave
            correct_answer: The correct answer (from feedback)
            answer_method: Method that produced the wrong answer
            answer_confidence: Confidence of the wrong answer

        Returns:
            dict with correction result, including:
            - corrected: bool — was the correction successful?
            - new_understanding_id: str — ID of new understanding (if created)
            - diagnosis: dict — what went wrong
            - reattempt_answer: str — answer after correction (if reattempted)
            - reattempt_confidence: float — confidence after correction
        """
        logger.info("Self-correction triggered: question='%s...' wrong='%s' correct='%s'",
                    question[:50], str(wrong_answer)[:30], str(correct_answer)[:30])

        self._correction_count += 1

        # Step 1: Find the understanding that produced the wrong answer
        failed_node = self._find_failed_understanding(
            text, question, answer_method, answer_confidence
        )
        failed_node_id = failed_node.id if failed_node else ''

        # Step 2: Diagnose the failure
        diagnosis = self._diagnose_failure(
            text, question, wrong_answer, correct_answer,
            failed_node, answer_method
        )

        # Step 3: Compose new understanding from the failure
        new_node = self._learn_from_failure(
            text, question, wrong_answer, correct_answer, diagnosis
        )

        # Step 4: Verify the new understanding
        verified = False
        new_node_id = ''
        if new_node is not None:
            verified = self._verify_correction(
                new_node, text, question, correct_answer
            )
            new_node_id = new_node.id

            if verified:
                # Promote confidence from probation to verified
                new_node.confidence = self.VERIFIED_CONFIDENCE
                self.graph._save()
                self._verification_count += 1
                self._successful_corrections += 1
                logger.info("Correction VERIFIED: new understanding %s promoted to %.2f",
                           new_node_id, self.VERIFIED_CONFIDENCE)
            else:
                # Verification failed — keep at probation confidence
                logger.info("Correction NOT verified: understanding %s stays at %.2f",
                           new_node_id, self.PROBATION_CONFIDENCE)

        # Step 5: Prune the failed understanding
        if failed_node is not None:
            self._prune_understanding(failed_node, diagnosis)

        # Step 6: Record the correction
        record = CorrectionRecord(
            question=question,
            wrong_answer=str(wrong_answer),
            correct_answer=str(correct_answer),
            failed_node_id=failed_node_id,
            new_node_id=new_node_id,
            diagnosis=diagnosis,
            verified=verified,
        )
        self._correction_history.append(record)

        # Step 7: Re-attempt the answer (if we have a new verified understanding)
        reattempt_result = None
        if verified and new_node is not None:
            reattempt_result = self._reattempt_answer(text, question)

        return {
            'corrected': verified,
            'new_understanding_id': new_node_id,
            'failed_understanding_id': failed_node_id,
            'diagnosis': diagnosis,
            'verified': verified,
            'reattempt_answer': reattempt_result.get('answer') if reattempt_result else None,
            'reattempt_confidence': reattempt_result.get('confidence') if reattempt_result else None,
            'reattempt_method': reattempt_result.get('method') if reattempt_result else None,
        }

    def proactive_correct(self, text: str, question: str,
                          answer: dict, from_correction: bool = False) -> Optional[dict]:
        """Proactive self-correction when confidence is low.

        This is called when SELF's answer has LOW confidence — before
        any external feedback. SELF tries to self-correct proactively.

        Strategy:
          1. If confidence < PROACTIVE_THRESHOLD, try to improve
          2. Diagnose the gap
          3. Try composing new understanding (from observation)
          4. Re-attempt with improved knowledge
          5. If re-attempt has higher confidence, use it

        v31 fix (P6): Added from_correction parameter to prevent cascade.
        When a probation understanding (just created by proactive correction)
        produces a low-confidence answer, we DON'T trigger another round
        of proactive correction — that would create an infinite loop of
        low-confidence understandings.

        Args:
            text: Source text
            question: The question
            answer: The low-confidence answer dict
            from_correction: True if this is a re-attempt from a previous
                            correction — prevents cascade (v31 P6 fix)

        Returns:
            Improved answer dict if correction improved things, None otherwise.
        """
        # v31 fix (P6): Prevent cascade — if this answer was already produced
        # by a correction, don't try to correct it again
        if from_correction:
            logger.debug("Proactive correction skipped — answer is from a previous correction (cascade prevention)")
            return None

        confidence = answer.get('confidence', 0.0)
        if confidence >= self.PROACTIVE_THRESHOLD:
            return None  # Confidence is fine, no correction needed

        logger.info("Proactive self-correction: confidence=%.2f for question='%s...'",
                    confidence, question[:50])

        # Diagnose the gap
        diagnosis = self._diagnose_low_confidence(text, question, answer)

        # Try to observe and create understanding
        new_node = None
        try:
            observation = {
                'text': text,
                'question': question,
                'answer': answer.get('answer', ''),
                'novelty_score': 1.0 - confidence,
                'concepts': [],
            }
            new_node = self.composer.compose_from_observation(observation)

            if new_node is not None:
                # Start at probation confidence
                new_node.confidence = self.PROBATION_CONFIDENCE
                logger.info("Proactive correction created understanding: %s", new_node.id)
        except Exception as e:
            logger.debug("Proactive observation composition failed: %s", e)

        # Re-attempt the answer
        if new_node is not None:
            reattempt = self._reattempt_answer(text, question)
            if reattempt is not None:
                new_confidence = reattempt.get('confidence', 0.0)
                # Only accept if it's genuinely better
                if new_confidence > confidence + 0.1:
                    reattempt['self_corrected'] = True
                    reattempt['correction_method'] = 'proactive'
                    reattempt['original_confidence'] = confidence
                    # v31 fix (P6): Mark as from_correction to prevent cascade
                    reattempt['from_correction'] = True
                    return reattempt

        return None

    def get_correction_stats(self) -> dict:
        """Get statistics about self-correction performance."""
        total = len(self._correction_history)
        verified = sum(1 for r in self._correction_history if r.verified)

        # Failure frequency by understanding node
        top_failed = sorted(
            self._failure_counts.items(),
            key=lambda x: x[1], reverse=True
        )[:5]

        return {
            'total_corrections': self._correction_count,
            'total_verified': self._verification_count,
            'successful_corrections': self._successful_corrections,
            'success_rate': self._successful_corrections / max(1, self._correction_count),
            'correction_history_size': total,
            'top_failed_understandings': [
                {'node_id': nid, 'fail_count': count}
                for nid, count in top_failed
            ],
        }

    def prune_deprecated(self) -> int:
        """Prune understandings that have dropped below the confidence threshold.

        This scans all nodes in the graph and marks those below the
        PRUNE_THRESHOLD as deprecated. Deprecated nodes are NOT deleted
        — they are kept for potential partial correctness in other contexts.
        They simply won't be retrieved for answering.

        Returns:
            Number of understandings deprecated.
        """
        deprecated_count = 0
        for node_id, node in self.graph._nodes.items():
            if node.confidence < self.PRUNE_THRESHOLD and node.source != 'deprecated':
                old_source = node.source
                node.source = f'deprecated:{old_source}'
                node.conditions = []  # Clear conditions so they won't match
                deprecated_count += 1
                logger.info("Deprecated understanding %s (confidence=%.3f, was %s)",
                           node_id, node.confidence, old_source)

        if deprecated_count > 0:
            self.graph._save()

        return deprecated_count

    # ═══════════════ STEP 1: FIND FAILED UNDERSTANDING ═══════════════

    def _find_failed_understanding(self, text: str, question: str,
                                    method: str, confidence: float) -> Optional[Any]:
        """Find which understanding node produced the wrong answer.

        Strategy:
          1. If method contains understanding ID, extract it directly
          2. Otherwise, find the best matching understanding for this question
          3. The best match is the most likely culprit
        """
        # Strategy 1: Extract from method string
        # Method format: "understanding_<node_id>" or "understanding_composition"
        if method and 'understanding_' in method:
            # Try to extract node ID
            parts = method.split('understanding_')
            if len(parts) > 1:
                potential_id = parts[1].split('_')[0] if '_' in parts[1] else parts[1]
                node = self.graph.get_node(potential_id)
                if node is not None:
                    return node

        # Strategy 2: Find best matching understanding
        try:
            matches = self.graph.find_matching_multi(text, question, top_k=1, threshold=0.1)
            if matches:
                return matches[0][0]
        except Exception as e:
            logger.debug("Could not find matching understanding: %s", e)

        return None

    # ═══════════════ STEP 2: DIAGNOSE FAILURE ═══════════════

    def _diagnose_failure(self, text: str, question: str,
                          wrong_answer: str, correct_answer: str,
                          failed_node: Any, method: str) -> dict:
        """Diagnose WHY the answer was wrong.

        Uses:
          1. GapIdentifier from meta_cognitive for gap classification
          2. Understanding-specific diagnosis based on the failed node
          3. Historical failure patterns
        """
        diagnosis = {
            'failed_node_id': failed_node.id if failed_node else '',
            'failed_kind': (failed_node.transformation.kind
                           if failed_node and failed_node.transformation else 'unknown'),
            'method': method,
            'failure_type': 'unknown',
            'gap_type': 'unknown',
            'root_cause': '',
            'suggested_fix': '',
        }

        # Use MetaCognitive gap identifier
        try:
            from derivation.meta_cognitive import GapIdentifier
            gap_id = GapIdentifier()
            answer_dict = {'answer': wrong_answer, 'confidence': 0.0, 'method': method}
            gap = gap_id.identify_gap(question, answer_dict, text)
            diagnosis['gap_type'] = gap.get('gap_type', 'unknown')
            diagnosis['suggested_fix'] = gap.get('suggested_action', '')
        except Exception as e:
            logger.debug("Meta-cognitive diagnosis failed: %s", e)

        # Understand-specific diagnosis
        if failed_node is not None:
            # Track failure count for this node
            self._failure_counts[failed_node.id] += 1
            fail_count = self._failure_counts[failed_node.id]

            # What kind of transformation failure?
            if failed_node.transformation:
                kind = failed_node.transformation.kind
                diagnosis['failure_type'] = f'{kind}_mismatch'

                # Analyze the specific transformation failure
                if kind == 'signal_flip':
                    diagnosis['root_cause'] = (
                        'Signal flip transformation applied incorrectly — '
                        'may have captured wrong signal word or wrong result position'
                    )
                elif kind == 'comparison_resolve':
                    diagnosis['root_cause'] = (
                        'Comparison resolution failed — '
                        'may have chosen wrong direction or misidentified entities'
                    )
                elif kind == 'quantity_compute':
                    diagnosis['root_cause'] = (
                        'Quantity computation failed — '
                        'may have extracted wrong numbers or used wrong operation'
                    )
                elif kind == 'negation_affirmation':
                    diagnosis['root_cause'] = (
                        'Negation/affirmation pattern not recognized correctly — '
                        'may have missed the negation or affirmation word'
                    )
                else:
                    diagnosis['root_cause'] = (
                        f'{kind} transformation produced wrong result — '
                        'transformation logic may need refinement'
                    )
            else:
                diagnosis['root_cause'] = (
                    'Understanding has no transformation — '
                    'cannot mechanically apply this understanding'
                )

            # Check for repeated failures
            if fail_count > 1:
                diagnosis['root_cause'] += (
                    f' (REPEATED FAILURE: {fail_count} times — '
                    'this understanding may be fundamentally wrong for this type of question)'
                )
        else:
            # No specific understanding found as the culprit
            if method and 'composition' in method:
                diagnosis['failure_type'] = 'composition_failure'
                diagnosis['root_cause'] = (
                    'Multi-understanding composition produced wrong answer — '
                    'the combination of understandings may be incorrect'
                )
            elif method and 'llm' in method:
                diagnosis['failure_type'] = 'llm_hallucination'
                diagnosis['root_cause'] = (
                    'LLM reasoning produced wrong answer — '
                    'may need teaching to build understanding for this type'
                )
            else:
                diagnosis['failure_type'] = 'no_understanding_found'
                diagnosis['root_cause'] = (
                    'No understanding was found for this question — '
                    'SELF needs to be taught or observe similar examples'
                )

        return diagnosis

    def _diagnose_low_confidence(self, text: str, question: str,
                                  answer: dict) -> dict:
        """Diagnose why confidence is low for proactive correction."""
        diagnosis = {
            'confidence': answer.get('confidence', 0.0),
            'method': answer.get('method', 'unknown'),
            'gap_type': 'unknown',
            'root_cause': '',
        }

        try:
            from derivation.meta_cognitive import GapIdentifier
            gap_id = GapIdentifier()
            gap = gap_id.identify_gap(question, answer, text)
            diagnosis['gap_type'] = gap.get('gap_type', 'unknown')
            diagnosis['root_cause'] = gap.get('gap_details', '')
        except Exception:
            pass

        return diagnosis

    # ═══════════════ STEP 3: LEARN FROM FAILURE ═══════════════

    def _learn_from_failure(self, text: str, question: str,
                            wrong_answer: str, correct_answer: str,
                            diagnosis: dict) -> Optional[Any]:
        """Compose a new understanding from the failure.

        Uses UnderstandingComposer to create a new understanding that
        would produce the CORRECT answer instead of the wrong one.

        The new understanding starts at PROBATION confidence — it must
        be verified before being promoted.
        """
        try:
            new_node = self.composer.compose_from_failure(
                text=text,
                question=question,
                wrong_answer=str(wrong_answer),
                correct_answer=str(correct_answer),
            )

            if new_node is not None:
                # Override confidence to probation level
                new_node.confidence = self.PROBATION_CONFIDENCE
                new_node.source = f'corrected:{diagnosis.get("failure_type", "unknown")}'

                # If we know which understanding failed, add an edge
                failed_id = diagnosis.get('failed_node_id', '')
                if failed_id and failed_id != new_node.id:
                    new_node.add_edge(failed_id, 'corrects')

                logger.info("Created correction understanding: %s (probation=%.2f)",
                           new_node.id, self.PROBATION_CONFIDENCE)

            return new_node

        except Exception as e:
            logger.warning("Failed to compose understanding from failure: %s", e)
            return None

    # ═══════════════ STEP 4: VERIFY CORRECTION ═══════════════

    def _verify_correction(self, new_node: Any, text: str, question: str,
                           correct_answer: str) -> bool:
        """Verify that the new understanding produces the correct answer.

        A correction is verified if:
          1. The new understanding's transformation, when applied to the
             SAME question, produces the correct answer
          2. The new understanding doesn't contradict verified understandings

        v31 fix (P4): Uses SEMANTIC similarity for verification instead of
        exact string match. "delapan" vs "8" should pass verification
        because they are semantically identical.

        This prevents false negatives where the correction IS correct
        but the string representation differs.
        """
        # Test 1: Apply the new understanding to the same question
        try:
            result = self.graph.apply(new_node, text, question)
            if result is not None:
                produced_answer = str(result.get('answer', '')).strip()
                expected_answer = str(correct_answer).strip()

                # v31 fix (P4): Use semantic matching instead of exact string match
                if self._semantic_verify(produced_answer, expected_answer):
                    logger.info("Verification PASSED: new understanding produces correct answer")
                    return True
        except Exception as e:
            logger.debug("Verification apply failed: %s", e)

        # Test 2: Cross-check against existing verified understandings
        # If the new understanding contradicts a high-confidence understanding,
        # it's not verified
        conflicts = self._check_conflicts(new_node)
        if conflicts:
            logger.info("Verification FAILED: conflicts with %d verified understandings",
                        len(conflicts))
            return False

        # If we can't apply the transformation but there are no conflicts,
        # give it a conditional pass — it might work for other questions
        # but we can't verify for this specific one
        logger.info("Verification INCONCLUSIVE: cannot apply transformation, "
                    "no conflicts found — keeping at probation level")
        return False

    def _semantic_verify(self, produced: str, expected: str) -> bool:
        """Verify answers using semantic similarity instead of exact string match.

        v31 fix (P4): The previous implementation used exact string comparison
        which failed when the correct answer was semantically identical but
        textually different (e.g., "delapan" vs "8", "sederhana" vs "Sederhana").

        Verification strategy (in order):
          1. Exact match (case-insensitive) — fastest path
          2. Substring containment — "8" in "8 kelereng"
          3. Semantic similarity via embedding — "delapan" ≈ "8"
        """
        if not produced or not expected:
            return False

        produced_lower = produced.lower()
        expected_lower = expected.lower()

        # Strategy 1: Exact match (case-insensitive)
        if produced_lower == expected_lower:
            return True

        # Strategy 2: Substring containment
        # e.g., "8" in "8 kelereng" or "sederhana" in "sederhana dan hemat"
        if produced_lower in expected_lower or expected_lower in produced_lower:
            return True

        # Strategy 3: Semantic similarity via embedding
        # e.g., "delapan" ≈ "8" (semantically identical)
        try:
            from derivation.understanding_builder import get_shared_graph
            graph = self.graph if self._graph else get_shared_graph()

            # Use the graph's embedding retriever for semantic similarity
            if graph._retriever is not None and graph._retriever.is_available():
                import numpy as np
                embeddings = graph._retriever.model.encode(
                    [produced_lower, expected_lower],
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                similarity = float(np.dot(embeddings[0], embeddings[1]))

                if similarity >= 0.75:
                    logger.info("Semantic verification PASSED: '%s' ≈ '%s' (sim=%.3f)",
                               produced_lower, expected_lower, similarity)
                    return True
                else:
                    logger.debug("Semantic verification below threshold: '%s' vs '%s' (sim=%.3f)",
                                produced_lower, expected_lower, similarity)
        except Exception as e:
            logger.debug("Semantic verification failed, falling back to string match: %s", e)

        return False

    def _check_conflicts(self, new_node: Any) -> List[str]:
        """Check if a new understanding conflicts with existing verified ones.

        Two understandings conflict if:
          - They have the same or overlapping conditions
          - They have different transformation kinds
          - Both have confidence > 0.5

        This is a simple heuristic — real contradiction detection would
        require semantic analysis, which SELF can learn over time.
        """
        conflicts = []
        for node_id, existing in self.graph._nodes.items():
            if node_id == new_node.id:
                continue
            if existing.confidence < 0.5:
                continue  # Not verified, ignore
            if existing.source.startswith('deprecated'):
                continue  # Deprecated, ignore

            # Check for overlapping conditions with different transformation kind
            if (new_node.transformation and existing.transformation and
                new_node.transformation.kind != existing.transformation.kind):
                # Check condition overlap
                new_conds = set(c.lower() for c in new_node.conditions)
                existing_conds = set(c.lower() for c in existing.conditions)
                overlap = new_conds & existing_conds
                if len(overlap) >= 2:  # Substantial overlap
                    conflicts.append(node_id)

        return conflicts

    # ═══════════════ STEP 5: PRUNE ═══════════════

    def _prune_understanding(self, node: Any, diagnosis: dict) -> None:
        """Weaken an understanding that produced a wrong answer.

        The weakening amount depends on:
          1. Whether this is a first failure or a repeat
          2. Whether the understanding has been verified before
          3. The severity of the failure

        The node is NOT deleted — it might be partially correct in
        other contexts. It's just weakened so it's less likely to be
        selected for similar questions in the future.
        """
        if node is None:
            return

        fail_count = self._failure_counts.get(node.id, 1)

        # Determine weakening amount
        if fail_count > 2:
            weaken_amount = self.REPEAT_FAIL_WEAKEN
        elif fail_count > 1:
            weaken_amount = (self.FIRST_FAIL_WEAKEN + self.REPEAT_FAIL_WEAKEN) / 2
        else:
            weaken_amount = self.FIRST_FAIL_WEAKEN

        # Apply weakening
        old_confidence = node.confidence
        node.weaken(weaken_amount)
        new_confidence = node.confidence

        logger.info("Weakened understanding %s: %.3f → %.3f (fail_count=%d)",
                    node.id, old_confidence, new_confidence, fail_count)

        # If below prune threshold, deprecate
        if new_confidence < self.PRUNE_THRESHOLD and not node.source.startswith('deprecated'):
            node.source = f'deprecated:{node.source}'
            node.conditions = []  # Clear conditions to prevent retrieval
            logger.info("Deprecated understanding %s (confidence=%.3f dropped below %.3f)",
                       node.id, new_confidence, self.PRUNE_THRESHOLD)

        # Record feedback on the graph
        self.graph.record_feedback(node.id, correct=False)

    # ═══════════════ STEP 6: RE-ATTEMPT ═══════════════

    def _reattempt_answer(self, text: str, question: str) -> Optional[dict]:
        """Re-attempt to answer a question after correction.

        After a correction has been verified, SELF re-tries to answer
        the question using its improved understanding graph.

        This is a SINGLE re-attempt — no recursion, no infinite loops.
        """
        try:
            # Try understanding pipeline first
            matches = self.graph.find_matching_multi(text, question, top_k=3, threshold=0.15)

            if not matches:
                return None

            # Try applying each understanding individually
            for node, score in matches:
                if node.transformation is None:
                    continue
                if node.source.startswith('deprecated'):
                    continue  # Skip deprecated

                result = self.graph.apply(node, text, question)
                if result and result.get('answer'):
                    result['reattempt'] = True
                    result['corrected_via'] = node.id
                    return result

            # Try composition with Qwen3
            non_deprecated = [
                (n, s) for n, s in matches
                if not n.source.startswith('deprecated')
            ]
            if non_deprecated:
                try:
                    composed = self.composer.compose_answer_from_understandings(
                        text, question, non_deprecated
                    )
                    if composed:
                        return {
                            'answer': composed,
                            'confidence': 0.55,
                            'method': 'correction_composition',
                            'reattempt': True,
                        }
                except Exception as e:
                    logger.debug("Re-attempt composition failed: %s", e)

        except Exception as e:
            logger.warning("Re-attempt answer failed: %s", e)

        return None

    # ═══════════════ STRENGTHEN ON SUCCESS ═══════════════

    def record_success(self, text: str, question: str,
                       answer: dict) -> None:
        """Record that an answer was correct — strengthen the understanding.

        This is the POSITIVE side of self-correction. When SELF gets
        an answer RIGHT, the understanding that produced it should be
        STRENGTHENED to reinforce correct behavior.

        Args:
            text: Source text
            question: The question
            answer: The correct answer dict (with method and confidence)
        """
        method = answer.get('method', '')

        # Find the understanding that produced the correct answer
        node = self._find_failed_understanding(
            text, question, method, answer.get('confidence', 0.5)
        )

        if node is not None:
            # Record positive feedback
            self.graph.record_feedback(node.id, correct=True)
            logger.debug("Strengthened understanding %s after correct answer", node.id)

    # ═══════════════ PERSISTENCE ═══════════════

    def get_correction_history(self, limit: int = 50) -> list:
        """Get recent correction history."""
        return [r.to_dict() for r in self._correction_history[-limit:]]

    def load_state(self, data: dict) -> None:
        """Load correction state from dict."""
        self._correction_count = data.get('correction_count', 0)
        self._verification_count = data.get('verification_count', 0)
        self._successful_corrections = data.get('successful_corrections', 0)
        self._failure_counts = defaultdict(int, data.get('failure_counts', {}))

    def save_state(self) -> dict:
        """Save correction state to dict."""
        return {
            'correction_count': self._correction_count,
            'verification_count': self._verification_count,
            'successful_corrections': self._successful_corrections,
            'failure_counts': dict(self._failure_counts),
        }

    # ═══════════════ DISK PERSISTENCE (v31 fix — P5) ═══════════════

    def _state_path(self) -> str:
        """Get the path for correction state persistence."""
        return os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'correction_state.json'
        )

    def save_state_to_disk(self) -> None:
        """Persist correction state to disk.

        v31 fix (P5): Previously, CorrectionRecord and failure_counts were
        only kept in memory and lost on restart. Now they're persisted so
        SELF remembers its correction history across sessions.

        Called automatically by SelfCore.save().
        """
        try:
            state = {
                'correction_count': self._correction_count,
                'verification_count': self._verification_count,
                'successful_corrections': self._successful_corrections,
                'failure_counts': dict(self._failure_counts),
                'correction_history': [r.to_dict() for r in self._correction_history[-100:]],
            }

            state_path = self._state_path()
            os.makedirs(os.path.dirname(state_path), exist_ok=True)

            # Atomic write
            tmp_path = state_path + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp_path, state_path)

            logger.debug("Correction state persisted to disk (%d history records)",
                        len(self._correction_history))
        except Exception as e:
            logger.warning("Failed to persist correction state: %s", e)

    def load_state_from_disk(self) -> None:
        """Load correction state from disk.

        v31 fix (P5): Load previously persisted correction state so SELF
        remembers its correction history across restarts.

        Called automatically by SelfCore._get_correction_loop() on init.
        """
        try:
            state_path = self._state_path()
            if not os.path.exists(state_path):
                return

            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self._correction_count = state.get('correction_count', 0)
            self._verification_count = state.get('verification_count', 0)
            self._successful_corrections = state.get('successful_corrections', 0)
            self._failure_counts = defaultdict(int, state.get('failure_counts', {}))

            # Restore correction history
            history_data = state.get('correction_history', [])
            for item in history_data[-100:]:
                try:
                    record = CorrectionRecord(
                        question=item.get('question', ''),
                        wrong_answer=item.get('wrong_answer', ''),
                        correct_answer=item.get('correct_answer', ''),
                        failed_node_id=item.get('failed_node_id', ''),
                        new_node_id=item.get('new_node_id', ''),
                        diagnosis=item.get('diagnosis'),
                        verified=item.get('verified', False),
                    )
                    record.timestamp = item.get('timestamp', time.time())
                    self._correction_history.append(record)
                except Exception:
                    pass

            logger.debug("Correction state loaded from disk (%d history, %d failure counts)",
                        len(self._correction_history), len(self._failure_counts))
        except Exception as e:
            logger.debug("Failed to load correction state from disk: %s", e)
