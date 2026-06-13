"""SELF - 8-Layer Cognitive Architecture
Layers: Sensory → Difference → Concept → Consistency → Axiom → Memory → Derivation → Curiosity
"""
import json, os, time, re, logging
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# v32: Removed _SimpleConsistencyChecker and _StubPatternDiscovery.
# These were LEGACY hardcoded components that violated SELF-AI's vision:
#   - _SimpleConsistencyChecker used hardcoded NEGATION_WORDS and ANTONYM_MAP
#   - _StubPatternDiscovery was a dead stub for a deleted module
# Consistency checking is now done via embedding-based semantic similarity
# (see text_comprehension.semantic_match and self_correction._semantic_verify).

class SelfCore:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Layer states
        self.sensory_buffer = []
        self.difference_log = []
        self.concept_graph = {}  # concept_id -> {embedding, metadata}
        self.consistency_checks = []
        self.axioms = {}  # axiom_id -> {subject, predicate, object, confidence, source}
        self.observations = []  # Memory store
        self.derivation_results = []
        self.curiosity_queue = []
        
        # Emergent relation registry
        self.relation_registry = {}  # relation_name -> {count, examples, operational_type}
        
        # Adaptive parameters
        self.confidence_alpha = 0.5
        self.learning_rate = 0.01
        self.feedback_history = []
        
        # Import sub-modules (lazy)
        self._derivation_engine = None
        self._composer = None
        self._correction_loop = None  # v31: Shared SelfCorrectionLoop
        
    @property
    def composer(self):
        """Lazy-init shared UnderstandingComposer (singleton via get_shared_composer)."""
        if self._composer is None:
            from derivation.understanding_composer import get_shared_composer
            self._composer = get_shared_composer()
        return self._composer

    @property
    def derivation_engine(self):
        if self._derivation_engine is None:
            from derivation.engine import DerivationEngine
            self._derivation_engine = DerivationEngine(self)
        return self._derivation_engine

    def _get_correction_loop(self):
        """Get the shared SelfCorrectionLoop instance.

        v31 fix (P1): Instead of creating a new SelfCorrectionLoop each time,
        this method uses the SAME instance that AnswerHandlers uses.
        This ensures correction history, failure counts, and stats are
        shared across all components.

        The instance is lazy-initialized on first access and cached.
        It also loads persisted correction state from disk (P5 fix).
        """
        if self._correction_loop is not None:
            return self._correction_loop

        try:
            from derivation.self_correction import SelfCorrectionLoop
            from derivation.understanding_builder import get_shared_graph

            loop = SelfCorrectionLoop(graph=get_shared_graph())
            # v31 P5 fix: Load persisted correction state from disk
            loop.load_state_from_disk()
            self._correction_loop = loop
            return loop
        except ImportError:
            logger.debug("SelfCorrectionLoop not available")
            return None
        except Exception as e:
            logger.warning("Failed to init shared SelfCorrectionLoop: %s", e)
            return None
    

    
    def process(self, text: str) -> dict:
        """Process input through all 8 layers"""
        # Layer 1: Sensory
        sensory_result = self._sensory(text)
        
        # Layer 2: Difference
        diff_result = self._difference(sensory_result)
        
        # Layer 3: Concept
        concept_result = self._concept(diff_result)
        
        # Layer 4: Consistency
        consistency_result = self._consistency(concept_result)
        
        # Layer 5: Axiom
        axiom_result = self._axiom(concept_result, consistency_result)
        
        # Layer 6: Memory
        memory_result = self._memory(axiom_result)
        
        # Layer 7: Derivation
        derivation_result = self._derivation(memory_result)
        
        # Layer 8: Curiosity
        curiosity_result = self._curiosity(derivation_result)
        
        return {
            'sensory': sensory_result,
            'difference': diff_result,
            'concept': concept_result,
            'consistency': consistency_result,
            'axiom': axiom_result,
            'memory': memory_result,
            'derivation': derivation_result,
            'curiosity': curiosity_result
        }
    
    def _sensory(self, text: str) -> dict:
        """Layer 1: Receive and preprocess input"""
        entry = {'text': text, 'timestamp': time.time(), 'id': len(self.sensory_buffer)}
        self.sensory_buffer.append(entry)
        return entry
    
    def _difference(self, sensory_result: dict) -> dict:
        """Layer 2: Detect what's new/changed"""
        text = sensory_result['text']
        known_texts = [o.get('text', '') for o in self.observations[-50:]]
        is_new = text not in known_texts
        novelty_score = 1.0 if is_new else 0.3
        self.difference_log.append({'text': text, 'is_new': is_new, 'novelty': novelty_score})
        return {**sensory_result, 'is_new': is_new, 'novelty': novelty_score}
    
    def _concept(self, diff_result: dict) -> dict:
        """Layer 3: Map to concepts"""
        text = diff_result['text']
        # Extract potential concepts (nouns, numbers, verbs)
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        words = re.findall(r'[a-zA-Z\u00C0-\u024F\u0400-\u04FF\u4e00-\u9fff\u0600-\u06FF]+', text.lower())
        
        concepts = []
        for num in numbers:
            concepts.append({'type': 'quantity', 'value': float(num) if '.' in num else int(num), 'raw': num})
        for word in words:
            if word not in concepts:
                concepts.append({'type': 'entity', 'value': word, 'raw': word})
        
        return {**diff_result, 'concepts': concepts}
    
    def _consistency(self, concept_result: dict) -> dict:
        """Layer 4: Check for contradictions using embedding-based semantic similarity.

        v32: Removed hardcoded _SimpleConsistencyChecker (NEGATION_WORDS, ANTONYM_MAP).
        Now uses embedding similarity to detect contradictions between the input
        text and existing axioms. This is consistent with SELF-AI's vision of
        NO hardcoded rules — consistency is detected semantically.

        If embedding model is unavailable, contradictions are not detected
        (returns consistent=True) — SELF will learn to detect them through
        the self-correction loop when answers are wrong.
        """
        text = concept_result.get('text', '')
        contradictions = []

        # Use embedding-based contradiction detection
        try:
            from derivation.understanding_builder import get_shared_graph
            graph = get_shared_graph()
            if graph._retriever is not None and graph._retriever.is_available():
                import numpy as np
                text_emb = graph._retriever.model.encode(
                    [text], show_progress_bar=False, normalize_embeddings=True
                )[0]

                for axiom_id, axiom in self.axioms.items():
                    axiom_text = f"{axiom.get('subject', '')} {axiom.get('predicate', '')} {axiom.get('object', '')}"
                    if not axiom_text.strip():
                        continue
                    axiom_emb = graph._retriever.model.encode(
                        [axiom_text], show_progress_bar=False, normalize_embeddings=True
                    )[0]
                    similarity = float(np.dot(text_emb, axiom_emb))

                    # High similarity + opposing structure → contradiction
                    # This is a simplified heuristic; SELF will learn better
                    # contradiction detection through the self-correction loop.
                    # For now, we use: if text is very similar to axiom but
                    # confidence is low, flag as potential contradiction.
                    if similarity > 0.85 and axiom.get('confidence', 0.5) < 0.3:
                        contradictions.append({
                            'axiom_id': axiom_id,
                            'axiom': axiom,
                            'reason': f'Semantic conflict detected (sim={similarity:.2f})',
                            'confidence': similarity * 0.7,
                        })
        except Exception:
            # If embedding model is unavailable, skip contradiction detection.
            # SELF will learn from failures through the self-correction loop.
            pass

        is_consistent = len(contradictions) == 0
        self.consistency_checks.append({'text': text, 'consistent': is_consistent, 'contradictions': contradictions})
        return {**concept_result, 'consistent': is_consistent, 'contradictions': contradictions}
    
    def _axiom(self, concept_result: dict, consistency_result: dict) -> dict:
        """Layer 5: Manage known facts/rules"""
        concepts = concept_result.get('concepts', [])
        text = concept_result.get('text', '')
        
        # Try to extract triplets (subject-predicate-object)
        triplets = self._extract_triplets(text, concepts)
        
        new_axioms = []
        for triplet in triplets:
            # Check if similar axiom exists (node similarity + axiom ID lookup - Bug #5 fix)
            existing_id = self._find_similar_axiom(triplet)
            if existing_id is None:
                axiom_id = f"ax_{len(self.axioms)}_{hash(frozenset(triplet.items())) & 0xFFFF}"
                self.axioms[axiom_id] = {
                    'subject': triplet.get('subject', ''),
                    'predicate': triplet.get('predicate', ''),
                    'object': triplet.get('object', ''),
                    'confidence': 0.5,
                    'source': 'observation',
                    'roles': triplet.get('roles', {})
                }
                new_axioms.append(axiom_id)
        
        return {**consistency_result, 'triplets': triplets, 'new_axioms': new_axioms}
    
    def _extract_triplets(self, text: str, concepts: list) -> list:
        """Extract subject-predicate-object triplets from text"""
        import re
        triplets = []
        numbers = [c for c in concepts if c['type'] == 'quantity']
        entities = [c for c in concepts if c['type'] == 'entity']
        
        if len(numbers) >= 2:
            # Might be an operational relationship
            # Use grammar parser for role detection
            try:
                from grammar.parser import GrammarParser
                parser = GrammarParser(self)
                parsed = parser.parse(text)
                if parsed.get('triplets'):
                    triplets.extend(parsed['triplets'])
            except ImportError:
                try:
                    from grammar.simple_parser import SimpleParser
                    parser = SimpleParser(self)
                    parsed = parser.parse(text)
                    if parsed.get('triplets'):
                        triplets.extend(parsed['triplets'])
                except Exception:
                    pass
            except Exception:
                pass
        
        return triplets
    
    def _find_similar_axiom(self, triplet: dict) -> Optional[str]:
        """Find existing axiom by similarity - Bug #5 fix"""
        sub = triplet.get('subject', '')
        pred = triplet.get('predicate', '')
        obj = triplet.get('object', '')
        
        for axiom_id, axiom in self.axioms.items():
            # Node similarity check
            if (axiom.get('subject', '') == sub and 
                axiom.get('predicate', '') == pred and
                axiom.get('object', '') == obj):
                return axiom_id
        return None
    
    def _memory(self, axiom_result: dict) -> dict:
        """Layer 6: Store and retrieve observations"""
        observation = {
            'text': axiom_result.get('text', ''),
            'concepts': axiom_result.get('concepts', []),
            'triplets': axiom_result.get('triplets', []),
            'new_axioms': axiom_result.get('new_axioms', []),
            'timestamp': time.time()
        }
        self.observations.append(observation)
        return {**axiom_result, 'stored': True}
    
    def _derivation(self, memory_result: dict) -> dict:
        """Layer 7: Compute answers.

        v28: If answer confidence is low and novelty is high,
        observe the novel input to potentially build new understanding.
        """
        text = memory_result.get('text', '')
        
        # v32: Check if it's a question using embedding-based detection.
        # Removed hardcoded question word list — SELF detects questions semantically.
        is_question = '?' in text
        if not is_question:
            try:
                from derivation.understanding_builder import get_shared_graph
                graph = get_shared_graph()
                if graph._retriever is not None and graph._retriever.is_available():
                    import numpy as np
                    q_emb = graph._retriever.model.encode(
                        [text], show_progress_bar=False, normalize_embeddings=True
                    )[0]
                    # Use a prototypical question embedding for comparison
                    q_proto = graph._retriever.model.encode(
                        ['Siapa apa dimana kapan berapa mengapa bagaimana?'],
                        show_progress_bar=False, normalize_embeddings=True
                    )[0]
                    q_sim = float(np.dot(q_emb, q_proto))
                    is_question = q_sim > 0.5
            except Exception:
                pass

        if is_question:
            try:
                # v15 fix: Use cached property instead of creating new engine each call
                result = self.derivation_engine.derive(text, memory_result)
                self.derivation_results.append(result)

                derivation_result = {**memory_result, 'answer': result.get('answer'), 'confidence': result.get('confidence', 0.0), 'method': result.get('method', 'unknown')}

                # v28: If answer confidence is low and novelty is high, observe
                confidence = result.get('confidence', 1.0)
                novelty = memory_result.get('novelty', 0)
                if confidence < 0.5 and novelty > 0.5:
                    self._observe_novel(memory_result)

                return derivation_result
            except Exception as e:
                return {**memory_result, 'answer': None, 'error': str(e)}
        
        return {**memory_result, 'answer': None, 'reason': 'not_a_question'}
    
    def _curiosity(self, derivation_result: dict) -> dict:
        """Layer 8: Generate questions for unknowns"""
        questions = []
        
        # If answer confidence is low, generate follow-up questions
        confidence = derivation_result.get('confidence', 1.0)
        if confidence < 0.5:
            questions.append(f"How can I be more certain about: {derivation_result.get('text', '')}?")
        
        # If new concepts were found, ask about them
        for concept in derivation_result.get('concepts', []):
            if concept['type'] == 'entity' and concept['value'] not in [a.get('subject', '') for a in self.axioms.values()]:
                questions.append(f"What is {concept['value']}?")
        
        self.curiosity_queue.extend(questions)
        return {**derivation_result, 'curiosity_questions': questions}
    
    def provide_feedback(self, text: str, correct_answer: Any, predicted_answer: Any, question: str = '',
                          answer_method: str = '', answer_confidence: float = 0.0):
        """Provide feedback to improve learning.

        v28: If the answer was wrong, try to learn from the failure by
        composing a new understanding via UnderstandingComposer.

        v29: Added question parameter for better failure learning context.

        v30: Full self-correction loop — when feedback shows a wrong answer,
        SELF runs the complete correction cycle:
          1. Detect which understanding failed
          2. Diagnose the failure
          3. Compose new understanding
          4. Verify the new understanding
          5. Prune the failed understanding
          6. Re-attempt the answer

        Also records success to STRENGTHEN correct understandings.
        """
        from config.thresholds import AdaptiveThresholds
        thresholds = AdaptiveThresholds.get_instance()

        is_correct = str(correct_answer) == str(predicted_answer)
        self.feedback_history.append({
            'text': text,
            'question': question,
            'correct': is_correct,
            'expected': correct_answer,
            'predicted': predicted_answer,
            'answer_method': answer_method,
            'answer_confidence': answer_confidence,
            'timestamp': time.time()
        })

        # Adjust confidence alpha based on feedback
        if len(self.feedback_history) > 5:
            recent = self.feedback_history[-10:]
            accuracy = sum(1 for f in recent if f['correct']) / len(recent)
            # v15 fix: Corrected logic — when accuracy is high, trust learned rules more;
            # when accuracy is low, give other strategies more weight (reduce alpha)
            self.confidence_alpha = max(0.1, min(0.9, 0.5 + (accuracy - 0.5) * 0.2))

        # Update thresholds
        thresholds.update_from_feedback(is_correct)

        # v30: Full self-correction loop
        if not is_correct:
            self._run_self_correction(text, question, str(predicted_answer),
                                      str(correct_answer), answer_method, answer_confidence)
        else:
            # v30: Strengthen the understanding that produced the correct answer
            self._record_success(text, question, answer_method, answer_confidence)

    def teach(self, problem: str, solution_steps: list, answer: str,
              explanation_why: str, question_type: str = ''):
        """Teach SELF through structured examples.

        Teaching goes through:
        1. UnderstandingComposer builds UnderstandingNode from the lesson
        2. Node is added to the UnderstandingGraph
        3. Graph evolves — SELF now has new understanding

        Args:
            problem: The question/soal being asked
            solution_steps: How to solve it (cara penyelesaian)
            answer: The correct answer (jawaban)
            explanation_why: WHY this is the answer (penjelasan kenapa)
            question_type: Category of question (e.g., 'peribahasa', 'ide_pokok')
        """
        from derivation.teaching_lessons import TeachingLesson
        lesson = TeachingLesson(
            problem=problem,
            solution_steps=solution_steps,
            answer=answer,
            explanation_why=explanation_why,
            question_type=question_type,
        )

        # Try UnderstandingComposer first (Qwen3 builds understanding)
        try:
            node = self.composer.compose_from_teaching(lesson)
            if node is not None:
                logger.info("SELF learned from teaching: %s", node.name)
                return node
        except ImportError:
            logger.debug("UnderstandingComposer not available for teach()")
        except Exception as e:
            logger.warning("UnderstandingComposer failed in teach(): %s", e)

        # Fallback: Use UnderstandingBuilder (rule-based observation)
        try:
            from derivation.understanding_builder import UnderstandingBuilder
            builder = UnderstandingBuilder()
            node = builder.observe(lesson)
            if node is not None:
                logger.info("SELF observed teaching lesson: %s", node.name)
                return node
        except Exception as e:
            logger.warning("UnderstandingBuilder failed in teach(): %s", e)

        return None

    def _run_self_correction(self, text: str, question: str, wrong_answer: str,
                              correct_answer: str, answer_method: str = '',
                              answer_confidence: float = 0.0):
        """Run the full self-correction loop when feedback shows a wrong answer.

        v31 fix: Uses the SHARED SelfCorrectionLoop instance from
        AnswerHandlers instead of creating a new one each time.
        This ensures correction history, failure counts, and stats are
        preserved across calls.

        v30: This replaces the simple _learn_from_failure with a complete
        self-correction cycle that:
          1. Finds which understanding produced the wrong answer
          2. Diagnoses the failure
          3. Composes new understanding
          4. VERIFIES the new understanding before committing
          5. PRUNES the failed understanding
          6. Re-attempts the answer
        """
        try:
            # v31 fix: Use the shared correction loop instead of creating new
            loop = self._get_correction_loop()
            if loop is None:
                self._learn_from_failure_simple(text, question, wrong_answer, correct_answer)
                return

            result = loop.correct(
                text=text,
                question=question,
                wrong_answer=wrong_answer,
                correct_answer=correct_answer,
                answer_method=answer_method,
                answer_confidence=answer_confidence,
            )

            if result.get('corrected'):
                logger.info("SELF-CORRECTION SUCCEEDED: question='%s...' new_understanding=%s",
                           question[:50], result.get('new_understanding_id', '?'))
            else:
                # Fall back to simple learning if full correction didn't verify
                logger.info("SELF-CORRECTION not verified, falling back to simple learning")
                self._learn_from_failure_simple(text, question, wrong_answer, correct_answer)

        except ImportError:
            logger.debug("SelfCorrectionLoop not available — using simple failure learning")
            self._learn_from_failure_simple(text, question, wrong_answer, correct_answer)
        except Exception as e:
            logger.warning("Self-correction loop failed: %s — falling back to simple learning", e)
            self._learn_from_failure_simple(text, question, wrong_answer, correct_answer)

    def _learn_from_failure_simple(self, text: str, question: str,
                                    wrong_answer: str, correct_answer: str):
        """Simple failure learning — fallback when full correction loop is unavailable.

        v30: This is the v28-v29 behavior preserved as a fallback.
        Just composes a new understanding without verification.
        """
        try:
            node = self.composer.compose_from_failure(
                text=text,
                question=question,
                wrong_answer=str(wrong_answer),
                correct_answer=str(correct_answer),
            )
            if node is not None:
                logger.info("SELF learned from failure (simple): %s", node.name)
        except ImportError:
            logger.debug("UnderstandingComposer not available for _learn_from_failure")
        except Exception as e:
            logger.debug("Failed to learn from failure: %s", e)

    def _record_success(self, text: str, question: str,
                         answer_method: str, answer_confidence: float):
        """Record that an answer was correct — strengthen the responsible understanding.

        v31 fix: Uses the SHARED SelfCorrectionLoop instance.

        v30: Positive side of self-correction. When SELF gets it RIGHT,
        the understanding that produced the answer is STRENGTHENED.
        """
        try:
            # v31 fix: Use the shared correction loop instead of creating new
            loop = self._get_correction_loop()
            if loop is not None:
                loop.record_success(
                    text=text, question=question,
                    answer={'answer': None, 'confidence': answer_confidence, 'method': answer_method}
                )
        except Exception as e:
            logger.debug("Failed to record success for strengthening: %s", e)

    def _observe_novel(self, memory_result: dict):
        """Observe a novel input to potentially build new understanding.

        v28: When SELF encounters a novel input with low confidence,
        it tries to observe it and extract understanding.

        This is a private method called from _derivation().
        """
        text = memory_result.get('text', '')
        novelty = memory_result.get('novelty', 0)

        if not text or novelty < 0.7:
            return

        try:
            # v29 fix: Extract question from the text for better context
            question = ''
            if '?' in text:
                # Try to extract the question part
                parts = text.split('?')
                question = parts[0].strip() if parts else text
            else:
                question = text

            observation = {
                'text': text,
                'question': question,
                'novelty_score': novelty,
                'concepts': memory_result.get('concepts', []),
                'answer': memory_result.get('answer', ''),
            }
            node = self.composer.compose_from_observation(observation)
            if node is not None:
                logger.info("SELF observed novel input: %s", node.name)
        except ImportError:
            logger.debug("UnderstandingComposer not available for _observe_novel")
        except Exception as e:
            logger.debug("Failed to observe novel input: %s", e)
    
    def save(self, filepath: str = None):
        """Save state to JSON - persistence with atomic write
        v15 fix: Write to temp file first, then rename — prevents corruption if
        the process crashes mid-write.

        v31 fix: Uses shared SelfCorrectionLoop for pruning.
        Also persists correction state to disk (P5 fix).
        """
        import tempfile

        # v31 fix: Prune using shared correction loop (preserves state)
        try:
            loop = self._get_correction_loop()
            if loop is not None:
                pruned = loop.prune_deprecated()
                if pruned > 0:
                    logger.info("Pruned %d deprecated understandings before save", pruned)
                # v31 P5 fix: Persist correction state to disk
                loop.save_state_to_disk()
        except Exception:
            pass  # Non-critical

        filepath = filepath or os.path.join(self.data_dir, 'self_state.json')
        state = {
            'axioms': self.axioms,
            'observations': self.observations[-100:],  # Keep last 100
            'relation_registry': self.relation_registry,
            'confidence_alpha': self.confidence_alpha,
            'learning_rate': self.learning_rate,
            'feedback_history': self.feedback_history[-50:],
            'concept_graph': self.concept_graph,
            'curiosity_queue': self.curiosity_queue[-50:],
            'derivation_results': self.derivation_results[-50:],
        }
        # Atomic write: write to temp file, then replace (Windows-robust)
        dir_name = os.path.dirname(filepath)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            try:
                os.replace(tmp_path, filepath)
            except OSError:
                # Windows: target may be locked — try removing it first
                try:
                    os.unlink(filepath)
                except OSError:
                    pass
                os.replace(tmp_path, filepath)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    
    def load(self, filepath: str = None):
        """Load state from JSON - persistence"""
        filepath = filepath or os.path.join(self.data_dir, 'self_state.json')
        if os.path.exists(filepath):
            with open(filepath) as f:
                state = json.load(f)
            self.axioms = state.get('axioms', {})
            self.observations = state.get('observations', [])
            self.relation_registry = state.get('relation_registry', {})
            self.confidence_alpha = state.get('confidence_alpha', 0.5)
            self.learning_rate = state.get('learning_rate', 0.01)
            self.feedback_history = state.get('feedback_history', [])
            self.concept_graph = state.get('concept_graph', {})
            self.curiosity_queue = state.get('curiosity_queue', [])
            self.derivation_results = state.get('derivation_results', [])


# Singleton
_self_instance = None

def get_self(data_dir: str = None) -> SelfCore:
    global _self_instance
    if _self_instance is None:
        _self_instance = SelfCore(data_dir)
    return _self_instance
