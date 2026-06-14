# @WHO:   self-ai/src/derivation/pattern_learner.py
# @WHAT:  SELF discovers semantic understandings through inner thinking — reusable without LLM
# @PART:  self-ai/derivation
# @ENTRY: PatternLearner (imported by answer_handlers.py, text_comprehension.py)

"""Pattern Learner — SELF discovers reusable semantic understandings.

Philosophy:
    SELF must discover PATTERNS through inner thinking, NOT from machine parsers.
    SELF must find and write its own semantic patterns.

    v26: Now uses UnderstandingNode/Transformation from understanding_builder.py
    instead of the deleted semantic_understanding.py. The UnderstandingGraph
    (with bge-m3 embedding retrieval) is the primary reasoning path.

    This module provides backward-compatible legacy pattern support.
    The PRIMARY reasoning path is UnderstandingGraph.find_matching() with
    UnderstandingRetriever (embedding-based retrieval).

    Flow:
        Teacher teaches (soal + cara + jawaban + kenapa)
            ↓
        SELF observes all examples
            ↓
        SELF discovers understanding through inner thinking (LLM reasoning)
            ↓
        SELF builds UnderstandingNode with Transformation
            ↓
        For new cases: UnderstandingGraph finds match via embedding → applies transformation (NO LLM)
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SimplePattern:
    """Simple pattern object for backward compatibility.

    Replaces SemanticUnderstanding from the deleted semantic_understanding.py.
    This is a minimal implementation that preserves the interface used by
    find_matching_pattern() and apply_pattern().
    """

    def __init__(self, name: str, concept: str = '', abstraction: str = '',
                 conditions: list = None, method: list = None,
                 examples: list = None, question_types: list = None,
                 source: str = 'self_discovered', confidence: float = 0.5):
        self.name = name
        self.concept = concept
        self.abstraction = abstraction
        self.conditions = conditions or []
        self.method = method or []
        self.examples = examples or []
        self.question_types = question_types or []
        self.source = source
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'concept': self.concept,
            'abstraction': self.abstraction,
            'conditions': self.conditions,
            'method': self.method,
            'examples': self.examples,
            'question_types': self.question_types,
            'source': self.source,
            'confidence': self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SimplePattern':
        return cls(
            name=d.get('name', ''),
            concept=d.get('concept', ''),
            abstraction=d.get('abstraction', ''),
            conditions=d.get('conditions', []),
            method=d.get('method', []),
            examples=d.get('examples', []),
            question_types=d.get('question_types', []),
            source=d.get('source', 'self_discovered'),
            confidence=d.get('confidence', 0.5),
        )


class PatternLearner:
    """SELF discovers semantic understandings through inner thinking.

    v26: This module now provides LEGACY pattern support only.
    The PRIMARY reasoning path is:
        UnderstandingGraph.find_matching() → UnderstandingRetriever (bge-m3)
        → apply transformation WITHOUT LLM

    This class is kept for backward compatibility and as a fallback when
    the understanding graph is not available.
    """

    def __init__(self, llm_engine=None, patterns_file: str = None,
                 understandings_file: str = None, embedding_model=None):
        self._llm_engine = llm_engine
        self._patterns_file = patterns_file or os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'self_discovered_patterns.json'
        )

        # v26: Understanding graph is now managed by UnderstandingBuilder
        # This module only stores legacy patterns
        self._patterns = {}
        self._load_patterns()

    # ═══════════════ LEARNING ═══════════════

    def learn_from_lessons(self, lessons):
        """Observe teaching lessons and discover patterns through inner thinking.

        @FLOW:     PATTERN_LEARN
        @CALLS:    _discover_pattern_through_thinking()
        @MUTATES:  _patterns
        """
        for q_type in lessons.get_types():
            type_lessons = lessons.get_by_type(q_type)
            if len(type_lessons) < 1:
                continue

            # Legacy: store as pattern for backward compat
            discovered = self._discover_pattern_through_thinking(q_type, type_lessons)
            if discovered:
                self._patterns[discovered.name] = discovered

        # Persist
        self._save_patterns()

    # ═══════════════ MATCHING ═══════════════

    def find_matching_pattern(self, text: str, question: str,
                               question_type: str) -> Optional[SimplePattern]:
        """Find a SELF-discovered pattern that matches the current question.

        DEPRECATED: Use UnderstandingGraph.find_matching() instead.
        Kept for backward compatibility.
        """
        if not self._patterns:
            return None

        text_lower = text.lower()
        question_lower = question.lower()

        scored = []
        for name, pattern in self._patterns.items():
            if question_type not in pattern.question_types:
                continue

            signal_matches = 0
            for signal in pattern.conditions:
                sig_lower = signal.lower() if isinstance(signal, str) else str(signal).lower()
                if sig_lower in text_lower or sig_lower in question_lower:
                    signal_matches += 1

            if signal_matches > 0:
                scored.append((pattern, signal_matches))

        if not scored:
            for name, pattern in self._patterns.items():
                if question_type in pattern.question_types:
                    scored.append((pattern, 0.5))

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]

        return None

    def apply_pattern(self, pattern, text: str,
                       question: str, question_type: str) -> Optional[dict]:
        """Apply a SELF-discovered pattern to answer a question.

        DEPRECATED: Use UnderstandingGraph.apply() instead.
        Kept for backward compatibility.
        """
        desc = getattr(pattern, 'description', None) or getattr(pattern, 'concept', '')
        reasoning = getattr(pattern, 'reasoning_template', None) or '; '.join(getattr(pattern, 'method', []))
        pattern_context = (
            f"Pattern: {pattern.name}\n"
            f"Description: {desc}\n"
            f"Reasoning: {reasoning}\n"
            f"Examples: {'; '.join(pattern.examples[:3])}"
        )

        prompt = (
            f"Based on this self-discovered pattern:\n{pattern_context}\n\n"
            f"Question type: {question_type}\n"
            f"Text: {text[:500]}\n"
            f"Question: {question}\n\n"
            "Apply the pattern to answer the question. "
            "Give ONLY the answer, nothing else."
        )

        if self._llm_engine is not None:
            try:
                answer = None
                if hasattr(self._llm_engine, '_try_sdk_fallback'):
                    answer = self._llm_engine._try_sdk_fallback(text, question, prompt)
                if not answer and hasattr(self._llm_engine, '_try_local_qwen'):
                    answer = self._llm_engine._try_local_qwen(prompt)

                if answer:
                    return {
                        'answer': answer.strip(),
                        'confidence': pattern.confidence,
                        'method': f'self_discovered_{pattern.name}',
                        'explanation': f'SELF applied pattern: {desc[:80]}',
                        'source': pattern.source,
                    }
            except Exception as e:
                logger.warning("Pattern application error: %s", e)

        return None

    def get_patterns(self) -> dict:
        """Get all SELF-discovered patterns."""
        return {name: p.to_dict() for name, p in self._patterns.items()}

    def get_pattern(self, name: str) -> Optional[SimplePattern]:
        """Get a specific SELF-discovered pattern."""
        return self._patterns.get(name)

    # ═══════════════ INTERNAL: LEGACY PATTERN DISCOVERY ═══════════════

    def _discover_pattern_through_thinking(self, question_type: str,
                                            lessons: list) -> Optional[SimplePattern]:
        """Legacy pattern discovery — kept for backward compatibility."""
        # Build observation from teaching examples
        observations = []
        for lesson in lessons:
            obs = (
                f"SOAL: {lesson.problem}\n"
                f"CARA: {'; '.join(lesson.solution_steps)}\n"
                f"JAWABAN: {lesson.answer}\n"
                f"KENAPA: {lesson.explanation_why}"
            )
            observations.append(obs)

        # Try LLM first
        if self._llm_engine is not None:
            try:
                pattern = self._discover_via_llm(question_type, observations)
                if pattern:
                    return pattern
            except Exception as e:
                logger.warning("LLM pattern discovery failed: %s — using observation", e)

        return self._discover_via_observation(question_type, lessons)

    def _discover_via_llm(self, question_type: str, observations: list) -> Optional[SimplePattern]:
        """Use LLM to discover the semantic pattern connecting examples."""
        prompt = (
            "You are SELF, a meta-learning AI. Observe these teaching examples "
            "and discover the SEMANTIC PATTERN that connects them.\n\n"
            f"Question type: {question_type}\n\n"
            "Examples:\n" + "\n---\n".join(observations) + "\n\n"
            "Discover the pattern. Write:\n"
            "1. PATTERN NAME (short, snake_case)\n"
            "2. DESCRIPTION (what pattern SELF discovered)\n"
            "3. TRIGGER SIGNALS (words/phrases that suggest this pattern)\n"
            "4. REASONING TEMPLATE (how to reason when this pattern applies)\n\n"
            "Format: JSON with keys: name, description, trigger_signals, reasoning_template"
        )

        try:
            answer = None
            if hasattr(self._llm_engine, '_try_sdk_fallback'):
                answer = self._llm_engine._try_sdk_fallback('', '', prompt)
            if not answer and hasattr(self._llm_engine, '_try_local_qwen'):
                answer = self._llm_engine._try_local_qwen(prompt)

            if answer:
                return self._parse_llm_pattern(answer, question_type, observations)
        except Exception as e:
            logger.warning("LLM pattern discovery error: %s", e)

        return None

    def _parse_llm_pattern(self, llm_output: str, question_type: str,
                            observations: list) -> Optional[SimplePattern]:
        """Parse LLM output into a SimplePattern."""
        import re as _re

        json_match = _re.search(r'\{[^{}]+\}', llm_output, _re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return SimplePattern(
                    name=data.get('name', f'{question_type}_pattern'),
                    concept=data.get('description', ''),
                    abstraction=data.get('description', ''),
                    conditions=data.get('trigger_signals', []),
                    method=[data.get('reasoning_template', '')],
                    examples=[obs[:200] for obs in observations[:3]],
                    question_types=[question_type],
                    source='self_discovered_from_teaching',
                    confidence=0.6,
                )
            except json.JSONDecodeError:
                pass

        name = f'{question_type}_pattern'
        description = ''
        trigger_signals = []
        reasoning_template = ''

        for line in llm_output.strip().split('\n'):
            line = line.strip()
            if 'name' in line.lower() and ':' in line:
                name = line.split(':', 1)[1].strip().strip('"\'')
            elif 'description' in line.lower() and ':' in line:
                description = line.split(':', 1)[1].strip().strip('"\'')
            elif 'trigger' in line.lower() and ':' in line:
                signals_str = line.split(':', 1)[1].strip()
                trigger_signals = [s.strip().strip('"\'') for s in signals_str.split(',')]
            elif 'reasoning' in line.lower() and ':' in line:
                reasoning_template = line.split(':', 1)[1].strip().strip('"\'')

        if description or trigger_signals or reasoning_template:
            return SimplePattern(
                name=name,
                concept=description or f'Pattern for {question_type}',
                abstraction=description or f'Pattern for {question_type}',
                conditions=trigger_signals,
                method=[reasoning_template] if reasoning_template else [],
                examples=[obs[:200] for obs in observations[:3]],
                question_types=[question_type],
                source='self_discovered_from_teaching',
                confidence=0.55,
            )

        return None

    def _discover_via_observation(self, question_type: str,
                                   lessons: list) -> Optional[SimplePattern]:
        """Discover pattern through basic observation (no LLM needed)."""
        trigger_signals = set()
        reasoning_parts = []
        example_summaries = []

        for lesson in lessons:
            problem_words = lesson.problem.lower().split()
            stop_words = {'apa', 'yang', 'di', 'ke', 'dari', 'untuk', 'dengan',
                          'adalah', 'itu', 'ini', 'dan', 'atau', 'juga', 'tidak',
                          'akan', 'telah', 'sudah', 'dalam', 'pada', 'oleh',
                          'bagaimana', 'mengapa', 'kapan', 'dimana', 'siapa'}
            meaningful_words = [w for w in problem_words
                                if len(w) > 3 and w not in stop_words]
            trigger_signals.update(meaningful_words[:5])

            if lesson.solution_steps:
                reasoning_parts.extend(lesson.solution_steps[:2])

            example_summaries.append(f"Q: {lesson.problem[:60]} → A: {lesson.answer[:40]}")

        pattern_name = f'{question_type}_observed_pattern'
        description = f'Pattern for {question_type} discovered from {len(lessons)} teaching examples'

        if reasoning_parts:
            reasoning_template = ' → '.join(reasoning_parts[:3])
        else:
            reasoning_template = f'Observe text, identify {question_type} signals, apply reasoning'

        return SimplePattern(
            name=pattern_name,
            concept=description,
            abstraction=description,
            conditions=list(trigger_signals)[:10],
            method=[reasoning_template],
            examples=example_summaries[:5],
            question_types=[question_type],
            source='self_discovered_from_teaching',
            confidence=0.5,
        )

    # ═══════════════ PERSISTENCE ═══════════════

    def _save_patterns(self):
        """Persist SELF's discovered patterns to disk (atomic write)."""
        try:
            os.makedirs(os.path.dirname(self._patterns_file), exist_ok=True)
            data = {name: p.to_dict() if hasattr(p, 'to_dict') else {}
                    for name, p in self._patterns.items()}
            tmp = self._patterns_file + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._patterns_file)
        except Exception as e:
            logger.warning("Failed to save patterns: %s", e)

    def _load_patterns(self):
        """Load SELF's previously discovered patterns."""
        if os.path.exists(self._patterns_file):
            try:
                with open(self._patterns_file) as f:
                    data = json.load(f)
                for name, pd in data.items():
                    self._patterns[name] = SimplePattern.from_dict(pd)
                logger.info("Loaded %d SELF-discovered patterns", len(self._patterns))
            except Exception as e:
                logger.warning("Failed to load patterns: %s", e)
