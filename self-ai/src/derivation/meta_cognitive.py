# @WHO:   self-ai/src/derivation/meta_cognitive.py
# @WHAT:  Meta-Cognitive Self-Reflection — system observes its own reasoning
# @PART:  self-ai/derivation
# @ENTRY: MetaCognitiveMonitor.reflect(), MetaCognitiveMonitor.identify_gaps()

"""Meta-Cognitive Self-Reflection — Idea 1 from deductive thinking.

Core principle: "What don't I know about why I don't know?"

Instead of blindly falling back to LLM when confidence is low,
the meta-cognitive monitor OBSERVES the reasoning process and
identifies SPECIFIC gaps in knowledge. This transforms the LLM
fallback from "blind retry" to "targeted knowledge acquisition".

Components:
  1. ReflectionLog — records reasoning failures with structured diagnostics
  2. GapIdentifier — classifies WHAT knowledge is missing
  3. TargetedQueryGenerator — creates specific LLM prompts for missing knowledge
  4. AdaptiveStrategySelector — learns which strategies work for which gap types

This follows the "NO hardcoded rules" philosophy:
  - Gap types are DISCOVERED from observation, not predefined
  - Strategy effectiveness is LEARNED from experience
  - LLM prompts are CONSTRUCTED based on identified gaps

v18: First implementation.
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any, List
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class ReflectionLog:
    """Record of a reasoning attempt — what worked, what didn't, why.

    This is the "memory of mistakes" — the system remembers what
    it got wrong and WHY, so it can avoid the same mistake patterns.
    """

    def __init__(self):
        self.entries = []
        self._gap_patterns = {}  # Learned gap patterns

    def record(self, question: str, answer: dict, gap_type: str,
               gap_details: str, strategy_used: str, success: bool):
        """Record a reasoning attempt.

        Args:
            question: The question that was attempted
            answer: The answer dict (with confidence, method, etc.)
            gap_type: Category of knowledge gap (if any)
            gap_details: Specific description of what was missing
            strategy_used: Which derivation strategy was used
            success: Whether the answer was correct
        """
        entry = {
            'question': question[:200],
            'answer_confidence': answer.get('confidence', 0.0),
            'answer_method': answer.get('method', 'unknown'),
            'gap_type': gap_type,
            'gap_details': gap_details[:200],
            'strategy_used': strategy_used,
            'success': success,
            'timestamp': datetime.now().isoformat(),
        }
        self.entries.append(entry)

        # Learn gap patterns: if this gap type appears repeatedly,
        # record what kind of questions trigger it
        if gap_type:
            key = gap_type
            if key not in self._gap_patterns:
                self._gap_patterns[key] = {
                    'count': 0,
                    'failed_strategies': defaultdict(int),
                    'question_patterns': [],
                    'success_rate': {'hit': 0, 'miss': 0},
                }
            self._gap_patterns[key]['count'] += 1
            if not success:
                self._gap_patterns[key]['failed_strategies'][strategy_used] += 1
            else:
                self._gap_patterns[key]['success_rate']['hit'] += 1
            self._gap_patterns[key]['success_rate']['miss'] += (0 if success else 1)

            # Keep last 5 question patterns for this gap type
            q_pattern = self._extract_question_pattern(question)
            patterns = self._gap_patterns[key]['question_patterns']
            if len(patterns) >= 5:
                patterns.pop(0)
            patterns.append(q_pattern)

    def _extract_question_pattern(self, question: str) -> str:
        """Extract a generalized pattern from a question.

        Replaces specific entities with placeholders to find
        recurring QUESTION STRUCTURES (not specific questions).
        """
        # Replace numbers with NUM
        pattern = re.sub(r'\d+\.?\d*', 'NUM', question)
        # Replace names (capitalized words) with NAME
        pattern = re.sub(r'\b[A-Z][a-z]+\b', 'NAME', pattern)
        return pattern.lower().strip()

    def get_gap_insights(self) -> dict:
        """Get insights about what gaps the system has been experiencing.

        Returns a summary of gap patterns and which strategies
        fail for each gap type.
        """
        insights = {}
        for gap_type, data in self._gap_patterns.items():
            total = data['success_rate']['hit'] + data['success_rate']['miss']
            success_rate = data['success_rate']['hit'] / total if total > 0 else 0.0

            # Find the most common failed strategy
            worst_strategy = ''
            if data['failed_strategies']:
                worst_strategy = max(
                    data['failed_strategies'],
                    key=data['failed_strategies'].get
                )

            insights[gap_type] = {
                'count': data['count'],
                'success_rate': success_rate,
                'worst_strategy': worst_strategy,
                'recent_patterns': data['question_patterns'][-3:],
            }
        return insights


class GapIdentifier:
    """Identifies WHAT knowledge is missing when confidence is low.

    Gap types are NOT predefined — they are DISCOVERED by analyzing
    the pattern of what the system doesn't know. However, we provide
    initial gap categories that emerge from the architecture:

    1. concept_gap: Missing concept clusters for the question domain
    2. proposition_gap: Text mentions something but no proposition captured it
    3. strategy_gap: No derivation strategy can handle this question type
    4. calibration_gap: Confidence doesn't reflect actual accuracy
    5. negation_gap: Cannot parse negation structure correctly
    6. multi_step_gap: Multi-step reasoning chain breaks
    7. entity_gap: Cannot track entities across sentences
    """

    # Initial gap types — discovered from architecture analysis
    GAP_TYPES = [
        'concept_gap',        # No concept cluster matches the text
        'proposition_gap',    # Key information not captured as proposition
        'strategy_gap',       # No handler for this question type
        'calibration_gap',    # Confidence mismatch with accuracy
        'negation_gap',       # Cannot parse negation structure
        'multi_step_gap',     # Multi-step reasoning fails
        'entity_gap',         # Cannot track entities across text
        'llm_gap',            # LLM also couldn't answer
    ]

    def identify_gap(self, question: str, answer: dict, text: str = '',
                     q_type: str = '') -> Dict[str, str]:
        """Identify the specific knowledge gap causing low confidence.

        Returns dict with:
            gap_type: Category of gap
            gap_details: Specific description
            suggested_action: What the system should do to fill this gap
        """
        confidence = answer.get('confidence', 0.0)
        method = answer.get('method', 'unknown')
        answer_val = answer.get('answer')

        # No answer at all → strategy gap or concept gap
        if answer_val is None:
            return self._diagnose_no_answer(question, text, q_type, method)

        # Low confidence with answer → calibration or concept gap
        if confidence < 0.4:
            return self._diagnose_low_confidence(question, text, q_type, method, confidence)

        # Medium confidence → might be entity or negation gap
        if confidence < 0.6:
            return self._diagnose_medium_confidence(question, text, q_type)

        # Reasonably confident → no major gap
        return {
            'gap_type': 'none',
            'gap_details': 'Confidence is adequate, no major gap identified',
            'suggested_action': 'verify_via_counterfactual',
        }

    def _diagnose_no_answer(self, question: str, text: str,
                            q_type: str, method: str) -> dict:
        """Diagnose why the system couldn't produce any answer."""
        q_lower = question.lower()

        # Check for negation patterns in the question
        negation_words = ['tidak', 'bukan', 'tanpa', 'belum']
        has_negation = any(nw in q_lower.split() for nw in negation_words)
        if has_negation and q_type in ('pertanyaan_negatif', 'benar_salah'):
            return {
                'gap_type': 'negation_gap',
                'gap_details': f'Cannot parse negation structure in question: {question[:80]}',
                'suggested_action': 'learn_negation_pattern',
            }

        # Check for multi-step indicators
        multi_step_keywords = ['kembalian', 'sisa', 'selisih', 'total', 'kemudian']
        has_multi_step = any(kw in q_lower for kw in multi_step_keywords)
        if has_multi_step:
            return {
                'gap_type': 'multi_step_gap',
                'gap_details': f'Multi-step reasoning required: {question[:80]}',
                'suggested_action': 'decompose_question',
            }

        # Check if question type is unknown
        if q_type in ('unknown_type', 'unknown') or not q_type:
            return {
                'gap_type': 'strategy_gap',
                'gap_details': f'No handler for question type: {q_type}',
                'suggested_action': 'learn_question_type',
            }

        # Check if text has concepts but they weren't detected
        if text:
            # Simple heuristic: if text is long but no concepts matched
            word_count = len(text.split())
            if word_count > 30 and method == 'unknown_type':
                return {
                    'gap_type': 'concept_gap',
                    'gap_details': f'Text has {word_count} words but no concepts detected',
                    'suggested_action': 'extend_concept_clusters',
                }

        # Default: proposition gap
        return {
            'gap_type': 'proposition_gap',
            'gap_details': f'Could not extract relevant propositions from text',
            'suggested_action': 'ask_for_clarification',
        }

    def _diagnose_low_confidence(self, question: str, text: str,
                                  q_type: str, method: str,
                                  confidence: float) -> dict:
        """Diagnose why confidence is low despite having an answer."""
        # If method is LLM, the concept-based approach failed
        if method in ('llm_reasoning', 'llm_fallback'):
            return {
                'gap_type': 'llm_gap',
                'gap_details': f'Concept-based methods failed (conf={confidence:.2f}), LLM used as fallback',
                'suggested_action': 'teach_llm_answer_to_concepts',
            }

        # If learned pattern was used, it might be too generalized
        if 'learned' in method:
            return {
                'gap_type': 'calibration_gap',
                'gap_details': f'Learned pattern matched but with low confidence ({confidence:.2f})',
                'suggested_action': 'refine_learned_pattern',
            }

        # Generic concept gap
        return {
            'gap_type': 'concept_gap',
            'gap_details': f'Answer found via {method} but confidence only {confidence:.2f}',
            'suggested_action': 'targeted_llm_query',
        }

    def _diagnose_medium_confidence(self, question: str, text: str,
                                     q_type: str) -> dict:
        """Diagnose medium confidence — might be entity tracking issue."""
        # Check for entity-heavy questions
        entity_keywords = ['siapa', 'nama', 'tokoh', 'yang mana']
        q_lower = question.lower()
        if any(ek in q_lower for ek in entity_keywords):
            return {
                'gap_type': 'entity_gap',
                'gap_details': 'Entity tracking across sentences may be incomplete',
                'suggested_action': 'improve_entity_resolution',
            }

        # Check for comparison questions
        comparison_keywords = ['perbedaan', 'persamaan', 'dibandingkan', 'lebih']
        if any(ck in q_lower for ck in comparison_keywords):
            return {
                'gap_type': 'entity_gap',
                'gap_details': 'Comparison requires tracking multiple entities',
                'suggested_action': 'improve_entity_resolution',
            }

        return {
            'gap_type': 'calibration_gap',
            'gap_details': f'Medium confidence — calibration may be off',
            'suggested_action': 'verify_via_counterfactual',
        }


class TargetedQueryGenerator:
    """Generate targeted LLM queries based on identified gaps.

    Instead of sending the raw question to the LLM (which is a blind retry),
    this generator creates SPECIFIC prompts that ask the LLM to fill the
    IDENTIFIED gap.

    This is the key difference: the system knows WHAT it doesn't know,
    so it can ask the right question.
    """

    def generate(self, text: str, question: str, gap: dict) -> str:
        """Generate a targeted LLM prompt based on the identified gap.

        Args:
            text: Source text
            question: Original question
            gap: Gap identification result from GapIdentifier

        Returns:
            A targeted prompt for the LLM
        """
        gap_type = gap.get('gap_type', 'unknown')
        gap_details = gap.get('gap_details', '')
        suggested_action = gap.get('suggested_action', '')

        # Truncate text for prompt
        max_text = 1200
        truncated_text = text[:max_text]
        if len(text) > max_text:
            truncated_text += '...'

        if gap_type == 'concept_gap':
            return self._concept_gap_prompt(truncated_text, question, gap_details)

        if gap_type == 'negation_gap':
            return self._negation_gap_prompt(truncated_text, question, gap_details)

        if gap_type == 'multi_step_gap':
            return self._multi_step_gap_prompt(truncated_text, question, gap_details)

        if gap_type == 'strategy_gap':
            return self._strategy_gap_prompt(truncated_text, question, gap_details)

        if gap_type == 'entity_gap':
            return self._entity_gap_prompt(truncated_text, question, gap_details)

        # Default: generic targeted query
        return self._generic_targeted_prompt(truncated_text, question, gap_details)

    def _concept_gap_prompt(self, text: str, question: str, details: str) -> str:
        """Targeted prompt for concept gap — ask LLM to identify concepts."""
        return (
            f"Teks berikut mengandung konsep yang belum saya kenali.\n\n"
            f"Teks:\n{text}\n\n"
            f"Pertanyaan: {question}\n\n"
            f"Masalah: {details}\n\n"
            f"Tugas:\n"
            f"1. Identifikasi konsep utama dalam teks yang relevan dengan pertanyaan\n"
            f"2. Jelaskan hubungan antara konsep-konsep tersebut\n"
            f"3. Berikan jawaban singkat untuk pertanyaan\n\n"
            f"Format:\n"
            f"Konsep: [daftar konsep]\n"
            f"Hubungan: [penjelasan]\n"
            f"Jawaban: [jawaban]"
        )

    def _negation_gap_prompt(self, text: str, question: str, details: str) -> str:
        """Targeted prompt for negation gap — ask LLM to parse negation structure."""
        return (
            f"Pertanyaan berikut mengandung struktur negasi yang kompleks.\n\n"
            f"Teks:\n{text}\n\n"
            f"Pertanyaan: {question}\n\n"
            f"Masalah: {details}\n\n"
            f"Tugas:\n"
            f"1. Identifikasi semua kata negasi dalam pertanyaan (tidak, bukan, belum, dll.)\n"
            f"2. Tentukan APA yang di-negasi oleh setiap kata negasi\n"
            f"3. Hitung berapa lapis negasi (1=negasi tunggal, 2=double negasi, dll.)\n"
            f"4. Jika negasi genap → makna positif. Jika ganjil → makna negatif.\n"
            f"5. Berikan jawaban berdasarkan interpretasi negasi yang benar\n\n"
            f"Format:\n"
            f"Lapis Negasi: [jumlah]\n"
            f"Interpretasi: [makna setelah negasi]\n"
            f"Jawaban: [jawaban]"
        )

    def _multi_step_gap_prompt(self, text: str, question: str, details: str) -> str:
        """Targeted prompt for multi-step gap — ask LLM to decompose."""
        return (
            f"Pertanyaan berikut memerlukan penalaran multi-langkah.\n\n"
            f"Teks:\n{text}\n\n"
            f"Pertanyaan: {question}\n\n"
            f"Masalah: {details}\n\n"
            f"Tugas:\n"
            f"1. Pecah pertanyaan menjadi langkah-langkah perhitungan\n"
            f"2. Untuk setiap langkah, identifikasi: operasi, angka, dan hasil\n"
            f"3. Rantaikan hasil setiap langkah ke langkah berikutnya\n"
            f"4. Berikan jawaban akhir\n\n"
            f"Format:\n"
            f"Langkah 1: [operasi] → [hasil]\n"
            f"Langkah 2: [operasi] → [hasil]\n"
            f"...\n"
            f"Jawaban: [jawaban akhir]"
        )

    def _strategy_gap_prompt(self, text: str, question: str, details: str) -> str:
        """Targeted prompt for strategy gap — ask LLM to classify and answer."""
        return (
            f"Pertanyaan berikut tidak cocok dengan strategi yang saya kenali.\n\n"
            f"Teks:\n{text}\n\n"
            f"Pertanyaan: {question}\n\n"
            f"Masalah: {details}\n\n"
            f"Tugas:\n"
            f"1. Klasifikasikan jenis pertanyaan ini\n"
            f"2. Identifikasi strategi penalaran yang tepat\n"
            f"3. Jelaskan langkah penalarannya\n"
            f"4. Berikan jawaban\n\n"
            f"Format:\n"
            f"Jenis: [klasifikasi]\n"
            f"Strategi: [nama strategi]\n"
            f"Penalaran: [langkah-langkah]\n"
            f"Jawaban: [jawaban]"
        )

    def _entity_gap_prompt(self, text: str, question: str, details: str) -> str:
        """Targeted prompt for entity gap — ask LLM to track entities."""
        return (
            f"Pertanyaan berikut memerlukan pelacakan entitas dalam teks.\n\n"
            f"Teks:\n{text}\n\n"
            f"Pertanyaan: {question}\n\n"
            f"Masalah: {details}\n\n"
            f"Tugas:\n"
            f"1. Identifikasi semua entitas (orang, benda, tempat) dalam teks\n"
            f"2. Untuk setiap entitas, daftarkan atribut yang disebutkan\n"
            f"3. Lacak perubahan atribut entitas dari kalimat ke kalimat\n"
            f"4. Jawab pertanyaan berdasarkan pelacakan entitas\n\n"
            f"Format:\n"
            f"Entitas: [daftar entitas dan atributnya]\n"
            f"Jawaban: [jawaban]"
        )

    def _generic_targeted_prompt(self, text: str, question: str, details: str) -> str:
        """Generic targeted prompt — still better than blind retry."""
        return (
            f"Saya tidak yakin tentang jawaban untuk pertanyaan ini.\n\n"
            f"Teks:\n{text}\n\n"
            f"Pertanyaan: {question}\n\n"
            f"Masalah yang saya hadapi: {details}\n\n"
            f"Jawab pertanyaan dengan menjelaskan penalaranmu.\n\n"
            f"Format:\n"
            f"Penalaran: [langkah-langkah]\n"
            f"Jawaban: [jawaban singkat]"
        )


class MetaCognitiveMonitor:
    """The brain of meta-cognitive self-reflection.

    This monitor OBSERVES the reasoning process, IDENTIFIES gaps,
    and GENERATES targeted knowledge acquisition strategies.

    It is NOT a separate reasoning engine — it is a META layer
    that watches the existing reasoning engine and makes it smarter.
    """

    def __init__(self, self_core=None):
        self.self_core = self_core
        self.reflection_log = ReflectionLog()
        self.gap_identifier = GapIdentifier()
        self.query_generator = TargetedQueryGenerator()
        self._strategy_effectiveness = defaultdict(lambda: {'hits': 0, 'misses': 0})

    def reflect(self, question: str, answer: dict, text: str = '',
                q_type: str = '', correct_answer: str = None) -> dict:
        """Reflect on a reasoning result — the main entry point.

        Called AFTER an answer is produced. Analyzes the result
        and identifies gaps if confidence is low.

        Args:
            question: The question that was asked
            answer: The answer dict from the reasoning engine
            text: Source text (if any)
            q_type: Question type classification
            correct_answer: If known, the correct answer for learning

        Returns:
            Reflection result with gap identification and suggested actions
        """
        confidence = answer.get('confidence', 0.0)
        method = answer.get('method', 'unknown')

        # Identify the gap (if any)
        gap = self.gap_identifier.identify_gap(question, answer, text, q_type)

        # Determine if answer was correct (if we know)
        success = None
        if correct_answer is not None:
            success = str(answer.get('answer', '')).lower().strip() == str(correct_answer).lower().strip()

        # Record in reflection log
        self.reflection_log.record(
            question=question,
            answer=answer,
            gap_type=gap['gap_type'],
            gap_details=gap['gap_details'],
            strategy_used=method,
            success=success if success is not None else (confidence >= 0.5),
        )

        # Update strategy effectiveness tracking
        if success is not None:
            if success:
                self._strategy_effectiveness[method]['hits'] += 1
            else:
                self._strategy_effectiveness[method]['misses'] += 1

        return {
            'gap': gap,
            'targeted_query': self.query_generator.generate(text, question, gap) if gap['gap_type'] != 'none' else None,
            'reflection_insights': self.reflection_log.get_gap_insights(),
            'strategy_performance': self._get_strategy_performance(),
        }

    def get_targeted_llm_query(self, text: str, question: str,
                                previous_answer: dict, q_type: str = '') -> str:
        """Generate a targeted LLM query based on identified gap.

        This replaces the blind LLM fallback with a TARGETED query.

        Instead of:
          "Jawab pertanyaan ini: {question}"

        We generate:
          "Pertanyaan ini memerlukan penalaran multi-langkah.
           Langkah 1: ..., Langkah 2: ..., Jawaban: ..."

        This dramatically improves LLM response quality because
        the LLM knows exactly WHAT kind of reasoning is needed.
        """
        gap = self.gap_identifier.identify_gap(question, previous_answer, text, q_type)
        return self.query_generator.generate(text, question, gap)

    def _get_strategy_performance(self) -> dict:
        """Get performance metrics for each strategy.

        This is used by the adaptive strategy selector to prefer
        strategies that have worked well in the past.
        """
        performance = {}
        for strategy, data in self._strategy_effectiveness.items():
            total = data['hits'] + data['misses']
            performance[strategy] = {
                'total': total,
                'success_rate': data['hits'] / total if total > 0 else 0.0,
            }
        return performance

    def recommend_strategy(self, q_type: str, gap_type: str = '') -> str:
        """Recommend the best strategy for a given question/gap type.

        Based on historical performance, NOT hardcoded rules.
        If no history exists, returns 'default'.
        """
        best_strategy = 'default'
        best_rate = 0.0

        for strategy, perf in self._strategy_effectiveness.items():
            if perf['total'] >= 3:  # Need at least 3 observations
                if perf['success_rate'] > best_rate:
                    best_rate = perf['success_rate']
                    best_strategy = strategy

        return best_strategy

    def save_reflections(self, path: str):
        """Persist reflection log to disk for cross-session learning."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                'entries': self.reflection_log.entries[-500:],  # Keep last 500
                'gap_patterns': {
                    k: {
                        'count': v['count'],
                        'failed_strategies': dict(v['failed_strategies']),
                        'question_patterns': v['question_patterns'],
                        'success_rate': v['success_rate'],
                    }
                    for k, v in self.reflection_log._gap_patterns.items()
                },
                'strategy_effectiveness': {
                    k: dict(v) for k, v in self._strategy_effectiveness.items()
                },
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save reflection log: %s", e)

    def load_reflections(self, path: str):
        """Load reflection log from disk."""
        try:
            if not os.path.exists(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.reflection_log.entries = data.get('entries', [])
            for k, v in data.get('gap_patterns', {}).items():
                self.reflection_log._gap_patterns[k] = {
                    'count': v['count'],
                    'failed_strategies': defaultdict(int, v.get('failed_strategies', {})),
                    'question_patterns': v.get('question_patterns', []),
                    'success_rate': v.get('success_rate', {'hit': 0, 'miss': 0}),
                }
            for k, v in data.get('strategy_effectiveness', {}).items():
                self._strategy_effectiveness[k] = defaultdict(int, v)
        except Exception as e:
            logger.warning("Failed to load reflection log: %s", e)
