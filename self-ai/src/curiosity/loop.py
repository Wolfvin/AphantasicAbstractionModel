# @WHO:   self-ai/src/curiosity/loop.py
# @WHAT:  Curiosity-Driven Learning Loop
# @PART:  self-ai/curiosity
# @ENTRY: CuriosityLoop

"""Curiosity-Driven Learning Loop
Layer 8 (Curiosity) generates questions about unknowns.
This module makes those questions answerable and feeds answers back into the system.

Flow:
1. Curiosity Queue → Export to curiosity_queue.json
2. CLI: python -m curiosity.answer → Interactive Q&A
3. Answers auto-taught back to TextComprehension
4. Confidence increases → Curiosity Queue shrinks
"""
import json
import os
from typing import List, Optional


class CuriosityLoop:
    def __init__(self, self_core=None, queue_file=None, text_comprehension=None):
        self.self_core = self_core
        self.queue_file = queue_file or os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'curiosity_queue.json'
        )
        # v18: Reuse shared TextComprehension instance instead of creating new one
        # each time (fixes C-05: teaching was being lost on each call)
        self._text_comprehension = text_comprehension

    def export_queue(self):
        """Export curiosity questions to JSON file"""
        if self.self_core is None:
            return []

        questions = self.self_core.curiosity_queue
        queue_data = []
        for q in questions:
            queue_data.append({
                'question': q,
                'answered': False,
                'answer': None,
                'source': None,
            })

        # Merge with existing file
        existing = self._load_queue()
        # Add new questions that aren't already in the file
        existing_questions = {item['question'] for item in existing}
        for item in queue_data:
            if item['question'] not in existing_questions:
                existing.append(item)

        self._save_queue(existing)
        return existing

    def answer_question(self, question: str, answer: str, source: str = 'user'):
        """Record an answer to a curiosity question and teach it"""
        queue = self._load_queue()

        for item in queue:
            if item['question'] == question and not item['answered']:
                item['answered'] = True
                item['answer'] = answer
                item['source'] = source
                break

        self._save_queue(queue)

        # v18: Teach the answer back to the system using SHARED TextComprehension
        # (fixes C-05: previously created new instance each time, losing all teaching)
        if self.self_core:
            try:
                tc = self._get_text_comprehension()
                if tc is not None:
                    tc.teach(answer, question, answer)
            except Exception:
                pass

    def get_unanswered(self) -> List[dict]:
        """Get all unanswered questions"""
        queue = self._load_queue()
        return [item for item in queue if not item['answered']]

    def get_answered(self) -> List[dict]:
        """Get all answered questions"""
        queue = self._load_queue()
        return [item for item in queue if item['answered']]

    def try_auto_answer(self, question: str) -> Optional[str]:
        """Try to answer using LLM reasoning"""
        try:
            from derivation.llm_reasoning import LLMReasoningEngine
            engine = LLMReasoningEngine()
            result = engine.reason('', question)
            if result.get('answer'):
                return result['answer']
        except Exception:
            pass
        return None

    def auto_answer_all(self):
        """Auto-answer all unanswered questions using LLM"""
        unanswered = self.get_unanswered()
        for item in unanswered:
            answer = self.try_auto_answer(item['question'])
            if answer:
                self.answer_question(item['question'], answer, source='llm_auto')

    def _load_queue(self) -> list:
        """Load curiosity queue from file"""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_queue(self, queue: list):
        """Save curiosity queue to file"""
        os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)
        with open(self.queue_file, 'w') as f:
            json.dump(queue, f, indent=2, ensure_ascii=False)

    def _get_text_comprehension(self):
        """Get or create a shared TextComprehension instance.

        v18: Reuses the same instance across all calls so that
        teaching persists within the same session. Also tries to
        use the TextComprehension from the DerivationEngine if
        available (sharing the same learned patterns).
        """
        if self._text_comprehension is not None:
            return self._text_comprehension

        # Try to get from DerivationEngine if available
        if self.self_core and hasattr(self.self_core, 'derivation_engine'):
            de = self.self_core.derivation_engine
            if de and hasattr(de, 'text_comprehension') and de.text_comprehension:
                self._text_comprehension = de.text_comprehension
                return self._text_comprehension

        # Create a new instance (but only once)
        try:
            from derivation.text_comprehension import TextComprehension
            self._text_comprehension = TextComprehension(self.self_core)
            return self._text_comprehension
        except Exception:
            return None
