# @WHO:   self-ai/src/derivation/understanding_composer.py
# @WHAT:  Qwen3 builds understanding nodes from observations — the HEART of SELF's vision
# @PART:  self-ai/derivation
# @ENTRY: UnderstandingComposer

"""Understanding Composer — Qwen3 builds understanding nodes from observations.

Vision:
    This is the HEART of SELF's vision: the LLM doesn't just answer —
    it BUILDS understanding that can be applied WITHOUT the LLM later.

    Flow:
        Teaching Example → Qwen3 "thinks" → UnderstandingNode → Graph

    The resulting UnderstandingNode has a Transformation that can be
    applied mechanically (no LLM needed), because Qwen3 extracted
    the STRUCTURAL PATTERN, not just the answer.

    System 1 = intuitive output (bge-m3 embedding)
    System 2 = learned understanding (Understanding Graph)
    The LLM can COMBINE multiple semantic understandings
    to generate appropriate answers.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Teaching Lesson / Observation / Failure                 │
    │  (soal + cara + jawaban + kenapa)                       │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  UnderstandingComposer.compose_from_teaching()          │
    │  compose_from_observation() / compose_from_failure()    │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Qwen3 (via SDK or local) extracts structural pattern   │
    │  → JSON output: name, concept, abstraction,             │
    │     conditions, transformation, schemas                  │
    └──────────────┬──────────────────────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Parse JSON → UnderstandingNode → Add to Graph          │
    │  (with regex fallback for malformed JSON)               │
    └─────────────────────────────────────────────────────────┘

    When Qwen3 is unavailable, compose_from_* returns None
    gracefully — the system falls back to seeded understandings.

Strategies (in order of preference):
  1. z-ai-web-dev-sdk via Node.js subprocess (cloud LLM)
  2. Local Qwen3-0.6B via transformers (if cached)
  3. Return None — caller falls back to seeded understandings
"""

import os
import re
import json
import logging
import subprocess
import tempfile
from typing import Optional, List

# v35: Structural memory types
from governance.states import UnderstandingMember

logger = logging.getLogger(__name__)


class UnderstandingComposer:
    """Qwen3 builds understanding nodes from observations.

    This is the HEART of SELF's vision: the LLM doesn't just answer —
    it BUILDS understanding that can be applied WITHOUT the LLM later.

    Flow:
        Teaching Example → Qwen3 "thinks" → UnderstandingNode → Graph

    The resulting UnderstandingNode has a Transformation that can be
    applied mechanically (no LLM needed), because Qwen3 extracted
    the STRUCTURAL PATTERN, not just the answer.
    """

    # Valid transformation kinds — matches those in understanding_builder.py
    VALID_KINDS = {
        'signal_flip', 'contrast_focus', 'negation_affirmation',
        'comparison_resolve', 'entity_extract', 'fact_extract',
        'quantity_compute', 'context_filter', 'word_sense',
    }

    def __init__(self, graph=None, llm_engine=None):
        """Initialize the UnderstandingComposer.

        Args:
            graph: UnderstandingGraph to add nodes to. If None, one will
                   be created lazily on first use.
            llm_engine: Optional LLMReasoningEngine instance. If None,
                        we'll try SDK/local Qwen3 ourselves.
        """
        self._graph = graph
        self._llm_engine = llm_engine
        self._sdk_checked = False
        self._sdk_available = False
        self._local_checked = False
        self._local_available = False
        self._local_model = None
        self._local_tokenizer = None

    @property
    def graph(self):
        """Lazy-initialize the understanding graph."""
        if self._graph is None:
            from derivation.understanding_builder import UnderstandingGraph
            self._graph = UnderstandingGraph()
        return self._graph

    # ═══════════════ PUBLIC API ═══════════════

    def compose_from_teaching(self, lesson) -> Optional['UnderstandingNode']:
        """Take a teaching lesson, ask Qwen3 to extract understanding.

        This is the CORE method — Qwen3 observes a teaching example and
        builds a reusable UnderstandingNode with a Transformation.

        Args:
            lesson: TeachingLesson instance with problem, solution_steps,
                    answer, explanation_why, question_type

        Returns:
            UnderstandingNode if Qwen3 successfully extracted understanding,
            None if Qwen3 is unavailable or output is unparseable.
        """
        prompt = self._build_teaching_prompt(lesson)
        raw_response = self._call_llm(prompt)

        if raw_response is None:
            logger.debug("LLM unavailable for compose_from_teaching — skipping")
            return None

        # Parse the JSON response
        parsed = self._parse_json_response(raw_response)
        if parsed is None:
            logger.warning("Failed to parse LLM response for teaching: %s",
                          raw_response[:200] if raw_response else 'None')
            return None

        # Validate and build the node
        node = self._build_node_from_parsed(parsed, source='composed_from_teaching')
        if node is None:
            return None

        # Add to graph
        self.graph.add_node(node)
        logger.info("Composed understanding from teaching: %s (kind=%s)",
                    node.id, node.transformation.kind if node.transformation else '?')
        return node

    def compose_from_observation(self, observation: dict) -> Optional['UnderstandingNode']:
        """Take a novel observation, ask Qwen3 what understanding can be built.

        This is for when SELF encounters something novel and wants to
        understand it — not from a structured teaching lesson, but from
        raw observation data.

        Args:
            observation: Dict with keys like 'text', 'question', 'answer',
                        'context', 'novelty_score', etc.

        Returns:
            UnderstandingNode if Qwen3 extracted understanding,
            None otherwise.
        """
        prompt = self._build_observation_prompt(observation)
        raw_response = self._call_llm(prompt)

        if raw_response is None:
            logger.debug("LLM unavailable for compose_from_observation — skipping")
            return None

        parsed = self._parse_json_response(raw_response)
        if parsed is None:
            logger.warning("Failed to parse LLM response for observation: %s",
                          raw_response[:200] if raw_response else 'None')
            return None

        node = self._build_node_from_parsed(parsed, source='composed_from_observation')
        if node is None:
            return None

        self.graph.add_node(node)
        logger.info("Composed understanding from observation: %s (kind=%s)",
                    node.id, node.transformation.kind if node.transformation else '?')
        return node

    def compose_from_failure(self, text: str, question: str,
                             wrong_answer: str,
                             correct_answer: str) -> Optional['UnderstandingNode']:
        """Learn from mistakes — compose understanding from a failure.

        When SELF gets a wrong answer, this method asks Qwen3 to figure
        out what understanding is missing or incorrect.

        Args:
            text: The source text
            question: The question that was asked
            wrong_answer: The answer SELF gave
            correct_answer: The correct answer

        Returns:
            UnderstandingNode representing the new/corrected understanding,
            None if Qwen3 is unavailable.
        """
        prompt = self._build_failure_prompt(text, question, wrong_answer, correct_answer)
        raw_response = self._call_llm(prompt)

        if raw_response is None:
            logger.debug("LLM unavailable for compose_from_failure — skipping")
            return None

        parsed = self._parse_json_response(raw_response)
        if parsed is None:
            logger.warning("Failed to parse LLM response for failure: %s",
                          raw_response[:200] if raw_response else 'None')
            return None

        node = self._build_node_from_parsed(parsed, source='composed_from_failure')
        if node is None:
            return None

        self.graph.add_node(node)
        logger.info("Composed understanding from failure: %s (kind=%s)",
                    node.id, node.transformation.kind if node.transformation else '?')
        return node

    def compose_answer_from_understandings(self, text: str, question: str,
                                            understandings: list) -> Optional[str]:
        """Compose an answer by combining multiple understandings via Qwen3.

        When multiple understandings are retrieved but none individually
        produce an answer, Qwen3 can compose them together.

        Args:
            text: The source text
            question: The question being asked
            understandings: List of (UnderstandingNode, score) tuples

        Returns:
            Composed answer string, or None if Qwen3 is unavailable.
        """
        if not understandings:
            return None

        prompt = self._build_composition_prompt(text, question, understandings)
        raw_response = self._call_llm(prompt)

        if raw_response is None:
            return None

        # Extract answer from response
        return self._extract_answer(raw_response)

    # ═══════════════ PROMPT CONSTRUCTION ═══════════════

    def _build_teaching_prompt(self, lesson) -> str:
        """Build the prompt for compose_from_teaching."""
        problem = lesson.problem or ''
        steps = '\n'.join(f'  {i+1}. {s}' for i, s in enumerate(lesson.solution_steps or []))
        answer = lesson.answer or ''
        why = lesson.explanation_why or ''
        q_type = lesson.question_type or ''

        return f"""Observe this teaching example and extract the STRUCTURAL UNDERSTANDING:

SOAL: {problem}
CARA: 
{steps}
JAWABAN: {answer}
KENAPA: {why}
TIPE: {q_type}

Extract the understanding as JSON:
{{
    "name": "short snake_case name",
    "concept": "what this understanding is about",
    "abstraction": "the generalized principle (replace specifics with variables)",
    "conditions": ["trigger signal 1", "trigger signal 2"],
    "transformation": {{
        "kind": "one of: signal_flip, contrast_focus, negation_affirmation, comparison_resolve, entity_extract, fact_extract, quantity_compute, context_filter, word_sense",
        "trigger": {{"signal_words": ["word1", "word2"], "result_position": "after/before"}},
        "action": "what the transformation does"
    }},
    "schemas": ["generalized pattern like: ALL [STATE] KECUALI [EXCEPTION] → [EXCEPTION] has OPPOSITE [STATE]"],
    "members": [
        {{"role": "trigger", "description": "what signal activates this", "confidence": 0.9}},
        {{"role": "default", "description": "what the normal/default answer would be", "confidence": 0.8}},
        {{"role": "exception", "description": "what overrides the default", "confidence": 0.9}},
        {{"role": "result", "description": "the final transformed answer", "confidence": 0.8}}
    ]
}}

IMPORTANT: 
- The abstraction must generalize away specifics — use [ENTITY], [STATE], [EXCEPTION] as variables
- The transformation kind must be one of the listed kinds
- The conditions are signal words that trigger this understanding
- The members describe the STRUCTURAL ROLES within this understanding — each has a role name, description, and confidence
- Common roles: trigger, default, exception, result, cause, agent, patient, context, evidence
- Respond ONLY with valid JSON, no other text"""

    def _build_observation_prompt(self, observation: dict) -> str:
        """Build the prompt for compose_from_observation."""
        text = observation.get('text', '')
        question = observation.get('question', '')
        answer = observation.get('answer', '')
        context = observation.get('context', '')
        novelty = observation.get('novelty_score', 0.0)

        return f"""Observe this novel input and determine what understanding can be built:

TEKS: {text}
PERTANYAAN: {question}
JAWABAN: {answer}
KONTEKS: {context}
NOVELTY: {novelty:.2f}

If this observation reveals a pattern that could be generalized, extract the understanding as JSON:
{{
    "name": "short snake_case name",
    "concept": "what this understanding is about",
    "abstraction": "the generalized principle",
    "conditions": ["trigger signal 1", "trigger signal 2"],
    "transformation": {{
        "kind": "one of: signal_flip, contrast_focus, negation_affirmation, comparison_resolve, entity_extract, fact_extract, quantity_compute, context_filter, word_sense",
        "trigger": {{"signal_words": ["word1"], "result_position": "after"}},
        "action": "what the transformation does"
    }},
    "schemas": ["generalized pattern"],
    "members": [
        {{"role": "trigger", "description": "what signal activates this", "confidence": 0.9}},
        {{"role": "result", "description": "the expected outcome", "confidence": 0.8}}
    ]
}}

If no clear understanding can be extracted, respond with: {{"skip": true}}
Respond ONLY with valid JSON."""

    def _build_failure_prompt(self, text: str, question: str,
                              wrong_answer: str,
                              correct_answer: str) -> str:
        """Build the prompt for compose_from_failure."""
        return f"""SELF answered incorrectly. Learn from this failure:

TEKS: {text}
PERTANYAAN: {question}
JAWABAN SALAH: {wrong_answer}
JAWABAN BENAR: {correct_answer}

What understanding is missing or incorrect? Extract the correct understanding as JSON:
{{
    "name": "short snake_case name",
    "concept": "what the correct understanding should be",
    "abstraction": "the generalized principle that would produce the correct answer",
    "conditions": ["trigger signal 1", "trigger signal 2"],
    "transformation": {{
        "kind": "one of: signal_flip, contrast_focus, negation_affirmation, comparison_resolve, entity_extract, fact_extract, quantity_compute, context_filter, word_sense",
        "trigger": {{"signal_words": ["word1"], "result_position": "after"}},
        "action": "what the transformation should do to get the correct answer"
    }},
    "schemas": ["generalized pattern that explains the correct answer"],
    "members": [
        {{"role": "trigger", "description": "what signal was missed or misapplied", "confidence": 0.9}},
        {{"role": "result", "description": "the correct result when this trigger is recognized", "confidence": 0.8}}
    ]
}}

Focus on WHY the wrong answer was wrong and what structural understanding would prevent it.
Respond ONLY with valid JSON."""

    def _build_composition_prompt(self, text: str, question: str,
                                   understandings: list) -> str:
        """Build the prompt for composing answer from multiple understandings."""
        u_descriptions = []
        for i, (node, score) in enumerate(understandings[:3]):
            kind = node.transformation.kind if node.transformation else 'unknown'
            action = node.transformation.action if node.transformation else ''
            abstraction = node.abstraction[:150] if node.abstraction else ''
            u_descriptions.append(
                f"  Understanding {i+1} (score={score:.3f}, kind={kind}):\n"
                f"    Abstraction: {abstraction}\n"
                f"    Action: {action}"
            )

        u_text = '\n'.join(u_descriptions)

        return f"""Given these understandings, compose the answer to the question.

TEKS: {text}
PERTANYAAN: {question}

UNDERSTANDINGS:
{u_text}

Based on these understandings, what is the answer? 
Jawab dengan format:
Penalaran: [langkah-langkah penalaran menggunakan understandings di atas]
Jawaban: [jawaban singkat]"""

    # ═══════════════ LLM CALLING ═══════════════

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM — try SDK first, then local Qwen3, then give up.

        This follows the same pattern as llm_reasoning.py:
        1. z-ai-web-dev-sdk (cloud LLM — fast, no local GPU needed)
        2. Local Qwen3-0.6B (if cached)
        3. Return None — caller handles gracefully
        """
        # Strategy 1: Use existing LLM engine if available
        if self._llm_engine is not None:
            try:
                # Use the LLM engine's reason method — it has its own SDK/local logic
                result = self._llm_engine.reason(
                    text=prompt,
                    question="",
                    previous_result={'answer': None, 'confidence': 0.0}
                )
                if result and result.get('answer'):
                    return str(result['answer'])
            except Exception as e:
                logger.debug("llm_engine failed: %s — falling through to SDK/local", e)

        # Strategy 2: z-ai-web-dev-sdk (cloud LLM)
        answer = self._try_sdk(prompt)
        if answer is not None:
            return answer

        # Strategy 3: Local Qwen3-0.6B
        answer = self._try_local_qwen(prompt)
        if answer is not None:
            return answer

        return None

    def _try_sdk(self, prompt: str) -> Optional[str]:
        """Use z-ai-web-dev-sdk via Node.js subprocess."""
        if not self._sdk_checked:
            self._sdk_available = self._check_sdk_available()
            self._sdk_checked = True
            if not self._sdk_available:
                logger.info("z-ai-web-dev-sdk not available for UnderstandingComposer")

        if not self._sdk_available:
            return None

        try:
            prompt_json = json.dumps(prompt[:2000], ensure_ascii=False)

            script = '''
const ZAI = require('z-ai-web-dev-sdk').default;

async function main() {
    try {
        const zai = await ZAI.create();
        const completion = await zai.chat.completions.create({
            messages: [
                { role: 'system', content: 'Kamu adalah sistem yang mengekstrak pola struktural dari contoh pembelajaran. Selalu jawab dengan JSON yang valid. Tidak ada teks lain selain JSON.' },
                { role: 'user', content: %s }
            ]
        });
        const content = completion.choices && completion.choices[0] && completion.choices[0].message && completion.choices[0].message.content;
        if (content) {
            console.log(content);
        } else {
            console.error('No content in response');
        }
    } catch (e) {
        console.error('SDK error: ' + e.message);
    }
}

main();
''' % prompt_json

            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(script)
                script_path = f.name

            try:
                # Ensure Node.js can find z-ai-web-dev-sdk
                import shutil
                env = os.environ.copy()
                npm_global = os.path.expanduser('~/.npm-global/lib/node_modules')
                if os.path.isdir(npm_global) and 'NODE_PATH' not in env:
                    env['NODE_PATH'] = npm_global

                result = subprocess.run(
                    ['node', script_path],
                    capture_output=True, text=True, timeout=30,
                    env=env
                )

                if result.returncode == 0 and result.stdout.strip():
                    output = result.stdout.strip()
                    if output.startswith('SDK error:') or output.startswith('No content'):
                        logger.warning("z-ai-web-dev-sdk returned error: %s", output[:200])
                        return None
                    return output
                else:
                    if result.stderr:
                        logger.warning("z-ai-web-dev-sdk subprocess error: %s", result.stderr[:200])
            finally:
                os.unlink(script_path)

        except subprocess.TimeoutExpired:
            logger.warning("z-ai-web-dev-sdk call timed out (30s)")
        except FileNotFoundError:
            self._sdk_available = False
            logger.info("Node.js not found — disabling cloud LLM for UnderstandingComposer")
        except Exception as e:
            logger.warning("z-ai-web-dev-sdk fallback failed: %s", e)

        return None

    def _check_sdk_available(self) -> bool:
        """Check if z-ai-web-dev-sdk is installed and node is available."""
        try:
            env = os.environ.copy()
            npm_global = os.path.expanduser('~/.npm-global/lib/node_modules')
            if os.path.isdir(npm_global) and 'NODE_PATH' not in env:
                env['NODE_PATH'] = npm_global

            result = subprocess.run(
                ['node', '-e', "require('z-ai-web-dev-sdk')"],
                capture_output=True, text=True, timeout=5,
                env=env
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    def _try_local_qwen(self, prompt: str) -> Optional[str]:
        """Try Qwen3-0.6B via shared singleton.

        Uses model_registry.get_shared_qwen() to avoid loading Qwen3
        multiple times. The model is loaded once and shared across
        all callers (LLMReasoningEngine, UnderstandingComposer, etc.).
        """
        if not self._local_checked:
            self._local_available = self._check_local_qwen_available()
            self._local_checked = True
            if not self._local_available:
                logger.info("Qwen3-0.6B not cached locally — skipping local LLM for UnderstandingComposer")

        if not self._local_available:
            return None

        try:
            from derivation.model_registry import get_shared_qwen

            model, tokenizer = get_shared_qwen()
            if model is None or tokenizer is None:
                logger.warning("Shared Qwen3-0.6B not available for UnderstandingComposer")
                return None

            inputs = tokenizer(
                prompt, return_tensors='pt', max_length=512, truncation=True
            )
            outputs = model.generate(
                **inputs, max_new_tokens=300, temperature=0.3
            )
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            return response
        except Exception as e:
            logger.warning("Local Qwen3-0.6B inference failed: %s", e)
            return None

    def _check_local_qwen_available(self) -> bool:
        """Check if Qwen3-0.6B is cached in HuggingFace hub."""
        try:
            from derivation.model_registry import is_qwen_available
            return is_qwen_available()
        except ImportError:
            pass
        # Fallback: check cache dir directly
        try:
            import transformers  # noqa: F401
            cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
            model_path = os.path.join(cache_dir, 'models--Qwen--Qwen3-0.6B')
            return os.path.exists(model_path)
        except ImportError:
            return False

    # ═══════════════ JSON PARSING ═══════════════

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse LLM response as JSON with graceful fallback.

        Tries:
        1. Direct JSON parse
        2. Extract JSON from markdown code blocks
        3. Regex fallback to extract key fields

        Returns parsed dict or None.
        """
        if not response or not response.strip():
            return None

        # Check for skip signal
        if '"skip"' in response and 'true' in response:
            return None

        # Strategy 1: Direct JSON parse
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from markdown code blocks
        # LLMs often wrap JSON in ```json ... ```
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON-like object in response
        # Look for the outermost { ... }
        brace_start = response.find('{')
        brace_end = response.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            json_str = response[brace_start:brace_end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Regex fallback — extract key fields manually
        return self._regex_fallback_parse(response)

    def _regex_fallback_parse(self, response: str) -> Optional[dict]:
        """Parse JSON-like content using regex when JSON parsing fails.

        This handles cases where Qwen3 outputs slightly malformed JSON,
        such as missing quotes, trailing commas, or comments.
        """
        result = {}

        # Extract name
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', response)
        if name_match:
            result['name'] = name_match.group(1)

        # Extract concept
        concept_match = re.search(r'"concept"\s*:\s*"([^"]+)"', response)
        if concept_match:
            result['concept'] = concept_match.group(1)

        # Extract abstraction
        abstraction_match = re.search(r'"abstraction"\s*:\s*"([^"]+)"', response)
        if abstraction_match:
            result['abstraction'] = abstraction_match.group(1)

        # Extract conditions (array of strings)
        conditions_match = re.search(r'"conditions"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if conditions_match:
            conditions_str = conditions_match.group(1)
            conditions = re.findall(r'"([^"]+)"', conditions_str)
            if conditions:
                result['conditions'] = conditions

        # Extract transformation.kind
        kind_match = re.search(r'"kind"\s*:\s*"([^"]+)"', response)
        if kind_match:
            kind = kind_match.group(1)
            if kind in self.VALID_KINDS:
                if 'transformation' not in result:
                    result['transformation'] = {}
                result['transformation']['kind'] = kind

        # Extract transformation.trigger.signal_words
        signal_words_match = re.search(
            r'"signal_words"\s*:\s*\[(.*?)\]', response, re.DOTALL
        )
        if signal_words_match:
            words_str = signal_words_match.group(1)
            words = re.findall(r'"([^"]+)"', words_str)
            if words:
                if 'transformation' not in result:
                    result['transformation'] = {}
                if 'trigger' not in result['transformation']:
                    result['transformation']['trigger'] = {}
                result['transformation']['trigger']['signal_words'] = words

        # Extract transformation.trigger.result_position
        pos_match = re.search(r'"result_position"\s*:\s*"([^"]+)"', response)
        if pos_match:
            if 'transformation' not in result:
                result['transformation'] = {}
            if 'trigger' not in result['transformation']:
                result['transformation']['trigger'] = {}
            result['transformation']['trigger']['result_position'] = pos_match.group(1)

        # Extract transformation.action
        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response)
        if action_match:
            if 'transformation' not in result:
                result['transformation'] = {}
            result['transformation']['action'] = action_match.group(1)

        # Extract schemas
        schemas_match = re.search(r'"schemas"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if schemas_match:
            schemas_str = schemas_match.group(1)
            schemas = re.findall(r'"([^"]+)"', schemas_str)
            if schemas:
                result['schemas'] = schemas

        # Validate: we need at least name + concept + abstraction + transformation.kind
        required = ['name', 'concept', 'abstraction']
        has_transformation_kind = (
            result.get('transformation', {}).get('kind') is not None
        )

        if all(result.get(k) for k in required) and has_transformation_kind:
            return result

        return None

    # ═══════════════ NODE CONSTRUCTION ═══════════════

    def _build_node_from_parsed(self, parsed: dict,
                                 source: str = 'composed') -> Optional['UnderstandingNode']:
        """Build an UnderstandingNode from parsed JSON data.

        Args:
            parsed: Dict with name, concept, abstraction, conditions,
                    transformation, schemas
            source: Source label for the node

        Returns:
            UnderstandingNode or None if required fields are missing.
        """
        from derivation.understanding_builder import (
            UnderstandingNode, Transformation
        )

        name = parsed.get('name', '')
        concept = parsed.get('concept', '')
        abstraction = parsed.get('abstraction', '')

        if not name or not concept or not abstraction:
            logger.warning("Parsed understanding missing required fields (name/concept/abstraction)")
            return None

        # Build transformation
        trans_data = parsed.get('transformation', {})
        kind = trans_data.get('kind', '')

        if kind and kind not in self.VALID_KINDS:
            logger.warning("Invalid transformation kind: %s — skipping", kind)
            return None

        transformation = None
        if kind:
            trigger = trans_data.get('trigger', {})
            action = trans_data.get('action', '')
            transformation = Transformation(
                kind=kind,
                trigger=trigger,
                action=action,
            )

        # Build conditions
        conditions = parsed.get('conditions', [])
        if not conditions:
            # Try to infer from transformation trigger
            if transformation and transformation.trigger:
                signal_words = transformation.trigger.get('signal_words', [])
                conditions = signal_words[:5]
            if not conditions:
                conditions = [name]

        # Build schemas
        schemas = parsed.get('schemas', [])

        # Generate unique ID
        import hashlib
        raw = f"{source}:{name}:{kind}:{':'.join(sorted(conditions[:5]))}"
        node_id = hashlib.md5(raw.encode()).hexdigest()[:12]

        # Check for duplicate
        existing = self.graph.get_node(node_id)
        if existing:
            logger.debug("Understanding node %s already exists — merging conditions", node_id)
            for cond in conditions:
                if cond not in existing.conditions:
                    existing.conditions.append(cond)
            self.graph._save()
            return existing

        # v35: Build members from parsed data (structural roles)
        members = []
        members_data = parsed.get('members', [])
        if isinstance(members_data, list):
            for m in members_data:
                if isinstance(m, dict) and m.get('role') and m.get('description'):
                    members.append(UnderstandingMember(
                        role=m['role'],
                        description=m['description'],
                        confidence=m.get('confidence', 0.8),
                    ))
                elif isinstance(m, dict) and m.get('role'):
                    # Partial member — role but no description
                    members.append(UnderstandingMember(
                        role=m['role'],
                        description=m.get('description', ''),
                        confidence=m.get('confidence', 0.7),
                    ))

        # v35: If no members were extracted but we have a transformation,
        # infer basic members from the transformation structure
        if not members and transformation:
            members = self._infer_members_from_transformation(transformation, conditions)

        # Create the node
        # v35: New nodes start as NEW (not yet verified)
        node = UnderstandingNode(
            id=node_id,
            name=name,
            concept=concept[:200],
            abstraction=abstraction[:300],
            schemas=schemas[:5],
            transformation=transformation,
            conditions=conditions[:15],
            source=source,
            confidence=0.55,  # LLM-composed understandings start with moderate confidence
            members=members,  # v35: Structural roles
            lifecycle='new',   # v35: Start as NEW — needs verification
            epistemic='observed',  # v35: Observed from LLM output
        )

        return node

    # ═══════════════ MEMBER INFERENCE ═══════════════

    def _infer_members_from_transformation(self, transformation, conditions: list) -> list:
        """Infer structural members from transformation when LLM didn't provide them.

        v35: When Qwen3 doesn't output members in its JSON response, we can
        still infer basic roles from the transformation structure. This ensures
        EVERY understanding has at least a trigger and result role, which is
        critical for:
          - Per-role injection (inject only Trigger when needed)
          - Gap detection (missing Result = gap)
          - Detailed introspection

        Args:
            transformation: The Transformation object
            conditions: The conditions list (signal words)

        Returns:
            List of UnderstandingMember objects.
        """
        members = []

        # Trigger role — from conditions/signal words
        if conditions:
            trigger_desc = ', '.join(conditions[:5])
            members.append(UnderstandingMember(
                role='trigger',
                description=trigger_desc,
                confidence=0.85,
            ))
        elif transformation and transformation.trigger:
            signal_words = transformation.trigger.get('signal_words', [])
            if signal_words:
                members.append(UnderstandingMember(
                    role='trigger',
                    description=', '.join(signal_words[:5]),
                    confidence=0.85,
                ))

        # Result role — from transformation action
        if transformation and transformation.action:
            members.append(UnderstandingMember(
                role='result',
                description=transformation.action[:200],
                confidence=0.75,
            ))

        # Kind-specific role inference
        if transformation:
            kind = transformation.kind

            if kind == 'signal_flip':
                # Signal flip: has default and exception
                members.append(UnderstandingMember(
                    role='default',
                    description='jawaban default sebelum sinyal flip',
                    confidence=0.7,
                ))
                members.append(UnderstandingMember(
                    role='exception',
                    description='jawaban setelah sinyal flip diterapkan',
                    confidence=0.7,
                ))

            elif kind == 'contrast_focus':
                members.append(UnderstandingMember(
                    role='contrast_point',
                    description='bagian yang menjadi fokus setelah kata kontras',
                    confidence=0.7,
                ))

            elif kind == 'comparison_resolve':
                members.append(UnderstandingMember(
                    role='comparator',
                    description='entitas yang dibandingkan',
                    confidence=0.7,
                ))

            elif kind == 'quantity_compute':
                members.append(UnderstandingMember(
                    role='values',
                    description='nilai-nilai yang perlu dihitung',
                    confidence=0.7,
                ))

            elif kind == 'context_filter':
                members.append(UnderstandingMember(
                    role='target_info',
                    description='informasi spesifik yang dicari dalam konteks',
                    confidence=0.7,
                ))

        return members

    # ═══════════════ ANSWER EXTRACTION ═══════════════

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer from LLM composition response.

        Strategy:
        1. Try "Jawaban:" section
        2. Try parsing JSON response and extracting answer/value fields
        3. Last sentence fallback
        """
        if not response or not response.strip():
            return None

        # Strategy 1: Try to find "Jawaban:" section
        jawaban_match = re.search(
            r'Jawaban:\s*(.+?)(?:\n|$)', response, re.IGNORECASE
        )
        if jawaban_match:
            answer = jawaban_match.group(1).strip()
            if answer:
                return answer

        # Strategy 2: Try "Answer:" as fallback
        answer_match = re.search(
            r'Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE
        )
        if answer_match:
            answer = answer_match.group(1).strip()
            if answer:
                return answer

        # Strategy 3: Parse JSON response
        json_answer = self._try_extract_from_json(response)
        if json_answer is not None:
            return json_answer

        # Strategy 4: Last sentence fallback
        # Clean up markdown code blocks first
        cleaned = re.sub(r'```\w*\n?', '', response)
        cleaned = re.sub(r'```', '', cleaned)
        sentences = re.split(r'[.!?]\s*', cleaned.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        if sentences:
            return sentences[-1][:200]

        return response.strip()[:200]

    def _try_extract_from_json(self, response: str) -> Optional[str]:
        """Try to extract answer from JSON-structured LLM response.

        Qwen3-0.6B often returns JSON in markdown code blocks.
        This method extracts the 'answer', 'jawaban', 'value', or
        first meaningful string field from the JSON.
        """
        # Clean markdown code blocks
        cleaned = re.sub(r'```\w*\n?', '', response)
        cleaned = re.sub(r'```', '', cleaned)
        cleaned = cleaned.strip()

        # Try direct JSON parse
        try:
            data = json.loads(cleaned)
            return self._extract_value_from_json(data)
        except json.JSONDecodeError:
            pass

        # Try to find JSON embedded in text
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._extract_value_from_json(data)
            except json.JSONDecodeError:
                pass

        # Try nested JSON
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._extract_value_from_json(data)
            except json.JSONDecodeError:
                pass

        return None

    def _extract_value_from_json(self, data) -> Optional[str]:
        """Extract the most relevant answer field from parsed JSON."""
        if not isinstance(data, dict):
            if isinstance(data, str):
                return data
            return None

        # Priority order for answer fields
        answer_keys = ['answer', 'jawaban', 'Jawaban', 'result', 'value',
                       'response', 'text', 'main_idea', 'message', 'amanat']
        for key in answer_keys:
            if key in data:
                val = data[key]
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, (int, float)):
                    return str(val)

        # If it's a list, try to get first element
        for key, val in data.items():
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
            if isinstance(val, str) and val.strip() and len(val.strip()) > 1:
                return val.strip()

        return None


# ═══════════════ COMPOSER SINGLETON (v31 fix — P2) ═══════════════

_shared_composer = None


def get_shared_composer() -> UnderstandingComposer:
    """Get the shared singleton UnderstandingComposer.

    v31 fix (P2): Previously, three separate places created their own
    UnderstandingComposer instances:
      1. SelfCorrectionLoop.composer property
      2. AnswerHandlers._try_understanding_pipeline()
      3. SelfCore.composer property

    Each instance could load Qwen3-0.6B independently (~600MB each),
    wasting up to 1.8GB RAM. Now all components share ONE instance.

    The composer shares the same UnderstandingGraph via get_shared_graph().
    """
    global _shared_composer
    if _shared_composer is None:
        from derivation.understanding_builder import get_shared_graph
        _shared_composer = UnderstandingComposer(graph=get_shared_graph())
    return _shared_composer
