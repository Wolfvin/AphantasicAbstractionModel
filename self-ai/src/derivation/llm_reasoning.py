# @WHO:   self-ai/src/derivation/llm_reasoning.py
# @WHAT:  LLM Reasoning Engine — Qwen3-0.6B fallback for low-confidence questions
# @PART:  self-ai/derivation
# @ENTRY: LLMReasoningEngine.reason()

"""LLM Reasoning Engine — Qwen3-0.6B fallback for low-confidence questions.

When the concept-based TextComprehension returns confidence < threshold,
route to Qwen3 for Chain-of-Thought reasoning.

Strategies (in order of preference):
  1. z-ai-web-dev-sdk via Node.js subprocess (cloud LLM)
  2. Local Qwen3-0.6B via transformers (if cached)
  3. Graceful degradation — return None so caller keeps concept-based answer

The LLM is OPTIONAL. If unavailable, the system falls back gracefully
without crashing or blocking the concept-based pipeline.
"""
import os
import re
import json
import logging
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class LLMReasoningEngine:
    """Qwen3-0.6B reasoning fallback — called when confidence is low.

    Usage:
        engine = LLMReasoningEngine(confidence_threshold=0.5)
        if engine.should_fallback(result):
            llm_result = engine.reason(text, question, result)
    """

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self._sdk_checked = False
        self._sdk_available = False
        self._local_checked = False
        self._local_available = False
        self.history = []  # Track LLM calls for debugging

    # ── Public API ──────────────────────────────────────────

    def should_fallback(self, result: dict) -> bool:
        """Check if LLM fallback should be used based on concept-based result.

        Returns True if:
          - confidence is below threshold, OR
          - answer is None
        """
        confidence = result.get('confidence', 0.0)
        answer = result.get('answer')
        return confidence < self.confidence_threshold or answer is None

    def reason(self, text: str, question: str, previous_result: dict = None) -> dict:
        """Use Qwen3 for Chain-of-Thought reasoning.

        Strategy:
          1. Try z-ai-web-dev-sdk via Node.js subprocess (cloud LLM)
          2. If unavailable, try local Qwen3 model (transformers)
          3. Parse response to extract answer

        Returns dict with: answer, confidence, method, explanation, previous_result
        """
        prompt = self._build_prompt(text, question)

        # Strategy 1: z-ai-web-dev-sdk (cloud LLM — fast, no local GPU needed)
        answer = self._try_sdk_fallback(text, question, prompt)

        # Strategy 2: Local Qwen3-0.6B (if cached locally)
        if answer is None:
            answer = self._try_local_qwen(prompt)

        # Record in history
        self.history.append({
            'question': question[:100],
            'answer': answer,
            'method': 'llm_reasoning' if answer else 'llm_unavailable',
        })

        if answer:
            return {
                'answer': answer,
                'confidence': 0.6,  # LLM confidence is approximate
                'method': 'llm_reasoning',
                'explanation': 'Answer derived from LLM Chain-of-Thought reasoning',
                'previous_result': previous_result,
            }

        return {
            'answer': None,
            'confidence': 0.0,
            'method': 'llm_unavailable',
            'explanation': 'LLM reasoning unavailable — no SDK or local model found',
        }

    # ── Prompt Construction ─────────────────────────────────

    def _build_prompt(self, text: str, question: str) -> str:
        """Build Chain-of-Thought prompt for Qwen3 (Bahasa Indonesia)."""
        # Truncate text to avoid token limit issues
        max_text_len = 1500
        truncated_text = text[:max_text_len]
        if len(text) > max_text_len:
            truncated_text += '...'

        return (
            f"Berdasarkan teks berikut, jawab pertanyaan dengan menjelaskan "
            f"penalaranmu langkah demi langkah.\n\n"
            f"Teks:\n{truncated_text}\n\n"
            f"Pertanyaan: {question}\n\n"
            f"Jawab dengan format:\n"
            f"Penalaran: [langkah-langkah penalaran]\n"
            f"Jawaban: [jawaban singkat]"
        )

    # ── Strategy 1: z-ai-web-dev-sdk ────────────────────────

    def _try_sdk_fallback(self, text: str, question: str, prompt: str) -> Optional[str]:
        """Use z-ai-web-dev-sdk via Node.js subprocess.

        Creates a temporary JS script, runs it with node, parses output.
        This avoids needing a Python binding for the JS SDK.
        """
        # Check SDK availability once
        if not self._sdk_checked:
            self._sdk_available = self._check_sdk_available()
            self._sdk_checked = True
            if not self._sdk_available:
                logger.info("z-ai-web-dev-sdk not available — skipping cloud LLM")

        if not self._sdk_available:
            return None

        try:
            # Escape text and question for safe JSON embedding in JS
            text_json = json.dumps(text[:1500], ensure_ascii=False)
            question_json = json.dumps(question, ensure_ascii=False)

            script = '''
const ZAI = require('z-ai-web-dev-sdk').default;

async function main() {
    try {
        const zai = await ZAI.create();
        const completion = await zai.chat.completions.create({
            messages: [
                { role: 'system', content: 'Kamu adalah asisten yang menjawab pertanyaan berdasarkan teks yang diberikan. Jawab dalam bahasa Indonesia dengan format:\\nPenalaran: [langkah-langkah]\\nJawaban: [jawaban singkat]' },
                { role: 'user', content: 'Berdasarkan teks berikut, jawab pertanyaan:\\n\\nTeks: ' + %s + '\\n\\nPertanyaan: ' + %s }
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
''' % (text_json, question_json)

            # Write to temp file (safer than -e for large scripts)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(script)
                script_path = f.name

            try:
                # Ensure Node.js can find z-ai-web-dev-sdk
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
                    # Check if output looks like an error message
                    if output.startswith('SDK error:') or output.startswith('No content'):
                        logger.warning("z-ai-web-dev-sdk returned error: %s", output[:200])
                        return None
                    return self._parse_response(output)
                else:
                    if result.stderr:
                        logger.warning("z-ai-web-dev-sdk subprocess error: %s", result.stderr[:200])
            finally:
                os.unlink(script_path)

        except subprocess.TimeoutExpired:
            logger.warning("z-ai-web-dev-sdk call timed out (30s)")
        except FileNotFoundError:
            # node not installed
            self._sdk_available = False
            logger.info("Node.js not found — disabling cloud LLM fallback")
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

    # ── Strategy 2: Local Qwen3-0.6B ───────────────────────

    def _try_local_qwen(self, prompt: str) -> Optional[str]:
        """Try Qwen3-0.6B via shared singleton.

        Uses model_registry.get_shared_qwen() to avoid loading Qwen3
        multiple times (~1.5GB RAM each). The model is loaded once
        and shared across all callers.
        """
        # Check local availability once
        if not self._local_checked:
            self._local_available = self._check_local_qwen_available()
            self._local_checked = True
            if not self._local_available:
                logger.info("Qwen3-0.6B not cached locally — skipping local LLM")

        if not self._local_available:
            return None

        try:
            from derivation.model_registry import get_shared_qwen, is_ollama_available, ollama_generate

            # Try Ollama first (no torch dependency)
            if is_ollama_available():
                response = ollama_generate(prompt, max_tokens=200)
                if response:
                    return self._parse_response(response)

            model, tokenizer = get_shared_qwen()
            if model is None or tokenizer is None:
                logger.warning("Shared Qwen3-0.6B not available")
                return None

            inputs = tokenizer(
                prompt, return_tensors='pt', max_length=512, truncation=True
            )
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.3
            )
            # Decode only the new tokens (skip the prompt)
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            return self._parse_response(response)
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

    # ── Response Parsing ────────────────────────────────────

    def _parse_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract the answer.

        Looks for "Jawaban:" section first, then falls back to
        the last meaningful sentence.
        """
        if not response or not response.strip():
            return None

        # Try to find "Jawaban:" section
        jawaban_match = re.search(
            r'Jawaban:\s*(.+?)(?:\n|$)', response, re.IGNORECASE
        )
        if jawaban_match:
            answer = jawaban_match.group(1).strip()
            if answer:
                return answer

        # Try "Answer:" as fallback (some models use English)
        answer_match = re.search(
            r'Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE
        )
        if answer_match:
            answer = answer_match.group(1).strip()
            if answer:
                return answer

        # If no structured format, return the last sentence
        sentences = re.split(r'[.!?]\s*', response.strip())
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            last = sentences[-1]
            if last:
                return last
            elif len(sentences) > 1:
                return sentences[-2]

        # Last resort: truncate raw response
        return response.strip()[:200]
