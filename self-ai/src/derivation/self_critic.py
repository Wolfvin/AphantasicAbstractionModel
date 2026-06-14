# @WHO:   self-ai/src/derivation/self_critic.py
# @WHAT:  Self-Critique Pipeline — model evaluates its own answers without user input
# @PART:  self-ai/derivation
# @ENTRY: SelfCritic.critique(), SelfCritic.should_learn()

"""Self-Critique Pipeline — autonomous answer evaluation and self-learning.

Vision:
    SELF-AI saat ini memerlukan input user untuk belajar:
    user tanya → model jawab → user koreksi → SELF simpan sebagai UnderstandingNode.

    Self-Critique Pipeline menghilangkan kebutuhan input user dengan membuat
    model mengevaluasi sendiri apakah jawabannya benar atau perlu diperbaiki.

    Flow:
        1. Model menghasilkan jawaban untuk sebuah pertanyaan
        2. SelfCritic.critique() mengevaluasi jawaban tersebut
        3. Jika should_learn() → True, otomatis buat UnderstandingNode
           dari critique dan simpan ke UnderstandingGraph

    Ini menutup loop belajar — SELF bisa memperbaiki dirinya sendiri
    tanpa menunggu koreksi eksternal.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  Model generates answer for question                      │
    └──────────────┬───────────────────────────────────────────┘
                   ▼
    ┌──────────────────────────────────────────────────────────┐
    │  SelfCritic.critique(question, answer)                    │
    │     - Prompt Qwen3 untuk evaluasi mandiri                │
    │     - Parse output → {confident, critique, correction}   │
    └──────────────┬───────────────────────────────────────────┘
                   ▼
    ┌──────────────────────────────────────────────────────────┐
    │  SelfCritic.should_learn(critique_result)                 │
    │     - confident=False → layak dipelajari                 │
    │     - suggested_correction != None → ada perbaikan       │
    └──────────────┬───────────────────────────────────────────┘
                   ▼ (if should_learn)
    ┌──────────────────────────────────────────────────────────┐
    │  Create UnderstandingNode from critique                   │
    │  Save to UnderstandingGraph                               │
    └──────────────────────────────────────────────────────────┘

Integration:
    CompositionLayer memanggil SelfCritic setelah setiap generate()
    ketika flag self_critique_enabled=True. Jika should_learn() → True,
    CompositionLayer otomatis membuat UnderstandingNode dan menyimpannya.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SelfCritic:
    """Self-critique pipeline — model evaluates its own answers autonomously.

    Uses Qwen3-0.6B (via shared singleton from model_registry) to
    evaluate whether a generated answer is correct for a given question.
    If the model determines the answer is incorrect or incomplete,
    it provides a critique and a suggested correction.

    Usage:
        critic = SelfCritic()
        result = critic.critique("Berapa 2+3?", "6")
        # result = {
        #     'confident': False,
        #     'critique': 'Jawaban salah, 2+3=5 bukan 6',
        #     'suggested_correction': '5'
        # }
        if critic.should_learn(result):
            # Save as UnderstandingNode
    """

    def __init__(self):
        """Initialize SelfCritic with lazy model loading.

        The Qwen3 model is NOT loaded at construction time.
        It is loaded lazily on first critique() call via
        model_registry.get_shared_qwen() to avoid wasting RAM
        if self-critique is never used.
        """
        self._model = None
        self._tokenizer = None
        self._model_loaded = False

    def _ensure_model(self):
        """Lazy-load Qwen3-0.6B via shared singleton.

        Uses get_shared_qwen() from model_registry so the same
        model instance is shared across all consumers (CompositionLayer,
        LLMReasoningEngine, etc). This avoids loading Qwen3 multiple
        times (~1.5GB RAM each).
        """
        if self._model_loaded:
            return

        self._model_loaded = True  # Mark as attempted, even if it fails

        try:
            from derivation.model_registry import get_shared_qwen
            model, tokenizer = get_shared_qwen()
            if model is not None and tokenizer is not None:
                self._model = model
                self._tokenizer = tokenizer
                logger.info("SelfCritic: Qwen3-0.6B loaded via shared singleton")
            else:
                logger.warning(
                    "SelfCritic: Qwen3-0.6B not available — "
                    "critique will use template-based fallback"
                )
        except ImportError:
            logger.warning(
                "SelfCritic: model_registry not available — "
                "critique will use template-based fallback"
            )
        except Exception as e:
            logger.warning("SelfCritic: failed to load Qwen3 model: %s", e)

    def critique(self, question: str, answer: str) -> dict:
        """Evaluate a generated answer for a given question.

        Prompts Qwen3 to self-evaluate whether its answer is correct.
        If the model is unavailable, falls back to heuristic-based
        evaluation (always confident=True to avoid false learning).

        Args:
            question: The original question that was asked.
            answer: The answer that the model generated.

        Returns:
            dict with keys:
                confident (bool): Whether the answer is deemed correct.
                critique (str): Explanation of why the answer is right or wrong.
                suggested_correction (str | None): Corrected answer if not confident.
        """
        if not question or not answer:
            return {
                'confident': True,
                'critique': 'Empty question or answer — cannot critique.',
                'suggested_correction': None,
            }

        self._ensure_model()

        if self._model is not None and self._tokenizer is not None:
            return self._critique_with_model(question, answer)
        else:
            return self._critique_fallback(question, answer)

    def _critique_with_model(self, question: str, answer: str) -> dict:
        """Use Qwen3-0.6B to evaluate the answer.

        Constructs a prompt asking the model to evaluate its own answer,
        then parses the structured output.
        """
        prompt = (
            f"Kamu baru menjawab: {answer}\n"
            f"Apakah jawabanmu sudah tepat untuk pertanyaan: {question}?\n"
            f"Jika tidak, apa yang seharusnya?\n\n"
            f"Jawab dengan format persis seperti ini:\n"
            f"TEPAT: ya/tidak\n"
            f"KRITIK: [penjelasan singkat mengapa tepat atau tidak]\n"
            f"KOREKSI: [jawaban yang benar, atau - jika sudah tepat]"
        )

        try:
            import torch

            # Build messages with /no_think for fast response
            messages = [
                {
                    "role": "system",
                    "content": (
                        "/no_think\nKamu adalah evaluator yang jujur. "
                        "Kamu mengevaluasi jawabanmu sendiri secara kritis. "
                        "Jawab dalam bahasa Indonesia dengan format yang diminta."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt")

            # Move to same device as model
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                )

            # Decode only generated tokens (skip prompt)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_response = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            # Strip thinking blocks (Qwen3 <think...</think/>)
            response = self._strip_thinking(raw_response)

            if not response or len(response.strip()) < 5:
                # Retry once — Qwen3 /no_think sometimes returns empty
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                    )
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                raw_response = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                response = self._strip_thinking(raw_response)

            return self._parse_critique_response(response)

        except Exception as e:
            logger.warning("SelfCritic: model inference failed: %s", e)
            return self._critique_fallback(question, answer)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip Qwen3 thinking blocks from generated text.

        Qwen3-0.6B may produce <think...reasoning...</think<actual_response.
        We only want the actual response after </think.
        """
        end_think = text.find('</think')
        if end_think != -1:
            text = text[end_think:]
            tag_end = text.find('>', 0)
            if tag_end != -1:
                text = text[tag_end + 1:]
            else:
                text = text[len('</think'):]

        # If there's <think without </think, model is still thinking
        if '<think' in text:
            return ''

        # Clean up residual special tokens <|...|>
        text = re.sub(r'<\|[^|]+\|>', '', text)

        return text.strip()

    @staticmethod
    def _parse_critique_response(response: str) -> dict:
        """Parse the structured critique response from Qwen3.

        Expected format:
            TEPAT: ya/tidak
            KRITIK: [explanation]
            KOREKSI: [correction or -]

        Falls back gracefully if the format is not followed exactly.
        """
        if not response or len(response.strip()) < 3:
            return {
                'confident': True,
                'critique': 'Model returned empty critique — assuming correct.',
                'suggested_correction': None,
            }

        response_lower = response.lower().strip()

        # Parse TEPAT field
        confident = True  # Default: trust the model's answer
        tepat_match = re.search(r'tepat\s*:\s*(ya|tidak|yes|no)', response_lower)
        if tepat_match:
            tepat_val = tepat_match.group(1)
            confident = tepat_val in ('ya', 'yes')
        else:
            # Heuristic: if response contains "tidak tepat" or "salah", not confident
            if any(phrase in response_lower for phrase in
                   ['tidak tepat', 'salah', 'kurang tepat', 'tidak benar',
                    'incorrect', 'wrong']):
                confident = False

        # Parse KRITIK field
        critique_text = ''
        kritik_match = re.search(
            r'kritik\s*:\s*(.+?)(?=\n\s*koreksi|\n\s*TEPAT|$)',
            response, re.IGNORECASE | re.DOTALL
        )
        if kritik_match:
            critique_text = kritik_match.group(1).strip()
        else:
            # Use entire response as critique if structured format not found
            critique_text = response.strip()

        # Parse KOREKSI field
        suggested_correction = None
        koreksi_match = re.search(r'koreksi\s*:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
        if koreksi_match:
            correction = koreksi_match.group(1).strip()
            # "-" or "tidak ada" means no correction needed
            if correction and correction not in ('-', 'tidak ada', 'tidak perlu', 'none', 'n/a'):
                suggested_correction = correction

        # If not confident but no correction suggested, still mark as learning-worthy
        # but without a specific correction
        if not confident and suggested_correction is None:
            suggested_correction = None  # Keep None — the critique itself is valuable

        return {
            'confident': confident,
            'critique': critique_text,
            'suggested_correction': suggested_correction,
        }

    @staticmethod
    def _critique_fallback(question: str, answer: str) -> dict:
        """Fallback critique when Qwen3 model is unavailable.

        Returns confident=True to avoid false learning when we cannot
        properly evaluate. This is the safe default — it's better to
        not learn from an uncertain critique than to learn wrong things.
        """
        return {
            'confident': True,
            'critique': (
                f'Model tidak tersedia untuk evaluasi mandiri. '
                f'Jawaban "{answer}" untuk pertanyaan "{question}" '
                f'dianggap benar secara default (tidak bisa dievaluasi).'
            ),
            'suggested_correction': None,
        }

    def should_learn(self, critique_result: dict) -> bool:
        """Determine if a critique result is worth saving as an UnderstandingNode.

        A critique is worth learning from when:
        1. The model is NOT confident in its answer (confident=False)
        2. There is substantive critique text (not just empty/placeholder)

        This prevents saving trivial or uncertain critiques that would
        pollute the understanding graph.

        Args:
            critique_result: The dict returned by critique().

        Returns:
            True if this critique should be saved as an UnderstandingNode.
        """
        if not critique_result:
            return False

        confident = critique_result.get('confident', True)
        critique_text = critique_result.get('critique', '')

        # Learn only when NOT confident AND critique is substantive
        if confident:
            return False

        # Critique must have real content, not just placeholder text
        if not critique_text or len(critique_text.strip()) < 5:
            return False

        # Don't learn from fallback critiques (model unavailable)
        if 'tidak tersedia untuk evaluasi' in critique_text:
            return False

        return True
