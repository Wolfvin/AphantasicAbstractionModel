"""
DEFINITION EXTRACTOR: Aphantasic Layer 2 — amodal verbal definition.

Biologis: Aphantasics do not store visual imagery. They compensate by
storing *verbal definitions* of concepts (Bainbridge 2021; Monzel 2024;
PMC11910157 calls this the "Semantic Reliance" compensatory strategy).
When asked about "api", they don't picture fire — they recall the verbal
fact "fenomena pembakaran yang menghasilkan panas dan cahaya".

AI: This module generates that verbal definition lazily, the first time
an Episome is articulated. The definition is cached forever on the
Episome (by text hash) so subsequent articulations skip the LLM call —
unless the node's confidence grows past the invalidate threshold (set
in AGNNCore), in which case the cache is invalidated and the next
articulation re-generates the definition.

Why lazy generation (not at ``learn()`` time):
  - ``learn()`` may be called many times in a burst (user teaches 20
    facts in a row). Blocking 0.5–2 seconds per node for definition
    generation is a poor UX.
  - ``_articulate()`` retrieves at most 3–5 nodes per query, so the
    worst-case per-query cost is 3–5 LLM calls — acceptable.
  - Most nodes are never articulated (the user teaches more than they
    ask), so eager generation wastes compute.

Why cache by text hash (not by node id):
  - Two Episomes with identical ``text`` (e.g. user re-learns the same
    fact after a penalize) should share the same definition. Caching
    by text hash deduplicates the LLM calls across the whole AGNNCore
    instance.
  - The cache lives on the AGNNCore instance (not on the Episome) so
    even after an Episome's ``definition_dirty`` flag invalidates its
    own cached value, the cross-Episome cache can still serve a fresh
    copy if the text matches.

Why a ≤15-word limit:
  - Mirrors the "Condensation of Inner Speech" compensatory strategy
    (PMC11910157): aphantasics use *dense* verbal internal speech, not
    long explanations. A short definition is more disambiguating per
    token than a paragraph.
  - Keeps the articulate prompt small. With 3–5 nodes per chain and
    15 words per definition, the DEFINISI section adds ~75 words —
    well within Qwen3-0.6B's 8K context budget alongside the Phase 0
    system message + the chain + the Q/A pair.

No-think mode:
  - Definition generation is an *extraction* task, not a *reasoning*
    task. Qwen3's thinking mode (Long-CoT) is overkill and tends to
    add speculative noise. We use the ``/no_think`` prefix in the
    system message (per the Qwen3 technical report) to keep the model
    in direct-generation mode. This is faster and more accurate for
    this task.

Failure contract:
  - Any exception (model not loaded, generation error, parse error)
    returns an empty string. The caller (``_articulate``) treats an
    empty definition as "Layer 2 unavailable — fall back to surface
    form only". This keeps the articulate pipeline robust to model
    unavailability.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Maximum words in a generated definition. Mirrors the "Condensation
# of Inner Speech" aphantasic strategy — short, dense, disambiguating.
_MAX_DEFINITION_WORDS = 15

# System message for definition generation. Uses /no_think to keep
# Qwen3 in direct-generation mode (definition extraction is not a
# reasoning task). The message is multilingual by design — it
# instructs the model to respond in the same language as the input
# text, so an Indonesian correction gets an Indonesian definition
# and an English correction gets an English definition. This matches
# the Phase 1 design decision (point 4 in the user's confirmation):
# "Ikut bahasa correction".
_DEFINITION_SYSTEM_MESSAGE = (
    "/no_think\n"
    "Tugasmu: berikan definisi singkat (maksimal 15 kata) untuk "
    "konsep utama dalam teks. Definisi harus berupa fakta konseptual "
    "verbal, bukan deskripsi visual. Jawab dalam bahasa yang sama "
    "dengan teks input. Jawab HANYA definisi, tanpa pembuka atau "
    "penutup."
)

# User-message template. The triple-brace ``{text}`` is filled with
# the raw correction text. We keep the template minimal so the model
# has maximum surface area to disambiguate the concept from context.
_DEFINITION_USER_TEMPLATE = "Teks: {text}\nDefinisi:"


class DefinitionExtractor:
    """Generate and cache amodal verbal definitions for Episomes.

    The extractor is owned by ``AGNNCore`` (one instance per brain).
    It holds:
      - ``_cache``: ``{text_hash: definition}`` — survives across
        Episomes. Keyed by SHA-256 of the normalized text so two
        Episomes with the same correction share one definition.
      - ``_model`` / ``_tokenizer``: borrowed from AGNNCore on each
        ``extract()`` call (lazy — the extractor does not own the
        model lifecycle). When either is None, ``extract()`` returns
        an empty string without raising.
      - ``_max_words``: the word limit, exposed for tests.

    Thread safety:
        Not thread-safe. AGNNCore is single-threaded by design (the
        trisynaptic circuit mutates shared state). If a future caller
        needs concurrent articulation, wrap ``extract()`` in a lock
        at the AGNNCore level.
    """

    def __init__(self, max_words: int = _MAX_DEFINITION_WORDS) -> None:
        """Allocate the cache + store the word limit.

        Args:
            max_words: Maximum words in a generated definition.
                Defaults to 15 (matches the aphantasic "condensed
                inner speech" compensatory strategy). Exposed for
                tests so they can assert the contract without
                hard-coding the constant.
        """
        self._cache: Dict[str, str] = {}
        self._max_words: int = max_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        model: Any,
        tokenizer: Any,
        force_refresh: bool = False,
    ) -> str:
        """Generate (or fetch from cache) an amodal definition for ``text``.

        Args:
            text: The correction text to define. Normalized internally
                (strip + collapse whitespace) before hashing.
            model: A HuggingFace causal LM (typically Qwen3-0.6B).
                ``None`` short-circuits to empty string.
            tokenizer: The matching tokenizer. ``None`` short-circuits.
            force_refresh: When True, bypass the cache and re-generate.
                Used by ``AGNNCore._articulate`` when the Episome's
                ``definition_dirty`` flag is set (i.e. confidence has
                grown past the invalidate threshold).

        Returns:
            The definition string (≤ ``max_words`` words, same
            language as ``text``). Empty string if:
              - ``text`` is empty/whitespace
              - ``model`` or ``tokenizer`` is None
              - generation raises any exception
              - the generated text is empty after cleaning
            The caller treats empty string as "Layer 2 unavailable"
            and falls back to surface-form-only articulation.
        """
        normalized = self._normalize(text)
        if not normalized:
            return ""
        if model is None or tokenizer is None:
            return ""

        cache_key = self._hash(normalized)
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Generate. Any exception → empty string (failure contract).
        try:
            definition = self._generate(normalized, model, tokenizer)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "DefinitionExtractor: generation failed for %r: %s",
                normalized[:60], exc,
            )
            return ""

        # Clean + enforce word limit (the prompt asks for ≤15 words
        # but models sometimes add a trailing period or extra clause).
        definition = self._clean(definition)
        if not definition:
            return ""

        # Cache and return. Even on a "weak" definition we cache —
        # re-generation would likely produce the same output (Qwen3
        # with do_sample=False is deterministic).
        self._cache[cache_key] = definition
        return definition

    def invalidate(self, text: str) -> None:
        """Drop the cached definition for ``text`` (if any).

        Called by ``AGNNCore.reinforce`` when the cumulative confidence
        delta on a node exceeds the invalidate threshold. The next
        ``extract()`` call for the same text will re-generate.

        Note: this only invalidates the cross-Episome text-hash cache.
        The Episome's own ``amodal_definition`` field is reset
        separately by the caller (set to "" so the lazy-populate path
        in ``_articulate`` re-fetches).
        """
        normalized = self._normalize(text)
        if not normalized:
            return
        cache_key = self._hash(normalized)
        self._cache.pop(cache_key, None)

    @property
    def max_words(self) -> int:
        """The word limit enforced on generated definitions."""
        return self._max_words

    @property
    def cache_size(self) -> int:
        """Number of cached definitions (for tests / introspection)."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Strip + collapse internal whitespace.

        We do NOT lowercase — the definition prompt passes the original
        case to Qwen3 so proper nouns ("Socrates", "Jakarta") survive.
        """
        if not text:
            return ""
        return " ".join(text.split())

    @staticmethod
    def _hash(normalized: str) -> str:
        """SHA-256 of the normalized text (hex digest)."""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _generate(
        self,
        text: str,
        model: Any,
        tokenizer: Any,
    ) -> str:
        """Call Qwen3 to generate the definition.

        Builds a 2-message chat (system + user), applies the chat
        template, generates with ``do_sample=False`` for determinism,
        and returns the decoded new tokens (prompt tokens stripped).

        We deliberately do NOT call ``AGNNCore._generate`` here — the
        extractor must be decoupled from AGNNCore so it can be unit
        tested with a mock model/tokenizer in isolation. The chat
        template + slicing logic is duplicated (small, ~15 lines) but
        the decoupling is worth it.

        The system message uses ``/no_think`` (Qwen3 spec) to disable
        Long-CoT — definition extraction is not a reasoning task, and
        thinking mode adds speculative noise + latency.
        """
        import torch  # noqa: WPS433 — local import keeps the module
        # importable in environments without torch (e.g. pure unit
        # tests of the cache logic).
        chat_messages = [
            {"role": "system", "content": _DEFINITION_SYSTEM_MESSAGE},
            {"role": "user", "content": _DEFINITION_USER_TEMPLATE.format(text=text)},
        ]
        chat_text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,  # 15 words ≈ 30 tokens; 60 is headroom
                do_sample=False,
                repetition_penalty=1.1,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        return raw or ""

    def _clean(self, definition: str) -> str:
        """Trim, strip quotes/periods, enforce the word limit.

        Qwen3 sometimes wraps the answer in quotes or adds a trailing
        period. We strip both so the definition is a clean phrase.
        The word limit is enforced as a hard cut — if the model
        ignored the ≤15-word instruction, we truncate at 15 words.
        """
        s = (definition or "").strip().strip('"').strip("'").strip()
        # Strip a single trailing period (Indonesian/English convention).
        if s.endswith("."):
            s = s[:-1].strip()
        if not s:
            return ""
        words = s.split()
        if len(words) > self._max_words:
            s = " ".join(words[: self._max_words])
        return s
