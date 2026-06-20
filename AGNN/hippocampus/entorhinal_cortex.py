"""
ENTORHINAL CORTEX: Primary input gateway - receive stimulus.

Biologis: EC is the main interface between neocortex and hippocampus.
AI: Pre-process incoming stimulus (tokenization, normalization).

Circuit: EC -> DG / CA3 (input layer of trisynaptic pathway).

This implementation is pure-Python (no torch) and uses a minimal
stop-word list + regex tokenizer. The output is a dict carrying the
normalized text, the raw token list, and the keyword list (tokens with
stop-words filtered out). Downstream structures (DG, CA3, CA1) consume
these fields.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Minimal stop-word list - keeps the math/semantic keywords that actually
# discriminate meaning. Larger lists (NLTK etc.) would add a dependency.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "as",
    "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "it", "its", "from", "into", "than", "so", "such", "not", "no",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EntorhinalCortex:
    """Input gateway for the hippocampal trisynaptic circuit.

    The EC normalizes incoming text and extracts keywords that downstream
    structures (DG, CA3) use for pattern separation and autoassociation.
    """

    def __init__(self) -> None:
        """Allocate an input buffer for recently processed stimuli."""
        # Buffer holds the last N normalized stimuli - mainly for debugging
        # and for downstream structures that want a sliding window of input.
        self.input_buffer: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_input(self, stimulus: str) -> Dict[str, object]:
        """Receive stimulus and prepare it for hippocampal processing.

        Biologis: EC layers II/III project to DG and CA3.
        AI: Normalize text, store in buffer, return processed stimulus.

        Args:
            stimulus: Raw input text.

        Returns:
            Dict with keys:
                - text: normalized stimulus (lowercased, whitespace-collapsed).
                - tokens: ordered list of word tokens.
                - keywords: tokens with stop-words filtered out.
        """
        if not isinstance(stimulus, str):
            raise TypeError(
                f"EntorhinalCortex.normalize_input expects str, got {type(stimulus).__name__}"
            )
        text = stimulus.strip().lower()
        # Collapse internal whitespace so downstream keyword extraction is stable.
        text = re.sub(r"\s+", " ", text)
        tokens = _TOKEN_RE.findall(text)
        keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
        record: Dict[str, object] = {"text": text, "tokens": tokens, "keywords": keywords}
        self.input_buffer.append(record)
        return record
