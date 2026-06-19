"""
APHANTASIC CHAIN FORMATTER: 3-layer chain for the articulate prompt.

Biologis: When an aphantasic recalls a chain of facts, they don't
picture a sequence of images — they verbally walk through each concept's
*definition* and its *causal relations* to the next concept. This is
the "External Recoding" compensatory strategy (PMC11910157): the
aphantasic converts the chain into a structured verbal form so they
can hold it in working memory without visual imagery.

AI: This module formats the retrieved episomes + semesomes into a
3-layer chain that Qwen3 can read during articulation:

    KONSEP: api
    DEFINISI: fenomena pembakaran yang menghasilkan panas
    RELASI:
      - (CAUSAL) → panas
      - (FUNCTIONAL) → oksigen

    KONSEP: panas
    DEFINISI: bentuk energi yang dirasakan sebagai suhu tinggi
    RELASI:
      - (FUNCTIONAL) → air

This is the critical disambiguator that solves "api"/"API" and
"air"/"air" at the *node* level (not just the prompt level). Even if
Qwen3's prior leans toward "API (programming)", the DEFINISI field
("fenomena pembakaran") and the RELASI field ("(CAUSAL) → panas")
make the intended meaning unambiguous — programming APIs don't cause
heat or require oxygen.

Why not just dump all Episome fields as JSON:
  - Qwen3-0.6B is a small model. JSON parsing eats context tokens
    and confuses the model with structural noise. Prose with clear
    section markers (KONSEP / DEFINISI / RELASI) is what Qwen3 was
    post-trained on.
  - The Phase 0 system message already tells Qwen3 to expect the
    [Knowledge Graph Context] block in a particular shape. The
    formatter's output is the body of that block.

Layer availability:
  - Layer 1 (surface): always available — it's the Episome.text.
  - Layer 2 (definition): available only after the lazy
    DefinitionExtractor has run for this node. Before that, the
    DEFINISI line is omitted (graceful degradation).
  - Layer 3 (causal anchors): available only if the
    CausalAnchorBuilder found a causal relation in the correction
    text. For CATEGORICAL corrections ("X adalah Y"), the RELASI
    line is omitted.
  When both Layer 2 and Layer 3 are unavailable, the formatter
  emits just the KONSEP line — equivalent to the pre-Phase-1
  surface-form chain, so existing tests that don't populate the
  new fields still get a working (if less disambiguating) prompt.

Backward compatibility:
  - When called with Episomes that have empty ``amodal_definition``
    and empty ``causal_anchors`` (the pre-Phase-1 default), the
    formatter's output reduces to a newline-separated list of
    KONSEP lines. This means existing tests that call ``_articulate``
    without setting the new fields still get a coherent chain (and
    the prompt-structure tests in test_qwen3_integration.py, which
    check the user-message prefix + Q:/A: markers, continue to pass
    because the formatter only affects the *body* of the
    [Knowledge Graph Context] block, not its delimiters).
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence

logger = logging.getLogger(__name__)


# Maximum characters of the Episome.text to include in the KONSEP line.
# Long corrections (e.g. multi-sentence teaching) are truncated so the
# chain stays within Qwen3-0.6B's context budget. 120 chars ≈ 30 tokens,
# generous for a single concept label.
_MAX_SURFACE_CHARS = 120

# Maximum number of causal anchors to render per node. Even if a node
# has 10 anchors, we only show the top 5 (sorted by relation type
# priority: CAUSAL > FUNCTIONAL > DIFFERENTIAL) to keep the prompt
# compact.
_MAX_ANCHORS_PER_NODE = 5

# Relation-type display priority (lower = higher priority). When a node
# has more than ``_MAX_ANCHORS_PER_NODE`` anchors, this ordering
# decides which ones make the cut.
_ANCHOR_PRIORITY = {
    "CAUSAL": 0,
    "FUNCTIONAL": 1,
    "DIFFERENTIAL": 2,
}


class AphantasicChainFormatter:
    """Format retrieved episomes into a 3-layer aphantasic chain.

    Stateless — the formatter holds no mutable state. It reads the
    Episome fields (text, amodal_definition, causal_anchors) and the
    optional Semesome edges, then emits the KONSEP/DEFINISI/RELASI
    block.

    The formatter is decoupled from AGNNCore so it can be unit-tested
    in isolation with synthetic Episome instances.
    """

    def __init__(
        self,
        max_surface_chars: int = _MAX_SURFACE_CHARS,
        max_anchors_per_node: int = _MAX_ANCHORS_PER_NODE,
    ) -> None:
        """Store the rendering limits.

        Args:
            max_surface_chars: Max chars of ``Episome.text`` to render
                in the KONSEP line. Exposed for tests.
            max_anchors_per_node: Max number of causal anchors to
                render per node (top-N by priority). Exposed for tests.
        """
        self._max_surface_chars = max_surface_chars
        self._max_anchors_per_node = max_anchors_per_node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(
        self,
        episomes: Sequence[Any],
        semesomes: Sequence[Any] = None,
    ) -> str:
        """Build the 3-layer chain string for the articulate prompt.

        Args:
            episomes: The retrieved Episome instances (typically 3–5
                from PapezCircuit.retrieve). Order matters — the
                formatter preserves retrieval order so Qwen3 reads
                the chain head-to-tail.
            semesomes: Optional list of Semesome edges between the
                retrieved episomes. Currently unused (the RELASI
                section draws from Episome.causal_anchors, not from
                the cross-node semesomes), but kept in the signature
                for future Phase 2 work where we may render both
                intra-node anchors and inter-node edges.

        Returns:
            A multi-line string with one block per episome:

                KONSEP: api
                DEFINISI: fenomena pembakaran yang menghasilkan panas
                RELASI:
                  - (CAUSAL) → panas
                  - (FUNCTIONAL) → oksigen

            Blocks are separated by a blank line. Layers 2 and 3 are
            omitted when the corresponding Episome fields are empty
            (graceful degradation to surface-form-only).

            Empty string if:
              - ``episomes`` is empty
              - all episomes have empty ``text``
        """
        if not episomes:
            return ""

        blocks: List[str] = []
        for epi in episomes:
            block = self._format_one(epi)
            if block:
                blocks.append(block)

        if not blocks:
            return ""
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_one(self, episome: Any) -> str:
        """Format a single Episome into a KONSEP/DEFINISI/RELASI block.

        Returns an empty string if the episome has no text (we can't
        render a KONSEP line without a surface form).
        """
        text = self._get_surface(episome)
        if not text:
            return ""

        lines: List[str] = [f"KONSEP: {text}"]

        definition = self._get_definition(episome)
        if definition:
            lines.append(f"DEFINISI: {definition}")

        anchors = self._get_anchors(episome)
        if anchors:
            lines.append("RELASI:")
            for rel_type, target in anchors:
                lines.append(f"  - ({rel_type}) → {target}")

        return "\n".join(lines)

    def _get_surface(self, episome: Any) -> str:
        """Return the truncated surface form (Layer 1)."""
        text = getattr(episome, "text", "") or ""
        text = text.strip()
        if not text:
            return ""
        if len(text) > self._max_surface_chars:
            return text[: self._max_surface_chars].rstrip() + "…"
        return text

    def _get_definition(self, episome: Any) -> str:
        """Return the amodal definition (Layer 2), or empty string."""
        definition = getattr(episome, "amodal_definition", "") or ""
        return definition.strip()

    def _get_anchors(
        self,
        episome: Any,
    ) -> List[tuple]:
        """Return the top-N causal anchors (Layer 3), sorted by priority.

        Returns an empty list if the episome has no causal_anchors or
        if the field is missing.
        """
        anchors = getattr(episome, "causal_anchors", None) or ()
        if not anchors:
            return []
        # Sort by relation-type priority (CAUSAL > FUNCTIONAL >
        # DIFFERENTIAL), then alphabetically by target for stable
        # ordering across runs.
        sorted_anchors = sorted(
            anchors,
            key=lambda a: (
                _ANCHOR_PRIORITY.get(a[0], 99),
                a[1] if len(a) > 1 else "",
            ),
        )
        return list(sorted_anchors[: self._max_anchors_per_node])
