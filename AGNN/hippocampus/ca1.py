"""
CA1: Context integration - infer edge types.

Biologis: CA1 integrates contextual information from EC and CA3.
AI: Infer edge type (CATEGORICAL, CAUSAL, DIFFERENTIAL, FUNCTIONAL).

CA1 looks at the *text* of the stimulus (+ correction, if any) and
counts how many cue-words from each relation type appear. The type
with the most hits wins; ties fall back to CATEGORICAL (the safest
default for new knowledge).
"""

from __future__ import annotations

import re
from typing import Dict


class CA1:
    """Context integration mechanism.

    The four edge types from ARCHITECTURE.md section 5 are kept as plain
    strings to match :class:`Episome.edge_type` and
    :class:`Semesome.type`.
    """

    EDGE_TYPES = {"CATEGORICAL", "CAUSAL", "DIFFERENTIAL", "FUNCTIONAL"}

    # Cue-words per relation type. Matched as whole words (case-insensitive).
    # These cues drive a simple bag-of-words classifier - no torch needed.
    _CUES: Dict[str, frozenset] = {
        "CATEGORICAL": frozenset({
            "is", "are", "was", "were", "member", "kind", "type",
            "example", "instance", "subclass", "category", "class",
            "belong", "belongs", "taxon", "taxa",
        }),
        "CAUSAL": frozenset({
            "causes", "caused", "cause", "leads", "produces", "produced",
            "results", "because", "due", "effect", "affects", "affected",
            "induces", "induced", "triggers", "triggered", "consequence",
        }),
        "DIFFERENTIAL": frozenset({
            "not", "unlike", "contrasts", "contrast", "opposite",
            "negates", "excepts", "without", "less", "fewer", "reduces",
            "inversely", "inversely_related", "anti", "non", "inhibit",
            "inhibits",
        }),
        "FUNCTIONAL": frozenset({
            "requires", "enables", "uses", "computes", "function",
            "needs", "depends", "powered", "operates", "transforms",
            "applies", "processes", "supports", "facilitates",
        }),
    }

    def __init__(self) -> None:
        """Allocate the inferred-types cache (episome_id -> edge_type)."""
        self.inferred_types: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def integrate_context(self, stimulus: str, correction: str = "") -> str:
        """Infer the edge type from the stimulus + correction text.

        Biologis: CA1 compares EC input with CA3 output to derive context.
        AI: Heuristic classifier - count cue-word hits per relation type
            and pick the winner. Ties fall back to CATEGORICAL.

        Args:
            stimulus: The input stimulus text.
            correction: Optional correction text. If provided, it is
                concatenated with the stimulus before inference - this is
                how CA1 sees the full learning context (the question +
                the corrected fact).

        Returns:
            One of :attr:`CA1.EDGE_TYPES` (a string).
        """
        text = f"{stimulus} {correction}".strip().lower()
        scores: Dict[str, int] = {t: 0 for t in self._CUES}
        for rel_type, cues in self._CUES.items():
            for cue in cues:
                # Word-boundary match so "is" does not match "this".
                if re.search(r"\b" + re.escape(cue) + r"\b", text):
                    scores[rel_type] += 1
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            # No cue-words at all - default to CATEGORICAL (the relation
            # type AGNN uses for "X is a Y" / "X is a member of Y", which
            # is the safest bet for freshly encoded facts).
            best = "CATEGORICAL"
        return best

    # ------------------------------------------------------------------
    # Overload accepting an episome_id (skeleton-era signature)
    # ------------------------------------------------------------------

    def integrate_context_for(self, episome_id: int, stimulus: str, correction: str = "") -> str:
        """Same as :meth:`integrate_context`, but also cache the result
        keyed by ``episome_id`` so other components can look it up later.

        This bridges the skeleton API (which took only ``episome_id``)
        with the real input CA1 needs (the text).
        """
        edge_type = self.integrate_context(stimulus, correction)
        self.inferred_types[episome_id] = edge_type
        return edge_type
