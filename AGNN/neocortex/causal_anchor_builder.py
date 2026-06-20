"""
CAUSAL ANCHOR BUILDER: Aphantasic Layer 3 — cause-effect relations.

Biologis: Aphantasics prioritise *cause-effect* relationships over
visual details when storing memories (Monzel 2024: "Rather they focus
on actions and cause-effect relationships, creating an emotional
distance from memory"). When asked about "api", an aphantasic does
not picture fire — they recall the causal structure: api *causes*
panas, api *requires* oksigen, api *requires* bahan bakar.

AI: This module extracts that causal structure from the correction
text by reusing the existing ``SemanticRoleClassifier``'s SPO parser.
The SPO parse already gives us (subject, predicate, object); we map
the predicate to a RelationType via the same classifier, then emit
``(relation_type_name, object)`` pairs as the causal anchors.

Why a separate module (not inline in TrisynapticCircuit):
  - The extraction logic is non-trivial (handle negation, multi-word
    predicates, fallback to middle-token heuristic) and deserves its
    own unit tests.
  - Future Phase 2+ work may extend causal anchors with cross-node
    inference (e.g. if (api, causes, panas) and (panas, requires,
    air) are both in the graph, infer (api, requires, air) as a
    transitive anchor). Keeping the builder separate makes that
    extension cleaner.

No LLM call:
  - Unlike ``DefinitionExtractor`` (Layer 2), this builder is 100%
    deterministic — it reuses the SPO parser + seed keyword tables
    that ``SemanticRoleClassifier`` already owns. No new LLM call,
    no caching, no invalidation. The anchors are computed once at
    ``learn()`` time (cheap) and stored immutably on the Episome.

Failure contract:
  - Any exception returns an empty tuple. The caller (``_articulate``)
    treats an empty tuple as "Layer 3 unavailable — fall back to
    surface form + definition only".
"""

from __future__ import annotations

import logging
from typing import Tuple

from neocortex.semantic_role_classifier import RelationType, SemanticRoleClassifier

logger = logging.getLogger(__name__)


# Relation types considered "causal" for the anchor layer. We include
# CAUSAL (X causes Y), FUNCTIONAL (X requires/enables Y), and
# DIFFERENTIAL (X negates Y) because all three carry cause-effect
# semantics that an aphantasic would prioritise. CATEGORICAL (X is Y)
# and DISCURSIVE (X according to Y) are excluded — they're identity
# or attribution, not cause-effect. TEMPORAL/SPATIAL are excluded
# because they're sequencing/containment, not causation.
# This set is deliberately a tuple (immutable) so callers can rely on
# it not being mutated.
_CAUSAL_RELATION_TYPES: Tuple[RelationType, ...] = (
    RelationType.CAUSAL,
    RelationType.FUNCTIONAL,
    RelationType.DIFFERENTIAL,
)


class CausalAnchorBuilder:
    """Extract ``(relation_type, target)`` pairs from correction text.

    Stateless — the builder holds no mutable state. It borrows the
    ``SemanticRoleClassifier`` instance owned by ``TrisynapticCircuit``
    so the SPO parse + frequency-table learning happens through the
    same classifier the circuit already uses for edge-type inference.
    This ensures the anchors and the edge type are always consistent
    (both derived from the same SPO parse).
    """

    def __init__(
        self,
        classifier: SemanticRoleClassifier,
        causal_types: Tuple[RelationType, ...] = _CAUSAL_RELATION_TYPES,
    ) -> None:
        """Wire the builder to a SemanticRoleClassifier.

        Args:
            classifier: The classifier to use for SPO parsing + relation
                typing. Typically the same instance
                ``TrisynapticCircuit`` owns.
            causal_types: Which RelationTypes count as "causal" for
                anchor extraction. Defaults to CAUSAL + FUNCTIONAL +
                DIFFERENTIAL. Exposed for tests so they can assert
                the contract without hard-coding the constant.
        """
        self._classifier = classifier
        self._causal_types = causal_types

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        correction: str,
        relation: "RelationType",
    ) -> Tuple[Tuple[str, str], ...]:
        """Extract causal anchors from ``correction``.

        Args:
            correction: The correction text (e.g. "api menyebabkan panas"
                or "X tidak menyebabkan Y"). Empty/whitespace returns
                an empty tuple.
            relation: Pre-computed RelationType for the correction. The
                builder uses this value directly — it never calls
                ``classify()`` itself. This avoids double-bumping the
                classifier's frequency table: the caller has already
                called ``classify()`` once to determine the edge type
                (see :meth:`TrisynapticCircuit.encode`). Required.

        Returns:
            Tuple of ``(relation_type_name, target)`` pairs. Empty
            tuple if:
              - ``correction`` is empty
              - the SPO parse finds no object (e.g. single-token
                correction like "api")
              - the supplied relation type is not in
                ``causal_types`` (e.g. a pure CATEGORICAL "X adalah Y"
                yields no anchors — that's identity, not cause-effect)

          Examples:
              "api menyebabkan panas", relation=CAUSAL
                → (("CAUSAL", "panas"),)
              "api membutuhkan oksigen", relation=FUNCTIONAL
                → (("FUNCTIONAL", "oksigen"),)
              "api tidak menyebabkan dingin", relation=DIFFERENTIAL
                → (("DIFFERENTIAL", "dingin"),)
              "api adalah phenomenon", relation=CATEGORICAL
                → ()  # CATEGORICAL — not causal
              "api"
                → ()  # single token, no SPO object

          The ``relation_type_name`` is the upper-case enum name
          (e.g. "CAUSAL") so it matches the existing
          ``_EDGE_TYPE_TO_RELATION`` mapping in TrisynapticCircuit.
          The ``target`` is the object noun phrase from the SPO parse
          (lower-cased by the classifier's normalizer).
        """
        if not correction or not correction.strip():
            return ()

        try:
            spo = self._classifier.spo(correction)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "CausalAnchorBuilder: SPO parse failed for %r: %s",
                correction[:60], exc,
            )
            return ()

        # No object → no anchor (single-token or parse failure).
        if not spo.object:
            return ()

        # Only emit anchors for causal relation types. CATEGORICAL /
        # DISCURSIVE / TEMPORAL / SPATIAL are skipped — they don't
        # carry the cause-effect semantics an aphantasic would
        # prioritise.
        if relation not in self._causal_types:
            return ()

        return ((relation.name, spo.object),)

    @property
    def causal_types(self) -> Tuple[RelationType, ...]:
        """The relation types this builder treats as causal."""
        return self._causal_types
