"""
EPISODIC ENGRAM: Episome (node) - labile, hippocampus-dependent.

Biologis: Episodic memory is fast-encoded in hippocampus, labile phase.
AI: Single fact node with confidence score.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Episome:
    """Episodic memory unit (node).

    Attributes:
        id: Unique node identifier.
        text: Content of the memory (surface form — the raw correction
            string as the user typed it). Phase 1 note: this is the
            *surface* layer of the aphantasic representation; the
            *amodal* layer is in ``amodal_definition``, and the
            *causal* layer is in ``causal_anchors``. Together these
            three layers mirror how aphantasics store knowledge (see
            ARCHITECTURE.md section 2 + Phase 1 design doc):
              - Layer 1 (surface):  ``text`` — "api menyebabkan panas"
              - Layer 2 (amodal):   ``amodal_definition`` —
                "fenomena pembakaran yang menghasilkan panas"
              - Layer 3 (causal):   ``causal_anchors`` —
                (("CAUSAL", "panas"), ("FUNCTIONAL", "oksigen"))
            Layer 2 + 3 are populated lazily by ``DefinitionExtractor``
            + ``CausalAnchorBuilder`` on the first ``_articulate()``
            call that touches this node, then cached on the instance
            so subsequent articulations skip the LLM call.
        confidence: Belief score in [0, 1].
        edge_type: Default edge type when binding to other episomes.
        type: Memory-type marker (always "episodic" for this class).
        amodal_definition: Phase 1 — short (≤15-word) verbal definition
            of the concept named in ``text``, generated lazily by
            ``DefinitionExtractor``. Empty string means "not yet
            generated". Once generated, the definition is cached
            forever on the instance unless the node's confidence
            increases by more than ``_DEFINITION_INVALIDATE_THRESHOLD``
            (set in ``AGNNCore``) via ``reinforce()``, in which case
            the cache is invalidated and the next ``_articulate()``
            call re-generates it. This mirrors the aphantasic
            compensatory strategy "Semantic Reliance" (PMC11910157,
            Bainbridge 2021): aphantasics do not store visual imagery,
            they store *verbal definitions* of concepts. The field is
            a plain string (not a structured object) so it can be
            serialized verbatim into the articulate prompt and into
            ``AGNNNode.metadata`` for graph persistence.
        causal_anchors: Phase 1 — tuple of ``(relation_type, target)``
            pairs extracted from the correction text by the
            ``SemanticRoleClassifier``'s SPO parser. ``relation_type``
            is the upper-case RelationType name (e.g. "CAUSAL",
            "FUNCTIONAL"); ``target`` is the object noun phrase from
            the SPO parse. Empty tuple means "no causal anchors
            extractable" (e.g. for a single-token correction like
            "api"). This mirrors the aphantasic focus on cause-effect
            relationships (Monzel 2024): aphantasics prioritise
            *causal/functional* relations over visual details when
            storing memories. The tuple is immutable so it can be
            safely shared across retrieval paths without copy risk.
            Stored as ``Tuple[Tuple[str, str], ...]`` (nested tuples)
            rather than a list so the dataclass remains hashable for
            future use in sets / dict keys.
        definition_dirty: Phase 1 — internal flag set by
            ``reinforce()`` when the cumulative confidence delta
            exceeds the invalidate threshold. When True, the next
            ``_articulate()`` call that needs this node's definition
            will re-generate it via ``DefinitionExtractor`` and reset
            the flag. Callers should not read or write this field
            directly — it is part of the lazy-cache contract.
    """
    id: int
    text: str
    confidence: float
    edge_type: str = "CATEGORICAL"
    type: str = "episodic"  # marker for memory type
    # Phase 1 (Aphantasic Node Representation) — see docstring above.
    amodal_definition: str = ""
    causal_anchors: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    definition_dirty: bool = False
