"""
TRISYNAPTIC CIRCUIT: EC -> DG -> CA3 -> CA1 -> Sub (encoding pathway).

Biologis: Classic hippocampal trisynaptic pathway for fast episodic encoding.
AI: Orchestrate the 5-stage encoding pipeline.

The circuit owns an :class:`EngramComplex` (which itself wraps an
``AGNNGraph``) so that every encoded Episome is also registered as a
graph node, with typed edges to its autoassociative neighbors (the
neighbors CA3 finds by keyword overlap). This lets downstream
components - :class:`SystemsConsolidation` and :class:`PapezCircuit` -
operate on the same graph the circuit populated, without the caller
having to manually wire the Episome into the graph.

Pure Python + numpy. No torch.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from engrams.episodic_engram import Episome
from hippocampus.ca1 import CA1
from hippocampus.ca3 import CA3
from hippocampus.dentate_gyrus import DentateGyrus
from hippocampus.entorhinal_cortex import EntorhinalCortex
from hippocampus.subiculum import DEFAULT_EPISODIC_CONFIDENCE, Subiculum
from neocortex.causal_anchor_builder import CausalAnchorBuilder
from neocortex.semantic_role_classifier import RelationType, SemanticRoleClassifier


# ----------------------------------------------------------------------
# Make self-ai/src/agnn importable for AGNNGraph / RelationType.
# We do this lazily so the module imports cleanly even when the
# self-ai/ tree is absent (e.g. during partial unit testing).
# ----------------------------------------------------------------------

def _resolve_agnn_graph():
    """Import AGNNGraph + RelationType from self-ai/src/agnn/graph.py.

    Adds self-ai/src to sys.path (idempotent) and returns the module.
    Raises ImportError if the module is unavailable.
    """
    self_ai_src = os.path.join(
        os.path.dirname(__file__), "..", "..", "self-ai", "src"
    )
    self_ai_src = os.path.abspath(self_ai_src)
    if self_ai_src not in sys.path:
        sys.path.insert(0, self_ai_src)
    from agnn import graph as agnn_graph  # noqa: WPS433 (lazy import)
    return agnn_graph


# Mapping from the plain-string edge types used by Episome / Semesome to
# the RelationType enum used by AGNNGraph's TypedEdge. Kept here so the
# circuit can create typed edges in the wrapped graph without exposing
# the enum to callers.
_EDGE_TYPE_TO_RELATION = {
    "CATEGORICAL": "categorical",
    "CAUSAL": "causal",
    "DIFFERENTIAL": "differential",
    "FUNCTIONAL": "functional",
    "TEMPORAL": "temporal",
    "SPATIAL": "spatial",
    "DISCURSIVE": "discursive",
}


class TrisynapticCircuit:
    """Encoding pathway orchestrator.

    Owns the five hippocampal substructures and (optionally) an
    :class:`EngramComplex` that records every encoded Episome as a
    graph node, plus typed edges to the neighbors CA3 autoassociated
    with it.

    Pipeline (per :meth:`encode`):
        1. EC.normalize_input(stimulus)
        2. DG.separate(normalized_text) -> new_id
        3. CA3.register(new_id, text, keywords) + CA3.autoassociate(new_id)
        4. CA1.integrate_context(stimulus, correction) -> edge_type
        5. Sub.relay_output(new_id, text, edge_type, confidence=0.6)
           -> Episome

    Side effect: the Episome is added as a node to the wrapped
    EngramComplex, with edges to its autoassociative neighbors.
    """

    def __init__(
        self,
        engram_complex=None,
        ec: Optional[EntorhinalCortex] = None,
        dg: Optional[DentateGyrus] = None,
        ca3: Optional[CA3] = None,
        ca1: Optional[CA1] = None,
        sub: Optional[Subiculum] = None,
        role_classifier: Optional[SemanticRoleClassifier] = None,
        classifier_persist_path: Optional[str] = None,
    ) -> None:
        """Wire up the five hippocampal substructures + engraph complex.

        Args:
            engram_complex: Optional pre-existing EngramComplex. If
                omitted, a fresh one is created. When supplied, the
                circuit populates *that* graph, so callers can share
                one graph across encode + consolidate + retrieve.
            ec, dg, ca3, ca1, sub: Optional pre-configured instances of
                each substructure. If omitted, fresh defaults are used.
            role_classifier: Optional pre-configured
                :class:`SemanticRoleClassifier`. If omitted, a fresh
                one is created. The classifier infers the RelationType
                of each encoded edge from the correction text so BA 44
                can fire the right deductive rule (CAUSAL_CHAIN,
                CATEGORICAL_TRANSITIVITY, etc.) downstream. The
                classifier is stateful (it carries a frequency table)
                so sharing one across encodes lets the system learn
                over time.
            classifier_persist_path: Optional path to a JSON file for
                persisting the role classifier's frequency table
                across process restarts. Only used when
                ``role_classifier`` is None (i.e. when we are
                constructing a fresh classifier); when the caller
                supplies their own ``role_classifier``, that instance's
                own ``persist_path`` is what governs its IO. ``None``
                (the default) means no persistence - matching the
                pre-persistence behaviour exactly.
        """
        # Lazy import so this module can be imported even when
        # self-ai/src/agnn is not on sys.path (matches the pattern in
        # engrams/engram_complex.py).
        if engram_complex is None:
            from engrams.engram_complex import EngramComplex
            engram_complex = EngramComplex()
        self.engram_complex = engram_complex

        self.ec = ec if ec is not None else EntorhinalCortex()
        self.dg = dg if dg is not None else DentateGyrus()
        self.ca3 = ca3 if ca3 is not None else CA3()
        self.ca1 = ca1 if ca1 is not None else CA1()
        self.sub = sub if sub is not None else Subiculum()
        # Phase 3: SemanticRoleClassifier replaces CA1 as the primary
        # edge-type inferrer. CA1 is still kept around for backward
        # compatibility (existing tests exercise it directly) and as a
        # fallback when the classifier returns CATEGORICAL on an
        # unknown predicate - in that case CA1's cue-word scan may
        # still pick up a more specific type from the surrounding
        # stimulus text.
        #
        # Worker 2: when the caller does not supply a role_classifier,
        # we honour classifier_persist_path by passing it down to the
        # fresh SemanticRoleClassifier. When the caller DOES supply a
        # role_classifier, that instance is used as-is - the caller is
        # responsible for its own persist_path (this avoids
        # double-wiring two paths to the same file).
        if role_classifier is not None:
            self.role_classifier = role_classifier
        elif classifier_persist_path is not None:
            self.role_classifier = SemanticRoleClassifier(
                persist_path=classifier_persist_path
            )
        else:
            self.role_classifier = SemanticRoleClassifier()

        # Phase 1 (Aphantasic Node Representation): the causal anchor
        # builder extracts (relation_type, target) pairs from the
        # correction text so each Episome carries its Layer-3 causal
        # structure alongside the surface text (Layer 1) and the
        # amodal definition (Layer 2, populated lazily by
        # DefinitionExtractor in AGNNCore). The builder borrows the
        # same role_classifier we just constructed so the SPO parse
        # + relation typing stays consistent with edge-type inference
        # above. The builder is stateless (it holds no mutable state
        # of its own), so sharing it across encode() calls is safe.
        self.causal_anchor_builder = CausalAnchorBuilder(self.role_classifier)

    # ------------------------------------------------------------------
    # Encoding pipeline
    # ------------------------------------------------------------------

    def encode(self, stimulus: str, correction: str = "") -> Episome:
        """Run ``stimulus`` through the full trisynaptic pathway.

        Pipeline: EC.normalize_input -> DG.separate -> CA3.register +
        CA3.autoassociate -> CA1.integrate_context -> Sub.relay_output.

        Side effect: the resulting Episome is added as a node to the
        wrapped EngramComplex (id = str(episome.id), label = text), and
        a typed edge is created from the new node to each autoassociative
        neighbor. Edge confidence is set to the new episome's confidence
        (0.6 by default) so downstream systems-consolidation can pick
        the "strongest" edge reliably.

        Args:
            stimulus: Input text (typically the question or wrong answer
                being corrected).
            correction: Optional correction text. CA1 uses both fields
                to infer the relation type. The Episome.text stores the
                correction when provided, else the stimulus - this is
                the new knowledge being encoded.

        Returns:
            The freshly encoded Episome (confidence = 0.6).

        .. warning::
            **``edge_type`` is snapshotted at encode time (issue #91).**
            The ``edge_type`` written onto the returned :class:`Episome`
            and onto each :class:`TypedEdge` in the graph is computed
            from :attr:`self.role_classifier` **at the moment of this
            ``encode()`` call**. If ``role_classifier`` is a
            :class:`PositionalClusterLearner` and its
            :meth:`~PositionalClusterLearner.label_clusters` is later
            called (or its state file is reloaded with different
            ``cluster_labels``), the **already-encoded edges retain
            their original ``relation_type``**. Only edges encoded
            *after* the re-labelling will reflect the new labels.

            If a predicate's cluster label changed between two
            ``encode()`` calls, the same predicate will have edges
            with **different** ``relation_type`` in the same graph.
            This mutes BA 44's transitivity rules
            (``CAUSAL_CHAIN``, ``CATEGORICAL_TRANSITIVITY``,
            ``FUNCTIONAL_COMPOSITION``) on chains that include those
            edges, because the rules require homogeneous-type chains
            to fire.

            **Safe ordering:** label clusters (or load the final
            state file) BEFORE the first ``encode()`` call. Once any
            edge exists in the graph, re-labelling requires either
            (a) accepting the mixed-type risk, (b) rebuilding the
            graph from scratch, or (c) implementing Option 2 from
            issue #91 (re-classify existing edges on PCL mutation —
            not yet implemented).

            See :meth:`PositionalClusterLearner.label_clusters` for
            the ``graph_has_existing_edges`` flag that surfaces this
            risk as a ``RuntimeWarning`` when re-labelling is
            attempted mid-session.
        """
        # 1. EC: normalize input.
        norm = self.ec.normalize_input(stimulus)
        # The "fact" we encode is the correction if one was supplied,
        # else the stimulus itself. Keywords come from the *combined*
        # text so CA3 can still find neighbors even when the correction
        # is short.
        fact_text = (correction.strip() or norm["text"])  # type: ignore[union-attr]
        # Re-normalize the fact so the stored text is canonical.
        fact_norm = self.ec.normalize_input(fact_text)

        # 2. DG: allocate a new unique ID.
        new_id = self.dg.separate(fact_norm["text"])  # type: ignore[index]

        # 3. CA3: register + autoassociate.
        # Use the union of keywords from stimulus + correction so CA3
        # can match against either surface form.
        combined_keywords = list(
            set(norm["keywords"]) | set(fact_norm["keywords"])  # type: ignore[index]
        )
        self.ca3.register(new_id, fact_norm["text"], combined_keywords)  # type: ignore[index]
        neighbor_ids = self.ca3.autoassociate(new_id)

        # 4. Infer edge type. Phase 3: prefer the SemanticRoleClassifier
        #    over CA1 - the classifier does proper SPO parsing, seed
        #    matching, negation detection, and frequency-table learning,
        #    so it can correctly classify "X tidak menyebabkan Y" as
        #    DIFFERENTIAL instead of CAUSAL (CA1's bag-of-words scan
        #    cannot tell predicate cues from object cues, so the
        #    "menyebabkan" token always fires CAUSAL regardless of the
        #    preceding negation).
        #
        #    The classifier returns a RelationType enum member; we
        #    store its ``.name`` (e.g. "CAUSAL") on the Episome so the
        #    existing _EDGE_TYPE_TO_RELATION mapping in
        #    _register_in_graph keeps working - that mapping already
        #    accepts the upper-case string form.
        #
        #    Failure contract: classify() never throws and always
        #    returns at least CATEGORICAL. CATEGORICAL can mean either
        #    of two things:
        #
        #      (a) CONFIDENT — PCL's labelled cluster for the action
        #          literally has RelationType.CATEGORICAL as its label.
        #          This is a high-confidence classification.
        #
        #      (b) FALLBACK — PCL fell through to its fallback
        #          (SemanticRoleClassifier), which returns CATEGORICAL
        #          as its default "I don't know" answer when no seed
        #          matches. This is a low-confidence classification.
        #
        #    CA1's stimulus-text scan may pick up a more specific cue
        #    ("causes" in the question rather than the correction) and
        #    we DO want to honor that signal — but ONLY in case (b),
        #    where the classifier's CATEGORICAL is a fallback. In case
        #    (a), CA1's bag-of-words scan is strictly weaker than the
        #    classifier's labelled-cluster decision, so overriding it
        #    is wrong (issue #88: CA1 was overriding PCL's correct
        #    CATEGORICAL for "X merupakan Y" sentences just because
        #    the user's English stimulus contained "causes"/"requires"
        #    /"affects").
        #
        #    Issue #88 fix: gate the CA1 override on
        #    ``role_classifier._last_classification_was_fallback``,
        #    which PCL exposes (see
        #    neocortex/positional_cluster_learner.py:classify).
        #    SemanticRoleClassifier does NOT expose this flag (it has
        #    no confident-vs-fallback distinction internally — every
        #    classification is "use the best signal I have"), so we
        #    fall back to the pre-#88 behaviour when the classifier
        #    is a plain SRC. This preserves every existing test's
        #    expectation while still letting the classifier drive the
        #    new CAUSAL / DIFFERENTIAL / TEMPORAL / DISCURSIVE cases.
        relation = self.role_classifier.classify(correction or stimulus)
        edge_type = relation.name
        # Issue #88: only let CA1 override when the classifier's
        # CATEGORICAL was a fallback (low-confidence). When the
        # classifier exposes ``_last_classification_was_fallback``,
        # check it; otherwise (SRC) preserve the pre-#88 behaviour
        # and let CA1 override as before.
        pcl_was_fallback = getattr(
            self.role_classifier, "_last_classification_was_fallback", True,
        )
        if (relation == RelationType.CATEGORICAL
                and correction.strip()
                and pcl_was_fallback):
            # Classifier fell back to default. Give CA1 one more shot
            # at the combined stimulus + correction text - its cue
            # list includes tokens ("causes", "requires") that may
            # appear in the stimulus half of the input.
            ca1_type = self.ca1.integrate_context(stimulus, correction)
            if ca1_type != "CATEGORICAL":
                edge_type = ca1_type

        # Phase 1: extract causal anchors (Layer 3 of the aphantasic
        # node representation). The builder reuses the SPO parse from
        # the same role_classifier we just used for edge-type inference,
        # so the anchors and the edge type are always consistent. The
        # builder returns an empty tuple for CATEGORICAL corrections
        # (e.g. "X adalah Y") — those carry identity, not cause-effect,
        # and aphantasics prioritise cause-effect (Monzel 2024).
        # ``correction or stimulus`` matches the classify() call above
        # so the anchors are derived from the same text the edge type
        # was inferred from.
        #
        # We pass the pre-computed ``relation`` so the builder does NOT
        # call ``classify()`` a second time — that would double-bump
        # the classifier's frequency table (one bump from the
        # edge-type classify above, one bump from the builder's own
        # classify). The pre-computed relation is the source of truth
        # here: the edge_type we just inferred IS the relation.
        # Note: when CA1 overrode the classifier's CATEGORICAL fallback
        # above (``edge_type = ca1_type``), we still pass the original
        # ``relation`` to the builder — the builder only emits anchors
        # for CAUSAL/FUNCTIONAL/DIFFERENTIAL, and CA1's override only
        # fires when the classifier returned CATEGORICAL, so passing
        # the original CATEGORICAL relation correctly suppresses
        # anchor extraction in that case.
        causal_anchors = self.causal_anchor_builder.build(
            correction or stimulus, relation=relation,
        )

        # 5. Sub: build + relay the Episome.
        episome = self.sub.relay_output(
            episome_id=new_id,
            text=fact_norm["text"],  # type: ignore[index]
            edge_type=edge_type,
            confidence=DEFAULT_EPISODIC_CONFIDENCE,
            neighbor_ids=neighbor_ids,
            causal_anchors=causal_anchors,
        )

        # 6. Side effect: register the Episome as a graph node + edges
        #    in the wrapped EngramComplex so downstream consolidation
        #    and retrieval can find it.
        self._register_in_graph(episome, neighbor_ids)

        return episome

    # ------------------------------------------------------------------
    # Graph side effects
    # ------------------------------------------------------------------

    def _register_in_graph(self, episome: Episome, neighbor_ids) -> None:
        """Add ``episome`` as a node + edges in the wrapped EngramComplex.

        - Node id = str(episome.id), label = episome.text, confidence
          = episome.confidence. Metadata carries the original int id
          and edge_type so PapezCircuit can reconstruct a faithful
          Episome from the graph later.
        - For every neighbor ID, look up the neighbor's graph node
          (by str id) and add a typed edge from the new node to it.

        Fails silently (logs to stderr) if the EngramComplex cannot be
        reached - this keeps encode() robust to environments where the
        AGNNGraph dependency is not installed.
        """
        try:
            agnn_graph = _resolve_agnn_graph()
        except ImportError:
            # No AGNNGraph available - skip graph side effects.
            return

        graph = self.engram_complex._graph  # EngraphComplex wraps AGNNGraph
        AGNNNode = agnn_graph.AGNNNode
        NodeType = agnn_graph.NodeType
        TypedEdge = agnn_graph.TypedEdge
        RelationType = agnn_graph.RelationType

        node_id = str(episome.id)
        # Add the new episome as a graph node (idempotent: if the node
        # already exists, add_node just overwrites the embedding init).
        # Phase 1: metadata now also carries the causal_anchors tuple
        # (Layer 3 of the aphantasic node representation) so downstream
        # retrieval + articulation can read it back from the graph
        # without needing to re-parse the correction text. The anchors
        # are stored as a list of [relation_type, target] pairs (JSON-
        # serializable) — the Episome field is a tuple-of-tuples, but
        # AGNNNode.metadata is a plain dict so we convert.
        node = AGNNNode(
            id=node_id,
            label=episome.text,
            node_type=NodeType.ENTITY,
            confidence=episome.confidence,
            metadata={
                "episome_id": episome.id,
                "edge_type": episome.edge_type,
                "type": episome.type,
                "causal_anchors": [
                    [rel, tgt] for rel, tgt in episome.causal_anchors
                ],
            },
        )
        if graph.get_node(node_id) is None:
            graph.add_node(node)

        # Add edges to each autoassociative neighbor.
        relation_str = _EDGE_TYPE_TO_RELATION.get(
            episome.edge_type, "categorical"
        )
        relation_type = RelationType(relation_str)
        for nid in neighbor_ids:
            target_id = str(nid)
            if graph.get_node(target_id) is None:
                continue
            # Avoid duplicate edges between the same pair.
            existing = graph.get_edges_from(node_id)
            if any(e.target_id == target_id for e in existing):
                continue
            edge = TypedEdge(
                source_id=node_id,
                target_id=target_id,
                relation_type=relation_type,
                confidence=episome.confidence,
            )
            graph.add_edge(edge)
