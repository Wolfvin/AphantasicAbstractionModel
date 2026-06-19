"""
CINGULATE GYRUS: Conflict detection + attention modulation.

Biologis: Anterior cingulate cortex (ACC) monitors for conflict.
AI: Detect contradictory edges -> resolve via weight aggregation.

Example: A->B (CAUSAL=0.7), A->B (DIFFERENTIAL=-0.8) => conflict
Resolution: (0.7 + -0.8)/2 = -0.05 (near zero = uncertain).

This module is intentionally kept compatible with
`InferiorFrontalGyrus.CAUSAL_DIFFERENTIAL_CONFLICT` so that BA 44 can
delegate conflict detection to the cingulate gyrus when needed.

Phase 2 (Cross-Node Definition Consistency Check):
    The cingulate gyrus now also detects a second class of conflict:
    when two Episomes share the same surface text (Layer 1) but carry
    divergent amodal definitions (Layer 2), the system has a
    *semantic* conflict — the user (or an adversarial input) is
    teaching two incompatible meanings for the same token. This is
    the cross-node counterpart to the adversarial-edge-poisoning test
    from PR #64: where the original ``detect_conflict`` catches
    "X causes Y" vs "X tidak menyebabkan Y" (relation-type conflict
    on the same edge), ``detect_definition_conflict`` catches
    "api = fenomena pembakaran" vs "api = application programming
    interface" (definition conflict on the same node label).

    See ``detect_definition_conflict`` below for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from engrams.semantic_engram import Semesome


# ─────────────────────────────────────────────────────────────────────
# Edge-type vocabulary - kept in sync with neocortex.inferior_frontal_gyrus.
# ─────────────────────────────────────────────────────────────────────

CAUSAL = "CAUSAL"
DIFFERENTIAL = "DIFFERENTIAL"

# A conflict is flagged when two edges share (source, target) but one is
# CAUSAL (positive weight, "X causes Y") and the other is DIFFERENTIAL
# (typically negative weight, "X is the opposite of Y" / "more X, less Y").
_CONFLICTING_TYPES = {CAUSAL, DIFFERENTIAL}


# ─────────────────────────────────────────────────────────────────────
# Phase 2 — definition-conflict defaults
# ─────────────────────────────────────────────────────────────────────

# Default Jaccard-similarity threshold below which two amodal definitions
# are considered "divergent". 0.3 means: if the two definitions share
# fewer than 30% of their (lower-cased, alpha) tokens, they describe
# different concepts and a conflict is flagged. The value is a
# compromise:
#   - Too high (e.g. 0.7) → false positives: short definitions naturally
#     have low token overlap even when they describe the same concept
#     ("fenomena pembakaran" vs "reaksi kimia eksotermik" — both valid
#     descriptions of "api" but share zero tokens).
#   - Too low (e.g. 0.05) → false negatives: "api = fenomena
#     pembakaran" vs "api = application programming interface" share
#     zero tokens, so any threshold > 0 would catch it, but a threshold
#     near 0 would also let near-duplicates ("api = fenomena
#     pembakaran" vs "api = fenomena pembakaran yang menghasilkan
#     panas") slip through.
# 0.3 is the same order as the equivalence_threshold used in the
# aam/ LANGUAGE_LINKS_DESIGN.md cross-language same_as contract (which
# uses 0.90 for full equivalence and 0.75 for candidate). We're
# stricter on the *conflict* side because we want to surface ambiguous
# cases for review, not auto-merge them.
_DEFAULT_DEFINITION_CONFLICT_THRESHOLD = 0.3


@dataclass
class Conflict:
    """Result of `CingulateGyrus.detect_conflict()`.

    Attributes:
        detected: True if the two premises conflict.
        resolution: Strategy name ("weight_aggregation" if detected,
            "none" otherwise).
        final_weight: Aggregated weight (arithmetic mean of the two premise
            weights). 0.0 if no conflict.
        premises: The two conflicting edges (or empty list if no conflict).
        note: Human-readable explanation.
    """
    detected: bool
    resolution: str
    final_weight: float
    premises: List[Semesome] = field(default_factory=list)
    note: str = ""


@dataclass
class DefinitionConflict:
    """Result of `CingulateGyrus.detect_definition_conflict()`.

    Phase 2 — the cross-node counterpart to :class:`Conflict`. Where
    ``Conflict`` flags two *edges* that share (source, target) but
    disagree on relation type, ``DefinitionConflict`` flags two
    *nodes* that share surface text but disagree on amodal definition.

    Attributes:
        detected: True if the two nodes conflict (same surface,
            divergent definitions below the similarity threshold).
        resolution: Strategy name. Currently always
            ``"surface_for_review"`` when detected — Phase 2 does NOT
            auto-merge or auto-quarantine; the caller (AGNNCore.learn)
            decides what to do (skip the new episome, mark it dirty,
            surface a warning, etc.). ``"none"`` when not detected.
        similarity: The Jaccard similarity score in [0.0, 1.0] between
            the two definitions' token sets. 0.0 when either definition
            is empty or when not detected.
        threshold: The threshold used for this check (mirrored back
            for audit — useful when the caller overrides the default).
        surface: The shared surface text (Layer 1) that triggered the
            check. Empty string when not detected.
        definition_a: The first node's amodal definition (Layer 2).
            Empty string when not detected.
        definition_b: The second node's amodal definition (Layer 2).
            Empty string when not detected.
        node_a_id: The first node's id (any type — typically int for
            Episome.id, str for graph node ids). ``None`` when not
            detected.
        node_b_id: The second node's id. ``None`` when not detected.
        note: Human-readable explanation.
    """
    detected: bool
    resolution: str
    similarity: float
    threshold: float
    surface: str
    definition_a: str
    definition_b: str
    node_a_id: Any
    node_b_id: Any
    note: str


class CingulateGyrus:
    """Conflict detection mechanism (anterior cingulate cortex analog)."""

    def __init__(self):
        """Initialize conflict counter and audit log."""
        self.conflict_count: int = 0
        # Lifetime log of all detected conflicts (for introspect / audit).
        self.conflict_log: List[Conflict] = []
        # Phase 2: separate log for definition conflicts. Kept separate
        # from ``conflict_log`` so existing callers that introspect
        # ``conflict_log`` (e.g. the adversarial test suite) are not
        # polluted with the new conflict type.
        self.definition_conflict_count: int = 0
        self.definition_conflict_log: List[DefinitionConflict] = []

    def detect_conflict(self, premise1: Semesome, premise2: Semesome) -> Conflict:
        """
        Detect conflict between two edges (same src/dst, different type).

        Biologis: ACC flags co-activation of incompatible representations.
        AI: Flag CAUSAL vs DIFFERENTIAL on same (source, target), aggregate
            weights via arithmetic mean.

        A conflict is detected when ALL of the following hold:
          1. `premise1.source == premise2.source`
          2. `premise1.target == premise2.target`
          3. `{premise1.type, premise2.type} == {CAUSAL, DIFFERENTIAL}`
             (i.e. one is CAUSAL, the other is DIFFERENTIAL, and they are
             not the same type).

        Resolution: `final_weight = (w1 + w2) / 2`.

        Args:
            premise1: First edge.
            premise2: Second edge.

        Returns:
            Conflict dataclass. If no conflict, `detected=False` and all
            other fields are zero/empty.
        """
        same_pair = (
            premise1.source == premise2.source
            and premise1.target == premise2.target
        )
        types_are_conflicting = (
            premise1.type in _CONFLICTING_TYPES
            and premise2.type in _CONFLICTING_TYPES
            and premise1.type != premise2.type
        )

        if not (same_pair and types_are_conflicting):
            return Conflict(
                detected=False,
                resolution="none",
                final_weight=0.0,
                premises=[],
                note="no conflict - premises do not match CAUSAL/DIFFERENTIAL pattern",
            )

        resolved = (premise1.weight + premise2.weight) / 2.0
        conflict = Conflict(
            detected=True,
            resolution="weight_aggregation",
            final_weight=resolved,
            premises=[premise1, premise2],
            note=(
                f"CONFLICT on {premise1.source}->{premise1.target}: "
                f"{premise1.type} {premise1.weight} vs "
                f"{premise2.type} {premise2.weight} "
                f"=> resolved weight = {resolved} "
                f"(near zero = uncertain)"
            ),
        )
        self.conflict_count += 1
        self.conflict_log.append(conflict)
        return conflict

    def scan_for_conflicts(self, edges: Sequence[Semesome]) -> List[Conflict]:
        """
        Convenience: scan a whole edge list and return all conflicts.

        Args:
            edges: Sequence of Semesome edges to scan pairwise.

        Returns:
            List of detected Conflict objects (empty if none).
        """
        conflicts: List[Conflict] = []
        seen_pairs: set = set()
        for i, e1 in enumerate(edges):
            for j, e2 in enumerate(edges):
                if i >= j:
                    continue
                key = (i, j)
                if key in seen_pairs:
                    continue
                c = self.detect_conflict(e1, e2)
                if c.detected:
                    conflicts.append(c)
                    seen_pairs.add(key)
        return conflicts

    # ------------------------------------------------------------------
    # Phase 2 — cross-node definition consistency check
    # ------------------------------------------------------------------

    def detect_definition_conflict(
        self,
        node_a: Any,
        node_b: Any,
        threshold: float = _DEFAULT_DEFINITION_CONFLICT_THRESHOLD,
    ) -> DefinitionConflict:
        """Detect conflict between two nodes' amodal definitions.

        Biologis: ACC also monitors for *conceptual* conflict — when
        the same label is bound to two incompatible meanings, the
        aphantasic verbal system flags it for review (this is the
        cognitive analog of "wait, which 'api' do you mean?").

        AI: A definition conflict is detected when ALL of the
        following hold:
          1. ``node_a.text`` and ``node_b.text`` are equal after
             normalization (lower-case, strip, collapse whitespace).
             This is the "same surface form" precondition — two
             Episomes with different texts cannot definition-conflict
             (they're just different concepts).
          2. Both nodes have non-empty ``amodal_definition``. If
             either definition is empty (e.g. the lazy
             DefinitionExtractor hasn't run yet, or the model was
             unavailable), we cannot judge divergence — return
             ``detected=False`` so the caller doesn't get a
             false-positive on a not-yet-populated node.
          3. The Jaccard similarity of the two definitions' token
             sets is strictly less than ``threshold`` (default 0.3).
             Token sets are built from lower-cased alpha tokens, so
             "Fenomena Pembakaran" and "fenomena pembakaran" match
             exactly (similarity = 1.0 → no conflict).

        Resolution strategy:
          Phase 2 does NOT auto-merge or auto-quarantine. The
          ``resolution`` field is always ``"surface_for_review"`` when
          detected — the caller (AGNNCore.learn) decides what to do
          with the conflict. This is deliberately conservative: auto-
          resolution of definition conflicts could silently drop a
          legitimate new meaning (e.g. the user genuinely wants to
          teach both "api = fire" and "api = API" as polysemy).

        Args:
            node_a: First node. Must have ``text`` and
                ``amodal_definition`` attributes (Episome does; any
                duck-typed object with those attributes works).
            node_b: Second node. Same attribute contract.
            threshold: Jaccard similarity below which the two
                definitions are considered divergent. Default 0.3
                (see ``_DEFAULT_DEFINITION_CONFLICT_THRESHOLD``
                docstring for the rationale). Callers can override
                per-call — e.g. set to 0.1 for a stricter check
                (only flag completely disjoint definitions) or 0.5
                for a looser check (flag even partial overlaps).

        Returns:
            DefinitionConflict dataclass. When not detected, all
            fields are zero/empty/None and ``note`` explains which
            precondition failed (different surface, empty definition,
            or similarity above threshold).

        Failure contract:
            Any exception (missing attribute, etc.) returns
            ``detected=False`` with a note explaining the failure.
            This keeps the conflict checker non-blocking — a broken
            node should not crash ``learn()``.
        """
        try:
            text_a = self._normalize_surface(getattr(node_a, "text", ""))
            text_b = self._normalize_surface(getattr(node_b, "text", ""))
            def_a = (getattr(node_a, "amodal_definition", "") or "").strip()
            def_b = (getattr(node_b, "amodal_definition", "") or "").strip()
            id_a = getattr(node_a, "id", None)
            id_b = getattr(node_b, "id", None)
        except Exception as exc:  # noqa: BLE001
            return DefinitionConflict(
                detected=False,
                resolution="none",
                similarity=0.0,
                threshold=threshold,
                surface="",
                definition_a="",
                definition_b="",
                node_a_id=None,
                node_b_id=None,
                note=f"definition-conflict check failed: {exc}",
            )

        # Precondition 1: same surface text (normalized).
        if not text_a or text_a != text_b:
            return DefinitionConflict(
                detected=False,
                resolution="none",
                similarity=0.0,
                threshold=threshold,
                surface=text_a,
                definition_a=def_a,
                definition_b=def_b,
                node_a_id=id_a,
                node_b_id=id_b,
                note=(
                    "no definition conflict - surfaces differ "
                    f"({text_a!r} vs {text_b!r})"
                ),
            )

        # Precondition 2: both definitions non-empty.
        if not def_a or not def_b:
            return DefinitionConflict(
                detected=False,
                resolution="none",
                similarity=0.0,
                threshold=threshold,
                surface=text_a,
                definition_a=def_a,
                definition_b=def_b,
                node_a_id=id_a,
                node_b_id=id_b,
                note=(
                    "no definition conflict - at least one amodal "
                    "definition is empty (lazy generation pending or "
                    "model unavailable)"
                ),
            )

        # Precondition 3: Jaccard similarity below threshold.
        similarity = self._jaccard(def_a, def_b)
        if similarity >= threshold:
            return DefinitionConflict(
                detected=False,
                resolution="none",
                similarity=similarity,
                threshold=threshold,
                surface=text_a,
                definition_a=def_a,
                definition_b=def_b,
                node_a_id=id_a,
                node_b_id=id_b,
                note=(
                    f"no definition conflict - similarity {similarity:.3f} "
                    f">= threshold {threshold:.3f}"
                ),
            )

        # All three preconditions met → conflict detected.
        conflict = DefinitionConflict(
            detected=True,
            resolution="surface_for_review",
            similarity=similarity,
            threshold=threshold,
            surface=text_a,
            definition_a=def_a,
            definition_b=def_b,
            node_a_id=id_a,
            node_b_id=id_b,
            note=(
                f"DEFINITION CONFLICT on {text_a!r}: "
                f"{def_a!r} vs {def_b!r} "
                f"(Jaccard {similarity:.3f} < threshold {threshold:.3f})"
            ),
        )
        self.definition_conflict_count += 1
        self.definition_conflict_log.append(conflict)
        return conflict

    def scan_for_definition_conflicts(
        self,
        nodes: Sequence[Any],
        threshold: float = _DEFAULT_DEFINITION_CONFLICT_THRESHOLD,
    ) -> List[DefinitionConflict]:
        """Convenience: scan a whole node list and return all conflicts.

        Pairwise scan — O(n²) — so callers should keep ``nodes``
        small (e.g. only the retrieved top-k from PapezCircuit, not
        the entire graph). For the typical 3–5 retrieved episomes
        this is 3–10 pairwise checks, negligible.

        Args:
            nodes: Sequence of nodes (Episomes or duck-typed objects
                with ``text`` + ``amodal_definition`` + ``id``).
            threshold: Jaccard threshold — see
                ``detect_definition_conflict``.

        Returns:
            List of detected DefinitionConflict objects (empty if none).
        """
        conflicts: List[DefinitionConflict] = []
        nodes_list = list(nodes)
        for i, a in enumerate(nodes_list):
            for j, b in enumerate(nodes_list):
                if i >= j:
                    continue
                c = self.detect_definition_conflict(a, b, threshold=threshold)
                if c.detected:
                    conflicts.append(c)
        return conflicts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_surface(text: str) -> str:
        """Lower-case + strip + collapse internal whitespace.

        We DO lower-case here (unlike DefinitionExtractor._normalize)
        because the conflict check is about *identity* — "Api" and
        "api" are the same concept for conflict purposes, even though
        they may differ for definition generation (proper-noun case
        matters there).
        """
        if not text:
            return ""
        return " ".join(text.lower().split())

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Jaccard similarity of two strings' lower-cased alpha-token sets.

        Returns 0.0 if either string has no alpha tokens (e.g. both
        are empty or only punctuation). Returns 1.0 if the two strings
        have identical token sets.
        """
        import re  # local import — keeps the module importable in
        # environments that strip ``re`` (rare, but matches the
        # defensive style of the rest of AGNN).
        tokens_a = set(re.findall(r"[a-zà-ÿ]+", a.lower()))
        tokens_b = set(re.findall(r"[a-zà-ÿ]+", b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0
