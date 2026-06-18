"""
CINGULATE GYRUS: Conflict detection + attention modulation.

Biologis: Anterior cingulate cortex (ACC) monitors for conflict.
AI: Detect contradictory edges -> resolve via weight aggregation.

Example: A->B (CAUSAL=0.7), A->B (DIFFERENTIAL=-0.8) => conflict
Resolution: (0.7 + -0.8)/2 = -0.05 (near zero = uncertain).

This module is intentionally kept compatible with
`InferiorFrontalGyrus.CAUSAL_DIFFERENTIAL_CONFLICT` so that BA 44 can
delegate conflict detection to the cingulate gyrus when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

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


class CingulateGyrus:
    """Conflict detection mechanism (anterior cingulate cortex analog)."""

    def __init__(self):
        """Initialize conflict counter and audit log."""
        self.conflict_count: int = 0
        # Lifetime log of all detected conflicts (for introspect / audit).
        self.conflict_log: List[Conflict] = []

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
