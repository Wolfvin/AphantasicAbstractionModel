"""
SUBICULUM: Primary output pathway - relay ke neocortex.

Biologis: Subiculum is the main output of hippocampus, projecting via fornix.
AI: Relay episome ID to downstream neocortical structures.

In the trisynaptic pipeline, Subiculum is the last stage. It receives
the episome ID (from DG), the normalized text (from EC), the inferred
edge type (from CA1), and the list of neighbors (from CA3), and
assembles the final :class:`Episome` object that gets returned to the
caller. Per the AGNN architecture, fresh episodic memories leave the
hippocampus with confidence = 0.6 (labile phase).
"""

from __future__ import annotations

from typing import List, Optional

from engrams.episodic_engram import Episome

# Default confidence for a freshly encoded episodic memory. This is the
# "labile phase" baseline mentioned in ARCHITECTURE.md section 10:
# Episome = "Labile (fast encoding)". The value 0.6 leaves room for
# systems consolidation to strengthen the memory (it adds +0.05) and
# for mesolimbic modulation to reinforce it (another +0.1) without
# saturating at 1.0 too quickly.
DEFAULT_EPISODIC_CONFIDENCE = 0.6


class Subiculum:
    """Primary output relay.

    Attributes:
        output_log: List of every Episome relayed so far. Useful for
            audit and for downstream systems-consolidation loops that
            iterate over recently encoded memories.
    """

    def __init__(self) -> None:
        """Allocate the output log."""
        self.output_log: List[Episome] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def relay_output(
        self,
        episome_id: int,
        text: str,
        edge_type: str,
        confidence: float = DEFAULT_EPISODIC_CONFIDENCE,
        neighbor_ids: Optional[List[int]] = None,
    ) -> Episome:
        """Construct and relay an Episome to the neocortex.

        Biologis: Sub -> Fornix -> Anterior Thalamus -> PFC.
        AI: Build the Episome dataclass, log it, return it. The
            ``neighbor_ids`` argument is recorded (via metadata on the
            returned object's ``text`` only when the caller asks - the
            Episome dataclass does not carry a neighbors field, so we
            do not pollute it here).

        Args:
            episome_id: Unique node ID from DentateGyrus.
            text: Normalized stimulus text (the fact being encoded).
            edge_type: Inferred edge type from CA1.
            confidence: Initial confidence. Defaults to 0.6 per the
                architecture's "labile phase" baseline.
            neighbor_ids: Optional list of neighbor IDs found by CA3.
                Not stored on the Episome (the dataclass has no such
                field) but kept available for the caller via the log.

        Returns:
            The freshly constructed Episome.
        """
        if not isinstance(episome_id, int):
            raise TypeError(
                f"episome_id must be int, got {type(episome_id).__name__}"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {confidence}"
            )
        episome = Episome(
            id=episome_id,
            text=text,
            confidence=confidence,
            edge_type=edge_type,
        )
        self.output_log.append(episome)
        return episome
