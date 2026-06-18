"""
EPISODIC ENGRAM: Episome (node) - labile, hippocampus-dependent.

Biologis: Episodic memory is fast-encoded in hippocampus, labile phase.
AI: Single fact node with confidence score.
"""

from dataclasses import dataclass


@dataclass
class Episome:
    """Episodic memory unit (node).

    Attributes:
        id: Unique node identifier.
        text: Content of the memory.
        confidence: Belief score in [0, 1].
        edge_type: Default edge type when binding to other episomes.
    """
    id: int
    text: str
    confidence: float
    edge_type: str = "CATEGORICAL"
    type: str = "episodic"  # marker for memory type
