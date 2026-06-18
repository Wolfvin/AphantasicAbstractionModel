"""
SEMANTIC ENGRAM: Semesome (edge) - stable, neocortical.

Biologis: Semantic memory is consolidated in neocortex, stable.
AI: Abstract edge relation with weight.
"""

from dataclasses import dataclass


@dataclass
class Semesome:
    """Semantic memory unit (edge).

    Attributes:
        type: CATEGORICAL | CAUSAL | DIFFERENTIAL | FUNCTIONAL.
        weight: Edge weight in [-1, 1].
        source: Source episome text.
        target: Target episome text.
    """
    type: str
    weight: float
    source: str
    target: str
    type_memory: str = "semantic"  # marker for memory type
