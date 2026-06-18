"""
GLOBUS PALLIDUS: Output gating.

Biologis: GP gates selected actions, inhibiting non-selected.
AI: Filter and gate outputs from striatum selection.
"""

from typing import List


class GlobusPallidus:
    """Output gating."""

    def __init__(self):
        """Initialize gate log."""
        # TODO: allocate gate_log list.
        raise NotImplementedError("GlobusPallidus.__init__ pending gate log")

    def gate(self, candidates: List[int], selected: List[int]) -> List[int]:
        """
        Pass through only selected candidates.

        Args:
            candidates: All candidate node IDs.
            selected: Node IDs chosen by striatum.

        Returns:
            Gated candidate list (intersection).
        """
        # TODO: filter candidates by selected set.
        raise NotImplementedError("gate() pending filtering logic")
