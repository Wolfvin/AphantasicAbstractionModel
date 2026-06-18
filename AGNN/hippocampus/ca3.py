"""
CA3: Rapid autoassociation - bind episome ke existing nodes.

Biologis: CA3 recurrent network binds co-activated representations.
AI: Add typed edges ke tetangga (CATEGORICAL, CAUSAL, etc.).
"""

from typing import List


class CA3:
    """Autoassociative binding mechanism."""

    def __init__(self):
        """Initialize bindings store."""
        # TODO: allocate bindings list/dict.
        raise NotImplementedError("CA3.__init__ pending bindings store setup")

    def bind(self, episome_id: int, neighbor_ids: List[int]) -> None:
        """
        Autoassociative binding: connect new episome to existing nodes.

        Biologis: CA3 recurrent synapses bind co-active representations.
        AI: Store binding tuple.

        Args:
            episome_id: New node ID.
            neighbor_ids: Existing node IDs to bind.
        """
        # TODO: persist (episome_id, neighbor_ids) binding.
        raise NotImplementedError("bind() pending binding persistence")
