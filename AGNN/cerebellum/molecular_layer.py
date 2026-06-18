"""
MOLECULAR LAYER: Spike propagation.

Biologis: Molecular layer of cerebellum propagates parallel fiber spikes.
AI: Propagate spikes between Purkinje cells across the graph.
"""

from typing import List


class MolecularLayer:
    """Spike propagation network."""

    def __init__(self):
        """Initialize propagation log."""
        # TODO: allocate propagation_log list.
        raise NotImplementedError("MolecularLayer.__init__ pending propagation log")

    def propagate(self, source_id: int, target_ids: List[int], spike: bool) -> dict:
        """
        Propagate spike from source to targets.

        Args:
            source_id: Firing neuron ID.
            target_ids: Downstream neuron IDs.
            spike: Whether source fired this step.

        Returns:
            Dict describing propagation outcome.
        """
        # TODO: deliver spike to targets if source fired.
        raise NotImplementedError("propagate() pending spike delivery")
