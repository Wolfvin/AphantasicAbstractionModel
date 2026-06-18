"""
SUBICULUM: Primary output pathway - relay ke neocortex.

Biologis: Subiculum is the main output of hippocampus, projecting via fornix.
AI: Relay episome ID to downstream neocortical structures.
"""


class Subiculum:
    """Primary output relay."""

    def __init__(self):
        """Initialize output log."""
        # TODO: allocate output_log list.
        raise NotImplementedError("Subiculum.__init__ pending output log setup")

    def relay_output(self, episome_id: int) -> int:
        """
        Relay episome ID to neocortex via fornix.

        Biologis: Sub -> Fornix -> Anterior Thalamus -> PFC.
        AI: Log output, return episome_id for downstream consumption.

        Args:
            episome_id: Node to relay.

        Returns:
            Same episome_id (pass-through).
        """
        # TODO: log + forward via fornix.
        raise NotImplementedError("relay_output() pending fornix relay")
