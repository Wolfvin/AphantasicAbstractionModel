"""
ANTERIOR THALAMUS: Memory consolidation relay.

Biologis: Anterior thalamic nuclei relay hippocampal output to PFC.
AI: Relay layer between hippocampal output and neocortical reasoning.

Circuit: Sub -> Anterior Thalamus -> Cingulate -> PFC.
"""


class AnteriorThalamus:
    """Relay station for memory consolidation."""

    def __init__(self):
        """Initialize relay buffer."""
        # TODO: allocate relay_buffer list.
        raise NotImplementedError("AnteriorThalamus.__init__ pending relay buffer")

    def relay(self, signal) -> object:
        """
        Relay signal from hippocampus to neocortex.

        Args:
            signal: Incoming signal (e.g. episome ID + payload).

        Returns:
            Same signal (pass-through relay).
        """
        # TODO: log + forward signal.
        raise NotImplementedError("relay() pending forward logic")
