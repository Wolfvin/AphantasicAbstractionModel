"""
STRIATUM: Value-based action selection + gating.

Biologis: Striatum selects actions based on value prediction.
AI: Beam search top-k selection (width=3).

Formula: P(action) = softmax(value_prediction).
"""

from typing import Dict, List


class Striatum:
    """Value-based action selection."""

    def __init__(self, width: int = 3):
        """Initialize selection log and beam width."""
        # TODO: store width + allocate selection_log.
        raise NotImplementedError("Striatum.__init__ pending width setup")

    def select_top_k(self, activation_scores: Dict[int, float]) -> List[int]:
        """
        Beam search top-k selection.

        Args:
            activation_scores: Mapping node_id -> score.

        Returns:
            Top-k node IDs sorted by score descending.
        """
        # TODO: sort + slice top-k.
        raise NotImplementedError("select_top_k() pending top-k selection")
