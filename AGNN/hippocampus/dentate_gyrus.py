"""
DENTATE GYRUS: Pattern separation - sparse coding ke new episomes.

Biologis: DG sparse-codes similar inputs into distinct representations.
AI: Create new node with unique ID, avoid collision.

The DG uses a monotonically increasing integer counter as the source of
new episome IDs. This guarantees that even identical stimuli receive
distinct IDs (pattern separation), forcing downstream CA3 to decide
whether to autoassociate them based on content overlap rather than
identity.
"""

from __future__ import annotations

from typing import Dict


class DentateGyrus:
    """Pattern separation mechanism - creates distinct episome IDs.

    Attributes:
        counter: Monotonic integer counter - source of all new IDs.
        sparse_cache: Mapping {episome_id: stimulus_text} used as the
            sparse-code cache. CA3 reads this to look up a stimulus by ID.
    """

    def __init__(self, start_id: int = 0) -> None:
        """Initialize the counter and the sparse-code cache.

        Args:
            start_id: Initial value of the counter. Defaults to 0 so the
                first allocated ID is 1 (1-indexed, matching the convention
                used in existing tests like ``Episome(id=1, ...)``).
        """
        if start_id < 0:
            raise ValueError(f"start_id must be >= 0, got {start_id}")
        self.counter: int = int(start_id)
        # sparse_cache maps episome_id -> stimulus text. The biological
        # analog is the DG's sparse distributed code; here we keep it
        # simple - one canonical stimulus string per ID.
        self.sparse_cache: Dict[int, str] = {}

    def separate(self, stimulus: str) -> int:
        """Pattern separation: create a new episome ID.

        Biologis: DG creates sparse, non-overlapping representations.
        AI: Increment counter, cache the stimulus, return the new ID.

        Args:
            stimulus: Input text (already normalized by EntorhinalCortex).

        Returns:
            episome_id: Unique node identifier.
        """
        if not isinstance(stimulus, str):
            raise TypeError(
                f"DentateGyrus.separate expects str, got {type(stimulus).__name__}"
            )
        self.counter += 1
        new_id = self.counter
        self.sparse_cache[new_id] = stimulus
        return new_id
