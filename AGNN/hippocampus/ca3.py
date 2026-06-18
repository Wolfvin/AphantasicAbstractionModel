"""
CA3: Rapid autoassociation - bind episome ke existing nodes.

Biologis: CA3 recurrent network binds co-activated representations.
AI: Add typed edges ke tetangga (CATEGORICAL, CAUSAL, etc.).

CA3 maintains a registry of every episome it has seen (id -> {text,
keywords}) and uses keyword-set overlap to find autoassociative
neighbors. The TrisynapticCircuit pipes the new episome through CA3
right after DG allocates its ID; CA3 returns the list of existing
episome IDs that share at least one keyword with the newcomer.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple


class CA3:
    """Autoassociative binding mechanism.

    Attributes:
        registry: Mapping {episome_id: {"text": str, "keywords": set[str]}}.
            Populated by :meth:`register` (called by TrisynapticCircuit).
        bindings: List of (episome_id, neighbor_ids) tuples recording every
            autoassociation decision. Useful for audit / replay.
    """

    def __init__(self) -> None:
        """Allocate the episome registry and the bindings log."""
        self.registry: Dict[int, Dict[str, object]] = {}
        self.bindings: List[Tuple[int, List[int]]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, episome_id: int, text: str, keywords: List[str]) -> None:
        """Register an episome so CA3 can find it as a neighbor later.

        Args:
            episome_id: Unique ID from DentateGyrus.
            text: Normalized stimulus text.
            keywords: Keyword list from EntorhinalCortex.
        """
        if not isinstance(episome_id, int):
            raise TypeError(
                f"episome_id must be int, got {type(episome_id).__name__}"
            )
        self.registry[episome_id] = {
            "text": text,
            "keywords": set(keywords),
        }

    # ------------------------------------------------------------------
    # Autoassociation
    # ------------------------------------------------------------------

    def autoassociate(self, episome_id: int, top_k: int = 5) -> List[int]:
        """Find neighbors of ``episome_id`` by keyword-set overlap.

        Biologis: CA3 recurrent synapses bind co-active representations.
        AI: For every other registered episome, compute the size of the
            keyword intersection; return the top-k IDs with non-zero
            overlap, sorted by descending overlap then by ID for stability.

        Args:
            episome_id: The newly-registered episome to find neighbors for.
            top_k: Maximum number of neighbors to return.

        Returns:
            List of neighbor episome IDs (may be empty).
        """
        if episome_id not in self.registry:
            # Nothing registered yet for this ID - no neighbors possible.
            return []
        cur_keywords: Set[str] = self.registry[episome_id]["keywords"]  # type: ignore[assignment]
        scored: List[Tuple[int, int]] = []
        for other_id, other in self.registry.items():
            if other_id == episome_id:
                continue
            overlap = len(cur_keywords & other["keywords"])  # type: ignore[arg-type]
            if overlap > 0:
                scored.append((other_id, overlap))
        # Sort by descending overlap, then ascending ID for deterministic ties.
        scored.sort(key=lambda x: (-x[1], x[0]))
        neighbor_ids = [oid for oid, _ in scored[:top_k]]
        self.bindings.append((episome_id, list(neighbor_ids)))
        return neighbor_ids

    # ------------------------------------------------------------------
    # Backwards-compatible alias
    # ------------------------------------------------------------------

    def bind(self, episome_id: int, neighbor_ids: List[int]) -> None:
        """Legacy entry point - persist a (episome_id, neighbor_ids) binding.

        New code should call :meth:`register` followed by
        :meth:`autoassociate`, which both compute and persist the binding.
        This method is kept so callers using the skeleton-era API continue
        to work.
        """
        if not isinstance(episome_id, int):
            raise TypeError("episome_id must be int")
        self.bindings.append((episome_id, list(neighbor_ids)))
