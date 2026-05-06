"""RSVS Protocol definitions for type-safe Rust core access (v8.3)."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RsvsCoreProtocol(Protocol):
    """Protocol defining the interface the Rust core must satisfy.

    This enables type-safe access to the PyO3-wrapped Rust Rsvs class
    without requiring the actual class to be imported at type-check time.

    Updated for RSVS v8.3:
    - v7.0: MCTS, reflection, consolidation, thinking mode, spreading activation
    - v6.0: grounding_info, revise_sense
    - v5.0: structural_similarity, substitution_analysis, compose (with compositions)
    - Updated: ingest, query, senses, node_info, relate return richer result objects
    """

    # --- Ingest / snapshot / events ---
    def ingest(self, text: str) -> Any:
        """Ingest text; returns PyIngestStats (with compositions_induced)."""
        ...

    def ingest_with_meta_v1(self, text: str, domain_id: int | None = None) -> Any: ...
    def query(self, text: str, context: str) -> Any | None:
        """Query; returns PyQueryResult (with layer, grounding_score, compositions)."""
        ...

    def similarity(self, label_a: str, label_b: str) -> Any | None: ...
    def appraise(self, text: str) -> Any: ...
    def relate(self, text: str, target: str = "") -> Any:
        """Relate; returns PyRelateResult (with structural_relations)."""
        ...

    def node_info(self, label: str) -> Any | None:
        """Node info; returns PyNodeInfo (with layer)."""
        ...

    def atom_info(self, label: str) -> Any | None:
        """Backward compat alias for node_info."""
        ...

    def senses(self, label: str) -> list[Any]:
        """Senses; returns list of PySenseInfo (with layer, grounding_score, compositions)."""
        ...

    def snapshot_v1(self) -> str: ...
    def consume_events_v1(self, since_seq: int = 0, limit: int = 10000) -> str: ...
    def latest_seq_v1(self) -> int: ...
    def status(self) -> dict: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> 'RsvsCoreProtocol': ...
    def set_domain(self, domain: int) -> None: ...
    def set_domain_attention(self, domain_id: int, alpha: float, beta: float, gamma: float) -> None:
        """v6.3.1: Set per-domain attention weights (alpha, beta, gamma).

        Creates or updates a DomainAttentionConfig for the given domain.
        Weights are automatically normalized to sum to 1.0.
        After at least 5 observations, these override the global config.
        """
        ...
    def entity_candidates(self, top_k: int = 10) -> list[tuple[str, float]]:
        """v6.3: Return entity candidates based on learned centrality + diversity scoring."""
        ...
    def top_atoms(self, n: int) -> list[str]: ...
    def atoms(self, include_seeds: bool = False) -> list[str]: ...
    def nodes(self, include_seeds: bool = False) -> list[str]: ...
    def confidence_map(self) -> dict[str, float]: ...

    # --- v5.0: Compositional architecture ---
    def structural_similarity(self, a: str, b: str) -> Any:
        """Structural similarity between two labels.

        Returns PyStructuralSimResult with:
            sense_idx_a, sense_idx_b, structural_similarity,
            shared_compositions, only_a_compositions, only_b_compositions,
            layer_a, layer_b
        """
        ...

    def substitution_analysis(self, a: str, b: str) -> Any:
        """Substitution analysis between two labels.

        Returns PySubstitutionResult with:
            sense_idx_a, sense_idx_b, structural_similarity,
            substitutions, unpaired_only_a, unpaired_only_b
        """
        ...

    def compose(self, label: str, compositions: list[tuple[str, int]], lang: str | None = None) -> int:
        """Compose a node from (label, sense_id) pairs.

        Args:
            label: The label for the new composite node.
            compositions: List of (atom_label, sense_id) tuples specifying
                which atoms compose this node and their sense bindings.
            lang: Optional language code (e.g. 'id', 'en').

        Returns:
            The node ID of the newly created composite node.
        """
        ...

    def compose_from_ids(self, label: str, atom_ids: list[int], lang: str | None = None) -> int:
        """Backward-compatible compose using raw atom IDs (sense_id=0).

        Args:
            label: The label for the new composite node.
            atom_ids: List of atom node IDs to compose from.
            lang: Optional language code.

        Returns:
            The node ID of the newly created composite node.
        """
        ...

    # --- v6.0: Grounding and sense revision ---
    def grounding_info(self, label: str, sense_id: int) -> Any:
        """Get detailed grounding evidence for a specific sense."""
        ...

    def revise_sense(self, label: str, sense_id: int) -> bool:
        """Trigger composition revision for an ungrounded sense."""
        ...

    # --- v6.1: Context-aware query ---
    def context_query(
        self,
        concept: str,
        context_atoms: list[str],
        max_depth: int | None = None,
        gamma: float | None = None,
        halt_confidence: float | None = None,
        tau_relevance: float | None = None,
    ) -> Any | None:
        """v6.1: Context-aware query with depth-controlled lazy traversal.

        Uses P(a|S,q) scoring, cycle detection, and adaptive halting.
        """
        ...

    # --- v6.2: Context similarity ---
    def context_similarity(self, a: str, b: str, context: list[str]) -> float | None:
        """v6.2: Context-weighted similarity between two concepts."""
        ...

    # --- v6.2: Sense annotations ---
    def set_sense_label(self, node_label: str, sense_idx: int, label: str | None) -> None:
        """v6.2: Set the condition label for a specific sense."""
        ...

    # --- v6.2: Pending removals ---
    def pending_removals(self) -> list[int]:
        """v6.2: Get node IDs that require approval before removal."""
        ...

    # --- v7.0: MCTS & Reflection ---
    def mcts_query(self, label: str, simulations: int = 100, exploration: float = 1.414) -> Any | None:
        """v7.0: MCTS reasoning path exploration.

        Returns PyMCTSResult with active_sense_idx, scored_atoms,
        best_path, depth_reached, halt_reason, simulations_run.
        """
        ...

    def consolidate(self) -> Any:
        """v7.0: Run a consolidation cycle.

        Returns PyConsolidationResult with senses_merged, senses_removed,
        edges_pruned, atoms_compacted.
        """
        ...

    def verify(self) -> dict[str, int]:
        """v7.0: Verify graph integrity. Returns verification stats."""
        ...

    def run_reflection(self) -> Any:
        """v7.0: Run a reflection cycle.

        Returns PyReflectionResult with actions_total, actions_applied.
        """
        ...

    def set_thinking_mode(self, mode: str) -> None:
        """v7.0: Set thinking mode ('fast' or 'deep')."""
        ...

    # --- v7.1: Composition index ---
    def composition_index_stats(self) -> dict[str, int]:
        """v7.1: Get composition index statistics."""
        ...
