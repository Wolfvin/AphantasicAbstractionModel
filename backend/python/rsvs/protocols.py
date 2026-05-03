"""RSVS Protocol definitions for type-safe Rust core access (v6.0)."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RsvsCoreProtocol(Protocol):
    """Protocol defining the interface the Rust core must satisfy.

    This enables type-safe access to the PyO3-wrapped Rust Rsvs class
    without requiring the actual class to be imported at type-check time.

    Updated for RSVS v6.0:
    - New: grounding_info, revise_sense
    - v5.0: structural_similarity, substitution_analysis, compose (with compositions), compose_from_ids
    - Updated: ingest, query, senses, node_info, relate now return richer result objects
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
    def top_atoms(self, n: int) -> list[str]: ...
    def atoms(self) -> list[str]: ...
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
        """Get detailed grounding evidence for a specific sense.

        Args:
            label: The atom label to inspect.
            sense_id: The sense ID within the atom to get grounding evidence for.

        Returns:
            A grounding info object containing evidence traces, source references,
            confidence breakdown, and composition grounding status for the sense.
        """
        ...

    def revise_sense(self, label: str, sense_id: int) -> bool:
        """Trigger composition revision for an ungrounded sense.

        This requests the Rust core to re-evaluate and potentially revise
        the compositions of a sense that has been flagged as ungrounded
        (low grounding_score). The revision may adjust compositions,
        re-assign tier, or merge with another sense.

        Args:
            label: The atom label whose sense to revise.
            sense_id: The sense ID within the atom to revise.

        Returns:
            True if a revision was successfully triggered, False if the
            sense was already grounded or could not be revised.
        """
        ...

    def __repr__(self) -> str: ...
