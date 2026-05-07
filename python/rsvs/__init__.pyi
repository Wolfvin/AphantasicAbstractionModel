"""Type stubs for the rsvs package.

These stubs provide static type information for the PyO3-native classes
that are compiled from Rust. When the Rust core is not built, these
types will not be available at runtime, but IDEs and mypy can still
use these stubs for type checking.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# PyO3-native data classes (from rsvs._rsvs)
# ---------------------------------------------------------------------------

class IngestStats:
    sentences_processed: int
    atoms_promoted: int
    sense_assigned: int
    sense_created: int
    confidence_updated: int
    frozen_batches: int
    compositions_induced: int
    atoms_flagged_inactive: int

    def __repr__(self) -> str: ...

class IngestMetaV1:
    api_version: str
    schema_version: str
    correlation_id: str
    seq_start: int
    seq_end: int
    sentences_processed: int
    atoms_promoted: int
    sense_assigned: int
    sense_created: int
    confidence_updated: int
    frozen_batches: int
    compositions_induced: int
    atoms_flagged_inactive: int

    def __repr__(self) -> str: ...

class QueryResult:
    sense_idx: int
    sense_n: int
    atoms: list[tuple[str, float]]
    layer: int
    grounding_score: float
    compositions: list[tuple[str, int]]
    convergence_contributors: list[tuple[str, float]]

    def __repr__(self) -> str: ...
    def top_atoms(self, k: int) -> list[str]: ...

class SimResult:
    jaccard: float
    shared: list[str]
    only_a: list[str]
    only_b: list[str]

    def __repr__(self) -> str: ...

class StructuralSimResult:
    sense_idx_a: int
    sense_idx_b: int
    structural_similarity: float
    shared_compositions: list[tuple[int, int]]
    only_a_compositions: list[tuple[int, int]]
    only_b_compositions: list[tuple[int, int]]
    layer_a: int
    layer_b: int

    def __repr__(self) -> str: ...
    def shared_labels(self, rsvs: Rsvs) -> list[tuple[str, int]]: ...

class SubstitutionResult:
    sense_idx_a: int
    sense_idx_b: int
    structural_similarity: float
    substitutions: list[tuple[int, int, int, int]]
    unpaired_only_a: list[tuple[int, int]]
    unpaired_only_b: list[tuple[int, int]]

    def __repr__(self) -> str: ...
    def substitution_labels(self, rsvs: Rsvs) -> list[tuple[str, int, str, int]]: ...

class NodeInfo:
    label: str
    surface_label: str
    id: int
    confidence: float
    tier: int
    status: str
    is_seed: bool
    is_locked: bool
    is_stable: bool
    compression_state: str
    layer: int
    atoms: list[int]
    derived_from_node_ids: list[int]
    compression_reason: Optional[str]

    def __repr__(self) -> str: ...

# Backward compat alias
AtomInfo = NodeInfo

class SenseInfo:
    sense_idx: int
    n_contexts: int
    coherence: float
    status: str
    core_atoms: list[str]
    layer: int
    grounding_score: float
    grounding_evidence: GroundingEvidence
    compositions: list[tuple[str, int]]
    condition_label: Optional[str]

    def __repr__(self) -> str: ...

class GroundingEvidence:
    confirming_contexts: int
    contradicting_contexts: int
    last_contradiction: Optional[str]
    revision_count: int

    def __repr__(self) -> str: ...
    def score(self) -> float: ...

class AppraiseResult:
    agree_pct: float
    disagree_pct: float
    verdict: str
    evidence: list[tuple[str, float]]
    convergence_info: list[tuple[str, float]]

    def __repr__(self) -> str: ...

class RelateResult:
    related_nodes: list[tuple[int, float]]
    related_edges: list[tuple[int, int, float]]
    structural_relations: list[tuple[int, float]]

    def __repr__(self) -> str: ...
    def node_labels(self, rsvs: Rsvs) -> list[tuple[str, float]]: ...
    def structural_labels(self, rsvs: Rsvs) -> list[tuple[str, float]]: ...

class MCTSResult:
    active_sense_idx: int
    total_senses: int
    scored_atoms: list[tuple[str, float]]
    depth_reached: int
    halt_reason: str
    simulations_run: int
    best_path: list[tuple[str, int]]
    layer: int
    grounding_score: float

    def __repr__(self) -> str: ...

class ConsolidationResult:
    senses_merged: int
    senses_removed: int
    edges_pruned: int
    atoms_compacted: int

    def __repr__(self) -> str: ...

class ReflectionResult:
    actions_total: int
    actions_applied: int

    def __repr__(self) -> str: ...

class ContextQueryResult:
    active_sense_idx: int
    total_senses: int
    scored_atoms: list[tuple[str, float]]
    depth_reached: int
    halt_reason: str
    cycles_detected: int
    layer: int
    grounding_score: float

    def __repr__(self) -> str: ...

class TransformerBridgeConfig:
    similarity_threshold: float
    max_compositions: int
    use_attention_weights: bool

    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Main Rsvs class (PyO3)
# ---------------------------------------------------------------------------

class Rsvs:
    """RSVS knowledge system — compositional symbolic engine with structural meaning.

    Create a new instance with optional hyperparameters:

        >>> from rsvs import Rsvs
        >>> r = Rsvs(entity_promote_n=3, theta_assign=0.12, n_warm=20, eta=0.1)
    """

    def __init__(
        self,
        entity_promote_n: int = 3,
        theta_assign: float = 0.12,
        n_warm: int = 20,
        eta: float = 0.1,
    ) -> None: ...

    # --- Core operations ---

    def ingest(self, text: str) -> IngestStats: ...
    def ingest_with_meta_v1(self, text: str, domain_id: Optional[int] = ...) -> IngestMetaV1: ...
    def query(self, concept: str, context: str) -> Optional[QueryResult]: ...
    def context_query(
        self,
        concept: str,
        context_atoms: list[str],
        max_depth: Optional[int] = ...,
        gamma: Optional[float] = ...,
        halt_confidence: Optional[float] = ...,
        tau_relevance: Optional[float] = ...,
    ) -> Optional[ContextQueryResult]: ...
    def similarity(self, a: str, b: str) -> Optional[SimResult]: ...
    def structural_similarity(self, a: str, b: str) -> Optional[StructuralSimResult]: ...
    def substitution_analysis(self, a: str, b: str) -> Optional[SubstitutionResult]: ...
    def context_similarity(self, a: str, b: str, context: list[str]) -> Optional[float]: ...
    def appraise(self, text: str) -> AppraiseResult: ...
    def relate(self, concept: str) -> Optional[RelateResult]: ...
    def compose(
        self, label: str, compositions: list[tuple[str, int]], lang: Optional[str] = ...
    ) -> int: ...
    def compose_from_ids(
        self, label: str, atom_ids: list[int], lang: Optional[str] = ...
    ) -> int: ...

    # --- Configuration ---

    def set_domain(self, domain_id: int) -> None: ...
    def set_domain_attention(self, domain_id: int, alpha: float, beta: float, gamma: float) -> None: ...
    def set_sense_label(self, node_label: str, sense_idx: int, label: Optional[str]) -> None: ...
    def set_thinking_mode(self, mode: str) -> None: ...

    # --- Inspection ---

    def node_info(self, label: str) -> NodeInfo: ...
    def atom_info(self, label: str) -> NodeInfo: ...
    def senses(self, concept: str) -> list[SenseInfo]: ...
    def nodes(self, include_seeds: bool = ...) -> list[str]: ...
    def atoms(self, include_seeds: bool = ...) -> list[str]: ...
    def confidence_map(self) -> dict[str, float]: ...
    def entity_candidates(self, top_k: int = ...) -> list[tuple[str, float]]: ...
    def pending_removals(self) -> list[int]: ...

    # --- MCTS / Reasoning ---

    def mcts_query(
        self, label: str, simulations: int = ..., exploration: float = ...
    ) -> Optional[MCTSResult]: ...

    # --- Maintenance ---

    def consolidate(self) -> ConsolidationResult: ...
    def verify(self) -> dict[str, int]: ...
    def run_reflection(self) -> ReflectionResult: ...
    def composition_index_stats(self) -> dict[str, int]: ...

    # --- Persistence ---

    def save(self, path: str) -> None: ...
    def load(path: str) -> Rsvs: ...
    def snapshot_v1(self) -> str: ...
    def consume_events_v1(self, after_seq: Optional[int] = ..., limit: int = ...) -> str: ...
    def latest_seq_v1(self) -> int: ...

    # --- Status ---

    def status(self) -> dict[str, float]: ...

# ---------------------------------------------------------------------------
# Lazy-loaded Python-side utilities
# ---------------------------------------------------------------------------

def get_rsvs_instance() -> Rsvs: ...
def run_mode(mode: str, text: str, **kwargs: object) -> dict[str, object]: ...

# ---------------------------------------------------------------------------
# Corpus data (lazy-loaded)
# ---------------------------------------------------------------------------

class _DOMAINS:
    geology: list[str]
    water: list[str]
    biology: list[str]
    physics: list[str]
    materials: list[str]
    kerajaan: list[str]
    konsep: list[str]

DOMAINS: _DOMAINS

def get_domain_text(domain: str) -> str: ...
def get_all_text() -> str: ...
def domain_names() -> list[str]: ...
