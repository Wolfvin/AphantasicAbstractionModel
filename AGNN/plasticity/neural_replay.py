"""
NEURAL REPLAY: Sharp-wave ripple simulation via Spiking GNN.

Biologis: Hippocampus replays episodes during sleep -> trains neocortex.
AI: Spiking message passing (LIF neurons) -> refine embeddings.

Formula (ARCHITECTURE.md section 9 — Spiking Dynamics):
    tau * dU/dt = -(U - U_reset) + I_input    (membrane potential)
    S = Theta(U - U_th)                        (spike threshold)
    U = U_reset if S=1                         (reset after fire)

Each node in an EngramComplex is treated as a PurkinjeCell. The input
current I_input is derived from the node's topology (sum of incident
edge confidences) — exactly as ARCHITECTURE.md prescribes: "I_input
derived from topology — Spike from neighbor activation". The resulting
per-node spike train is then aggregated into a new node embedding,
mirroring the biological sharp-wave-ripple -> neocortical consolidation
loop.

CONSTRAINT: Pure numpy — no torch in this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cerebellum.purkinje_cell import PurkinjeCell


class NeuralReplay:
    """Spiking neural replay simulation.

    Drives a population of PurkinjeCells (one per graph node) for
    ``timesteps`` steps, records the resulting binary spike train per
    node, and aggregates that spike train into a fresh embedding. The
    embedding aggregation is deterministic and topology-aware: a node's
    new embedding is a function of its own spike train, so distinct
    spike patterns always yield distinct embeddings.

    Attributes:
        tau: Membrane time constant passed through to every PurkinjeCell.
        timesteps: Length of the simulated replay window. The returned
            spike_matrix has this many columns.
        threshold: Spike threshold U_th, shared by all cells.
        u_reset: Reset potential, shared by all cells. Defaults to 0.0
            to match ARCHITECTURE.md.
        embedding_dim: Dimensionality of the embeddings produced by
            ``aggregate_spikes``. Defaults to 64 (DEFAULT_EMBEDDING_DIM
            of AGNNGraph). Override per-instance if your graph uses a
            different embedding dimensionality.
    """

    def __init__(
        self,
        tau: float = 0.5,
        timesteps: int = 10,
        threshold: float = 1.0,
        embedding_dim: int = 64,
    ):
        """Initialise LIF + replay parameters."""
        if tau <= 0:
            raise ValueError(f"tau must be > 0 (got {tau!r}).")
        if timesteps <= 0:
            raise ValueError(f"timesteps must be > 0 (got {timesteps!r}).")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0 (got {embedding_dim!r}).")

        self.tau = float(tau)
        self.timesteps = int(timesteps)
        self.threshold = float(threshold)
        self.u_reset = 0.0
        self.embedding_dim = int(embedding_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inner_graph(graph: Any):
        """Resolve the underlying AGNNGraph from an EngramComplex.

        EngramComplex wraps (does not replace) AGNNGraph and exposes it
        via the ``_graph`` attribute. If the caller hands us a raw
        AGNNGraph directly, accept it as-is.
        """
        inner = getattr(graph, "_graph", None)
        return inner if inner is not None else graph

    def _topology_currents(self, inner) -> np.ndarray:
        """Per-node input current derived from graph topology.

        For each node we sum the confidences of every incident edge
        (both incoming and outgoing). This implements the
        ARCHITECTURE.md prescription "I_input derived from topology" in
        a deterministic, parameter-free way: well-connected nodes get a
        stronger injected current and therefore spike earlier and more
        often than isolated nodes.
        """
        node_ids = inner.all_node_ids()
        n = len(node_ids)
        currents = np.zeros(n, dtype=np.float64)
        for i, nid in enumerate(node_ids):
            for edge in inner.get_edges_from(nid):
                currents[i] += float(edge.confidence)
            for edge in inner.get_edges_to(nid):
                currents[i] += float(edge.confidence)
        return currents

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replay(self, graph: Any) -> np.ndarray:
        """Simulate sharp-wave ripple replay across all graph nodes.

        Args:
            graph: EngramComplex (or raw AGNNGraph) whose nodes will be
                driven as LIF neurons for ``self.timesteps`` steps.

        Returns:
            spike_matrix: ``np.ndarray`` of shape
            ``(num_nodes, timesteps)`` and dtype ``int8``. Entry
            ``[i, t]`` is 1 iff node ``i`` fired at step ``t``. An empty
            graph (num_nodes == 0) yields a ``(0, timesteps)`` array so
            callers can index safely.
        """
        inner = self._inner_graph(graph)
        node_ids = inner.all_node_ids()
        n = len(node_ids)

        # Empty-graph guard: keep shape contract intact so downstream
        # code does not need a special case.
        if n == 0:
            return np.zeros((0, self.timesteps), dtype=np.int8)

        # Topology-derived input current per node — held constant across
        # the replay window. The LIF dynamics do the integration.
        input_currents = self._topology_currents(inner)

        # One PurkinjeCell per node, all freshly at rest.
        cells = [
            PurkinjeCell(
                tau=self.tau,
                threshold=self.threshold,
                u_reset=self.u_reset,
            )
            for _ in range(n)
        ]

        spike_matrix = np.zeros((n, self.timesteps), dtype=np.int8)
        for t in range(self.timesteps):
            for i, cell in enumerate(cells):
                spiked = cell.integrate_and_fire(float(input_currents[i]))
                spike_matrix[i, t] = 1 if spiked else 0
        return spike_matrix

    def aggregate_spikes(self, node_idx: int, spike_matrix: np.ndarray) -> np.ndarray:
        """Aggregate a node's spike train into a new embedding.

        The aggregation is deterministic and topology-aware: the spike
        train is treated as a binary code, tiled to ``embedding_dim``,
        modulated by a fixed sinusoidal carrier (so distinct spike
        patterns stay distinct even when one is a cyclic shift of
        another), and L2-normalised to keep embeddings on the unit
        hypersphere regardless of how many spikes occurred.

        Args:
            node_idx: Row index of the target node in ``spike_matrix``.
            spike_matrix: Spike train matrix as returned by ``replay``.
                Shape ``(num_nodes, timesteps)``.

        Returns:
            np.ndarray of shape ``(embedding_dim,)`` and dtype
            ``float32``. A node that never spiked (all-zero row) returns
            a zero vector — callers may treat this as "no consolidation
            signal for this node".
        """
        if not isinstance(spike_matrix, np.ndarray):
            raise TypeError(
                f"spike_matrix must be a numpy.ndarray, got {type(spike_matrix).__name__}."
            )
        if spike_matrix.ndim != 2:
            raise ValueError(
                f"spike_matrix must be 2-D (num_nodes, timesteps); "
                f"got shape {spike_matrix.shape}."
            )
        num_nodes, timesteps = spike_matrix.shape
        if not (0 <= node_idx < num_nodes):
            raise IndexError(
                f"node_idx {node_idx} out of range for spike_matrix "
                f"with {num_nodes} nodes."
            )

        dim = self.embedding_dim
        spikes = spike_matrix[node_idx].astype(np.float64)

        # All-silent node -> zero embedding. Distinguishes "no signal"
        # from "weak signal" without inventing noise.
        if timesteps == 0 or spikes.sum() == 0:
            return np.zeros(dim, dtype=np.float32)

        # Tile the spike pattern to fill the embedding dimension. The
        # tile boundary deliberately does NOT align with the embedding
        # boundary, so two spike patterns that are cyclic shifts of
        # each other still produce different tiled vectors.
        repeats = (dim + timesteps - 1) // timesteps
        tiled = np.tile(spikes, repeats)[:dim]

        # Fixed (data-independent) sinusoidal carrier. This adds
        # position-dependent variation so the tiled binary vector does
        # not collapse to a tiny set of distinct values.
        pos = np.arange(dim, dtype=np.float64)
        carrier = 0.7 + 0.3 * np.cos(pos * (2.0 * np.pi / max(dim, 1)))
        embedding = tiled * carrier

        # L2-normalise so embeddings live on the unit hypersphere,
        # matching the scale of AGNNGraph's random-init embeddings
        # (which are scaled by 0.1) is NOT required — we want the new
        # embedding to dominate the residual update, so unit-norm is
        # appropriate.
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def pass_messages(self, graph: Any) -> None:
        """Update every node's embedding via spike aggregation.

        Implements the sharp-wave-ripple -> neocortex consolidation
        step: run ``replay`` once over the whole graph, then for every
        node replace its embedding with the aggregated spike embedding.
        Nodes that did not spike at all are left untouched (their
        aggregate would be a zero vector, which would erase their
        prior embedding — undesirable for isolated nodes).

        Args:
            graph: EngramComplex (or raw AGNNGraph). Mutated in place:
                each node's ``embedding`` attribute is overwritten with
                the aggregated spike embedding (where spikes occurred).
        """
        inner = self._inner_graph(graph)
        node_ids = inner.all_node_ids()
        if not node_ids:
            return

        spike_matrix = self.replay(graph)

        # Sync embedding_dim with the graph's own dimensionality so the
        # updated embeddings slot in cleanly. AGNNGraph stores this on
        # the private _embedding_dim attribute; fall back to our own
        # default if the underlying graph does not expose it.
        graph_dim = getattr(inner, "_embedding_dim", None)
        if graph_dim is not None and int(graph_dim) != self.embedding_dim:
            self.embedding_dim = int(graph_dim)

        for i, nid in enumerate(node_ids):
            new_embedding = self.aggregate_spikes(i, spike_matrix)
            # Skip nodes that produced no signal: replacing a node's
            # embedding with a zero vector would destroy its identity
            # without adding information. Leave the prior embedding
            # in place instead.
            if not np.any(new_embedding):
                continue
            node = inner.get_node(nid)
            if node is None:
                continue
            node.embedding = new_embedding.astype(np.float32)
