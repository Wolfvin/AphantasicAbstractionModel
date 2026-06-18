"""
Tests for PurkinjeCell (LIF neuron) and NeuralReplay (spiking message
passing), as specified in ARCHITECTURE.md section 9 — Spiking Dynamics.

Covers the 6 user-pinned behaviours plus additional edge-case coverage:
  1. PurkinjeCell fires when input_current drives U >= threshold.
  2. PurkinjeCell does NOT fire when input_current keeps U < threshold.
  3. Membrane potential decays toward U_reset under zero input (tau).
  4. NeuralReplay.replay() returns spike_matrix of shape (N, timesteps).
  5. NeuralReplay.aggregate_spikes() returns shape (embedding_dim,).
  6. NeuralReplay.pass_messages() mutates node embeddings.
  7-14. Reset-after-fire, periodic spiking under constant input,
        empty-graph handling, single isolated node, deterministic
        aggregation, distinct-pattern separation, graph-structure
        preservation, and threshold boundary (>=) semantics.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_neural_replay.py -v
"""

import math
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Make AGNN package importable when running from repo root.
# Tests are invoked as: python -m pytest AGNN/tests/ -v
# So we add the AGNN/ directory (parent of tests/) to sys.path.
_AGNN_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNN_ROOT))

# Also ensure self-ai/src is importable for AGNNGraph in engram_complex.
_SELF_AI_SRC = _AGNN_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


from cerebellum.purkinje_cell import PurkinjeCell  # noqa: E402
from plasticity.neural_replay import NeuralReplay  # noqa: E402
from engrams.episodic_engram import Episome  # noqa: E402


# ----------------------------------------------------------------------
# Optional AGNNGraph integration — tests that need a real graph are
# skipped if self-ai/src/agnn/graph.py is not importable.
# ----------------------------------------------------------------------

def _agnn_graph_available() -> bool:
    try:
        from agnn.graph import AGNNGraph, AGNNNode, NodeType, TypedEdge, RelationType  # noqa: F401
        return True
    except Exception:
        return False


def _make_connected_graph(num_nodes: int = 3, embedding_dim: int = 64):
    """Build an EngramComplex with `num_nodes` fully-connected nodes.

    Each pair (i, i+1) gets a CATEGORICAL edge with confidence 1.0, so
    every node has incident edges and therefore non-zero topology
    current under NeuralReplay's default I_input derivation.
    """
    from agnn.graph import (
        AGNNGraph,
        AGNNNode,
        NodeType,
        RelationType,
        TypedEdge,
    )
    from engrams.engram_complex import EngramComplex

    ec = EngramComplex()
    # Force a known embedding_dim on the wrapped graph so assertions
    # are stable across runs.
    ec._graph = AGNNGraph(embedding_dim=embedding_dim)
    for i in range(num_nodes):
        node = AGNNNode(
            id=f"n{i}",
            label=f"node_{i}",
            node_type=NodeType.ENTITY,
            confidence=0.8,
        )
        ec._graph.add_node(node)
    # Linear chain: n0 -> n1 -> n2 -> ... so middle nodes have higher
    # degree (and therefore stronger topology current).
    for i in range(num_nodes - 1):
        ec._graph.add_edge(
            TypedEdge(
                source_id=f"n{i}",
                target_id=f"n{i+1}",
                relation_type=RelationType.CATEGORICAL,
                confidence=1.0,
            )
        )
    return ec


# ======================================================================
# PurkinjeCell — LIF dynamics
# ======================================================================

def test_purkinje_fires_when_input_at_or_above_threshold():
    """Requirement 1: fire when input drives U >= threshold."""
    cell = PurkinjeCell(tau=0.5, threshold=1.0, u_reset=0.0)
    # I_input = 1.0 with tau=0.5, dt=1.0, decay=exp(-2)≈0.1353
    # U(1) = 0 + 0 + 1.0 * (1 - 0.1353) ≈ 0.8647  -> no spike yet
    # We need a slightly larger input (or a second step) to cross 1.0.
    # Use I_input = 1.5 to push U above threshold in a single step.
    spiked = cell.integrate_and_fire(input_current=1.5, dt=1.0)
    assert spiked is True, "PurkinjeCell should fire when input >= threshold-equivalent."
    # After firing, membrane must be at u_reset (hard reset rule).
    assert cell.u == pytest.approx(0.0), f"Membrane should reset to u_reset after fire, got {cell.u}."


def test_purkinje_does_not_fire_below_threshold():
    """Requirement 2: do NOT fire when U stays below threshold."""
    cell = PurkinjeCell(tau=0.5, threshold=1.0, u_reset=0.0)
    # I_input = 0.5 gives U(1) = 0.5 * (1 - exp(-2)) ≈ 0.432 — well under 1.0.
    spiked = cell.integrate_and_fire(input_current=0.5, dt=1.0)
    assert spiked is False, "PurkinjeCell should not fire when U < threshold."
    # And the membrane should NOT have been reset.
    assert cell.u == pytest.approx(0.5 * (1.0 - math.exp(-2.0)))


def test_purkinje_membrane_decay_with_tau():
    """Requirement 3: under zero input, U decays exponentially toward U_reset.

    With tau=0.5 and dt=1.0, decay factor is exp(-2) ≈ 0.1353.
    Starting from U = U_reset + 1.0 (manually set), after one zero-input
    step U should be U_reset + 1.0 * exp(-2). With U_reset=0: 0.1353.
    """
    cell = PurkinjeCell(tau=0.5, threshold=10.0, u_reset=0.0)  # high threshold so no fire
    cell.u = 1.0  # inject a non-rest potential
    expected = math.exp(-1.0 / 0.5)  # exp(-dt/tau) with dt=1.0
    cell.integrate_and_fire(input_current=0.0, dt=1.0)
    assert cell.u == pytest.approx(expected, rel=1e-9), (
        f"Expected decay to {expected}, got {cell.u}."
    )

    # After many steps, the membrane should approach U_reset = 0.
    for _ in range(50):
        cell.integrate_and_fire(input_current=0.0, dt=1.0)
    assert abs(cell.u) < 1e-9, f"Membrane should converge to u_reset, got {cell.u}."


def test_purkinje_reset_after_fire():
    """Extra: after firing, U is clamped to u_reset (not held over)."""
    cell = PurkinjeCell(tau=0.5, threshold=1.0, u_reset=0.0)
    cell.integrate_and_fire(input_current=2.0)  # definitely spikes
    assert cell.u == pytest.approx(0.0)
    # Next step with zero input should keep U at 0 (decay of 0 is 0).
    spiked = cell.integrate_and_fire(input_current=0.0)
    assert spiked is False
    assert cell.u == pytest.approx(0.0)


def test_purkinje_periodic_spiking_under_constant_input():
    """Extra: constant I_input above threshold produces periodic spiking."""
    # tau=0.5, threshold=1.0, u_reset=0.0. Need I_input high enough that
    # the asymptote U* = I_input exceeds threshold, AND the membrane
    # actually reaches threshold within `timesteps` steps from reset.
    # I_input = 1.5 -> U(1) = 1.5 * (1 - exp(-2)) ≈ 1.297 -> spike at t=1.
    # After reset, the cycle repeats -> spike at every step from t=1 on.
    cell = PurkinjeCell(tau=0.5, threshold=1.0, u_reset=0.0)
    spikes: List[bool] = []
    for _ in range(10):
        spikes.append(cell.integrate_and_fire(input_current=1.5, dt=1.0))
    # First step might not spike (depends on initial condition), but
    # every subsequent step should spike under this input regime.
    assert all(spikes[1:]), f"Expected periodic spiking after warmup; got {spikes}."


def test_purkinje_threshold_boundary_is_inclusive():
    """Extra: U == threshold counts as a spike (>= semantics)."""
    # Construct a cell whose membrane will land exactly on threshold.
    # Using closed-form: U(1) = I_input * (1 - exp(-dt/tau)).
    # Set this equal to threshold = 1.0 with dt=1.0, tau=0.5:
    # I_input = 1.0 / (1 - exp(-2)) ≈ 1.1565
    cell = PurkinjeCell(tau=0.5, threshold=1.0, u_reset=0.0)
    exact_input = 1.0 / (1.0 - math.exp(-2.0))
    spiked = cell.integrate_and_fire(input_current=exact_input, dt=1.0)
    assert spiked is True, "Spike must trigger when U == threshold (>= semantics)."


def test_purkinje_invalid_tau_raises():
    """Extra: non-positive tau is physically meaningless and rejected."""
    with pytest.raises(ValueError):
        PurkinjeCell(tau=0.0)
    with pytest.raises(ValueError):
        PurkinjeCell(tau=-1.0)


# ======================================================================
# NeuralReplay — topology-driven LIF over a graph
# ======================================================================

@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_replay_returns_correct_spike_matrix_shape():
    """Requirement 4: replay() -> spike_matrix shape (num_nodes, timesteps)."""
    ec = _make_connected_graph(num_nodes=3, embedding_dim=64)
    replayer = NeuralReplay(tau=0.5, timesteps=10, threshold=1.0)
    spike_matrix = replayer.replay(ec)
    assert isinstance(spike_matrix, np.ndarray)
    assert spike_matrix.shape == (3, 10), f"Expected (3, 10), got {spike_matrix.shape}."
    # Binary spike values only.
    assert set(np.unique(spike_matrix)).issubset({0, 1}), (
        f"Spike matrix must be binary; got unique values {np.unique(spike_matrix)}."
    )


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_replay_empty_graph_returns_zero_rows():
    """Extra: empty graph -> shape (0, timesteps), not (0, 0)."""
    from engrams.engram_complex import EngramComplex
    ec = EngramComplex()
    replayer = NeuralReplay(timesteps=10)
    spike_matrix = replayer.replay(ec)
    assert spike_matrix.shape == (0, 10), f"Empty graph -> (0, 10); got {spike_matrix.shape}."


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_replay_isolated_node_does_not_spike():
    """Extra: a node with no edges has zero topology current -> no spikes."""
    ec = _make_connected_graph(num_nodes=1, embedding_dim=64)  # 1 node, no edges
    replayer = NeuralReplay(tau=0.5, timesteps=10, threshold=1.0)
    spike_matrix = replayer.replay(ec)
    assert spike_matrix.shape == (1, 10)
    assert spike_matrix.sum() == 0, "Isolated node has I_input=0, should never spike."


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_replay_connected_nodes_produce_spikes():
    """Extra: connected nodes get non-zero topology current -> some spikes occur."""
    # 3 nodes in a chain: middle node has 2 incident edges with conf=1.0,
    # so I_input = 2.0 > threshold = 1.0 -> guaranteed to spike.
    ec = _make_connected_graph(num_nodes=3, embedding_dim=64)
    replayer = NeuralReplay(tau=0.5, timesteps=10, threshold=1.0)
    spike_matrix = replayer.replay(ec)
    # Middle node (index 1) MUST have spiked at least once.
    assert spike_matrix[1].sum() > 0, "Middle node with 2 edges should spike."
    # End nodes (1 edge each, I_input=1.0) may or may not spike depending
    # on dynamics — no assertion on them here.


# ======================================================================
# NeuralReplay.aggregate_spikes
# ======================================================================

def test_aggregate_spikes_returns_correct_shape():
    """Requirement 5: aggregate_spikes -> shape (embedding_dim,)."""
    replayer = NeuralReplay(tau=0.5, timesteps=10, threshold=1.0, embedding_dim=64)
    # Fabricate a spike_matrix with at least one spike so the aggregator
    # does not short-circuit to a zero vector.
    spike_matrix = np.zeros((3, 10), dtype=np.int8)
    spike_matrix[0, 0] = 1
    spike_matrix[0, 3] = 1
    embedding = replayer.aggregate_spikes(0, spike_matrix)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (64,), f"Expected (64,), got {embedding.shape}."
    assert embedding.dtype == np.float32


def test_aggregate_spikes_deterministic():
    """Extra: same spike pattern -> identical embedding (no RNG)."""
    replayer = NeuralReplay(embedding_dim=64)
    spike_matrix = np.array(
        [[1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=np.int8,
    )
    e1 = replayer.aggregate_spikes(0, spike_matrix)
    e2 = replayer.aggregate_spikes(0, spike_matrix)
    assert np.array_equal(e1, e2), "aggregate_spikes must be deterministic."


def test_aggregate_spikes_distinct_patterns_separate():
    """Extra: different spike patterns -> different embeddings."""
    replayer = NeuralReplay(embedding_dim=64)
    spike_matrix = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pattern A
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # pattern B (different position)
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # all-silent
        ],
        dtype=np.int8,
    )
    e_a = replayer.aggregate_spikes(0, spike_matrix)
    e_b = replayer.aggregate_spikes(1, spike_matrix)
    e_silent = replayer.aggregate_spikes(2, spike_matrix)
    assert not np.array_equal(e_a, e_b), "Distinct spike patterns must yield distinct embeddings."
    # Silent node -> zero vector (no signal -> no fabricated noise).
    assert np.all(e_silent == 0), "Silent node should produce a zero embedding."


def test_aggregate_spikes_normalised_unit_norm():
    """Extra: a non-silent embedding has unit L2 norm (consistent scale)."""
    replayer = NeuralReplay(embedding_dim=32)
    spike_matrix = np.zeros((1, 8), dtype=np.int8)
    spike_matrix[0, 0] = 1
    spike_matrix[0, 4] = 1
    e = replayer.aggregate_spikes(0, spike_matrix)
    assert np.linalg.norm(e) == pytest.approx(1.0, rel=1e-6), (
        f"Non-silent embedding should have unit L2 norm; got {np.linalg.norm(e)}."
    )


def test_aggregate_spikes_out_of_range_raises():
    """Extra: node_idx bounds are enforced."""
    replayer = NeuralReplay(embedding_dim=16)
    spike_matrix = np.zeros((2, 5), dtype=np.int8)
    with pytest.raises(IndexError):
        replayer.aggregate_spikes(-1, spike_matrix)
    with pytest.raises(IndexError):
        replayer.aggregate_spikes(2, spike_matrix)


# ======================================================================
# NeuralReplay.pass_messages
# ======================================================================

@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_pass_messages_changes_node_embeddings():
    """Requirement 6: pass_messages mutates node embeddings in place."""
    ec = _make_connected_graph(num_nodes=4, embedding_dim=64)
    replayer = NeuralReplay(tau=0.5, timesteps=10, threshold=1.0)

    # Snapshot original embeddings.
    original_embeddings = {
        nid: ec._graph.get_node(nid).embedding.copy()
        for nid in ec._graph.all_node_ids()
    }

    replayer.pass_messages(ec)

    changed_count = 0
    for nid, original in original_embeddings.items():
        current = ec._graph.get_node(nid).embedding
        # Shape must be preserved.
        assert current.shape == original.shape == (64,), (
            f"Node {nid}: embedding shape changed from {original.shape} to {current.shape}."
        )
        if not np.allclose(current, original, atol=1e-6):
            changed_count += 1

    # At least one node (a spiking one) must have a different embedding.
    assert changed_count > 0, (
        "pass_messages must mutate at least one node's embedding (the spiking ones)."
    )


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_pass_messages_preserves_graph_structure():
    """Extra: pass_messages updates embeddings only — it does not add/remove nodes or edges."""
    ec = _make_connected_graph(num_nodes=3, embedding_dim=64)
    original_node_count = ec._graph.node_count()
    original_edge_count = ec._graph.edge_count()
    original_node_ids = set(ec._graph.all_node_ids())

    replayer = NeuralReplay(tau=0.5, timesteps=10, threshold=1.0)
    replayer.pass_messages(ec)

    assert ec._graph.node_count() == original_node_count
    assert ec._graph.edge_count() == original_edge_count
    assert set(ec._graph.all_node_ids()) == original_node_ids, (
        "Node IDs must not change during pass_messages."
    )


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_pass_messages_empty_graph_is_noop():
    """Extra: pass_messages on an empty graph does not raise."""
    from engrams.engram_complex import EngramComplex
    ec = EngramComplex()
    replayer = NeuralReplay(tau=0.5, timesteps=10, threshold=1.0)
    # Should be a no-op without raising.
    replayer.pass_messages(ec)
    assert ec._graph.node_count() == 0


def test_episome_dataclass_used_as_node_payload():
    """Extra: Episome type from engrams/episodic_engram.py is importable and well-formed."""
    e = Episome(id=0, text="hello", confidence=0.9)
    assert e.id == 0
    assert e.text == "hello"
    assert e.confidence == 0.9
    assert e.type == "episodic"
    assert e.edge_type == "CATEGORICAL"  # default


def test_neural_replay_invalid_params_raise():
    """Extra: constructor validates LIF/replay parameters."""
    with pytest.raises(ValueError):
        NeuralReplay(tau=0.0)
    with pytest.raises(ValueError):
        NeuralReplay(timesteps=0)
    with pytest.raises(ValueError):
        NeuralReplay(embedding_dim=0)


def test_no_torch_imported_in_either_module():
    """Constraint check: neither module imports torch."""
    import importlib
    import sys

    # Ensure neither module nor its imports pulled in torch.
    cerebellum_mod = importlib.import_module("cerebellum.purkinje_cell")
    replay_mod = importlib.import_module("plasticity.neural_replay")
    assert "torch" not in sys.modules or False, (
        "torch should not be imported by PurkinjeCell or NeuralReplay."
    )
    # Sanity: modules loaded successfully and expose the right symbols.
    assert hasattr(cerebellum_mod, "PurkinjeCell")
    assert hasattr(replay_mod, "NeuralReplay")
