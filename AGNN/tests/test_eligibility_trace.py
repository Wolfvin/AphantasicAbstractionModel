"""
Tests for three-factor learning (R-STDP / neoHebbian eligibility traces)
in AGNNCore.reinforce() / penalize().

Validates the credit-assignment fix described in
AGNN/docs/research-spiking-neural-networks.md §1.1-1.3 + §4 + B1:
reinforce()/penalize() must distribute the ±0.1 modulatory signal
across recently-traversed edges proportionally to their eligibility
trace, instead of uniformly bumping the target node's confidence.

Covered behaviors:
  1. test_reinforce_weights_by_eligibility_trace — the Definition-of-Done
     test: edges traversed during process() receive a confidence boost;
     edges never traversed receive zero.
  2. test_edges_traversed_more_get_larger_boost — among traversed edges,
     the one with the larger trace (more visits) gets the larger delta.
  3. test_cold_start_falls_back_to_uniform_node_bump — when no traversal
     has happened, reinforce() falls back to the legacy +0.1 node bump
     (backward compatibility with the pre-three-factor behavior).
  4. test_penalize_uses_eligibility_trace — penalize() mirrors
     reinforce(): traversed edges lose confidence, never-traversed
     edges keep their confidence.
  5. test_eligibility_trace_decays_between_calls — older traversals
     fade exponentially; a freshly-traversed edge gets a larger boost
     than one traversed several calls ago.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_eligibility_trace.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
_AGGN_ROOT = Path(__file__).resolve().parent.parent
_SELF_AI_SRC = _AGGN_ROOT.parent / "self-ai" / "src"

# Insert self-ai/src FIRST (lower priority) so that the AGNN package
# (inserted next) wins on name collisions. This matters because both
# trees expose a "core" name: AGNN/core.py vs self-ai/src/core/.
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGGN_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGGN_ROOT))


# Load AGNN/core.py directly by path. This avoids the name collision
# with self-ai/src/core/ (a package) which can otherwise shadow the
# AGNN core module depending on sys.path order and pytest's rootdir.
import importlib.util as _ilu  # noqa: E402

_core_path = _AGGN_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_module_elig", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_module_elig"] = agnn_core_module  # register before exec
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore

from engrams.episodic_engram import Episome  # noqa: E402


# ----------------------------------------------------------------------
# Test graph availability — tests that need real typed edges are
# skipped if self-ai/src/agnn/graph.py is not importable (same gate
# the rest of the AGNN suite uses).
# ----------------------------------------------------------------------

def _agnn_graph_available() -> bool:
    try:
        from agnn.graph import (  # noqa: F401
            AGNNGraph,
            AGNNNode,
            NodeType,
            TypedEdge,
            RelationType,
        )
        return True
    except Exception:
        return False


def _make_core_with_chain_graph() -> AGNNCore:
    """Build an AGNNCore whose graph holds a 4-node chain.

    Topology:  n0 -> n1 -> n2 -> n3   (CATEGORICAL edges, conf=0.5)

    Each Episome is registered in the core's _episomes registry so
    reinforce(episome_id) can find it. The graph nodes' ids match the
    Episome ids (str) so Papez retrieval can find them by keyword.
    """
    from agnn.graph import (
        AGNNGraph,
        AGNNNode,
        NodeType,
        RelationType,
        TypedEdge,
    )
    from engrams.engram_complex import EngramComplex

    core = AGNNCore(use_cluster_learner=False)
    # Replace the wrapped graph with a known-shape one we control.
    ec = EngramComplex()
    ec._graph = AGNNGraph(embedding_dim=64)
    core.graph = ec

    labels = ["alpha", "beta", "gamma", "delta"]
    for i, label in enumerate(labels):
        nid = f"n{i}"
        node = AGNNNode(
            id=nid,
            label=label,
            node_type=NodeType.ENTITY,
            confidence=0.5,
        )
        ec._graph.add_node(node)
        # Register a matching Episome so reinforce(nid) resolves.
        epi = Episome(id=nid, text=label, confidence=0.5)
        # Use a string id — graph nodes use string ids.
        epi.id = nid
        core._episomes.append(epi)
    # Linear chain of edges with confidence 0.5 each — the baseline
    # we measure boosts against.
    for i in range(3):
        ec._graph.add_edge(
            TypedEdge(
                source_id=f"n{i}",
                target_id=f"n{i + 1}",
                relation_type=RelationType.CATEGORICAL,
                confidence=0.5,
            )
        )
    return core


def _edge_confidence(core: AGNNCore, src: str, tgt: str) -> float:
    """Read the live TypedEdge.confidence from the wrapped graph."""
    inner = core.graph._graph
    for edge in inner.get_edges_from(src):
        if edge.target_id == tgt:
            return float(edge.confidence)
    raise AssertionError(f"Edge {src} -> {tgt} not found in graph")


def _force_traverse_edges(core: AGNNCore, edge_pairs):
    """Make process() traverse a specific set of edges.

    We do this by directly calling _build_semesomes_from_graph with a
    hand-built list of pseudo-Episomes whose ids match the graph node
    ids — that triggers the eligibility-trace increment loop in
    _build_semesomes_from_graph for exactly the edges whose both
    endpoints are in the list.
    """
    pseudo_episomes = [
        type("E", (), {"id": nid, "text": label, "confidence": 0.5})()
        for nid, label in edge_pairs
    ]
    # Flatten to the set of node ids we want connected.
    node_ids = set()
    for nid, _ in edge_pairs:
        node_ids.add(nid)
    pseudo_episomes = [
        type("E", (), {"id": nid, "text": nid, "confidence": 0.5})()
        for nid in node_ids
    ]
    core._build_semesomes_from_graph(pseudo_episomes)


# ======================================================================
# DoD test: traversed edges get boosted, never-traversed edges don't
# ======================================================================

@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_reinforce_weights_by_eligibility_trace():
    """DoD: edge yang baru dilalui dapat confidence boost lebih besar
    dari edge yang tidak dilalui sama sekali.

    Setup: 4-node chain  n0->n1->n2->n3  with all edges at conf=0.5.
    Traverse only n0->n1 and n1->n2 (i.e. involve nodes n0, n1, n2).
    Leave n3 out of the traversal entirely so edge n2->n3 never gets
    an eligibility-trace increment. Then call reinforce(n1).

    Expected:
      - conf(n0->n1) > 0.5  (traversed, gets a positive delta)
      - conf(n1->n2) > 0.5  (traversed, gets a positive delta)
      - conf(n2->n3) == 0.5 (never traversed, gets zero delta)
    """
    core = _make_core_with_chain_graph()
    # Sanity: all edges start at 0.5.
    assert _edge_confidence(core, "n0", "n1") == pytest.approx(0.5)
    assert _edge_confidence(core, "n1", "n2") == pytest.approx(0.5)
    assert _edge_confidence(core, "n2", "n3") == pytest.approx(0.5)

    # Traverse only edges among n0, n1, n2 — leaves n2->n3 cold.
    _force_traverse_edges(core, [("n0", "alpha"), ("n1", "beta"),
                                  ("n2", "gamma")])

    # The trace should now contain exactly two edges: n0->n1, n1->n2.
    assert len(core._eligibility) == 2, (
        f"Expected 2 edges in trace, got {len(core._eligibility)}: "
        f"{list(core._eligibility.keys())}"
    )

    # Reinforce any node — the trace is global, so the +0.1 budget
    # is distributed across the traced edges regardless of episome_id.
    core.reinforce("n1")

    boost_01 = _edge_confidence(core, "n0", "n1") - 0.5
    boost_12 = _edge_confidence(core, "n1", "n2") - 0.5
    boost_23 = _edge_confidence(core, "n2", "n3") - 0.5

    assert boost_01 > 0, (
        f"Traversed edge n0->n1 must get a positive boost, got {boost_01}"
    )
    assert boost_12 > 0, (
        f"Traversed edge n1->n2 must get a positive boost, got {boost_12}"
    )
    assert boost_23 == pytest.approx(0.0, abs=1e-9), (
        f"Never-traversed edge n2->n3 must get zero boost, got {boost_23}"
    )

    # Conservation: total boost across all edges should equal the
    # +_REINFORCE_DELTA modulatory budget (0.1), within float tolerance.
    total_boost = boost_01 + boost_12 + boost_23
    assert total_boost == pytest.approx(core._REINFORCE_DELTA, abs=1e-6), (
        f"Total boost {total_boost} must equal modulatory budget "
        f"{core._REINFORCE_DELTA}"
    )


# ======================================================================
# More traversals → larger share of the modulatory budget
# ======================================================================

@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_edges_traversed_more_get_larger_boost():
    """An edge visited twice during traversal should get a larger
    confidence boost than one visited once.

    We force two traversals of edge n0->n1 and one traversal of
    n1->n2 by calling _build_semesomes_from_graph twice with the
    n0,n1 pair (so n0->n1 gets +2 to its trace) and once with the
    n1,n2 pair. Then reinforce and check n0->n1's boost > n1->n2's.
    """
    core = _make_core_with_chain_graph()

    # First traversal: n0, n1, n2 — traces n0->n1 (1.0) and n1->n2 (1.0).
    _force_traverse_edges(core, [("n0", "alpha"), ("n1", "beta"),
                                  ("n2", "gamma")])
    # Second traversal: only n0, n1 — bumps n0->n1 trace by another 1.0.
    _force_traverse_edges(core, [("n0", "alpha"), ("n1", "beta")])

    trace_01 = core._eligibility[("n0", "n1", "RelationType.CATEGORICAL")]
    trace_12 = core._eligibility[("n1", "n2", "RelationType.CATEGORICAL")]
    assert trace_01 > trace_12, (
        f"n0->n1 trace ({trace_01}) must exceed n1->n2 trace ({trace_12}) "
        f"after extra traversal"
    )

    core.reinforce("n0")

    boost_01 = _edge_confidence(core, "n0", "n1") - 0.5
    boost_12 = _edge_confidence(core, "n1", "n2") - 0.5
    assert boost_01 > boost_12, (
        f"Edge with larger trace must get larger boost: "
        f"n0->n1={boost_01}, n1->n2={boost_12}"
    )


# ======================================================================
# Cold-start fallback: no trace → legacy uniform node bump
# ======================================================================

@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_cold_start_falls_back_to_uniform_node_bump():
    """When no traversal has happened (empty trace), reinforce() must
    fall back to the pre-three-factor behavior: bump the target
    episome's confidence by +_REINFORCE_DELTA, and leave all edges
    untouched. This preserves backward compatibility for callers that
    never call process()/traverse().
    """
    core = _make_core_with_chain_graph()
    assert core._eligibility == {}, (
        "Freshly-constructed core must have an empty eligibility trace"
    )

    before = core._episomes[1].confidence
    core.reinforce("n1")
    after = core._episomes[1].confidence

    assert after == pytest.approx(before + core._REINFORCE_DELTA), (
        f"Cold-start reinforce must bump episome confidence by "
        f"{core._REINFORCE_DELTA}: before={before}, after={after}"
    )
    # Edges must be untouched.
    assert _edge_confidence(core, "n0", "n1") == pytest.approx(0.5)
    assert _edge_confidence(core, "n1", "n2") == pytest.approx(0.5)
    assert _edge_confidence(core, "n2", "n3") == pytest.approx(0.5)


# ======================================================================
# penalize() mirrors reinforce() — traversed edges lose confidence
# ======================================================================

@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_penalize_uses_eligibility_trace():
    """penalize() must distribute the -_REINFORCE_DELTA modulatory
    signal across traced edges (mirroring reinforce), while
    never-traversed edges keep their confidence unchanged.
    """
    core = _make_core_with_chain_graph()

    _force_traverse_edges(core, [("n0", "alpha"), ("n1", "beta"),
                                  ("n2", "gamma")])

    core.penalize("n1")

    delta_01 = _edge_confidence(core, "n0", "n1") - 0.5
    delta_12 = _edge_confidence(core, "n1", "n2") - 0.5
    delta_23 = _edge_confidence(core, "n2", "n3") - 0.5

    assert delta_01 < 0, (
        f"Traversed edge n0->n1 must lose confidence under penalize, "
        f"got delta {delta_01}"
    )
    assert delta_12 < 0, (
        f"Traversed edge n1->n2 must lose confidence under penalize, "
        f"got delta {delta_12}"
    )
    assert delta_23 == pytest.approx(0.0, abs=1e-9), (
        f"Never-traversed edge n2->n3 must be unchanged, got {delta_23}"
    )

    total_delta = delta_01 + delta_12 + delta_23
    assert total_delta == pytest.approx(-core._REINFORCE_DELTA, abs=1e-6), (
        f"Total delta {total_delta} must equal -modulatory_budget "
        f"{-core._REINFORCE_DELTA}"
    )


# ======================================================================
# Trace decays between calls — older traversals fade
# ======================================================================

@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_eligibility_trace_decays_between_calls():
    """_decay_eligibility() multiplies every trace by
    _ELIGIBILITY_DECAY and drops values below _ELIGIBILITY_EPSILON.

    A trace that started at 1.0 should drop to 0.9 after one decay
    call, 0.81 after two, and so on. This is the time-decay half of
    Izhikevich's distal-reward solution: older traversals contribute
    less to the next reinforce()/penalize() credit assignment.
    """
    core = _make_core_with_chain_graph()
    _force_traverse_edges(core, [("n0", "alpha"), ("n1", "beta"),
                                  ("n2", "gamma")])

    initial = core._eligibility[("n0", "n1", "RelationType.CATEGORICAL")]
    assert initial == pytest.approx(core._ELIGIBILITY_INCREMENT)

    core._decay_eligibility()
    after_one = core._eligibility[("n0", "n1", "RelationType.CATEGORICAL")]
    assert after_one == pytest.approx(
        initial * core._ELIGIBILITY_DECAY, rel=1e-9
    )

    core._decay_eligibility()
    after_two = core._eligibility[("n0", "n1", "RelationType.CATEGORICAL")]
    assert after_two == pytest.approx(
        initial * (core._ELIGIBILITY_DECAY ** 2), rel=1e-9
    ), (
        f"Trace must decay exponentially: expected "
        f"{initial * (core._ELIGIBILITY_DECAY ** 2)}, got {after_two}"
    )

    # A freshly-traversed edge (after decay) should have a higher
    # trace than an edge traversed only once and then decayed twice.
    # Traverse n2->n3 fresh and compare.
    _force_traverse_edges(core, [("n2", "gamma"), ("n3", "delta")])
    fresh_trace = core._eligibility[
        ("n2", "n3", "RelationType.CATEGORICAL")
    ]
    stale_trace = core._eligibility[
        ("n0", "n1", "RelationType.CATEGORICAL")
    ]
    assert fresh_trace > stale_trace, (
        f"Freshly-traversed edge ({fresh_trace}) must have higher "
        f"trace than decayed older edge ({stale_trace})"
    )
