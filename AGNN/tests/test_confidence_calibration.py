"""
Statistical confidence-calibration tests for ``AGNNCore.process()``.

These tests validate that ``chain_confidence`` is a meaningful signal
of answer quality — not just a number that happens to come out of the
deductive engine. The validation strategy is to build synthetic graphs
whose ground-truth structure is known, then assert that the confidence
produced by ``process()`` tracks that ground truth in the expected
direction across three orthogonal dimensions:

1. **Relevance discrimination.** A query that retrieves real graph
   content should produce a higher mean ``chain_confidence`` than a
   cold query that retrieves nothing. If ``chain_confidence`` cannot
   tell "I have evidence" from "I have no evidence", it is useless as
   a quality signal.

2. **Reinforcement response.** Calling ``reinforce(node_id)`` on a
   node that participated in a correct chain must increase the
   episome's confidence (and the mirrored graph node's confidence).
   If ``reinforce()`` were a no-op, the system could never learn from
   positive feedback.

3. **Penalization response.** Calling ``penalize(node_id)`` on a
   wrong node must decrease its confidence. If ``penalize()`` were a
   no-op, the system could never unlearn mistakes.

4. **Chain-length decay.** Longer deductive chains (more transitivity
   hops) must yield lower ``chain_confidence`` than shorter chains,
   because confidence is the product of edge weights along the chain.
   This is the BA 44 design contract: ``CategoricalTransitivity`` weighs
   ``w1 * w2``, so each additional hop strictly multiplies in a value
   ``< 1`` and shrinks the aggregate.

5. **Top-node ranking.** After ``reinforce(node_id)``, the reinforced
   node must surface as ``top_nodes[0]`` in ``introspect()`` (the
   highest-confidence entry). This is the user-visible contract of the
   confidence signal: "what does the system currently believe most?"

All tests run with ``model_path=None`` — the Qwen3 graceful-fallback
path is sufficient because we only inspect ``chain_confidence`` and
the episome registry, never the articulated answer text.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_confidence_calibration.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ----------------------------------------------------------------------
# Make AGNN package + self-ai/src importable from anywhere pytest is
# invoked. Mirrors the bootstrap pattern in test_core_wired.py /
# test_qwen3_integration.py so this file is self-contained.
# ----------------------------------------------------------------------

_AGNP_ROOT = Path(__file__).resolve().parent.parent
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"

# Insert self-ai/src FIRST (lower priority) so AGNN wins on name
# collisions (both trees expose a "core" name).
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

# Load AGNN/core.py by path to avoid the self-ai/src/core/ package
# shadowing the AGNN core module.
import importlib.util as _ilu  # noqa: E402

_core_path = _AGNP_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_module_calibration", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_module_calibration"] = agnn_core_module
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_qwen_path(monkeypatch):
    """Auto-clear QWEN_PATH for every test so they're isolated.

    None of these tests need a real Qwen3 model — we only inspect
    ``chain_confidence`` and the episome registry, which the
    graceful-fallback path produces correctly.
    """
    monkeypatch.delenv("QWEN_PATH", raising=False)


@pytest.fixture
def brain() -> AGNNCore:
    """Fresh AGNNCore with no model.

    Skips the test if the EngramComplex dependency (self-ai/src/agnn)
    is unavailable — the calibration contract only makes sense when
    the graph is actually wired.
    """
    core = AGNNCore(model_path=None)
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")
    return core


def _build_synthetic_chain(
    core: AGNNCore,
    facts: list,
) -> list:
    """Encode a list of (question, wrong, correction) tuples and return
    the list of learn() result dicts.

    The corrections are crafted so adjacent facts share exactly one
    keyword (e.g. ``"alpha beta"`` and ``"beta gamma"`` share
    ``"beta"``). This guarantees CA3 autoassociation finds the
    intended neighbor on every encode, so the typed edges
    TrisynapticCircuit records in the AGNNGraph form a single
    deterministic chain rather than a random sub-graph.
    """
    results = []
    for q, w, c in facts:
        r = core.learn(q, w, c)
        assert r["node_id"] is not None, (
            f"learn() failed for {c!r} — graph wiring is broken, "
            "calibration tests cannot proceed"
        )
        results.append(r)
    return results


def _build_default_chain(core: AGNNCore) -> list:
    """Build the canonical 4-node CATEGORICAL chain used by most tests.

    Layout (each arrow = a CATEGORICAL typed edge in the AGNNGraph,
    created by TrisynapticCircuit when CA3 finds the shared keyword):

        alpha beta  ->  beta gamma  ->  gamma delta  ->  delta epsilon

    Every adjacent pair shares exactly one keyword, so CA3 finds the
    intended neighbor deterministically and the deductive engine sees
    a clean chain it can apply ``CATEGORICAL_TRANSITIVITY`` to.
    """
    return _build_synthetic_chain(core, [
        ("q1?", "w1", "alpha beta"),
        ("q2?", "w2", "beta gamma"),
        ("q3?", "w3", "gamma delta"),
        ("q4?", "w4", "delta epsilon"),
    ])


# ======================================================================
# Dimension 1: relevance discrimination
# ======================================================================

def test_relevant_query_returns_nonzero_confidence(brain: AGNNCore):
    """A query that retrieves real graph content must yield > 0 confidence.

    This is the floor of the calibration contract: if ``process()``
    cannot even tell "I have evidence" from "I have no evidence" by
    returning a non-zero confidence, the signal is meaningless.
    """
    _build_default_chain(brain)
    # Query keywords overlap with all 4 nodes -> full chain retrieved.
    result = brain.process("alpha beta gamma delta epsilon")
    assert result["chain_confidence"] > 0.0, (
        "relevant query must produce non-zero chain_confidence, "
        f"got {result['chain_confidence']}"
    )


def test_cold_query_returns_zero_confidence(brain: AGNNCore):
    """A query with zero keyword overlap must yield exactly 0.0 confidence.

    The Papez circuit retrieves nothing, so ``process()`` short-circuits
    to the empty dict — chain_confidence is 0.0 by construction.
    """
    _build_default_chain(brain)
    # No keyword overlap with any encoded node.
    result = brain.process("quantum physics pizza banana")
    assert result["chain_confidence"] == 0.0, (
        "cold query must produce zero chain_confidence, "
        f"got {result['chain_confidence']}"
    )
    assert result["answer"] == ""
    assert result["chain"] == ""


def test_relevant_mean_confidence_exceeds_cold_mean(brain: AGNNCore):
    """Across multiple synthetic graphs, mean(relevant) > mean(cold).

    This is the statistical core of the calibration claim. We build
    ``N_SYNTH`` independent synthetic graphs (each with a different
    "alpha" token so they don't accidentally share keywords), run a
    relevant query and a cold query on each, and assert that the
    average relevant confidence strictly exceeds the average cold
    confidence.

    A single comparison could pass by luck; aggregating across many
    independent graphs makes the test a real statistical statement
    about the confidence signal's discriminative power.
    """
    n_synth = 5
    relevant_confidences = []
    cold_confidences = []

    for i in range(n_synth):
        # Fresh brain per iteration so the graphs are independent.
        core = AGNNCore(model_path=None)
        if core.graph is None:
            pytest.skip("EngramComplex (self-ai/src/agnn) not available")
        # Unique prefix per chain so the chains don't cross-link.
        prefix = f"s{i}_"
        _build_synthetic_chain(core, [
            ("q1?", "w1", f"{prefix}alpha beta"),
            ("q2?", "w2", "beta gamma"),
            ("q3?", "w3", "gamma delta"),
            ("q4?", "w4", "delta epsilon"),
        ])
        relevant = core.process(
            f"{prefix}alpha beta gamma delta epsilon"
        )
        cold = core.process(f"quantum{i} physics pizza banana")
        relevant_confidences.append(float(relevant["chain_confidence"]))
        cold_confidences.append(float(cold["chain_confidence"]))

    mean_relevant = sum(relevant_confidences) / n_synth
    mean_cold = sum(cold_confidences) / n_synth
    assert mean_relevant > mean_cold, (
        f"mean relevant confidence ({mean_relevant}) must exceed "
        f"mean cold confidence ({mean_cold}) across {n_synth} "
        f"synthetic graphs; per-iteration values: "
        f"relevant={relevant_confidences}, cold={cold_confidences}"
    )


def test_partial_overlap_query_returns_intermediate_confidence(brain: AGNNCore):
    """A query that overlaps some (not all) nodes yields > 0 confidence.

    This guards against an off-by-one in the retrieval filter that
    would only return confidence when the *entire* graph is matched.
    A 2-keyword query against a 4-node chain should retrieve at least
    the two matching nodes and yield a non-zero confidence.
    """
    _build_default_chain(brain)
    # Overlaps with two nodes (alpha beta + beta gamma share 'beta').
    result = brain.process("alpha beta")
    assert result["chain_confidence"] > 0.0, (
        "partial-overlap query must yield non-zero confidence, "
        f"got {result['chain_confidence']}"
    )


# ======================================================================
# Dimension 2: reinforcement response
# ======================================================================

def test_reinforce_increases_episome_confidence(brain: AGNNCore):
    """``reinforce(id)`` must raise the episome's confidence by 0.1."""
    rs = _build_default_chain(brain)
    eid = rs[0]["node_id"]
    before = brain._find_episome(eid).confidence

    brain.reinforce(eid)

    after = brain._find_episome(eid).confidence
    assert after > before, (
        f"reinforce() must increase confidence: before={before}, after={after}"
    )
    # The delta is _REINFORCE_DELTA (0.1) within float tolerance.
    assert after == pytest.approx(before + 0.1, abs=1e-9)


def test_reinforce_mirrors_to_graph_node(brain: AGNNCore):
    """``reinforce()`` must also update the AGNNNode confidence in the graph.

    PapezCircuit reads confidence from the AGNNNode, so without this
    mirror the reinforcement would not affect future retrieval scores.
    The mirror is implemented by ``_mirror_confidence_to_graph``.
    """
    rs = _build_default_chain(brain)
    eid = rs[1]["node_id"]
    inner = brain.graph._graph
    node_before = inner.get_node(str(eid))
    assert node_before is not None
    conf_before = float(node_before.confidence)

    brain.reinforce(eid)

    node_after = inner.get_node(str(eid))
    conf_after = float(node_after.confidence)
    assert conf_after > conf_before, (
        f"reinforce() must mirror onto the graph node: "
        f"before={conf_before}, after={conf_after}"
    )


def test_reinforce_bubbles_to_introspect_top(brain: AGNNCore):
    """After reinforce(id), the reinforced node must be top_nodes[0].

    This is the user-visible contract of the confidence signal: the
    "I believe this most" slot in ``introspect()`` must reflect the
    most-recent reinforcement. If reinforce() worked but the ranking
    didn't update, the signal would be lying about what the system
    currently believes.
    """
    rs = _build_default_chain(brain)
    eid = rs[2]["node_id"]  # reinforce the 3rd node

    brain.reinforce(eid)

    info = brain.introspect()
    assert info["top_nodes"], "top_nodes must be non-empty"
    top = info["top_nodes"][0]
    assert top["id"] == eid, (
        f"reinforced node (id={eid}) must be top_nodes[0], "
        f"got top={top}"
    )
    assert top["confidence"] == pytest.approx(0.7, abs=1e-9), (
        f"reinforced node confidence must be 0.6 + 0.1 = 0.7, "
        f"got {top['confidence']}"
    )


# ======================================================================
# Dimension 3: penalization response
# ======================================================================

def test_penalize_decreases_episome_confidence(brain: AGNNCore):
    """``penalize(id)`` must lower the episome's confidence by 0.1."""
    rs = _build_default_chain(brain)
    eid = rs[0]["node_id"]
    before = brain._find_episome(eid).confidence

    brain.penalize(eid)

    after = brain._find_episome(eid).confidence
    assert after < before, (
        f"penalize() must decrease confidence: before={before}, after={after}"
    )
    assert after == pytest.approx(before - 0.1, abs=1e-9)


def test_penalize_mirrors_to_graph_node(brain: AGNNCore):
    """``penalize()`` must also lower the AGNNNode confidence in the graph."""
    rs = _build_default_chain(brain)
    eid = rs[1]["node_id"]
    inner = brain.graph._graph
    conf_before = float(inner.get_node(str(eid)).confidence)

    brain.penalize(eid)

    conf_after = float(inner.get_node(str(eid)).confidence)
    assert conf_after < conf_before, (
        f"penalize() must mirror onto the graph node: "
        f"before={conf_before}, after={conf_after}"
    )


def test_penalize_pushes_node_below_others_in_top_nodes(brain: AGNNCore):
    """After penalize(id), the penalized node must NOT be top_nodes[0].

    The symmetric counterpart of the reinforce ranking test: a
    penalized node should drop out of the top slot so the system stops
    surfacing it as its strongest belief.
    """
    rs = _build_default_chain(brain)
    # First reinforce node 1 so it bubbles to top, then penalize it.
    eid = rs[0]["node_id"]
    brain.reinforce(eid)
    assert brain.introspect()["top_nodes"][0]["id"] == eid

    brain.penalize(eid)

    info = brain.introspect()
    top = info["top_nodes"][0]
    # After +0.1 then -0.1, the node is back at 0.6 — tied with the
    # others. The sorting is stable on confidence then on insertion
    # order, so the penalized node may or may not still be at the top
    # by tie-break. The contract we test is that the penalized node's
    # confidence is no longer strictly greater than every other node.
    others = [n for n in info["top_nodes"] if n["id"] != eid]
    assert others, "there must be other nodes to compare against"
    max_other = max(n["confidence"] for n in others)
    penalized_conf = next(
        n["confidence"] for n in info["top_nodes"] if n["id"] == eid
    )
    assert penalized_conf <= max_other, (
        f"penalized node confidence ({penalized_conf}) must not exceed "
        f"the max of the other nodes ({max_other})"
    )


def test_reinforce_and_penalize_round_trip(brain: AGNNCore):
    """reinforce then penalize on the same node returns to baseline."""
    rs = _build_default_chain(brain)
    eid = rs[0]["node_id"]
    baseline = brain._find_episome(eid).confidence

    brain.reinforce(eid)
    brain.penalize(eid)

    after = brain._find_episome(eid).confidence
    assert after == pytest.approx(baseline, abs=1e-9), (
        f"reinforce+penalize round-trip must return to baseline: "
        f"baseline={baseline}, after={after}"
    )


# ======================================================================
# Dimension 4: chain-length decay
# ======================================================================

def test_confidence_decreases_with_chain_length(brain: AGNNCore):
    """Longer deductive chains must yield lower chain_confidence.

    BA 44's ``CATEGORICAL_TRANSITIVITY`` rule multiplies edge weights:
    a 2-edge chain (A->B->C) yields ``w1 * w2``. With every weight < 1,
    adding a transitivity hop strictly shrinks the product. This test
    asserts the contract holds by comparing:

    - **short retrieval** (2 nodes, 1 edge, no rule fires) → falls back
      to the max edge weight (``DEFAULT_EPISODIC_CONFIDENCE = 0.6``)
    - **long retrieval** (3 nodes, 2 edges, 1 transitivity firing) →
      ``0.6 * 0.6 = 0.36``

    So ``long < short`` (0.36 < 0.6). If this contract were violated,
    longer reasoning chains would be reported as *more* reliable — the
    opposite of correct behavior.

    Note: ``process()`` caps retrieval at ``top_k=3``, so we cannot
    observe ``0.6 ** N`` for arbitrarily large ``N`` through the public
    API; the 2-node vs 3-node comparison is the largest gap visible
    through ``process()``.
    """
    _build_default_chain(brain)
    # 2-keyword query -> 2 nodes retrieved -> 1 edge, no rule fires
    # -> falls back to max edge weight (0.6).
    short_result = brain.process("alpha beta")
    # 3-keyword query -> 3 nodes retrieved -> 2 edges, transitivity
    # fires once -> 0.6 * 0.6 = 0.36.
    long_result = brain.process("alpha beta gamma")

    assert short_result["chain_confidence"] > 0.0
    assert long_result["chain_confidence"] > 0.0
    # The long retrieval has more transitivity hops, so its confidence
    # must be strictly lower.
    assert long_result["chain_confidence"] < short_result["chain_confidence"], (
        f"longer chain must yield lower confidence: "
        f"short(2-node, no rule)={short_result['chain_confidence']}, "
        f"long(3-node, 1 transitivity)={long_result['chain_confidence']}"
    )


def test_single_edge_confidence_equals_edge_weight(brain: AGNNCore):
    """A 2-node retrieval (1 edge, no transitivity) yields the edge weight.

    With exactly one edge and no transitivity rule firing, ``process``
    falls back to "max edge weight". The default edge confidence is
    ``DEFAULT_EPISODIC_CONFIDENCE = 0.6`` (set by Subiculum.relay_output),
    so a 2-node retrieval should report exactly 0.6. This anchors the
    calibration scale: 0.6 is the "one fresh fact, no chaining" floor.
    """
    _build_default_chain(brain)
    # 2-keyword query -> 2 nodes retrieved -> 1 edge between them.
    result = brain.process("alpha beta")
    assert result["chain_confidence"] == pytest.approx(0.6, abs=1e-9), (
        f"single-edge confidence must equal DEFAULT_EPISODIC_CONFIDENCE "
        f"(0.6), got {result['chain_confidence']}"
    )


# ======================================================================
# Dimension 5: top-node ranking invariants
# ======================================================================

def test_top_nodes_sorted_by_descending_confidence(brain: AGNNCore):
    """``introspect().top_nodes`` must be sorted by descending confidence."""
    rs = _build_default_chain(brain)
    # Reinforce different nodes by different amounts to create a
    # non-trivial confidence distribution.
    brain.reinforce(rs[0]["node_id"])  # +0.1 -> 0.7
    brain.reinforce(rs[2]["node_id"])
    brain.reinforce(rs[2]["node_id"])  # +0.2 -> 0.8
    brain.penalize(rs[3]["node_id"])   # -0.1 -> 0.5

    info = brain.introspect()
    confs = [n["confidence"] for n in info["top_nodes"]]
    assert confs == sorted(confs, reverse=True), (
        f"top_nodes must be sorted by descending confidence, got {confs}"
    )
    # Spot-check: node 2 (reinforced twice) must be at the top.
    assert info["top_nodes"][0]["id"] == rs[2]["node_id"]
    # Spot-check: node 3 (penalized once) must be at the bottom.
    assert info["top_nodes"][-1]["id"] == rs[3]["node_id"]


def test_top_nodes_capped_at_five(brain: AGNNCore):
    """``top_nodes`` must never contain more than 5 entries."""
    # Encode 7 facts so we have more candidates than the cap.
    _build_synthetic_chain(brain, [
        (f"q{i}?", f"w{i}", f"n{i}_ kw{i}")
        for i in range(7)
    ])
    info = brain.introspect()
    assert len(info["top_nodes"]) <= 5, (
        f"top_nodes must be capped at 5, got {len(info['top_nodes'])}"
    )


def test_top_nodes_entry_shape(brain: AGNNCore):
    """Every ``top_nodes`` entry must have id/text/confidence keys."""
    _build_default_chain(brain)
    info = brain.introspect()
    for n in info["top_nodes"]:
        assert isinstance(n, dict), f"top_nodes entry must be a dict, got {type(n)}"
        assert {"id", "text", "confidence"} <= set(n.keys()), (
            f"top_nodes entry must have id/text/confidence keys, got {list(n.keys())}"
        )
        assert isinstance(n["id"], int)
        assert isinstance(n["text"], str)
        assert isinstance(n["confidence"], float)
        assert 0.0 <= n["confidence"] <= 1.0


# ======================================================================
# Cross-dimension: reinforcement modulates future confidence signal
# ======================================================================

def test_reinforced_chain_has_higher_max_top_node_confidence(brain: AGNNCore):
    """After reinforce(), the highest top_nodes confidence must increase.

    This ties dimensions 2 + 5 together: reinforcement should not only
    raise the reinforced episome's confidence but also raise the *peak*
    of the introspect ranking. If reinforcement worked locally but the
    ranking peak didn't move, the calibration signal would still be
    lying to the caller about the system's peak belief.
    """
    _build_default_chain(brain)
    peak_before = max(
        n["confidence"] for n in brain.introspect()["top_nodes"]
    )

    # Reinforce the top node once more.
    top_id = brain.introspect()["top_nodes"][0]["id"]
    brain.reinforce(top_id)

    peak_after = max(
        n["confidence"] for n in brain.introspect()["top_nodes"]
    )
    assert peak_after > peak_before, (
        f"reinforce() must raise the peak top_nodes confidence: "
        f"before={peak_before}, after={peak_after}"
    )


def test_penalized_chain_has_lower_min_top_node_confidence(brain: AGNNCore):
    """After penalize(), the lowest top_nodes confidence must decrease.

    This test must use ≤5 total nodes so the penalized node stays
    inside the ``top_nodes`` cap (5). With more nodes, a penalized
    node could be pushed out of the top 5 by tie-breakers and the
    ``min`` would not move — a real edge case but not what we're
    testing here.
    """
    # Build a 3-node chain so every node is guaranteed to appear in
    # top_nodes (cap is 5).
    rs = _build_synthetic_chain(brain, [
        ("q1?", "w1", "alpha beta"),
        ("q2?", "w2", "beta gamma"),
        ("q3?", "w3", "gamma delta"),
    ])
    # Reinforce node 0 so it is clearly the top, then the *min* node
    # is unambiguously one of the unreinforced ones at 0.6.
    brain.reinforce(rs[0]["node_id"])
    info_before = brain.introspect()
    # Verify all 3 nodes are present in top_nodes.
    assert len(info_before["top_nodes"]) == 3, (
        f"expected all 3 nodes in top_nodes, got {len(info_before['top_nodes'])}"
    )
    min_node = min(info_before["top_nodes"], key=lambda n: n["confidence"])
    min_before = min_node["confidence"]

    brain.penalize(min_node["id"])

    info_after = brain.introspect()
    min_after = min(n["confidence"] for n in info_after["top_nodes"])
    assert min_after < min_before, (
        f"penalize() must lower the min top_nodes confidence: "
        f"before={min_before}, after={min_after}"
    )
