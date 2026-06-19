"""
End-to-end tests for Phase 3 logical validity.

Verifies the Definition-of-Done from the Phase 3 brief:
    1. Learn 3 facts with a CAUSAL chain -> verify the encoded edges
       in the AGNNGraph are CAUSAL -> verify the BA 44 CAUSAL_CHAIN
       rule fires when those edges are retrieved + deduced.
    2. Learn 3 facts with a CATEGORICAL chain -> verify the
       CATEGORICAL_TRANSITIVITY rule fires.
    3. Mixed chains (one CAUSAL + one CATEGORICAL) should NOT fire
       either transitivity rule - this is the *negative* control that
       proves the typed-edge wiring actually matters (pre-Phase-3 every
       edge was CATEGORICAL and CAUSAL_CHAIN never fired; with the new
       classifier, two CATEGORICAL + one CAUSAL won't fire CAUSAL_CHAIN
       because not all edges are CAUSAL).

The tests construct a chain by encoding three correction sentences
that share keywords so CA3 autoassociation wires them as neighbors.
The shared keyword is the "bridge" node (e.g. "lung" + "damage"
bridges fact 1 and fact 2). The TrisynapticCircuit creates a typed
edge from each new node to its autoassociative neighbors - the
RelationType on that edge is now driven by the SemanticRoleClassifier
reading the correction text.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_e2e_logical_validity.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
_AGNP_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

# Also ensure self-ai/src is importable for AGNNGraph in engram_complex.
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


# ----------------------------------------------------------------------
# Imports under test
# ----------------------------------------------------------------------

from engrams.engram_complex import EngramComplex  # noqa: E402
from circuits.trisynaptic_circuit import TrisynapticCircuit  # noqa: E402
from circuits.papez_circuit import PapezCircuit  # noqa: E402
from neocortex.inferior_frontal_gyrus import InferiorFrontalGyrus  # noqa: E402
from neocortex.semantic_role_classifier import RelationType  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_circuit() -> TrisynapticCircuit:
    """Build a TrisynapticCircuit with a fresh EngramComplex.

    Skips the test if the AGNNGraph dependency (self-ai/src/agnn) is
    unavailable - the EngramComplex constructor raises ImportError in
    that case.
    """
    try:
        ec = EngramComplex()
    except ImportError:
        pytest.skip("AGNNGraph (self-ai/src/agnn) not available")
    return TrisynapticCircuit(engram_complex=ec)


@pytest.fixture
def circuit() -> TrisynapticCircuit:
    """Fresh TrisynapticCircuit + EngramComplex per test."""
    return _make_circuit()


@pytest.fixture
def papez() -> PapezCircuit:
    """Fresh PapezCircuit per test."""
    return PapezCircuit()


@pytest.fixture
def ba44() -> InferiorFrontalGyrus:
    """Fresh InferiorFrontalGyrus (BA 44) per test."""
    return InferiorFrontalGyrus()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _edge_relation_types(circuit: TrisynapticCircuit):
    """Yield every TypedEdge.relation_type in the wrapped graph.

    Used to verify that the edges the TrisynapticCircuit recorded
    actually carry the RelationType the classifier committed to (not
    just the Episome.edge_type string).
    """
    inner = circuit.engram_complex._graph
    for node_id in inner.all_node_ids():
        for edge in inner.get_edges_from(node_id):
            yield edge.relation_type


def _build_semesomes_from_graph(
    circuit: TrisynapticCircuit,
    episomes,
):
    """Mirror of AGNNCore._build_semesomes_from_graph for the test.

    Lets us inspect exactly what BA 44 will see without going through
    the model-articulate path.
    """
    from engrams.semantic_engram import Semesome

    inner = circuit.engram_complex._graph
    retrieved_ids = {str(getattr(e, "id", "")) for e in episomes}
    id_to_text = {}
    for e in episomes:
        id_to_text[str(getattr(e, "id", ""))] = getattr(e, "text", str(e.id))
    for nid in retrieved_ids:
        node = inner.get_node(nid)
        if node is not None:
            id_to_text[nid] = node.label

    semesomes = []
    seen_pairs = set()
    for nid in retrieved_ids:
        for edge in inner.get_edges_from(nid):
            if edge.target_id not in retrieved_ids:
                continue
            pair = (edge.source_id, edge.target_id, str(edge.relation_type))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            semesomes.append(Semesome(
                type=str(edge.relation_type.value).upper(),
                weight=float(edge.confidence),
                source=id_to_text.get(edge.source_id, edge.source_id),
                target=id_to_text.get(edge.target_id, edge.target_id),
            ))
    # Order so adjacent pairs chain (same logic as AGNNCore._order_chain).
    if len(semesomes) <= 1:
        return semesomes
    remaining = list(semesomes)
    ordered = []
    sources = {e.source for e in remaining}
    targets = {e.target for e in remaining}
    head_candidates = [e for e in remaining if e.source not in targets]
    current = head_candidates[0] if head_candidates else remaining[0]
    ordered.append(current)
    remaining.remove(current)
    while remaining:
        next_edge = None
        for e in remaining:
            if e.source == current.target:
                next_edge = e
                break
        if next_edge is None:
            next_edge = remaining[0]
        ordered.append(next_edge)
        remaining.remove(next_edge)
        current = next_edge
    return ordered


# ======================================================================
# CAUSAL chain
# ======================================================================


def test_causal_chain_edges_are_causal_in_graph(circuit: TrisynapticCircuit):
    """Encoding 3 CAUSAL-bridge facts creates CAUSAL TypedEdges in the graph.

    Fact sequence (each shares a keyword with the next so CA3
    autoassociation wires them):
        e1: "smoking causes lung damage"
        e2: "lung damage causes cancer"
        e3: "cancer causes death"

    Every correction contains the seed "causes", so the
    SemanticRoleClassifier should classify each as CAUSAL, and the
    TrisynapticCircuit should record every TypedEdge as
    RelationType.CAUSAL.
    """
    circuit.encode("q1?", correction="smoking causes lung damage")
    circuit.encode("q2?", correction="lung damage causes cancer")
    circuit.encode("q3?", correction="cancer causes death")

    relations = list(_edge_relation_types(circuit))
    assert len(relations) >= 2, (
        f"expected at least 2 typed edges in the graph, got {len(relations)}"
    )
    non_causal = [r for r in relations if r != RelationType.CAUSAL]
    assert not non_causal, (
        f"all edges should be CAUSAL, found non-causal: {non_causal}"
    )


def test_causal_chain_fires_causal_chain_rule(
    circuit: TrisynapticCircuit,
    papez: PapezCircuit,
    ba44: InferiorFrontalGyrus,
):
    """Retrieving the CAUSAL chain + deducing fires BA 44's CAUSAL_CHAIN.

    End-to-end:
        1. Encode 3 facts forming a CAUSAL chain.
        2. Retrieve them via PapezCircuit.
        3. Build Semesome edges from the wrapped graph.
        4. Run InferiorFrontalGyrus.deduce() on the chain.
        5. Assert "CAUSAL_CHAIN" appears in applied_rules.
    """
    circuit.encode("q1?", correction="smoking causes lung damage")
    circuit.encode("q2?", correction="lung damage causes cancer")
    circuit.encode("q3?", correction="cancer causes death")

    # Retrieve nodes whose keywords overlap with "lung damage cancer"
    # (this should pull in all three episomes).
    episomes = papez.retrieve(
        "smoking lung damage cancer death", circuit.engram_complex, top_k=5
    )
    assert len(episomes) >= 2, (
        f"expected at least 2 retrieved episomes for chain deduction, "
        f"got {len(episomes)}"
    )

    semesomes = _build_semesomes_from_graph(circuit, episomes)
    assert semesomes, "expected at least one Semesome edge between retrieved episomes"

    deduction = ba44.deduce(semesomes)
    assert "CAUSAL_CHAIN" in deduction.applied_rules, (
        f"CAUSAL_CHAIN rule should fire on a CAUSAL A->B->C chain; "
        f"applied_rules={deduction.applied_rules}, "
        f"semesomes={[(s.type, s.source, s.target) for s in semesomes]}"
    )


# ======================================================================
# CATEGORICAL chain
# ======================================================================


def test_categorical_chain_edges_are_categorical_in_graph(
    circuit: TrisynapticCircuit,
):
    """Encoding 3 CATEGORICAL-bridge facts creates CATEGORICAL TypedEdges.

    Fact sequence:
        e1: "socrates is a human"
        e2: "human is a mammal"
        e3: "mammal is an animal"

    Every correction contains the seed "is a" (or "is an"), so the
    classifier should classify each as CATEGORICAL.
    """
    circuit.encode("q1?", correction="socrates is a human")
    circuit.encode("q2?", correction="human is a mammal")
    circuit.encode("q3?", correction="mammal is an animal")

    relations = list(_edge_relation_types(circuit))
    assert len(relations) >= 2, (
        f"expected at least 2 typed edges, got {len(relations)}"
    )
    non_cat = [r for r in relations if r != RelationType.CATEGORICAL]
    assert not non_cat, (
        f"all edges should be CATEGORICAL, found non-categorical: {non_cat}"
    )


def test_categorical_chain_fires_categorical_transitivity(
    circuit: TrisynapticCircuit,
    papez: PapezCircuit,
    ba44: InferiorFrontalGyrus,
):
    """Retrieving the CATEGORICAL chain + deducing fires
    CATEGORICAL_TRANSITIVITY.

    This is the *only* rule BA 44 could fire pre-Phase-3 (when every
    edge defaulted to CATEGORICAL). The test is included for parity
    with the CAUSAL case so we can prove the new typed-edge wiring
    preserved the old behaviour.
    """
    circuit.encode("q1?", correction="socrates is a human")
    circuit.encode("q2?", correction="human is a mammal")
    circuit.encode("q3?", correction="mammal is an animal")

    episomes = papez.retrieve(
        "socrates human mammal animal", circuit.engram_complex, top_k=5
    )
    assert len(episomes) >= 2

    semesomes = _build_semesomes_from_graph(circuit, episomes)
    assert semesomes

    deduction = ba44.deduce(semesomes)
    assert "CATEGORICAL_TRANSITIVITY" in deduction.applied_rules, (
        f"CATEGORICAL_TRANSITIVITY should fire on a CAT A->B->C chain; "
        f"applied_rules={deduction.applied_rules}, "
        f"semesomes={[(s.type, s.source, s.target) for s in semesomes]}"
    )


# ======================================================================
# Mixed-chain negative control
# ======================================================================


def test_mixed_chain_does_not_fire_either_transitivity_rule(
    circuit: TrisynapticCircuit,
    papez: PapezCircuit,
    ba44: InferiorFrontalGyrus,
):
    """A chain with one CAUSAL + one CATEGORICAL edge fires NEITHER rule.

    This is the negative control that proves the typed-edge wiring
    actually matters: BA 44's CategoricalTransitivity requires *both*
    edges to be CATEGORICAL, and CausalChain requires *both* to be
    CAUSAL. A mixed chain fires neither - exactly the discrimination
    that was impossible pre-Phase-3 (when everything was CATEGORICAL
    and CAUSAL_CHAIN never fired).

    Fact sequence:
        e1: "socrates is a human"     (CATEGORICAL)
        e2: "human causes death"      (CAUSAL — odd, but it lets us
                                       mix types in a single chain)
    """
    circuit.encode("q1?", correction="socrates is a human")
    circuit.encode("q2?", correction="human causes death")

    episomes = papez.retrieve(
        "socrates human death", circuit.engram_complex, top_k=5
    )
    assert len(episomes) >= 2

    semesomes = _build_semesomes_from_graph(circuit, episomes)
    if not semesomes:
        pytest.skip("no edges between retrieved episomes - keyword overlap too thin")

    deduction = ba44.deduce(semesomes)
    # Mixed chain: neither CATEGORICAL_TRANSITIVITY nor CAUSAL_CHAIN
    # should fire (each requires a homogeneous pair).
    assert "CATEGORICAL_TRANSITIVITY" not in deduction.applied_rules, (
        f"CATEGORICAL_TRANSITIVITY must not fire on a mixed-type chain; "
        f"applied_rules={deduction.applied_rules}, "
        f"semesomes={[(s.type, s.source, s.target) for s in semesomes]}"
    )
    assert "CAUSAL_CHAIN" not in deduction.applied_rules, (
        f"CAUSAL_CHAIN must not fire on a mixed-type chain; "
        f"applied_rules={deduction.applied_rules}, "
        f"semesomes={[(s.type, s.source, s.target) for s in semesomes]}"
    )


# ======================================================================
# AGNNCore end-to-end (process() returns chain_confidence > 0)
# ======================================================================


def test_agnn_core_process_returns_positive_confidence_on_causal_chain():
    """AGNNCore.process() on a CAUSAL chain returns chain_confidence > 0.

    This exercises the full public-API path: learn -> process. The
    pre-Phase-3 bug was that BA 44 never fired CAUSAL_CHAIN (everything
    was CATEGORICAL), so chain_confidence stayed at 0.0 or at the
    fallback max-edge-weight value. With the classifier wiring
    CAUSAL edges, BA 44 fires CAUSAL_CHAIN and the deduction
    confidence (0.7 * 0.7 = 0.49) flows through to chain_confidence.
    """
    # Import AGNNCore via path to avoid the self-ai/src/core name
    # collision (same pattern as tests/test_core_wired.py).
    import importlib.util as _ilu
    _core_path = _AGNP_ROOT / "core.py"
    _spec = _ilu.spec_from_file_location("agnn_core_e2e", _core_path)
    agnn_core_module = _ilu.module_from_spec(_spec)
    sys.modules["agnn_core_e2e"] = agnn_core_module
    _spec.loader.exec_module(agnn_core_module)
    AGNNCore = agnn_core_module.AGNNCore

    core = AGNNCore(model_path=None)
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    core.learn("q1?", "wrong1", "smoking causes lung damage")
    core.learn("q2?", "wrong2", "lung damage causes cancer")
    core.learn("q3?", "wrong3", "cancer causes death")

    result = core.process("smoking lung damage cancer death")
    assert isinstance(result, dict)
    assert "chain_confidence" in result
    assert result["chain_confidence"] > 0.0, (
        f"CAUSAL chain should produce a positive chain_confidence via "
        f"CAUSAL_CHAIN firing; got {result['chain_confidence']}, "
        f"chain={result['chain']!r}"
    )
