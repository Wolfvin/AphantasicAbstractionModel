"""
Tests for BA 44 DeductiveReasoner (InferiorFrontalGyrus) and
CingulateGyrus conflict detection.

Covers all 5 rules from AGNN/ARCHITECTURE.md section 5 plus conflict
detection and edge cases.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_deductive_reasoning.py -v
"""

import os
import sys
from pathlib import Path

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


# ----------------------------------------------------------------------
# Fixtures & helpers
# ----------------------------------------------------------------------

from engrams.episodic_engram import Episome  # noqa: E402
from engrams.semantic_engram import Semesome  # noqa: E402
from neocortex.inferior_frontal_gyrus import (  # noqa: E402
    CATEGORICAL,
    CAUSAL,
    DIFFERENTIAL,
    FUNCTIONAL,
    EdgeChain,
    InferiorFrontalGyrus,
    Inference,
    Deduction,
)
from limbic_system.cingulate_gyrus import CingulateGyrus, Conflict  # noqa: E402


def _edge(t: str, w: float, src: str, dst: str) -> Semesome:
    """Helper: build a Semesome edge."""
    return Semesome(type=t, weight=w, source=src, target=dst)


@pytest.fixture
def ba44() -> InferiorFrontalGyrus:
    """Fresh InferiorFrontalGyrus (BA 44) instance per test."""
    return InferiorFrontalGyrus()


@pytest.fixture
def acc() -> CingulateGyrus:
    """Fresh CingulateGyrus (ACC) instance per test."""
    return CingulateGyrus()


# ======================================================================
# Rule 1: CATEGORICAL_TRANSITIVITY
# ======================================================================

def test_categorical_transitivity_basic(ba44: InferiorFrontalGyrus):
    """A->B (CAT 1.0), B->C (CAT 1.0) => A->C (CAT 1.0)."""
    edges = [
        _edge(CATEGORICAL, 1.0, "human", "mammal"),
        _edge(CATEGORICAL, 1.0, "mammal", "animal"),
    ]
    result = ba44.deduce(edges)

    assert result.rule_count == 1
    assert "CATEGORICAL_TRANSITIVITY" in result.applied_rules
    assert len(result.inferred_edges) == 1
    inferred = result.inferred_edges[0]
    assert inferred.type == CATEGORICAL
    assert inferred.source == "human"
    assert inferred.target == "animal"
    assert inferred.weight == pytest.approx(1.0)


def test_categorical_transitivity_chain_of_three(ba44: InferiorFrontalGyrus):
    """A->B->C->D (all CAT 1.0) => two transitivity firings: A->C and B->D."""
    edges = [
        _edge(CATEGORICAL, 1.0, "A", "B"),
        _edge(CATEGORICAL, 1.0, "B", "C"),
        _edge(CATEGORICAL, 1.0, "C", "D"),
    ]
    result = ba44.deduce(edges)

    # Two adjacent pairs: (A->B,B->C) and (B->C,C->D).
    assert result.rule_count == 2
    sources = sorted(e.source for e in result.inferred_edges)
    targets = sorted(e.target for e in result.inferred_edges)
    assert sources == ["A", "B"]
    assert targets == ["C", "D"]
    # All weights = 1.0 * 1.0 = 1.0.
    for e in result.inferred_edges:
        assert e.weight == pytest.approx(1.0)


# ======================================================================
# Rule 2: CAUSAL_CHAIN
# ======================================================================

def test_causal_chain_basic(ba44: InferiorFrontalGyrus):
    """A->B (CAUSAL 0.7), B->C (CAUSAL 0.7) => A->C (CAUSAL 0.49)."""
    edges = [
        _edge(CAUSAL, 0.7, "smoking", "lung_damage"),
        _edge(CAUSAL, 0.7, "lung_damage", "cancer"),
    ]
    result = ba44.deduce(edges)

    assert "CAUSAL_CHAIN" in result.applied_rules
    assert len(result.inferred_edges) == 1
    inferred = result.inferred_edges[0]
    assert inferred.type == CAUSAL
    assert inferred.source == "smoking"
    assert inferred.target == "cancer"
    assert inferred.weight == pytest.approx(0.49, rel=1e-9)


def test_causal_chain_mismatched_types_does_not_fire(ba44: InferiorFrontalGyrus):
    """A->B (CAUSAL), B->C (CATEGORICAL) - CAUSAL_CHAIN must NOT fire."""
    edges = [
        _edge(CAUSAL, 0.7, "smoking", "lung_damage"),
        _edge(CATEGORICAL, 1.0, "lung_damage", "organ_issue"),
    ]
    result = ba44.deduce(edges)

    assert "CAUSAL_CHAIN" not in result.applied_rules
    # CATEGORICAL_TRANSITIVITY also won't fire because first edge isn't CATEGORICAL.
    assert result.rule_count == 0


# ======================================================================
# Rule 3: DIFFERENTIAL_INVERSION
# ======================================================================

def test_differential_inversion_basic(ba44: InferiorFrontalGyrus):
    """A->B (DIFF -0.8) => B->A (DIFF -0.8). Weight is symmetric."""
    edges = [_edge(DIFFERENTIAL, -0.8, "exercise", "body_fat")]
    result = ba44.deduce(edges)

    assert "DIFFERENTIAL_INVERSION" in result.applied_rules
    assert len(result.inferred_edges) == 1
    inverted = result.inferred_edges[0]
    assert inverted.type == DIFFERENTIAL
    assert inverted.source == "body_fat"
    assert inverted.target == "exercise"
    assert inverted.weight == pytest.approx(-0.8)


def test_differential_inversion_does_not_fire_on_other_types(ba44: InferiorFrontalGyrus):
    """CATEGORICAL edge alone must NOT trigger DIFFERENTIAL_INVERSION."""
    edges = [_edge(CATEGORICAL, 1.0, "A", "B")]
    result = ba44.deduce(edges)
    assert "DIFFERENTIAL_INVERSION" not in result.applied_rules


# ======================================================================
# Rule 4: CAUSAL_DIFFERENTIAL_CONFLICT
# ======================================================================

def test_causal_differential_conflict(ba44: InferiorFrontalGyrus):
    """A->B (CAUSAL 0.7) + A->B (DIFF -0.8) => resolved weight -0.05."""
    edges = [
        _edge(CAUSAL, 0.7, "stress", "ulcer"),
        _edge(DIFFERENTIAL, -0.8, "stress", "ulcer"),
    ]
    result = ba44.deduce(edges)

    assert "CAUSAL_DIFFERENTIAL_CONFLICT" in result.applied_rules
    # Conflict rule produces ONE inference (canonicalized - no double-fire).
    conflict_inferences = [i for i in result.inferences if i.rule == "CAUSAL_DIFFERENTIAL_CONFLICT"]
    assert len(conflict_inferences) == 1
    inf = conflict_inferences[0]
    assert inf.weight == pytest.approx(-0.05, rel=1e-9)
    assert inf.conclusion is not None
    assert inf.conclusion.source == "stress"
    assert inf.conclusion.target == "ulcer"
    # Confidence drops to 0 because resolved weight is negative.
    assert result.confidence == 0.0


def test_causal_differential_conflict_no_fire_when_only_one_type(ba44: InferiorFrontalGyrus):
    """Two CAUSAL edges on same pair do NOT fire CAUSAL_DIFFERENTIAL_CONFLICT."""
    edges = [
        _edge(CAUSAL, 0.7, "X", "Y"),
        _edge(CAUSAL, 0.5, "X", "Y"),
    ]
    result = ba44.deduce(edges)
    assert "CAUSAL_DIFFERENTIAL_CONFLICT" not in result.applied_rules


# ======================================================================
# Rule 5: FUNCTIONAL_COMPOSITION
# ======================================================================

def test_functional_composition_basic(ba44: InferiorFrontalGyrus):
    """A->B (FUNC 0.6), B->C (FUNC 0.6) => A->C (FUNC 0.36)."""
    edges = [
        _edge(FUNCTIONAL, 0.6, "heart", "blood"),
        _edge(FUNCTIONAL, 0.6, "blood", "oxygen_transport"),
    ]
    result = ba44.deduce(edges)

    assert "FUNCTIONAL_COMPOSITION" in result.applied_rules
    assert len(result.inferred_edges) == 1
    inferred = result.inferred_edges[0]
    assert inferred.type == FUNCTIONAL
    assert inferred.source == "heart"
    assert inferred.target == "oxygen_transport"
    assert inferred.weight == pytest.approx(0.36, rel=1e-9)


def test_functional_composition_weight_propagation(ba44: InferiorFrontalGyrus):
    """A->B (FUNC 0.5), B->C (FUNC 0.8) => A->C (FUNC 0.4)."""
    edges = [
        _edge(FUNCTIONAL, 0.5, "X", "Y"),
        _edge(FUNCTIONAL, 0.8, "Y", "Z"),
    ]
    result = ba44.deduce(edges)
    assert result.inferred_edges[0].weight == pytest.approx(0.4, rel=1e-9)


# ======================================================================
# Conflict detection (CingulateGyrus)
# ======================================================================

def test_cingulate_detects_conflict(acc: CingulateGyrus):
    """CingulateGyrus flags CAUSAL vs DIFFERENTIAL on same (src, dst)."""
    e1 = _edge(CAUSAL, 0.7, "stress", "ulcer")
    e2 = _edge(DIFFERENTIAL, -0.8, "stress", "ulcer")
    c = acc.detect_conflict(e1, e2)

    assert c.detected is True
    assert c.resolution == "weight_aggregation"
    assert c.final_weight == pytest.approx(-0.05, rel=1e-9)
    assert len(c.premises) == 2
    assert acc.conflict_count == 1


def test_cingulate_no_conflict_when_types_same(acc: CingulateGyrus):
    """Two CAUSAL edges on same pair do NOT conflict per ACC rule."""
    e1 = _edge(CAUSAL, 0.7, "X", "Y")
    e2 = _edge(CAUSAL, 0.5, "X", "Y")
    c = acc.detect_conflict(e1, e2)
    assert c.detected is False
    assert c.resolution == "none"
    assert c.final_weight == 0.0
    assert acc.conflict_count == 0


def test_cingulate_no_conflict_when_pairs_differ(acc: CingulateGyrus):
    """CAUSAL and DIFFERENTIAL on DIFFERENT (src, dst) pairs do NOT conflict."""
    e1 = _edge(CAUSAL, 0.7, "X", "Y")
    e2 = _edge(DIFFERENTIAL, -0.8, "A", "B")
    c = acc.detect_conflict(e1, e2)
    assert c.detected is False


def test_cingulate_scan_for_conflicts(acc: CingulateGyrus):
    """CingulateGyrus.scan_for_conflicts() finds all conflicts in an edge list."""
    edges = [
        _edge(CAUSAL, 0.7, "X", "Y"),
        _edge(DIFFERENTIAL, -0.8, "X", "Y"),  # conflicts with #0
        _edge(CAUSAL, 0.6, "A", "B"),
        _edge(DIFFERENTIAL, -0.5, "A", "B"),  # conflicts with #2
        _edge(CATEGORICAL, 1.0, "P", "Q"),    # no conflict
    ]
    conflicts = acc.scan_for_conflicts(edges)
    assert len(conflicts) == 2
    final_weights = sorted(c.final_weight for c in conflicts)
    # (0.7 + -0.8)/2 = -0.05 ; (0.6 + -0.5)/2 = 0.05
    assert final_weights == pytest.approx([-0.05, 0.05], rel=1e-9)


# ======================================================================
# End-to-end: 3-node chain, multiple rules in one pass
# ======================================================================

def test_end_to_end_3_node_chain_categorical(ba44: InferiorFrontalGyrus):
    """
    End-to-end: 3-node chain A->B->C (CATEGORICAL) fires one transitivity.

    Nodes (Episome):
      A: "Socrates is a human"
      B: "human is a mortal"
      C: (inferred) "Socrates is a mortal"

    Edges (Semesome):
      Socrates -> human (CATEGORICAL 1.0)
      human    -> mortal (CATEGORICAL 1.0)

    Inference: Socrates -> mortal (CATEGORICAL 1.0)
    """
    # 3 Episome nodes (just to exercise the dataclass).
    socrates = Episome(id=1, text="Socrates", confidence=1.0)
    human = Episome(id=2, text="human", confidence=1.0)
    mortal = Episome(id=3, text="mortal", confidence=1.0)
    assert socrates.text == "Socrates"
    assert human.text == "human"
    assert mortal.text == "mortal"

    # 2 Semesome edges forming a chain.
    edges = [
        _edge(CATEGORICAL, 1.0, socrates.text, human.text),
        _edge(CATEGORICAL, 1.0, human.text, mortal.text),
    ]

    result = ba44.deduce(edges)

    assert result.rule_count == 1
    assert "CATEGORICAL_TRANSITIVITY" in result.applied_rules
    assert result.confidence == pytest.approx(1.0)

    inferred = result.inferred_edges[0]
    assert inferred.source == "Socrates"
    assert inferred.target == "mortal"
    assert inferred.type == CATEGORICAL
    assert inferred.weight == pytest.approx(1.0)

    # Human-readable context string should mention the rule and the conclusion.
    assert "CATEGORICAL_TRANSITIVITY" in result.context
    assert "Socrates->mortal" in result.context


def test_end_to_end_3_node_chain_causal(ba44: InferiorFrontalGyrus):
    """
    End-to-end: 3-node chain A->B->C (CAUSAL) fires one CAUSAL_CHAIN.

    Edges:
      smoking     -> lung_damage (CAUSAL 0.7)
      lung_damage -> cancer      (CAUSAL 0.7)

    Inference: smoking -> cancer (CAUSAL 0.49)
    Confidence: 0.49 (product of positive weights)
    """
    edges = [
        _edge(CAUSAL, 0.7, "smoking", "lung_damage"),
        _edge(CAUSAL, 0.7, "lung_damage", "cancer"),
    ]
    result = ba44.deduce(edges)

    assert result.rule_count == 1
    assert "CAUSAL_CHAIN" in result.applied_rules
    assert result.confidence == pytest.approx(0.49, rel=1e-9)

    inferred = result.inferred_edges[0]
    assert inferred.source == "smoking"
    assert inferred.target == "cancer"
    assert inferred.type == CAUSAL
    assert inferred.weight == pytest.approx(0.49, rel=1e-9)


def test_end_to_end_mixed_rules_in_one_pass(ba44: InferiorFrontalGyrus):
    """
    Multiple rules fire in one deduce() call:
      - CATEGORICAL transitivity on A->B->C
      - DIFFERENTIAL_INVERSION on D->E
    """
    edges = [
        _edge(CATEGORICAL, 1.0, "A", "B"),
        _edge(CATEGORICAL, 1.0, "B", "C"),
        _edge(DIFFERENTIAL, -0.8, "D", "E"),
    ]
    result = ba44.deduce(edges)

    assert result.rule_count == 2
    assert "CATEGORICAL_TRANSITIVITY" in result.applied_rules
    assert "DIFFERENTIAL_INVERSION" in result.applied_rules

    # Confidence is 0 because one inference has a negative weight.
    assert result.confidence == 0.0


def test_deduce_chain_legacy_entry_point(ba44: InferiorFrontalGyrus):
    """deduce_chain(EdgeChain) returns the same as deduce(edges)."""
    edges = [
        _edge(CATEGORICAL, 1.0, "A", "B"),
        _edge(CATEGORICAL, 1.0, "B", "C"),
    ]
    chain = EdgeChain(edges=edges, confidence=1.0)
    result = ba44.deduce_chain(chain)
    assert result.rule_count == 1
    assert result.inferred_edges[0].source == "A"
    assert result.inferred_edges[0].target == "C"


def test_empty_edges_no_inferences(ba44: InferiorFrontalGyrus):
    """Empty edge list => no inferences, confidence 0.0."""
    result = ba44.deduce([])
    assert result.rule_count == 0
    assert result.inferred_edges == []
    assert result.applied_rules == []
    assert result.confidence == 0.0


def test_rule_count_increments_across_calls(ba44: InferiorFrontalGyrus):
    """InferiorFrontalGyrus.rule_count is a lifetime counter across calls."""
    assert ba44.rule_count == 0
    ba44.deduce([_edge(CATEGORICAL, 1.0, "A", "B"),
                 _edge(CATEGORICAL, 1.0, "B", "C")])
    assert ba44.rule_count == 1
    ba44.deduce([_edge(CATEGORICAL, 1.0, "X", "Y"),
                 _edge(CATEGORICAL, 1.0, "Y", "Z")])
    assert ba44.rule_count == 2


# ======================================================================
# Backwards-compatibility: keep the original 4 skeleton tests working.
# (These passed before this PR; they must still pass after.)
# ======================================================================

def test_episome_dataclass():
    """Episome can be constructed with required fields and has expected attributes."""
    e = Episome(id=1, text="insulin resistance causes T2D", confidence=0.6)
    assert e.id == 1
    assert e.text == "insulin resistance causes T2D"
    assert e.confidence == 0.6
    assert e.edge_type == "CATEGORICAL"
    assert e.type == "episodic"


def test_semesome_dataclass():
    """Semesome can be constructed with required fields and has expected attributes."""
    s = Semesome(type="CAUSAL", weight=0.7, source="smoking", target="cancer")
    assert s.type == "CAUSAL"
    assert s.weight == 0.7
    assert s.source == "smoking"
    assert s.target == "cancer"
    assert s.type_memory == "semantic"


def test_engram_complex_wraps_agnn_graph():
    """
    EngramComplex wraps (does NOT replace) AGNNGraph from self-ai/src/agnn/.
    Skipped if self-ai/src/agnn/graph.py is not importable.
    """
    try:
        from agnn.graph import AGNNGraph  # noqa: F401
    except ImportError:
        pytest.skip("self-ai/src/agnn/graph.py not available - skipping wrap test")

    from engrams.engram_complex import EngramComplex

    try:
        ec = EngramComplex()
    except ImportError:
        pytest.skip("EngramComplex() could not resolve AGNNGraph at runtime")

    assert hasattr(ec, "_graph"), "EngramComplex must expose _graph attribute"
    assert isinstance(ec._graph, AGNNGraph), (
        f"_graph must be an AGNNGraph instance, got {type(ec._graph).__name__}"
    )


# Required-methods check preserved from the original skeleton test file.
REQUIRED_METHODS = [
    "learn",
    "process",
    "introspect",
    "traverse",
    "consolidate",
    "reinforce",
    "penalize",
]


def test_agnncore_has_required_methods():
    """AGNNCore exposes all 7 public methods (+ __init__ = 8 total)."""
    from core import AGNNCore

    assert callable(AGNNCore), "AGNNCore must be a class"
    for method_name in REQUIRED_METHODS:
        assert hasattr(AGNNCore, method_name), (
            f"AGNNCore missing required method: {method_name}"
        )
        assert callable(getattr(AGNNCore, method_name)), (
            f"AGNNCore.{method_name} must be callable"
        )


# ======================================================================
# Configurable t-norm: behaviour of non-default t-norms must differ
# from the legacy "product" t-norm.
#
# Background: see AGNN/docs/research-neuro-symbolic-reasoning.md §4 and
# §B1. BA44's hardcoded `a * b` IS the product t-norm; making it
# explicit & swappable is a pure structural refactor that must NOT
# change the default behaviour (every test above still passes), but
# MUST let callers opt into "lukasiewicz" or "godel" and observe
# different weights.
# ======================================================================

def test_default_t_norm_is_product(ba44: InferiorFrontalGyrus):
    """InferiorFrontalGyrus() with no args defaults to the product t-norm.

    This is what guarantees the refactor is behaviour-preserving: every
    pre-existing call site that does `InferiorFrontalGyrus()` keeps
    getting `a * b` composition.
    """
    assert ba44.t_norm == "product"

    # Sanity: 0.7 * 0.7 = 0.49 under the default.
    edges = [
        _edge(CAUSAL, 0.7, "smoking", "lung_damage"),
        _edge(CAUSAL, 0.7, "lung_damage", "cancer"),
    ]
    result = ba44.deduce(edges)
    assert result.inferred_edges[0].weight == pytest.approx(0.49, rel=1e-9)


def test_unknown_t_norm_raises():
    """An invalid t_norm name must raise ValueError, not silently fall back."""
    with pytest.raises(ValueError):
        InferiorFrontalGyrus(t_norm="not-a-t-norm")


def test_lukasiewicz_tnorm_differs_from_product():
    """Łukasiewicz t-norm: T(a, b) = max(0, a + b - 1).

    For a CAUSAL_CHAIN over (0.7, 0.7):
        product      →  0.7 * 0.7        = 0.49  (legacy)
        lukasiewicz  →  max(0, 0.7+0.7-1) = 0.40

    The two must differ; otherwise the refactor is a no-op and the
    configurability promise is hollow.
    """
    edges = [
        _edge(CAUSAL, 0.7, "smoking", "lung_damage"),
        _edge(CAUSAL, 0.7, "lung_damage", "cancer"),
    ]

    product_result = InferiorFrontalGyrus(t_norm="product").deduce(edges)
    lukasiewicz_result = InferiorFrontalGyrus(t_norm="lukasiewicz").deduce(edges)

    # Sanity: same rule fired in both regimes.
    assert "CAUSAL_CHAIN" in product_result.applied_rules
    assert "CAUSAL_CHAIN" in lukasiewicz_result.applied_rules

    product_w = product_result.inferred_edges[0].weight
    lukasiewicz_w = lukasiewicz_result.inferred_edges[0].weight

    # Sanity: product reproduces the legacy value exactly.
    assert product_w == pytest.approx(0.49, rel=1e-9)
    # Sanity: lukasiewicz matches its closed-form value.
    assert lukasiewicz_w == pytest.approx(0.40, rel=1e-9)
    # The whole point of the refactor: they must differ.
    assert product_w != pytest.approx(lukasiewicz_w, rel=1e-9)


def test_lukasiewicz_tnorm_collapses_low_premises_to_zero():
    """Łukasiewicz t-norm gives 0 when both premises are <= 0.5.

    This is the characteristic Łukasiewicz "either premise must be
    quite true" semantics that distinguishes it from the product t-norm
    (which only scales down). For (0.4, 0.5):
        product      →  0.4 * 0.5             = 0.20
        lukasiewicz  →  max(0, 0.4+0.5-1)     = 0.00  (boundary clamp)
    """
    edges = [
        _edge(FUNCTIONAL, 0.4, "X", "Y"),
        _edge(FUNCTIONAL, 0.5, "Y", "Z"),
    ]

    product_w = InferiorFrontalGyrus(t_norm="product").deduce(edges).inferred_edges[0].weight
    lukasiewicz_w = InferiorFrontalGyrus(t_norm="lukasiewicz").deduce(edges).inferred_edges[0].weight

    assert product_w == pytest.approx(0.20, rel=1e-9)
    assert lukasiewicz_w == pytest.approx(0.0, rel=1e-9)


def test_godel_tnorm_differs_from_product():
    """Gödel (minimum) t-norm: T(a, b) = min(a, b).

    For a FUNCTIONAL_COMPOSITION over (0.6, 0.6):
        product  →  0.6 * 0.6  = 0.36  (legacy)
        godel    →  min(0.6, 0.6) = 0.60

    They clearly differ. (For equal-weight premises godel returns the
    common value, product scales it down.)
    """
    edges = [
        _edge(FUNCTIONAL, 0.6, "heart", "blood"),
        _edge(FUNCTIONAL, 0.6, "blood", "oxygen_transport"),
    ]

    product_result = InferiorFrontalGyrus(t_norm="product").deduce(edges)
    godel_result = InferiorFrontalGyrus(t_norm="godel").deduce(edges)

    assert "FUNCTIONAL_COMPOSITION" in product_result.applied_rules
    assert "FUNCTIONAL_COMPOSITION" in godel_result.applied_rules

    product_w = product_result.inferred_edges[0].weight
    godel_w = godel_result.inferred_edges[0].weight

    assert product_w == pytest.approx(0.36, rel=1e-9)
    assert godel_w == pytest.approx(0.60, rel=1e-9)
    assert product_w != pytest.approx(godel_w, rel=1e-9)


def test_godel_tnorm_is_min_for_mismatched_premises():
    """Gödel t-norm returns the weaker premise. For (0.5, 0.8):
        product  →  0.5 * 0.8  = 0.40
        godel    →  min(0.5, 0.8) = 0.50
    """
    edges = [
        _edge(FUNCTIONAL, 0.5, "X", "Y"),
        _edge(FUNCTIONAL, 0.8, "Y", "Z"),
    ]

    product_w = InferiorFrontalGyrus(t_norm="product").deduce(edges).inferred_edges[0].weight
    godel_w = InferiorFrontalGyrus(t_norm="godel").deduce(edges).inferred_edges[0].weight

    assert product_w == pytest.approx(0.40, rel=1e-9)
    assert godel_w == pytest.approx(0.50, rel=1e-9)


def test_t_norm_does_not_affect_conflict_rule():
    """CAUSAL_DIFFERENTIAL_CONFLICT uses arithmetic mean, NOT a t-norm.

    So changing the t-norm must NOT change the conflict-resolved
    weight. This guards the design contract documented in
    inferior_frontal_gyrus.py: the arithmetic mean is a fuzzy
    aggregation operator but not a t-norm, and is intentionally
    excluded from t-norm configurability.
    """
    edges = [
        _edge(CAUSAL, 0.7, "stress", "ulcer"),
        _edge(DIFFERENTIAL, -0.8, "stress", "ulcer"),
    ]

    weights = {}
    for tn in ("product", "lukasiewicz", "godel"):
        result = InferiorFrontalGyrus(t_norm=tn).deduce(edges)
        conflict = [
            i for i in result.inferences
            if i.rule == "CAUSAL_DIFFERENTIAL_CONFLICT"
        ]
        assert len(conflict) == 1
        weights[tn] = conflict[0].weight

    # All three regimes must give the same arithmetic-mean weight (-0.05).
    for tn, w in weights.items():
        assert w == pytest.approx(-0.05, rel=1e-9), (
            f"conflict weight drifted under t_norm={tn!r}: got {w}"
        )


def test_t_norm_does_not_affect_differential_inversion():
    """DIFFERENTIAL_INVERSION is a unary, weight-preserving rule.

    The t-norm parameter must therefore have no effect on its output.
    """
    edges = [_edge(DIFFERENTIAL, -0.8, "exercise", "body_fat")]

    weights = {}
    for tn in ("product", "lukasiewicz", "godel"):
        result = InferiorFrontalGyrus(t_norm=tn).deduce(edges)
        inv = [
            i for i in result.inferences
            if i.rule == "DIFFERENTIAL_INVERSION"
        ]
        assert len(inv) == 1
        weights[tn] = inv[0].weight

    for tn, w in weights.items():
        assert w == pytest.approx(-0.8, rel=1e-9), (
            f"inversion weight drifted under t_norm={tn!r}: got {w}"
        )
