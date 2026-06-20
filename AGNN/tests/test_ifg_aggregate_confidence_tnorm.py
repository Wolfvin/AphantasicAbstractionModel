"""
Tests for issue #90 fix: aggregate ``deduction.confidence`` must use
the selected t-norm (not plain multiplication) for multi-inference
chains.

Background
----------
Pre-fix: ``InferiorFrontalGyrus.deduce()`` computed the aggregate
``Deduction.confidence`` with ``confidence *= w`` (plain
multiplication), regardless of the ``t_norm`` parameter. The t-norm
choice only affected per-edge ``inference.weight`` (via each rule's
``apply()``). The aggregate — which is what ``AGNNCore.process()``
surfaces as ``chain_confidence`` — silently used product semantics
even when the user selected ``"godel"`` or ``"lukasiewicz"``.

For single-inference chains (1 rule firing), the bug was invisible:
``confidence = 1.0 * w`` and ``w`` was computed via ``tnorm_fn``, so
the t-norm choice appeared to propagate. The bug only surfaced for
chains with 2+ inferences, where the fold over multiple ``w_i``
silently reverted to product.

Post-fix: the fold uses ``self._tnorm_fn(confidence, w)`` instead of
``confidence *= w``. The identity element 1.0 is correct for all
three canonical t-norms (product, lukasiewicz, godel), so the seed
value is unchanged. For ``t_norm="product"`` (the default), the fold
is bit-for-bit identical to the pre-fix arithmetic — every existing
test passes unchanged.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_ifg_aggregate_confidence_tnorm.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_AGNP_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


from engrams.semantic_engram import Semesome  # noqa: E402
from neocortex.inferior_frontal_gyrus import (  # noqa: E402
    CAUSAL,
    CATEGORICAL,
    FUNCTIONAL,
    InferiorFrontalGyrus,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _causal_chain_3_edges():
    """Build a 3-edge CAUSAL chain A->B->C->D that fires 2 CAUSAL_CHAIN rules.

    With premise weight 0.7 on every edge:
      - product t-norm:      each inference weight = 0.49 (0.7*0.7),
                              aggregate = 0.49 * 0.49 = 0.2401
      - lukasiewicz t-norm:  each inference weight = 0.4 (max(0, 0.7+0.7-1)),
                              aggregate = max(0, 0.4+0.4-1) = 0
      - godel t-norm:        each inference weight = 0.7 (min(0.7, 0.7)),
                              aggregate = min(0.7, 0.7) = 0.7

    Pre-fix bug: godel and lukasiewicz aggregates were computed via
    plain multiplication (0.49 and 0.16 respectively), silently
    reverting to product semantics.
    """
    return [
        Semesome(type=CAUSAL, weight=0.7, source="A", target="B"),
        Semesome(type=CAUSAL, weight=0.7, source="B", target="C"),
        Semesome(type=CAUSAL, weight=0.7, source="C", target="D"),
    ]


def _categorical_chain_3_edges():
    """Build a 3-edge CATEGORICAL chain A->B->C->D, fires 2 transitivity rules.

    With premise weight 1.0 on every edge (the canonical CATEGORICAL
    weight per ARCHITECTURE.md §5):
      - product:      1.0 * 1.0 = 1.0, aggregate = 1.0 * 1.0 = 1.0
      - lukasiewicz:  max(0, 1+1-1) = 1.0, aggregate = max(0, 1+1-1) = 1.0
      - godel:        min(1, 1) = 1.0, aggregate = min(1, 1) = 1.0

    All three t-norms agree at weight 1.0 — this case is degenerate
    and not useful for distinguishing them. Kept here as a sanity
    check that the fix doesn't break the canonical CATEGORICAL chain
    (which is what ``test_e2e_logical_validity.py`` exercises).
    """
    return [
        Semesome(type=CATEGORICAL, weight=1.0, source="A", target="B"),
        Semesome(type=CATEGORICAL, weight=1.0, source="B", target="C"),
        Semesome(type=CATEGORICAL, weight=1.0, source="C", target="D"),
    ]


# ======================================================================
# Issue #90 — multi-inference aggregate uses t-norm fold
# ======================================================================


def test_aggregate_confidence_uses_tnorm_for_multistep_chain():
    """Multi-inference ``deduction.confidence`` must use ``self._tnorm_fn``.

    This is the regression guard for issue #90. The 3-edge CAUSAL chain
    A->B->C->D fires two ``CAUSAL_CHAIN`` rules (A->C and B->D, or
    A->D depending on rule matching). The aggregate confidence is the
    t-norm fold over both inference weights.

    Pre-fix: aggregate always used plain multiplication, so godel and
    lukasiewicz silently reverted to product semantics.

    Post-fix: aggregate uses ``self._tnorm_fn``, so:
      - product:      0.49 * 0.49 = 0.2401 (backward compat, unchanged)
      - lukasiewicz:  max(0, 0.4 + 0.4 - 1) = 0.0 (was 0.16 pre-fix)
      - godel:        min(0.7, 0.7) = 0.7 (was 0.49 pre-fix)

    The test asserts:
      1. All three t-norms fire the same number of rules (sanity).
      2. Per-inference weights differ by t-norm (the per-edge wiring
         was already correct pre-fix — this is a sanity check that
         the fix didn't break it).
      3. Aggregate confidence differs by t-norm (the actual #90 fix).
      4. Specifically, godel aggregate > product aggregate > lukasiewicz
         aggregate. (Gödel is the most permissive conjunction;
         Lukasiewicz is the strictest; Product is in between for
         weights in (0, 1).)
    """
    edges = _causal_chain_3_edges()

    results = {}
    for t_norm in ("product", "lukasiewicz", "godel"):
        ba44 = InferiorFrontalGyrus(t_norm=t_norm)
        ded = ba44.deduce(edges)
        results[t_norm] = {
            "rule_count": ded.rule_count,
            "inference_weights": [round(float(i.weight), 6) for i in ded.inferences],
            "confidence": round(float(ded.confidence), 6),
        }

    # 1. All three t-norms fire the same number of rules (the rule
    #    matcher is t-norm-agnostic; only the weight computation
    #    differs).
    assert results["product"]["rule_count"] == results["godel"]["rule_count"]
    assert results["product"]["rule_count"] == results["lukasiewicz"]["rule_count"]
    # The chain A->B->C->D should fire at least 2 transitivity rules
    # (A->C and B->D, or A->C and A->D — the exact pair depends on
    # rule implementation, but it's >= 2).
    assert results["product"]["rule_count"] >= 2, (
        f"Expected at least 2 inferences on a 3-edge CAUSAL chain; "
        f"got {results['product']['rule_count']}. The test relies on "
        f"multi-inference aggregation to surface the bug."
    )

    # 2. Per-inference weights differ by t-norm (per-edge wiring was
    #    already correct pre-fix; this guards against regressions).
    assert results["product"]["inference_weights"] != results["godel"]["inference_weights"], (
        f"Per-inference weights must differ between product and godel; "
        f"got identical {results['product']['inference_weights']}."
    )
    assert results["product"]["inference_weights"] != results["lukasiewicz"]["inference_weights"], (
        f"Per-inference weights must differ between product and lukasiewicz; "
        f"got identical {results['product']['inference_weights']}."
    )

    # 3. THE ACTUAL #90 FIX: aggregate confidence differs by t-norm.
    #    Pre-fix, godel and lukasiewicz aggregates were equal to
    #    product's (because the fold used plain multiplication).
    conf_product = results["product"]["confidence"]
    conf_godel = results["godel"]["confidence"]
    conf_luka = results["lukasiewicz"]["confidence"]

    assert conf_product != conf_godel, (
        f"Aggregate confidence for product ({conf_product}) must differ "
        f"from godel ({conf_godel}). If they're equal, the aggregate "
        f"fold is still using plain multiplication — issue #90 not fixed."
    )
    assert conf_product != conf_luka, (
        f"Aggregate confidence for product ({conf_product}) must differ "
        f"from lukasiewicz ({conf_luka}). If they're equal, the aggregate "
        f"fold is still using plain multiplication — issue #90 not fixed."
    )

    # 4. Ordering: godel > product > lukasiewicz for weights in (0, 1).
    #    Gödel = min — most permissive (chain is as strong as strongest
    #    single link's weakest input).
    #    Product = multiply — middle ground.
    #    Lukasiewicz = max(0, a+b-1) — strictest (collapses to 0 unless
    #    both premises are quite strong).
    assert conf_godel > conf_product > conf_luka, (
        f"Expected godel ({conf_godel}) > product ({conf_product}) > "
        f"lukasiewicz ({conf_luka}) for premise weights in (0, 1). "
        f"Ordering violation suggests t-norm semantics are wrong."
    )


def test_aggregate_confidence_product_backward_compat_unchanged():
    """``t_norm="product"`` aggregate must be bit-for-bit identical to pre-fix.

    Pre-fix: ``confidence = 1.0; for w in weights: confidence *= w``.
    Post-fix: ``confidence = 1.0; for w in weights: confidence = product_tnorm(confidence, w)``.

    Since ``product_tnorm(a, b) = a * b``, the two are mathematically
    identical. This test asserts the bit-for-bit equality on the
    canonical CAUSAL chain, so any accidental change to the product
    path surfaces immediately.

    The expected value 0.2401 = 0.49 * 0.49 = (0.7 * 0.7) * (0.7 * 0.7)
    is the pre-fix value that every existing test (e.g.
    ``test_deductive_reasoning.py`` weight assertions) was written
    against. If this test passes, all existing tests pass.
    """
    edges = _causal_chain_3_edges()
    ba44 = InferiorFrontalGyrus(t_norm="product")
    ded = ba44.deduce(edges)
    # 0.49 * 0.49 = 0.2401 (with IEEE 754 floating-point, this is
    # 0.2401 to within 1e-15 — use approx).
    assert ded.confidence == pytest.approx(0.2401, abs=1e-9), (
        f"product aggregate must equal 0.2401 (backward compat); "
        f"got {ded.confidence!r}. If this changed, existing tests in "
        f"test_deductive_reasoning.py and test_e2e_logical_validity.py "
        f"will fail."
    )


def test_aggregate_confidence_godel_matches_min_fold():
    """Gödel aggregate must equal ``min(w1, w2, ...)`` over inference weights.

    For the 3-edge CAUSAL chain with premise weight 0.7:
      - each inference weight = min(0.7, 0.7) = 0.7
      - aggregate = min(0.7, 0.7) = 0.7

    Pre-fix bug: aggregate was 0.49 (plain product of 0.7 and 0.7).
    """
    edges = _causal_chain_3_edges()
    ba44 = InferiorFrontalGyrus(t_norm="godel")
    ded = ba44.deduce(edges)
    inf_weights = [float(i.weight) for i in ded.inferences]
    expected = min(inf_weights) if inf_weights else 0.0
    assert ded.confidence == pytest.approx(expected, abs=1e-9), (
        f"godel aggregate must equal min of inference weights "
        f"({expected!r}); got {ded.confidence!r}."
    )
    # Specifically for the 0.7-premise chain, this is 0.7.
    assert ded.confidence == pytest.approx(0.7, abs=1e-9), (
        f"godel aggregate for 0.7-premise 3-edge chain must equal 0.7; "
        f"got {ded.confidence!r}."
    )


def test_aggregate_confidence_lukasiewicz_collapses_to_zero():
    """Lukasiewicz aggregate must collapse to 0 for weak-premise multi-step chains.

    For the 3-edge CAUSAL chain with premise weight 0.7:
      - each inference weight = max(0, 0.7+0.7-1) = 0.4
      - aggregate = max(0, 0.4+0.4-1) = 0.0

    Pre-fix bug: aggregate was 0.16 (plain product of 0.4 and 0.4).

    This collapse is the intended Lukasiewicz semantics: "weak premises
    produce weak (rather than merely reduced) evidence" — see the
    lukasiewicz_tnorm docstring.
    """
    edges = _causal_chain_3_edges()
    ba44 = InferiorFrontalGyrus(t_norm="lukasiewicz")
    ded = ba44.deduce(edges)
    assert ded.confidence == pytest.approx(0.0, abs=1e-9), (
        f"lukasiewicz aggregate for 0.7-premise 3-edge chain must "
        f"collapse to 0.0; got {ded.confidence!r}. The whole point "
        f"of Lukasiewicz is that weak premises produce zero evidence "
        f"once combined."
    )


# ======================================================================
# Categorical chain sanity (all 3 t-norms agree at weight 1.0)
# ======================================================================


def test_aggregate_confidence_categorical_chain_all_tnorms_agree_at_weight_1():
    """All t-norms must produce confidence 1.0 on a 1.0-premise chain.

    For weight 1.0:
      - product:      1.0 * 1.0 = 1.0
      - lukasiewicz:  max(0, 1+1-1) = 1.0
      - godel:        min(1, 1) = 1.0

    This is the degenerate case where all three t-norms agree. The
    test guards against accidentally breaking the canonical
    CATEGORICAL chain (which ``test_e2e_logical_validity.py`` also
    exercises, but only with the default product t-norm).
    """
    edges = _categorical_chain_3_edges()
    for t_norm in ("product", "lukasiewicz", "godel"):
        ba44 = InferiorFrontalGyrus(t_norm=t_norm)
        ded = ba44.deduce(edges)
        assert ded.confidence == pytest.approx(1.0, abs=1e-9), (
            f"{t_norm} aggregate for 1.0-premise CATEGORICAL chain must "
            f"equal 1.0; got {ded.confidence!r}."
        )


# ======================================================================
# Single-inference case (was already correct pre-fix; sanity guard)
# ======================================================================


def test_aggregate_confidence_single_inference_uses_tnorm():
    """Single-inference ``deduction.confidence`` must equal ``tnorm_fn(1.0, w)``.

    For a single inference, ``confidence = self._tnorm_fn(1.0, w) = w``
    (since 1.0 is the identity for all three canonical t-norms). This
    case was already correct pre-fix (because ``1.0 * w == w``), but
    the test guards against accidentally changing the identity seed.
    """
    # 2-edge CAUSAL chain A->B->C fires 1 CAUSAL_CHAIN rule (A->C).
    edges = [
        Semesome(type=CAUSAL, weight=0.7, source="A", target="B"),
        Semesome(type=CAUSAL, weight=0.7, source="B", target="C"),
    ]
    for t_norm in ("product", "lukasiewicz", "godel"):
        ba44 = InferiorFrontalGyrus(t_norm=t_norm)
        ded = ba44.deduce(edges)
        # Should fire exactly 1 inference.
        assert ded.rule_count == 1, (
            f"{t_norm}: expected 1 inference on 2-edge CAUSAL chain; "
            f"got {ded.rule_count}."
        )
        inf_weight = float(ded.inferences[0].weight)
        # Aggregate = tnorm(1.0, inf_weight) = inf_weight for all 3 t-norms.
        assert ded.confidence == pytest.approx(inf_weight, abs=1e-9), (
            f"{t_norm}: single-inference aggregate ({ded.confidence!r}) "
            f"must equal the inference weight ({inf_weight!r})."
        )
