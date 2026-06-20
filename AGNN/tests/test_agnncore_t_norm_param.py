"""
Tests for AGNNCore's exposed ``t_norm`` parameter.

Background
----------
``AGNN/docs/dead-code-audit.md`` §3.5 flagged the alternative t-norms
in ``InferiorFrontalGyrus`` (``lukasiewicz_tnorm``, ``godel_tnorm``,
the ``_TNORMS`` dict, and the ``t_norm`` constructor parameter) as
"mungkin dead *for production*" — production callers previously had no
way to set them without reaching into ``core.deductive.t_norm``
directly. §9 of the audit explicitly recommends wiring this research
surface up to ``AGNNCore`` so researchers can pick a t-norm without
internal reach-in.

This test file verifies the wiring introduced in this PR:

1. ``AGNNCore`` accepts a ``t_norm`` parameter.
2. The default ``"product"`` reproduces the pre-PR arithmetic
   (backward-compat — every existing test still passes unchanged).
3. Non-default values (``"lukasiewicz"``, ``"godel"``) are plumbed
   through to ``InferiorFrontalGyrus.t_norm``.
4. Invalid values raise ``ValueError`` at construction time (so a
   typo doesn't silently leave ``core.deductive`` as None via
   ``_safe_init``'s exception-swallowing fallback).
5. ``_t_norm_requested`` is always set (even when BA 44 itself is
   unavailable), for test introspection.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_agnncore_t_norm_param.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
_AGNP_ROOT = Path(__file__).resolve().parent.parent
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"

# Insert self-ai/src FIRST (lower priority) so AGNN package wins on
# name collisions - same pattern as test_core_wired.py and
# test_bootstrap_classifier.py.
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))


# ----------------------------------------------------------------------
# Imports under test
# ----------------------------------------------------------------------

# Load AGNN/core.py by path to avoid name collision with
# self-ai/src/core/ (same pattern as test_core_wired.py and
# test_bootstrap_classifier.py).
import importlib.util as _ilu  # noqa: E402

_core_path = _AGNP_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_tnorm_test", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_tnorm_test"] = agnn_core_module
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore

from neocortex.inferior_frontal_gyrus import (  # noqa: E402
    godel_tnorm,
    lukasiewicz_tnorm,
    product_tnorm,
)


# ----------------------------------------------------------------------
# Skip-if-missing helpers
# ----------------------------------------------------------------------


def _require_self_ai_graph():
    """Skip the test if self-ai/src/agnn/graph.py is not importable.

    Several tests in this file construct a full AGNNCore (which tries
    to build EngramComplex -> AGNNGraph). In environments without
    self-ai/src/, that init path degrades gracefully but
    ``core.deductive`` becomes the only meaningful attribute to check.
    """
    if not _SELF_AI_SRC.exists():
        pytest.skip(
            "self-ai/src/agnn/graph.py not available - AGNNCore's "
            "full init pipeline requires EngramComplex."
        )


# ======================================================================
# Default behaviour — backward compat with pre-PR AGNNCore
# ======================================================================


def test_agnncore_default_t_norm_is_product():
    """AGNNCore() with no ``t_norm`` arg must default to ``"product"``.

    This is the backward-compatibility contract: every existing test
    in the repo (test_deductive_reasoning, test_e2e_logical_validity,
    test_bootstrap_classifier, etc.) constructs AGNNCore with no
    ``t_norm`` argument, and their weight-related assertions assume
    the product t-norm (e.g. ``0.7 * 0.7 = 0.49`` for CAUSAL_CHAIN,
    ``0.6 * 0.6 = 0.36`` for FUNCTIONAL_COMPOSITION,
    ``1.0 * 1.0 = 1.0`` for CATEGORICAL_TRANSITIVITY).

    If this default changes, every weight assertion in the test suite
    will break.
    """
    core = AGNNCore(model_path=None, use_cluster_learner=False)
    assert core._t_norm_requested == "product", (
        f"Default t_norm must be 'product' (backward compat); "
        f"got {core._t_norm_requested!r}."
    )
    # If BA 44 is available, it must also carry the same value.
    if core.deductive is not None:
        assert core.deductive.t_norm == "product", (
            f"core.deductive.t_norm must be 'product' by default; "
            f"got {core.deductive.t_norm!r}."
        )


def test_agnncore_default_t_norm_reproduces_product_arithmetic():
    """The default t-norm must compute ``a * b`` bit-for-bit.

    Concrete check: a CAUSAL chain with two 0.7-weight premises must
    produce an inferred weight of exactly 0.49 under the default
    t-norm. This is the pre-PR arithmetic — any t-norm other than
    "product" would produce a different value (Lukasiewicz: 0.4,
    Gödel: 0.7).
    """
    core = AGNNCore(model_path=None, use_cluster_learner=False)
    if core.deductive is None:
        pytest.skip("InferiorFrontalGyrus not available")

    # Reach into the resolved t-norm callable directly. This is the
    # function BA 44 uses to combine premise weights in
    # CATEGORICAL_TRANSITIVITY / CAUSAL_CHAIN / FUNCTIONAL_COMPOSITION.
    tnorm_fn = core.deductive._tnorm_fn
    assert tnorm_fn is product_tnorm, (
        f"Default t-norm callable must be product_tnorm; "
        f"got {tnorm_fn!r}."
    )

    # The arithmetic check itself. Use approx for floating-point
    # comparison: 0.7 * 0.7 = 0.48999999999999993 in IEEE 754.
    assert tnorm_fn(0.7, 0.7) == pytest.approx(0.49), (
        f"product(0.7, 0.7) must equal 0.49; got {tnorm_fn(0.7, 0.7)!r}."
    )
    assert tnorm_fn(0.6, 0.6) == pytest.approx(0.36)
    assert tnorm_fn(1.0, 1.0) == pytest.approx(1.0)


# ======================================================================
# Non-default t-norms plumbed through correctly
# ======================================================================


@pytest.mark.parametrize("t_norm_name,expected_fn", [
    ("product", product_tnorm),
    ("lukasiewicz", lukasiewicz_tnorm),
    ("godel", godel_tnorm),
])
def test_agnncore_t_norm_plumbed_to_inferior_frontal_gyrus(
    t_norm_name, expected_fn,
):
    """AGNNCore(t_norm=X) wires X through to InferiorFrontalGyrus.

    The whole point of the audit's §9 "Wire to existing pipeline"
    recommendation: the user/researcher must be able to pick a t-norm
    from the AGNNCore constructor without reaching into
    ``core.deductive`` directly. This test verifies the plumbing by
    inspecting ``core.deductive.t_norm`` (the string) and
    ``core.deductive._tnorm_fn`` (the resolved callable).
    """
    core = AGNNCore(
        model_path=None,
        use_cluster_learner=False,
        t_norm=t_norm_name,
    )
    assert core._t_norm_requested == t_norm_name, (
        f"_t_norm_requested must echo the constructor argument; "
        f"got {core._t_norm_requested!r} (expected {t_norm_name!r})."
    )
    if core.deductive is None:
        pytest.skip("InferiorFrontalGyrus not available")
    assert core.deductive.t_norm == t_norm_name, (
        f"core.deductive.t_norm must equal the constructor argument; "
        f"got {core.deductive.t_norm!r} (expected {t_norm_name!r})."
    )
    assert core.deductive._tnorm_fn is expected_fn, (
        f"core.deductive._tnorm_fn must be {expected_fn.__name__}; "
        f"got {core.deductive._tnorm_fn!r}."
    )


def test_agnncore_t_norm_lukasiewicz_arithmetic_differs_from_product():
    """Sanity: Lukasiewicz t-norm produces a different result than product.

    Specifically, T(0.7, 0.7):
      - product:      0.49
      - lukasiewicz:  max(0, 0.7 + 0.7 - 1) = 0.4

    This confirms the wiring actually changes BA 44's arithmetic, not
    just the string label. A "lukasiewicz" selection that secretly
    still called ``product_tnorm`` would fail this test.
    """
    core = AGNNCore(
        model_path=None,
        use_cluster_learner=False,
        t_norm="lukasiewicz",
    )
    if core.deductive is None:
        pytest.skip("InferiorFrontalGyrus not available")
    fn = core.deductive._tnorm_fn
    assert fn(0.7, 0.7) == pytest.approx(0.4), (
        f"lukasiewicz(0.7, 0.7) must equal 0.4; got {fn(0.7, 0.7)!r}."
    )
    # Boundary: weak premises collapse to 0 under Lukasiewicz.
    assert fn(0.3, 0.4) == pytest.approx(0.0), (
        f"lukasiewicz(0.3, 0.4) must collapse to 0; "
        f"got {fn(0.3, 0.4)!r}."
    )


def test_agnncore_t_norm_godel_arithmetic_differs_from_product():
    """Sanity: Gödel t-norm produces a different result than product.

    Specifically, T(0.7, 0.4):
      - product:  0.28
      - godel:    min(0.7, 0.4) = 0.4

    Same rationale as the Lukasiewicz test above.
    """
    core = AGNNCore(
        model_path=None,
        use_cluster_learner=False,
        t_norm="godel",
    )
    if core.deductive is None:
        pytest.skip("InferiorFrontalGyrus not available")
    fn = core.deductive._tnorm_fn
    assert fn(0.7, 0.4) == pytest.approx(0.4), (
        f"godel(0.7, 0.4) must equal 0.4 (min); got {fn(0.7, 0.4)!r}."
    )
    assert fn(1.0, 0.6) == pytest.approx(0.6)
    assert fn(0.6, 1.0) == pytest.approx(0.6)


# ======================================================================
# Error handling — invalid t_norm values
# ======================================================================


def test_agnncore_invalid_t_norm_raises_at_construction():
    """An invalid ``t_norm`` value must raise ValueError immediately.

    Without explicit validation, ``_safe_init`` would swallow the
    ValueError from ``InferiorFrontalGyrus(t_norm="bogus")`` and
    silently leave ``core.deductive`` as None. The user would then
    hit a confusing AttributeError on the first ``process()`` call.

    The fix: AGNNCore validates ``t_norm`` *before* calling
    ``_safe_init``, so the typo surfaces at construction time with a
    clear error message.
    """
    with pytest.raises(ValueError, match=r"Unknown t_norm"):
        AGNNCore(
            model_path=None,
            use_cluster_learner=False,
            t_norm="bogus",
        )


def test_agnncore_invalid_t_norm_error_message_lists_valid_options():
    """The ValueError message must list all valid t_norm names.

    A user typing ``t_norm="godel"`` as ``t_norm="Gödel"`` (with the
    diacritic) or ``t_norm="minimum"`` (Gödel's alternate name)
    should be able to read the error and immediately see the correct
    spelling.
    """
    with pytest.raises(ValueError) as exc_info:
        AGNNCore(
            model_path=None,
            use_cluster_learner=False,
            t_norm="Gödel",  # wrong casing + diacritic
        )
    msg = str(exc_info.value)
    assert "product" in msg, f"Error message must mention 'product': {msg!r}"
    assert "lukasiewicz" in msg, (
        f"Error message must mention 'lukasiewicz': {msg!r}"
    )
    assert "godel" in msg, f"Error message must mention 'godel': {msg!r}"


# ======================================================================
# Introspection — _t_norm_requested always set
# ======================================================================


def test_agnncore_t_norm_requested_attribute_set_even_when_deductive_unavailable():
    """``_t_norm_requested`` must be set even if BA 44 fails to init.

    When ``_safe_init`` returns None for ``InferiorFrontalGyrus``
    (e.g. an unrelated import error in the module), tests can still
    inspect ``core._t_norm_requested`` to verify what the caller
    asked for. This mirrors the pattern of
    ``_use_cluster_learner_requested`` (which is set even when the
    cluster learner itself fails to load).
    """
    # We can't easily force InferiorFrontalGyrus to be unavailable
    # without mocking, but we can verify the attribute exists and is
    # correct on a normal construction. The "even when unavailable"
    # path is exercised implicitly by tests that run in environments
    # where self-ai/src is missing.
    core = AGNNCore(
        model_path=None,
        use_cluster_learner=False,
        t_norm="lukasiewicz",
    )
    assert hasattr(core, "_t_norm_requested"), (
        "AGNNCore must always set _t_norm_requested, even when "
        "InferiorFrontalGyrus is unavailable."
    )
    assert core._t_norm_requested == "lukasiewicz"


# ======================================================================
# Full-pipeline smoke test — t_norm flows through end-to-end
# ======================================================================


def test_agnncore_t_norm_with_full_pipeline_does_not_crash():
    """AGNNCore with a non-default t_norm runs ``process()`` end-to-end.

    This is the "wire to existing pipeline" smoke test: a researcher
    running an A/B t-norm experiment needs to know that picking
    ``"lukasiewicz"`` or ``"godel"`` doesn't break the
    ``learn() -> articulate()`` pipeline. We don't assert on the
    specific output text (which depends on the LLM); we only check
    that no exception escapes.
    """
    _require_self_ai_graph()

    for t_norm_name in ("product", "lukasiewicz", "godel"):
        core = AGNNCore(
            model_path=None,
            use_cluster_learner=False,
            t_norm=t_norm_name,
        )
        # BA 44 + graph must both be available for the pipeline to
        # actually exercise the t-norm. If either is None, skip.
        if core.deductive is None or core.graph is None:
            pytest.skip(
                "InferiorFrontalGyrus or EngramComplex unavailable - "
                "cannot exercise the full pipeline."
            )

        # Learn a simple CAUSAL fact. This exercises PCL/SRC -> CA3 ->
        # CA1 -> Subiculum -> graph wiring. The t-norm itself only
        # fires during ``process()``'s deduction step, but a clean
        # ``learn()`` confirms the pipeline up to that point is
        # unaffected by the t_norm choice.
        result = core.learn(
            question="q",
            wrong="w",
            correction="api menyebabkan kebakaran",
        )
        assert result["node_id"] is not None, (
            f"learn() must succeed with t_norm={t_norm_name!r}; "
            f"got result={result!r}."
        )

        # process() runs the full pipeline including BA 44 deduction.
        # We only check it returns without raising. The result shape
        # may vary (with/without model), so we just check the call
        # succeeds.
        try:
            core.process(question="apa akibat api?")
        except Exception as e:
            # A non-None exception here means the t-norm choice broke
            # the pipeline somewhere — fail loudly.
            pytest.fail(
                f"process() raised with t_norm={t_norm_name!r}: "
                f"{type(e).__name__}: {e}"
            )
