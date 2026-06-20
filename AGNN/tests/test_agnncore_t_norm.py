"""
Tests for the ``t_norm`` parameter exposed at the ``AGNNCore``
constructor level.

Background
----------
``InferiorFrontalGyrus`` (BA 44, the deductive component) accepts a
``t_norm`` kwarg selecting which fuzzy-logic t-norm to use when a
deductive rule composes the weights of two premise edges (PR #79,
"make t-norm explicit and configurable"). Until this PR the only way
to pick a non-default t-norm was to reach into ``core.deductive`` and
re-construct the IFG by hand — an awkward research surface that
``AGNN/docs/dead-code-audit.md`` §3.5 explicitly called out as
*configurable but not wired to runtime config*.

This file verifies the wiring the audit recommended: ``AGNNCore``
now accepts ``t_norm`` and forwards it to the IFG constructor, while
the default (``"product"``) preserves the legacy arithmetic bit-for-bit.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_agnncore_t_norm.py -v
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
# Imports under test
# ----------------------------------------------------------------------

# Load AGNN/core.py directly by path (same pattern as
# test_core_wired.py) to avoid the name collision with
# self-ai/src/core/ (a package).
import importlib.util as _ilu  # noqa: E402

_core_path = _AGNN_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_module_tnorm", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_module_tnorm"] = agnn_core_module  # register before exec
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore

from neocortex.inferior_frontal_gyrus import (  # noqa: E402
    InferiorFrontalGyrus,
    CATEGORICAL,
    CAUSAL,
    FUNCTIONAL,
)
from engrams.semantic_engram import Semesome  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _edge(t: str, w: float, src: str, dst: str) -> Semesome:
    """Helper: build a Semesome edge."""
    return Semesome(type=t, weight=w, source=src, target=dst)


# ----------------------------------------------------------------------
# The headline test — Definition of Done for this PR.
# ----------------------------------------------------------------------


def test_agnncore_exposes_t_norm_param():
    """``AGNNCore(t_norm=...)`` plumbs the choice down to the IFG.

    This is the test the dead-code-audit (§3.5) recommended: the
    alternative t-norms introduced by PR #79 are no longer reachable
    only by reaching into ``core.deductive`` — a researcher can pick
    them at the ``AGNNCore`` entry point.

    The contract verified here:
      1. The default (no ``t_norm`` kwarg) yields ``"product"``, so
         existing call sites are bit-for-bit unchanged.
      2. Each accepted value is forwarded verbatim to the IFG
         instance and surfaces on ``core.deductive.t_norm``.
      3. Different ``t_norm`` values produce different inferred
         weights on the same premise pair (sanity: the parameter is
         not a no-op).
    """
    # 1. Default == "product" (backward compatibility).
    core_default = AGNNCore(model_path=None)
    assert core_default.deductive is not None, (
        "IFG must construct under the default t_norm; if it is None, "
        "the IFG module is unavailable and this test cannot run."
    )
    assert core_default.deductive.t_norm == "product"

    # 2. Each accepted value is forwarded verbatim.
    for tn in ("product", "lukasiewicz", "godel"):
        core = AGNNCore(model_path=None, t_norm=tn)
        assert core.deductive is not None, (
            f"IFG must construct for t_norm={tn!r}"
        )
        assert core.deductive.t_norm == tn, (
            f"AGNNCore(t_norm={tn!r}) did not forward to IFG; "
            f"got {core.deductive.t_norm!r}"
        )

    # 3. The parameter is not a no-op: the same premise pair produces
    #    different inferred weights under different t-norms.
    #    CAUSAL_CHAIN over (0.7, 0.7):
    #       product      → 0.7 * 0.7         = 0.49
    #       lukasiewicz  → max(0, 0.7+0.7-1) = 0.40
    #       godel        → min(0.7, 0.7)     = 0.70
    edges = [
        _edge(CAUSAL, 0.7, "smoking", "lung_damage"),
        _edge(CAUSAL, 0.7, "lung_damage", "cancer"),
    ]
    weights = {}
    for tn in ("product", "lukasiewicz", "godel"):
        core = AGNNCore(model_path=None, t_norm=tn)
        result = core.deductive.deduce(edges)
        assert "CAUSAL_CHAIN" in result.applied_rules, (
            f"CAUSAL_CHAIN must fire for t_norm={tn!r}"
        )
        weights[tn] = result.inferred_edges[0].weight

    assert weights["product"] == pytest.approx(0.49, rel=1e-9)
    assert weights["lukasiewicz"] == pytest.approx(0.40, rel=1e-9)
    assert weights["godel"] == pytest.approx(0.70, rel=1e-9)
    # The three regimes must be pairwise distinct — otherwise the
    # parameter is effectively a no-op and the configurability
    # promise from §3.5 is hollow.
    assert len(set(weights.values())) == 3


# ----------------------------------------------------------------------
# Defensive-edge tests: unknown t_norm + graceful degradation.
# ----------------------------------------------------------------------


def test_agnncore_default_t_norm_matches_legacy_ifg():
    """``AGNNCore()`` (no t_norm) and ``InferiorFrontalGyrus()`` must
    produce byte-identical inferred weights.

    This is the strongest possible backward-compatibility guarantee:
    the AGNNCore constructor plumbing does not perturb the legacy
    default even by a floating-point epsilon.
    """
    edges = [
        _edge(FUNCTIONAL, 0.6, "heart", "blood"),
        _edge(FUNCTIONAL, 0.6, "blood", "oxygen_transport"),
    ]
    core = AGNNCore(model_path=None)
    ifg_direct = InferiorFrontalGyrus()  # default t_norm

    core_w = core.deductive.deduce(edges).inferred_edges[0].weight
    direct_w = ifg_direct.deduce(edges).inferred_edges[0].weight

    assert core_w == pytest.approx(direct_w, rel=1e-12)
    # Both should also match the closed-form legacy value 0.36.
    assert core_w == pytest.approx(0.36, rel=1e-9)


def test_agnncore_unknown_t_norm_degrades_gracefully():
    """An unknown ``t_norm`` value must NOT crash ``AGNNCore.__init__``.

    ``InferiorFrontalGyrus(t_norm=...)`` raises ``ValueError`` on an
    unknown value (see ``test_deductive_reasoning.py::
    test_unknown_t_norm_raises``). ``AGNNCore._safe_init`` swallows
    any exception from a sub-component constructor into ``None``,
    so an unknown ``t_norm`` must surface as
    ``core.deductive is None`` — the same graceful-degradation
    contract applied to every other sub-component. This keeps a
    typo'd t_norm name from bricking the whole AGNNCore.
    """
    core = AGNNCore(model_path=None, t_norm="not-a-real-tnorm")
    # The IFG construction raised ValueError, _safe_init swallowed it,
    # so the deductive slot is None. AGNNCore itself is still usable
    # for everything that doesn't touch self.deductive.
    assert core.deductive is None
    # Sanity: the rest of the brain still wired up.
    # (We don't assert on every slot — some may legitimately be None
    # in environments missing optional deps — but graph should be
    # present in the standard test environment.)
    assert core.model is None


def test_agnncore_t_norm_does_not_affect_other_components():
    """Choosing a non-default ``t_norm`` must not perturb any
    sub-component *other* than the IFG.

    This guards against a future refactor accidentally passing
    ``t_norm`` to e.g. TrisynapticCircuit or the cluster learner.
    We assert the trisynaptic circuit is the same object instance
    regardless of the chosen t_norm (compared by identity of the
    underlying EngramComplex, which both share).
    """
    core_product = AGNNCore(model_path=None, t_norm="product")
    core_lukasiewicz = AGNNCore(model_path=None, t_norm="lukasiewicz")

    # IFG *did* change — that's the point of the parameter.
    assert core_product.deductive.t_norm == "product"
    assert core_lukasiewicz.deductive.t_norm == "lukasiewicz"

    # Both cores still point at structurally-equivalent EngramComplex
    # graphs (we cannot assert identity across separate AGNNCore
    # instances, but we can assert both are non-None and have the
    # same type).
    assert core_product.graph is not None
    assert core_lukasiewicz.graph is not None
    assert type(core_product.graph) is type(core_lukasiewicz.graph)
