"""
Phase 2 (Cross-Node Definition Consistency Check) — unit tests.

Covers:
  A. CingulateGyrus.detect_definition_conflict — direct unit tests
     (surface mismatch, empty definitions, similarity threshold,
     identical definitions, api-vs-API-style conflict).
  B. CingulateGyrus.scan_for_definition_conflicts — batch scan.
  C. DefinitionConflict dataclass — field defaults + audit fields.
  D. AGNNCore.learn integration — surface_collision flag, lazy
     conflict detection, conflict surfacing in return dict.
  E. Audit log separation — definition_conflict_log kept separate
     from conflict_log (existing adversarial tests unaffected).
  F. Failure contract — broken nodes / missing attributes don't
     crash learn().

Design notes:
  - All tests construct Episomes manually (no learn() / no model
    load) so the suite runs without a real Qwen3-0.6B checkpoint.
  - The api-vs-API test is the Phase 2 acceptance test: it asserts
    that two Episomes with surface "api" and divergent definitions
    ("fenomena pembakaran" vs "application programming interface")
    are flagged as a conflict. This is the user-reported bug that
    Phase 1 + Phase 2 together solve.
  - The "no conflict on identical defn" test is the false-positive
    guard: two Episomes with the same surface AND same definition
    must NOT be flagged (that's reinforcement, not conflict).

Run:
    python -m pytest AGNN/tests/test_definition_conflict.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Path setup — same convention as the other AGNN test modules.
_AGNP_ROOT = Path(__file__).resolve().parent.parent
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

# Load AGNN/core.py by path to avoid the self-ai/src/core/ name collision.
import importlib.util as _ilu  # noqa: E402

_core_path = _AGNP_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_phase2_module", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_phase2_module"] = agnn_core_module
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore

from engrams.episodic_engram import Episome  # noqa: E402
from limbic_system.cingulate_gyrus import (  # noqa: E402
    CingulateGyrus,
    DefinitionConflict,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_epi(
    text: str,
    amodal_definition: str = "",
    epi_id: int = 1,
    confidence: float = 0.6,
) -> Episome:
    """Construct an Episome with the Phase 1 fields populated."""
    return Episome(
        id=epi_id,
        text=text,
        confidence=confidence,
        amodal_definition=amodal_definition,
    )


# ======================================================================
# Component A — detect_definition_conflict direct unit tests
# ======================================================================


@pytest.fixture
def cingulate() -> CingulateGyrus:
    """Fresh CingulateGyrus per test (empty logs)."""
    return CingulateGyrus()


def test_definition_conflict_api_vs_api(cingulate: CingulateGyrus):
    """The Phase 2 acceptance test: 'api' = fire vs 'api' = programming.

    Two Episomes share the surface "api" but carry divergent amodal
    definitions (one Indonesian "fenomena pembakaran", one English
    "application programming interface"). The Jaccard similarity of
    their token sets is 0.0 (zero shared tokens) — well below the
    default 0.3 threshold. A conflict must be flagged.
    """
    epi_fire = _make_epi("api", "fenomena pembakaran", epi_id=1)
    epi_prog = _make_epi("api", "application programming interface", epi_id=2)

    result = cingulate.detect_definition_conflict(epi_fire, epi_prog)

    assert result.detected is True
    assert result.resolution == "surface_for_review"
    assert result.similarity == 0.0
    assert result.surface == "api"
    assert result.definition_a == "fenomena pembakaran"
    assert result.definition_b == "application programming interface"
    assert result.node_a_id == 1
    assert result.node_b_id == 2
    assert "DEFINITION CONFLICT" in result.note


def test_definition_conflict_no_conflict_on_identical_definitions(cingulate: CingulateGyrus):
    """Same surface + same definition → NOT a conflict (reinforcement, not conflict).

    This is the false-positive guard: the user re-learning the same
    fact (or reinforcing it) should not be flagged as a conflict.
    Jaccard similarity = 1.0 (identical token sets) → no conflict.
    """
    epi1 = _make_epi("api", "fenomena pembakaran", epi_id=1)
    epi2 = _make_epi("api", "fenomena pembakaran", epi_id=2)

    result = cingulate.detect_definition_conflict(epi1, epi2)

    assert result.detected is False
    assert result.similarity == 1.0


def test_definition_conflict_no_conflict_on_different_surfaces(cingulate: CingulateGyrus):
    """Different surfaces → no definition conflict (they're different concepts).

    Even if the definitions are wildly divergent, two Episomes with
    different surface texts cannot definition-conflict — they're
    simply different concepts.
    """
    epi1 = _make_epi("api", "fenomena pembakaran", epi_id=1)
    epi2 = _make_epi("air", "cairan untuk minum", epi_id=2)

    result = cingulate.detect_definition_conflict(epi1, epi2)

    assert result.detected is False
    assert "surfaces differ" in result.note


def test_definition_conflict_no_conflict_when_one_definition_empty(cingulate: CingulateGyrus):
    """If either definition is empty, no conflict can be judged.

    This covers the lazy-populate case: a freshly-learned Episome has
    amodal_definition="" until the first articulate call. We must not
    false-positive on it.
    """
    epi1 = _make_epi("api", "fenomena pembakaran", epi_id=1)
    epi2 = _make_epi("api", "", epi_id=2)  # not yet populated

    result = cingulate.detect_definition_conflict(epi1, epi2)

    assert result.detected is False
    assert "empty" in result.note


def test_definition_conflict_no_conflict_when_both_definitions_empty(cingulate: CingulateGyrus):
    """Both definitions empty → no conflict (lazy populate pending on both)."""
    epi1 = _make_epi("api", "", epi_id=1)
    epi2 = _make_epi("api", "", epi_id=2)

    result = cingulate.detect_definition_conflict(epi1, epi2)

    assert result.detected is False


def test_definition_conflict_threshold_respected(cingulate: CingulateGyrus):
    """Custom threshold overrides the default 0.3.

    Two definitions with partial overlap ("api fenomena pembakaran"
    vs "api aplikasi pemrograman") — Jaccard should be ~0.17 (1
    shared token "api" out of 5 unique tokens). With default
    threshold 0.3, this is a conflict. With threshold 0.1, it's not.
    """
    epi1 = _make_epi("api", "api fenomena pembakaran", epi_id=1)
    epi2 = _make_epi("api", "api aplikasi pemrograman", epi_id=2)

    # Default threshold 0.3 → conflict (similarity 0.17 < 0.3).
    result_default = cingulate.detect_definition_conflict(epi1, epi2)
    assert result_default.detected is True
    assert result_default.threshold == 0.3

    # Stricter threshold 0.1 → no conflict (similarity 0.17 >= 0.1).
    result_strict = cingulate.detect_definition_conflict(
        epi1, epi2, threshold=0.1
    )
    assert result_strict.detected is False
    assert result_strict.threshold == 0.1


def test_definition_conflict_case_insensitive_surface(cingulate: CingulateGyrus):
    """Surface comparison is case-insensitive ("Api" == "api")."""
    epi1 = _make_epi("Api", "fenomena pembakaran", epi_id=1)
    epi2 = _make_epi("api", "application programming interface", epi_id=2)

    result = cingulate.detect_definition_conflict(epi1, epi2)

    assert result.detected is True
    assert result.surface == "api"  # normalized to lower-case


def test_definition_conflict_whitespace_collapsed_in_surface(cingulate: CingulateGyrus):
    """Surface comparison collapses internal whitespace."""
    epi1 = _make_epi("api   menyebabkan   panas", "definisi a", epi_id=1)
    epi2 = _make_epi("api menyebabkan panas", "definisi b", epi_id=2)

    result = cingulate.detect_definition_conflict(epi1, epi2)

    # Surfaces match after whitespace collapse → check proceeds to
    # definition comparison. "definisi a" vs "definisi b" share
    # "definisi" → Jaccard = 1/3 ≈ 0.33 → above default threshold
    # 0.3 → no conflict. We assert the surface normalization worked
    # (the check reached the similarity stage, not the surface-differ
    # stage).
    assert result.detected is False
    assert "similarity" in result.note  # reached the similarity stage
    assert result.surface == "api menyebabkan panas"


def test_definition_conflict_increments_counter_and_logs(cingulate: CingulateGyrus):
    """A detected conflict increments the counter + appends to the log."""
    epi1 = _make_epi("api", "fenomena pembakaran", epi_id=1)
    epi2 = _make_epi("api", "application programming interface", epi_id=2)

    assert cingulate.definition_conflict_count == 0
    assert cingulate.definition_conflict_log == []

    cingulate.detect_definition_conflict(epi1, epi2)

    assert cingulate.definition_conflict_count == 1
    assert len(cingulate.definition_conflict_log) == 1
    assert cingulate.definition_conflict_log[0].detected is True


def test_definition_conflict_no_increment_on_no_conflict(cingulate: CingulateGyrus):
    """A non-detected conflict does NOT increment the counter."""
    epi1 = _make_epi("api", "fenomena pembakaran", epi_id=1)
    epi2 = _make_epi("air", "cairan untuk minum", epi_id=2)  # different surface

    cingulate.detect_definition_conflict(epi1, epi2)

    assert cingulate.definition_conflict_count == 0
    assert cingulate.definition_conflict_log == []


def test_definition_conflict_failure_contract_missing_attributes(cingulate: CingulateGyrus):
    """Broken nodes (missing text/amodal_definition) don't crash the checker.

    Returns detected=False with a note explaining the failure.
    """

    class _BrokenNode:
        # No text, no amodal_definition — getattr returns the default.
        id = 99

    broken = _BrokenNode()
    epi = _make_epi("api", "fenomena pembakaran", epi_id=1)

    result = cingulate.detect_definition_conflict(broken, epi)

    # broken.text defaults to "" → surfaces differ ("" vs "api") →
    # no conflict, but no crash either.
    assert result.detected is False


# ======================================================================
# Component B — scan_for_definition_conflicts
# ======================================================================


def test_scan_for_definition_conflicts_finds_all(cingulate: CingulateGyrus):
    """Batch scan finds all pairwise conflicts in a node list."""
    nodes = [
        _make_epi("api", "fenomena pembakaran", epi_id=1),
        _make_epi("api", "application programming interface", epi_id=2),
        _make_epi("air", "cairan untuk minum", epi_id=3),
        _make_epi("air", "gas di atmosfer", epi_id=4),
    ]

    conflicts = cingulate.scan_for_definition_conflicts(nodes)

    assert len(conflicts) == 2
    surfaces = {c.surface for c in conflicts}
    assert surfaces == {"api", "air"}


def test_scan_for_definition_conflicts_empty_list(cingulate: CingulateGyrus):
    """Empty node list → no conflicts."""
    assert cingulate.scan_for_definition_conflicts([]) == []


def test_scan_for_definition_conflicts_no_conflicts(cingulate: CingulateGyrus):
    """All-distinct surfaces → no conflicts."""
    nodes = [
        _make_epi("api", "definisi api", epi_id=1),
        _make_epi("air", "definisi air", epi_id=2),
        _make_epi("tanah", "definisi tanah", epi_id=3),
    ]
    assert cingulate.scan_for_definition_conflicts(nodes) == []


# ======================================================================
# Component C — DefinitionConflict dataclass
# ======================================================================


def test_definition_conflict_dataclass_fields():
    """DefinitionConflict has all the documented fields with correct types."""
    conflict = DefinitionConflict(
        detected=True,
        resolution="surface_for_review",
        similarity=0.0,
        threshold=0.3,
        surface="api",
        definition_a="fenomena pembakaran",
        definition_b="application programming interface",
        node_a_id=1,
        node_b_id=2,
        note="test note",
    )
    assert conflict.detected is True
    assert conflict.resolution == "surface_for_review"
    assert conflict.similarity == 0.0
    assert conflict.threshold == 0.3
    assert conflict.surface == "api"
    assert conflict.definition_a == "fenomena pembakaran"
    assert conflict.definition_b == "application programming interface"
    assert conflict.node_a_id == 1
    assert conflict.node_b_id == 2
    assert conflict.note == "test note"


# ======================================================================
# Component D — AGNNCore.learn integration
# ======================================================================


@pytest.fixture
def brain():
    """Fresh AGNNCore with the EngramComplex backend available.

    Skips the test when the backend is missing (sparse checkout
    without self-ai/src/agnn).
    """
    core = AGNNCore(model_path=None)
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")
    return core


def test_learn_returns_phase2_fields(brain: AGNNCore):
    """learn() return dict now includes definition_conflict + surface_collision."""
    result = brain.learn("q?", "wrong", "api menyebabkan panas")

    assert "definition_conflict" in result
    assert "surface_collision" in result
    # First learn — no pre-existing episome → no collision, no conflict.
    assert result["surface_collision"] is False
    assert result["definition_conflict"] is None


def test_learn_surfaces_collision_on_duplicate_surface(brain: AGNNCore):
    """learn() with a duplicate surface sets surface_collision=True.

    The new episome's amodal_definition is empty at learn time (lazy
    generation pending), so no *conflict* is detected — but the
    collision flag tells the caller there's a pre-existing episome
    with the same surface.
    """
    brain.learn("q1?", "wrong1", "api menyebabkan panas")
    result = brain.learn("q2?", "wrong2", "api menyebabkan panas")

    assert result["surface_collision"] is True
    # No conflict detected yet — both definitions are empty.
    assert result["definition_conflict"] is None


def test_learn_detects_conflict_when_both_definitions_populated(brain: AGNNCore):
    """When both episomes have definitions populated, conflict is detected.

    This simulates the post-articulate case: the user has already
    asked about "api" once (so the first episome's definition is
    populated), then learns a *different* meaning of "api" with a
    pre-populated definition (e.g. via direct field assignment or a
    pre-populated episome passed through learn()).
    """
    # First learn — surface "api", no definition yet.
    r1 = brain.learn("q1?", "wrong1", "api")
    epi1 = brain._find_episome(r1["node_id"])
    # Manually populate the first episome's definition (simulating
    # what _populate_definition would do at articulate time).
    epi1.amodal_definition = "fenomena pembakaran"

    # Second learn — same surface "api", but this time we pre-
    # populate the new episome's definition by intercepting the
    # trisynaptic circuit. The simplest way: learn, then immediately
    # populate the new episome's definition, then re-check.
    r2 = brain.learn("q2?", "wrong2", "api")
    epi2 = brain._find_episome(r2["node_id"])
    epi2.amodal_definition = "application programming interface"

    # The learn() return for r2 reported surface_collision=True but
    # definition_conflict=None (epi2's definition was empty at learn
    # time). Now we re-run the check — both definitions are populated.
    conflict, collision = brain._check_definition_conflict(epi2)

    assert collision is True
    assert conflict is not None
    assert conflict.detected is True
    assert conflict.surface == "api"


def test_learn_no_collision_on_distinct_surfaces(brain: AGNNCore):
    """Two learns with different surfaces → no collision, no conflict."""
    brain.learn("q1?", "wrong1", "api menyebabkan panas")
    result = brain.learn("q2?", "wrong2", "air membasahi tanah")

    assert result["surface_collision"] is False
    assert result["definition_conflict"] is None


def test_learn_fallback_dict_has_phase2_fields():
    """The fallback return dict (trisynaptic unavailable) has Phase 2 fields.

    We construct an AGNNCore, force trisynaptic to None, then call
    learn() — the fallback path must return the same shape as the
    success path so callers don't have to special-case None.
    """
    core = AGNNCore(model_path=None)
    # Force the trisynaptic circuit to None to trigger the fallback.
    core.trisynaptic = None

    result = core.learn("q?", "wrong", "api")

    assert result["node_id"] is None
    assert result["confidence"] == 0.0
    assert "definition_conflict" in result
    assert "surface_collision" in result
    assert result["definition_conflict"] is None
    assert result["surface_collision"] is False


# ======================================================================
# Component E — audit log separation
# ======================================================================


def test_definition_conflict_log_separate_from_conflict_log(brain: AGNNCore):
    """definition_conflict_log is NOT populated into conflict_log.

    Existing callers (e.g. the adversarial test suite) introspect
    ``conflict_log`` and ``conflict_count`` — Phase 2 must not
    pollute those with definition conflicts.
    """
    # Trigger a definition conflict by populating two episomes with
    # the same surface + divergent definitions.
    r1 = brain.learn("q1?", "wrong1", "api")
    epi1 = brain._find_episome(r1["node_id"])
    epi1.amodal_definition = "fenomena pembakaran"

    r2 = brain.learn("q2?", "wrong2", "api")
    epi2 = brain._find_episome(r2["node_id"])
    epi2.amodal_definition = "application programming interface"

    # Run the conflict check explicitly (it's the same code path
    # learn() uses, just exposed for testing).
    brain._check_definition_conflict(epi2)

    assert brain.cingulate.definition_conflict_count >= 1
    assert len(brain.cingulate.definition_conflict_log) >= 1
    # The original conflict_log (for edge-type conflicts) must NOT
    # contain any DefinitionConflict instances.
    for entry in brain.cingulate.conflict_log:
        assert not isinstance(entry, DefinitionConflict), (
            "DefinitionConflict must NOT leak into conflict_log — "
            "it belongs in definition_conflict_log only"
        )


# ======================================================================
# Component F — failure contract
# ======================================================================


def test_check_definition_conflict_returns_none_none_when_cingulate_none():
    """When self.cingulate is None, _check_definition_conflict returns (None, False)."""
    core = AGNNCore(model_path=None)
    core.cingulate = None  # force the None path

    epi = _make_epi("api", "fenomena pembakaran")
    conflict, collision = core._check_definition_conflict(epi)

    assert conflict is None
    assert collision is False


def test_check_definition_conflict_swallows_exceptions(brain: AGNNCore):
    """A broken episome (raises on attribute access) doesn't crash the check.

    The _check_definition_conflict method wraps its body in try/except
    so a broken episome can't crash learn(). We simulate a broken
    episome by patching its .text property to raise.
    """

    class _RaisingEpisome:
        id = 1

        @property
        def text(self):
            raise RuntimeError("boom")

        @property
        def amodal_definition(self):
            return ""

    # Insert the raising episome + a normal one with the same surface.
    brain._episomes.append(_RaisingEpisome())
    normal = _make_epi("api", "fenomena pembakaran", epi_id=2)
    brain._episomes.append(normal)

    # The check must not raise — it should return (None, False).
    conflict, collision = brain._check_definition_conflict(normal)
    # Either (None, False) on exception, or (None, False) because the
    # raising episome was skipped — both are acceptable.
    assert conflict is None
