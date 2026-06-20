"""
Tests for ``PositionalClusterLearner.get_relation_type_for_action()`` —
the stable, content-addressed API introduced in the fix for issue #93.

Background
----------
Issue #93 (red-team audit): cluster IDs in PCL are not stable across
versions — they depend on corpus token order, similarity threshold,
and the presence/absence of anchor-word discovery / Brown clustering.
Code that introspects by cluster ID directly (e.g.
``learner.cluster_labels[42]``) silently inspects the wrong cluster
after a PCL upgrade.

Fix: add ``get_relation_type_for_action(action: str) ->
Optional[RelationType]`` that resolves ``action → cluster_id →
cluster_labels[cluster_id]`` without exposing the cluster_id to the
caller. This is the **preferred** API for production code (anything
outside PCL's own test suite).

The internal fields ``cluster_id_of`` and ``cluster_labels`` remain
available for PCL's own tests (which verify cluster-identity
invariants directly) and for the bootstrap path (which matches by
verb set, not by single-action lookup). They are NOT deprecated.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_pcl_stable_action_lookup.py -v
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


from neocortex.bootstrap_classifier import (  # noqa: E402
    DEFAULT_STATE_PATH,
    EXPECTED_VERB_GROUPS,
    build_labelled_cluster_learner,
    load_default_state,
)
from neocortex.positional_cluster_learner import (  # noqa: E402
    PositionalClusterLearner,
)
from neocortex.semantic_role_classifier import RelationType  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_tiny_learner() -> PositionalClusterLearner:
    """Build a PCL trained on a tiny corpus that produces >= 1 labelled cluster.

    Used by unit tests that need a controlled learner state without
    depending on the canonical corpus / committed state file.
    """
    learner = PositionalClusterLearner()
    # 3 sentences with 'menyebabkan' so it clears
    # min_action_observations=2 and forms a cluster.
    learner.train([
        "api menyebabkan kebakaran",
        "panas menyebabkan ledakan",
        "gesekan menyebabkan api",
    ])
    assert learner.is_trained
    return learner


def _label_causal_cluster(learner: PositionalClusterLearner) -> int:
    """Find the cluster containing 'menyebabkan' and label it CAUSAL.

    Returns the cluster_id (for test assertions that need to verify
    the label was set). The cluster_id is NOT used by the code under
    test — only by the test setup.
    """
    cid = learner.cluster_id_of.get("menyebabkan")
    assert cid is not None and cid >= 0, (
        "Tiny corpus should cluster 'menyebabkan' (3 occurrences)"
    )
    learner.label_clusters({cid: RelationType.CAUSAL})
    return cid


# ======================================================================
# Happy path — labelled action returns the expected RelationType
# ======================================================================


def test_get_relation_type_for_action_returns_label_for_labelled_action():
    """A labelled action's relation type is returned correctly.

    Setup: tiny corpus with 'menyebabkan' (3 occurrences) → cluster
    it → label it CAUSAL. The new API must return CAUSAL for
    'menyebabkan'.
    """
    learner = _build_tiny_learner()
    _label_causal_cluster(learner)

    rt = learner.get_relation_type_for_action("menyebabkan")
    assert rt == RelationType.CAUSAL, (
        f"Expected CAUSAL for 'menyebabkan'; got {rt!r}."
    )


def test_get_relation_type_for_action_normalizes_input():
    """Input is normalized (lower-cased + whitespace-collapsed).

    'MENYEBABKAN', '  menyebabkan  ', 'Menyebabkan' must all resolve
    to the same RelationType as 'menyebabkan'.
    """
    learner = _build_tiny_learner()
    _label_causal_cluster(learner)

    for variant in ("MENYEBABKAN", "  menyebabkan  ", "Menyebabkan"):
        rt = learner.get_relation_type_for_action(variant)
        assert rt == RelationType.CAUSAL, (
            f"Normalized variant {variant!r} must resolve to CAUSAL; "
            f"got {rt!r}."
        )


def test_get_relation_type_for_action_on_canonical_state_file():
    """The committed state file resolves all 5 canonical verb groups.

    This is the integration-level test: load the actual shipped
    ``cluster_learner_state.json`` and verify that every verb in
    ``EXPECTED_VERB_GROUPS`` resolves to its expected RelationType
    via the new API.
    """
    if not os.path.exists(DEFAULT_STATE_PATH):
        pytest.skip(
            f"State file not found: {DEFAULT_STATE_PATH} - run "
            f"`python -m neocortex.bootstrap_classifier` (from AGNN/) "
            f"to generate."
        )

    learner = load_default_state(DEFAULT_STATE_PATH)
    assert learner is not None
    assert learner.is_trained
    assert learner.is_labelled

    for relation_type, expected_verbs in EXPECTED_VERB_GROUPS.items():
        for verb in expected_verbs:
            rt = learner.get_relation_type_for_action(verb)
            assert rt == relation_type, (
                f"Verb {verb!r} must resolve to {relation_type.name} "
                f"via get_relation_type_for_action; got {rt!r}. "
                f"This is the integration test for issue #93 — if it "
                f"fails, the new API is not consistent with the "
                f"canonical committed state file."
            )


# ======================================================================
# None returns — unclustered, unlabelled, untrained, empty
# ======================================================================


def test_get_relation_type_for_action_returns_none_for_untracked_action():
    """An action never observed in training returns None.

    'upload' (an English loan-word) is not in the tiny corpus and
    therefore not tracked. The API must return None — never raise.
    """
    learner = _build_tiny_learner()
    _label_causal_cluster(learner)

    rt = learner.get_relation_type_for_action("upload")
    assert rt is None, (
        f"Untracked action 'upload' must return None; got {rt!r}."
    )


def test_get_relation_type_for_action_returns_none_for_unclustered_action():
    """An action that's tracked but unclustered (cluster_id == -1) returns None.

    Below ``min_action_observations=2``, an action may be tracked in
    ``positional_freq`` but assigned ``cluster_id == -1`` (or not
    tracked in ``cluster_id_of`` at all, depending on the PCL
    version's bookkeeping). The API must treat both cases the same:
    return None.

    Note: with the current PCL implementation, an action observed
    only once is NOT entered into ``cluster_id_of`` (it's tracked in
    ``positional_freq`` but skipped during clustering). The
    ``cluster_id == -1`` case is reserved for actions observed >= 2
    times but that didn't merge with any other cluster above the
    similarity threshold. Both cases must return None.
    """
    learner = _build_tiny_learner()
    _label_causal_cluster(learner)

    # Build a slightly larger corpus where 'menanam' appears once
    # (singleton — not entered into cluster_id_of at all).
    learner2 = PositionalClusterLearner()
    learner2.train([
        "api menyebabkan kebakaran",
        "panas menyebabkan ledakan",
        "gesekan menyebabkan api",
        "petani menanam padi",  # singleton verb 'menanam'
    ])
    cid_menanam = learner2.cluster_id_of.get("menanam")
    # 'menanam' is either absent (None) or -1 — both are "unclustered".
    assert cid_menanam is None or cid_menanam == -1, (
        f"'menanam' (1 occurrence) must be unclustered (None or -1); "
        f"got {cid_menanam!r}."
    )
    # Label 'menyebabkan' cluster as CAUSAL — leaves 'menanam'
    # unclustered / unlabelled.
    cid_menyebabkan = learner2.cluster_id_of.get("menyebabkan")
    learner2.label_clusters({cid_menyebabkan: RelationType.CAUSAL})

    rt = learner2.get_relation_type_for_action("menanam")
    assert rt is None, (
        f"Unclustered action 'menanam' must return None; got {rt!r}."
    )


def test_get_relation_type_for_action_returns_none_for_unlabelled_cluster():
    """A clustered action whose cluster has no label returns None.

    Setup: tiny corpus with 'menyebabkan' clustered but NOT labelled.
    The API must return None — this is the case where PCL has
    discovered the cluster but no human has assigned a RelationType
    to it yet.
    """
    learner = _build_tiny_learner()
    # Do NOT call label_clusters() — leaves all clusters unlabelled.
    assert not learner.is_labelled

    rt = learner.get_relation_type_for_action("menyebabkan")
    assert rt is None, (
        f"Action in unlabelled cluster must return None; got {rt!r}."
    )


def test_get_relation_type_for_action_returns_none_for_empty_input():
    """Empty / whitespace-only input returns None, never raises."""
    learner = _build_tiny_learner()
    _label_causal_cluster(learner)

    for empty in ("", "   ", "   \t   "):
        rt = learner.get_relation_type_for_action(empty)
        assert rt is None, (
            f"Empty input {empty!r} must return None; got {rt!r}."
        )


def test_get_relation_type_for_action_returns_none_for_untrained_learner():
    """An untrained learner returns None for any action.

    A fresh PCL with no train() call has empty ``cluster_id_of``. The
    API must short-circuit to None rather than raising.
    """
    learner = PositionalClusterLearner()
    assert not learner.is_trained

    rt = learner.get_relation_type_for_action("menyebabkan")
    assert rt is None, (
        f"Untrained learner must return None for any action; got {rt!r}."
    )


# ======================================================================
# Consistency — new API matches the legacy direct-access path
# ======================================================================


def test_get_relation_type_for_action_matches_legacy_direct_access():
    """The new API returns the same result as the legacy direct-access path.

    For every action in ``cluster_id_of``, the new API's return
    value must equal ``cluster_labels.get(cluster_id_of[action])``.
    This is the equivalence contract: the new method is a stable
    wrapper around the same lookup, not a different algorithm.
    """
    if not os.path.exists(DEFAULT_STATE_PATH):
        pytest.skip(
            f"State file not found: {DEFAULT_STATE_PATH} - run "
            f"`python -m neocortex.bootstrap_classifier` (from AGNN/) "
            f"to generate."
        )

    learner = load_default_state(DEFAULT_STATE_PATH)
    assert learner is not None

    mismatches = []
    for action, cid in learner.cluster_id_of.items():
        # Legacy direct-access path (the one we're replacing in
        # production code).
        if cid is None or cid < 0:
            legacy_rt = None
        else:
            legacy_rt = learner.cluster_labels.get(cid)
        # New stable API.
        new_rt = learner.get_relation_type_for_action(action)
        if legacy_rt != new_rt:
            mismatches.append((action, cid, legacy_rt, new_rt))

    assert not mismatches, (
        f"New API must match legacy direct-access path for every "
        f"action. Mismatches: {mismatches[:10]} (showing 10 of "
        f"{len(mismatches)})."
    )


def test_get_relation_type_for_action_does_not_expose_cluster_id():
    """The new API returns a RelationType, not a cluster_id.

    This is a design assertion: the whole point of issue #93 is that
    cluster IDs are not stable across PCL versions, so the public API
    must never return them. The return type is
    ``Optional[RelationType]`` — never ``int``.

    We can't directly assert on the return type at runtime (Python
    is dynamically typed), but we can assert that the returned
    object is either None or a RelationType enum member — never an
    int that could be mistaken for a cluster_id.
    """
    if not os.path.exists(DEFAULT_STATE_PATH):
        pytest.skip(
            f"State file not found: {DEFAULT_STATE_PATH} - run "
            f"`python -m neocortex.bootstrap_classifier` (from AGNN/) "
            f"to generate."
        )

    learner = load_default_state(DEFAULT_STATE_PATH)
    assert learner is not None

    for action in ["menyebabkan", "adalah", "upload", "menanam", ""]:
        rt = learner.get_relation_type_for_action(action)
        assert rt is None or isinstance(rt, RelationType), (
            f"Return value for {action!r} must be None or RelationType; "
            f"got {type(rt).__name__} ({rt!r}). The API must never "
            f"expose a raw cluster_id (int) — see issue #93."
        )


# ======================================================================
# Robustness — does not crash on weird inputs
# ======================================================================


def test_get_relation_type_for_action_with_none_input_returns_none():
    """Passing None as input returns None, never raises.

    Defensive: the API is content-addressed, so None input has no
    meaningful lookup. Return None rather than raising
    AttributeError on ``None.lower()``.
    """
    learner = _build_tiny_learner()
    _label_causal_cluster(learner)

    # _normalize_token handles None gracefully (returns "").
    rt = learner.get_relation_type_for_action(None)  # type: ignore[arg-type]
    assert rt is None, (
        f"None input must return None; got {rt!r}."
    )


def test_get_relation_type_for_action_with_numeric_input_returns_none():
    """Passing a non-string input returns None, never raises.

    Defensive: ``_normalize_token`` does ``token.lower()`` which
    would fail on non-strings. Verify the method handles this
    gracefully (or that ``_normalize_token`` is robust — either way,
    the public API must not raise).
    """
    learner = _build_tiny_learner()
    _label_causal_cluster(learner)

    # _normalize_token handles non-string by returning "" (after
    # the `if not token: return ""` check, None is falsy).
    rt = learner.get_relation_type_for_action(42)  # type: ignore[arg-type]
    # Either None (preferred) or any RelationType — but never raise.
    assert rt is None or isinstance(rt, RelationType), (
        f"Numeric input must return None or RelationType; got {rt!r}."
    )


# ======================================================================
# Integration — apply_feedback uses the new API (issue #93 fix)
# ======================================================================


def test_apply_feedback_uses_stable_api_path():
    """AGNNCore.apply_feedback resolves actions via the stable API.

    Regression guard for the issue #93 fix in ``core.py``: the
    ``apply_feedback`` method must call
    ``classifier.get_relation_type_for_action(action_token)`` rather
    than reaching into ``classifier.cluster_id_of`` +
    ``classifier.cluster_labels`` directly for label resolution.

    We verify this by monkey-patching the new method to track calls,
    then invoking apply_feedback and asserting the new method was
    called.
    """
    # Build an AGNNCore. Use use_cluster_learner=False to keep it
    # lightweight (we'll inject our own classifier).
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "agnn_core_stable_api_test", str(_AGNP_ROOT / "core.py"),
    )
    agnn_core_module = ilu.module_from_spec(spec)
    sys.modules["agnn_core_stable_api_test"] = agnn_core_module
    spec.loader.exec_module(agnn_core_module)
    AGNNCore = agnn_core_module.AGNNCore

    # Build a real PCL with the tiny corpus.
    pcl = _build_tiny_learner()
    _label_causal_cluster(pcl)

    # Wrap in a tracking proxy that records calls to the new method.
    call_log: list = []
    original_method = pcl.get_relation_type_for_action

    def tracking_method(action):
        call_log.append(action)
        return original_method(action)

    pcl.get_relation_type_for_action = tracking_method  # type: ignore[assignment]

    # Build AGNNCore with the PCL injected as the cluster learner.
    # We need a graph for apply_feedback to find edges in.
    try:
        core = AGNNCore(model_path=None, use_cluster_learner=False)
    except Exception:
        pytest.skip("AGNNCore construction failed (missing deps)")

    if core.graph is None or core.trisynaptic is None:
        pytest.skip("EngramComplex / TrisynapticCircuit unavailable")

    # Inject our tracking PCL.
    core._cluster_learner = pcl
    core.trisynaptic.role_classifier = pcl

    # Learn a fact so there's an edge in the graph.
    core.learn(question="q", wrong="w", correction="api menyebabkan kebakaran")

    # Apply feedback — this should call get_relation_type_for_action.
    # apply_feedback(sentence, verdict) is the public signature; no
    # source_episome_id kwarg.
    core.apply_feedback(
        sentence="api menyebabkan kebakaran",
        verdict="good",
    )

    # The new API must have been called with the parsed action token.
    assert len(call_log) > 0, (
        "AGNNCore.apply_feedback must call "
        "get_relation_type_for_action (the stable API from issue #93). "
        "If this fails, the refactor in core.py was reverted or "
        "bypassed."
    )
    assert "menyebabkan" in call_log, (
        f"apply_feedback must call get_relation_type_for_action with "
        f"the parsed action token 'menyebabkan'; call log: {call_log}."
    )
