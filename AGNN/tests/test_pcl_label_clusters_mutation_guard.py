"""
Tests for issue #91 fix: ``PositionalClusterLearner.label_clusters()``
warns when called with ``graph_has_existing_edges=True``.

Background
----------
Issue #91 (red-team audit): if ``PCL.label_clusters()`` is called
after edges already exist in the graph (simulating a retrain/upgrade
mid-session), the existing edges retain their old ``relation_type``
while new edges get the new labels. This produces mixed-type edges
for the same predicate, which mutes BA 44's transitivity rules.

Fix (Option 1 from the issue — documentation + warning guard):
- ``PCL.label_clusters()`` accepts a new optional flag
  ``graph_has_existing_edges: bool = False``. When True, it emits a
  ``RuntimeWarning`` describing the mixed-type-edge risk. Non-blocking
  (warning, not exception) so legitimate A/B experiment use cases
  still work.
- ``PCL.load()`` and ``TrisynapticCircuit.encode()`` docstrings
  document the same invariant (``edge_type`` is snapshotted at encode
  time; loading a state file with different labels mid-session has
  the same effect as ``label_clusters()`` mid-session).

This test file verifies the warning behavior. The invariant itself
(snapshotted edge_type) is documented in the docstrings but not
enforced in code — full re-classification of existing edges is
Option 2 from the issue, intentionally out of scope.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_pcl_label_clusters_mutation_guard.py -v
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

_AGNP_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


from neocortex.positional_cluster_learner import (  # noqa: E402
    PositionalClusterLearner,
)
from neocortex.semantic_role_classifier import RelationType  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_minimally_trained_pcl() -> PositionalClusterLearner:
    """Build a PCL trained on a tiny corpus that produces >= 1 cluster.

    The corpus is intentionally tiny — we only need the learner to
    have ``action_clusters`` populated so ``label_clusters()`` has
    something to label. The actual cluster contents don't matter for
    the warning-behavior tests.
    """
    learner = PositionalClusterLearner()
    # Tiny corpus with one CAUSAL predicate used 3x so it clears
    # min_action_observations=2 and forms a cluster.
    corpus = [
        "api menyebabkan kebakaran",
        "panas menyebabkan ledakan",
        "gesekan menyebabkan api",
    ]
    learner.train(corpus)
    assert learner.is_trained, "Tiny corpus should produce a trained learner"
    assert len(learner.action_clusters) >= 1, (
        f"Tiny corpus should produce >= 1 action_cluster; "
        f"got {len(learner.action_clusters)}"
    )
    return learner


# ======================================================================
# DoD test — label_clusters warns when graph_has_existing_edges=True
# ======================================================================


def test_label_clusters_warns_when_graph_has_existing_edges():
    """``label_clusters(graph_has_existing_edges=True)`` emits RuntimeWarning.

    This is the Definition-of-Done test for issue #91. The warning
    must:
      1. Be a ``RuntimeWarning`` (not UserWarning, not Exception).
      2. Mention the key risk terms: "mixed-type", "BA 44",
         "transitivity", "issue #91" (or equivalent — see assertion).
      3. Be non-blocking — ``label_clusters()`` must still apply the
         mapping despite the warning (callers who know what they're
         doing can suppress the warning and proceed).
      4. Have ``stacklevel=2`` so the warning points at the caller
         of ``label_clusters()``, not at the warning() call site
         inside the method.
    """
    learner = _build_minimally_trained_pcl()
    # Find the first cluster_id to label
    cluster_id = next(iter(learner.action_clusters.keys()))

    # Capture warnings
    with warnings.catch_warnings(record=True) as caught:
        # Don't suppress — we want to see the warning.
        warnings.simplefilter("always")
        learner.label_clusters(
            {cluster_id: RelationType.CAUSAL},
            graph_has_existing_edges=True,
        )

    # 1. At least one warning was emitted.
    assert len(caught) >= 1, (
        "label_clusters(graph_has_existing_edges=True) must emit at "
        "least one warning; got none."
    )

    # 2. The first warning is a RuntimeWarning.
    w = caught[0]
    assert issubclass(w.category, RuntimeWarning), (
        f"Warning must be RuntimeWarning; got {w.category.__name__}."
    )

    # 3. The warning message mentions the key risk terms.
    msg = str(w.message)
    assert "label_clusters" in msg or "PositionalClusterLearner" in msg, (
        f"Warning message must mention label_clusters/PositionalCluster"
        f"ClusterLearner; got: {msg!r}"
    )
    assert "mixed-type" in msg.lower() or "mixed type" in msg.lower(), (
        f"Warning message must mention 'mixed-type' risk; got: {msg!r}"
    )
    assert "BA 44" in msg or "BA44" in msg or "transitivity" in msg.lower(), (
        f"Warning message must mention BA 44 / transitivity rules; "
        f"got: {msg!r}"
    )
    assert "#91" in msg, (
        f"Warning message must reference issue #91 for the full "
        f"analysis; got: {msg!r}"
    )

    # 4. The label was still applied (non-blocking).
    assert cluster_id in learner.cluster_labels, (
        f"Cluster {cluster_id} must be labelled despite the warning "
        f"(non-blocking contract)."
    )
    assert learner.cluster_labels[cluster_id] == RelationType.CAUSAL


def test_label_clusters_silent_when_graph_has_existing_edges_false():
    """``label_clusters(graph_has_existing_edges=False)`` is silent.

    Default behavior — the safe case where labelling happens before
    any edges are encoded. No warning should fire.
    """
    learner = _build_minimally_trained_pcl()
    cluster_id = next(iter(learner.action_clusters.keys()))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.label_clusters(
            {cluster_id: RelationType.CAUSAL},
            graph_has_existing_edges=False,
        )

    # Filter out any warnings unrelated to label_clusters (e.g.
    # DeprecationWarning from other modules that fire during the call).
    relevant = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(relevant) == 0, (
        f"label_clusters(graph_has_existing_edges=False) must NOT emit "
        f"any RuntimeWarning; got: {[str(w.message) for w in relevant]}"
    )

    # Label was applied.
    assert learner.cluster_labels.get(cluster_id) == RelationType.CAUSAL


def test_label_clusters_default_flag_value_is_false():
    """Default ``graph_has_existing_edges`` is False (backward compat).

    Existing callers of ``label_clusters(mapping)`` (without the new
    flag) must continue to work silently. This guards against
    accidentally defaulting the flag to True.
    """
    learner = _build_minimally_trained_pcl()
    cluster_id = next(iter(learner.action_clusters.keys()))

    # Call without the flag — should be silent.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.label_clusters({cluster_id: RelationType.CAUSAL})

    relevant = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(relevant) == 0, (
        f"label_clusters() without the flag must be silent (default "
        f"False); got RuntimeWarnings: {[str(w.message) for w in relevant]}"
    )

    # Label was applied.
    assert learner.cluster_labels.get(cluster_id) == RelationType.CAUSAL


def test_label_clusters_warning_stacklevel_points_at_caller():
    """Warning ``stacklevel=2`` so it points at the caller, not internals.

    Without correct stacklevel, the warning's filename:lineno would
    point at the ``warnings.warn()`` call inside ``label_clusters()``,
    which is useless for debugging. With ``stacklevel=2``, it points
    at the caller of ``label_clusters()``.
    """
    learner = _build_minimally_trained_pcl()
    cluster_id = next(iter(learner.action_clusters.keys()))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.label_clusters(
            {cluster_id: RelationType.CAUSAL},
            graph_has_existing_edges=True,
        )

    assert len(caught) >= 1
    w = caught[0]
    # The warning's filename should be THIS test file, not the
    # positional_cluster_learner.py module.
    assert w.filename == __file__, (
        f"Warning stacklevel must point at the caller (this test file, "
        f"{__file__}); got filename={w.filename!r}. "
        f"Check that stacklevel=2 is set in the warnings.warn() call."
    )


def test_label_clusters_warning_can_be_suppressed():
    """Researchers can suppress the warning with warnings.filterwarnings.

    Legitimate A/B experiment use case: researcher knows the risk
    and wants to flip labels mid-session without warning noise. They
    should be able to suppress the warning with a standard
    ``warnings.filterwarnings('ignore', RuntimeWarning)`` scoped to
    the call.
    """
    learner = _build_minimally_trained_pcl()
    cluster_id = next(iter(learner.action_clusters.keys()))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", RuntimeWarning)
        learner.label_clusters(
            {cluster_id: RelationType.CAUSAL},
            graph_has_existing_edges=True,
        )

    # No warning surfaced (suppressed).
    runtime_warnings = [
        w for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    assert len(runtime_warnings) == 0, (
        f"RuntimeWarning should have been suppressed; got: "
        f"{[str(w.message) for w in runtime_warnings]}"
    )

    # Label was still applied.
    assert learner.cluster_labels.get(cluster_id) == RelationType.CAUSAL


def test_label_clusters_idempotent_with_warning():
    """Calling label_clusters twice with the flag both times warns twice.

    Each call is independent — the warning is not sticky. This
    matches the contract: every mid-session re-labelling attempt is
    a separate risk event.
    """
    learner = _build_minimally_trained_pcl()
    cluster_id = next(iter(learner.action_clusters.keys()))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.label_clusters(
            {cluster_id: RelationType.CAUSAL},
            graph_has_existing_edges=True,
        )
        learner.label_clusters(
            {cluster_id: RelationType.CATEGORICAL},
            graph_has_existing_edges=True,
        )

    runtime_warnings = [
        w for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    assert len(runtime_warnings) == 2, (
        f"Two label_clusters calls with the flag must emit two "
        f"warnings; got {len(runtime_warnings)}."
    )

    # Last label wins (idempotent overwrite).
    assert learner.cluster_labels.get(cluster_id) == RelationType.CATEGORICAL


# ======================================================================
# Sanity: existing label_clusters callers (no flag) still work
# ======================================================================


def test_bootstrap_classifier_still_labels_without_warning():
    """``bootstrap_classifier.build_labelled_cluster_learner()`` is silent.

    The bootstrap path calls ``label_clusters(mapping)`` without the
    new flag (it always labels BEFORE any edges are encoded, so the
    flag is correctly False). This test verifies no warning leaks
    out of the bootstrap path.
    """
    from neocortex.bootstrap_classifier import (
        DEFAULT_CORPUS_PATHS,
        build_labelled_cluster_learner,
    )

    # Skip if corpus files are missing (partial checkout).
    missing = [p for p in DEFAULT_CORPUS_PATHS if not os.path.exists(p)]
    if missing:
        pytest.skip(f"Canonical corpus files missing: {missing}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner = build_labelled_cluster_learner()

    # No RuntimeWarning from label_clusters.
    label_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "label_clusters" in str(w.message)
    ]
    assert len(label_warnings) == 0, (
        f"Bootstrap path must not trigger label_clusters warning "
        f"(it labels before any edges exist); got: "
        f"{[str(w.message) for w in label_warnings]}"
    )
    assert learner.is_labelled
    assert len(learner.cluster_labels) == 5
