"""
Tests for ``neocortex.bootstrap_classifier`` and AGNNCore cluster-learner
integration.

Definition of Done covered:

    1. test_bootstrap_classifier_finds_all_5_clusters  - build_labelled
       _cluster_learner() on the canonical corpus discovers all 5
       RelationType cluster IDs (matched by verb-set, not by index).
    2. test_bootstrap_classifier_raises_on_corpus_mismatch - a fake
       corpus that does not produce the expected 5 clusters raises
       RuntimeError (no silent fallback).
    3. test_agnncore_uses_cluster_learner_by_default - AGNNCore's
       default construction loads the labelled PositionalClusterLearner
       and brain.learn() with a CAUSAL correction stores edge_type=CAUSAL
       in the graph (NOT from the SemanticRoleClassifier fallback).
    4. test_agnncore_fallback_when_disabled - use_cluster_learner=False
       produces behaviour identical to the pre-cluster-learner AGNNCore
       (role_classifier is a fresh SemanticRoleClassifier, no
       PositionalClusterLearner loaded).

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_bootstrap_classifier.py -v
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
# name collisions - same pattern as test_core_wired.py.
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))


# ----------------------------------------------------------------------
# Imports under test
# ----------------------------------------------------------------------

from neocortex.bootstrap_classifier import (  # noqa: E402
    DEFAULT_CORPUS_PATHS,
    DEFAULT_STATE_PATH,
    EXPECTED_VERB_GROUPS,
    build_labelled_cluster_learner,
    load_default_state,
    save_default_state,
)
from neocortex.positional_cluster_learner import (  # noqa: E402
    PositionalClusterLearner,
)
from neocortex.semantic_role_classifier import (  # noqa: E402
    RelationType,
    SemanticRoleClassifier,
)

# Load AGNN/core.py by path to avoid name collision with
# self-ai/src/core/ (same pattern as test_core_wired.py and
# test_frequency_table_persistence.py).
import importlib.util as _ilu  # noqa: E402

_core_path = _AGNP_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_bootstrap", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_bootstrap"] = agnn_core_module
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore


# ----------------------------------------------------------------------
# Skip-if-missing helpers
# ----------------------------------------------------------------------

def _require_corpus_files():
    """Skip the test if the canonical corpus files are not present.

    The bootstrap tests need pretrain_corpus.txt + pretrain_corpus_depth.txt
    to be on disk so they can train the learner from scratch. In
    environments that only check out AGNN/tests/ (e.g. partial CI),
    these files may be absent - skip rather than fail.
    """
    missing = [p for p in DEFAULT_CORPUS_PATHS if not os.path.exists(p)]
    if missing:
        pytest.skip(
            f"Canonical corpus files missing: {missing} - bootstrap "
            f"tests require the full AGNN/data/ directory."
        )


def _require_self_ai_graph():
    """Skip the test if self-ai/src/agnn/graph.py is not importable.

    AGNNCore's learn() pipeline needs EngramComplex, which wraps
    AGNNGraph. Without it, learn() returns a fallback dict and we
    cannot verify the edge_type stored in the graph.
    """
    if not _SELF_AI_SRC.exists():
        pytest.skip(
            "self-ai/src/agnn/graph.py not available - AGNNCore.learn() "
            "requires EngramComplex (which wraps AGNNGraph)."
        )


# ======================================================================
# DoD #1: bootstrap finds all 5 clusters on the canonical corpus
# ======================================================================


def test_bootstrap_classifier_finds_all_5_clusters():
    """build_labelled_cluster_learner() discovers all 5 RelationType clusters.

    The canonical corpus (pretrain_corpus.txt + pretrain_corpus_depth.txt,
    3290 sentences) produces 5 clean action clusters whose verb sets
    match the EXPECTED_VERB_GROUPS mapping. This test verifies that
    build_labelled_cluster_learner() successfully:

      1. Trains the learner on the combined corpus.
      2. Finds the cluster_id for each of the 5 RelationTypes by
         matching the action set (not by hardcoded index).
      3. Labels the clusters with the correct RelationType.
      4. Returns a learner whose classify() returns the correct
         RelationType for each pattern.
    """
    _require_corpus_files()

    learner = build_labelled_cluster_learner()

    # 1. The learner must be trained AND labelled.
    assert learner.is_trained, "learner must be trained after bootstrap"
    assert learner.is_labelled, "learner must be labelled after bootstrap"

    # 2. All 5 RelationTypes must be present in the cluster_labels.
    labelled_types = set(learner.cluster_labels.values())
    expected_types = set(EXPECTED_VERB_GROUPS.keys())
    assert labelled_types == expected_types, (
        f"Bootstrap must label all 5 RelationTypes. "
        f"Expected: {sorted(rt.name for rt in expected_types)}, "
        f"got: {sorted(rt.name for rt in labelled_types)}"
    )

    # 3. For each RelationType, verify the cluster's action set is a
    #    superset of the expected verbs (the cluster may have picked
    #    up extra verbs from corpus drift, but the expected verbs must
    #    all be there).
    for relation_type, expected_verbs in EXPECTED_VERB_GROUPS.items():
        # Find the cluster_id labelled with this RelationType.
        cid = None
        for cluster_id, rt in learner.cluster_labels.items():
            if rt is relation_type:
                cid = cluster_id
                break
        assert cid is not None, (
            f"No cluster labelled {relation_type.name} - this should "
            f"have been caught by the labelled_types check above, but "
            f"sanity-checking anyway."
        )
        cluster_actions = set(learner.action_clusters[cid])
        missing = expected_verbs - cluster_actions
        assert not missing, (
            f"Cluster {cid} (labelled {relation_type.name}) is missing "
            f"expected verbs: {sorted(missing)}. "
            f"Cluster actions: {sorted(cluster_actions)}"
        )

    # 4. classify() returns the correct RelationType for representative
    #    sentences of each pattern.
    test_cases = [
        ("api menyebabkan kebakaran", RelationType.CAUSAL),
        ("manusia membutuhkan air", RelationType.FUNCTIONAL),
        ("anjing adalah mamalia", RelationType.CATEGORICAL),
        ("setelah hujan, jalanan basah", RelationType.TEMPORAL),
        ("tomat bukan sayuran", RelationType.DIFFERENTIAL),
    ]
    for text, expected in test_cases:
        result = learner.classify(text)
        assert result == expected, (
            f"classify({text!r}) returned {result}, expected {expected}"
        )


def test_bootstrap_classifier_uses_verb_set_match_not_index():
    """The bootstrap matches clusters by action set, not by hardcoded ID.

    This is the "no silent fallback" contract: cluster IDs shift if the
    corpus or clustering algorithm changes. The bootstrap must look up
    the cluster_id fresh on every call by matching the action set, and
    must NOT hardcode IDs like 42, 57, 60, 98, 124.

    We verify this by training on a *subset* of the canonical corpus
    (just pretrain_corpus_depth.txt) and checking that the bootstrap
    still finds the 5 clusters even though their IDs will differ from
    the canonical run.
    """
    _require_corpus_files()

    # Train on just the depth corpus - this is a different input than
    # the default (which concatenates both files), so cluster IDs WILL
    # differ from the canonical 42/57/60/98/124.
    depth_only = [
        p for p in DEFAULT_CORPUS_PATHS
        if p.endswith("pretrain_corpus_depth.txt")
    ]
    assert len(depth_only) == 1, "depth corpus must be in DEFAULT_CORPUS_PATHS"

    learner = build_labelled_cluster_learner(corpus_paths=depth_only)

    # The bootstrap must still have found 5 labelled clusters.
    assert learner.is_labelled
    assert len(learner.cluster_labels) == 5

    # And classify() must still return correct RelationTypes.
    assert learner.classify("api menyebabkan kebakaran") == RelationType.CAUSAL
    assert learner.classify("anjing adalah mamalia") == RelationType.CATEGORICAL


def test_bootstrap_classifier_state_file_is_loadable():
    """The committed state file (cluster_learner_state.json) loads cleanly.

    This guards against the state file getting out of sync with the
    bootstrap code - if someone regenerates the state file with a
    different corpus or a different version of PositionalClusterLearner,
    load_default_state() must still return a working labelled learner.
    """
    if not os.path.exists(DEFAULT_STATE_PATH):
        pytest.skip(
            f"State file not found: {DEFAULT_STATE_PATH} - run "
            f"`python AGNN/neocortex/bootstrap_classifier.py` to generate."
        )

    learner = load_default_state(DEFAULT_STATE_PATH)
    assert learner is not None, "load_default_state() returned None"
    assert learner.is_trained
    assert learner.is_labelled
    # All 5 RelationTypes must be in the saved state.
    assert len(learner.cluster_labels) == 5

    # And classify() must work.
    assert learner.classify("api menyebabkan kebakaran") == RelationType.CAUSAL


# ======================================================================
# DoD #2: bootstrap raises on corpus mismatch (no silent fallback)
# ======================================================================


def test_bootstrap_classifier_raises_on_corpus_mismatch(tmp_path):
    """A fake corpus that doesn't produce the expected clusters raises.

    The "no silent fallback" contract: if the corpus changes such that
    the 5 expected verb groups cannot all be matched to clusters, the
    bootstrap MUST raise RuntimeError with a clear message. It must
    NOT silently label the wrong cluster or return an unlabelled learner.
    """
    # A corpus that contains NEITHER the expected CATEGORICAL verbs
    # (adalah, merupakan, termasuk) NOR the TEMPORAL markers (setelah,
    # etc). The bootstrap will fail to find those clusters and must
    # raise.
    fake_corpus = [
        # Only SVO sentences with verbs that won't cluster into the
        # expected groups.
        "saya makan nasi",
        "dia makan ayam",
        "kamu minum susu",
        "ibu masak sayur",
        "bapak baca koran",
        "anak tulis surat",
    ]

    fake_path = tmp_path / "fake_corpus.txt"
    fake_path.write_text("\n".join(fake_corpus) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        build_labelled_cluster_learner(corpus_paths=[str(fake_path)])

    # The error message must mention which group(s) failed.
    msg = str(exc_info.value)
    assert "cluster verification FAILED" in msg, (
        f"Error message must mention 'cluster verification FAILED'. "
        f"Got: {msg}"
    )
    # Must mention at least one of the expected RelationTypes that
    # couldn't be found.
    assert "CAUSAL" in msg or "CATEGORICAL" in msg, (
        f"Error message must name the missing RelationType(s). "
        f"Got: {msg}"
    )


def test_bootstrap_classifier_raises_on_split_clusters(tmp_path):
    """If expected verbs split across clusters, bootstrap raises.

    Construct a corpus where the CATEGORICAL verbs (adalah, merupakan,
    termasuk) each end up in their own singleton cluster (because they
    don't share enough objects). The bootstrap must detect this split
    and raise.
    """
    # Craft a corpus where each CATEGORICAL verb appears with totally
    # different objects, so they won't cluster together.
    fake_corpus = [
        # 'adalah' with food objects
        "nasi adalah karbohidrat",
        "roti adalah karbohidrat",
        # 'merupakan' with animal objects
        "anjing merupakan mamalia",
        "kucing merupakan mamalia",
        # 'termasuk' with science objects
        "atom termasuk partikel",
        "molekul termasuk partikel",
    ]

    fake_path = tmp_path / "split_corpus.txt"
    fake_path.write_text("\n".join(fake_corpus) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        build_labelled_cluster_learner(corpus_paths=[str(fake_path)])

    msg = str(exc_info.value)
    # The error must mention CATEGORICAL (the group that split).
    assert "CATEGORICAL" in msg, (
        f"Error must mention CATEGORICAL (the split group). "
        f"Got: {msg}"
    )


# ======================================================================
# DoD #3: AGNNCore uses cluster learner by default
# ======================================================================


def test_agnncore_uses_cluster_learner_by_default():
    """AGNNCore() with no args loads the labelled PositionalClusterLearner.

    The default use_cluster_learner=True must:
      1. Load the cluster learner state from DEFAULT_STATE_PATH.
      2. Pass it to TrisynapticCircuit as role_classifier.
      3. Make brain.learn() with a CAUSAL correction store edge_type=CAUSAL
         in the graph, sourced from the cluster learner (NOT from the
         SemanticRoleClassifier fallback).
    """
    _require_self_ai_graph()
    if not os.path.exists(DEFAULT_STATE_PATH):
        pytest.skip(
            f"State file not found: {DEFAULT_STATE_PATH} - run "
            f"`python AGNN/neocortex/bootstrap_classifier.py` to generate."
        )

    core = AGNNCore(model_path=None)

    # Pre-condition: graph + trisynaptic must be available.
    if core.graph is None or core.trisynaptic is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    # 1. The cluster learner was loaded.
    assert core._cluster_learner is not None, (
        "AGNNCore(use_cluster_learner=True) must load the cluster learner"
    )
    assert isinstance(core._cluster_learner, PositionalClusterLearner)
    assert core._cluster_learner.is_labelled, (
        "Loaded cluster learner must be labelled (state file must "
        "contain cluster_labels)"
    )

    # 2. The cluster learner is wired as TrisynapticCircuit's role_classifier.
    role_classifier = core.trisynaptic.role_classifier
    assert isinstance(role_classifier, PositionalClusterLearner), (
        f"TrisynapticCircuit.role_classifier must be a "
        f"PositionalClusterLearner when use_cluster_learner=True, "
        f"got {type(role_classifier).__name__}"
    )
    assert role_classifier is core._cluster_learner, (
        "TrisynapticCircuit.role_classifier must be the SAME instance "
        "as core._cluster_learner (no copy)"
    )

    # 3. brain.learn() with a CAUSAL correction stores edge_type=CAUSAL
    #    in the graph, sourced from the cluster learner.
    result = core.learn(
        question="test",
        wrong="x",
        correction="api menyebabkan kebakaran",
    )
    assert result["node_id"] is not None, "learn() must return a node_id"

    # Inspect the stored episome.
    episome = core._episomes[-1]
    assert episome.edge_type == "CAUSAL", (
        f"episome.edge_type must be 'CAUSAL' (from cluster learner), "
        f"got {episome.edge_type!r}"
    )

    # Verify the edge_type is also in the graph node metadata.
    graph = core.graph._graph
    node = graph.get_node(str(episome.id))
    assert node is not None, "Episome must be registered as a graph node"
    assert node.metadata.get("edge_type") == "CAUSAL", (
        f"Graph node metadata edge_type must be 'CAUSAL', "
        f"got {node.metadata.get('edge_type')!r}"
    )


def test_agnncore_cluster_learner_classifies_correctly():
    """AGNNCore with cluster learner classifies all 5 relation types correctly.

    This is a broader version of test_agnncore_uses_cluster_learner_by
    _default: it learns one fact per RelationType and verifies the
    edge_type stored matches.
    """
    _require_self_ai_graph()
    if not os.path.exists(DEFAULT_STATE_PATH):
        pytest.skip(
            f"State file not found: {DEFAULT_STATE_PATH} - run "
            f"`python AGNN/neocortex/bootstrap_classifier.py` to generate."
        )

    core = AGNNCore(model_path=None)
    if core.graph is None or core.trisynaptic is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    test_cases = [
        ("api menyebabkan kebakaran", "CAUSAL"),
        ("manusia membutuhkan air", "FUNCTIONAL"),
        ("anjing adalah mamalia", "CATEGORICAL"),
        ("setelah hujan, jalanan basah", "TEMPORAL"),
        ("tomat bukan sayuran", "DIFFERENTIAL"),
    ]

    for correction, expected_edge_type in test_cases:
        result = core.learn(question="q", wrong="w", correction=correction)
        assert result["node_id"] is not None, (
            f"learn() failed for {correction!r}"
        )
        episome = core._episomes[-1]
        assert episome.edge_type == expected_edge_type, (
            f"learn({correction!r}) -> edge_type={episome.edge_type!r}, "
            f"expected {expected_edge_type!r}"
        )


# ======================================================================
# DoD #4: AGNNCore falls back when use_cluster_learner=False
# ======================================================================


def test_agnncore_fallback_when_disabled():
    """use_cluster_learner=False produces behaviour identical to pre-cluster-learner.

    When use_cluster_learner=False:
      1. No PositionalClusterLearner is loaded (core._cluster_learner is None).
      2. TrisynapticCircuit.role_classifier is a fresh SemanticRoleClassifier
         (NOT a PositionalClusterLearner).
      3. brain.learn() still works - the SemanticRoleClassifier handles
         classification via its seed keyword table (the legacy path).
    """
    _require_self_ai_graph()

    core = AGNNCore(model_path=None, use_cluster_learner=False)

    # Pre-condition: graph + trisynaptic must be available.
    if core.graph is None or core.trisynaptic is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    # 1. No cluster learner loaded.
    assert core._cluster_learner is None, (
        "AGNNCore(use_cluster_learner=False) must NOT load the cluster learner"
    )

    # 2. role_classifier is a fresh SemanticRoleClassifier (legacy path).
    role_classifier = core.trisynaptic.role_classifier
    assert isinstance(role_classifier, SemanticRoleClassifier), (
        f"role_classifier must be a SemanticRoleClassifier when "
        f"use_cluster_learner=False, got {type(role_classifier).__name__}"
    )
    # Specifically NOT a PositionalClusterLearner.
    assert not isinstance(role_classifier, PositionalClusterLearner), (
        "role_classifier must NOT be a PositionalClusterLearner when "
        "use_cluster_learner=False"
    )

    # 3. brain.learn() still works via the legacy path. The
    #    SemanticRoleClassifier's seed table contains 'menyebabkan' ->
    #    CAUSAL, so this should still classify as CAUSAL (just from a
    #    different code path - the seed table, not the cluster label).
    result = core.learn(
        question="test",
        wrong="x",
        correction="api menyebabkan kebakaran",
    )
    assert result["node_id"] is not None
    episome = core._episomes[-1]
    # The legacy SemanticRoleClassifier's seed table has 'menyebabkan'
    # -> CAUSAL, so this still classifies as CAUSAL. The point is that
    # it came from the seed table, not from a cluster label.
    assert episome.edge_type == "CAUSAL", (
        f"Legacy SemanticRoleClassifier should still classify "
        f"'menyebabkan' as CAUSAL via its seed table. "
        f"Got: {episome.edge_type!r}"
    )


def test_agnncore_fallback_when_state_file_missing(tmp_path):
    """AGNNCore gracefully falls back when the state file is missing.

    When use_cluster_learner=True but the state file path doesn't exist
    (or fails to load), AGNNCore must fall back to the legacy
    SemanticRoleClassifier - never crash. This is the "graceful
    degradation" contract.
    """
    _require_self_ai_graph()

    # Point to a non-existent state file.
    bogus_path = str(tmp_path / "does_not_exist.json")
    core = AGNNCore(
        model_path=None,
        use_cluster_learner=True,
        cluster_learner_state_path=bogus_path,
    )

    if core.graph is None or core.trisynaptic is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    # cluster_learner must be None (file didn't load).
    assert core._cluster_learner is None, (
        "Missing state file must result in cluster_learner=None "
        "(graceful degradation)"
    )

    # role_classifier must be a fresh SemanticRoleClassifier (fallback).
    role_classifier = core.trisynaptic.role_classifier
    assert isinstance(role_classifier, SemanticRoleClassifier)
    assert not isinstance(role_classifier, PositionalClusterLearner)


def test_agnncore_use_cluster_learner_requested_is_recorded():
    """AGNNCore records the caller's use_cluster_learner choice for introspection.

    Tests and downstream code can read core._use_cluster_learner_requested
    to verify which path was requested (regardless of whether the
    cluster learner actually loaded).
    """
    _require_self_ai_graph()

    core_on = AGNNCore(model_path=None, use_cluster_learner=True)
    assert core_on._use_cluster_learner_requested is True

    core_off = AGNNCore(model_path=None, use_cluster_learner=False)
    assert core_off._use_cluster_learner_requested is False


# ======================================================================
# Supplementary: EXPECTED_VERB_GROUPS sanity checks
# ======================================================================


def test_expected_verb_groups_covers_all_5_relation_types():
    """EXPECTED_VERB_GROUPS has an entry for every required RelationType.

    The 5 required types: CAUSAL, FUNCTIONAL, CATEGORICAL, TEMPORAL,
    DIFFERENTIAL. If a new RelationType is added in the future, this
    test will catch whether EXPECTED_VERB_GROUPS was updated.
    """
    required = {
        RelationType.CAUSAL,
        RelationType.FUNCTIONAL,
        RelationType.CATEGORICAL,
        RelationType.TEMPORAL,
        RelationType.DIFFERENTIAL,
    }
    actual = set(EXPECTED_VERB_GROUPS.keys())
    assert required <= actual, (
        f"EXPECTED_VERB_GROUPS must cover all 5 required RelationTypes. "
        f"Missing: {sorted(rt.name for rt in required - actual)}"
    )


def test_expected_verb_groups_are_non_empty():
    """Every entry in EXPECTED_VERB_GROUPS has at least 2 verbs.

    A single-verb group would make the "split detection" meaningless
    (a single verb can't split across clusters). The minimum of 2
    ensures the bootstrap's "all verbs in same cluster" check is
    meaningful.
    """
    for rt, verbs in EXPECTED_VERB_GROUPS.items():
        assert len(verbs) >= 2, (
            f"EXPECTED_VERB_GROUPS[{rt.name}] must have >= 2 verbs, "
            f"got {len(verbs)}: {verbs}"
        )


def test_expected_verb_groups_do_not_overlap():
    """No verb appears in two different RelationType groups.

    If a verb appeared in two groups, the bootstrap's "no cross-group
    leakage" check would always fail (the verb would be in one cluster,
    causing the other group's check to flag it as leaked).
    """
    all_verbs: list[str] = []
    for verbs in EXPECTED_VERB_GROUPS.values():
        all_verbs.extend(verbs)
    # Each verb must appear exactly once.
    seen: set[str] = set()
    for v in all_verbs:
        assert v not in seen, (
            f"Verb {v!r} appears in multiple EXPECTED_VERB_GROUPS - "
            f"this would cause the bootstrap's leakage check to always fail."
        )
        seen.add(v)


# ======================================================================
# Regression: committed state file vs fresh build (issue #92)
# ======================================================================


def test_committed_state_file_matches_fresh_build():
    """The committed ``cluster_learner_state.json`` must match a fresh build.

    Regression test for issue #92: the committed state file drifted
    from the output of :func:`build_labelled_cluster_learner` because
    it was last regenerated before PR #81 (anchor-word discovery +
    Brown clustering for objects) landed. The fresh build produced
    different cluster IDs and a different token set, but every
    individual test passed because:

      * ``test_bootstrap_classifier_finds_all_5_clusters`` only checks
        the *fresh* build, and
      * ``test_bootstrap_classifier_state_file_is_loadable`` only
        checks the *committed* file's loadability + the easy
        ``classify("api menyebabkan kebakaran") == CAUSAL`` assertion.

    No test compared the two until now.

    Contract pinned by this test:

      * The set of action tokens tracked by the committed state file
        must equal the set tracked by a fresh build on the canonical
        corpus. Any drift means the state file is stale — run
        ``python -m neocortex.bootstrap_classifier`` (from the
        ``AGNN/`` directory) to regenerate.
      * Each expected verb group (``EXPECTED_VERB_GROUPS``) must map
        to exactly one cluster in BOTH the fresh build and the
        committed state file. Cluster IDs are allowed to differ
        between the two (they are an implementation detail of the
        clustering algorithm); what matters is that every expected
        verb is clustered in both, and that all verbs for a given
        relation type land in the SAME cluster within each.
    """
    _require_corpus_files()

    if not os.path.exists(DEFAULT_STATE_PATH):
        pytest.skip(
            f"State file not found: {DEFAULT_STATE_PATH} - run "
            f"`python -m neocortex.bootstrap_classifier` (from AGNN/) "
            f"to generate."
        )

    fresh = build_labelled_cluster_learner()
    committed = load_default_state(DEFAULT_STATE_PATH)
    assert committed is not None, (
        "load_default_state() returned None - committed state file is "
        "missing or corrupt. Run "
        "`python -m neocortex.bootstrap_classifier` (from AGNN/) to "
        "regenerate."
    )

    # 1. Token-set equality. This is the strongest invariant: any
    #    change to the clustering algorithm that adds or removes
    #    tracked action tokens will be caught here. The drift in
    #    issue #92 was 236 tokens (119 only-in-committed, 117
    #    only-in-fresh); this assertion makes that kind of drift a
    #    hard test failure.
    fresh_tokens = set(fresh.cluster_id_of.keys())
    committed_tokens = set(committed.cluster_id_of.keys())
    only_in_committed = sorted(committed_tokens - fresh_tokens)
    only_in_fresh = sorted(fresh_tokens - committed_tokens)
    assert fresh_tokens == committed_tokens, (
        f"Committed cluster_learner_state.json is stale (issue #92). "
        f"Token set drifted from a fresh build.\n"
        f"  fresh tokens:     {len(fresh_tokens)}\n"
        f"  committed tokens: {len(committed_tokens)}\n"
        f"  in committed but not fresh ({len(only_in_committed)}): "
        f"{only_in_committed[:15]} (showing 15)\n"
        f"  in fresh but not committed ({len(only_in_fresh)}): "
        f"{only_in_fresh[:15]} (showing 15)\n"
        f"Run `python -m neocortex.bootstrap_classifier` (from AGNN/) "
        f"to regenerate the state file."
    )

    # 2. Each expected verb group maps to exactly one cluster in BOTH
    #    fresh and committed. Cluster IDs themselves are allowed to
    #    differ (the algorithm's exact ID assignment is not part of
    #    the contract — only the grouping is).
    for relation_type, expected_verbs in EXPECTED_VERB_GROUPS.items():
        # Fresh: every expected verb must be clustered (not -1 / absent)
        # and they must all share one cluster ID.
        fresh_cids = {fresh.cluster_id_of[v] for v in expected_verbs}
        assert len(fresh_cids) == 1, (
            f"Fresh build: {relation_type.name} verbs {sorted(expected_verbs)} "
            f"did not land in a single cluster (got {sorted(fresh_cids)}). "
            f"This is a bootstrap_classifier regression - "
            f"test_bootstrap_classifier_finds_all_5_clusters should have caught it."
        )
        # Committed: same invariant.
        committed_cids = {
            committed.cluster_id_of[v] for v in expected_verbs
        }
        assert len(committed_cids) == 1, (
            f"Committed state file: {relation_type.name} verbs "
            f"{sorted(expected_verbs)} did not land in a single cluster "
            f"(got {sorted(committed_cids)}). The committed state file is "
            f"internally inconsistent - regenerate it."
        )
        # And the committed cluster must carry the right label.
        committed_cid = next(iter(committed_cids))
        assert committed.cluster_labels.get(committed_cid) == relation_type, (
            f"Committed state file: cluster {committed_cid} (which contains "
            f"the {relation_type.name} verbs) is labelled "
            f"{committed.cluster_labels.get(committed_cid)} - expected "
            f"{relation_type.name}. Regenerate the state file."
        )

