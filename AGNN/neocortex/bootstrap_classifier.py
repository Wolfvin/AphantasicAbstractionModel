"""
BOOTSTRAP CLASSIFIER - build a labelled PositionalClusterLearner from corpus.

The PositionalClusterLearner (see ``positional_cluster_learner.py``)
discovers action clusters from raw text via positional co-occurrence,
with zero human-authored seeds. After training, clusters are unnamed
integer IDs - a human (or this bootstrap module) must inspect them and
assign RelationType labels.

This module automates that labelling step. It is the bridge between
zero-bias discovery and a usable classifier:

    1. train() on the combined corpus
       (AGNN/data/pretrain_corpus.txt + pretrain_corpus_depth.txt)
    2. inspect_cluster_details() to find the cluster_id for each of the
       five known verb groups (CAUSAL, FUNCTIONAL, CATEGORICAL, TEMPORAL,
       DIFFERENTIAL).
    3. Verify - by exact verb-set match - that each expected group maps
       to exactly one cluster whose action set is a *superset* of the
       expected verbs. Raise ``RuntimeError`` if any group is missing
       or split - never silently label the wrong cluster.
    4. label_clusters() with the discovered mapping.
    5. Return the labelled learner.

The expected verb groups are pinned in :data:`EXPECTED_VERB_GROUPS` -
they are the result of manual inspection of the cluster output on the
canonical corpus (see the PR that introduced this module for the
verification log). If the corpus changes such that the clusters
re-fragment or re-merge differently, this module raises instead of
producing a mis-labelled learner.

Public API
----------
    build_labelled_cluster_learner(corpus_paths) -> PositionalClusterLearner
        Train + label + return. Raises on cluster mismatch.

    EXPECTED_VERB_GROUPS: Dict[RelationType, Set[str]]
        The canonical mapping. Read-only - tests assert against this
        to detect drift.

    DEFAULT_CORPUS_PATHS: List[str]
        The canonical corpus files used to produce
        ``AGNN/data/cluster_learner_state.json``.

    DEFAULT_STATE_PATH: str
        Where the labelled learner state is persisted.

    save_default_state(corpus_paths, state_path) -> PositionalClusterLearner
        Convenience: build + save in one call. Used by the one-shot
        generator script and by tests that need a fresh state file.

State-file regeneration contract
---------------------------------
``AGNN/data/cluster_learner_state.json`` is a **build artifact**, not
source code. It MUST be regenerated whenever the
``PositionalClusterLearner`` clustering algorithm changes — including
(but not limited to):

  * any change to ``_cluster_actions``, ``_weighted_jaccard``, the
    connector-signal partition, or the anchor-word discovery logic;
  * any change to the Brown-clustering configuration for objects;
  * any change to ``DEFAULT_CORPUS_PATHS`` or the contents of the
    canonical corpus files;
  * any change to ``EXPECTED_VERB_GROUPS`` (which the labelling step
    matches against).

Regeneration procedure::

    cd AGNN/
    python -m neocortex.bootstrap_classifier

This rebuilds the learner from the canonical corpus and overwrites
``DEFAULT_STATE_PATH`` in place. Commit the regenerated file alongside
the algorithm change that necessitated it.

The regression test
``test_committed_state_file_matches_fresh_build`` (in
``AGNN/tests/test_bootstrap_classifier.py``) enforces this contract:
it compares the committed state file against a fresh build and fails
on any token-set drift. PR #81 (anchor-word discovery + Brown
clustering for objects) was the original case where this contract was
violated — the state file went stale and silently shipped different
cluster IDs in production vs test (see issue #92).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Make sibling AGNN subpackages importable when this module is loaded
# as ``neocortex.bootstrap_classifier`` (the normal case inside AGNN/).
_AGNN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGNN_ROOT not in sys.path:
    sys.path.insert(0, _AGNN_ROOT)

from neocortex.positional_cluster_learner import PositionalClusterLearner  # noqa: E402
from neocortex.semantic_role_classifier import RelationType  # noqa: E402


# ----------------------------------------------------------------------
# Canonical verb groups (the "ground truth" the bootstrap matches against)
# ----------------------------------------------------------------------
#
# Each group lists the verbs that MUST all be in the same cluster after
# training on the canonical corpus. The cluster may contain additional
# verbs (superset match) - that's fine, the bootstrap only requires
# that every verb in the expected set lands in one cluster together.
#
# If the corpus changes such that:
#   - any of these verbs ends up unclustered (cluster_id = -1), OR
#   - the verbs split across multiple clusters, OR
#   - a verb from one group mixes into another group's cluster
# then build_labelled_cluster_learner() raises RuntimeError.
#
# These verb sets came from manual inspection of
# inspect_cluster_details() output on the combined corpus
# (pretrain_corpus.txt + pretrain_corpus_depth.txt, 3290 sentences).
# Cluster IDs in the canonical run post-PR #81 are 42 (CAUSAL),
# 56 (FUNCTIONAL), 61 (CATEGORICAL), 110 (TEMPORAL),
# 111 (DIFFERENTIAL) - but we DO NOT hardcode those IDs. We look up
# the cluster_id fresh on every call by matching the action set,
# because cluster IDs shift if the corpus or clustering algorithm
# changes. (Pre-PR #81 the IDs were 42/57/60/98/124 - issue #92.)
EXPECTED_VERB_GROUPS: Dict[RelationType, Set[str]] = {
    RelationType.CAUSAL: {
        "berakibat", "membuat", "memicu", "mengakibatkan",
        "menghasilkan", "menyebabkan",
    },
    RelationType.FUNCTIONAL: {
        "membutuhkan", "memerlukan", "perlu", "tergantung",
    },
    RelationType.CATEGORICAL: {
        "adalah", "merupakan", "termasuk",
    },
    RelationType.TEMPORAL: {
        "kemudian", "ketika", "lalu", "saat", "sebelum", "setelah",
    },
    RelationType.DIFFERENTIAL: {
        "berbeda", "berlawanan",
    },
}

# Clause-coordinator tokens (round 5 of the BOS training sprint).
# "dan"/"atau" join two independent clauses ("X melakukan A dan Y
# melakukan B") — they are action-bucket-anchored (true, discovered),
# but are NOT a semantic relation between an agent and a patient like
# RelationType represents; they're a structural/syntactic operator.
# Matched by content (same robust pattern as EXPECTED_VERB_GROUPS)
# AFTER zero-bias clustering, then marked via
# PositionalClusterLearner.mark_clause_coordinator_clusters() — never
# consulted during clustering itself.
EXPECTED_COORDINATOR_TOKENS: Set[str] = {"dan", "atau"}


# ----------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------
#
# The canonical corpus is the two pretrain files shipped in AGNN/data/.
# Together they are 3290 sentences and produce the 5 clean clusters
# listed in EXPECTED_VERB_GROUPS. Tests that need to exercise the
# bootstrap end-to-end use these paths.

# AGNN/data/ - resolved relative to this module so the paths work
# regardless of the current working directory.
_DATA_DIR = os.path.join(_AGNN_ROOT, "data")

DEFAULT_CORPUS_PATHS: List[str] = [
    os.path.join(_DATA_DIR, "pretrain_corpus.txt"),
    os.path.join(_DATA_DIR, "pretrain_corpus_depth.txt"),
    os.path.join(_DATA_DIR, "pretrain_corpus_passive.txt"),
    os.path.join(_DATA_DIR, "pretrain_corpus_ditransitive.txt"),
    os.path.join(_DATA_DIR, "pretrain_corpus_subordinate.txt"),
    os.path.join(_DATA_DIR, "pretrain_corpus_coordinate.txt"),
    os.path.join(_DATA_DIR, "pretrain_corpus_pronoun.txt"),
]

# Where the labelled learner state is persisted. AGNNCore loads this
# file at init time when use_cluster_learner=True (the default).
DEFAULT_STATE_PATH: str = os.path.join(
    _DATA_DIR, "cluster_learner_state.json"
)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def build_labelled_cluster_learner(
    corpus_paths: Optional[List[str]] = None,
) -> PositionalClusterLearner:
    """Train + label a PositionalClusterLearner from corpus files.

    Pipeline:
        1. Load + concatenate all corpus files (skipping blank lines
           and ``##``-prefixed comments).
        2. ``train()`` the learner on the combined corpus.
        3. ``inspect_cluster_details()`` to enumerate every cluster's
           action set.
        4. For each entry in :data:`EXPECTED_VERB_GROUPS`, find the
           cluster whose action set is a superset of the expected
           verbs. Exactly one such cluster must exist; otherwise raise.
        5. ``label_clusters()`` with the discovered
           ``{cluster_id: RelationType}`` mapping.
        6. Return the labelled learner.

    Args:
        corpus_paths: List of corpus file paths. ``None`` (the
            default) uses :data:`DEFAULT_CORPUS_PATHS` - the two
            canonical pretrain files shipped in ``AGNN/data/``.

    Returns:
        A trained AND labelled ``PositionalClusterLearner``. Calling
        ``classify()`` on it returns the labelled RelationType for any
        action whose cluster was labelled; other actions fall back to
        the wrapped SemanticRoleClassifier (no behaviour change).

    Raises:
        FileNotFoundError: if any path in ``corpus_paths`` does not
            exist.
        RuntimeError: if any expected verb group cannot be matched to
            exactly one cluster. The error message lists which group
            failed and why (verb unclustered, split across clusters,
            or mixed into another group). This is the "no silent
            fallback" contract: a corpus change that breaks the
            expected clustering must be visible, not papered over.
    """
    if corpus_paths is None:
        corpus_paths = DEFAULT_CORPUS_PATHS

    # 1. Load + concat corpus.
    lines: List[str] = []
    for path in corpus_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {path}"
            )
        text = p.read_text(encoding="utf-8")
        for ln in text.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("##"):
                lines.append(ln)

    if not lines:
        raise RuntimeError(
            "No corpus lines loaded from "
            f"{corpus_paths} - cannot train cluster learner"
        )

    # 2. Train.
    learner = PositionalClusterLearner()
    learner.train(lines)

    # 3. Inspect + build action -> cluster_id reverse map.
    details = learner.inspect_cluster_details()
    action_to_cluster: Dict[str, int] = {}
    for cid, info in details.items():
        for action in info["actions"]:
            action_to_cluster[action] = cid

    # 4. Find cluster_id for each expected verb group.
    #
    # Match policy: for each group, every expected verb MUST map to
    # the SAME cluster_id (and that cluster's action set must be a
    # superset of the expected verbs). If the verbs split across
    # clusters, that's a corpus drift signal - raise, don't guess.
    mapping: Dict[int, RelationType] = {}
    failures: List[str] = []

    for relation_type, expected_verbs in EXPECTED_VERB_GROUPS.items():
        # Step 4a: every expected verb must be clustered (not -1).
        unclustered = [
            v for v in expected_verbs
            if v not in action_to_cluster
        ]
        if unclustered:
            failures.append(
                f"{relation_type.name}: verbs not clustered "
                f"(cluster_id = -1 or absent): {sorted(unclustered)}"
            )
            continue

        # Step 4b: all expected verbs must land in the SAME cluster.
        cluster_ids = {action_to_cluster[v] for v in expected_verbs}
        if len(cluster_ids) != 1:
            # Verbs split across clusters - corpus drift.
            per_verb = {
                v: action_to_cluster[v] for v in sorted(expected_verbs)
            }
            failures.append(
                f"{relation_type.name}: expected verbs split across "
                f"clusters {sorted(cluster_ids)}. Per-verb: {per_verb}"
            )
            continue

        # Step 4c: verify the matched cluster is a superset of the
        # expected verbs. This catches the case where the cluster
        # picked up extra verbs from another relation type (e.g.
        # "adalah" and "menyebabkan" both ended up in the same cluster
        # due to corpus drift).
        cid = next(iter(cluster_ids))
        cluster_actions = set(details[cid]["actions"])

        # Cross-check: ensure no verb from ANOTHER expected group has
        # leaked into this cluster. If it has, the clusters are mixing
        # - corpus drift, raise.
        other_expected: Set[str] = set()
        for other_rt, other_verbs in EXPECTED_VERB_GROUPS.items():
            if other_rt is not relation_type:
                other_expected |= other_verbs
        leaked = cluster_actions & other_expected
        if leaked:
            failures.append(
                f"{relation_type.name}: cluster {cid} contains verbs "
                f"from another expected group: {sorted(leaked)}. "
                f"Full cluster actions: {sorted(cluster_actions)}"
            )
            continue

        # All checks passed - record the mapping.
        mapping[cid] = relation_type

    if failures:
        # Build a comprehensive error message so the operator can see
        # exactly which group(s) failed and why. Include the full
        # cluster details for debugging.
        msg_lines = [
            "bootstrap_classifier: cluster verification FAILED.",
            "",
            "The corpus did not produce the expected 5 action clusters.",
            "This usually means the corpus changed and the clustering",
            "re-fragmented or re-merged differently. Refusing to label",
            "clusters silently - manual inspection required.",
            "",
            "Failures:",
        ]
        for f in failures:
            msg_lines.append(f"  - {f}")
        msg_lines.append("")
        msg_lines.append("All discovered clusters (for debugging):")
        for cid in sorted(details.keys()):
            info = details[cid]
            msg_lines.append(
                f"  cluster {cid}: actions={info['actions']}, "
                f"top_objects={info['top_objects'][:5]}"
            )
        raise RuntimeError("\n".join(msg_lines))

    # 5. Label the clusters.
    learner.label_clusters(mapping)

    # 5b. Mark clause-coordinator clusters (round 5). Unlike
    # EXPECTED_VERB_GROUPS, coordinators are NOT required to land in
    # the SAME cluster as each other — "dan" and "atau" have different
    # "object" distributions (whatever starts the second clause after
    # an "and" vs an "or") and may legitimately form separate
    # clusters. We mark whichever cluster(s) each coordinator token
    # ended up in. Missing coordinator tokens are non-fatal (warn,
    # don't raise) — coordinator-cluster detection is an additive
    # refinement on top of the core 5-RelationType contract, not part
    # of it; a corpus that doesn't yet have "dan"/"atau" depth simply
    # doesn't get this refinement, the same graceful-degradation
    # contract as the rest of the zero-bias pipeline.
    coordinator_cluster_ids: Set[int] = set()
    missing_coordinators: List[str] = []
    for token in EXPECTED_COORDINATOR_TOKENS:
        cid = action_to_cluster.get(token)
        if cid is None:
            missing_coordinators.append(token)
            continue
        coordinator_cluster_ids.add(cid)
    if coordinator_cluster_ids:
        learner.mark_clause_coordinator_clusters(coordinator_cluster_ids)

    # 6. Return the labelled learner.
    return learner


def save_default_state(
    corpus_paths: Optional[List[str]] = None,
    state_path: Optional[str] = None,
) -> PositionalClusterLearner:
    """Build the labelled learner and persist it to ``state_path``.

    Convenience wrapper: calls :func:`build_labelled_cluster_learner`
    and then ``learner.save(state_path)``. Used by the one-shot
    generator script and by tests that need a fresh state file.

    Args:
        corpus_paths: See :func:`build_labelled_cluster_learner`.
            ``None`` uses the defaults.
        state_path: Where to write the JSON state. ``None`` (the
            default) uses :data:`DEFAULT_STATE_PATH`.

    Returns:
        The labelled learner (also saved to ``state_path``).
    """
    if state_path is None:
        state_path = DEFAULT_STATE_PATH
    learner = build_labelled_cluster_learner(corpus_paths)
    learner.save(state_path)
    return learner


def load_default_state(
    state_path: Optional[str] = None,
) -> Optional[PositionalClusterLearner]:
    """Load the labelled learner from ``state_path``, or None if absent.

    Used by AGNNCore at init time when ``use_cluster_learner=True``
    (the default). Returns ``None`` if the state file does not exist
    or fails to load - the caller then falls back to a fresh
    SemanticRoleClassifier (matching the pre-cluster-learner behaviour).

    Args:
        state_path: Where to load from. ``None`` (the default) uses
            :data:`DEFAULT_STATE_PATH`.

    Returns:
        The loaded learner, or ``None`` if the file is missing or
        corrupt. Never raises - a missing/corrupt state file is a
        graceful-degradation signal, not a crash.
    """
    if state_path is None:
        state_path = DEFAULT_STATE_PATH
    if not os.path.exists(state_path):
        return None
    try:
        return PositionalClusterLearner.load(state_path)
    except Exception:
        # Corrupt JSON / schema mismatch / future-version file.
        # Graceful degradation: return None so AGNNCore falls back
        # to SemanticRoleClassifier. The operator can regenerate
        # the state file via save_default_state().
        return None


# ----------------------------------------------------------------------
# CLI entry point (for regenerating the state file)
# ----------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    # One-shot generator: train on the canonical corpus, save state.
    print(f"Building labelled cluster learner from canonical corpus...")
    print(f"  corpus paths: {DEFAULT_CORPUS_PATHS}")
    print(f"  state path:   {DEFAULT_STATE_PATH}")
    learner = save_default_state()
    print(f"\nDone. Trained + labelled learner saved.")
    print(f"  total clusters: {len(learner.action_clusters)}")
    print(f"  labelled clusters: {len(learner.cluster_labels)}")
    for cid, rt in sorted(learner.cluster_labels.items()):
        actions = sorted(learner.action_clusters.get(cid, set()))
        print(f"    cluster {cid:3d} -> {rt.name:12s}  actions={actions}")
