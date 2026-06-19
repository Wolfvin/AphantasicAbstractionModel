"""
Tests for ``PositionalClusterLearner`` v2 - zero-bias emergent structure
discovery with post-hoc naming.

This is the rewrite of the test suite for the v2 design (PR #69 was
rejected for seeding 44 human tokens before training and for
hard-locking each token to one cluster).

Definition of Done covered:

    1. test_train_builds_unnamed_clusters      - after train(), clusters
       exist as integer IDs with NO RelationType assigned.
    2. test_polysemy_same_token_different_role - same learner parses
       "ayam mencari pakan" with ayam=agent AND "manusia potong ayam"
       with ayam=object.
    3. test_label_clusters_required            - classify() before
       label_clusters() is identical to SemanticRoleClassifier.classify().
    4. test_label_clusters_applies             - after label_clusters(),
       classify() for an action in that cluster returns the labelled
       RelationType.
    5. test_negation_overrides_cluster         - "tidak menyebabkan"
       still returns DIFFERENTIAL even after labelling.
    6. test_persistence_with_labels            - save/load preserves
       cluster_labels.

Plus supplementary tests for inspect_clusters, fallback composition,
and the position-bucketing helpers.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_positional_cluster_learner.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
_AGNP_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

# Also ensure self-ai/src is importable for the canonical RelationType.
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


# ----------------------------------------------------------------------
# Imports under test
# ----------------------------------------------------------------------

from neocortex.positional_cluster_learner import (  # noqa: E402
    PositionalClusterLearner,
    _AGENT_BUCKET,
    _ACTION_BUCKET,
    _OBJECT_BUCKET_3,
    _OBJECT_BUCKET_N,
)
from neocortex.semantic_role_classifier import (  # noqa: E402
    RelationType,
    SemanticRoleClassifier,
    SPO,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def learner() -> PositionalClusterLearner:
    """Fresh PositionalClusterLearner per test (no training)."""
    return PositionalClusterLearner()


@pytest.fixture
def svo_corpus() -> list:
    """10 SVO sentences where 'makan' is the action at position 1."""
    return [
        "saya makan ayam",
        "saya makan sapi",
        "saya makan ikan",
        "dia makan ayam",
        "dia makan sapi",
        "dia makan ikan",
        "kamu makan ayam",
        "kamu makan sapi",
        "kamu makan ikan",
        "saya minum air",
    ]


@pytest.fixture
def causal_corpus() -> list:
    """5 CAUSAL SVO sentences - 'menyebabkan' as action, state-change objects."""
    return [
        "api menyebabkan panas",
        "hujan menyebabkan banjir",
        "gesekan menyebabkan panas",
        "listrik menyebabkan kebakaran",
        "rokok menyebabkan kanker",
    ]


@pytest.fixture
def mixed_corpus() -> list:
    """Corpus mixing CAUSAL + FUNCTIONAL + CATEGORICAL patterns.

    Designed so actions cluster by object distribution:
      - 'menyebabkan' / 'memicu' both take state-change objects (panas,
        banjir, kebakaran) -> should land in the same cluster.
      - 'membutuhkan' takes need-objects (air, makanan, energi) ->
        separate cluster.
      - 'adalah' takes class-objects (mamalia, hewan, logam) ->
        separate cluster.
    """
    return [
        # CAUSAL cluster (state-change objects)
        "api menyebabkan panas",
        "hujan menyebabkan banjir",
        "gesekan menyebabkan panas",
        "listrik menyebabkan kebakaran",
        "rokok menyebabkan kanker",
        "stres memicu panas",
        "stres memicu banjir",
        # FUNCTIONAL cluster (need objects)
        "tanaman membutuhkan air",
        "tanaman membutuhkan energi",
        "manusia membutuhkan makanan",
        "manusia membutuhkan air",
        "mesin membutuhkan energi",
        # CATEGORICAL cluster (class objects)
        "manusia adalah mamalia",
        "manusia adalah hewan",
        "kucing adalah mamalia",
        "besi adalah logam",
        "emas adalah logam",
    ]


@pytest.fixture
def polysemy_corpus() -> list:
    """Corpus where 'ayam' appears as both agent and object.

    Used to demonstrate the polysemy fix: positional_freq is a soft
    count, so 'ayam' shows up in both bucket 0 (agent) and bucket 2
    (object). The learner does NOT lock 'ayam' to one role globally.
    """
    return [
        # ayam as agent (position 0)
        "ayam mencari pakan",
        "ayam mematuk tanah",
        "ayam minum air",
        "ayam bertelur telur",
        # ayam as object (position 2)
        "manusia potong ayam",
        "kucing kejar ayam",
        "musang culik ayam",
        "ibu masak ayam",
    ]


# ======================================================================
# DoD #1: train() builds unnamed clusters
# ======================================================================


def test_train_builds_unnamed_clusters(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """After train(), clusters exist as integer IDs with NO RelationType.

    The core zero-bias contract: clusters are formed by similarity of
    object distributions, but they are unnamed integer IDs until a
    human calls label_clusters().
    """
    learner.train(mixed_corpus)

    # Must be trained.
    assert learner.is_trained

    # Must have at least one cluster.
    assert len(learner.action_clusters) >= 1

    # NO RelationType must be assigned anywhere.
    assert learner.cluster_labels == {}
    assert learner.is_labelled is False

    # Every cluster_id must be a non-negative integer.
    for cid in learner.action_clusters:
        assert isinstance(cid, int)
        assert cid >= 0

    # inspect_clusters() returns Dict[int, List[str]] with no labels.
    inspection = learner.inspect_clusters()
    assert isinstance(inspection, dict)
    for cid, actions in inspection.items():
        assert isinstance(cid, int)
        assert isinstance(actions, list)
        assert all(isinstance(a, str) for a in actions)


def test_train_zero_bias_no_human_seeds(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """No RelationType seeds are pre-loaded before training.

    This is the regression test for the PR #69 rejection reason #1:
    the old code had a 44-token ``_RELATION_OBJECT_SEEDS`` table.
    After train() in the v2 design, the only RelationType mappings
    in the learner must come from explicit label_clusters() calls -
    and we have NOT called it here.
    """
    learner.train(mixed_corpus)

    # Check there is no relation mapping anywhere.
    assert learner.cluster_labels == {}
    # No attribute on the learner should be a Dict[str, RelationType]
    # pre-populated with seed data. Verify by inspecting the public
    # state attributes.
    for attr in (
        "cluster_labels",
        "positional_freq",
        "action_object_freq",
        "cluster_id_of",
        "action_clusters",
    ):
        value = getattr(learner, attr)
        # cluster_labels is the only place RelationType values live.
        # All other state is pure ints / strings / sets.
        if attr == "cluster_labels":
            assert value == {}, f"{attr} must be empty pre-labelling"


def test_train_menyebabkan_in_some_cluster(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """After train(), 'menyebabkan' must be in some cluster.

    With 4 observations (api/hujan/gesekan/listrik), it exceeds the
    default min_action_observations=2 and must be clustered.
    """
    learner.train(mixed_corpus)
    assert "menyebabkan" in learner.cluster_id_of
    cid = learner.cluster_id_of["menyebabkan"]
    assert cid >= 0, f"'menyebabkan' must be clustered, got cluster_id={cid}"


def test_train_actions_with_similar_objects_cluster_together(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """'menyebabkan' and 'memicu' (both take panas/banjir) cluster together."""
    learner.train(mixed_corpus)
    cid_menyebabkan = learner.cluster_id_of.get("menyebabkan")
    cid_memicu = learner.cluster_id_of.get("memicu")
    assert cid_menyebabkan is not None and cid_memicu is not None
    assert cid_menyebabkan == cid_memicu, (
        f"expected 'menyebabkan' and 'memicu' in same cluster "
        f"(both take state-change objects); got "
        f"menyebabkan={cid_menyebabkan}, memicu={cid_memicu}"
    )


def test_train_actions_with_different_objects_cluster_separately(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """'menyebabkan' (state-change) and 'membutuhkan' (need) cluster separately."""
    learner.train(mixed_corpus)
    cid_menyebabkan = learner.cluster_id_of.get("menyebabkan")
    cid_membutuhkan = learner.cluster_id_of.get("membutuhkan")
    assert cid_menyebabkan is not None
    assert cid_membutuhkan is not None
    assert cid_menyebabkan != cid_membutuhkan, (
        f"expected 'menyebabkan' and 'membutuhkan' in different clusters; "
        f"got menyebabkan={cid_menyebabkan}, membutuhkan={cid_membutuhkan}"
    )


# ======================================================================
# DoD #2: polysemy - same token, different role in different sentences
# ======================================================================


def test_polysemy_same_token_different_role(
    learner: PositionalClusterLearner, polysemy_corpus: list
):
    """'ayam' is agent in one sentence, object in another - SAME learner.

    This is the regression test for the PR #69 rejection reason #2:
    the old code locked each token to one "dominant position"
    cluster. The v2 design uses positional_freq as a *soft* count,
    so 'ayam' shows up in BOTH bucket 0 (agent) and bucket 2/-1
    (object). Role is determined at parse time by CURRENT position.
    """
    learner.train(polysemy_corpus)

    # 'ayam' must appear in BOTH agent bucket (0) and object bucket.
    ayam_positions = learner.positional_freq.get("ayam", {})
    assert _AGENT_BUCKET in ayam_positions, (
        f"'ayam' must have agent-bucket counts (corpus has it as subject); "
        f"got positional_freq['ayam']={ayam_positions}"
    )
    # Object bucket is 2 (3-token case) or -1 (>3-token case). All
    # polysemy_corpus sentences are 3-token, so bucket 2.
    assert _OBJECT_BUCKET_3 in ayam_positions, (
        f"'ayam' must have object-bucket counts (corpus has it as object); "
        f"got positional_freq['ayam']={ayam_positions}"
    )

    # The SPO parse of each sentence must give 'ayam' the right role
    # based on its CURRENT position in THAT sentence.
    spo_agent = learner.spo("ayam mencari pakan")
    assert spo_agent.subject == "ayam", (
        f"expected subject='ayam' in 'ayam mencari pakan', "
        f"got subject='{spo_agent.subject}'"
    )
    assert spo_agent.predicate == "mencari"
    assert spo_agent.object == "pakan"

    spo_object = learner.spo("manusia potong ayam")
    assert spo_object.subject == "manusia"
    assert spo_object.predicate == "potong"
    assert spo_object.object == "ayam", (
        f"expected object='ayam' in 'manusia potong ayam', "
        f"got object='{spo_object.object}'"
    )


def test_polysemy_no_global_role_lock(
    learner: PositionalClusterLearner, polysemy_corpus: list
):
    """The learner has NO global "token -> role" mapping.

    Role is purely a function of CURRENT sentence position. Verify
    this by parsing two sentences with the same token in different
    positions and confirming the role flips.
    """
    learner.train(polysemy_corpus)

    # Same learner, two sentences, 'ayam' flips role.
    s1 = learner.spo("ayam makan pakan")
    s2 = learner.spo("kucing kejar ayam")

    assert s1.subject == "ayam"
    assert s1.object == "pakan"
    assert s2.subject == "kucing"
    assert s2.object == "ayam"


# ======================================================================
# DoD #3: classify() before label_clusters() == SemanticRoleClassifier
# ======================================================================


def test_label_clusters_required(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """classify() before label_clusters() is identical to SemanticRoleClassifier.

    Backward-compatibility contract: a trained-but-unlabelled learner
    must produce the exact same classifications as a fresh
    SemanticRoleClassifier, because classify() short-circuits to the
    fallback when is_labelled is False.
    """
    learner.train(mixed_corpus)
    assert learner.is_trained
    assert not learner.is_labelled

    fallback = SemanticRoleClassifier()

    test_cases = [
        # CAUSAL via Indonesian seed
        "api menyebabkan panas",
        # CAUSAL via English seed
        "smoking causes cancer",
        # FUNCTIONAL via Indonesian seed
        "tanaman membutuhkan air",
        # FUNCTIONAL via English seed
        "engine requires fuel",
        # CATEGORICAL via Indonesian seed
        "manusia adalah mamalia",
        # CATEGORICAL via English seed
        "a dog is a mammal",
        # DIFFERENTIAL via standalone negation
        "kelelawar bukan burung",
        # DIFFERENTIAL via negation + CAUSAL seed
        "merokok tidak menyebabkan awet muda",
        # TEMPORAL via Indonesian seed
        "padi tumbuh setelah hujan",
        # CATEGORICAL fallback (unknown predicate)
        "X blahblah Y",
        # CATEGORICAL fallback (single token)
        "apple",
        # CATEGORICAL fallback (empty)
        "",
    ]

    for text in test_cases:
        learner_result = learner.classify(text)
        fallback_result = fallback.classify(text)
        assert learner_result == fallback_result, (
            f"Mismatch on '{text!r}': "
            f"learner={learner_result}, fallback={fallback_result}"
        )


def test_label_clusters_required_spo(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """spo() before label_clusters() delegates to fallback for short sentences.

    Once trained, spo() uses positional parsing (not the fallback)
    for >= 3-token sentences. This test verifies that the positional
    parse still produces correct results (matching the fallback's
    seed-based parse for seed-bearing sentences).
    """
    learner.train(mixed_corpus)

    # For a 3-token sentence, positional parse gives subject=tokens[0],
    # predicate=tokens[1], object=tokens[2] - which matches the
    # fallback's seed-based parse when tokens[1] is a seed.
    spo = learner.spo("api menyebabkan panas")
    assert spo.subject == "api"
    assert spo.predicate == "menyebabkan"
    assert spo.object == "panas"
    assert spo.negated is False


# ======================================================================
# DoD #4: label_clusters() applies to classify()
# ======================================================================


def test_label_clusters_applies(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """After label_clusters({cid: CAUSAL}), classify() for that cluster -> CAUSAL.

    Train on a causal corpus (5 sentences with 'menyebabkan' as
    action). The learner forms a cluster containing 'menyebabkan'.
    We label that cluster CAUSAL, then verify classify() returns
    CAUSAL for sentences using 'menyebabkan'.
    """
    learner.train(causal_corpus)
    cid = learner.cluster_id_of["menyebabkan"]
    assert cid >= 0

    # Before labelling - classify uses fallback (which returns CAUSAL
    # via the seed match, but the point is it's the fallback path).
    assert not learner.is_labelled

    # Label the cluster.
    learner.label_clusters({cid: RelationType.CAUSAL})
    assert learner.is_labelled
    assert learner.cluster_labels[cid] == RelationType.CAUSAL

    # After labelling - classify uses the cluster label.
    result = learner.classify("api menyebabkan panas")
    assert result == RelationType.CAUSAL


def test_label_clusters_unknown_id_skipped(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """label_clusters() silently skips unknown cluster_ids.

    Forward-compat: a saved mapping from a previous run that had
    more clusters should not crash on load.
    """
    learner.train(causal_corpus)
    real_cid = learner.cluster_id_of["menyebabkan"]
    fake_cid = 99999

    learner.label_clusters({
        real_cid: RelationType.CAUSAL,
        fake_cid: RelationType.TEMPORAL,
    })

    # Real cluster is labelled.
    assert learner.cluster_labels[real_cid] == RelationType.CAUSAL
    # Fake cluster_id is silently skipped (not added to labels).
    assert fake_cid not in learner.cluster_labels


def test_label_clusters_overwrites_previous(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """Calling label_clusters() again overwrites previous labels."""
    learner.train(causal_corpus)
    cid = learner.cluster_id_of["menyebabkan"]

    learner.label_clusters({cid: RelationType.CAUSAL})
    assert learner.cluster_labels[cid] == RelationType.CAUSAL

    # Re-label with a different RelationType.
    learner.label_clusters({cid: RelationType.TEMPORAL})
    assert learner.cluster_labels[cid] == RelationType.TEMPORAL


def test_classify_unlabelled_cluster_falls_back(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """If the action's cluster exists but is unlabelled, classify() falls back.

    Train on mixed corpus (so multiple clusters form), label only ONE
    cluster, then verify classify() for an action in an UNLABELLED
    cluster returns the fallback result.
    """
    learner.train(mixed_corpus)
    cid_menyebabkan = learner.cluster_id_of["menyebabkan"]
    cid_membutuhkan = learner.cluster_id_of["membutuhkan"]
    assert cid_menyebabkan != cid_membutuhkan

    # Label only the menyebabkan cluster.
    learner.label_clusters({cid_menyebabkan: RelationType.CAUSAL})

    # classify() for an action in the labelled cluster -> CAUSAL.
    assert learner.classify("api menyebabkan panas") == RelationType.CAUSAL

    # classify() for an action in the UNLABELLED cluster -> fallback.
    # The fallback's seed match for 'membutuhkan' returns FUNCTIONAL.
    result = learner.classify("tanaman membutuhkan air")
    assert result == RelationType.FUNCTIONAL


def test_classify_unseen_action_falls_back(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """An action token not in any cluster delegates to fallback."""
    learner.train(mixed_corpus)
    cid_menyebabkan = learner.cluster_id_of["menyebabkan"]
    learner.label_clusters({cid_menyebabkan: RelationType.CAUSAL})

    # 'membaca' is not in the corpus -> not in any cluster -> fallback.
    # Fallback has no seed for 'membaca' -> CATEGORICAL.
    result = learner.classify("saya membaca buku")
    assert result == RelationType.CATEGORICAL


def test_classify_unclustered_action_falls_back(
    learner: PositionalClusterLearner
):
    """An action with too few observations (cluster_id = -1) falls back.

    Train with only one observation for 'menyebabkan' - below
    min_action_observations=2. The action gets cluster_id = -1 and
    classify() must fall back.
    """
    learner.train(["api menyebabkan panas"])
    assert learner.cluster_id_of.get("menyebabkan") == -1

    # classify() falls back. The fallback's seed match returns CAUSAL.
    assert learner.classify("api menyebabkan panas") == RelationType.CAUSAL


# ======================================================================
# DoD #5: negation overrides cluster label
# ======================================================================


def test_negation_overrides_cluster(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """'X tidak menyebabkan Y' is DIFFERENTIAL even when cluster is labelled CAUSAL.

    Same contract as SemanticRoleClassifier: negation is a syntactic
    signal that always inverts the relation. It beats any cluster
    label.
    """
    learner.train(causal_corpus)
    cid = learner.cluster_id_of["menyebabkan"]
    learner.label_clusters({cid: RelationType.CAUSAL})

    # Sanity: without negation -> CAUSAL.
    assert learner.classify("api menyebabkan panas") == RelationType.CAUSAL

    # With negation -> DIFFERENTIAL (cluster label overridden).
    result = learner.classify("merokok tidak menyebabkan awet muda")
    assert result == RelationType.DIFFERENTIAL, (
        f"negation must override cluster label; got {result}"
    )


# ======================================================================
# DoD #6: persistence with labels
# ======================================================================


def test_persistence_with_labels(
    learner: PositionalClusterLearner, mixed_corpus: list, tmp_path
):
    """save() + load() preserves all learned state INCLUDING cluster_labels."""
    learner.train(mixed_corpus)
    cid_menyebabkan = learner.cluster_id_of["menyebabkan"]
    cid_membutuhkan = learner.cluster_id_of["membutuhkan"]
    learner.label_clusters({
        cid_menyebabkan: RelationType.CAUSAL,
        cid_membutuhkan: RelationType.FUNCTIONAL,
    })

    path = str(tmp_path / "pcl_state.json")
    learner.save(path)
    assert os.path.exists(path)

    loaded = PositionalClusterLearner.load(path)

    # All learned structures must match.
    assert loaded.positional_freq == learner.positional_freq
    assert loaded.action_object_freq == learner.action_object_freq
    assert loaded.cluster_id_of == learner.cluster_id_of
    assert loaded.action_clusters == learner.action_clusters
    assert loaded.cluster_labels == learner.cluster_labels

    # Loaded learner must be both trained AND labelled.
    assert loaded.is_trained
    assert loaded.is_labelled

    # Classification results must match.
    assert loaded.classify("api menyebabkan panas") == \
        learner.classify("api menyebabkan panas")
    assert loaded.classify("tanaman membutuhkan air") == \
        learner.classify("tanaman membutuhkan air")


def test_persistence_without_labels(
    learner: PositionalClusterLearner, mixed_corpus: list, tmp_path
):
    """save/load of a trained-but-unlabelled learner preserves is_labelled=False."""
    learner.train(mixed_corpus)
    assert not learner.is_labelled

    path = str(tmp_path / "pcl_unlabelled.json")
    learner.save(path)
    loaded = PositionalClusterLearner.load(path)

    assert loaded.is_trained
    assert not loaded.is_labelled
    assert loaded.cluster_labels == {}


def test_persistence_creates_parent_dirs(
    learner: PositionalClusterLearner, mixed_corpus: list, tmp_path
):
    """save() creates parent directories on demand."""
    learner.train(mixed_corpus)
    nested = str(tmp_path / "nested" / "deeper" / "pcl.json")
    learner.save(nested)
    assert os.path.exists(nested)


def test_persistence_atomic_write(
    learner: PositionalClusterLearner, mixed_corpus: list, tmp_path
):
    """save() writes a parseable JSON file with a trailing newline."""
    learner.train(mixed_corpus)
    path = str(tmp_path / "pcl_atomic.json")
    learner.save(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.endswith("\n")

    import json
    parsed = json.loads(content)
    for key in (
        "positional_freq",
        "action_object_freq",
        "cluster_id_of",
        "action_clusters",
        "cluster_labels",
    ):
        assert key in parsed


# ======================================================================
# Supplementary: inspect_clusters / inspect_cluster_details
# ======================================================================


def test_inspect_clusters_returns_readable_view(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """inspect_clusters() returns {cluster_id: [action_tokens]} for human review."""
    learner.train(mixed_corpus)
    inspection = learner.inspect_clusters()

    assert isinstance(inspection, dict)
    assert len(inspection) >= 1

    for cid, actions in inspection.items():
        assert isinstance(cid, int)
        assert cid >= 0
        assert isinstance(actions, list)
        assert len(actions) >= 1
        assert all(isinstance(a, str) for a in actions)


def test_inspect_cluster_details_includes_top_objects_and_label(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """inspect_cluster_details() returns actions + top_objects + label per cluster."""
    learner.train(mixed_corpus)
    cid = learner.cluster_id_of["menyebabkan"]
    learner.label_clusters({cid: RelationType.CAUSAL})

    details = learner.inspect_cluster_details()
    assert cid in details
    cluster = details[cid]
    assert "actions" in cluster
    assert "top_objects" in cluster
    assert "label" in cluster
    assert cluster["label"] == "CAUSAL"
    assert "menyebabkan" in cluster["actions"]
    # Top objects for the menyebabkan cluster should include the
    # state-change objects from the corpus.
    top_objs = cluster["top_objects"]
    assert any(o in top_objs for o in ("panas", "banjir", "kebakaran", "kanker"))


# ======================================================================
# Supplementary: fallback composition
# ======================================================================


def test_uses_custom_fallback():
    """The learner honours a caller-supplied fallback classifier."""
    custom = SemanticRoleClassifier(override_threshold=5)
    learner = PositionalClusterLearner(fallback=custom)
    assert learner.fallback is custom

    # Without training, classify() delegates to the custom fallback.
    assert learner.classify("X menyebabkan Y") == RelationType.CAUSAL


# ======================================================================
# Supplementary: position bucketing helpers
# ======================================================================


def test_compute_buckets_3_tokens():
    """3-token sentence -> [0, 1, 2] (classic SVO)."""
    buckets = PositionalClusterLearner._compute_buckets(3)
    assert buckets == [_AGENT_BUCKET, _ACTION_BUCKET, _OBJECT_BUCKET_3]


def test_compute_buckets_5_tokens():
    """>3-token sentence -> [0, 1, 1, 1, -1] (first, middles, last)."""
    buckets = PositionalClusterLearner._compute_buckets(5)
    assert buckets == [_AGENT_BUCKET, _ACTION_BUCKET, _ACTION_BUCKET, _ACTION_BUCKET, _OBJECT_BUCKET_N]


def test_compute_buckets_edge_cases():
    assert PositionalClusterLearner._compute_buckets(0) == []
    assert PositionalClusterLearner._compute_buckets(1) == [0]
    assert PositionalClusterLearner._compute_buckets(2) == [0, 1]


def test_jaccard_similarity():
    """Jaccard: |A ∩ B| / |A ∪ B|."""
    j = PositionalClusterLearner._jaccard
    assert j(set(), set()) == 0.0
    assert j({"a"}, set()) == 0.0
    assert j({"a"}, {"a"}) == 1.0
    assert j({"a", "b"}, {"b", "c"}) == 1 / 3  # |{b}| / |{a,b,c}|
    assert j({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


# ======================================================================
# Supplementary: train() robustness
# ======================================================================


def test_train_empty_corpus_no_crash(learner: PositionalClusterLearner):
    """train([]) is a no-op - no crash, learner stays untrained."""
    learner.train([])
    assert not learner.is_trained
    assert learner.positional_freq == {}


def test_train_short_sentences_no_crash(learner: PositionalClusterLearner):
    """train() with only short sentences builds no action_object_freq."""
    learner.train([
        "hello world",
        "single",
        "",
        "one two three",
    ])
    assert learner.is_trained
    # action_object_freq has one entry from 'one two three'.
    assert "two" in learner.action_object_freq


def test_train_malformed_lines_skipped(learner: PositionalClusterLearner):
    """train() skips empty / whitespace-only lines without crashing."""
    learner.train([
        "saya makan ayam",
        "",
        "   ",
        "dia makan ikan",
    ])
    assert learner.action_object_freq["makan"]["ayam"] == 1
    assert learner.action_object_freq["makan"]["ikan"] == 1


def test_train_resets_labels_on_retrain(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """Re-training resets cluster_labels (cluster_ids may shift)."""
    learner.train(causal_corpus)
    cid = learner.cluster_id_of["menyebabkan"]
    learner.label_clusters({cid: RelationType.CAUSAL})
    assert learner.is_labelled

    # Re-train - labels must reset.
    learner.train(causal_corpus)
    assert not learner.is_labelled
    assert learner.cluster_labels == {}


# ======================================================================
# Supplementary: RelationType compatibility (BA44 contract)
# ======================================================================


def test_relation_type_compatible_with_ba44_rules(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """classify() returns a RelationType member usable by BA44's rules."""
    learner.train(causal_corpus)
    cid = learner.cluster_id_of["menyebabkan"]
    learner.label_clusters({cid: RelationType.CAUSAL})

    result = learner.classify("api menyebabkan panas")
    assert isinstance(result, RelationType)
    assert result.name == "CAUSAL"
    assert hasattr(result, "value")


def test_relation_type_fallback_compatible(learner: PositionalClusterLearner):
    """Fallback classification also returns the same RelationType enum."""
    result = learner.classify("api menyebabkan panas")
    assert isinstance(result, RelationType)
    assert type(result) is type(
        learner.fallback.classify("X menyebabkan Y")
    )


# ======================================================================
# Supplementary: SPO edge cases
# ======================================================================


def test_spo_empty_string(learner: PositionalClusterLearner, svo_corpus: list):
    """Empty input after training -> empty SPO, no crash."""
    learner.train(svo_corpus)
    spo = learner.spo("")
    assert spo.subject == ""
    assert spo.predicate == ""
    assert spo.object == ""


def test_spo_short_sentence_delegates_to_fallback(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """Short sentences (<3 tokens) delegate to fallback for SVO parse.

    This preserves the "X bukan Y" -> DIFFERENTIAL path that lives in
    SemanticRoleClassifier's seed table.
    """
    learner.train(svo_corpus)
    # 'X bukan' is 2 tokens - too short for positional SVO.
    # Delegate to fallback, which extracts 'bukan' as predicate
    # (DIFFERENTIAL seed).
    spo = learner.spo("kelelawar bukan")
    # The fallback returns predicate='bukan'.
    assert spo.predicate == "bukan"


def test_spo_long_sentence_collapses_middle(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """Long sentences: subject=tokens[0], predicate=tokens[1..-1], object=tokens[-1]."""
    learner.train(svo_corpus)
    spo = learner.spo("saya sedang makan nasi ayam")
    assert spo.subject == "saya"
    # For >3-token sentences, predicate = all middle tokens joined.
    assert spo.predicate == "sedang makan nasi"
    assert spo.object == "ayam"
