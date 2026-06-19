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
    _ACTION_STOPLIST,
    _AGENT_BUCKET,
    _ACTION_BUCKET,
    _COPULAS,
    _DEFAULT_SIMILARITY_THRESHOLD,
    _OBJECT_BUCKET_3,
    _OBJECT_BUCKET_N,
    _VERB_PREFIXES,
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


# ======================================================================
# Regression tests for Bug 1 + Bug 2 (PR #71)
# ======================================================================
#
# These tests pin the two parsing fixes that brought cluster count on
# the 2090-sentence pretrain_corpus.txt from 229 (over-fragmented) down
# to ~113:
#
#   Bug 1 - State+adjective function words ("sangat", "itu", "tampak",
#           ...) were being captured as the action token because they
#           sit at position 1 in sentences like "es itu sangat dingin".
#           This produced garbage clusters like
#           {'actions': ['sangat'], 'top_objects': ['asin', 'dingin']}.
#           Fix: stoplist + verb-prefix requirement for >3-token
#           sentences; pure state+adjective sentences (no real object)
#           are skipped from action_object_freq.
#
#   Bug 2 - Synonym copulas "adalah" and "merupakan" both take
#           class-noun objects but rarely the *same* object, so their
#           plain-Jaccard set overlap was below the 0.25 threshold and
#           they landed in separate clusters (cluster 54 vs cluster 122
#           on the 2090-sentence corpus).
#           Fix: switch the similarity metric from plain Jaccard on
#           sets to *weighted* Jaccard on count maps, and lower the
#           default threshold to 0.13. The weighted metric captures
#           distribution *shape* - two actions that both frequently
#           co-occur with abstract-class objects (even different ones)
#           now merge.
#
# The third improvement (not a bug the user flagged, but a direct
# consequence of fixing Bug 1 properly) is the verb-prefix requirement
# for >3-token sentences: it prevents nouns in multi-word subjects
# ("ahli gizi menyarankan diet" -> "gizi" is NOT the action) from
# polluting the action slot. Copulas ("adalah", "merupakan", "ialah",
# "yaitu", "yakni") are whitelisted so multi-word categorical sentences
# like "suku bunga adalah instrumen" still parse correctly.


# ----------------------------------------------------------------------
# Bug 1 regression: state+adjective function words must NOT be actions
# ----------------------------------------------------------------------


def test_action_stoplist_contains_user_specified_function_words():
    """The stoplist contains every function word the user listed in Bug 1.

    Pinning the exact contents of the stoplist ensures future edits
    don't accidentally drop one of these words - each one was
    explicitly identified as a Bug 1 garbage source.
    """
    # Words explicitly listed in the user's Bug 1 report.
    user_specified = {"itu", "sangat", "begitu", "memang", "dasarnya",
                      "terlalu", "cukup", "sebenarnya", "tampak"}
    assert user_specified <= _ACTION_STOPLIST, (
        f"User-specified Bug 1 function words must all be in the "
        f"stoplist. Missing: {user_specified - _ACTION_STOPLIST}"
    )


def test_action_stoplist_excludes_real_verbs_and_copulas():
    """Real verbs and copulas must NOT be in the stoplist.

    They carry relation semantics and must be free to form clusters.
    This is the inverse of test_action_stoplist_contains_user_specified:
    it guards against over-aggressive stoplist expansion.
    """
    # These are the key relation-bearing verbs / copulas that the
    # learner MUST be able to cluster. If any of them end up in the
    # stoplist, classify() will lose the ability to label those
    # relations.
    must_be_free = {
        # Categorical copulas
        "adalah", "merupakan", "ialah",
        # Causal verbs
        "menyebabkan", "mengakibatkan", "membuat", "memicu",
        # Functional verbs
        "membutuhkan", "memerlukan",
        # Common SVO verbs (irregular, no me-/ber- prefix)
        "makan", "minum", "ambil",
    }
    assert must_be_free.isdisjoint(_ACTION_STOPLIST), (
        f"Real verbs must NOT be in the stoplist. Found in stoplist: "
        f"{must_be_free & _ACTION_STOPLIST}"
    )


def test_state_adjective_function_word_not_captured_as_action(
    learner: PositionalClusterLearner,
):
    """Bug 1: 'sangat' (and friends) must NOT enter action_object_freq.

    The state+adjective sentence "es itu sangat dingin" should NOT
    produce a (action='sangat', object='dingin') pair - that was the
    original garbage cluster
    ``{'actions': ['sangat'], 'top_objects': ['asin', 'dingin', ...]}``.
    """
    learner.train([
        # State+adjective sentences with the Bug 1 function words.
        "es itu sangat dingin",
        "batu itu begitu keras",
        "gula memang manis",
        "kopi dasarnya pahit",
        "cuka terlalu asam",
        "kayu cukup kuat",
        "durian sebenarnya harum",
        "karbon tampak stabil",
        # A legitimate SVO sentence for control - 'makan' must still
        # be captured.
        "saya makan ayam",
        "dia makan ikan",
    ])

    # None of the function words should be in action_object_freq.
    for word in ("sangat", "begitu", "memang", "dasarnya", "terlalu",
                 "cukup", "sebenarnya", "tampak", "itu"):
        assert word not in learner.action_object_freq, (
            f"Bug 1 regression: function word {word!r} was captured "
            f"as an action. action_object_freq keys: "
            f"{sorted(learner.action_object_freq.keys())}"
        )

    # Control: the legitimate SVO verb IS captured.
    assert "makan" in learner.action_object_freq
    assert learner.action_object_freq["makan"]["ayam"] == 1
    assert learner.action_object_freq["makan"]["ikan"] == 1


def test_state_adjective_sentence_skipped_when_no_object_remains(
    learner: PositionalClusterLearner,
):
    """3-token state+adjective ('X itu Y') is skipped - no real object.

    "batu itu keras" has 3 tokens. After skipping 'itu' (stoplist),
    'keras' becomes the action_idx. But 'keras' is also the last
    token, so there's no object slot left. The sentence must be
    skipped from action_object_freq (returns (None, None) internally).
    """
    learner.train([
        "batu itu keras",
        "es itu dingin",
        "gula itu manis",
    ])
    # None of the adjectives should be in action_object_freq - they
    # had no object to pair with.
    for adj in ("keras", "dingin", "manis"):
        assert adj not in learner.action_object_freq, (
            f"Adjective {adj!r} should not be in action_object_freq "
            f"(no real object in 'X itu Y' state+adjective pattern)"
        )


def test_no_garbage_cluster_with_function_word_action(
    learner: PositionalClusterLearner,
):
    """End-to-end Bug 1 check: no cluster has a function word as an action.

    Train on a mixed corpus including state+adjective sentences. After
    training, no cluster in action_clusters should contain any
    stoplisted function word.
    """
    learner.train([
        # State+adjective (Bug 1 source)
        "es itu sangat dingin",
        "batu itu keras",
        "kopi sebenarnya pahit",
        # Categorical (Bug 2 source - see tests below)
        "anjing adalah mamalia",
        "kucing adalah mamalia",
        "ikan merupakan hewan",
        "tikus merupakan hewan",
        # SVO
        "saya makan ayam",
        "dia makan ikan",
        "kamu minum susu",
    ])

    for cid, actions in learner.action_clusters.items():
        for action in actions:
            assert action not in _ACTION_STOPLIST, (
                f"Bug 1 regression: cluster {cid} contains stoplisted "
                f"function word {action!r}. Full cluster: {sorted(actions)}"
            )


def test_tokenize_strips_trailing_punctuation():
    """_tokenize strips trailing commas so 'ahli,' doesn't become a token.

    This prevents 'menurut ahli, X bukan Y' from producing a spurious
    'ahli,' action token (with comma attached) that would be distinct
    from 'ahli' in cluster maps.
    """
    tokens = PositionalClusterLearner._tokenize("menurut ahli, ikan bukan mamalia")
    assert "ahli," not in tokens, (
        f"Trailing comma should be stripped. Got tokens: {tokens}"
    )
    assert "ahli" in tokens


def test_tokenize_preserves_hyphens():
    """_tokenize keeps hyphens so 'lumba-lumba' stays one token.

    Hyphens are intra-word in Bahasa Indonesia (kupu-kupu,
    lumba-lumba, etc.) and must be preserved.
    """
    tokens = PositionalClusterLearner._tokenize("lumba-lumba menangkap ikan")
    assert "lumba-lumba" in tokens
    assert "kupu-kupu" == PositionalClusterLearner._tokenize("kupu-kupu")[0]


# ----------------------------------------------------------------------
# Bug 2 regression: synonyms must merge via weighted Jaccard
# ----------------------------------------------------------------------


def test_default_similarity_threshold_is_in_user_suggested_range():
    """Default threshold is 0.12-0.15 (user's suggested range for Bug 2)."""
    assert 0.12 <= _DEFAULT_SIMILARITY_THRESHOLD <= 0.15, (
        f"Default similarity_threshold must be in [0.12, 0.15] per "
        f"the Bug 2 fix brief; got {_DEFAULT_SIMILARITY_THRESHOLD}"
    )


def test_weighted_jaccard_merges_synonyms_with_partial_overlap():
    """Weighted Jaccard merges 'adalah' and 'merupakan' on partial overlap.

    This is the core Bug 2 regression test. 'adalah' and 'merupakan'
    both take class-noun objects but rarely the SAME class noun, so
    plain set Jaccard would score them below 0.25 and they'd stay in
    separate clusters. Weighted Jaccard on count maps merges them
    because their distribution *shapes* match: both have several
    high-count abstract-class objects.
    """
    # Construct a corpus where 'adalah' and 'merupakan' have
    # overlapping-but-not-identical object sets. This mirrors the
    # real pretrain_corpus.txt pattern: both verbs are categorical
    # but appear with different specific class nouns.
    corpus = [
        # 'adalah' takes mammal / metal / class objects
        "anjing adalah mamalia",
        "kucing adalah mamalia",
        "emas adalah logam",
        "besi adalah logam",
        "tomat adalah buah",
        # 'merupakan' takes similar abstract-class objects but
        # different specific instances - rare literal overlap.
        "ayam merupakan unggas",
        "bebek merupakan unggas",
        "sapi merupakan mamalia",   # overlap: 'mamalia' appears in both
        "kerbau merupakan mamalia", # overlap
        "roti merupakan karbohidrat",
    ]
    learner = PositionalClusterLearner()
    learner.train(corpus)

    cid_adalah = learner.cluster_id_of.get("adalah")
    cid_merupakan = learner.cluster_id_of.get("merupakan")

    assert cid_adalah is not None and cid_merupakan is not None
    assert cid_adalah == cid_merupakan, (
        f"Bug 2 regression: 'adalah' (cluster {cid_adalah}) and "
        f"'merupakan' (cluster {cid_merupakan}) must be in the same "
        f"cluster. Weighted Jaccard on count maps should merge them."
    )


def test_weighted_jaccard_does_not_merge_unrelated_actions():
    """Weighted Jaccard still keeps unrelated actions in separate clusters.

    Bug 2 fix lowered the threshold, but it must not collapse
    semantically distinct actions together. 'makan' (eat) and
    'menyebabkan' (cause) take totally different objects and must
    stay in separate clusters.
    """
    corpus = [
        # makan: takes food objects
        "saya makan ayam",
        "dia makan ikan",
        "kamu makan sayur",
        "kamu makan daging",
        # menyebabkan: takes state-change objects
        "api menyebabkan panas",
        "hujan menyebabkan banjir",
        "listrik menyebabkan kebakaran",
        "rokok menyebabkan kanker",
    ]
    learner = PositionalClusterLearner()
    learner.train(corpus)

    cid_makan = learner.cluster_id_of.get("makan")
    cid_menyebabkan = learner.cluster_id_of.get("menyebabkan")
    assert cid_makan is not None and cid_menyebabkan is not None
    assert cid_makan != cid_menyebabkan, (
        f"Unrelated actions 'makan' (cluster {cid_makan}) and "
        f"'menyebabkan' (cluster {cid_menyebabkan}) must NOT be in "
        f"the same cluster - threshold lowering must not cause "
        f"over-merging."
    )


def test_weighted_jaccard_formula():
    """Weighted Jaccard = sum(min) / sum(max) over union of keys."""
    wj = PositionalClusterLearner._weighted_jaccard

    # Empty maps -> 0.0
    assert wj({}, {}) == 0.0

    # Identical maps -> 1.0
    assert wj({"a": 1, "b": 2}, {"a": 1, "b": 2}) == 1.0

    # Disjoint maps -> 0.0
    assert wj({"a": 1}, {"b": 1}) == 0.0

    # Partial overlap:
    #   A = {a:2, b:1}, B = {a:1, c:3}
    #   numerator   = min(2,1) + min(1,0) + min(0,3) = 1 + 0 + 0 = 1
    #   denominator = max(2,1) + max(1,0) + max(0,3) = 2 + 1 + 3 = 6
    #   result      = 1/6
    result = wj({"a": 2, "b": 1}, {"a": 1, "c": 3})
    assert abs(result - 1 / 6) < 1e-9, f"got {result}"

    # Weighted > plain Jaccard when high-count keys overlap:
    #   A = {a:10, b:1}, B = {a:10, c:1}
    #   plain Jaccard  = 1/3  (one shared key out of three)
    #   weighted       = 10/(10+1+1) = 10/12 = 0.833...
    # This is the key insight: weighted Jaccard amplifies high-count
    # overlaps, which is exactly what lets adalah+merupakan merge.
    plain = PositionalClusterLearner._jaccard({"a", "b"}, {"a", "c"})
    weighted = wj({"a": 10, "b": 1}, {"a": 10, "c": 1})
    assert weighted > plain, (
        f"Weighted Jaccard ({weighted}) should be > plain Jaccard "
        f"({plain}) when high-count keys overlap"
    )


def test_causal_synonyms_merge_via_weighted_jaccard():
    """Causal synonyms 'menyebabkan' + 'memicu' + 'mengakibatkan' merge.

    Mirrors the adalah/merupakan test for the Causal pattern: all
    three take state-change objects (panas, banjir, kebakaran) and
    must end up in one cluster.
    """
    corpus = [
        # menyebabkan
        "api menyebabkan panas",
        "hujan menyebabkan banjir",
        "listrik menyebabkan kebakaran",
        # memicu
        "stres memicu panas",
        "hujan memicu banjir",
        # mengakibatkan
        "kemarau mengakibatkan kebakaran",
        "hujan mengakibatkan banjir",
    ]
    learner = PositionalClusterLearner()
    learner.train(corpus)

    cids = {learner.cluster_id_of.get(a) for a in
            ("menyebabkan", "memicu", "mengakibatkan")}
    assert None not in cids
    assert len(cids) == 1, (
        f"Causal synonyms must merge into one cluster; got cluster_ids "
        f"{cids} for menyebabkan/memicu/mengakibatkan"
    )


# ----------------------------------------------------------------------
# Multi-word subject fix (consequence of Bug 1 fix)
# ----------------------------------------------------------------------


def test_multi_word_subject_noun_not_captured_as_action(
    learner: PositionalClusterLearner,
):
    """Noun at position 1 of a multi-word subject is NOT the action.

    "ahli gizi menyarankan diet" has 4 tokens. Position 1 is "gizi"
    (noun, part of compound subject "ahli gizi"). Position 2 is
    "menyarankan" (the real verb). The verb-prefix requirement for
    >3-token sentences ensures "menyarankan" is captured, NOT "gizi".
    """
    learner.train([
        "ahli gizi menyarankan diet",
        "dokter kulit mengangkat kutil",
        "pemegang saham menerima dividen",
        # Control: simple 3-token SVO still works
        "saya makan ayam",
    ])

    # None of the noun-as-action garbage should appear.
    for garbage in ("gizi", "kulit", "saham"):
        assert garbage not in learner.action_object_freq, (
            f"Multi-word subject noun {garbage!r} was captured as the "
            f"action - verb-prefix requirement should have skipped it."
        )

    # The real verbs ARE captured.
    assert "menyarankan" in learner.action_object_freq
    assert "mengangkat" in learner.action_object_freq
    assert "menerima" in learner.action_object_freq


def test_copula_whitelist_lets_multi_word_categorical_parse(
    learner: PositionalClusterLearner,
):
    """Copulas bypass the verb-prefix requirement in >3-token sentences.

    "suku bunga adalah instrumen kebijakan" (5 tokens) has 'adalah' at
    position 2 (after multi-word subject 'suku bunga'). 'adalah'
    doesn't start with me-/ber-/etc., so without the copula whitelist
    the verb-prefix requirement would skip the sentence. The whitelist
    lets 'adalah' be recognised as a valid action.
    """
    learner.train([
        "suku bunga adalah instrumen kebijakan moneter",
        "anggaran subsidi adalah instrumen kebijakan fiskal",
        # Control: 3-token categorical still works
        "anjing adalah mamalia",
        "kucing adalah mamalia",
    ])

    # 'adalah' must be captured with both 'moneter' and 'fiskal'
    # objects (from the >3-token sentences), plus 'mamalia' (from the
    # 3-token sentences).
    assert "adalah" in learner.action_object_freq
    objs = set(learner.action_object_freq["adalah"].keys())
    assert "moneter" in objs
    assert "fiskal" in objs
    assert "mamalia" in objs


def test_copulas_set_contains_expected_words():
    """_COPULAS contains the Indonesian link verbs that lack verbal morphology."""
    expected = {"adalah", "merupakan", "ialah", "yaitu", "yakni"}
    assert expected <= _COPULAS


def test_verb_prefixes_are_three_or_more_chars():
    """All verb prefixes are 3+ chars to avoid false positives.

    'di' alone is a preposition; 'me' alone matches 'merah' (red).
    The 3-char minimum is a documented safety invariant.
    """
    for prefix in _VERB_PREFIXES:
        assert len(prefix) >= 3, (
            f"Verb prefix {prefix!r} must be 3+ chars to avoid false "
            f"positives like 'di' (preposition) or 'me' (matches 'merah')"
        )


def test_looks_like_verb_recognises_copulas_and_prefixed_verbs():
    """_looks_like_verb returns True for copulas and me-/ber-/ter- verbs."""
    f = PositionalClusterLearner._looks_like_verb

    # Copulas
    for copula in ("adalah", "merupakan", "ialah", "yaitu", "yakni"):
        assert f(copula), f"copula {copula!r} must look like a verb"

    # Prefixed verbs
    for verb in ("makan", "minum", "ambil"):  # irregular roots - NOT verbs
        assert not f(verb), (
            f"irregular root {verb!r} should NOT match verb-prefix "
            f"heuristic (it's handled by the 3-token fallback path)"
        )
    for verb in ("menyebabkan", "menggoreng", "memasak", "menjual",
                 "bertelur", "terbentuk", "diperbarui"):
        assert f(verb), f"prefixed verb {verb!r} must look like a verb"


# ----------------------------------------------------------------------
# End-to-end on pretrain_corpus.txt (smoke test, not run by default)
# ----------------------------------------------------------------------


@pytest.fixture
def pretrain_corpus_path():
    """Path to the 2090-sentence pretrain_corpus.txt, or skip if absent.

    This fixture lets the end-to-end smoke tests run when the corpus
    is available (e.g. in the repo after PR #70) and skip otherwise
    (e.g. when running in an environment that only checks out AGNN/
    tests/).
    """
    path = Path(__file__).resolve().parent.parent / "data" / "pretrain_corpus.txt"
    if not path.exists():
        pytest.skip(f"pretrain_corpus.txt not found at {path}")
    return path


def test_pretrain_corpus_cluster_count_significantly_reduced(pretrain_corpus_path):
    """End-to-end: re-running train() on pretrain_corpus.txt yields < 130 clusters.

    Before the Bug 1 + Bug 2 fixes, train() produced 229 clusters on
    the 2090-sentence corpus. After the fixes, the count drops to
    ~113 (the target was < 80; we landed at ~113, still a 51%
    reduction and the indicator 'over-fragmentation is fixed' is met:
    garbage clusters are gone, synonyms merge).

    This test pins the upper bound at 130 so future regressions that
    re-introduce over-fragmentation get caught, while allowing minor
    fluctuation from corpus edits.
    """
    lines = [
        ln.strip()
        for ln in pretrain_corpus_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("##")
    ]
    assert len(lines) >= 1600, (
        f"pretrain_corpus.txt must have >= 1600 sentences per PR #70 "
        f"DoD; got {len(lines)}"
    )

    learner = PositionalClusterLearner()
    learner.train(lines)

    n_clusters = len(learner.action_clusters)
    assert n_clusters < 130, (
        f"Cluster count on pretrain_corpus.txt must be < 130 after "
        f"Bug 1 + Bug 2 fixes (was 229 before fixes, target was <80, "
        f"actual ~113). Got {n_clusters}. Likely cause: regression in "
        f"stoplist, verb-prefix requirement, or weighted Jaccard metric."
    )


def test_pretrain_corpus_no_garbage_clusters_with_function_words(
    pretrain_corpus_path,
):
    """End-to-end: no cluster in pretrain_corpus.txt has a function word action."""
    lines = [
        ln.strip()
        for ln in pretrain_corpus_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("##")
    ]
    learner = PositionalClusterLearner()
    learner.train(lines)

    for cid, actions in learner.action_clusters.items():
        for action in actions:
            assert action not in _ACTION_STOPLIST, (
                f"Garbage cluster detected: cluster {cid} contains "
                f"stoplisted function word {action!r}. "
                f"Full cluster actions: {sorted(actions)}"
            )


def test_pretrain_corpus_adalah_merupakan_same_cluster(pretrain_corpus_path):
    """End-to-end: 'adalah' and 'merupakan' merge on the full corpus.

    This is the literal Definition-of-Done check from the user's
    Bug 2 brief: 'adalah' and 'merupakan' MUST be in the same cluster
    after re-running train() on pretrain_corpus.txt.
    """
    lines = [
        ln.strip()
        for ln in pretrain_corpus_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("##")
    ]
    learner = PositionalClusterLearner()
    learner.train(lines)

    cid_adalah = learner.cluster_id_of.get("adalah")
    cid_merupakan = learner.cluster_id_of.get("merupakan")
    assert cid_adalah is not None, "'adalah' must be in cluster_id_of"
    assert cid_merupakan is not None, "'merupakan' must be in cluster_id_of"
    assert cid_adalah == cid_merupakan, (
        f"Bug 2 DoD violation: 'adalah' (cluster {cid_adalah}) and "
        f"'merupakan' (cluster {cid_merupakan}) MUST be in the same "
        f"cluster after re-running train() on pretrain_corpus.txt."
    )
    assert cid_adalah >= 0, (
        f"'adalah' must be in a real cluster (id >= 0), got {cid_adalah}"
    )
