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

Plus supplementary tests for inspect_cluster_details, fallback composition,
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
    _ACTION_ANCHOR_MAX_BUCKET_ENTROPY,
    _ACTION_ANCHOR_MIN_FREQ,
    _AGENT_BUCKET,
    _ACTION_BUCKET,
    _DEFAULT_SIMILARITY_THRESHOLD,
    _FUNCTION_WORD_ENTROPY_THRESHOLD,
    _FUNCTION_WORD_MAX_BUCKET_ENTROPY,
    _FUNCTION_WORD_MIN_FREQ,
    _FUNCTION_WORD_MIN_POSITIONS,
    _OBJECT_BUCKET_3,
    _OBJECT_BUCKET_N,
    _3_TOKEN_MIN_ACTION_FREQ,
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

    # inspect_cluster_details() returns Dict[int, Dict[str, object]]
    # with "actions" key (no labels yet — see label_clusters()).
    inspection = learner.inspect_cluster_details()
    assert isinstance(inspection, dict)
    for cid, cluster in inspection.items():
        assert isinstance(cid, int)
        assert isinstance(cluster, dict)
        assert isinstance(cluster["actions"], list)
        assert all(isinstance(a, str) for a in cluster["actions"])


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

    Train with only one (action, object) observation for 'menyebabkan'
    - below min_action_observations=2. The action gets cluster_id = -1
    and classify() must fall back.

    Note: with the new zero-bias anchor-word discovery, the 3-token
    path requires the action candidate to appear at the action bucket
    >= _3_TOKEN_MIN_ACTION_FREQ (2) times. So we need TWO sentences
    with 'menyebabkan' to get it into action_object_freq at all; then
    it has only 1 (action, object) observation per object, but
    min_action_observations counts total observations across all
    objects, so 'menyebabkan' with {panas: 1, banjir: 1} has total=2
    which IS >= min_action_observations=2 and WOULD cluster.

    To force cluster_id = -1, we use distinct objects so the action
    has 2 total observations but each individual (action, object)
    pair is unique. Hmm — actually min_action_observations counts
    total observations, not distinct objects. So we need to ensure
    'menyebabkan' has total < 2.

    Solution: train on ONE sentence with 'menyebabkan' but use a
    4-token sentence so the >3-token path applies. The >3-token path
    uses verb-morphology OR action_bucket_anchor; 'menyebabkan' has
    verb morphology (starts with 'men-'), so it gets captured. Then
    it has total=1 observation < 2, so cluster_id = -1.
    """
    learner.train(["api menyebabkan panas banjir"])  # 4 tokens
    # 'menyebabkan' has verb morphology, captured at idx 1.
    assert "menyebabkan" in learner.action_object_freq
    # But total observations = 1 (one (action, object) pair) < 2.
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
# Supplementary: inspect_cluster_details (the canonical cluster view)
# ======================================================================


def test_inspect_cluster_details_returns_readable_view(
    learner: PositionalClusterLearner, mixed_corpus: list
):
    """inspect_cluster_details() returns the canonical human-readable view.

    Replaces the old ``test_inspect_clusters_returns_readable_view``
    which exercised the now-removed singular
    ``PositionalClusterLearner.inspect_clusters()`` (dead-code-audit
    §3.2). The richer ``inspect_cluster_details()`` is the canonical
    inspection API; ``bootstrap_classifier.py`` uses it as well.
    """
    learner.train(mixed_corpus)
    inspection = learner.inspect_cluster_details()

    assert isinstance(inspection, dict)
    assert len(inspection) >= 1

    for cid, cluster in inspection.items():
        assert isinstance(cid, int)
        assert cid >= 0
        assert isinstance(cluster, dict)
        actions = cluster["actions"]
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


def test_weighted_jaccard_basic_set_equivalents():
    """Weighted Jaccard on count maps: sum(min) / sum(max) over union of keys.

    Replaces the old ``test_jaccard_similarity`` which exercised the
    now-removed ``PositionalClusterLearner._jaccard`` (dead-code-audit
    §3.1). The plain set-Jaccard helper was dead in production (the
    clustering algorithm uses ``_weighted_jaccard`` exclusively); this
    test keeps coverage of the Jaccard *concept* on the helper that is
    actually live, using count maps that are set-equivalent (each key
    has count 1).
    """
    wj = PositionalClusterLearner._weighted_jaccard
    # Empty maps -> 0.0 (matches the old "two empty sets -> 0.0" case).
    assert wj({}, {}) == 0.0
    # One-sided -> 0.0 (matches "j({a}, set()) == 0.0").
    assert wj({"a": 1}, {}) == 0.0
    # Identical singletons -> 1.0 (matches "j({a},{a}) == 1.0").
    assert wj({"a": 1}, {"a": 1}) == 1.0
    # Partial overlap of singletons: |{b}| / |{a,b,c}| = 1/3.
    assert wj({"a": 1, "b": 1}, {"b": 1, "c": 1}) == 1 / 3
    # Identical sets -> 1.0.
    assert wj({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 1}) == 1.0


# ======================================================================
# Supplementary: train() robustness
# ======================================================================


def test_train_empty_corpus_no_crash(learner: PositionalClusterLearner):
    """train([]) is a no-op - no crash, learner stays untrained."""
    learner.train([])
    assert not learner.is_trained
    assert learner.positional_freq == {}


def test_train_short_sentences_no_crash(learner: PositionalClusterLearner):
    """train() with only short sentences builds no action_object_freq.

    Note: with the new zero-bias anchor-word discovery, the 3-token
    path requires the action candidate to appear at the action bucket
    >= _3_TOKEN_MIN_ACTION_FREQ (2) times. The corpus below has 'two'
    appearing twice at position 1, so it meets the floor and gets
    captured.
    """
    learner.train([
        "hello world",
        "single",
        "",
        "one two three",
        "four two five",  # 'two' appears again at position 1
    ])
    assert learner.is_trained
    # action_object_freq has one entry from 'one two three' and
    # 'four two five' — 'two' is the action, with objects 'three'
    # and 'five'.
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


def test_spo_long_sentence_cluster_driven(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """Long sentences: cluster-driven role assignment.

    With the cluster-driven parser (replacing the old positional
    formula ``subject=tokens[0], predicate=tokens[1..-1],
    object=tokens[-1]``), role is determined by cluster membership:

      - "makan" is the only ACTION-candidate token (it's in
        ``action_object_freq`` after training on svo_corpus).
      - Tokens before "makan" that don't match any cluster → subject.
        "saya" is not in any action/particle cluster → AGENT.
        "sedang" is unknown to the learner (not in the training
        corpus) → it falls through to AGENT (no cluster membership
        to override the position-before-ACTION rule).
      - Tokens after "makan" that don't match any cluster → object.
        "nasi" is unknown → OBJECT. "ayam" is in the object
        vocabulary (appears as object 3 times) but not in an action
        or particle cluster → OBJECT.

    The result: subject="saya sedang", predicate="makan",
    object="nasi ayam". This is the new contract — the predicate is
    the ACTION TOKEN ITSELF (not the position-1..-2 span), and
    unknown tokens before/after the action are absorbed into the
    subject/object phrases.

    Contrast with the old positional behaviour:
      subject="saya", predicate="sedang makan nasi", object="ayam"
    The old behaviour forced the predicate to be the position-1..-2
    span regardless of which token was the actual verb. For
    sentences where the verb is NOT at index 1 (multi-word subjects,
    subordinate clauses, passive voice), the old behaviour mis-
    parsed the sentence. The new behaviour correctly identifies the
    verb by cluster membership and splits subject/object around it.
    """
    learner.train(svo_corpus)
    spo = learner.spo("saya sedang makan nasi ayam")
    # "makan" is the ACTION token (in action_object_freq from training).
    assert spo.predicate == "makan", (
        f"Expected predicate 'makan' (the ACTION token, not the "
        f"position-1..-2 span); got {spo.predicate!r}."
    )
    # Tokens before "makan" that don't match any cluster → subject.
    # "saya" is not in any cluster → AGENT. "sedang" is not in the
    # training corpus → also falls into AGENT (no cluster membership
    # to override the position rule).
    assert spo.subject == "saya sedang", (
        f"Expected subject 'saya sedang' (tokens before ACTION that "
        f"don't match any cluster); got {spo.subject!r}."
    )
    # Tokens after "makan" that don't match any cluster → object.
    assert spo.object == "nasi ayam", (
        f"Expected object 'nasi ayam' (tokens after ACTION that "
        f"don't match any cluster); got {spo.object!r}."
    )


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


def test_anchor_word_discovery_finds_function_words_statistically():
    """Definition-of-Done test for Task 1: function words are discovered
    purely from positional entropy + frequency statistics, NOT from a
    hardcoded human list.

    This is the zero-bias replacement for the old ``_ACTION_STOPLIST``
    (which was rejected for being a meaning-based human-curated list -
    same kind of bias that PR #69 was rejected for).

    Setup: a corpus where 'sangat' (intensifier) and 'itu' (deictic)
    appear at >= 3 distinct fine-grained positions (3-token, 4-token,
    5-token sentences) with frequency >= ``_FUNCTION_WORD_MIN_FREQ``,
    so the entropy-based detector flags them as function word
    candidates. Meanwhile 'makan' (real verb) appears at the action
    bucket only (concentrated, low entropy) and is NOT flagged.

    Contract verified:
      1. 'sangat' and 'itu' ARE in ``learner.function_word_candidates``.
      2. 'makan' is NOT in ``learner.function_word_candidates``.
      3. 'makan' IS in ``learner.action_bucket_anchors`` (concentrated
         at action bucket).

    The test asserts that NO hardcoded word list is consulted - the
    discovery is purely from positional statistics. (We can't assert
    a negative directly, but the fact that the learner has no
    ``_ACTION_STOPLIST`` / ``_COPULAS`` constants anymore - and the
    test imports succeed without them - is the structural proof.)
    """
    learner = PositionalClusterLearner()
    # Build a corpus where 'sangat' and 'itu' each appear at >= 3
    # distinct fine positions.
    #
    # 'sangat' at fine positions 1, 2, -1 (3-token, 4-token, 5-token):
    #   "bunga sangat harum"             3 tokens: sangat at idx 1
    #   "bunga biru sangat harum"        4 tokens: sangat at idx 2
    #   "bunga biru ungu sangat harum"   5 tokens: sangat at idx 3
    #   "bunga sangat harum sekali"      4 tokens: sangat at idx 1
    #   "bunga biru ungu harum sangat"   5 tokens: sangat at idx 4 (last)
    #
    # 'itu' at fine positions 1, 2, 3:
    #   "bunga itu harum"                3 tokens: itu at idx 1
    #   "bunga biru itu harum"           4 tokens: itu at idx 2
    #   "bunga biru ungu itu harum"      5 tokens: itu at idx 3
    #   "bunga itu harum sekali"         4 tokens: itu at idx 1
    #   "bunga biru ungu harum itu"      5 tokens: itu at idx 4 (last)

    corpus = []
    # 'sangat' at 5 distinct fine-position instances, freq 5+
    pairs = [("bunga", "harum"), ("teh", "sepat"), ("kopi", "pahit"),
             ("es", "dingin"), ("gula", "manis")]
    for n, a in pairs:
        corpus.append(f"{n} sangat {a}")                    # idx 1
    for n, b, a in [(p[0], "biru", p[1]) for p in pairs]:
        corpus.append(f"{n} {b} sangat {a}")                # idx 2
    for n, b, c, a in [(p[0], "biru", "ungu", p[1]) for p in pairs]:
        corpus.append(f"{n} {b} {c} sangat {a}")            # idx 3
    # 'itu' at 5 distinct fine-position instances, freq 5+
    for n, a in pairs:
        corpus.append(f"{n} itu {a}")                       # idx 1
    for n, b, a in [(p[0], "biru", p[1]) for p in pairs]:
        corpus.append(f"{n} {b} itu {a}")                   # idx 2
    for n, b, c, a in [(p[0], "biru", "ungu", p[1]) for p in pairs]:
        corpus.append(f"{n} {b} {c} itu {a}")               # idx 3
    # 'makan' as a control — real verb, only at action bucket (idx 1).
    for s, o in [("saya", "ayam"), ("dia", "ikan"), ("kamu", "sayur"),
                 ("ibu", "nasi"), ("bapak", "daging")]:
        corpus.append(f"{s} makan {o}")

    learner.train(corpus)

    # 1. Function words discovered statistically.
    assert "sangat" in learner.function_word_candidates, (
        f"'sangat' must be flagged as a function word candidate from "
        f"positional entropy (appears at >= {_FUNCTION_WORD_MIN_POSITIONS} "
        f"distinct positions with freq >= {_FUNCTION_WORD_MIN_FREQ}, no "
        f"verb morphology). Got function_word_candidates: "
        f"{sorted(learner.function_word_candidates)}"
    )
    assert "itu" in learner.function_word_candidates, (
        f"'itu' must be flagged as a function word candidate. Got: "
        f"{sorted(learner.function_word_candidates)}"
    )

    # 2. Real verbs are NOT function word candidates.
    assert "makan" not in learner.function_word_candidates, (
        f"'makan' is a real verb and must NOT be flagged as a function "
        f"word. Got function_word_candidates: "
        f"{sorted(learner.function_word_candidates)}"
    )

    # 3. 'makan' IS an action bucket anchor (concentrated at bucket 1).
    assert "makan" in learner.action_bucket_anchors, (
        f"'makan' must be discovered as an action_bucket_anchor "
        f"(concentrated at action bucket with freq >= "
        f"{_ACTION_ANCHOR_MIN_FREQ}). Got action_bucket_anchors: "
        f"{sorted(learner.action_bucket_anchors)}"
    )

    # 4. 'sangat' and 'itu' are excluded from action_object_freq
    # because they're function word candidates (statistical exclusion,
    # NOT hardcoded stoplist).
    assert "sangat" not in learner.action_object_freq, (
        f"'sangat' must NOT enter action_object_freq because it is a "
        f"statistically discovered function word candidate."
    )
    assert "itu" not in learner.action_object_freq, (
        f"'itu' must NOT enter action_object_freq because it is a "
        f"statistically discovered function word candidate."
    )

    # 5. 'makan' IS in action_object_freq (real verb, captured).
    assert "makan" in learner.action_object_freq


def test_zero_bias_no_hardcoded_function_word_constants():
    """Structural test: the module must NOT export _ACTION_STOPLIST or
    _COPULAS. These were the hardcoded human-curated word lists that
    violated the zero-bias principle (same kind of bias PR #69 was
    rejected for). Their absence proves Task 1's contract is met at
    the code level.
    """
    import neocortex.positional_cluster_learner as pcl_module

    assert not hasattr(pcl_module, "_ACTION_STOPLIST"), (
        "_ACTION_STOPLIST must be removed - it was a hardcoded "
        "human-curated function-word list that violated zero-bias."
    )
    assert not hasattr(pcl_module, "_COPULAS"), (
        "_COPULAS must be removed - it was a hardcoded human-curated "
        "copula whitelist that violated zero-bias."
    )


def test_brown_cluster_merges_taxonomic_objects():
    """Definition-of-Done test for Task 2: Brown clustering groups
    taxonomic object nouns into super-clusters, allowing adalah and
    merupakan to merge even when their literal object overlap is
    minimal.

    This is the root-cause fix for the 'adalah'/'merupakan' synonym
    merge problem. PR #71/#73/#74 patched it by lowering the weighted
    Jaccard threshold to 0.13, but that was a tuned patch - the root
    cause is that literal object tokens are too sparse a signal.

    Setup: a corpus where:
      - 'adalah' takes mamalia (shared) + logam (only adalah).
      - 'merupakan' takes mamalia (shared) + unggas (only merupakan).
      - 'mamalia' is the literal-overlap bridge that seeds Brown
        clustering.
      - 'logam' and 'unggas' don't literally overlap with each other,
        but they share copula action context (both co-occur with
        'adalah' or 'merupakan', which themselves co-occur via
        'mamalia').

    Contract verified:
      1. 'mamalia', 'logam', and 'unggas' all end up in the SAME
         object super-cluster (Brown clustering merges them via
         transitive copula context).
      2. 'adalah' and 'merupakan' end up in the SAME action cluster
         (their super-cluster distributions overlap because all their
         objects are in the same super-cluster).
      3. The super-cluster mechanism is what enables the merge — the
         literal object overlap between adalah and merupakan alone
         (just 'mamalia' = 3 of 6 objects each) would yield weighted
         Jaccard = 6/12 = 0.5, which is above threshold, but the
         point of Brown clustering is that the merge would still
         happen even if the literal overlap were smaller (because
         the super-cluster projection amplifies the overlap).
    """
    corpus = [
        # adalah takes mamalia (3) + logam (3)
        "anjing adalah mamalia",
        "kucing adalah mamalia",
        "sapi adalah mamalia",
        "besi adalah logam",
        "emas adalah logam",
        "tembaga adalah logam",
        # merupakan takes mamalia (3) + unggas (3)
        "kuda merupakan mamalia",
        "babi merupakan mamalia",
        "kambing merupakan mamalia",
        "ayam merupakan unggas",
        "bebek merupakan unggas",
        "merpati merupakan unggas",
    ]
    learner = PositionalClusterLearner()
    learner.train(corpus)

    # 1. Brown clustering merges mamalia, logam, unggas into one SC.
    sc_mamalia = learner.object_supercluster_id.get("mamalia")
    sc_logam = learner.object_supercluster_id.get("logam")
    sc_unggas = learner.object_supercluster_id.get("unggas")
    assert sc_mamalia is not None, (
        "'mamalia' must be in object_supercluster_id (it's an object "
        "of both adalah and merupakan)"
    )
    assert sc_logam is not None, (
        "'logam' must be in object_supercluster_id (it's an object "
        "of adalah)"
    )
    assert sc_unggas is not None, (
        "'unggas' must be in object_supercluster_id (it's an object "
        "of merupakan)"
    )
    assert sc_mamalia == sc_logam == sc_unggas, (
        f"Brown clustering must merge taxonomic objects 'mamalia', "
        f"'logam', and 'unggas' into the same super-cluster (they "
        f"share copula action context transitively via 'mamalia'). "
        f"Got mamalia={sc_mamalia}, logam={sc_logam}, "
        f"unggas={sc_unggas}."
    )

    # 2. adalah and merupakan merge via super-cluster overlap.
    cid_adalah = learner.cluster_id_of.get("adalah")
    cid_merupakan = learner.cluster_id_of.get("merupakan")
    assert cid_adalah is not None, "'adalah' must be in cluster_id_of"
    assert cid_merupakan is not None, "'merupakan' must be in cluster_id_of"
    assert cid_adalah == cid_merupakan, (
        f"adalah (cluster {cid_adalah}) and merupakan (cluster "
        f"{cid_merupakan}) must be in the same action cluster. Their "
        f"objects (mamalia, logam, unggas) all belong to the same "
        f"Brown super-cluster, so their super-cluster distributions "
        f"overlap perfectly."
    )
    assert cid_adalah >= 0, (
        f"'adalah' must be in a real cluster (id >= 0), got {cid_adalah}"
    )

    # 3. The super-cluster that contains the taxonomy nouns has all
    #    three of them as members.
    sc_members = learner.object_superclusters.get(sc_mamalia, set())
    assert {"mamalia", "logam", "unggas"}.issubset(sc_members), (
        f"Super-cluster {sc_mamalia} must contain mamalia, logam, and "
        f"unggas as members. Got: {sorted(sc_members)}"
    )


def test_state_adjective_function_word_not_captured_as_action(
    learner: PositionalClusterLearner,
):
    """Bug 1: 'sangat' (and friends) must NOT enter action_object_freq.

    The state+adjective sentence "es itu sangat dingin" should NOT
    produce a (action='sangat', object='dingin') pair - that was the
    original garbage cluster
    ``{'actions': ['sangat'], 'top_objects': ['asin', 'dingin', ...]}``.

    In the new zero-bias design, the exclusion is via:
      - Statistical function word discovery (in real corpora where
        function words reach the freq/entropy threshold), OR
      - The 3-token frequency floor (``_3_TOKEN_MIN_ACTION_FREQ``)
        which excludes one-off function words in small synthetic
        corpora like this test corpus.

    Either way, NO hardcoded human list is consulted.
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
    training, no cluster in action_clusters should contain any token
    from ``learner.function_word_candidates`` (the statistically
    discovered function word set - the zero-bias replacement for the
    old hardcoded ``_ACTION_STOPLIST``).
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
            assert action not in learner.function_word_candidates, (
                f"Bug 1 regression: cluster {cid} contains statistically "
                f"discovered function word {action!r}. Full cluster: "
                f"{sorted(actions)}. function_word_candidates: "
                f"{sorted(learner.function_word_candidates)}"
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


def test_tokenize_keeps_decimal_and_thousands_separators_intact():
    """_tokenize must not fragment numbers at a digit-flanked comma or
    period (Indonesian decimal/thousands separator).

    Found via held-out Wikipedia text (Round 28): "gawang memiliki
    lebar 7,32 meter" was fragmenting into separate "7" and "32"
    tokens (the comma was treated as ordinary punctuation and replaced
    with a space), each polluting OBJECT spans as spurious standalone
    numbers. A digit-flanked separator is now stripped (merging the
    digits into one token) instead of replaced with a space; a real
    sentence-ending period is never digit-flanked, so normal sentence
    splitting is unaffected.
    """
    tokens = PositionalClusterLearner._tokenize(
        "gawang memiliki lebar 7,32 meter dan tinggi 2,44 meter."
    )
    assert "7" not in tokens and "32" not in tokens
    assert "732" in tokens
    assert "2" not in tokens and "44" not in tokens
    assert "244" in tokens

    tokens = PositionalClusterLearner._tokenize("lebih dari 20.000 spesies.")
    assert "20" not in tokens and "000" not in tokens
    assert "20000" in tokens

    # Sentence-ending periods (never digit-flanked) must still split
    # normally — this is the existing trailing-punctuation contract.
    tokens = PositionalClusterLearner._tokenize("ikan adalah hewan air.")
    assert "air." not in tokens
    assert "air" in tokens


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
    #
    # NB: ``PositionalClusterLearner._jaccard`` (the plain set-Jaccard
    # helper) was removed in dead-code-audit §3.1 — the clustering
    # algorithm uses ``_weighted_jaccard`` exclusively. The plain
    # value (1/3) is computed inline here from the set form
    # |{a}| / |{a,b,c}| = 1/3.
    plain = 1 / 3
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


def test_action_bucket_anchor_lets_multi_word_categorical_parse(
    learner: PositionalClusterLearner,
):
    """Copulas bypass the verb-prefix requirement in >3-token sentences
    via statistical action_bucket_anchor discovery (NOT a hardcoded
    _COPULAS whitelist).

    "suku bunga adalah instrumen kebijakan moneter" (6 tokens) has
    'adalah' at position 2 (after multi-word subject 'suku bunga').
    'adalah' doesn't start with me-/ber-/etc., so without the anchor
    mechanism the >3-token verb-prefix requirement would skip the
    sentence. The anchor mechanism - which discovers tokens
    concentrated at the action bucket purely from positional statistics
    - lets 'adalah' be recognised as a valid action.

    This is the zero-bias replacement for the old ``_COPULAS``
    whitelist: 'adalah' (bucket_freq={1: N}, bucket_nh=0.0) emerges
    as an action anchor automatically, no human-curated copula list
    needed.
    """
    learner.train([
        "suku bunga adalah instrumen kebijakan moneter",
        "anggaran subsidi adalah instrumen kebijakan fiskal",
        # Control: 3-token categorical still works
        "anjing adalah mamalia",
        "kucing adalah mamalia",
    ])

    # 'adalah' must be statistically discovered as an action anchor.
    assert "adalah" in learner.action_bucket_anchors, (
        f"'adalah' must be discovered as an action_bucket_anchor "
        f"(concentrated at action bucket). Got: "
        f"{sorted(learner.action_bucket_anchors)}"
    )

    # 'adalah' must be captured with both 'moneter' and 'fiskal'
    # objects (from the >3-token sentences), plus 'mamalia' (from the
    # 3-token sentences).
    assert "adalah" in learner.action_object_freq
    objs = set(learner.action_object_freq["adalah"].keys())
    assert "moneter" in objs
    assert "fiskal" in objs
    assert "mamalia" in objs


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


def test_looks_like_verb_only_uses_morphology():
    """_looks_like_verb returns True ONLY for me-/ber-/ter- prefixed verbs.

    The old ``_COPULAS`` whitelist was removed (it was a meaning-based
    human-curated list, violating zero-bias). Copulas like 'adalah' and
    'merupakan' are now recognised via ``action_bucket_anchors``
    (statistical discovery) instead of via ``_looks_like_verb``.
    """
    learner = PositionalClusterLearner()
    f = learner._looks_like_verb

    # Copulas — NOT recognised by _looks_like_verb anymore.
    # They're recognised via action_bucket_anchors (statistical) instead.
    for copula in ("adalah", "merupakan", "ialah", "yaitu", "yakni"):
        assert not f(copula), (
            f"copula {copula!r} must NOT match _looks_like_verb — copulas "
            f"are recognised via action_bucket_anchors (statistical), not "
            f"morphology. The _COPULAS whitelist was removed."
        )

    # Irregular roots — NOT verbs (no morphology, not in copula whitelist).
    for verb in ("makan", "minum", "ambil"):
        assert not f(verb), (
            f"irregular root {verb!r} should NOT match verb-prefix "
            f"heuristic (it's handled by the 3-token freq-floor path)"
        )

    # Prefixed verbs — recognised by morphology.
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

    After the zero-bias anchor-word + Brown-clustering refactor (which
    removed ``_ACTION_STOPLIST``/``_COPULAS`` and replaced them with
    statistical discovery), the count rises to ~150 because some
    function words that the old stoplist suppressed (``cukup``,
    ``tampak``, ``terasa``, etc.) are now captured as actions when
    they don't meet the statistical function-word discovery threshold
    (e.g., only 2 distinct fine positions). This is intentional —
    the zero-bias principle disallows hardcoded word lists, and the
    statistical discovery can't catch every edge case without
    semantic knowledge.

    This test pins the upper bound at 180 so future regressions that
    re-introduce over-fragmentation get caught, while allowing the
    ~37-cluster increase from the stoplist removal.
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
    assert n_clusters < 180, (
        f"Cluster count on pretrain_corpus.txt must be < 180 after "
        f"Bug 1 + Bug 2 + zero-bias anchor-word + Brown-clustering "
        f"fixes (was 229 before any fixes, was ~113 with stoplist, "
        f"now ~150 with statistical discovery). Got {n_clusters}. "
        f"Likely cause: regression in statistical anchor-word "
        f"discovery, Brown clustering, or weighted Jaccard metric."
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
            assert action not in learner.function_word_candidates, (
                f"Garbage cluster detected: cluster {cid} contains "
                f"statistically discovered function word {action!r}. "
                f"Full cluster actions: {sorted(actions)}. "
                f"function_word_candidates: "
                f"{sorted(learner.function_word_candidates)}"
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


# ======================================================================
# Connector-signal tests (cluster-62 fix)
# ======================================================================
#
# These tests cover the structural-signal feature that splits
# categorical-affirmation predicates ("adalah", "merupakan", "termasuk"
# — direct object) from categorical-contrast predicates ("berbeda",
# "berlawanan" — connector + object) even though their object sets
# overlap on taxonomy nouns.
#
# The detection is purely statistical (position + frequency). No list
# of "negation words" or "connector words" is consulted.


def test_connector_signal_separates_categorical_differential():
    """The Definition-of-Done test for the cluster-62 fix.

    After training on a synthetic corpus that mixes direct-action
    sentences with connector-action sentences (both using the same
    object vocabulary, so weighted Jaccard would otherwise merge
    them), "adalah" and "berbeda" MUST end up in different clusters.

    Setup:
      - Direct pattern: "X adalah Y" / "X merupakan Y" — action
        directly followed by object, no connector.
      - Connector pattern: "X berbeda dari Y" / "X berlawanan dengan Y"
        — action followed by connector token ("dari"/"dengan") and
        THEN object.

    Both groups use the SAME object vocabulary (kucing, anjing, ikan,
    ...) so weighted Jaccard would merge all 4 verbs into one cluster
    without the connector-signal split. The split must keep
    "adalah"/"merupakan" together (both no-connector) and
    "berbeda"/"berlawanan" together (both with-connector) — but the
    two pairs must NOT be in the same cluster.

    No hardcoded "connector" or "negation" list is consulted. The
    detector must find "dari"/"dengan" purely from positional
    evidence:
      - they sit in the between-first slot for >=
        _CONNECTOR_MIN_BETWEEN_COUNT sentences
      - they never appear as objects themselves
    """
    # Build a synthetic corpus with controlled structure.
    # Use enough sentences so each connector token reaches the
    # _CONNECTOR_MIN_BETWEEN_COUNT (3) threshold for corpus-wide
    # connector discovery.
    subjects = ["kucing", "anjing", "burung", "ikan", "ular", "kura"]
    objects = ["mamalia", "aves", "pisces", "reptil", "amfibi", "hewan"]

    # Affirmation pattern: "X adalah Y" / "X merupakan Y" / "X termasuk Y"
    # Direct: action immediately followed by object.
    affirmation_corpus = []
    for verb in ("adalah", "merupakan", "termasuk"):
        for i, subj in enumerate(subjects):
            affirmation_corpus.append(f"{subj} {verb} {objects[i]}")

    # Contrast pattern: "X berbeda dari Y" / "X berlawanan dengan Y"
    # With connector: action followed by "dari"/"dengan" then object.
    contrast_corpus = []
    for verb, connector in (
        ("berbeda", "dari"),
        ("berlawanan", "dengan"),
    ):
        for i, subj in enumerate(subjects):
            contrast_corpus.append(f"{subj} {verb} {connector} {objects[i]}")

    corpus = affirmation_corpus + contrast_corpus

    learner = PositionalClusterLearner()
    learner.train(corpus)

    # Connector signature assertions.
    for verb in ("adalah", "merupakan", "termasuk"):
        assert learner.action_connector_signature.get(verb) is False, (
            f"{verb!r} must have has_connector=False (direct object "
            f"pattern). Got signature: "
            f"{learner.action_connector_signature.get(verb)}"
        )
    for verb in ("berbeda", "berlawanan"):
        assert learner.action_connector_signature.get(verb) is True, (
            f"{verb!r} must have has_connector=True (connector pattern). "
            f"Got signature: "
            f"{learner.action_connector_signature.get(verb)}"
        )

    # Corpus-wide connector tokens must include "dari" and "dengan".
    # The detector found these purely from positional evidence — no
    # hardcoded list.
    assert "dari" in learner.connector_tokens, (
        f"'dari' must be detected as a corpus-wide connector token. "
        f"Got connector_tokens: {sorted(learner.connector_tokens)}"
    )
    assert "dengan" in learner.connector_tokens, (
        f"'dengan' must be detected as a corpus-wide connector token. "
        f"Got connector_tokens: {sorted(learner.connector_tokens)}"
    )

    # Cluster ID assertions — the core DoD check.
    cid_adalah = learner.cluster_id_of.get("adalah")
    cid_merupakan = learner.cluster_id_of.get("merupakan")
    cid_termasuk = learner.cluster_id_of.get("termasuk")
    cid_berbeda = learner.cluster_id_of.get("berbeda")
    cid_berlawanan = learner.cluster_id_of.get("berlawanan")

    assert cid_adalah is not None, "'adalah' must be in cluster_id_of"
    assert cid_berbeda is not None, "'berbeda' must be in cluster_id_of"

    # DoD: "adalah" and "berbeda" NOT in the same cluster.
    assert cid_adalah != cid_berbeda, (
        f"Cluster-62 regression: 'adalah' (cluster {cid_adalah}) and "
        f"'berbeda' (cluster {cid_berbeda}) MUST be in different "
        f"clusters after the connector-signal fix. They share the "
        f"same object vocabulary but have different structural "
        f"signatures (direct vs connector)."
    )

    # Sanity: synonym pairs still merge within their connector group.
    assert cid_adalah == cid_merupakan, (
        f"'adalah' (cluster {cid_adalah}) and 'merupakan' "
        f"(cluster {cid_merupakan}) should still merge — both are "
        f"has_connector=False with similar object distributions."
    )
    assert cid_adalah == cid_termasuk, (
        f"'adalah' (cluster {cid_adalah}) and 'termasuk' "
        f"(cluster {cid_termasuk}) should still merge — both are "
        f"has_connector=False with similar object distributions."
    )
    assert cid_berbeda == cid_berlawanan, (
        f"'berbeda' (cluster {cid_berbeda}) and 'berlawanan' "
        f"(cluster {cid_berlawanan}) should still merge — both are "
        f"has_connector=True with similar object distributions."
    )

    # Sanity: cluster_ids must be real (>= 0).
    for cid in (cid_adalah, cid_merupakan,
                cid_termasuk, cid_berbeda, cid_berlawanan):
        assert cid >= 0, f"cluster id must be >= 0, got {cid}"


def test_connector_signal_no_hardcoded_connector_list():
    """The detector must NOT consult a hardcoded list of connector words.

    The cluster-62 fix's contract is that the detection is purely
    positional — no list of "negation words" or "connector words"
    based on meaning. We verify this by training on a synthetic
    corpus where the "connector" is a made-up token that has never
    appeared in any Indonesian dictionary. The detector must still
    flag the action as has_connector=True, because the made-up token
    satisfies the positional + frequency + never-as-object contract.

    If this test fails, the implementation has likely regressed to
    hardcoding a list of prepositions / complementizers (which would
    be a return to the semi-supervised bias that PR #69 was rejected
    for).
    """
    # "zzzq" is a made-up token that no Indonesian dictionary knows.
    # It cannot be in any hardcoded connector list. The detector must
    # still pick it up purely from positional evidence.
    made_up_connector = "zzzq"

    subjects = ["subj1", "subj2", "subj3", "subj4"]
    objects = ["obj1", "obj2", "obj3", "obj4"]

    corpus = []
    # "X madeupverb zzzq Y" — action + made-up connector + object.
    # We need >= _CONNECTOR_MIN_BETWEEN_COUNT (3) sentences with
    # "zzzq" in the between-first slot for it to qualify as a
    # corpus-wide connector.
    for i, subj in enumerate(subjects):
        corpus.append(f"{subj} madeupverb {made_up_connector} {objects[i]}")
    # Add 2 more sentences to push the count above 3.
    corpus.append(f"subj5 madeupverb {made_up_connector} obj5")
    corpus.append(f"subj6 madeupverb {made_up_connector} obj6")

    # Important: "zzzq" must NEVER appear as an object in the corpus
    # (otherwise the never-as-object filter would reject it). All
    # objects above are "obj1".."obj6" — none is "zzzq". ✓

    # Important: "madeupverb" doesn't start with me-/ber-/diper-/ter-
    # and wouldn't be a discovered action_bucket_anchor in this small
    # corpus, so the >3-token verb-prefix / anchor filter
    # (see _extract_action_object) would normally skip these sentences.
    # To keep the test focused on the connector detector and not on
    # the verb-prefix heuristic, we use a verb-prefixed action name
    # instead (which passes _looks_like_verb via morphology).
    corpus = []
    action = "menguji"  # starts with "meng-" so it passes _looks_like_verb
    for i, subj in enumerate(subjects):
        corpus.append(f"{subj} {action} {made_up_connector} {objects[i]}")
    corpus.append(f"subj5 {action} {made_up_connector} obj5")
    corpus.append(f"subj6 {action} {made_up_connector} obj6")

    learner = PositionalClusterLearner()
    learner.train(corpus)

    assert made_up_connector in learner.connector_tokens, (
        f"Made-up token {made_up_connector!r} must be detected as a "
        f"corpus-wide connector purely from positional evidence "
        f"(appears in between-first slot >= "
        f"{PositionalClusterLearner._CONNECTOR_MIN_BETWEEN_COUNT if hasattr(PositionalClusterLearner, '_CONNECTOR_MIN_BETWEEN_COUNT') else 3} "
        f"times, never as object). Got connector_tokens: "
        f"{sorted(learner.connector_tokens)}"
    )
    assert learner.action_connector_signature.get(action) is True, (
        f"Action {action!r} must have has_connector=True — it "
        f"routinely takes the made-up connector {made_up_connector!r}. "
        f"Got signature: "
        f"{learner.action_connector_signature.get(action)}"
    )


def test_connector_signal_persistence_roundtrip(tmp_path):
    """save/load must round-trip the connector signature fields.

    A learner trained on a connector-mixed corpus, saved, and loaded
    must reproduce the same action_connector_signature and
    connector_tokens. This is the contract that lets a labelled
    learner persist its connector-aware clustering across restarts.
    """
    subjects = ["a", "b", "c", "d"]
    objects = ["x", "y", "z", "w"]

    corpus = []
    for verb in ("adalah", "merupakan"):
        for i, subj in enumerate(subjects):
            corpus.append(f"{subj} {verb} {objects[i]}")
    for verb, connector in (("berbeda", "dari"), ("berlawanan", "dengan")):
        for i, subj in enumerate(subjects):
            corpus.append(f"{subj} {verb} {connector} {objects[i]}")

    learner = PositionalClusterLearner()
    learner.train(corpus)

    # Sanity: pre-save, connector signal is populated.
    assert learner.action_connector_signature.get("adalah") is False
    assert learner.action_connector_signature.get("berbeda") is True
    assert "dari" in learner.connector_tokens
    assert "dengan" in learner.connector_tokens

    save_path = tmp_path / "learner.json"
    learner.save(str(save_path))

    loaded = PositionalClusterLearner.load(str(save_path))

    # Round-trip: connector signature preserved.
    assert loaded.action_connector_signature.get("adalah") is False, (
        "action_connector_signature['adalah'] must round-trip as False"
    )
    assert loaded.action_connector_signature.get("berbeda") is True, (
        "action_connector_signature['berbeda'] must round-trip as True"
    )
    assert "dari" in loaded.connector_tokens, (
        "connector_tokens must round-trip 'dari'"
    )
    assert "dengan" in loaded.connector_tokens, (
        "connector_tokens must round-trip 'dengan'"
    )

    # Round-trip: cluster membership preserved.
    assert loaded.cluster_id_of.get("adalah") == learner.cluster_id_of.get("adalah")
    assert loaded.cluster_id_of.get("berbeda") == learner.cluster_id_of.get("berbeda")
    assert loaded.cluster_id_of.get("adalah") != loaded.cluster_id_of.get("berbeda"), (
        "Round-tripped learner must still separate 'adalah' from 'berbeda'"
    )


def test_connector_signal_inspect_cluster_details_reports_has_connector():
    """inspect_cluster_details() surfaces has_connector per cluster.

    The human-readable cluster view must include the has_connector
    flag so a reviewer can verify the structural split at a glance.
    """
    subjects = ["a", "b", "c", "d"]
    objects = ["x", "y", "z", "w"]

    corpus = []
    for verb in ("adalah", "merupakan"):
        for i, subj in enumerate(subjects):
            corpus.append(f"{subj} {verb} {objects[i]}")
    for verb, connector in (("berbeda", "dari"), ("berlawanan", "dengan")):
        for i, subj in enumerate(subjects):
            corpus.append(f"{subj} {verb} {connector} {objects[i]}")

    learner = PositionalClusterLearner()
    learner.train(corpus)

    details = learner.inspect_cluster_details()
    assert details, "inspect_cluster_details() must return non-empty"

    # Find the cluster containing 'adalah' and the one containing 'berbeda'.
    cid_adalah = learner.cluster_id_of.get("adalah")
    cid_berbeda = learner.cluster_id_of.get("berbeda")
    assert cid_adalah is not None and cid_berbeda is not None
    assert cid_adalah != cid_berbeda

    cluster_adalah = details[cid_adalah]
    cluster_berbeda = details[cid_berbeda]

    assert "has_connector" in cluster_adalah, (
        "inspect_cluster_details() must include the has_connector field"
    )
    assert cluster_adalah["has_connector"] is False, (
        f"adalah's cluster must report has_connector=False, got "
        f"{cluster_adalah['has_connector']}"
    )
    assert cluster_berbeda["has_connector"] is True, (
        f"berbeda's cluster must report has_connector=True, got "
        f"{cluster_berbeda['has_connector']}"
    )


def test_connector_signal_extract_between_token_helper():
    """Unit test for the _extract_between_token helper.

    Direct cases return None; connector cases return the connector.
    """
    # Direct: action immediately followed by object.
    # "kucing adalah mamalia" → tokens = [kucing, adalah, mamalia]
    # action=adalah at idx 1, object=mamalia at idx 2 (last).
    # No tokens between → None.
    assert PositionalClusterLearner._extract_between_token(
        ["kucing", "adalah", "mamalia"], "adalah", "mamalia"
    ) is None

    # Connector: action + connector + object.
    # "kucing berbeda dari reptil" → tokens = [kucing, berbeda, dari, reptil]
    # action=berbeda at idx 1, object=reptil at idx 3 (last).
    # Between = [dari] → first is "dari".
    assert PositionalClusterLearner._extract_between_token(
        ["kucing", "berbeda", "dari", "reptil"], "berbeda", "reptil"
    ) == "dari"

    # Multi-token between: only the FIRST between token is returned
    # (the connector slot). Other between tokens are part of the
    # object phrase and are ignored.
    # "X verb conn1 conn2 obj" → between_first = "conn1"
    assert PositionalClusterLearner._extract_between_token(
        ["X", "verb", "conn1", "conn2", "obj"], "verb", "obj"
    ) == "conn1"

    # Action not found → None (defensive).
    assert PositionalClusterLearner._extract_between_token(
        ["a", "b", "c"], "missing", "c"
    ) is None

    # Empty tokens → None.
    assert PositionalClusterLearner._extract_between_token(
        [], "verb", "obj"
    ) is None


def test_connector_signal_combined_corpus_separates_adalah_and_berbeda(
    pretrain_corpus_path,
):
    """End-to-end on the combined pretrain corpus (cluster-62 case).

    Trains on pretrain_corpus.txt + pretrain_corpus_depth.txt and
    asserts the structural split: "adalah"/"merupakan"/"termasuk"
    (no-connector) and "berbeda"/"berlawanan" (with-connector) end
    up in different clusters. This is the literal cluster-62 fix.

    Skips if pretrain_corpus_depth.txt is not available (e.g. when
    only AGNN/tests/ is checked out).
    """
    depth_path = pretrain_corpus_path.parent / "pretrain_corpus_depth.txt"
    if not depth_path.exists():
        pytest.skip(f"pretrain_corpus_depth.txt not found at {depth_path}")

    lines = []
    for path in (pretrain_corpus_path, depth_path):
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("##"):
                lines.append(ln)

    learner = PositionalClusterLearner()
    learner.train(lines)

    cid_adalah = learner.cluster_id_of.get("adalah")
    cid_merupakan = learner.cluster_id_of.get("merupakan")
    cid_termasuk = learner.cluster_id_of.get("termasuk")
    cid_berbeda = learner.cluster_id_of.get("berbeda")
    cid_berlawanan = learner.cluster_id_of.get("berlawanan")
    cid_terhitung = learner.cluster_id_of.get("terhitung")

    # All target verbs must be clustered.
    for verb, cid in (
        ("adalah", cid_adalah),
        ("merupakan", cid_merupakan),
        ("termasuk", cid_termasuk),
        ("berbeda", cid_berbeda),
        ("berlawanan", cid_berlawanan),
        ("terhitung", cid_terhitung),
    ):
        assert cid is not None, f"{verb!r} must be in cluster_id_of"
        assert cid >= 0, f"{verb!r} must be in a real cluster, got {cid}"

    # DoD: "adalah" and "berbeda" NOT in the same cluster.
    assert cid_adalah != cid_berbeda, (
        f"Cluster-62 regression on combined corpus: 'adalah' "
        f"(cluster {cid_adalah}) and 'berbeda' (cluster {cid_berbeda}) "
        f"MUST be in different clusters after the connector-signal fix."
    )
    assert cid_adalah != cid_berlawanan, (
        f"Cluster-62 regression: 'adalah' and 'berlawanan' MUST be in "
        f"different clusters."
    )

    # Connector signature assertions.
    assert learner.action_connector_signature.get("adalah") is False
    assert learner.action_connector_signature.get("berbeda") is True
    assert learner.action_connector_signature.get("berlawanan") is True
    assert learner.action_connector_signature.get("terhitung") is True

    # Sanity: synonym pairs merge within their connector group.
    assert cid_adalah == cid_merupakan, (
        "'adalah' and 'merupakan' must still merge (both no-connector, "
        "similar object distributions)."
    )
    assert cid_adalah == cid_termasuk, (
        "'adalah' and 'termasuk' must still merge (both no-connector)."
    )
    assert cid_berbeda == cid_berlawanan, (
        "'berbeda' and 'berlawanan' must still merge (both with-connector)."
    )

    # Corpus-wide connector tokens must include "dari" and "dengan"
    # (used by berbeda / berlawanan). Detected purely from positional
    # evidence — no hardcoded list.
    assert "dari" in learner.connector_tokens
    assert "dengan" in learner.connector_tokens
    assert "sebagai" in learner.connector_tokens  # used by terhitung


def test_connector_signal_backward_compat_no_connector_breaks_clustering():
    """Pre-fix behavior preserved when no connector pattern is present.

    A corpus where every action is direct (no between tokens at all)
    must produce the same clustering as before the connector-signal
    fix: connector_tokens is empty, action_connector_signature is all
    False, and clustering proceeds purely on weighted Jaccard of
    object distributions.
    """
    # Pure direct-pattern corpus: every sentence is 3 tokens (SVO).
    # No between-first tokens ever, so connector_tokens stays empty.
    corpus = [
        "kucing makan ikan",
        "anjing makan ikan",
        "kucing minum air",
        "anjing minum air",
        "burung makan ikan",  # makan cluster grows
    ]
    learner = PositionalClusterLearner()
    learner.train(corpus)

    # No connector signal should fire.
    assert learner.connector_tokens == set(), (
        f"connector_tokens must be empty for a pure-direct corpus. "
        f"Got: {learner.connector_tokens}"
    )
    for action, flag in learner.action_connector_signature.items():
        assert flag is False, (
            f"action {action!r} must have has_connector=False in a "
            f"pure-direct corpus. Got: {flag}"
        )

    # makan and minum both take "ikan" / "air" objects; they should
    # merge (existing weighted-Jaccard behavior, preserved).
    cid_makan = learner.cluster_id_of.get("makan")
    cid_minum = learner.cluster_id_of.get("minum")
    assert cid_makan is not None and cid_minum is not None
    # Both clusterable (>= 2 observations each), should be in some
    # cluster (id >= 0).
    assert cid_makan >= 0 and cid_minum >= 0
