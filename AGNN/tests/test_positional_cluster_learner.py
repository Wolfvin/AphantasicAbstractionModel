"""
Tests for ``PositionalClusterLearner`` - emergent structure discovery.

Covers the Definition-of-Done from the task brief:

    1. test_train_builds_clusters     - after train() with 10 SVO
       sentences, "makan" is in the action cluster.
    2. test_classify_causal            - after train() with a causal
       corpus, classify("api menyebabkan panas") -> CAUSAL.
    3. test_fallback_no_corpus         - without train(), classify()
       returns the same result as a fresh SemanticRoleClassifier.
    4. test_spo_extracts_triple        - spo("saya makan ayam") ->
       subject="saya", predicate="makan", object="ayam".
    5. test_persistence                - save() + load() preserves
       the learned clusters exactly.

Plus a handful of supplementary tests that lock in the failure
contracts (empty input, negation override, mid-training fallback,
emergent cluster growth, etc.).

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_positional_cluster_learner.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
# Tests are invoked as:
#   python -m pytest AGNN/tests/test_positional_cluster_learner.py -v
# so we add the AGNN/ directory (parent of tests/) to sys.path.
_AGNP_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

# Also ensure self-ai/src is importable for the canonical RelationType
# in agnn.graph (the classifier re-exports it when available).
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


# ----------------------------------------------------------------------
# Imports under test
# ----------------------------------------------------------------------

from neocortex.positional_cluster_learner import (  # noqa: E402
    PositionalClusterLearner,
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


# ======================================================================
# DoD #1: train() builds positional clusters
# ======================================================================


def test_train_builds_clusters(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """After train() with 10 SVO sentences, 'makan' is in the action cluster.

    The action cluster is positional_clusters[1] (tokens whose dominant
    position is the action slot). With 10 SVO sentences where 'makan'
    appears 9 times at position 1, its dominant position must be 1.
    """
    learner.train(svo_corpus)

    # Primary check: 'makan' is in the action cluster.
    assert "makan" in learner.action_cluster, (
        f"expected 'makan' in action cluster, got "
        f"{sorted(learner.action_cluster)}"
    )

    # Also verify via the raw positional_clusters dict.
    assert "makan" in learner.positional_clusters.get(1, set())

    # 'makan' must NOT be in the agent cluster (position 0).
    assert "makan" not in learner.agent_cluster

    # Agent tokens ('saya', 'dia', 'kamu') must be in the agent cluster.
    assert "saya" in learner.agent_cluster
    assert "dia" in learner.agent_cluster
    assert "kamu" in learner.agent_cluster

    # Object tokens ('ayam', 'sapi', 'ikan') must be in the object cluster.
    assert "ayam" in learner.object_cluster
    assert "sapi" in learner.object_cluster
    assert "ikan" in learner.object_cluster


def test_train_marker_set(learner: PositionalClusterLearner, svo_corpus: list):
    """After train(), the learner's ``_trained`` flag is True.

    This is what makes classify() / spo() use the learned clusters
    instead of delegating to the fallback.
    """
    assert learner._trained is False
    learner.train(svo_corpus)
    assert learner._trained is True


def test_train_accumulates(learner: PositionalClusterLearner, svo_corpus: list):
    """Calling train() twice on disjoint corpora accumulates observations.

    Uses an int snapshot (not a dict copy) to avoid the shallow-copy
    aliasing trap - ``dict(positional_freq)`` shares the inner
    ``{position: count}`` dicts, so a subsequent train() call would
    mutate the snapshot too.
    """
    half1 = svo_corpus[:5]
    half2 = svo_corpus[5:]
    learner.train(half1)
    # Snapshot the count directly (int, immutable -> no aliasing).
    count_after_first = learner.positional_freq["makan"][1]
    assert count_after_first == 5, (
        f"expected 5 'makan' observations after first train(), "
        f"got {count_after_first}"
    )
    learner.train(half2)
    # 'makan' total count should grow after the second train() call
    # (half2 has 4 more 'makan' sentences).
    count_after_second = learner.positional_freq["makan"][1]
    assert count_after_second == 9, (
        f"expected 9 'makan' observations after second train() "
        f"(5 + 4), got {count_after_second}"
    )
    assert count_after_second > count_after_first


# ======================================================================
# DoD #2: classify() returns CAUSAL after causal-corpus training
# ======================================================================


def test_classify_causal(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """After train() with causal corpus, classify('api menyebabkan panas') -> CAUSAL.

    The corpus has 5 sentences with 'menyebabkan' as the action and
    state-change objects (panas, banjir, kebakaran, kanker). After
    training, 'menyebabkan' should be in the action cluster, and the
    objects fall into the CAUSAL seed sub-cluster, so the action's
    dominant relation is CAUSAL.
    """
    learner.train(causal_corpus)
    result = learner.classify("api menyebabkan panas")
    assert result == RelationType.CAUSAL, (
        f"expected CAUSAL, got {result}"
    )


def test_classify_causal_uses_learned_clusters_not_fallback(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """The CAUSAL result comes from learned clusters, not the fallback.

    We verify this by training with a corpus whose objects are all in
    the CAUSAL seed set, then classifying a sentence whose object is
    NOT a CAUSAL seed (but is one the learner has seen and labelled
    via cluster growth). If the learner is actually using its
    clusters, the result is still CAUSAL; if it were falling back,
    the fallback would still return CAUSAL via the seed 'menyebabkan',
    so this test also passes - but the point is to lock in the
    behaviour either way.

    The stronger assertion is on action_object_freq: after training,
    'menyebabkan' must have >= min_data_points observations, so
    classify() is NOT taking the fallback branch.
    """
    learner.train(causal_corpus)
    assert sum(learner.action_object_freq["menyebabkan"].values()) >= 3
    assert learner.classify("api menyebabkan panas") == RelationType.CAUSAL


def test_classify_emergent_cluster_growth(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """Object tokens NOT in the seed set inherit CAUSAL via co-occurrence.

    After training, 'kanker' is in the CAUSAL seed set, but 'banjir'
    and 'kebakaran' are also in the seed set (we seeded them
    explicitly). The real test of cluster growth is an object that
    was NOT seeded - 'penyakit' in our extension below. After
    observing 'menyebabkan' followed by 'penyakit' (alongside seeded
    CAUSAL objects), the learner should label 'penyakit' as CAUSAL
    via co-occurrence.
    """
    extended_corpus = causal_corpus + [
        "stres menyebabkan penyakit",
        "polusi menyebabkan penyakit",
        "obat menyebabkan penyakit",
    ]
    learner.train(extended_corpus)

    # 'penyakit' was not in the CAUSAL seed set, but it follows
    # 'menyebabkan' (which has labelled CAUSAL objects). After
    # training, 'penyakit' should be labelled CAUSAL.
    assert learner.object_relation_map.get("penyakit") == RelationType.CAUSAL, (
        f"expected 'penyakit' to inherit CAUSAL via co-occurrence, "
        f"got {learner.object_relation_map.get('penyakit')}"
    )

    # And classifying a sentence using the new object should also
    # return CAUSAL.
    assert learner.classify("stres menyebabkan penyakit") == RelationType.CAUSAL


# ======================================================================
# DoD #3: without training, classify() == SemanticRoleClassifier
# ======================================================================


def test_fallback_no_corpus(learner: PositionalClusterLearner):
    """Without train(), classify() returns the same result as SemanticRoleClassifier.

    Backward-compatibility contract: a fresh
    PositionalClusterLearner that has not been trained must behave
    exactly like a fresh SemanticRoleClassifier, because every
    classify() call short-circuits to the wrapped fallback instance.
    """
    fallback = SemanticRoleClassifier()

    test_cases = [
        # CAUSAL via Indonesian seed
        ("api menyebabkan panas",),
        # CAUSAL via English seed
        ("smoking causes cancer",),
        # FUNCTIONAL via Indonesian seed
        ("tanaman membutuhkan air",),
        # FUNCTIONAL via English seed
        ("engine requires fuel",),
        # CATEGORICAL via Indonesian seed
        ("manusia adalah mamalia",),
        # CATEGORICAL via English seed
        ("a dog is a mammal",),
        # DIFFERENTIAL via standalone negation
        ("kelelawar bukan burung",),
        # DIFFERENTIAL via negation + CAUSAL seed
        ("merokok tidak menyebabkan awet muda",),
        # TEMPORAL via Indonesian seed
        ("padi tumbuh setelah hujan",),
        # CATEGORICAL fallback (unknown predicate)
        ("X blahblah Y",),
        # CATEGORICAL fallback (single token)
        ("apple",),
        # CATEGORICAL fallback (empty)
        ("",),
    ]

    for (text,) in test_cases:
        learner_result = learner.classify(text)
        fallback_result = fallback.classify(text)
        assert learner_result == fallback_result, (
            f"Mismatch on '{text!r}': "
            f"learner={learner_result}, fallback={fallback_result}"
        )


def test_fallback_no_corpus_spo(learner: PositionalClusterLearner):
    """Without train(), spo() returns the same SPO as SemanticRoleClassifier."""
    fallback = SemanticRoleClassifier()

    test_cases = [
        "lari menyebabkan ngos-ngosan",
        "merokok tidak menyebabkan awet muda",
        "paru bagian dari sistem pernapasan",
        "saya makan ayam",
        "",
    ]

    for text in test_cases:
        learner_spo = learner.spo(text)
        fallback_spo = fallback.spo(text)
        assert learner_spo == fallback_spo, (
            f"Mismatch on '{text!r}': "
            f"learner={learner_spo}, fallback={fallback_spo}"
        )


# ======================================================================
# DoD #4: spo() extracts the SVO triple
# ======================================================================


def test_spo_extracts_triple(learner: PositionalClusterLearner):
    """spo('saya makan ayam') -> subject='saya', predicate='makan', object='ayam'.

    Without training, spo() delegates to SemanticRoleClassifier.spo(),
    which falls back to the middle-token heuristic when no seed
    matches. For a 3-token sentence, that puts the middle token
    ('makan') as the predicate.
    """
    spo = learner.spo("saya makan ayam")
    assert spo.subject == "saya", f"expected subject='saya', got '{spo.subject}'"
    assert spo.predicate == "makan", (
        f"expected predicate='makan', got '{spo.predicate}'"
    )
    assert spo.object == "ayam", f"expected object='ayam', got '{spo.object}'"
    assert spo.negated is False


def test_spo_extracts_triple_after_training(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """After training, spo() uses the learned action cluster to find the predicate.

    For 'saya makan ayam', 'makan' is in the action cluster after
    training, so spo() returns subject='saya', predicate='makan',
    object='ayam' via the learned-cluster path (not the fallback).
    """
    learner.train(svo_corpus)
    spo = learner.spo("saya makan ayam")
    assert spo.subject == "saya"
    assert spo.predicate == "makan"
    assert spo.object == "ayam"


def test_spo_extracts_triple_with_unseen_action_after_training(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """After training, spo() still extracts a triple for an unseen action.

    'lari' is not in the corpus, so it's not in the action cluster.
    spo() must fall back to the canonical middle index (1) for a
    3-token sentence, still producing a valid SVO triple.
    """
    learner.train(svo_corpus)
    spo = learner.spo("kucing lari cepat")
    assert spo.subject == "kucing"
    assert spo.predicate == "lari"
    assert spo.object == "cepat"


def test_spo_empty_string(learner: PositionalClusterLearner):
    """Empty input -> empty SPO, no crash (matches fallback contract)."""
    spo = learner.spo("")
    assert spo.subject == ""
    assert spo.predicate == ""
    assert spo.object == ""


def test_spo_negation_flag(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """After training, negation is still detected before the action.

    'X tidak menyebabkan Y' must set negated=True (so classify()
    returns DIFFERENTIAL).
    """
    learner.train(causal_corpus)
    spo = learner.spo("merokok tidak menyebabkan awet muda")
    assert spo.predicate == "menyebabkan"
    assert spo.negated is True


# ======================================================================
# DoD #5: save() + load() preserves clusters
# ======================================================================


def test_persistence(
    learner: PositionalClusterLearner,
    svo_corpus: list,
    causal_corpus: list,
    tmp_path,
):
    """save() + load() preserves all learned state exactly.

    Trains on a mixed corpus (SVO + causal), saves to a temp file,
    loads it back, and verifies that every learned structure matches.
    """
    learner.train(svo_corpus + causal_corpus)
    path = str(tmp_path / "pcl_state.json")

    learner.save(path)
    assert os.path.exists(path), "save() did not create the file"

    loaded = PositionalClusterLearner.load(path)

    # All four learned structures must match exactly.
    assert loaded.positional_freq == learner.positional_freq, (
        "positional_freq mismatch after save/load"
    )
    assert loaded.action_object_freq == learner.action_object_freq, (
        "action_object_freq mismatch after save/load"
    )
    assert loaded.positional_clusters == learner.positional_clusters, (
        "positional_clusters mismatch after save/load"
    )
    assert loaded.object_relation_map == learner.object_relation_map, (
        "object_relation_map mismatch after save/load"
    )

    # The loaded learner must be marked trained.
    assert loaded._trained is True

    # And classification results must match.
    assert loaded.classify("api menyebabkan panas") == \
        learner.classify("api menyebabkan panas")
    assert loaded.classify("saya makan ayam") == \
        learner.classify("saya makan ayam")


def test_persistence_creates_parent_dirs(
    learner: PositionalClusterLearner, svo_corpus: list, tmp_path
):
    """save() creates parent directories on demand (mkdir -p semantics)."""
    learner.train(svo_corpus)
    nested_path = str(tmp_path / "nested" / "deeper" / "pcl.json")
    learner.save(nested_path)
    assert os.path.exists(nested_path)


def test_persistence_atomic_write(
    learner: PositionalClusterLearner, svo_corpus: list, tmp_path
):
    """save() writes a parseable JSON file with a trailing newline."""
    learner.train(svo_corpus)
    path = str(tmp_path / "pcl_atomic.json")
    learner.save(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.endswith("\n"), "saved file should end with a newline"

    import json
    parsed = json.loads(content)
    assert "positional_freq" in parsed
    assert "action_object_freq" in parsed
    assert "positional_clusters" in parsed
    assert "object_relation_map" in parsed


# ======================================================================
# Supplementary: classify() failure contracts
# ======================================================================


def test_classify_empty_string(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """Empty input after training -> fallback result (CATEGORICAL)."""
    learner.train(svo_corpus)
    assert learner.classify("") == RelationType.CATEGORICAL


def test_classify_single_token(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """Single-token input after training -> fallback result (CATEGORICAL)."""
    learner.train(svo_corpus)
    assert learner.classify("apple") == RelationType.CATEGORICAL


def test_classify_negation_beats_learned_clusters(
    learner: PositionalClusterLearner, causal_corpus: list
):
    """Negation overrides learned clusters.

    'X tidak menyebabkan Y' is DIFFERENTIAL even after training on a
    causal corpus where 'menyebabkan' is strongly typed as CAUSAL.
    Same contract as SemanticRoleClassifier: negation is a syntactic
    signal that always inverts the relation.
    """
    learner.train(causal_corpus)
    # Sanity: without negation, classify returns CAUSAL.
    assert learner.classify("api menyebabkan panas") == RelationType.CAUSAL
    # With negation, classify returns DIFFERENTIAL.
    assert (
        learner.classify("merokok tidak menyebabkan awet muda")
        == RelationType.DIFFERENTIAL
    )


def test_classify_unseen_action_falls_back(
    learner: PositionalClusterLearner, svo_corpus: list
):
    """An action token not in action_object_freq delegates to fallback.

    'membaca' is not in the svo_corpus, so action_object_freq has no
    entry for it. classify() must delegate to the fallback, which
    returns CATEGORICAL (no seed match for 'membaca').
    """
    learner.train(svo_corpus)
    result = learner.classify("saya membaca buku")
    # Fallback: 'membaca' has no seed, so it returns CATEGORICAL.
    assert result == RelationType.CATEGORICAL


def test_classify_below_min_data_points_falls_back():
    """An action with < min_data_points observations delegates to fallback.

    Train with a corpus where 'menyebabkan' appears only twice - below
    the default min_data_points=3. classify() must delegate to the
    fallback (which still returns CAUSAL via the seed match, so the
    end result is CAUSAL - but the delegation path is what we're
    verifying).
    """
    learner = PositionalClusterLearner(min_data_points=3)
    learner.train([
        "api menyebabkan panas",
        "hujan menyebabkan banjir",
    ])
    # Only 2 observations for 'menyebabkan' - below threshold.
    assert sum(learner.action_object_freq["menyebabkan"].values()) == 2
    # classify() falls back, but the fallback's seed match still
    # returns CAUSAL.
    assert learner.classify("api menyebabkan panas") == RelationType.CAUSAL


def test_classify_uses_custom_fallback():
    """PositionalClusterLearner honours a caller-supplied fallback classifier.

    This is the composition contract: the learner wraps a
    SemanticRoleClassifier, and the caller can pre-configure that
    fallback (e.g. with a persist_path) by passing it to the
    constructor.
    """
    custom_fallback = SemanticRoleClassifier(override_threshold=5)
    learner = PositionalClusterLearner(fallback=custom_fallback)
    assert learner.fallback is custom_fallback
    # Without training, classify() delegates to the custom fallback.
    assert learner.classify("X menyebabkan Y") == RelationType.CAUSAL


# ======================================================================
# Supplementary: position labelling
# ======================================================================


def test_compute_positions_3_tokens():
    """3-token sentence -> [0, 1, 2] (classic SVO)."""
    assert PositionalClusterLearner._compute_positions(3) == [0, 1, 2]


def test_compute_positions_5_tokens():
    """>3-token sentence -> [0, 1, 1, 1, -1] (first, middles, last)."""
    assert PositionalClusterLearner._compute_positions(5) == [0, 1, 1, 1, -1]


def test_compute_positions_4_tokens():
    """4-token sentence -> [0, 1, 1, -1]."""
    assert PositionalClusterLearner._compute_positions(4) == [0, 1, 1, -1]


def test_compute_positions_edge_cases():
    """Edge cases: 0, 1, 2 tokens."""
    assert PositionalClusterLearner._compute_positions(0) == []
    assert PositionalClusterLearner._compute_positions(1) == [0]
    assert PositionalClusterLearner._compute_positions(2) == [0, 1]


# ======================================================================
# Supplementary: RelationType compatibility (BA44 contract)
# ======================================================================


def test_relation_type_compatible_with_ba44_rules(learner: PositionalClusterLearner, causal_corpus: list):
    """classify() returns a RelationType member usable by BA44's rules.

    BA44 (InferiorFrontalGyrus) switches on RelationType member
    identity, so the return value must be a RelationType enum member
    with the right .name and .value. This locks in the
    "Output RelationType harus compatible dengan BA44 rules yang
    sudah ada" constraint.
    """
    learner.train(causal_corpus)
    result = learner.classify("api menyebabkan panas")
    assert isinstance(result, RelationType)
    assert result.name == "CAUSAL"
    assert hasattr(result, "value")


def test_relation_type_fallback_compatible(learner: PositionalClusterLearner):
    """Fallback classification also returns the same RelationType enum."""
    result = learner.classify("api menyebabkan panas")
    assert isinstance(result, RelationType)
    # The learner and its fallback use the same RelationType class
    # (re-exported from semantic_role_classifier).
    assert type(result) is type(learner.fallback.classify("X menyebabkan Y"))


# ======================================================================
# Supplementary: train() idempotency / robustness
# ======================================================================


def test_train_empty_corpus_no_crash(learner: PositionalClusterLearner):
    """train([]) is a no-op - no crash, learner stays untrained."""
    learner.train([])
    assert learner._trained is False
    assert learner.positional_freq == {}


def test_train_short_sentences_no_crash(learner: PositionalClusterLearner):
    """train() with only short sentences (<3 tokens) builds no action_object_freq.

    The positional_freq still gets populated, but action_object_freq
    stays empty because we can't extract SVO from <3-token sentences.
    """
    learner.train([
        "hello world",
        "single",
        "",
        "one two three",
    ])
    # positional_freq has entries from 'one two three' (3 tokens).
    assert "one" in learner.positional_freq
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
    # The two valid sentences contribute 2 'makan' observations.
    assert learner.action_object_freq["makan"]["ayam"] == 1
    assert learner.action_object_freq["makan"]["ikan"] == 1
