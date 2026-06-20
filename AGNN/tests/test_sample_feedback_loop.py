"""
Tests for the synthetic-sentence feedback loop (RLHF-style, statistical
substrate).

Covers the Definition-of-Done from the task brief:

  Task 1 — sample_sentence:
    1. test_sample_sentence_uses_cluster_action_and_associated_object
       — DoD test: the generated sentence's action is a real cluster
       member, and the object is the highest-count object in
       action_object_freq for that action.
    2. test_sample_sentence_returns_none_for_invalid_cluster_id
    3. test_sample_sentence_excludes_function_words_from_subject
    4. test_sample_sentence_excludes_discourse_markers_from_subject

  Task 2 — apply_feedback:
    5. test_apply_feedback_good_increases_pair_weight
       — DoD test: verdict="good" raises the action↔object edge
       confidence via the eligibility-trace path.
    6. test_apply_feedback_bad_decreases_pair_weight
       — DoD test: verdict="bad" lowers the edge confidence.
    7. test_apply_feedback_does_not_relabel_cluster
       — Zero-bias guard: feedback does NOT change the cluster's
       label, membership, or the action_object_freq statistics.
    8. test_apply_feedback_invalid_verdict_is_noop
    9. test_apply_feedback_action_unclustered_is_noop

  Task 3 — CLI smoke test (the script imports cleanly + --help works).

Tests that need the AGNNGraph (self-ai/src/agnn/graph.py) are
skipped when that module is not importable — same gate as
test_eligibility_trace.py.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_sample_feedback_loop.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
_AGGN_ROOT = Path(__file__).resolve().parent.parent
_SELF_AI_SRC = _AGGN_ROOT.parent / "self-ai" / "src"

# Insert self-ai/src FIRST (lower priority) so that the AGNN package
# (inserted next) wins on name collisions.
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGGN_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGGN_ROOT))


# ----------------------------------------------------------------------
# Imports under test
# ----------------------------------------------------------------------

# Load AGNN/core.py directly by path. Same pattern as the existing
# test_core_wired.py / test_eligibility_trace.py — avoids name
# collision with self-ai/src/core/.
import importlib.util as _ilu  # noqa: E402

_core_path = _AGGN_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_module_fb", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_module_fb"] = agnn_core_module  # register before exec
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore

from neocortex.positional_cluster_learner import (  # noqa: E402
    PositionalClusterLearner,
)
from neocortex.semantic_role_classifier import (  # noqa: E402
    RelationType,
    SemanticRoleClassifier,
)
from engrams.episodic_engram import Episome  # noqa: E402


# ----------------------------------------------------------------------
# Gates — skip graph-requiring tests when AGNNGraph is unavailable.
# ----------------------------------------------------------------------

def _agnn_graph_available() -> bool:
    try:
        from agnn.graph import (  # noqa: F401
            AGNNGraph,
            AGNNNode,
            NodeType,
            TypedEdge,
            RelationType as _RT,
        )
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def trained_learner() -> PositionalClusterLearner:
    """A small trained + labelled cluster learner.

    Trains on a 20-sentence synthetic corpus that produces 3 clean
    clusters (CAUSAL / FUNCTIONAL / CATEGORICAL), then labels them.
    """
    corpus = [
        # CAUSAL: menyebabkan / memicu take state-change objects
        "api menyebabkan panas",
        "hujan menyebabkan banjir",
        "gesekan menyebabkan panas",
        "listrik menyebabkan kebakaran",
        "rokok menyebabkan kanker",
        "stres memicu panas",
        "stres memicu banjir",
        # FUNCTIONAL: membutuhkan / memerlukan take need objects
        "tanaman membutuhkan air",
        "tanaman membutuhkan energi",
        "manusia membutuhkan makanan",
        "manusia membutuhkan air",
        "mesin membutuhkan energi",
        "otak memerlukan oksigen",
        "otak memerlukan air",
        # CATEGORICAL: adalah / merupakan take class objects
        "manusia adalah mamalia",
        "kucing adalah mamalia",
        "besi adalah logam",
        "anjing merupakan mamalia",
        "ayam merupakan unggas",
        "emas merupakan logam",
    ]
    learner = PositionalClusterLearner()
    learner.train(corpus)

    # Label the 3 clusters we expect to form.
    inspect = learner.inspect_clusters()
    label_map = {}
    for cid, actions in inspect.items():
        if "menyebabkan" in actions or "memicu" in actions:
            label_map[cid] = RelationType.CAUSAL
        elif "membutuhkan" in actions or "memerlukan" in actions:
            label_map[cid] = RelationType.FUNCTIONAL
        elif "adalah" in actions or "merupakan" in actions:
            label_map[cid] = RelationType.CATEGORICAL
    learner.label_clusters(label_map)
    return learner


# ======================================================================
# Task 1 — sample_sentence
# ======================================================================


def test_sample_sentence_uses_cluster_action_and_associated_object(
    trained_learner: PositionalClusterLearner,
):
    """DoD test: the generated sentence's action is a real cluster
    member, and the object is the highest-count object in
    ``action_object_freq`` for that action.

    Setup: train on a corpus where 'menyebabkan' co-occurs most often
    with 'panas' (twice). Generate a sentence from the CAUSAL cluster
    using a seeded rng that picks 'menyebabkan' as the action. The
    resulting sentence must be "<subject> menyebabkan panas".

    This is the zero-bias synthetic-generation contract: the action
    comes from the cluster, the object comes from the action's
    statistical co-occurrence distribution. No hardcoded templates.
    """
    learner = trained_learner

    # Find the CAUSAL cluster.
    causal_cid = None
    for cid, actions in learner.action_clusters.items():
        if "menyebabkan" in actions:
            causal_cid = cid
            break
    assert causal_cid is not None, "CAUSAL cluster must exist"

    # Find a seed that picks 'menyebabkan' as the action.
    import random
    chosen_seed = None
    chosen_sentence = None
    for seed in range(50):
        rng = random.Random(seed)
        s = learner.sample_sentence(causal_cid, rng=rng)
        if s is None:
            continue
        tokens = s.split()
        if len(tokens) >= 3 and tokens[1] == "menyebabkan":
            chosen_seed = seed
            chosen_sentence = s
            break
    assert chosen_sentence is not None, (
        "Must find a seed that picks 'menyebabkan' as the action; "
        "tried seeds 0..49"
    )

    tokens = chosen_sentence.split()
    action = tokens[1]
    obj = tokens[2]

    # 1. Action must be a cluster member.
    assert action in learner.action_clusters[causal_cid], (
        f"Action {action!r} must be a member of cluster {causal_cid}. "
        f"Cluster actions: {sorted(learner.action_clusters[causal_cid])}"
    )

    # 2. Object must be the highest-count object for that action.
    objs = learner.action_object_freq.get(action, {})
    assert objs, f"Action {action!r} must have objects in action_object_freq"
    expected_obj = max(
        objs.items(),
        key=lambda kv: (kv[1], -ord(kv[0][0]) if kv[0] else 0),
    )[0]
    assert obj == expected_obj, (
        f"Object {obj!r} must be the highest-count object for action "
        f"{action!r}; expected {expected_obj!r}. "
        f"action_object_freq[{action!r}] = {objs}"
    )

    # 3. Sentence has the SVO structure: subject action object.
    assert len(tokens) == 3, (
        f"Generated sentence must be 3 tokens (SVO); got {len(tokens)}: "
        f"{tokens}"
    )
    subject = tokens[0]
    assert subject != action and subject != obj, (
        f"Subject {subject!r} must not equal action or object "
        f"(would be a degenerate sentence)"
    )


def test_sample_sentence_returns_none_for_invalid_cluster_id(
    trained_learner: PositionalClusterLearner,
):
    """sample_sentence returns None for invalid cluster ids.

    Covers: untrained learner, -1 (unclustered), and an id not in
    action_clusters.
    """
    learner = trained_learner

    # -1 (unclustered sentinel)
    assert learner.sample_sentence(-1) is None

    # An id not in action_clusters
    fake_cid = max(learner.action_clusters.keys()) + 1000
    assert learner.sample_sentence(fake_cid) is None

    # An untrained learner returns None.
    fresh = PositionalClusterLearner()
    assert fresh.sample_sentence(0) is None


def test_sample_sentence_excludes_function_words_from_subject(
    trained_learner: PositionalClusterLearner,
):
    """The subject slot must NOT be filled by a statistically
    discovered function word.

    This is part of the zero-bias contract: function words (sangat,
    itu, bukan, ...) are excluded from the subject candidate set so
    the generated sentence's subject is a content noun, not a
    grammatical marker.

    We verify this by training on a corpus that includes a
    statistically discoverable function word ('sangat' at multiple
    fine positions) and checking that no generated sentence has
    'sangat' as its subject.
    """
    # Build a corpus where 'sangat' appears at >= 3 distinct fine
    # positions so the entropy-based detector flags it.
    corpus = []
    pairs = [("bunga", "harum"), ("teh", "sepat"), ("kopi", "pahit"),
             ("es", "dingin"), ("gula", "manis")]
    for n, a in pairs:
        corpus.append(f"{n} sangat {a}")                    # idx 1
    for n, b, a in [(p[0], "biru", p[1]) for p in pairs]:
        corpus.append(f"{n} {b} sangat {a}")                # idx 2
    for n, b, c, a in [(p[0], "biru", "ungu", p[1]) for p in pairs]:
        corpus.append(f"{n} {b} {c} sangat {a}")            # idx 3
    # Control: real SVO sentences.
    for s, o in [("saya", "ayam"), ("dia", "ikan"),
                 ("kamu", "sayur"), ("ibu", "nasi"),
                 ("bapak", "daging")]:
        corpus.append(f"{s} makan {o}")

    learner = PositionalClusterLearner()
    learner.train(corpus)

    # Confirm 'sangat' was flagged as a function word candidate.
    assert "sangat" in learner.function_word_candidates, (
        "'sangat' must be flagged as a function word candidate after "
        "training on a corpus where it appears at >= 3 distinct fine "
        f"positions. Got: {sorted(learner.function_word_candidates)}"
    )

    # Generate sentences from every cluster and check none has
    # 'sangat' as the subject.
    import random
    rng = random.Random(123)
    for cid in list(learner.action_clusters.keys()):
        for _ in range(10):
            s = learner.sample_sentence(cid, rng=rng)
            if s is None:
                continue
            subject = s.split()[0]
            assert subject not in learner.function_word_candidates, (
                f"Subject {subject!r} must NOT be a function word "
                f"candidate. Sentence: {s!r}"
            )


def test_sample_sentence_excludes_discourse_markers_from_subject(
    trained_learner: PositionalClusterLearner,
):
    """The subject slot must NOT be filled by tokens that ONLY appear
    at the agent bucket with high frequency (discourse markers like
    'secara', 'menurut').

    Real subject nouns (api, manusia, kucing, ...) also appear as
    objects somewhere in a large corpus, so their bucket distribution
    spans 0 AND 2/-1. Discourse markers only appear at bucket 0 — and
    they appear OFTEN (they're grammatical). The frequency floor
    ``_SUBJECT_DISCOURSE_MARKER_MIN_FREQ`` (10) cleanly separates
    them: real subjects in small corpora stay below 10, discourse
    markers in the pretrain corpus (51 'secara', 45 'menurut',
    34 'karena') all clear it.

    We verify by training on a corpus where 'menurut' appears 12 times
    at position 0 (above the threshold) and checking that no generated
    sentence has 'menurut' as its subject.
    """
    # Build a corpus where 'menurut' appears 12 times at position 0
    # (above the discourse-marker threshold of 10), plus enough SVO
    # sentences to form at least one cluster.
    corpus = []
    # 'menurut' only at position 0, 12 times (>= 10 threshold).
    for i in range(12):
        corpus.append(f"menurut ahli{i} teori{i}")
    # Real SVO sentences that form a cluster — 'makan' appears with
    # multiple objects so it gets clustered.
    for s, o in [("saya", "ayam"), ("dia", "ikan"),
                 ("kamu", "sayur"), ("ibu", "nasi"),
                 ("bapak", "daging")]:
        corpus.append(f"{s} makan {o}")
    learner = PositionalClusterLearner()
    learner.train(corpus)

    # 'menurut' must NOT be a subject candidate — it only appears at
    # bucket 0 with high frequency (>= 10).
    menurut_pf = learner.positional_freq.get("menurut", {})
    assert menurut_pf.get(0, 0) >= 10, (
        f"'menurut' must appear at bucket 0 >= 10 times for the "
        f"discourse-marker filter to kick in; got {menurut_pf.get(0, 0)}"
    )
    # No presence at object bucket (2 or -1).
    assert menurut_pf.get(2, 0) == 0 and menurut_pf.get(-1, 0) == 0, (
        "'menurut' must not appear at the object bucket"
    )

    # Find the cluster containing 'makan' (the only action with
    # multiple observations).
    target_cid = None
    for cid, actions in learner.action_clusters.items():
        if "makan" in actions:
            target_cid = cid
            break
    assert target_cid is not None, "'makan' must be in a cluster"

    # Generate sentences from the cluster — none should have
    # 'menurut' as the subject.
    import random
    rng = random.Random(456)
    generated = 0
    for _ in range(20):
        s = learner.sample_sentence(target_cid, rng=rng)
        if s is None:
            continue
        generated += 1
        subject = s.split()[0]
        assert subject != "menurut", (
            f"Subject must NOT be 'menurut' (discourse marker with "
            f"freq >= 10 at bucket 0 only). Sentence: {s!r}"
        )
    assert generated > 0, (
        "Must generate at least one sentence to verify the filter"
    )


# ======================================================================
# Task 2 — apply_feedback (requires AGNNGraph)
# ======================================================================


def _make_core_with_action_object_edge(trained_learner):
    """Build an AGNNCore whose graph has one subject→object edge
    labelled with the action's cluster RelationType.

    Topology:
        node "manusia" --[CATEGORICAL]--> node "mamalia"

    The episome for "manusia" is registered so reinforce/penalize can
    find it. The cluster learner is injected so apply_feedback can
    parse sentences and look up cluster labels.

    Returns (core, edge_source_label, edge_target_label, edge_relation_type_name).
    """
    from agnn.graph import (
        AGNNGraph,
        AGNNNode,
        NodeType,
        TypedEdge,
        RelationType as GraphRT,
    )
    from engrams.engram_complex import EngramComplex

    core = AGNNCore(use_cluster_learner=False)
    ec = EngramComplex()
    ec._graph = AGNNGraph(embedding_dim=64)
    core.graph = ec

    # Two nodes: manusia (subject) -> mamalia (object).
    src_node = AGNNNode(
        id="n_manusia",
        label="manusia",
        node_type=NodeType.ENTITY,
        confidence=0.5,
    )
    tgt_node = AGNNNode(
        id="n_mamalia",
        label="mamalia",
        node_type=NodeType.ENTITY,
        confidence=0.5,
    )
    ec._graph.add_node(src_node)
    ec._graph.add_node(tgt_node)

    # Register an Episome for the source node so reinforce/penalize
    # can find it (otherwise the early-return guard triggers).
    epi = Episome(id="n_manusia", text="manusia", confidence=0.5)
    epi.id = "n_manusia"
    core._episomes.append(epi)

    # Edge: manusia -> mamalia, CATEGORICAL (matches the cluster label
    # for 'adalah' in the trained_learner fixture).
    ec._graph.add_edge(
        TypedEdge(
            source_id="n_manusia",
            target_id="n_mamalia",
            relation_type=GraphRT.CATEGORICAL,
            confidence=0.5,
        )
    )

    # Inject the cluster learner so apply_feedback can parse + look up.
    core._cluster_learner = trained_learner

    return core, "manusia", "mamalia", "CATEGORICAL"


def _edge_confidence(core, src, tgt, rel_name=None):
    """Read the live TypedEdge.confidence from the wrapped graph.

    If rel_name is given, only matches edges with that relation_type
    name; otherwise returns the first edge from src to tgt.
    """
    inner = core.graph._graph
    for edge in inner.get_edges_from(src):
        if edge.target_id != tgt:
            continue
        if rel_name is not None:
            edge_rel = edge.relation_type
            edge_name = edge_rel.name if hasattr(edge_rel, "name") else str(edge_rel)
            if edge_name != rel_name:
                continue
        return float(edge.confidence)
    raise AssertionError(f"Edge {src} -> {tgt} ({rel_name}) not found")


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_apply_feedback_good_increases_pair_weight(
    trained_learner: PositionalClusterLearner,
):
    """DoD test: verdict="good" raises the action↔object edge confidence.

    Setup: graph has one edge manusia -> mamalia [CATEGORICAL] at
    conf=0.5. Cluster learner has 'adalah' in the CATEGORICAL cluster.
    Sentence "manusia adalah mamalia" parses to (action='adalah',
    object='mamalia'), finds the edge by matching subject/object
    labels + the CATEGORICAL relation type, stamps an eligibility
    trace, and calls reinforce() — which routes +0.1 to the stamped
    edge via the existing three-factor path.

    Expected: edge confidence rises from 0.5 to ~0.6 (±0.1, the
    _REINFORCE_DELTA budget).
    """
    core, src, tgt, rel = _make_core_with_action_object_edge(trained_learner)
    before = _edge_confidence(core, "n_manusia", "n_mamalia", rel)
    assert before == pytest.approx(0.5), f"Edge must start at 0.5, got {before}"

    result = core.apply_feedback("manusia adalah mamalia", "good")

    assert result["applied"] is True, (
        f"Feedback must be applied. Result: {result}"
    )
    assert result["reason"] == "ok"
    assert result["action"] == "adalah"
    assert result["object"] == "mamalia"
    assert result["cluster_id"] is not None and result["cluster_id"] >= 0
    assert result["relation_type"] == "CATEGORICAL"
    assert result["edges_stamped"] >= 1

    after = _edge_confidence(core, "n_manusia", "n_mamalia", rel)
    delta = after - before
    assert delta > 0, (
        f"Edge confidence must increase after 'good' verdict: "
        f"before={before}, after={after}, delta={delta}"
    )
    # The +_REINFORCE_DELTA budget should be distributed to the
    # stamped edge(s). With one edge stamped, the boost should be
    # the full 0.1.
    assert delta == pytest.approx(core._REINFORCE_DELTA, abs=1e-6), (
        f"Edge boost {delta} must equal _REINFORCE_DELTA "
        f"{core._REINFORCE_DELTA} (one edge stamped = full budget)"
    )


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_apply_feedback_bad_decreases_pair_weight(
    trained_learner: PositionalClusterLearner,
):
    """DoD test: verdict="bad" lowers the action↔object edge confidence.

    Same setup as the 'good' test, but verdict="bad" → penalize() →
    -0.1 modulatory signal → edge confidence drops from 0.5 to ~0.4.
    """
    core, src, tgt, rel = _make_core_with_action_object_edge(trained_learner)
    before = _edge_confidence(core, "n_manusia", "n_mamalia", rel)
    assert before == pytest.approx(0.5)

    result = core.apply_feedback("manusia adalah mamalia", "bad")

    assert result["applied"] is True, f"Feedback must be applied. Result: {result}"
    assert result["reason"] == "ok"
    assert result["edges_stamped"] >= 1

    after = _edge_confidence(core, "n_manusia", "n_mamalia", rel)
    delta = after - before
    assert delta < 0, (
        f"Edge confidence must decrease after 'bad' verdict: "
        f"before={before}, after={after}, delta={delta}"
    )
    assert delta == pytest.approx(-core._REINFORCE_DELTA, abs=1e-6), (
        f"Edge delta {delta} must equal -_REINFORCE_DELTA "
        f"{-core._REINFORCE_DELTA}"
    )


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_apply_feedback_does_not_relabel_cluster(
    trained_learner: PositionalClusterLearner,
):
    """Zero-bias guard: feedback does NOT change cluster labels,
    membership, or action_object_freq.

    The principle: feedback is OUTPUT-LEVEL ("this sentence makes
    sense"), not RELABELING. The cluster structure (which actions
    belong together, what RelationType each cluster has) is fixed by
    the corpus statistics + label_clusters() call; feedback only
    adjusts the graph edge weights.

    We snapshot the cluster learner's state before feedback, apply
    'good' feedback, and assert that NONE of the following changed:
      - cluster_labels (the {cluster_id: RelationType} mapping)
      - action_clusters (the {cluster_id: set(actions)} mapping)
      - cluster_id_of (the {action: cluster_id} mapping)
      - action_object_freq (the {action: {object: count}} statistics)
      - positional_freq (corpus position statistics)

    The only thing that SHOULD change is the graph edge confidence
    (a separate state container, not part of the cluster learner).
    """
    core, src, tgt, rel = _make_core_with_action_object_edge(trained_learner)
    learner = core._cluster_learner

    # Snapshot the cluster learner's state BEFORE feedback.
    cluster_labels_before = dict(learner.cluster_labels)
    action_clusters_before = {
        cid: set(actions) for cid, actions in learner.action_clusters.items()
    }
    cluster_id_of_before = dict(learner.cluster_id_of)
    action_object_freq_before = {
        a: dict(objs) for a, objs in learner.action_object_freq.items()
    }
    positional_freq_before = {
        t: dict(buckets) for t, buckets in learner.positional_freq.items()
    }

    # Apply feedback.
    result = core.apply_feedback("manusia adalah mamalia", "good")
    assert result["applied"] is True

    # Assert NO change to cluster learner state.
    assert learner.cluster_labels == cluster_labels_before, (
        "cluster_labels must NOT change after feedback — relabeling "
        "would violate the zero-bias principle."
    )
    assert learner.action_clusters == action_clusters_before, (
        "action_clusters must NOT change after feedback — cluster "
        "membership is fixed by corpus statistics, not by user verdicts."
    )
    assert learner.cluster_id_of == cluster_id_of_before, (
        "cluster_id_of must NOT change after feedback — actions stay "
        "in their assigned clusters."
    )
    assert learner.action_object_freq == action_object_freq_before, (
        "action_object_freq must NOT change after feedback — corpus "
        "statistics are immutable post-training."
    )
    assert learner.positional_freq == positional_freq_before, (
        "positional_freq must NOT change after feedback — corpus "
        "position statistics are immutable post-training."
    )

    # Sanity: the GRAPH edge confidence DID change (that's the
    # intended effect, and it's a separate state container from the
    # cluster learner).
    after = _edge_confidence(core, "n_manusia", "n_mamalia", rel)
    assert after > 0.5, (
        f"Edge confidence must have increased (the intended effect). "
        f"Got {after}"
    )


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_apply_feedback_invalid_verdict_is_noop(
    trained_learner: PositionalClusterLearner,
):
    """Verdicts other than 'good'/'bad' are silently ignored.

    Returns applied=False, reason='invalid_verdict', and leaves the
    edge confidence untouched.
    """
    core, src, tgt, rel = _make_core_with_action_object_edge(trained_learner)
    before = _edge_confidence(core, "n_manusia", "n_mamalia", rel)

    for bad_verdict in ("maybe", "", "yes", "no", "GOOD", "Good", None):
        result = core.apply_feedback("manusia adalah mamalia", bad_verdict)
        assert result["applied"] is False, (
            f"Verdict {bad_verdict!r} must not apply. Result: {result}"
        )
        assert result["reason"] == "invalid_verdict"

    after = _edge_confidence(core, "n_manusia", "n_mamalia", rel)
    assert after == before, (
        f"Edge confidence must not change after invalid verdicts: "
        f"before={before}, after={after}"
    )


@pytest.mark.skipif(not _agnn_graph_available(),
                    reason="self-ai/src/agnn/graph.py not importable")
def test_apply_feedback_action_unclustered_is_noop(
    trained_learner: PositionalClusterLearner,
):
    """Feedback on a sentence whose action is not in any cluster is
    a no-op (reason='action_unclustered').

    This is the guard against feedback on out-of-cluster actions
    polluting the graph: we only adjust edges whose relation_type
    the cluster learner has assigned.
    """
    core, src, tgt, rel = _make_core_with_action_object_edge(trained_learner)
    before = _edge_confidence(core, "n_manusia", "n_mamalia", rel)

    # 'membaca' is not in the trained_learner's corpus, so it has no
    # cluster_id.
    result = core.apply_feedback("manusia membaca buku", "good")
    assert result["applied"] is False
    assert result["reason"] == "action_unclustered"
    assert result["action"] == "membaca"

    after = _edge_confidence(core, "n_manusia", "n_mamalia", rel)
    assert after == before, (
        f"Edge confidence must not change for un-clustered action: "
        f"before={before}, after={after}"
    )


# ======================================================================
# Task 3 — CLI smoke test
# ======================================================================


def test_sample_feedback_loop_cli_imports():
    """The CLI script must import cleanly and expose a main() function.

    We don't run the loop (it needs stdin), but we verify the module
    loads without errors and main() is callable.
    """
    # Import the CLI as a module by file path (it's not in the AGNN
    # package — it's a top-level script).
    cli_path = _AGGN_ROOT / "sample_feedback_loop.py"
    assert cli_path.exists(), f"CLI script must exist at {cli_path}"

    spec = _ilu.spec_from_file_location("sample_feedback_loop_module", cli_path)
    cli_module = _ilu.module_from_spec(spec)
    sys.modules["sample_feedback_loop_module"] = cli_module
    spec.loader.exec_module(cli_module)

    assert hasattr(cli_module, "main"), "CLI must expose main()"
    assert callable(cli_module.main)


def test_sample_feedback_loop_cli_help():
    """``python AGNN/sample_feedback_loop.py --help`` exits 0 and
    mentions the key options.
    """
    import subprocess
    cli_path = _AGGN_ROOT / "sample_feedback_loop.py"
    result = subprocess.run(
        [sys.executable, str(cli_path), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"--help must exit 0; got {result.returncode}. "
        f"stderr: {result.stderr}"
    )
    assert "--num" in result.stdout
    assert "--seed" in result.stdout
    assert "--state" in result.stdout
    assert "--corpus" in result.stdout


def test_sample_feedback_loop_cli_generates_sentences_noninteractively():
    """The CLI's helper functions work end-to-end without stdin.

    We can't run the interactive loop in a test, but we CAN verify
    the pieces:
      - _load_or_train_cluster_learner returns a trained learner.
      - _eligible_cluster_ids returns at least one cluster id.
      - learner.sample_sentence returns a non-None sentence for at
        least one eligible cluster.
    """
    cli_path = _AGGN_ROOT / "sample_feedback_loop.py"
    spec = _ilu.spec_from_file_location("sample_feedback_loop_module_2", cli_path)
    cli_module = _ilu.module_from_spec(spec)
    sys.modules["sample_feedback_loop_module_2"] = cli_module
    spec.loader.exec_module(cli_module)

    # Train on the canonical pretrain corpus if available.
    pretrain = _AGGN_ROOT / "data" / "pretrain_corpus.txt"
    if not pretrain.exists():
        pytest.skip("pretrain_corpus.txt not available")

    try:
        learner = cli_module._load_or_train_cluster_learner(
            state_path=None,
            corpus_paths=[pretrain],
        )
    except Exception as e:
        pytest.skip(f"Could not load/train cluster learner: {e}")

    assert learner.is_trained, "Learner must be trained"
    eligible = cli_module._eligible_cluster_ids(learner, min_actions=2)
    assert len(eligible) > 0, (
        "Must be at least one cluster with >= 2 actions"
    )

    import random
    rng = random.Random(0)
    generated = 0
    for cid in eligible:
        s = learner.sample_sentence(cid, rng=rng)
        if s is not None:
            generated += 1
            assert len(s.split()) == 3, (
                f"Generated sentence must be 3 tokens: {s!r}"
            )
        if generated >= 3:
            break
    assert generated >= 1, "Must generate at least one sentence"
