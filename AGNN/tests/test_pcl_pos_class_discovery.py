"""
Tests for PositionalClusterLearner POS-class discovery: MODIFIER
detection, object super-cluster exposure, and unified tag_sentence()
POS tagging API.

These tests verify the three DoD items from the task brief:

    1. test_modifier_discovery_distinguishes_from_connector — 'sangat'
       (modifier) and 'dari' (connector) must be in DIFFERENT
       grammar classes.
    2. test_object_supercluster_groups_taxonomic_nouns — mamalia/
       reptil/logam must be in the SAME super-cluster and queryable
       via the new public API.
    3. test_tag_sentence_full_coverage — for corpus training
       sentences, tag_sentence() returns a pos_class for EVERY token,
       no exceptions/crashes.

Plus supplementary tests for edge cases, persistence, and the
zero-bias contract (no hardcoded word lists).

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_pcl_pos_class_discovery.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

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

# A small corpus designed to exercise all grammar classes:
#   - CAUSAL: "api menyebabkan kebakaran" (3-token SVO with verb)
#   - CATEGORICAL: "anjing adalah mamalia" (3-token SVO with copula)
#   - STATE+ADJ: "es sangat dingin" (3-token SVO with modifier at action slot)
#   - DIFFERENTIAL: "kucing berbeda dari reptil" (4-token with connector)
#   - FUNCTIONAL: "manusia membutuhkan air" (3-token SVO with verb)
#   - NEGATION: "tomat bukan sayuran" (3-token SVO with negation modifier)
#
# 'sangat' needs to appear at >=3 distinct fine positions AND >=5 total
# freq AND high positional entropy to be flagged as a
# function_word_candidate (which is required for MODIFIER detection).
# We give it positions 1, 2, and -1 by using it in 3-token, 4-token,
# and 5-token sentences.
_TINY_CORPUS = [
    # CAUSAL (3x)
    "api menyebabkan kebakaran",
    "rokok menyebabkan kanker",
    "panas menyebabkan ledakan",
    # CATEGORICAL (3x)
    "anjing adalah mamalia",
    "kucing adalah karnivora",
    "tomat adalah buah",
    # STATE + ADJ with 'sangat' — need 5+ occurrences at 3+ distinct
    # fine positions with high entropy for function_word_candidates.
    # Position 1 (3-token sentences — 'sangat' at fine index 1):
    "es sangat dingin",
    "bunga sangat harum",
    "kopi sangat pahit",
    # Position 2 (4-token sentences — 'sangat' at fine index 2):
    "es batu sangat dingin",
    "bunga melati sangat harum",
    # Position 3 (6-token sentences — 'sangat' at fine index 3):
    # This gives 'sangat' a 3rd distinct fine position.
    "air es batu sangat dingin ini",
    # DIFFERENTIAL with 'dari' (5x) — 'dari' as between-first connector
    "kucing berbeda dari reptil",
    "tomat berbeda dari sayuran",
    "kelelawar berbeda dari burung",
    "lumba lumba berbeda dari ikan",
    "ubin berbeda dari keramik",
    # FUNCTIONAL (3x)
    "manusia membutuhkan air",
    "tanaman membutuhkan cahaya",
    "mesin membutuhkan bensin",
    # NEGATION with 'bukan' (3x) — 'bukan' at action slot
    "tomat bukan sayuran",
    "paus bukan ikan",
    "kelelawar bukan burung",
]


def _train_tiny_learner() -> PositionalClusterLearner:
    """Build a PCL trained on _TINY_CORPUS.

    The tiny corpus is designed so that:
      - 'sangat' has 3+ pre-object 3-token observations → MODIFIER
      - 'dari' has 3+ between-first observations → CONNECTOR
      - 'bukan' has 3+ pre-object 3-token observations → MODIFIER
      - 'menyebabkan', 'adalah', 'berbeda', 'membutuhkan' are in
        action_object_freq → ACTION (not MODIFIER)
    """
    learner = PositionalClusterLearner()
    learner.train(_TINY_CORPUS)
    assert learner.is_trained, "Tiny corpus should produce a trained learner"
    return learner


def _label_particle_clusters_by_content(
    learner: PositionalClusterLearner,
) -> None:
    """Post-hoc-label the learner's particle clusters by content match.

    Mirrors how RelationType clusters are labelled in
    bootstrap_classifier.py (match by what's actually IN the cluster,
    not by index/ID — cluster ids are an implementation detail that
    shifts between train() calls). PR replacing the PRE-named
    MODIFIER/CONNECTOR detection (which baked the name into the
    detection function itself, calibrated against known tokens — the
    same bias pattern PR #69 was rejected for) with a proper two-stage
    discovery: train() clusters particle tokens by feature-vector
    distance UNNAMED, and a human (this helper, playing that role in
    tests) reviews + labels them afterward. tag_sentence() no longer
    auto-names particle clusters, so tests that need MODIFIER /
    CONNECTOR output must call this AFTER train() and BEFORE asserting.
    """
    clusters = learner.inspect_particle_clusters()
    mapping: Dict[int, str] = {}
    for cid, detail in clusters.items():
        tokens = set(detail["tokens"])
        if "sangat" in tokens or "bukan" in tokens:
            mapping[cid] = "MODIFIER"
        if "dari" in tokens or "dengan" in tokens:
            mapping[cid] = "CONNECTOR"
    learner.label_particle_clusters(mapping)


def _train_canonical_learner() -> PositionalClusterLearner:
    """Build a PCL trained on the canonical corpus files.

    Used by tests that need the full corpus vocabulary (e.g. object
    super-cluster grouping of taxonomy nouns).
    """
    corpus_paths = [
        _AGNP_ROOT / "data" / "pretrain_corpus.txt",
        _AGNP_ROOT / "data" / "pretrain_corpus_depth.txt",
        _AGNP_ROOT / "data" / "pretrain_corpus_passive.txt",
        _AGNP_ROOT / "data" / "pretrain_corpus_ditransitive.txt",
    ]
    missing = [p for p in corpus_paths if not p.exists()]
    if missing:
        pytest.skip(f"Canonical corpus files missing: {missing}")

    lines = []
    for p in corpus_paths:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("##"):
                lines.append(ln)

    learner = PositionalClusterLearner()
    learner.train(lines)
    return learner


# ======================================================================
# DoD #1: MODIFIER discovery distinguishes from CONNECTOR
# ======================================================================


def test_modifier_discovery_distinguishes_from_connector():
    """'sangat' (modifier) and 'dari' (connector) must be in DIFFERENT
    grammar classes.

    This is the primary DoD test for MODIFIER discovery. The tiny
    corpus is designed so that:
      - 'sangat' sits at the action slot in 3-token "state + adj"
        sentences ("es sangat dingin"). Its pre-object observations
        are dominated by 3-token sentences → MODIFIER.
      - 'dari' sits at the between-first slot in 4-token "action +
        connector + object" sentences ("kucing berbeda dari reptil").
        Its pre-object observations are dominated by >3-token
        sentences → CONNECTOR.

    The test asserts:
      1. 'sangat' is in modifier_tokens.
      2. 'dari' is in connector_tokens.
      3. 'sangat' is NOT in connector_tokens (priority rule: MODIFIER
         classification removes the token from connector_tokens).
      4. 'dari' is NOT in modifier_tokens.
      5. The two sets are disjoint (no token is both MODIFIER and
         CONNECTOR).
    """
    learner = _train_tiny_learner()

    # 1. 'sangat' is a MODIFIER.
    assert "sangat" in learner.modifier_tokens, (
        f"'sangat' must be in modifier_tokens (it sits at the action "
        f"slot in 3-token 'state + adj' sentences). "
        f"modifier_tokens={sorted(learner.modifier_tokens)}"
    )

    # 2. 'dari' is a CONNECTOR.
    assert "dari" in learner.connector_tokens, (
        f"'dari' must be in connector_tokens (it sits at the "
        f"between-first slot in 4-token 'action + connector + object' "
        f"sentences). connector_tokens={sorted(learner.connector_tokens)}"
    )

    # 3. 'sangat' is NOT in connector_tokens (priority rule).
    assert "sangat" not in learner.connector_tokens, (
        "'sangat' must NOT be in connector_tokens — the MODIFIER "
        "classification takes priority and removes it from "
        "connector_tokens. The priority rule in _compute_modifiers() "
        "may have regressed."
    )

    # 4. 'dari' is NOT in modifier_tokens.
    assert "dari" not in learner.modifier_tokens, (
        "'dari' must NOT be in modifier_tokens — it's a connector, "
        "not a modifier."
    )

    # 5. The two sets are disjoint.
    overlap = learner.modifier_tokens & learner.connector_tokens
    assert not overlap, (
        f"modifier_tokens and connector_tokens must be disjoint. "
        f"Overlap: {sorted(overlap)}"
    )


def test_modifier_discovery_on_canonical_corpus():
    """Canonical corpus also distinguishes MODIFIER from CONNECTOR.

    Integration test on the full pretrain corpus. 'sangat' should be
    a MODIFIER and 'dari' should be a CONNECTOR — same contract as
    the tiny corpus test, but on real data.
    """
    learner = _train_canonical_learner()

    assert "sangat" in learner.modifier_tokens, (
        f"'sangat' must be in modifier_tokens on the canonical corpus. "
        f"modifier_tokens={sorted(learner.modifier_tokens)}"
    )
    assert "dari" in learner.connector_tokens, (
        f"'dari' must be in connector_tokens on the canonical corpus."
    )
    assert "sangat" not in learner.connector_tokens, (
        "'sangat' must NOT be in connector_tokens (priority rule)."
    )


def test_modifier_not_classified_as_action():
    """A MODIFIER token must not be classified as an ACTION.

    'sangat' sits at the action slot (index 1) in "es sangat dingin",
    but it's NOT a real action — it's a modifier. The action
    extraction (Pass 2) must have excluded it from action_object_freq
    (because it's in function_word_candidates). This test verifies
    that exclusion.
    """
    learner = _train_tiny_learner()

    # 'sangat' is a MODIFIER and NOT in action_object_freq.
    assert "sangat" in learner.modifier_tokens
    assert "sangat" not in learner.action_object_freq, (
        "'sangat' must NOT be in action_object_freq — it was excluded "
        "from action extraction because it's a function_word_candidate "
        "(or modifier). If it's in action_object_freq, the MODIFIER "
        "discovery would have excluded it via the action_object_freq "
        "check."
    )

    # Contrast: 'menyebabkan' IS in action_object_freq (real action).
    assert "menyebabkan" in learner.action_object_freq
    assert "menyebabkan" not in learner.modifier_tokens


def test_connector_not_in_modifier():
    """A CONNECTOR token must not be classified as a MODIFIER.

    'dari' sits at the between-first slot in >3-token sentences. Its
    pre-object observations are dominated by >3-token sentences
    (pre_object_long_freq), so the 3tok rate is < 0.5 → not a
    MODIFIER.
    """
    learner = _train_tiny_learner()

    assert "dari" in learner.connector_tokens
    assert "dari" not in learner.modifier_tokens

    # Verify the 3tok rate is low for 'dari'.
    pre_3tok = learner.pre_object_3tok_freq.get("dari", 0)
    pre_long = learner.pre_object_long_freq.get("dari", 0)
    total = pre_3tok + pre_long
    if total > 0:
        rate = pre_3tok / total
        assert rate < 0.5, (
            f"'dari' 3tok rate must be < 0.5 (it's a connector, not a "
            f"modifier); got {rate:.2f} (3tok={pre_3tok}, long={pre_long})."
        )


# ======================================================================
# DoD #2: Object super-cluster groups taxonomic nouns
# ======================================================================


def test_object_supercluster_groups_taxonomic_nouns():
    """mamalia/reptil/logam must be in the SAME super-cluster.

    This is the primary DoD test for object sub-type exposure. The
    canonical corpus contains CATEGORICAL sentences like "anjing
    adalah mamalia", "ular adalah reptil", "besi adalah logam". The
    Brown clustering of object vocabulary should group these taxonomy
    nouns into the same super-cluster because they share the same
    action-context distribution (they all co-occur with the copulas
    'adalah' / 'merupakan').

    The test asserts:
      1. get_object_supercluster('mamalia') returns a non-None int.
      2. get_object_supercluster('reptil') returns the SAME int.
      3. get_object_supercluster('logam') returns the SAME int.
      4. inspect_object_superclusters() returns a dict containing
         these tokens in the same cluster.
    """
    learner = _train_canonical_learner()

    # 1. 'mamalia' has a super-cluster ID.
    sc_mamalia = learner.get_object_supercluster("mamalia")
    assert sc_mamalia is not None, (
        "'mamalia' must have a non-None object super-cluster ID on "
        "the canonical corpus. It appears as an object of 'adalah' "
        "and should be in the Brown-clustered object vocabulary."
    )

    # 2. 'reptil' has the SAME super-cluster ID.
    sc_reptil = learner.get_object_supercluster("reptil")
    assert sc_reptil is not None, (
        "'reptil' must have a non-None object super-cluster ID."
    )
    assert sc_reptil == sc_mamalia, (
        f"'mamalia' (sc={sc_mamalia}) and 'reptil' (sc={sc_reptil}) "
        f"must be in the SAME super-cluster — they share the same "
        f"action-context distribution (both co-occur with copulas)."
    )

    # 3. 'logam' has the SAME super-cluster ID.
    sc_logam = learner.get_object_supercluster("logam")
    assert sc_logam is not None, (
        "'logam' must have a non-None object super-cluster ID."
    )
    assert sc_logam == sc_mamalia, (
        f"'mamalia' (sc={sc_mamalia}) and 'logam' (sc={sc_logam}) "
        f"must be in the SAME super-cluster."
    )

    # 4. inspect_object_superclusters() returns a dict containing
    # these tokens in the same cluster.
    all_sc = learner.inspect_object_superclusters()
    assert sc_mamalia in all_sc, (
        f"Super-cluster {sc_mamalia} must be in inspect_object_superclusters() output."
    )
    cluster_tokens = all_sc[sc_mamalia]
    assert "mamalia" in cluster_tokens
    assert "reptil" in cluster_tokens
    assert "logam" in cluster_tokens


def test_get_object_supercluster_returns_none_for_unknown():
    """get_object_supercluster() returns None for tokens not in the
    object vocabulary.
    """
    learner = _train_canonical_learner()

    # 'upload' is an English loan-word not in the Indonesian corpus.
    assert learner.get_object_supercluster("upload") is None
    # Empty string.
    assert learner.get_object_supercluster("") is None
    # Non-string input (defensive).
    assert learner.get_object_supercluster(None) is None  # type: ignore[arg-type]
    assert learner.get_object_supercluster(42) is None  # type: ignore[arg-type]


def test_get_object_supercluster_normalizes_input():
    """get_object_supercluster() normalizes input (lower-case + strip)."""
    learner = _train_canonical_learner()

    sc_lower = learner.get_object_supercluster("mamalia")
    sc_upper = learner.get_object_supercluster("MAMALIA")
    sc_mixed = learner.get_object_supercluster("  Mamalia  ")

    assert sc_lower == sc_upper == sc_mixed
    assert sc_lower is not None


def test_inspect_object_superclusters_returns_sorted_lists():
    """inspect_object_superclusters() returns {sc_id: sorted_list}."""
    learner = _train_canonical_learner()
    all_sc = learner.inspect_object_superclusters()

    assert isinstance(all_sc, dict)
    for sc_id, tokens in all_sc.items():
        assert isinstance(sc_id, int)
        assert isinstance(tokens, list)
        assert tokens == sorted(tokens), (
            f"Tokens in super-cluster {sc_id} must be sorted; "
            f"got {tokens[:5]}..."
        )


def test_object_supercluster_api_on_untrained_learner():
    """Object super-cluster API returns None/empty on untrained learner."""
    learner = PositionalClusterLearner()
    assert not learner.is_trained

    assert learner.get_object_supercluster("mamalia") is None
    assert learner.inspect_object_superclusters() == {}


# ======================================================================
# DoD #3: tag_sentence() full coverage — no crashes
# ======================================================================


def test_tag_sentence_full_coverage():
    """tag_sentence() returns a pos_class for EVERY token in corpus
    sentences — no exceptions, no crashes.

    This is the primary DoD test for the unified POS tagging API.
    We train on the canonical corpus, then run tag_sentence() on
    every sentence in the corpus. The test asserts:
      1. tag_sentence() never raises for any corpus sentence.
      2. The returned list has the same length as the tokenized
         sentence (every token gets a tag).
      3. Every tag is one of the 6 valid pos_class values.
    """
    learner = _train_canonical_learner()

    corpus_paths = [
        _AGNP_ROOT / "data" / "pretrain_corpus.txt",
        _AGNP_ROOT / "data" / "pretrain_corpus_depth.txt",
    ]
    valid_pos_classes = {
        "AGENT", "ACTION", "OBJECT", "MODIFIER", "CONNECTOR", "UNKNOWN",
    }

    total_sentences = 0
    total_tokens = 0
    tag_distribution: dict = {}
    errors: list = []

    for path in corpus_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            total_sentences += 1
            try:
                tags = learner.tag_sentence(line)
            except Exception as e:
                errors.append(f"tag_sentence({line!r}) raised {type(e).__name__}: {e}")
                continue

            # Every token gets a tag.
            expected_len = len(learner._tokenize(line))
            assert len(tags) == expected_len, (
                f"tag_sentence({line!r}) returned {len(tags)} tags for "
                f"{expected_len} tokens. Every token must get a tag."
            )

            for tok, pos in tags:
                total_tokens += 1
                assert pos in valid_pos_classes, (
                    f"Invalid pos_class {pos!r} for token {tok!r} in "
                    f"sentence {line!r}. Must be one of {valid_pos_classes}."
                )
                tag_distribution[pos] = tag_distribution.get(pos, 0) + 1

    assert not errors, (
        f"tag_sentence() raised on {len(errors)} sentences:\n"
        + "\n".join(errors[:5])
    )
    assert total_sentences > 100, (
        f"Expected to test >100 corpus sentences; got {total_sentences}."
    )
    assert total_tokens > 500, (
        f"Expected to test >500 tokens; got {total_tokens}."
    )

    # Print the tag distribution for visibility (shows up in pytest -v).
    dist_str = ", ".join(
        f"{k}={v}" for k, v in sorted(tag_distribution.items())
    )
    assert True, (
        f"tag_sentence() full coverage: {total_sentences} sentences, "
        f"{total_tokens} tokens, distribution: {dist_str}"
    )


def test_tag_sentence_causal_pattern():
    """tag_sentence() correctly tags a 3-token CAUSAL sentence.

    "api menyebabkan kebakaran" → AGENT/ACTION/OBJECT.
    """
    learner = _train_tiny_learner()
    tags = learner.tag_sentence("api menyebabkan kebakaran")
    assert tags == [("api", "AGENT"), ("menyebabkan", "ACTION"), ("kebakaran", "OBJECT")], (
        f"Expected AGENT/ACTION/OBJECT; got {tags}."
    )


def test_tag_sentence_state_with_modifier():
    """tag_sentence() correctly tags a 3-token STATE+ADJ sentence.

    "es sangat dingin" → AGENT/MODIFIER/OBJECT.
    'sangat' is at the action slot but is a MODIFIER (not a real
    action), so it gets tagged MODIFIER, not ACTION.
    """
    learner = _train_tiny_learner()
    _label_particle_clusters_by_content(learner)
    tags = learner.tag_sentence("es sangat dingin")
    assert tags == [("es", "AGENT"), ("sangat", "MODIFIER"), ("dingin", "OBJECT")], (
        f"Expected AGENT/MODIFIER/OBJECT; got {tags}. "
        f"'sangat' must be MODIFIER (it's in modifier_tokens and not "
        f"a real action)."
    )


def test_tag_sentence_differential_with_connector():
    """tag_sentence() correctly tags a 4-token DIFFERENTIAL sentence.

    "kucing berbeda dari reptil" → AGENT/ACTION/CONNECTOR/OBJECT.
    'dari' is a CONNECTOR sitting between the action and object.
    """
    learner = _train_tiny_learner()
    _label_particle_clusters_by_content(learner)
    tags = learner.tag_sentence("kucing berbeda dari reptil")
    assert tags == [
        ("kucing", "AGENT"),
        ("berbeda", "ACTION"),
        ("dari", "CONNECTOR"),
        ("reptil", "OBJECT"),
    ], (
        f"Expected AGENT/ACTION/CONNECTOR/OBJECT; got {tags}."
    )


def test_tag_sentence_empty_input():
    """tag_sentence() returns [] for empty input."""
    learner = _train_tiny_learner()
    assert learner.tag_sentence("") == []
    assert learner.tag_sentence("   ") == []
    assert learner.tag_sentence("!!! ???") == []


def test_tag_sentence_short_input():
    """tag_sentence() handles <3-token sentences gracefully.

    For <3-token sentences, positional SVO is ambiguous. The method
    falls back to per-token grammar-class lookup (MODIFIER /
    CONNECTOR / UNKNOWN).
    """
    learner = _train_tiny_learner()

    # 1-token sentence.
    tags = learner.tag_sentence("sangat")
    assert len(tags) == 1
    assert tags[0][0] == "sangat"
    # 'sangat' is a MODIFIER even in a 1-token sentence.
    assert tags[0][1] in ("MODIFIER", "UNKNOWN")

    # 2-token sentence.
    tags = learner.tag_sentence("es dingin")
    assert len(tags) == 2


def test_tag_sentence_untrained_learner():
    """tag_sentence() on an untrained learner returns UNKNOWN for all."""
    learner = PositionalClusterLearner()
    assert not learner.is_trained

    tags = learner.tag_sentence("api menyebabkan kebakaran")
    assert len(tags) == 3
    for tok, pos in tags:
        assert pos == "UNKNOWN", (
            f"Untrained learner must return UNKNOWN for all tokens; "
            f"got {pos!r} for {tok!r}."
        )


# ======================================================================
# Persistence — save/load preserves MODIFIER and object super-cluster state
# ======================================================================


def test_persistence_preserves_modifier_and_object_superclusters():
    """save/load preserves modifier_tokens, pre_object_*_freq, and
    object_supercluster state.
    """
    learner = _train_tiny_learner()

    # Capture pre-save state.
    pre_modifiers = sorted(learner.modifier_tokens)
    pre_connectors = sorted(learner.connector_tokens)
    pre_pre3tok = dict(learner.pre_object_3tok_freq)
    pre_pre_long = dict(learner.pre_object_long_freq)
    pre_obj_sc_id = dict(learner.object_supercluster_id)

    # Save + load.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        tmp_path = f.name
    try:
        learner.save(tmp_path)
        loaded = PositionalClusterLearner.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Verify post-load state matches.
    assert sorted(loaded.modifier_tokens) == pre_modifiers, (
        f"modifier_tokens not preserved by save/load. "
        f"Pre: {pre_modifiers}, Post: {sorted(loaded.modifier_tokens)}."
    )
    assert sorted(loaded.connector_tokens) == pre_connectors
    assert loaded.pre_object_3tok_freq == pre_pre3tok
    assert loaded.pre_object_long_freq == pre_pre_long
    assert loaded.object_supercluster_id == pre_obj_sc_id


def test_load_backward_compat_with_old_state_file():
    """load() handles state files that lack the new MODIFIER fields.

    Older state files (pre-this-PR) don't have pre_object_3tok_freq,
    pre_object_long_freq, or modifier_tokens. load() must backfill
    them as empty without crashing.
    """
    # Create a minimal state file that lacks the new fields.
    import json

    minimal_state = {
        "similarity_threshold": 0.13,
        "min_action_observations": 2,
        "positional_freq": {"menyebabkan": {"1": 3}},
        "action_object_freq": {"menyebabkan": {"kebakaran": 1}},
        "cluster_id_of": {"menyebabkan": 0},
        "action_clusters": {"0": ["menyebabkan"]},
        "cluster_labels": {"0": "CAUSAL"},
        "action_connector_signature": {},
        "connector_tokens": [],
        "function_word_candidates": [],
        "action_bucket_anchors": [],
        "object_supercluster_id": {},
        "object_superclusters": {},
        # NOTE: no pre_object_3tok_freq, pre_object_long_freq, or
        # modifier_tokens — simulates an old state file.
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(minimal_state, f)
        tmp_path = f.name
    try:
        loaded = PositionalClusterLearner.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    # New fields are backfilled as empty.
    assert loaded.pre_object_3tok_freq == {}
    assert loaded.pre_object_long_freq == {}
    assert loaded.modifier_tokens == set()
    # tag_sentence() works (returns UNKNOWN for modifiers since the
    # sets are empty).
    tags = loaded.tag_sentence("es sangat dingin")
    assert len(tags) == 3


# ======================================================================
# Zero-bias contract — no hardcoded word lists
# ======================================================================


def test_zero_bias_no_hardcoded_modifier_list():
    """MODIFIER discovery uses only positional + frequency statistics.

    Verify that the PCL module does NOT contain any hardcoded list of
    adverb / modifier words (like 'sangat', 'begitu', 'terlalu',
    'cukup'). The discovery must be purely statistical.
    """
    # Read the PCL source and check that none of the common Indonesian
    # adverbs appear as string literals in the module (they may appear
    # in comments/docstrings, but not as values in sets/lists/dicts).
    pcl_source = (_AGNP_ROOT / "neocortex" / "positional_cluster_learner.py").read_text()

    # These are the Indonesian adverbs that a hardcoded approach would
    # use. They must NOT appear as string literals in the module code
    # (outside of comments/docstrings/test fixtures).
    #
    # We check for them in the context of set/list/dict literals or
    # string assignments. A simple substring check would have false
    # positives (e.g. in docstrings), so we check for patterns like
    # "sangat" appearing in a quoted context that's NOT a comment.
    forbidden_adverbs = ["sangat", "begitu", "terlalu", "cukup", "memang"]

    for adv in forbidden_adverbs:
        # Check if the adverb appears in a code context (not just in
        # comments/docstrings). We look for it in lines that don't
        # start with # and aren't inside triple-quoted strings.
        # This is a heuristic — a perfect check would require AST
        # parsing. For our purposes, the heuristic is sufficient: if
        # the adverb appears in a set/list/dict literal, it would be
        # on a line like `{"sangat", "begitu"}` or `modifiers =
        # {"sangat"}`.
        for line in pcl_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            # Check for the adverb in a quoted string context.
            if f'"{adv}"' in line or f"'{adv}'" in line:
                # Allow appearances in docstrings (multi-line strings
                # that span the line). This is a heuristic — we
                # check that the line doesn't look like a set/dict
                # literal assignment.
                if (
                    "=" in line
                    and ("{" in line or "[" in line)
                    and not line.strip().startswith('"""')
                    and not line.strip().startswith("'''")
                ):
                    pytest.fail(
                        f"Hardcoded adverb {adv!r} found in PCL source "
                        f"at line: {line.strip()!r}. The MODIFIER "
                        f"discovery must be purely statistical (zero-bias)."
                    )


def test_zero_bias_no_hardcoded_connector_list():
    """CONNECTOR discovery also uses only positional + frequency
    statistics (existing contract, verified here for completeness).
    """
    pcl_source = (_AGNP_ROOT / "neocortex" / "positional_cluster_learner.py").read_text()

    forbidden_connectors = ["dari", "dengan", "sebagai", "ke", "di"]

    for conn in forbidden_connectors:
        for line in pcl_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if f'"{conn}"' in line or f"'{conn}'" in line:
                if (
                    "=" in line
                    and ("{" in line or "[" in line)
                    and not line.strip().startswith('"""')
                    and not line.strip().startswith("'''")
                ):
                    pytest.fail(
                        f"Hardcoded connector {conn!r} found in PCL "
                        f"source at line: {line.strip()!r}."
                    )


# ----------------------------------------------------------------------
# PARTICLE clustering: zero-bias two-stage discovery (replaces the
# PRE-named MODIFIER/CONNECTOR detection critiqued during review — see
# the "PARTICLE clustering" section in positional_cluster_learner.py)
# ----------------------------------------------------------------------

def test_particle_clusters_are_unnamed_before_labelling():
    """train() produces particle clusters with empty labels.

    This is the core contract: clustering happens with ZERO awareness
    of "MODIFIER"/"CONNECTOR" or any other name. particle_cluster_labels
    must be empty immediately after train(), before any human calls
    label_particle_clusters().
    """
    learner = _train_tiny_learner()
    assert learner.particle_clusters, (
        "Expected at least one particle cluster to form from the tiny "
        "corpus's function words (sangat, dari, bukan, ...)."
    )
    assert learner.particle_cluster_labels == {}, (
        "particle_cluster_labels must be empty right after train() — "
        "naming is a separate, post-hoc, opt-in step via "
        "label_particle_clusters(), never automatic."
    )


def test_tag_sentence_reports_unknown_before_labelling():
    """tag_sentence() must NOT silently use the old pre-named sets.

    Before label_particle_clusters() is called, a particle token
    (e.g. 'sangat') must tag as UNKNOWN — not MODIFIER. Falling back
    to the legacy modifier_tokens/connector_tokens membership would
    defeat the entire point of the two-stage discovery: it would mean
    every caller gets the OLD pre-named answer for free, with no
    actual review step ever required.
    """
    learner = _train_tiny_learner()
    tags = dict(learner.tag_sentence("es sangat dingin"))
    assert tags["sangat"] == "UNKNOWN", (
        f"Expected 'sangat' to be UNKNOWN before label_particle_clusters() "
        f"is called (no auto-naming); got {tags['sangat']!r}."
    )


def test_particle_clustering_excludes_real_content_actions():
    """Real action verbs must never appear in a particle cluster.

    Regression guard for a bug found during implementation: the
    connector-candidate pool's "never appears as object" check is
    weak (true of nearly every verb, since verbs aren't nouns), so
    verbs that incidentally sit in a between-first slot a few times
    used to leak into the particle-clustering candidate pool. Any
    token that has its own action_object_freq entry (i.e. is itself
    the predicate of at least one sentence) must be excluded from
    particle clustering — it's already a content action, not a
    leftover grammar particle.
    """
    learner = _train_canonical_learner()
    real_actions = set(learner.action_object_freq.keys())
    for cluster_id, tokens in learner.particle_clusters.items():
        leaked = tokens & real_actions
        assert not leaked, (
            f"Particle cluster {cluster_id} contains real content "
            f"action(s) {leaked!r} — these have their own "
            f"action_object_freq entry and must not compete for a "
            f"leftover-particle cluster slot."
        )


def test_particle_cluster_can_discover_a_category_beyond_modifier_connector():
    """The architecture must not presuppose only 2 particle categories.

    On the canonical corpus, negators ('tidak', 'melainkan') form a
    cluster behaviourally distinct from both the dari/dengan-style
    CONNECTOR cluster and the sangat/begitu-style MODIFIER cluster.
    This test asserts that AGNN is free to surface this as its own
    named category (e.g. "NEGATOR") rather than being forced to
    relabel it as one of the two pre-existing ideas — proving the
    clustering doesn't hardcode "there are exactly 2 particle
    classes".
    """
    learner = _train_canonical_learner()
    clusters = learner.inspect_particle_clusters()

    tidak_cluster_id = None
    for cid, detail in clusters.items():
        if "tidak" in set(detail["tokens"]):
            tidak_cluster_id = cid
            break
    assert tidak_cluster_id is not None, (
        "'tidak' should land in some particle cluster on the canonical "
        "corpus (it has high positional entropy and is excluded from "
        "the action slot)."
    )

    # Find the dari/dengan-style connector cluster (if 'tidak' isn't
    # already in it).
    connector_cluster_id = None
    for cid, detail in clusters.items():
        toks = set(detail["tokens"])
        if {"dari", "dengan"} & toks:
            connector_cluster_id = cid
            break

    if connector_cluster_id is not None:
        assert tidak_cluster_id != connector_cluster_id, (
            "'tidak' (negator) and 'dari'/'dengan' (connectors) should "
            "be behaviourally distinct enough to land in different "
            "unnamed clusters on the canonical corpus — they are "
            "different grammar phenomena (negation vs. preposition)."
        )

    # The naming step is free to call this cluster anything —
    # including a name that didn't exist before this test was written.
    learner.label_particle_clusters({tidak_cluster_id: "NEGATOR"})
    tags = dict(learner.tag_sentence("kucing tidak termasuk reptil"))
    assert tags.get("tidak") == "NEGATOR", (
        f"Expected 'tidak' to report the freshly-assigned 'NEGATOR' "
        f"label; got {tags.get('tidak')!r}."
    )


def test_particle_cluster_purity_no_chain_merge_regression():
    """Regression guard: the negator cluster must stay small and clean.

    Found during BOS review of the Q/K/V soft-clustering PR: cosine
    similarity on the 4-dim particle signature is degenerate (two of
    the four dimensions are 0.0 for nearly every candidate, so cosine
    measures direction over ~2 effective dims and is blind to
    magnitude). This let an unrelated noise token ('di', a locative
    preposition) chain-merge into the cluster seeded by 'tidak' on the
    very first comparison (cosine('tidak', 'di') = 0.909, well above
    the 0.85 threshold then in use), and the cluster ballooned to 29
    tokens including clearly unrelated nouns ('asupan', 'benda',
    'galah', 'instrumen', ...).

    test_particle_cluster_can_discover_a_category_beyond_modifier_connector
    (above) did NOT catch this — it only checks that 'tidak' lands in
    a cluster distinct from 'dari'/'dengan', never the cluster's SIZE
    or whether obviously-unrelated tokens leaked in. This test closes
    that gap: on the canonical corpus, the cluster containing 'tidak'
    must stay small (real negators: tidak, tak, melainkan — at most a
    handful of tokens), not balloon into a grab-bag.
    """
    learner = _train_canonical_learner()
    clusters = learner.inspect_particle_clusters()

    tidak_cluster_tokens = None
    for cid, detail in clusters.items():
        if "tidak" in set(detail["tokens"]):
            tidak_cluster_tokens = detail["tokens"]
            break
    assert tidak_cluster_tokens is not None, (
        "'tidak' should land in some particle cluster on the canonical corpus."
    )
    assert len(tidak_cluster_tokens) <= 6, (
        f"The cluster containing 'tidak' has {len(tidak_cluster_tokens)} "
        f"tokens: {tidak_cluster_tokens!r}. Real Indonesian negators "
        f"(tidak/tak/melainkan/bukan-as-negator) are a small closed "
        f"class — a cluster this large indicates a chain-merge "
        f"regression (unrelated nouns/verbs absorbed via a degenerate "
        f"similarity metric), not genuine negator discovery."
    )
    # Obviously unrelated tokens (locative prepositions, random nouns
    # from the corpus) must NOT be in the negator cluster.
    unrelated_examples = {"di", "asupan", "benda", "galah", "instrumen"}
    leaked = unrelated_examples & set(tidak_cluster_tokens)
    assert not leaked, (
        f"Unrelated token(s) {leaked!r} leaked into the 'tidak' "
        f"cluster {tidak_cluster_tokens!r} — chain-merge regression."
    )


# ----------------------------------------------------------------------
# Passive voice (sprint round 1 — BOS-driven structural diversity test)
# ----------------------------------------------------------------------

def test_spo_parses_passive_voice_with_oleh_agent_marker():
    """spo() must correctly extract subject/predicate/object from
    Indonesian passive voice ("X di-V oleh Y").

    Found during BOS training sprint: "oleh" (the passive-voice agent
    marker) is, on its own statistical profile, indistinguishable from
    a copula like 'adalah' — it concentrates 100% at the action bucket
    because it NEVER appears sentence-initially (unlike 'sebelum'/
    'setelah', which the existing _is_soft_particle check already
    handles). Left unfixed, "oleh" gets independently recognised as an
    action_bucket_anchor and the extraction loop in
    _extract_action_object picks it (or stops at it) instead of the
    real verb, leaving spo().object empty.

    Root cause was a DATA issue, not purely an algorithm bug: the
    initial passive-voice corpus used 25 different di-verbs at 1
    occurrence each (all below the action-anchor frequency floor),
    while "oleh" itself accumulated frequency 25 — making "oleh" look
    MORE like a real anchor than any individual verb. Fixed by (a)
    giving the corpus verb depth (5+ occurrences per di-verb, so each
    individually clears the frequency floor) and (b) a structural
    exclusion in _extract_action_object: a bucket-1 candidate that is
    NOT verb-morphology AND immediately follows another already-
    recognised action token is treated as a post-verbal particle, not
    an independent verb — a positional/structural signal (adjacency),
    not a hardcoded word.
    """
    learner = _train_canonical_learner()

    cases = [
        ("ikan dimakan oleh kucing", "ikan", "dimakan", "kucing"),
        ("buku dibaca oleh siswa", "buku", "dibaca", "siswa"),
        ("mobil diperbaiki oleh montir", "mobil", "diperbaiki", "montir"),
    ]
    for sentence, expected_subj, expected_pred, expected_obj in cases:
        spo = learner.spo(sentence)
        assert spo.subject == expected_subj, (
            f"{sentence!r}: expected subject={expected_subj!r}, "
            f"got {spo.subject!r}"
        )
        assert spo.predicate == expected_pred, (
            f"{sentence!r}: expected predicate={expected_pred!r}, "
            f"got {spo.predicate!r}"
        )
        assert spo.object == expected_obj, (
            f"{sentence!r}: expected object={expected_obj!r} (this was "
            f"the original bug — object came out empty), got {spo.object!r}"
        )


def test_oleh_is_not_misclassified_as_action():
    """'oleh' must never be tagged ACTION — it has no verb morphology
    and exists purely as the passive-voice agent marker.
    """
    learner = _train_canonical_learner()
    assert not learner._is_action_token("oleh"), (
        "'oleh' was tagged as a valid ACTION candidate — this is the "
        "passive-voice misclassification bug (oleh looks like a copula "
        "by bucket concentration alone, since it never appears "
        "sentence-initially)."
    )


# ----------------------------------------------------------------------
# Ditransitive verbs (sprint round 3 — verb + 2 objects)
# ----------------------------------------------------------------------

def test_spo_parses_ditransitive_verbs():
    """spo() must correctly extract subject/predicate/object-phrase
    from ditransitive verbs ("X verb Y Z" — indirect + direct object).

    Found during the BOS training sprint: this pattern already works
    correctly with NO architecture changes needed — the cluster-driven
    parser naturally absorbs multiple post-verbal tokens into the
    object phrase (no particle sits between them to confuse it, unlike
    passive voice's "oleh"). Recorded as a permanent regression guard
    now that pretrain_corpus_ditransitive.txt is part of the default
    training corpus.
    """
    learner = _train_canonical_learner()

    cases = [
        ("ayah membelikan adik sepeda", "ayah", "membelikan"),
        ("guru mengajari murid matematika", "guru", "mengajari"),
        ("ibu mengirim nenek surat", "ibu", "mengirim"),
        ("pemandu menunjukkan turis arah", "pemandu", "menunjukkan"),
    ]
    for sentence, expected_subj, expected_pred in cases:
        spo = learner.spo(sentence)
        assert spo.subject == expected_subj, (
            f"{sentence!r}: expected subject={expected_subj!r}, "
            f"got {spo.subject!r}"
        )
        assert spo.predicate == expected_pred, (
            f"{sentence!r}: expected predicate={expected_pred!r}, "
            f"got {spo.predicate!r}"
        )
        assert spo.object, (
            f"{sentence!r}: expected a non-empty object phrase "
            f"(both post-verbal arguments), got empty string"
        )
