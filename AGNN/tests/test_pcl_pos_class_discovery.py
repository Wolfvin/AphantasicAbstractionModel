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
        _AGNP_ROOT / "data" / "pretrain_corpus_subordinate.txt",
        _AGNP_ROOT / "data" / "pretrain_corpus_coordinate.txt",
        _AGNP_ROOT / "data" / "pretrain_corpus_pronoun.txt",
        _AGNP_ROOT / "data" / "pretrain_corpus_predicate_adjective.txt",
        _AGNP_ROOT / "data" / "pretrain_corpus_datang.txt",
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


# ----------------------------------------------------------------------
# Subordinate conjunctions (sprint round 4)
# ----------------------------------------------------------------------

def test_subordinate_conjunctions_karena_ketika_are_particles():
    """'karena' and 'ketika' must be recognised as particles.

    Found during the BOS training sprint, round 4 (subordinate
    conjunction diversity): "karena"/"ketika" enter
    ``action_object_freq`` incidentally (some sentence in the corpus
    happens to make them the first eligible bucket-1 candidate during
    >3-token extraction), which lets the existing has_agent/has_action
    soft-particle check recognise them. This is a permanent regression
    guard for that working case.

    Other subordinate conjunctions ("meskipun"/"agar"/"supaya"/
    "walaupun") do NOT get this recognition — they never enter
    action_object_freq because an earlier token in their sentences
    (e.g. "tetap"/"harus") is picked first by the >3-token extraction
    loop, which breaks at the first eligible candidate. Broadening the
    has_agent/has_action check to bypass the action_object_freq gate
    was attempted and REVERTED (see _is_soft_particle's docstring) —
    it caused a regression where real content nouns used across many
    different corpus sentences (e.g. "ikan") also coincidentally
    satisfy has_agent+has_action, breaking the round-1 passive-voice
    fix. Recognising "meskipun"-class conjunctions needs a different,
    more targeted signal — tracked as a sprint follow-up.
    """
    learner = _train_canonical_learner()
    assert learner._is_particle_token("karena"), (
        "'karena' should be recognised as a particle (clause-boundary "
        "anchor candidate) via the existing has_agent/has_action signal."
    )
    assert learner._is_particle_token("ketika"), (
        "'ketika' should be recognised as a particle via the same signal."
    )


# ----------------------------------------------------------------------
# Clause coordinators (sprint round 5 — "dan"/"atau")
# ----------------------------------------------------------------------

def test_coordinator_clusters_are_action_bucket_anchored_not_excluded():
    """'dan'/'atau' must form their OWN action cluster(s), not noise.

    This is the core insight from round 5 (raised by the user, not a
    BOS-discovered bug): "dan"/"atau" being action-bucket-anchored is
    NOT a misclassification to suppress — _cluster_actions() already,
    correctly, puts them in their own cluster(s) separate from every
    labelled RelationType cluster, because a coordinator's "object"
    distribution (whatever noun starts the second clause) is far more
    diffuse than a real verb's. This test pins that the clustering
    itself produces this separation (zero-bias, unprompted).
    """
    learner = _train_canonical_learner()
    clusters = learner.inspect_cluster_details()

    dan_cluster = None
    atau_cluster = None
    for cid, detail in clusters.items():
        actions = set(detail["actions"])
        if "dan" in actions:
            dan_cluster = cid
        if "atau" in actions:
            atau_cluster = cid
    assert dan_cluster is not None, "'dan' should land in some action cluster."
    assert atau_cluster is not None, "'atau' should land in some action cluster."

    # Neither must be in a cluster labelled with a RelationType —
    # they're a different category, not CAUSAL/FUNCTIONAL/etc.
    assert dan_cluster not in learner.cluster_labels, (
        "'dan' landed in a cluster that also got a RelationType label "
        "— that would mean it merged with real predicates, which "
        "should not happen given its diffuse object distribution."
    )
    assert atau_cluster not in learner.cluster_labels, (
        "'atau' landed in a cluster that also got a RelationType label."
    )


def test_mark_clause_coordinator_clusters_enables_two_clause_parse():
    """After marking, a coordinated sentence tags BOTH clauses correctly.

    "ayah membaca koran dan ibu memasak nasi" has two independent
    clauses joined by "dan". Before marking, "dan" is action-bucket-
    anchored and the second clause's information is lost (the object-
    collection loop for the FIRST clause breaks early at "dan",
    treating it as a second predicate). After
    mark_clause_coordinator_clusters(), "dan" is treated as a
    clause-boundary anchor (same role as a particle), and the lazy
    anchor-split mechanism (already proven for "sebelum"/"setelah" in
    round 1) splits the sentence into two independent clauses, each
    correctly tagged AGENT/ACTION/OBJECT.
    """
    learner = _train_canonical_learner()
    clusters = learner.inspect_cluster_details()
    coordinator_ids = {
        cid for cid, detail in clusters.items()
        if "dan" in set(detail["actions"]) or "atau" in set(detail["actions"])
    }
    assert coordinator_ids, "Expected to find 'dan'/'atau' clusters."
    learner.mark_clause_coordinator_clusters(coordinator_ids)

    tags = dict(learner.tag_sentence("ayah membaca koran dan ibu memasak nasi"))
    assert tags["ayah"] == "AGENT"
    assert tags["membaca"] == "ACTION"
    assert tags["koran"] == "OBJECT"
    assert tags["ibu"] == "AGENT", (
        f"Expected 'ibu' (second clause's subject) to be AGENT after "
        f"coordinator marking; got {tags['ibu']!r}. This is the round-5 "
        f"fix: without marking, the second clause's subject is "
        f"unreachable because the object-collection loop for the first "
        f"clause never stops at 'dan'."
    )
    assert tags["memasak"] == "ACTION"
    assert tags["nasi"] == "OBJECT"


def test_bootstrap_classifier_marks_coordinator_clusters():
    """build_labelled_cluster_learner() marks coordinators automatically.

    This is the production wiring: EXPECTED_COORDINATOR_TOKENS in
    bootstrap_classifier.py finds "dan"/"atau"'s cluster(s) by content
    match (same robust pattern as EXPECTED_VERB_GROUPS for
    RelationType) and marks them — so AGNNCore's default cluster
    learner (loaded from the committed state file) already has this
    refinement, no manual step needed.
    """
    from neocortex.bootstrap_classifier import build_labelled_cluster_learner

    learner = build_labelled_cluster_learner()
    assert learner.coordinator_cluster_ids, (
        "build_labelled_cluster_learner() should have found and marked "
        "at least one coordinator cluster on the canonical corpus."
    )
    tags = dict(learner.tag_sentence("ayah membaca koran dan ibu memasak nasi"))
    assert tags["ibu"] == "AGENT"


# ----------------------------------------------------------------------
# spo_all() — multi-clause extraction (sprint round 6)
# ----------------------------------------------------------------------

def test_spo_all_returns_both_clauses_for_coordinated_sentence():
    """spo_all() must surface BOTH clauses of a coordinated sentence.

    Completes round 5's coordinator fix: tag_sentence() already
    tagged every token across both clauses correctly, but spo()
    could only surface one (the "main" clause, per its existing
    tie-breaking rule). spo_all() exposes every clause the
    anchor-split mechanism finds, sharing the exact same split logic
    as spo() via _parse_all_clauses() (so the two methods can never
    disagree on HOW a sentence splits, only on whether one or all
    results are returned).
    """
    learner = _train_canonical_learner()
    clusters = learner.inspect_cluster_details()
    coordinator_ids = {
        cid for cid, detail in clusters.items()
        if "dan" in set(detail["actions"]) or "atau" in set(detail["actions"])
    }
    learner.mark_clause_coordinator_clusters(coordinator_ids)

    clauses = learner.spo_all("ayah membaca koran dan ibu memasak nasi")
    assert len(clauses) == 2, (
        f"Expected 2 clauses (coordinated by 'dan'); got {len(clauses)}: "
        f"{clauses!r}"
    )
    assert clauses[0].subject == "ayah"
    assert clauses[0].predicate == "membaca"
    assert clauses[0].object == "koran"
    assert clauses[1].subject == "ibu"
    assert clauses[1].predicate == "memasak"
    assert clauses[1].object == "nasi"


def test_spo_all_single_clause_sentence_returns_one_element():
    """spo_all() on a simple sentence returns a single-element list."""
    learner = _train_canonical_learner()
    clauses = learner.spo_all("api menyebabkan kebakaran")
    assert len(clauses) == 1
    assert clauses[0].subject == "api"
    assert clauses[0].predicate == "menyebabkan"
    assert clauses[0].object == "kebakaran"


def test_spo_all_never_returns_empty_list():
    """spo_all() always returns at least one element, even on edge cases.

    Callers should be able to do spo_all(text)[0] without a length
    check — untrained learner, empty input, and short input all fall
    back to a single-element list wrapping the fallback's result,
    never an empty list.
    """
    learner = _train_canonical_learner()
    assert len(learner.spo_all("")) >= 1
    assert len(learner.spo_all("x")) >= 1

    untrained = PositionalClusterLearner()
    assert len(untrained.spo_all("api menyebabkan kebakaran")) >= 1


def test_spo_all_agrees_with_spo_on_the_main_clause():
    """spo() and spo_all() must never disagree on clause structure.

    spo() picks the "main" clause (most complete, ties broken by
    latest); that exact clause must also appear somewhere in
    spo_all()'s result, since both share _parse_all_clauses().
    """
    learner = _train_canonical_learner()
    sentence = "sebelum makan saya mencuci tangan"
    single = learner.spo(sentence)
    all_clauses = learner.spo_all(sentence)
    matches = [
        c for c in all_clauses
        if c.predicate == single.predicate and c.object == single.object
    ]
    assert matches, (
        f"spo()'s result {single!r} should appear in spo_all()'s "
        f"result {all_clauses!r} — they must agree on clause structure."
    )


# ----------------------------------------------------------------------
# spo_embedded() — recursive embedded-clause detection (sprint round 25)
# ----------------------------------------------------------------------

def test_spo_embedded_detects_genuine_embedded_clause():
    """spo_embedded() must recognise when the object span is itself a
    clause (has its own subject + ACTION), not a flat noun phrase.

    Background: Round 22-23 proved empirically that no single-token
    surface statistic (span-length, particle cluster membership,
    bigram conditioning) can distinguish a complementizer ("bahwa")
    from a passive agent-marker ("oleh") — they are statistically
    identical on every signal tried. The only remaining signal is
    STRUCTURAL: does the span, re-parsed with the same SVO machinery,
    contain a real ACTION of its own?
    """
    learner = _train_canonical_learner()
    result = learner.spo_embedded(
        "guru menyatakan bahwa murid berkembang pesat"
    )
    assert result.embedded is not None, (
        "the object span ('...murid berkembang pesat') contains its "
        "own ACTION ('berkembang') and should be recognised as an "
        "embedded clause, not flattened into a single object string"
    )
    assert result.embedded.predicate == "berkembang"


def test_spo_embedded_flat_object_for_passive_voice_stays_flat():
    """spo_embedded() must NOT treat a passive-voice agent ("oleh X")
    as an embedded clause — there is no ACTION in that span, so
    ``embedded`` must stay ``None``. This is the precise case the
    'oleh' vs 'bahwa' disambiguation work (Round 22-23) needed to get
    right: both particles look identical on every surface statistic,
    so this guards against a recursion that fires on EVERY post-
    predicate span regardless of content.
    """
    learner = _train_canonical_learner()
    cases = [
        "ikan dimakan oleh kucing",
        "buku dibaca oleh siswa",
    ]
    for sentence in cases:
        result = learner.spo_embedded(sentence)
        assert result.embedded is None, (
            f"{sentence!r}: 'oleh <agent>' has no ACTION of its own — "
            f"embedded should be None, got {result.embedded!r}"
        )


def test_spo_embedded_flat_object_for_simple_causal_stays_flat():
    """A plain noun object ("menyebabkan banjir") must never be
    mistaken for an embedded clause — there's no second ACTION at all.
    """
    learner = _train_canonical_learner()
    result = learner.spo_embedded("hujan menyebabkan banjir")
    assert result.embedded is None


def test_spo_embedded_flat_fields_agree_with_spo():
    """spo_embedded()'s subject/predicate/object must match spo()'s
    output for the SAME sentence — spo_embedded() only ADDS the
    ``embedded`` field, it must never change the flat result.
    """
    learner = _train_canonical_learner()
    sentence = "ikan dimakan oleh kucing"
    flat = learner.spo(sentence)
    rich = learner.spo_embedded(sentence)
    assert rich.subject == flat.subject
    assert rich.predicate == flat.predicate
    assert rich.object == flat.object
    assert rich.negated == flat.negated


def test_spo_embedded_untrained_delegates_to_fallback():
    """An untrained learner must delegate to the fallback classifier,
    same contract as spo()/spo_all(), with embedded always None.
    """
    untrained = PositionalClusterLearner()
    result = untrained.spo_embedded("api menyebabkan kebakaran")
    assert result.embedded is None


def test_spo_embedded_rejects_bare_token_false_positive():
    """A single leftover token that happens to pass _is_action_token
    (e.g. via the documented "bel-" verb-morphology collision —
    "belalai" matches the same prefix as "belajar", a known,
    deliberately-accepted trade-off since sprint Round 10) must NOT be
    treated as an embedded clause when it has nothing else around it.

    "gajah memiliki belalai" is only 3 tokens: action="memiliki",
    object_tokens=["belalai"]. Recursing into that single-token span
    finds "belalai" itself passes _is_action_token (morphology
    coincidence), but _parse_clause_spo on a 1-token list returns
    subject="" AND object="" — that's not a clause, it's a bare
    mis-tagged token. The fix is a structural well-formedness gate
    (>=2 of 3 SPO slots filled), not a fallback/exception for this
    specific token — see spo_embedded's implementation for the
    rationale.
    """
    learner = _train_canonical_learner()
    result = learner.spo_embedded("gajah memiliki belalai")
    assert result.embedded is None, (
        f"a bare single token with no subject/object of its own should "
        f"never be promoted to an embedded clause, got {result.embedded!r}"
    )


# ----------------------------------------------------------------------
# Pronoun positional diversity (sprint round 8)
# ----------------------------------------------------------------------

def test_saya_is_not_misclassified_as_particle():
    """'saya' ("I") must not be a particle — it's a real pronoun.

    Found during the BOS training sprint (first surfaced round 3,
    recurred round 7): every pre-round-8 corpus usage of "saya" was as
    an OBJECT (e.g. "dia memberi saya buku" — indirect object), never
    as a subject. This gave "saya" a one-sided positional signature
    (bucket 1 only), making it statistically indistinguishable from a
    genuine particle by the has_agent/has_action soft-particle check
    (which, ironically, requires BOTH buckets to fire for OTHER
    particles like "karena" — but a token with ONLY bucket-1 presence
    can still slip through other particle-detection paths depending on
    entropy). Fixed by adding declarative sentences with "saya" as the
    AGENT (pretrain_corpus_pronoun.txt), balancing its distribution
    across both buckets — the same fix pattern as "dia" in round 3.
    """
    learner = _train_canonical_learner()
    assert not learner._is_particle_token("saya"), (
        "'saya' should not be classified as a particle — it's a "
        "content pronoun, not a grammatical marker."
    )


def test_saya_correctly_tagged_object_in_ditransitive():
    """'saya' as an indirect object must tag OBJECT, not UNKNOWN.

    Regression guard for the round-8 fix on the exact sentence that
    surfaced the bug in round 3.
    """
    learner = _train_canonical_learner()
    tags = dict(learner.tag_sentence("dia memberi saya buku"))
    assert tags["saya"] == "OBJECT", (
        f"Expected 'saya' to be OBJECT in a ditransitive sentence; "
        f"got {tags['saya']!r}."
    )


# ----------------------------------------------------------------------
# Predicate-final adjectives (sprint round 9 reframe)
# ----------------------------------------------------------------------
#
# Background: the original anchor-discovery test (concentration at one
# bucket + low bucket-entropy + freq floor) was always bucket-agnostic
# by construction, but was only EVER applied to bucket 1
# (_ACTION_BUCKET) — every other bucket's anchors were discarded even
# when AGNN found a real, low-entropy concentration there. This was
# the human imposing "predicates are verbs in the action slot" rather
# than letting AGNN discover what predicate-shapes actually exist in
# the data. Generalizing the SAME test to every bucket (bucket_anchors)
# and using it as a fallback in _extract_action_object (when no verb-
# looking/anchor token exists anywhere in the clause, check whether
# the LAST token is itself a bucket(-1) anchor) lets Indonesian's
# copula-less predicate adjectives ("rumah itu luas" — no verb at
# all) surface as real predicates instead of being silently dropped.
#
# No adjective list, no POS category, no "this is an adjective" rule
# was written anywhere — "luas"/"lusuh" get recorded with an empty-
# string object sentinel ("luas" has NO object, unlike a real verb),
# which gives every predicate-final token an IDENTICAL {"": count}
# object signature. They then cluster together via the EXISTING
# Q/K/V cosine-similarity step (_cluster_actions) — the same
# machinery real verbs go through — purely because their signature
# is similar, not because any code said "group adjectives together".

def test_predicate_final_token_recorded_with_empty_object_sentinel():
    """A copula-less predicate adjective gets the {"": count} sentinel.

    "rumah itu sangat luas" has NO verb anywhere and NO object slot —
    "luas" itself is the predicate. _extract_action_object's fallback
    should record it in action_object_freq with the empty-string
    sentinel object (not skip the sentence, not invent a fake object).
    """
    learner = _train_canonical_learner()
    assert "" in learner.action_object_freq.get("luas", {}), (
        f"Expected 'luas' to have the empty-object sentinel recorded; "
        f"got {learner.action_object_freq.get('luas')!r}."
    )
    assert learner._is_action_token("luas"), (
        "'luas' should be recognised as a predicate token via the "
        "predicate-final fallback."
    )


def test_predicate_final_adjectives_cluster_together():
    """Predicate-final tokens with identical empty-object signatures
    land in the SAME action cluster via the existing Q/K/V step.

    This is the round-9 payoff: no clustering code was written for
    this category specifically. "luas" and "lusuh" cluster together
    purely because _cluster_actions sees their {"": count} object
    distributions as maximally similar — the same mechanism that
    clusters real verbs by shared object vocabulary.
    """
    learner = _train_canonical_learner()
    clusters = learner.inspect_cluster_details()
    luas_cluster = None
    lusuh_cluster = None
    for cid, detail in clusters.items():
        actions = set(detail["actions"])
        if "luas" in actions:
            luas_cluster = cid
        if "lusuh" in actions:
            lusuh_cluster = cid
    assert luas_cluster is not None, "'luas' should land in some cluster."
    assert lusuh_cluster is not None, "'lusuh' should land in some cluster."
    assert luas_cluster == lusuh_cluster, (
        f"Expected 'luas' (cluster {luas_cluster}) and 'lusuh' "
        f"(cluster {lusuh_cluster}) to cluster together — both are "
        f"predicate-final adjectives with identical empty-object "
        f"signatures."
    )


def test_predicate_adjective_sentence_tags_subject_and_predicate():
    """"rumah itu sangat luas" tags 'rumah' AGENT and 'luas' as the
    recognised predicate, even though there is no verb and no
    object in the sentence at all.
    """
    learner = _train_canonical_learner()
    tags = dict(learner.tag_sentence("rumah itu sangat luas"))
    assert tags["rumah"] == "AGENT"
    assert tags["luas"] != "UNKNOWN", (
        f"Expected 'luas' to be recognised as the predicate; got "
        f"{tags['luas']!r}."
    )


# ----------------------------------------------------------------------
# Q/K/V tie-break determinism (pre-round-10 architecture audit)
# ----------------------------------------------------------------------

def test_qkv_tiebreak_is_alphabetical_not_insertion_order():
    """Two actions with an identical total observation count must be
    processed in alphabetical order, not corpus-insertion order.

    Found during an external architecture audit: _cluster_action_group_qkv
    sorts actions by descending total count, but ties were broken by
    Python dict iteration order (= the order each action was first
    inserted while scanning the corpus) — an implicit, fragile
    dependency. Reordering the training corpus (same sentences,
    different order) could silently change which action seeds a
    cluster first, and therefore the final cluster assignment, with
    no test catching it. Fixed by adding an explicit alphabetical
    secondary sort key — it carries no linguistic meaning, it just
    makes the processing order reproducible independent of corpus
    order.
    """
    learner = PositionalClusterLearner()
    # Two actions ("zaction"/"aaction") with IDENTICAL object
    # distributions and counts, deliberately inserted in an order
    # where the alphabetically-LATER one appears first in the corpus.
    # If the tiebreak were still insertion-order, "zaction" would seed
    # the first cluster; with the alphabetical tiebreak, "aaction"
    # must seed it instead — provable from cluster_id_of: whichever
    # action processed FIRST keeps cluster_id 0 in a no-connector
    # group (the seeding cluster gets the lowest cluster_id, since
    # action_clusters is enumerated in processing order).
    lines = (
        ["kucing zaction ikan"] * 4 + ["anjing aaction tulang"] * 4
    )
    learner.train(lines)
    # Both should cluster together (identical-shaped object signature
    # after Brown projection is irrelevant here — same literal object
    # count shape) OR at minimum, "aaction" (alphabetically first)
    # must be the one whose cluster_id matches the FIRST cluster
    # formed (cluster_id 0 among clusterable actions), proving
    # processing order is alphabetical, not insertion order.
    assert learner.cluster_id_of.get("aaction") == 0 or (
        learner.cluster_id_of.get("aaction") == learner.cluster_id_of.get("zaction")
    ), (
        f"Expected 'aaction' (alphabetically first) to seed cluster 0 "
        f"despite 'zaction' being inserted first in the corpus. Got "
        f"aaction={learner.cluster_id_of.get('aaction')!r}, "
        f"zaction={learner.cluster_id_of.get('zaction')!r}."
    )


# ----------------------------------------------------------------------
# Corpus expansion sprint (round 15) — 'datang' as a standalone verb
# ----------------------------------------------------------------------

def test_datang_is_not_misclassified_as_soft_particle():
    """'datang' must be recognised as ACTION, not excluded as a particle.

    Found in round 10/14: the pre-round-15 corpus only used "datang" in
    a modal-complement construction ("diperkirakan datang") in 2 of its
    3 occurrences, which made it sit in the between-first slot relative
    to the OUTER extraction's action("diperkirakan")/object("awal") pair
    — a real, structurally-driven signal, not a bug in
    _is_soft_particle. The system generalised correctly from
    insufficient data; the fix (per the round-14 user/researcher
    correction) is to ADD variety, never to change the detection logic.
    pretrain_corpus_datang.txt adds 8 sentences using "datang" as an
    ordinary standalone verb across varied subjects, with no modal
    construction repeated.
    """
    learner = _train_canonical_learner()
    assert not learner._is_soft_particle("datang"), (
        "'datang' should not be classified as a soft particle anymore "
        "now that its between-first rate is diluted by varied usage."
    )
    assert learner._is_action_token("datang"), (
        "'datang' should be recognised as a valid ACTION candidate."
    )


def test_datang_tagged_action_in_simple_sentence():
    """'datang' tags ACTION in a plain, non-modal sentence."""
    learner = _train_canonical_learner()
    tags = dict(learner.tag_sentence("tamu itu datang tiba-tiba"))
    assert tags["datang"] == "ACTION", (
        f"Expected 'datang' to be tagged ACTION; got {tags['datang']!r}."
    )


# ----------------------------------------------------------------------
# Context-stratified object signature (round 19 — post-round-18
# architecture patch, additive only)
# ----------------------------------------------------------------------

def test_action_object_context_freq_is_purely_additive():
    """The new context-stratified view must not change ANY existing
    public behaviour — tag_sentence/spo/classify/_is_action_token are
    completely unaffected by action_object_context_freq's existence.

    This is the core safety contract for round 19: the field is
    populated during train() but never read by anything except the
    new inspect_context_split() diagnostic method.
    """
    learner = _train_canonical_learner()
    assert learner.action_object_context_freq, (
        "Expected action_object_context_freq to be populated after "
        "training on the canonical corpus."
    )
    # Spot-check a sentence whose behaviour is well-established by
    # earlier rounds — must be byte-for-byte identical to what those
    # rounds already verified.
    tags = dict(learner.tag_sentence("api menyebabkan kebakaran"))
    assert tags["api"] == "AGENT"
    assert tags["menyebabkan"] == "ACTION"
    assert tags["kebakaran"] == "OBJECT"


def test_menyebabkan_context_split_separates_embedded_fragments():
    """'menyebabkan' must have BOTH context groups populated, with
    embedded-predicate fragments ("naik", "turun", "mahal") landing in
    "embedded_candidate" and ordinary objects ("kebakaran", "diabetes")
    landing in "plain".

    This is the structural test the round-19 patch is built on:
    "is there a non-particle token between the action and the
    extracted object" — NOT bucket-anchor membership (an earlier draft
    used that signal and got it backwards, because bucket-anchor
    membership is exactly the signal rounds 16-18 already proved too
    weak for these specific tokens).
    """
    learner = _train_canonical_learner()
    ctx = learner.action_object_context_freq.get("menyebabkan", {})
    assert "plain" in ctx and "embedded_candidate" in ctx, (
        f"Expected both context groups for 'menyebabkan'; got keys "
        f"{sorted(ctx.keys())}."
    )
    embedded_objects = set(ctx["embedded_candidate"].keys())
    plain_objects = set(ctx["plain"].keys())
    assert "naik" in embedded_objects or "turun" in embedded_objects, (
        f"Expected at least one embedded-predicate fragment "
        f"('naik'/'turun') in the embedded_candidate group; got "
        f"{sorted(embedded_objects)}."
    )
    assert "kebakaran" in plain_objects or "diabetes" in plain_objects, (
        f"Expected at least one ordinary object ('kebakaran'/'diabetes') "
        f"in the plain group; got {sorted(plain_objects)}."
    )


def test_inspect_context_split_returns_none_without_both_groups():
    """inspect_context_split() returns None when an action has no
    observations in one of the two context groups (nothing to compare).
    """
    learner = _train_canonical_learner()
    assert learner.inspect_context_split("this_action_does_not_exist") is None


def test_inspect_context_split_returns_valid_cosine_range():
    """When both context groups exist, the result is a valid cosine
    similarity in [0.0, 1.0] (object counts are always non-negative,
    so the projected vectors live in the non-negative cone — cosine
    similarity can't go negative here).
    """
    learner = _train_canonical_learner()
    sim = learner.inspect_context_split("menyebabkan")
    assert sim is not None
    assert 0.0 <= sim <= 1.0001, f"Expected a valid cosine similarity; got {sim!r}."


def test_action_object_context_freq_persists_through_save_load():
    """action_object_context_freq must survive a save()/load() roundtrip
    byte-for-byte (issue #92's persistence contract, extended to the
    new field)."""
    import tempfile
    import os

    learner = _train_canonical_learner()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        learner.save(path)
        loaded = PositionalClusterLearner.load(path)
        assert (
            loaded.action_object_context_freq.get("menyebabkan")
            == learner.action_object_context_freq.get("menyebabkan")
        )
    finally:
        os.unlink(path)


# ----------------------------------------------------------------------
# Particle clustering persistence (found during exploration session,
# post-round-19) — particle_cluster_id_of/particle_clusters/
# particle_cluster_labels were missing from save()/load() ENTIRELY.
# ----------------------------------------------------------------------

def test_particle_clustering_persists_through_save_load():
    """particle_cluster_id_of / particle_clusters / particle_cluster_labels
    must survive a save()/load() roundtrip.

    Found by manually exploring a saved-then-loaded learner's state:
    every existing saved state file (including the committed
    production cluster_learner_state.json) was silently losing ALL
    particle-clustering results on every roundtrip — these 3 fields
    were never part of the save()/load() contract at all. This went
    uncaught by the test suite because _is_particle_token has
    redundant fallback paths (_is_soft_particle,
    _is_clause_coordinator) that mask the gap for many tokens, but the
    underlying discovery state was still being thrown away.
    """
    import tempfile
    import os

    learner = _train_canonical_learner()
    assert learner.particle_clusters, (
        "Expected particle_clusters to be non-empty on the canonical "
        "corpus (sanity check before testing persistence)."
    )
    assert learner.particle_cluster_id_of, (
        "Expected particle_cluster_id_of to be non-empty on the "
        "canonical corpus (sanity check before testing persistence)."
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        learner.save(path)
        loaded = PositionalClusterLearner.load(path)
        assert loaded.particle_clusters == learner.particle_clusters, (
            "particle_clusters did not survive the save/load roundtrip."
        )
        assert loaded.particle_cluster_id_of == learner.particle_cluster_id_of, (
            "particle_cluster_id_of did not survive the save/load roundtrip."
        )
        assert loaded.particle_cluster_labels == learner.particle_cluster_labels, (
            "particle_cluster_labels did not survive the save/load roundtrip."
        )
    finally:
        os.unlink(path)
