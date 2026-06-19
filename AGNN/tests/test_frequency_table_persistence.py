"""
Tests for SemanticRoleClassifier frequency-table persistence (Worker 2).

Covers the Definition-of-Done from the Worker 2 brief:
    1. Save -> load -> frequency_table is byte-identical.
    2. classify() with persist_path set -> file is created automatically
       after every confident call (one that bumps the table).
    3. Loading from a non-existent file -> fresh empty table, no crash.
    4. AGNNCore with classifier_persist_path -> the classifier persists
       its table across two AGNNCore instances pointed at the same file.

Plus targeted edge cases:
    - persist_path=None -> no file IO happens at all (matches the
      pre-persistence behaviour).
    - Auto-save is atomic: a half-written tmp file never appears at
      the target path. We verify this indirectly by checking the
      target file is valid JSON after a sequence of classify() calls
      - any non-atomic write would risk a truncated file.
    - Forward compatibility: a JSON file that contains an unknown
      relation-type name (e.g. "FOO") loads without crashing; the
      unknown entry is dropped, known entries are kept.
    - Save creates parent directories on demand.
    - Save with an empty frequency table produces a valid empty JSON
      object rather than crashing.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_frequency_table_persistence.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
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

from neocortex.semantic_role_classifier import (  # noqa: E402
    RelationType,
    SemanticRoleClassifier,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def tmp_path_str(tmp_path: Path) -> str:
    """str path inside pytest's tmp_path - matches the API's str type."""
    return str(tmp_path / "freq.json")


def _populate_classifier(c: SemanticRoleClassifier) -> None:
    """Bump the classifier's table with a few confident classify() calls."""
    c.classify("lari menyebabkan ngos-ngosan")     # CAUSAL x1
    c.classify("lari menyebabkan ngos-ngosan")     # CAUSAL x2
    c.classify("manusia adalah mamalia")           # CATEGORICAL x1


def _table_to_str(c: SemanticRoleClassifier) -> dict:
    """Render the frequency_table as {pred: {rt_name: count}} for asserts."""
    return {
        pred: {rt.name: n for rt, n in counts.items()}
        for pred, counts in c.frequency_table.items()
    }


# ======================================================================
# DoD #1: save -> load -> frequency_table is identical
# ======================================================================


def test_save_then_load_round_trips_identically(tmp_path_str: str):
    """save(path) -> load(path) yields a classifier with the same table."""
    c1 = SemanticRoleClassifier()
    _populate_classifier(c1)
    c1.save(tmp_path_str)

    c2 = SemanticRoleClassifier.load(tmp_path_str)
    assert _table_to_str(c2) == _table_to_str(c1), (
        "round-trip should preserve the frequency_table exactly"
    )


def test_save_then_load_preserves_specific_counts(tmp_path_str: str):
    """Round-trip preserves the actual integer counts, not just the keys."""
    c1 = SemanticRoleClassifier()
    _populate_classifier(c1)
    c1.save(tmp_path_str)

    c2 = SemanticRoleClassifier.load(tmp_path_str)
    assert c2.frequency_table["menyebabkan"][RelationType.CAUSAL] == 2
    assert c2.frequency_table["adalah"][RelationType.CATEGORICAL] == 1


def test_save_writes_json_format_matching_spec(tmp_path_str: str):
    """The on-disk JSON matches the format documented in the spec.

    Spec example::

        {"menyebabkan": {"CAUSAL": 5, "CATEGORICAL": 1},
         "adalah":      {"CATEGORICAL": 12}}

    Keys are predicate strings, values are {relation_type_name: count}.
    """
    c = SemanticRoleClassifier()
    _populate_classifier(c)
    c.save(tmp_path_str)

    with open(tmp_path_str, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert isinstance(raw, dict)
    assert "menyebabkan" in raw
    assert raw["menyebabkan"]["CAUSAL"] == 2
    assert raw["adalah"]["CATEGORICAL"] == 1


# ======================================================================
# DoD #2: classify() with persist_path -> file created automatically
# ======================================================================


def test_classify_with_persist_path_creates_file(tmp_path_str: str):
    """A single confident classify() call writes the file at persist_path."""
    c = SemanticRoleClassifier(persist_path=tmp_path_str)
    assert not os.path.exists(tmp_path_str), (
        "file should not exist before any confident classify() call"
    )
    c.classify("lari menyebabkan ngos-ngosan")
    assert os.path.exists(tmp_path_str), (
        "auto-save should create the file after a confident classify() call"
    )
    with open(tmp_path_str, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw == {"menyebabkan": {"CAUSAL": 1}}, (
        f"expected one CAUSAL count for 'menyebabkan', got {raw}"
    )


def test_classify_with_persist_path_grows_file_on_each_call(
    tmp_path_str: str,
):
    """Each subsequent confident classify() call re-saves the grown table."""
    c = SemanticRoleClassifier(persist_path=tmp_path_str)
    c.classify("lari menyebabkan ngos-ngosan")
    with open(tmp_path_str) as f:
        assert json.load(f) == {"menyebabkan": {"CAUSAL": 1}}
    c.classify("lari menyebabkan ngos-ngosan")
    with open(tmp_path_str) as f:
        assert json.load(f) == {"menyebabkan": {"CAUSAL": 2}}
    c.classify("manusia adalah mamalia")
    with open(tmp_path_str) as f:
        assert json.load(f) == {
            "menyebabkan": {"CAUSAL": 2},
            "adalah": {"CATEGORICAL": 1},
        }


def test_classify_without_confidence_does_not_save(tmp_path_str: str):
    """A non-confident classify() call (no seed match) does NOT auto-save.

    Rationale: only confident (seed-match) calls bump the frequency
    table; a non-confident call returns CATEGORICAL without changing
    state, so there is nothing new to persist.
    """
    c = SemanticRoleClassifier(persist_path=tmp_path_str)
    # "blahblah" is not a seed - classify returns CATEGORICAL without
    # bumping the table.
    result = c.classify("X blahblah Y")
    assert result == RelationType.CATEGORICAL
    assert not os.path.exists(tmp_path_str), (
        "non-confident classify() should not trigger auto-save"
    )


# ======================================================================
# DoD #3: load from non-existent file -> fresh empty table, no crash
# ======================================================================


def test_load_nonexistent_file_returns_empty_table(tmp_path: Path):
    """load() on a path that does not exist -> empty frequency_table.

    The returned instance still has persist_path set so future
    classify() calls will create the file.
    """
    missing = str(tmp_path / "does_not_exist.json")
    c = SemanticRoleClassifier.load(missing)
    assert c.frequency_table == {}, (
        "loading from a non-existent file should yield an empty table"
    )
    # The returned instance keeps persist_path wired so the next
    # confident classify() will start writing to that path.
    assert c.persist_path == missing


def test_persist_path_to_nonexistent_file_does_not_crash_on_init(
    tmp_path: Path,
):
    """Constructing with persist_path to a missing file is fine.

    Auto-load on __post_init__ must swallow the FileNotFoundError
    rather than crashing the constructor.
    """
    missing = str(tmp_path / "subdir" / "does_not_exist.json")
    c = SemanticRoleClassifier(persist_path=missing)
    assert c.frequency_table == {}


# ======================================================================
# DoD #4: AGNNCore with classifier_persist_path -> classifier persists
#         across instances
# ======================================================================


def _import_agnn_core():
    """Load AGNN/core.py by path to avoid the self-ai/src/core name
    collision (same pattern as tests/test_core_wired.py).
    """
    import importlib.util as _ilu
    _core_path = _AGNP_ROOT / "core.py"
    _spec = _ilu.spec_from_file_location("agnn_core_persistence", _core_path)
    agnn_core_module = _ilu.module_from_spec(_spec)
    sys.modules["agnn_core_persistence"] = agnn_core_module
    _spec.loader.exec_module(agnn_core_module)
    return agnn_core_module.AGNNCore


def test_agnn_core_with_persist_path_propagates_to_classifier(tmp_path_str: str):
    """AGNNCore(classifier_persist_path=P) wires P into the classifier.

    Note: this test exercises the legacy SemanticRoleClassifier path, so
    it explicitly passes ``use_cluster_learner=False`` to opt out of the
    PositionalClusterLearner (which is the new default and does not use
    a persist_path).
    """
    AGNNCore = _import_agnn_core()
    core = AGNNCore(
        classifier_persist_path=tmp_path_str,
        use_cluster_learner=False,
    )
    if core.graph is None or core.trisynaptic is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")
    assert core.trisynaptic.role_classifier.persist_path == tmp_path_str


def test_agnn_core_persistence_survives_across_instances(tmp_path_str: str):
    """Two AGNNCore instances pointed at the same path share the table.

    Instance 1 learns three facts (each learn() triggers a classify()
    on the classifier) -> file is created with the accumulated table.
    Instance 2 is then constructed with the same path -> its
    classifier's frequency_table is loaded from the file and matches
    what instance 1 wrote.

    Note: this test exercises the legacy SemanticRoleClassifier path, so
    it explicitly passes ``use_cluster_learner=False`` to opt out of the
    PositionalClusterLearner (which is the new default and does not use
    a persist_path).
    """
    AGNNCore = _import_agnn_core()
    core1 = AGNNCore(
        classifier_persist_path=tmp_path_str,
        use_cluster_learner=False,
    )
    if core1.graph is None or core1.trisynaptic is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    core1.learn("q1?", "wrong1", "smoking causes lung damage")
    core1.learn("q2?", "wrong2", "lung damage causes cancer")
    core1.learn("q3?", "wrong3", "socrates is a human")

    assert os.path.exists(tmp_path_str), (
        "persistence file should exist after core1 learns facts"
    )
    with open(tmp_path_str) as f:
        on_disk = json.load(f)
    # Two "causes" + one "is a" -> 3 entries across 2 predicates.
    assert on_disk.get("causes", {}).get("CAUSAL") == 2
    assert on_disk.get("is a", {}).get("CATEGORICAL") == 1

    # Instance 2: should auto-load on construction.
    core2 = AGNNCore(
        classifier_persist_path=tmp_path_str,
        use_cluster_learner=False,
    )
    assert core2.trisynaptic is not None
    classifier2 = core2.trisynaptic.role_classifier
    assert classifier2.frequency_table.get("causes", {}).get(
        RelationType.CAUSAL
    ) == 2, "instance 2 should have loaded the CAUSAL count from disk"
    assert classifier2.frequency_table.get("is a", {}).get(
        RelationType.CATEGORICAL
    ) == 1, "instance 2 should have loaded the CATEGORICAL count from disk"


def test_agnn_core_default_classifier_persist_path_is_none():
    """AGNNCore() with no classifier_persist_path -> no persistence.

    Note: this test exercises the legacy SemanticRoleClassifier path, so
    it explicitly passes ``use_cluster_learner=False`` to opt out of the
    PositionalClusterLearner (which is the new default and does not use
    a persist_path).
    """
    AGNNCore = _import_agnn_core()
    core = AGNNCore(use_cluster_learner=False)
    assert core._classifier_persist_path is None
    if core.trisynaptic is not None:
        assert core.trisynaptic.role_classifier.persist_path is None


# ======================================================================
# Edge cases
# ======================================================================


def test_persist_path_none_disables_all_file_io(tmp_path: Path):
    """persist_path=None -> no file is ever created, classify still works."""
    c = SemanticRoleClassifier(persist_path=None)
    c.classify("lari menyebabkan ngos-ngosan")
    c.classify("manusia adalah mamalia")
    # No file should have been created anywhere in tmp_path.
    files_created = list(tmp_path.rglob("*.json"))
    assert files_created == [], (
        f"persist_path=None should not create any files, got {files_created}"
    )
    # But the in-memory table should still be bumped.
    assert c.frequency_table["menyebabkan"][RelationType.CAUSAL] == 1
    assert c.frequency_table["adalah"][RelationType.CATEGORICAL] == 1


def test_save_creates_parent_directories(tmp_path: Path):
    """save() into a not-yet-existing directory creates it on demand."""
    nested = tmp_path / "a" / "b" / "c" / "freq.json"
    c = SemanticRoleClassifier()
    _populate_classifier(c)
    c.save(str(nested))
    assert nested.exists()
    with open(nested) as f:
        raw = json.load(f)
    assert "menyebabkan" in raw


def test_save_empty_table_produces_empty_json_object(tmp_path_str: str):
    """save() with an empty frequency_table writes ``{}``, not ``null``."""
    c = SemanticRoleClassifier()
    assert c.frequency_table == {}
    c.save(tmp_path_str)
    with open(tmp_path_str) as f:
        raw = json.load(f)
    assert raw == {}


def test_load_skips_unknown_relation_type_names(tmp_path_str: str):
    """A JSON file with an unknown RelationType name loads without crashing.

    Forward compatibility: a future version that adds new
    RelationTypes (e.g. "SPATIAL_RELATION_V2") should not crash this
    version's loader. Unknown entries are silently dropped; known
    entries are kept.
    """
    payload = {
        "menyebabkan": {"CAUSAL": 5, "FOO_BAR_V2": 3},
        "adalah": {"CATEGORICAL": 12},
    }
    with open(tmp_path_str, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    c = SemanticRoleClassifier.load(tmp_path_str)
    # The known entries are preserved exactly...
    assert c.frequency_table["menyebabkan"][RelationType.CAUSAL] == 5
    assert c.frequency_table["adalah"][RelationType.CATEGORICAL] == 12
    # ...and the unknown name did not leak into the table as a string
    # key or anything else.
    assert all(
        isinstance(rt, RelationType)
        for bucket in c.frequency_table.values()
        for rt in bucket
    ), "all RelationType keys in the loaded table must be enum members"


def test_auto_save_is_atomic_under_normal_load(tmp_path_str: str):
    """After many rapid classify() calls the file is always valid JSON.

    Indirect atomic-write test: if the write were not atomic (e.g.
    open(path, 'w') + write), a crash mid-write could leave a truncated
    JSON. We can't easily crash the process mid-write in a unit test,
    but we *can* verify that after a sequence of rapid saves the file
    parses as valid JSON every time we read it back - and that the
    final content matches the in-memory table exactly.

    We deliberately use ``override_threshold=99`` so the seed-match
    path keeps bumping the table on every call (once the default
    threshold of 3 is hit, the override path takes over and the table
    stops growing - that's a separate, intended behaviour tested in
    test_semantic_role_classifier.py).
    """
    c = SemanticRoleClassifier(
        persist_path=tmp_path_str, override_threshold=99
    )
    for i in range(10):
        c.classify("X menyebabkan Y")
        # Re-read after every save: any non-atomic write would risk
        # reading a half-written file here.
        with open(tmp_path_str) as f:
            raw = json.load(f)
        assert raw == {"menyebabkan": {"CAUSAL": i + 1}}, (
            f"after {i + 1} calls, file should contain CAUSAL={i + 1}, "
            f"got {raw}"
        )
    # Final in-memory table matches the final on-disk table.
    assert c.frequency_table["menyebabkan"][RelationType.CAUSAL] == 10


def test_save_is_idempotent_for_unchanged_table(tmp_path_str: str):
    """save() twice with no changes writes byte-identical content.

    This makes the persistence file friendly to version control and
    rsync - re-saving an unchanged table produces no diff.
    """
    c = SemanticRoleClassifier()
    _populate_classifier(c)
    c.save(tmp_path_str)
    with open(tmp_path_str, "rb") as f:
        bytes_1 = f.read()
    c.save(tmp_path_str)
    with open(tmp_path_str, "rb") as f:
        bytes_2 = f.read()
    assert bytes_1 == bytes_2, (
        "saving an unchanged table should produce byte-identical output"
    )
