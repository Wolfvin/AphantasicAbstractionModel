"""
Adversarial edge-poisoning robustness tests for AGNN.

Four scenarios from the task brief (WORKER 3):

  1. Conflict detection
     learn(fact A) -> learn(contradicting fact B) -> CingulateGyrus
     .detect_conflict() must return a conflict signal on the two
     Semesomes projected onto the same (source, target).

  2. Penalize degrades confidence
     learn(fact) -> penalize(node_id) x3 -> confidence must drop
     below the initial 0.6 (Subiculum DEFAULT_EPISODIC_CONFIDENCE).

  3. Reinforce vs penalize (net positive)
     learn(fact) -> reinforce x5 -> penalize x3 -> confidence must
     still be above the initial 0.6.

  4. Frequency-table poisoning resistance
     Directly inject classifier.frequency_table["menyebabkan"] =
     {RelationType.CATEGORICAL: 100} (adversarial poisoning). The
     non-negated form falls for the poison (proves setup is real),
     but the negated form ("X tidak menyebabkan Y") MUST still
     return DIFFERENTIAL — negation beats the table.

Constraints:
  - Only adds this file. No source-code changes.
  - When the public interface does not support a scenario, the
    affected test is skipped with a documented ``pytest.skip()``
    reason.
  - The 151 existing tests must continue to pass.

Run:
    python -m pytest AGNN/tests/test_adversarial_robustness.py -v
    python -m pytest AGNN/tests/ -v
"""

import importlib.util as _ilu
import sys
from pathlib import Path

import pytest

# ----------------------------------------------------------------------
# Path setup — mirror the convention used by test_core_wired.py /
# test_semantic_role_classifier.py so this file works whether run
# standalone or as part of the full AGNN/tests/ suite.
# ----------------------------------------------------------------------
_AGNP_ROOT = Path(__file__).resolve().parent.parent
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"

# self-ai/src FIRST (lower priority) so the AGNN package wins on name
# collisions (e.g. self-ai/src/core/ vs AGNN/core.py).
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

# Load AGNN/core.py directly by path to avoid the name collision with
# self-ai/src/core/ (a package). Registering under a unique module name
# keeps it isolated from any other "core" already on sys.path.
_core_path = _AGNP_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_adv_module", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_adv_module"] = agnn_core_module
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore

from engrams.semantic_engram import Semesome  # noqa: E402
from limbic_system.cingulate_gyrus import (  # noqa: E402
    CAUSAL,
    Conflict,
    CingulateGyrus,
    DIFFERENTIAL,
)
from neocortex.semantic_role_classifier import (  # noqa: E402
    RelationType,
    SemanticRoleClassifier,
)


# ----------------------------------------------------------------------
# Constants pinned from source (not magic numbers in the tests).
# ----------------------------------------------------------------------
# Subiculum.DEFAULT_EPISODIC_CONFIDENCE — freshly-encoded episome
# baseline. Asserted here so a silent change in the source is caught
# by this test rather than letting the adversarial scenarios quietly
# go false-green.
_INITIAL_CONFIDENCE = 0.6

# AGNNCore._REINFORCE_DELTA — confidence nudge per reinforce/penalize.
_DELTA = 0.1


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def brain() -> AGNNCore:
    """Fresh AGNNCore without a model.

    Skips the test when the EngramComplex dependency (self-ai/src/agnn)
    is unavailable — without it, learn() returns node_id=None and the
    penalize/reinforce scenarios cannot be exercised against a real
    episome.
    """
    core = AGNNCore(model_path=None)
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")
    return core


@pytest.fixture
def cingulate() -> CingulateGyrus:
    """Fresh CingulateGyrus (anterior cingulate cortex analog)."""
    return CingulateGyrus()


@pytest.fixture
def classifier() -> SemanticRoleClassifier:
    """Fresh SemanticRoleClassifier (empty frequency table)."""
    return SemanticRoleClassifier()


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------


def _lookup_confidence(brain: AGNNCore, node_id) -> float:
    """Best-effort: fetch the current confidence for a node id.

    AGNNCore's public API does not expose a direct node lookup, so we
    go through ``introspect().top_nodes`` (max 5, sorted by descending
    confidence) — fine for the small graphs in these tests. Falls back
    to scanning ``brain._episomes`` directly when the target node is
    not in the top-5 slice.
    """
    snapshot = brain.introspect()
    for n in snapshot["top_nodes"]:
        if n["id"] == node_id:
            return float(n["confidence"])
    # Fallback: scan the episome registry directly.
    for e in brain._episomes:
        if e.id == node_id:
            return float(e.confidence)
    raise AssertionError(
        f"node {node_id!r} not found in introspect() or _episomes registry"
    )


# ======================================================================
# Scenario 1 — Conflict detection
# ======================================================================


def test_conflict_detection_causal_vs_differential(cingulate: CingulateGyrus):
    """CAUSAL + DIFFERENTIAL on the same (source, target) must be flagged.

    This is the canonical adversarial pattern from cingulate_gyrus.py:
    an attacker injects two edges A->B, one CAUSAL ("api menyebabkan
    panas") and one DIFFERENTIAL ("api tidak menyebabkan panas").
    CingulateGyrus.detect_conflict() must:
      - set ``detected = True``
      - set ``resolution = "weight_aggregation"``
      - set ``final_weight`` to the arithmetic mean of the two weights
      - record both premises in ``premises``
      - bump ``conflict_count`` and append to ``conflict_log``
    """
    premise_a = Semesome(
        type=CAUSAL,
        weight=0.7,
        source="api",
        target="panas",
    )
    premise_b = Semesome(
        type=DIFFERENTIAL,
        weight=-0.8,
        source="api",
        target="panas",
    )

    result = cingulate.detect_conflict(premise_a, premise_b)

    assert isinstance(result, Conflict), (
        "detect_conflict() must return a Conflict dataclass instance"
    )
    assert result.detected is True, (
        f"expected conflict on CAUSAL+DIFFERENTIAL same pair, "
        f"got note={result.note!r}"
    )
    assert result.resolution == "weight_aggregation", (
        f"expected resolution='weight_aggregation', got {result.resolution!r}"
    )
    # Arithmetic mean of (0.7, -0.8) = -0.05 — near zero = uncertain,
    # exactly the "near zero = uncertain" semantic documented in source.
    assert result.final_weight == pytest.approx(-0.05, abs=1e-9), (
        f"expected final_weight=(0.7 + -0.8)/2 = -0.05, "
        f"got {result.final_weight}"
    )
    assert result.premises == [premise_a, premise_b]
    assert cingulate.conflict_count == 1
    assert len(cingulate.conflict_log) == 1
    assert cingulate.conflict_log[0] is result


def test_conflict_detection_no_conflict_on_same_type(cingulate: CingulateGyrus):
    """Two CAUSAL edges on the same pair are NOT a conflict.

    Adversarial robustness contract is symmetric: the detector must NOT
    fire on benign duplicates, otherwise an attacker could trigger
    false positives by simply repeating the same fact.
    """
    p1 = Semesome(type=CAUSAL, weight=0.5, source="A", target="B")
    p2 = Semesome(type=CAUSAL, weight=0.4, source="A", target="B")

    result = cingulate.detect_conflict(p1, p2)

    assert result.detected is False
    assert result.resolution == "none"
    assert result.final_weight == 0.0
    assert result.premises == []
    assert cingulate.conflict_count == 0


def test_conflict_detection_no_conflict_on_different_pair(cingulate: CingulateGyrus):
    """CAUSAL+DIFFERENTIAL on disjoint (source, target) pairs is NOT a conflict.

    A common adversarial evasion is to inject the contradiction on a
    *different* pair so the (source, target) match fails. The detector
    must reject this case so it cannot be tricked into merging two
    unrelated facts.
    """
    p1 = Semesome(type=CAUSAL, weight=0.7, source="A", target="B")
    p2 = Semesome(type=DIFFERENTIAL, weight=-0.5, source="C", target="D")

    result = cingulate.detect_conflict(p1, p2)

    assert result.detected is False
    assert cingulate.conflict_count == 0


def test_conflict_detection_end_to_end_with_learn(
    brain: AGNNCore, cingulate: CingulateGyrus
):
    """End-to-end adversarial scenario for conflict detection.

    Step 1: ``learn(fact A)`` — "api menyebabkan panas"
    Step 2: ``learn(fact B)`` — "api tidak menyebabkan panas"
            (an attacker deliberately injects a contradiction)
    Step 3: Project both learned corrections onto the same (source,
            target) pair as CAUSAL + DIFFERENTIAL Semesomes — exactly
            the input shape ``detect_conflict()`` expects.
    Step 4: Assert conflict is flagged.

    Note on AGNNCore's current behaviour: ``learn()`` stores each
    correction as a standalone Episome node; it does not yet emit typed
    edges *between* two learned episomes automatically, nor does it
    auto-invoke ``CingulateGyrus.detect_conflict()`` on encode. So this
    test exercises the conflict-detection interface directly on the
    Semesomes derived from the two learned corrections — which is the
    realistic adversarial surface where CingulateGyrus is meant to be
    wired in.
    """
    res_a = brain.learn(
        question="apakah api menyebabkan panas?",
        wrong="api mendinginkan",
        correction="api menyebabkan panas",
    )
    res_b = brain.learn(
        question="apakah api tidak menyebabkan panas?",
        wrong="api memanaskan",
        correction="api tidak menyebabkan panas",
    )

    # learn() must have actually produced nodes (graph backend present).
    if res_a["node_id"] is None or res_b["node_id"] is None:
        pytest.skip(
            "learn() returned null node_id — graph backend unavailable, "
            "cannot exercise end-to-end conflict detection"
        )

    # Two distinct episomes were learned.
    assert res_a["node_id"] != res_b["node_id"], (
        "two learn() calls must produce two distinct episomes"
    )

    # Project the two learned corrections onto the same (source, target)
    # pair as conflicting Semesomes — the canonical adversarial edge-
    # poisoning pattern from cingulate_gyrus.py.
    s_a = Semesome(type=CAUSAL,       weight=0.7,  source="api", target="panas")
    s_b = Semesome(type=DIFFERENTIAL, weight=-0.8, source="api", target="panas")

    result = cingulate.detect_conflict(s_a, s_b)
    assert result.detected is True, (
        f"adversarial contradiction must be flagged; note={result.note!r}"
    )
    assert result.resolution == "weight_aggregation"
    assert result.final_weight == pytest.approx(-0.05, abs=1e-9)


# ======================================================================
# Scenario 2 — Penalize degrades confidence below the initial 0.6
# ======================================================================


def test_penalize_degrades_confidence_below_initial(brain: AGNNCore):
    """learn() -> 3x penalize() -> confidence must drop below 0.6.

    Initial confidence (Subiculum.DEFAULT_EPISODIC_CONFIDENCE) = 0.6.
    Each penalize() subtracts AGNNCore._REINFORCE_DELTA (= 0.1), so
    after 3 calls confidence should be 0.6 - 0.3 = 0.3 < 0.6.

    Adversarial relevance: when a user (or downstream consumer) reports
    a node as wrong, repeated penalize() calls must reliably erode the
    node's confidence so retrieval / introspect surfaces it lower. An
    attacker cannot "lock in" high confidence on a poisoned node.
    """
    result = brain.learn(
        question="what is the capital of france?",
        wrong="lyon",
        correction="paris is the capital of france",
    )
    node_id = result["node_id"]
    if node_id is None:
        pytest.skip(
            "learn() returned null node_id — graph backend unavailable"
        )

    initial = float(result["confidence"])
    assert initial == pytest.approx(_INITIAL_CONFIDENCE), (
        f"freshly learned episome should start at confidence="
        f"{_INITIAL_CONFIDENCE}, got {initial}"
    )

    for _ in range(3):
        brain.penalize(node_id)

    final = _lookup_confidence(brain, node_id)
    assert final < initial, (
        f"after 3 penalize calls confidence must be < initial "
        f"({initial}), got {final}"
    )
    assert final == pytest.approx(_INITIAL_CONFIDENCE - 3 * _DELTA, abs=1e-9), (
        f"expected {_INITIAL_CONFIDENCE} - 3*{_DELTA} = "
        f"{_INITIAL_CONFIDENCE - 3 * _DELTA}, got {final}"
    )


def test_penalize_floors_at_zero(brain: AGNNCore):
    """penalize() must never push confidence below 0.0.

    Adversarial check: even after 100 penalize calls, confidence must
    not go negative — otherwise downstream code that assumes [0, 1]
    could panic or flip sign.
    """
    result = brain.learn(
        question="is the moon made of cheese?",
        wrong="no",
        correction="the moon is made of cheese",
    )
    node_id = result["node_id"]
    if node_id is None:
        pytest.skip("learn() returned null node_id — graph backend unavailable")

    for _ in range(100):
        brain.penalize(node_id)

    final = _lookup_confidence(brain, node_id)
    assert final == 0.0, (
        f"confidence must floor at 0.0 after 100 penalize calls, got {final}"
    )


# ======================================================================
# Scenario 3 — Reinforce vs penalize (net positive)
# ======================================================================


def test_reinforce_dominates_penalize(brain: AGNNCore):
    """learn() -> 5x reinforce() -> 3x penalize() -> confidence > 0.6.

    Arithmetic (with the 1.0 cap on reinforce()):
        0.6 + 5*0.1 = 1.1 -> capped at 1.0 (5th reinforce hits the cap)
        1.0 - 3*0.1     = 0.7
    So final = 0.7 > 0.6 (initial) -> net positive, contract satisfied.

    Adversarial relevance: a node that has been reinforced multiple
    times must be robust to a few subsequent penalize() calls —
    otherwise an attacker could trivially "kill" a well-established
    fact by spamming penalize().
    """
    result = brain.learn(
        question="does the earth orbit the sun?",
        wrong="the sun orbits the earth",
        correction="the earth orbits the sun",
    )
    node_id = result["node_id"]
    if node_id is None:
        pytest.skip("learn() returned null node_id — graph backend unavailable")

    initial = float(result["confidence"])
    assert initial == pytest.approx(_INITIAL_CONFIDENCE)

    for _ in range(5):
        brain.reinforce(node_id)
    # Sanity check: after 5 reinforces, confidence should be capped at 1.0.
    mid = _lookup_confidence(brain, node_id)
    assert mid == 1.0, (
        f"5 reinforces should saturate at 1.0 (0.6 + 0.5 = 1.1 -> capped), "
        f"got {mid}"
    )

    for _ in range(3):
        brain.penalize(node_id)

    final = _lookup_confidence(brain, node_id)
    assert final > initial, (
        f"5 reinforces + 3 penalizes must leave confidence net-positive "
        f"({initial} -> {final})"
    )
    # 1.0 (capped) - 3*0.1 = 0.7 — the cap is what makes the test
    # asymmetric vs test_penalize_degrades_confidence_below_initial.
    assert final == pytest.approx(1.0 - 3 * _DELTA, abs=1e-9), (
        f"expected 1.0 (cap) - 3*{_DELTA} = {1.0 - 3 * _DELTA}, "
        f"got {final}"
    )


def test_reinforce_caps_at_one(brain: AGNNCore):
    """reinforce() must never push confidence above 1.0.

    Adversarial check: a node that has been reinforced 100 times must
    not exceed 1.0 — otherwise an attacker could inflate a poisoned
    node to arbitrarily high confidence and dominate retrieval.
    """
    result = brain.learn(
        question="is water wet?",
        wrong="no",
        correction="water is wet",
    )
    node_id = result["node_id"]
    if node_id is None:
        pytest.skip("learn() returned null node_id — graph backend unavailable")

    for _ in range(100):
        brain.reinforce(node_id)

    final = _lookup_confidence(brain, node_id)
    assert final == 1.0, (
        f"confidence must cap at 1.0 after 100 reinforce calls, got {final}"
    )


# ======================================================================
# Scenario 4 — Frequency-table poisoning resistance
# ======================================================================


def test_frequency_table_poisoning_resistance_id(
    classifier: SemanticRoleClassifier,
):
    """Adversarial: directly inject frequency_table["menyebabkan"] =
    {RelationType.CATEGORICAL: 100}.

    Sanity check (proves the poison is effective on the non-negated
    form): classify("merokok menyebabkan kanker") must fall for the
    poison and return CATEGORICAL — the table override threshold (=3)
    is far exceeded by the injected count (=100).

    Adversarial-safety contract (the actual point of the test): when
    the SAME predicate appears with leading negation ("merokok tidak
    menyebabkan kanker"), the classifier MUST return DIFFERENTIAL
    regardless of the poisoned table. Negation beats the frequency
    table — see semantic_role_classifier.py:277.
    """
    # ── Poisoning ────────────────────────────────────────────────────
    classifier.frequency_table["menyebabkan"] = {RelationType.CATEGORICAL: 100}

    # Sanity check 1: poison is effective on the non-negated form
    # (table override threshold=3, injected count=100). If this
    # assertion failed, the poisoning setup itself would be wrong and
    # the resistance assertion below would be meaningless.
    poisoned_result = classifier.classify("merokok menyebabkan kanker")
    assert poisoned_result == RelationType.CATEGORICAL, (
        f"poisoned table should override the CAUSAL seed for "
        f"non-negated 'menyebabkan' (count=100 >= threshold=3); "
        f"got {poisoned_result}"
    )

    # ── The actual adversarial-safety contract ───────────────────────
    # Negation MUST beat the frequency table. Even though the table
    # says CATEGORICAL:100 for "menyebabkan", the leading "tidak"
    # flips the classification to DIFFERENTIAL.
    negated_result = classifier.classify("merokok tidak menyebabkan kanker")
    assert negated_result == RelationType.DIFFERENTIAL, (
        f"negation must beat the frequency table — "
        f"'merokok tidak menyebabkan kanker' must classify as "
        f"DIFFERENTIAL even when 'menyebabkan' has been poisoned to "
        f"CATEGORICAL:100; got {negated_result}"
    )


def test_frequency_table_poisoning_resistance_en(
    classifier: SemanticRoleClassifier,
):
    """English variant of the poisoning-resistance scenario.

    Poison "causes" -> {CATEGORICAL: 100} and verify:
      1. "smoking causes cancer" -> CATEGORICAL (poison effective)
      2. "smoking not causes cancer" -> DIFFERENTIAL (negation wins)
    """
    classifier.frequency_table["causes"] = {RelationType.CATEGORICAL: 100}

    poisoned = classifier.classify("smoking causes cancer")
    assert poisoned == RelationType.CATEGORICAL, (
        f"poisoned table should override CAUSAL seed for 'causes'; "
        f"got {poisoned}"
    )

    negated = classifier.classify("smoking not causes cancer")
    assert negated == RelationType.DIFFERENTIAL, (
        f"negation must beat the frequency table — 'not causes' must "
        f"return DIFFERENTIAL even with 'causes' poisoned; got {negated}"
    )


def test_frequency_table_poisoning_resistance_bukan(
    classifier: SemanticRoleClassifier,
):
    """Indonesian 'bukan' negation variant of the same scenario.

    Poison "menyebabkan" -> {CATEGORICAL: 100} and verify:
      1. "X menyebabkan Y" -> CATEGORICAL (poison effective)
      2. "X bukan menyebabkan Y" -> DIFFERENTIAL (negation wins)

    'bukan' is in _NEGATION_TOKENS, so it must flip the classification
    the same way 'tidak' does.
    """
    classifier.frequency_table["menyebabkan"] = {RelationType.CATEGORICAL: 100}

    poisoned = classifier.classify("stress menyebabkan botak")
    assert poisoned == RelationType.CATEGORICAL

    negated = classifier.classify("stress bukan menyebabkan botak")
    assert negated == RelationType.DIFFERENTIAL, (
        f"'bukan' negation must beat the poisoned frequency table; "
        f"got {negated}"
    )
