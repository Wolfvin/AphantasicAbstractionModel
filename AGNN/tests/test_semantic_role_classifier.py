"""
Tests for ``SemanticRoleClassifier`` - Phase 3 edge-type inference.

Covers the Definition-of-Done from the Phase 3 task brief:
    1. CAUSAL classification (Indonesian + English).
    2. CATEGORICAL classification (Indonesian + English).
    3. FUNCTIONAL classification (Indonesian + English).
    4. DIFFERENTIAL classification, including negation+verb
       ("X tidak menyebabkan Y" -> DIFFERENTIAL, not CAUSAL).
    5. TEMPORAL classification (Indonesian + English).
    6. DISCURSIVE classification.
    7. Fallback to CATEGORICAL on unknown predicates.
    8. Frequency-table learning: confident calls populate the table,
       and once a type reaches the override threshold it wins over
       the seed rules (including for predicates that have no seed
       entry at all).
    9. Negation beats the frequency table - "X tidak menyebabkan Y"
       stays DIFFERENTIAL even when "menyebabkan" has been voted
       CAUSAL many times.
   10. SPO parser exposes the parsed subject/predicate/object for
       audit and downstream use.
   11. Standalone import + call works as documented in the spec:
       ``SemanticRoleClassifier().classify("lari menyebabkan ngos-ngosan")``
       returns ``RelationType.CAUSAL``.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_semantic_role_classifier.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
# Tests are invoked as: python -m pytest AGNN/tests/test_semantic_role_classifier.py -v
# So we add the AGNN/ directory (parent of tests/) to sys.path.
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
    SPO,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def classifier() -> SemanticRoleClassifier:
    """Fresh SemanticRoleClassifier per test (empty frequency table)."""
    return SemanticRoleClassifier()


# ======================================================================
# CAUSAL
# ======================================================================


def test_causal_menyebabkan(classifier: SemanticRoleClassifier):
    """'X menyebabkan Y' -> CAUSAL (Indonesian seed)."""
    assert classifier.classify("lari menyebabkan ngos-ngosan") == RelationType.CAUSAL


def test_causal_mengakibatkan(classifier: SemanticRoleClassifier):
    """'X mengakibatkan Y' -> CAUSAL (Indonesian seed)."""
    assert classifier.classify("banjir mengakibatkan kerugian") == RelationType.CAUSAL


def test_causal_causes_english(classifier: SemanticRoleClassifier):
    """'X causes Y' -> CAUSAL (English seed)."""
    assert classifier.classify("smoking causes lung cancer") == RelationType.CAUSAL


def test_causal_sehingga(classifier: SemanticRoleClassifier):
    """'X sehingga Y' -> CAUSAL (Indonesian seed)."""
    assert classifier.classify("hujan deras sehingga banjir") == RelationType.CAUSAL


# ======================================================================
# CATEGORICAL
# ======================================================================


def test_categorical_adalah(classifier: SemanticRoleClassifier):
    """'X adalah Y' -> CATEGORICAL (Indonesian seed)."""
    assert classifier.classify("manusia adalah mamalia") == RelationType.CATEGORICAL


def test_categorical_is_a_english(classifier: SemanticRoleClassifier):
    """'X is a Y' -> CATEGORICAL (English seed)."""
    assert classifier.classify("a dog is a mammal") == RelationType.CATEGORICAL


def test_categorical_merupakan(classifier: SemanticRoleClassifier):
    """'X merupakan Y' -> CATEGORICAL (Indonesian seed)."""
    assert classifier.classify("Jakarta merupakan ibu kota") == RelationType.CATEGORICAL


def test_categorical_bagian_dari_multiword(classifier: SemanticRoleClassifier):
    """'X bagian dari Y' -> CATEGORICAL (multi-word seed).

    This exercises the longest-first seed sorting: "bagian dari" must
    win over any single-word "dari" that might appear elsewhere.
    """
    assert classifier.classify("paru bagian dari sistem pernapasan") == RelationType.CATEGORICAL


# ======================================================================
# FUNCTIONAL
# ======================================================================


def test_functional_membutuhkan(classifier: SemanticRoleClassifier):
    """'X membutuhkan Y' -> FUNCTIONAL (Indonesian seed)."""
    assert classifier.classify("tanaman membutuhkan air") == RelationType.FUNCTIONAL


def test_functional_requires_english(classifier: SemanticRoleClassifier):
    """'X requires Y' -> FUNCTIONAL (English seed)."""
    assert classifier.classify("the engine requires fuel") == RelationType.FUNCTIONAL


# ======================================================================
# DIFFERENTIAL (incl. negation)
# ======================================================================


def test_differential_bukan(classifier: SemanticRoleClassifier):
    """'X bukan Y' -> DIFFERENTIAL (Indonesian seed)."""
    assert classifier.classify("kelelawar bukan burung") == RelationType.DIFFERENTIAL


def test_differential_is_not_english(classifier: SemanticRoleClassifier):
    """'X is not Y' -> DIFFERENTIAL (English seed)."""
    assert classifier.classify("a whale is not a fish") == RelationType.DIFFERENTIAL


def test_differential_negation_overrides_causal_seed(classifier: SemanticRoleClassifier):
    """'X tidak menyebabkan Y' -> DIFFERENTIAL, NOT CAUSAL.

    This is the headline bug the Phase 3 brief calls out: CA1's
    bag-of-words scan saw "menyebabkan" and fired CAUSAL regardless of
    the preceding "tidak". The classifier must detect the negation
    and flip to DIFFERENTIAL.
    """
    result = classifier.classify("merokok tidak menyebabkan awet muda")
    assert result == RelationType.DIFFERENTIAL, (
        f"negation 'tidak' must flip CAUSAL seed to DIFFERENTIAL, got {result}"
    )


def test_differential_negation_overrides_causal_english(classifier: SemanticRoleClassifier):
    """'X does not cause Y' -> DIFFERENTIAL, NOT CAUSAL (English)."""
    result = classifier.classify("exercise does not cause weight gain")
    assert result == RelationType.DIFFERENTIAL


# ======================================================================
# TEMPORAL
# ======================================================================


def test_temporal_setelah(classifier: SemanticRoleClassifier):
    """'X setelah Y' -> TEMPORAL (Indonesian seed)."""
    assert classifier.classify("padi tumbuh setelah hujan") == RelationType.TEMPORAL


def test_temporal_before_english(classifier: SemanticRoleClassifier):
    """'X before Y' -> TEMPORAL (English seed)."""
    assert classifier.classify("breakfast before lunch") == RelationType.TEMPORAL


# ======================================================================
# DISCURSIVE
# ======================================================================


def test_discursive_menurut(classifier: SemanticRoleClassifier):
    """'menurut X, Y' -> DISCURSIVE (Indonesian seed)."""
    assert classifier.classify("menurut buku ini manusia adalah mamalia") == RelationType.DISCURSIVE


def test_discursive_according_to(classifier: SemanticRoleClassifier):
    """'according to X, Y' -> DISCURSIVE (English seed)."""
    assert classifier.classify("according to einstein relativity holds") == RelationType.DISCURSIVE


# ======================================================================
# Fallback
# ======================================================================


def test_fallback_unknown_predicate_is_categorical(classifier: SemanticRoleClassifier):
    """'X blahblah Y' (no seed match, no frequency entry) -> CATEGORICAL.

    This matches TrisynapticCircuit's pre-Phase-3 behaviour so existing
    pipelines keep working when the classifier cannot commit.
    """
    assert classifier.classify("X blahblah Y") == RelationType.CATEGORICAL


def test_fallback_empty_string_is_categorical(classifier: SemanticRoleClassifier):
    """Empty input -> CATEGORICAL (no crash)."""
    assert classifier.classify("") == RelationType.CATEGORICAL


def test_fallback_single_word_is_categorical(classifier: SemanticRoleClassifier):
    """Single-token input -> CATEGORICAL (no predicate extractable)."""
    assert classifier.classify("apple") == RelationType.CATEGORICAL


# ======================================================================
# Frequency-table learning
# ======================================================================


def test_frequency_table_populated_by_confident_calls(classifier: SemanticRoleClassifier):
    """Each confident classify() call bumps the predicate's count."""
    classifier.classify("stress memicu insomnia")
    classifier.classify("kopi memicu jantung berdebar")
    # "memicu" is a CAUSAL seed - two confident calls should bump its
    # CAUSAL count to 2.
    counts = classifier.frequency_table.get("memicu", {})
    assert counts.get(RelationType.CAUSAL) == 2


def test_frequency_table_override_threshold_wins_over_seed():
    """After override_threshold counts, the table wins - even for a
    predicate that has no seed entry at all.

    We simulate a predicate ("mendorong") that has been voted CAUSAL
    `threshold` times via prior confident classifications (in practice
    those would come from repeated seed matches before the seed was
    removed, or from a learning loop that injects counts manually).
    The classifier must consult the table even when the predicate has
    no seed match - otherwise the table would never override anything
    new.
    """
    c = SemanticRoleClassifier(override_threshold=3)
    # Manually seed the table for a non-seed predicate to simulate 3
    # prior confident CAUSAL votes.
    c.frequency_table["mendorong"] = {RelationType.CAUSAL: 3}
    assert c.classify("X mendorong Y") == RelationType.CAUSAL


def test_frequency_table_below_threshold_does_not_override():
    """Below the override threshold, the seed rules still win."""
    c = SemanticRoleClassifier(override_threshold=3)
    # One prior CAUSAL vote for "menyebabkan" - below threshold.
    c.frequency_table["menyebabkan"] = {RelationType.CAUSAL: 1}
    # The seed rule should still fire CAUSAL anyway (consistent result),
    # but the point of this test is that the table alone is not yet
    # authoritative. We verify by forcing an ambiguous scenario:
    # inject a single FUNCTIONAL vote for a CAUSAL seed predicate.
    # Below threshold, the seed CAUSAL should still win.
    c2 = SemanticRoleClassifier(override_threshold=3)
    c2.frequency_table["menyebabkan"] = {RelationType.FUNCTIONAL: 1}
    assert c2.classify("X menyebabkan Y") == RelationType.CAUSAL


def test_negation_beats_frequency_table():
    """Negation must override the frequency table - 'X tidak menyebabkan Y'
    is DIFFERENTIAL even when "menyebabkan" has 100 CAUSAL votes.

    Rationale: negation is a *syntactic* signal that always inverts the
    relation semantics. Letting the table override it would re-introduce
    the original bug (CA1 saw "menyebabkan" and fired CAUSAL regardless
    of "tidak").
    """
    c = SemanticRoleClassifier()
    c.frequency_table["menyebabkan"] = {RelationType.CAUSAL: 100}
    assert c.classify("X tidak menyebabkan Y") == RelationType.DIFFERENTIAL


# ======================================================================
# SPO parser
# ======================================================================


def test_spo_extracts_predicate_menyebabkan(classifier: SemanticRoleClassifier):
    """SPO parse of 'X menyebabkan Y' has predicate='menyebabkan'."""
    spo = classifier.spo("lari menyebabkan ngos-ngosan")
    assert spo.predicate == "menyebabkan"
    assert spo.subject == "lari"
    assert spo.object == "ngos-ngosan"
    assert spo.negated is False


def test_spo_negation_flag_set_for_tidak_menyebabkan(classifier: SemanticRoleClassifier):
    """SPO parse of 'X tidak menyebabkan Y' has negated=True."""
    spo = classifier.spo("merokok tidak menyebabkan awet muda")
    assert spo.predicate == "menyebabkan"
    assert spo.negated is True


def test_spo_multiword_seed_wins_over_single_word(classifier: SemanticRoleClassifier):
    """'X bagian dari Y' picks the multi-word seed 'bagian dari'.

    This exercises the longest-first seed sorting.
    """
    spo = classifier.spo("paru bagian dari sistem pernapasan")
    assert spo.predicate == "bagian dari"


def test_spo_empty_string_returns_empty_spo(classifier: SemanticRoleClassifier):
    """Empty input -> SPO with all empty fields, no crash."""
    spo = classifier.spo("")
    assert spo.subject == ""
    assert spo.predicate == ""
    assert spo.object == ""


# ======================================================================
# Standalone-call smoke test (from the Definition of Done)
# ======================================================================


def test_standalone_call_returns_relation_type():
    """The DoD's one-liner must work:

        SemanticRoleClassifier().classify('lari menyebabkan ngos-ngosan')
        -> RelationType.CAUSAL
    """
    result = SemanticRoleClassifier().classify("lari menyebabkan ngos-ngosan")
    assert result == RelationType.CAUSAL
    assert hasattr(result, "value")  # enum-like, not a bare string
