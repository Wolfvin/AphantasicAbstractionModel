"""
Tests for TrisynapticCircuit (encoding), SystemsConsolidation (HPC->NC
transfer), and PapezCircuit (retrieval).

Covers the four Definition-of-Done requirements from the task brief:
    1. TrisynapticCircuit.encode() returns an Episome with the right fields.
    2. SystemsConsolidation.consolidate() returns a Semesome.
    3. Consolidation increases the episome's confidence (+0.05).
    4. PapezCircuit.retrieve() returns a list of Episomes sorted by
       confidence.
    5. End-to-end: encode -> consolidate -> retrieve.

Plus targeted unit tests for each hippocampal substructure and for
edge-type inference. Total: 16 tests (> 10 required).

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_consolidation.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
# Tests are invoked as: python -m pytest AGNN/tests/test_consolidation.py -v
# So we add the AGNN/ directory (parent of tests/) to sys.path.
_AGNN_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNN_ROOT))

# Also ensure self-ai/src is importable for AGNNGraph in engram_complex.
_SELF_AI_SRC = _AGNN_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


# ----------------------------------------------------------------------
# Imports under test
# ----------------------------------------------------------------------

from engrams.episodic_engram import Episome  # noqa: E402
from engrams.semantic_engram import Semesome  # noqa: E402
from engrams.engram_complex import EngramComplex  # noqa: E402
from hippocampus.dentate_gyrus import DentateGyrus  # noqa: E402
from hippocampus.entorhinal_cortex import EntorhinalCortex  # noqa: E402
from hippocampus.ca1 import CA1  # noqa: E402
from hippocampus.ca3 import CA3  # noqa: E402
from hippocampus.subiculum import Subiculum  # noqa: E402
from circuits.trisynaptic_circuit import TrisynapticCircuit  # noqa: E402
from circuits.papez_circuit import PapezCircuit  # noqa: E402
from plasticity.systems_consolidation import (  # noqa: E402
    SystemsConsolidation,
    CONSOLIDATION_CONFIDENCE_DELTA,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_circuit() -> TrisynapticCircuit:
    """Build a TrisynapticCircuit with a fresh EngramComplex.

    Skips the test if the AGNNGraph dependency (self-ai/src/agnn) is
    unavailable, since the EngramComplex constructor raises ImportError
    in that case.
    """
    try:
        ec = EngramComplex()
    except ImportError:
        pytest.skip("AGNNGraph (self-ai/src/agnn) not available")
    return TrisynapticCircuit(engram_complex=ec)


@pytest.fixture
def circuit() -> TrisynapticCircuit:
    """Fresh TrisynapticCircuit + EngramComplex per test."""
    return _make_circuit()


@pytest.fixture
def consolidator() -> SystemsConsolidation:
    """Fresh SystemsConsolidation per test."""
    return SystemsConsolidation()


@pytest.fixture
def retriever() -> PapezCircuit:
    """Fresh PapezCircuit per test."""
    return PapezCircuit()


# ======================================================================
# Requirement 1: TrisynapticCircuit.encode() returns an Episome
# ======================================================================


def test_encode_returns_episome(circuit: TrisynapticCircuit):
    """encode() must return an Episome instance (not a dict or other type)."""
    result = circuit.encode("Socrates is a human")
    assert isinstance(result, Episome), (
        f"encode() must return Episome, got {type(result).__name__}"
    )


def test_encode_episome_has_correct_fields(circuit: TrisynapticCircuit):
    """The returned Episome must have all required fields with correct types.

    Required by Definition-of-Done:
        - id is int (and unique)
        - text is str (the encoded fact)
        - confidence == 0.6 (labile-phase baseline)
        - edge_type is one of CA1.EDGE_TYPES
        - type == "episodic"
    """
    epi = circuit.encode("Socrates is a human")
    assert isinstance(epi.id, int), f"episome.id must be int, got {type(epi.id).__name__}"
    assert epi.id > 0, "episome.id must be positive (DG allocates 1-indexed IDs)"
    assert isinstance(epi.text, str) and epi.text, "episome.text must be a non-empty str"
    assert epi.confidence == pytest.approx(0.6), (
        f"freshly encoded episome must have confidence=0.6, got {epi.confidence}"
    )
    assert epi.edge_type in CA1.EDGE_TYPES, (
        f"episome.edge_type must be in {CA1.EDGE_TYPES}, got {epi.edge_type!r}"
    )
    assert epi.type == "episodic", (
        f"episome.type must be 'episodic', got {epi.type!r}"
    )


def test_encode_assigns_unique_ids(circuit: TrisynapticCircuit):
    """Encoding two stimuli must produce two distinct episome IDs (DG pattern separation)."""
    e1 = circuit.encode("Socrates is a human")
    e2 = circuit.encode("Plato is a philosopher")
    assert e1.id != e2.id, "DG must allocate unique IDs (pattern separation)"


def test_encode_stores_correction_text_when_provided(circuit: TrisynapticCircuit):
    """When a correction is supplied, the Episome.text should be the correction.

    Rationale: in AGNNCore.learn(question, wrong, correction), the new
    knowledge being encoded is the correction, not the wrong answer. So
    Episome.text should reflect what was learned, not what was wrong.
    """
    epi = circuit.encode(stimulus="What is photosynthesis?", correction="Photosynthesis converts light into chemical energy")
    assert "photosynthesis" in epi.text.lower()
    assert "converts light" in epi.text.lower(), (
        "Episome.text should carry the correction (the new knowledge)"
    )


# ======================================================================
# Requirement 1 (cont'd): CA1 edge-type inference
# ======================================================================


def test_ca1_infers_categorical_for_default_text():
    """CA1.integrate_context defaults to CATEGORICAL when no cues fire."""
    ca1 = CA1()
    # Text with no relation cues - just a noun phrase.
    assert ca1.integrate_context("red apple") == "CATEGORICAL"


def test_ca1_infers_causal_for_causal_cues():
    """CA1 detects 'causes' and infers CAUSAL."""
    ca1 = CA1()
    assert ca1.integrate_context("smoking causes lung damage") == "CAUSAL"


def test_ca1_infers_differential_for_negation_cues():
    """CA1 detects 'not' / 'unlike' / 'inhibits' and infers DIFFERENTIAL."""
    ca1 = CA1()
    assert ca1.integrate_context("exercise reduces body fat") == "DIFFERENTIAL"


def test_ca1_infers_functional_for_functional_cues():
    """CA1 detects 'requires' / 'enables' and infers FUNCTIONAL."""
    ca1 = CA1()
    assert ca1.integrate_context("heart requires blood to pump") == "FUNCTIONAL"


# ======================================================================
# CA3 autoassociation
# ======================================================================


def test_ca3_finds_neighbors_by_keyword_overlap(circuit: TrisynapticCircuit):
    """Two episomes sharing a keyword must autoassociate.

    Encode "Socrates is a human" then encode "Every human is mortal".
    The second episome must list the first as a neighbor (overlap on
    "human").
    """
    e1 = circuit.encode("Socrates is a human")
    e2 = circuit.encode("Every human is mortal")
    # The CA3 bindings list records (episome_id, neighbor_ids) tuples.
    last_binding = circuit.ca3.bindings[-1]
    assert last_binding[0] == e2.id, "binding should be recorded for e2"
    assert e1.id in last_binding[1], (
        f"e2 should list e1 as a neighbor (keyword 'human' overlaps), "
        f"got neighbors={last_binding[1]}"
    )


# ======================================================================
# Requirement 2: SystemsConsolidation.consolidate() returns a Semesome
# ======================================================================


def test_consolidate_returns_semesome_when_neighbors_exist(
    circuit: TrisynapticCircuit,
    consolidator: SystemsConsolidation,
):
    """consolidate() must return a Semesome when the episome has neighbors."""
    e1 = circuit.encode("Socrates is a human")
    e2 = circuit.encode("Every human is mortal")  # shares "human" with e1

    result = consolidator.consolidate(e2, circuit.engram_complex)
    assert isinstance(result, Semesome), (
        f"consolidate() must return Semesome when neighbors exist, "
        f"got {type(result).__name__}"
    )
    assert result.source in {e1.text, e2.text}
    assert result.target in {e1.text, e2.text}
    assert result.source != result.target
    assert result.type in CA1.EDGE_TYPES
    assert -1.0 <= result.weight <= 1.0


def test_consolidate_returns_none_when_no_neighbors(
    circuit: TrisynapticCircuit,
    consolidator: SystemsConsolidation,
):
    """consolidate() returns None when the episome has no neighbors.

    A single isolated episome has nothing to consolidate into a
    Semesome - there's no relation to abstract. The function should
    return None so callers know no semantic memory was produced.
    """
    e = circuit.encode("A lonely unrelated fact about xyzzy")
    result = consolidator.consolidate(e, circuit.engram_complex)
    assert result is None, (
        "consolidate() should return None when the episome has no neighbors"
    )


# ======================================================================
# Requirement 3: Consolidation increases episome confidence by 0.05
# ======================================================================


def test_consolidation_increases_confidence(
    circuit: TrisynapticCircuit,
    consolidator: SystemsConsolidation,
):
    """consolidate() must add CONSOLIDATION_CONFIDENCE_DELTA (0.05) to episome.confidence."""
    e1 = circuit.encode("Socrates is a human")
    e2 = circuit.encode("Every human is mortal")
    before = e2.confidence
    consolidator.consolidate(e2, circuit.engram_complex)
    after = e2.confidence
    assert after == pytest.approx(before + CONSOLIDATION_CONFIDENCE_DELTA), (
        f"consolidation must add {CONSOLIDATION_CONFIDENCE_DELTA} to confidence, "
        f"got {before} -> {after}"
    )


def test_consolidation_strengthens_even_without_neighbors(
    circuit: TrisynapticCircuit,
    consolidator: SystemsConsolidation,
):
    """The +0.05 confidence boost applies even when no Semesome is produced.

    Per the task spec: "Update episome.confidence += 0.05 (consolidation
    strengthens memory)". The strengthening is unconditional - only the
    Semesome construction is conditional on having neighbors.
    """
    e = circuit.encode("A lonely unrelated fact about xyzzy")
    before = e.confidence
    consolidator.consolidate(e, circuit.engram_complex)
    after = e.confidence
    assert after == pytest.approx(before + CONSOLIDATION_CONFIDENCE_DELTA)


def test_consolidation_caps_confidence_at_one(
    circuit: TrisynapticCircuit,
    consolidator: SystemsConsolidation,
):
    """Confidence must never exceed 1.0 even after many consolidation passes."""
    e1 = circuit.encode("Socrates is a human")
    e2 = circuit.encode("Every human is mortal")
    # Run consolidation 20 times - each adds 0.05, so we'd hit 1.6 uncapped.
    for _ in range(20):
        consolidator.consolidate(e2, circuit.engram_complex)
    assert e2.confidence <= 1.0, (
        f"confidence must be capped at 1.0, got {e2.confidence}"
    )


# ======================================================================
# Requirement 4: PapezCircuit.retrieve() returns Episomes sorted by confidence
# ======================================================================


def test_retrieve_returns_list_of_episomes(
    circuit: TrisynapticCircuit,
    retriever: PapezCircuit,
):
    """retrieve() must return a list of Episome instances."""
    circuit.encode("Socrates is a human")
    circuit.encode("Every human is mortal")
    results = retriever.retrieve("human", circuit.engram_complex, top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, Episome) for r in results), (
        "every retrieved item must be an Episome"
    )


def test_retrieve_sorted_by_confidence_descending(
    circuit: TrisynapticCircuit,
    retriever: PapezCircuit,
    consolidator: SystemsConsolidation,
):
    """retrieve() must return results sorted by descending confidence.

    Setup: encode two episomes sharing the keyword "human", then
    consolidate one of them so its confidence goes 0.6 -> 0.65. The
    consolidated one must rank first.
    """
    e1 = circuit.encode("Socrates is a human")
    e2 = circuit.encode("Every human is mortal")
    consolidator.consolidate(e2, circuit.engram_complex)  # e2 -> 0.65

    results = retriever.retrieve("human", circuit.engram_complex, top_k=3)
    assert len(results) >= 2
    # First result must have higher (or equal) confidence than the second.
    confidences = [r.confidence for r in results]
    assert confidences == sorted(confidences, reverse=True), (
        f"results must be sorted by descending confidence, got {confidences}"
    )
    # The consolidated episome should rank first.
    assert results[0].id == e2.id, (
        f"consolidated episome (id={e2.id}, conf=0.65) should rank first, "
        f"got id={results[0].id}"
    )


def test_retrieve_respects_top_k(
    circuit: TrisynapticCircuit,
    retriever: PapezCircuit,
):
    """retrieve() must never return more than top_k results."""
    for i in range(5):
        circuit.encode(f"human fact number {i}")
    results = retriever.retrieve("human", circuit.engram_complex, top_k=2)
    assert len(results) <= 2, (
        f"top_k=2 must bound the result count, got {len(results)}"
    )


def test_retrieve_returns_empty_when_no_keyword_match(
    circuit: TrisynapticCircuit,
    retriever: PapezCircuit,
):
    """retrieve() returns an empty list when nothing matches the query keywords."""
    circuit.encode("Socrates is a human")
    results = retriever.retrieve("quantum", circuit.engram_complex, top_k=3)
    assert results == [], (
        "no keyword overlap -> no results"
    )


# ======================================================================
# Requirement 5: End-to-end encode -> consolidate -> retrieve
# ======================================================================


def test_end_to_end_encode_consolidate_retrieve(
    circuit: TrisynapticCircuit,
    consolidator: SystemsConsolidation,
    retriever: PapezCircuit,
):
    """End-to-end: encode two related facts, consolidate the second,
    then retrieve and verify the consolidated episome ranks first.
    """
    # 1. Encode two related facts (shared keyword "human").
    e1 = circuit.encode("Socrates is a human")
    e2 = circuit.encode("Every human is mortal")
    assert e1.id != e2.id
    assert e1.confidence == pytest.approx(0.6)
    assert e2.confidence == pytest.approx(0.6)

    # 2. Consolidate e2 - should produce a Semesome + strengthen e2.
    semesome = consolidator.consolidate(e2, circuit.engram_complex)
    assert isinstance(semesome, Semesome)
    assert semesome.source in {e1.text, e2.text}
    assert semesome.target in {e1.text, e2.text}
    assert e2.confidence == pytest.approx(0.65)

    # 3. Retrieve by the shared keyword - the consolidated episome must
    #    rank first (higher confidence).
    results = retriever.retrieve("human", circuit.engram_complex, top_k=3)
    assert len(results) >= 2
    assert results[0].id == e2.id, (
        "consolidated episome should rank first after retrieval"
    )
    assert results[0].confidence > results[1].confidence, (
        "consolidated episome should have strictly higher confidence"
    )


# ======================================================================
# Hippocampal substructure unit tests
# ======================================================================


def test_dentate_gyrus_allocates_incrementing_ids():
    """DG.separate() must return monotonically increasing IDs."""
    dg = DentateGyrus()
    ids = [dg.separate(f"stimulus {i}") for i in range(5)]
    assert ids == [1, 2, 3, 4, 5]


def test_entorhinal_cortex_normalizes_and_extracts_keywords():
    """EC.normalize_input returns text + tokens + keywords with stop-words removed."""
    ec = EntorhinalCortex()
    out = ec.normalize_input("  Socrates   IS a Human  ")
    assert out["text"] == "socrates is a human"
    assert "socrates" in out["keywords"]
    assert "human" in out["keywords"]
    # Stop-words "is" and "a" must not appear in keywords.
    assert "is" not in out["keywords"]
    assert "a" not in out["keywords"]


def test_ca3_autoassociation_returns_empty_for_first_episome():
    """The first registered episome has no neighbors (registry is empty)."""
    ca3 = CA3()
    ca3.register(1, "first fact", ["first", "fact"])
    assert ca3.autoassociate(1) == [], (
        "first episome should have no neighbors"
    )


def test_subiculum_relay_builds_episome_with_default_confidence():
    """Sub.relay_output must produce an Episome with confidence=0.6 by default."""
    sub = Subiculum()
    epi = sub.relay_output(episome_id=42, text="hello", edge_type="CATEGORICAL")
    assert isinstance(epi, Episome)
    assert epi.id == 42
    assert epi.text == "hello"
    assert epi.edge_type == "CATEGORICAL"
    assert epi.confidence == pytest.approx(0.6)
    # The subiculum must log the relay.
    assert epi in sub.output_log
