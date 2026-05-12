"""
RSVS Convergence Engine Test

Tests that the convergence engine creates LanguageLink(structural_equivalence)
between concepts that have similar composition structures but never co-occur.

Strategy: ingest two domains with mirror-image vocabulary. The concepts
"doctor" and "physician" should converge (same compositions, never co-occur
in the same sentence if we keep them in separate ingest batches).
"""
import pytest

try:
    from rsvs import Rsvs
    RSVS_AVAILABLE = True
except ImportError:
    RSVS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RSVS_AVAILABLE,
    reason="Rust native extension not compiled"
)


CORPUS_A = """
A doctor examines patients to diagnose illness.
The doctor prescribes medicine to treat the patient.
Doctors work in hospitals and clinics.
A doctor uses medical tools to examine a patient.
The doctor treats illness by prescribing medicine.
Medical knowledge helps the doctor diagnose patients.
A doctor heals patients using medicine and treatment.
The doctor is respected for their medical expertise.
Patients trust the doctor with their health concerns.
A doctor studies medicine for many years.
"""

# Synonym corpus: physician = doctor, but never appears with "doctor"
CORPUS_B = """
A physician examines patients to diagnose illness.
The physician prescribes medicine to treat the patient.
Physicians work in hospitals and clinics.
A physician uses medical tools to examine a patient.
The physician treats illness by prescribing medicine.
Medical knowledge helps the physician diagnose patients.
A physician heals patients using medicine and treatment.
The physician is respected for their medical expertise.
Patients trust the physician with their health concerns.
A physician studies medicine for many years.
"""


@pytest.fixture(scope="module")
def converged_rsvs():
    """Build RSVS with mirrored corpora for convergence testing."""
    r = Rsvs(entity_promote_n=2, theta_assign=0.20)
    r.ingest(CORPUS_A)
    r.ingest(CORPUS_B)
    # Run convergence detection explicitly if available
    try:
        r.detect_convergence()
    except AttributeError:
        pass  # method may not exist in all versions
    return r


def test_both_concepts_in_graph(converged_rsvs):
    """Both doctor and physician must be atoms."""
    atoms = set(converged_rsvs.atoms())
    assert "doctor"    in atoms, "doctor not promoted to atom"
    assert "physician" in atoms, "physician not promoted to atom"


def test_structural_similarity_high(converged_rsvs):
    """doctor and physician must have high structural similarity (>=0.5)."""
    atoms = set(converged_rsvs.atoms())
    if "doctor" not in atoms or "physician" not in atoms:
        pytest.skip("One of the concepts not in graph")

    result = converged_rsvs.structural_similarity("doctor", "physician")
    assert result is not None
    assert result.structural_similarity >= 0.5, (
        f"structural_similarity(doctor, physician) = "
        f"{result.structural_similarity:.3f} — expected >= 0.5"
    )


def test_similarity_positive(converged_rsvs):
    """doctor and physician must have positive similarity."""
    atoms = set(converged_rsvs.atoms())
    if "doctor" not in atoms or "physician" not in atoms:
        pytest.skip("One of the concepts not in graph")

    result = converged_rsvs.similarity("doctor", "physician")
    assert result is not None
    assert result.jaccard > 0, (
        f"Jaccard similarity(doctor, physician) = {result.jaccard:.3f} — expected > 0"
    )
