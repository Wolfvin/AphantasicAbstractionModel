"""
RSVS Accuracy Integration Test

Tests that the system discriminates true from false statements.
Skips if Rust native extension is not compiled.

Run: pytest python/tests/test_accuracy.py -v
"""
import pytest

# Try to import the native extension; skip entire module if unavailable
try:
    from rsvs import Rsvs
    RSVS_AVAILABLE = True
except ImportError:
    RSVS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RSVS_AVAILABLE,
    reason="Rust native extension not compiled — run 'maturin develop' first"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_rsvs():
    """Train RSVS on full English corpus once per module."""
    from rsvs.corpus import DOMAINS
    import tempfile
    import os

    r = Rsvs(entity_promote_n=2, theta_assign=0.20, n_warm=20, eta=0.1)

    # Ingest all domains
    for domain_name, sentences in DOMAINS.items():
        for sentence in sentences:
            r.ingest(sentence)

    yield r


# ---------------------------------------------------------------------------
# Smoke: graph is non-trivial after training
# ---------------------------------------------------------------------------

def test_atoms_promoted(trained_rsvs):
    """After ingesting 9 domains, at least 40 atoms must be promoted."""
    atoms = trained_rsvs.atoms()
    assert len(atoms) >= 40, (
        f"Only {len(atoms)} atoms promoted — corpus may be too small or "
        f"entity_promote_n too high"
    )


def test_cross_domain_atoms_present(trained_rsvs):
    """Key cross-domain atoms must be in the graph."""
    atoms = set(trained_rsvs.atoms())
    required = ["energy", "material", "water"]
    missing = [a for a in required if a not in atoms]
    assert not missing, f"Cross-domain atoms missing from graph: {missing}"


# ---------------------------------------------------------------------------
# Structural similarity: related concepts score higher than unrelated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anchor,related,unrelated", [
    ("solid",    "hard",      "water"),
    ("doctor",   "patient",   "crop"),
    ("computer", "processor", "farmer"),
    ("empire",   "ruler",     "software"),
    ("energy",   "heat",      "crop"),
])
def test_structural_similarity_ranking(trained_rsvs, anchor, related, unrelated):
    """sim(anchor, related) >= sim(anchor, unrelated) for known pairs."""
    atoms = set(trained_rsvs.atoms())
    if not all(a in atoms for a in [anchor, related, unrelated]):
        pytest.skip(f"Atoms not in graph: {[a for a in [anchor, related, unrelated] if a not in atoms]}")

    sim_rel = trained_rsvs.similarity(anchor, related)
    sim_unr = trained_rsvs.similarity(anchor, unrelated)

    assert sim_rel is not None and sim_unr is not None
    assert sim_rel.jaccard >= sim_unr.jaccard, (
        f"sim({anchor},{related})={sim_rel.jaccard:.3f} < "
        f"sim({anchor},{unrelated})={sim_unr.jaccard:.3f}"
    )


# ---------------------------------------------------------------------------
# Discriminability: appraise(true) > appraise(false) for clear pairs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("true_stmt,false_stmt", [
    ("doctor treats patients",    "doctor plants crops"),
    ("farmer grows crops",        "farmer treats patients"),
    ("computer processes data",   "computer grows crops"),
    ("teacher explains lessons",  "teacher harvests crops"),
    ("empire controls territory", "empire treats patients"),
])
def test_discriminability(trained_rsvs, true_stmt, false_stmt):
    """appraise(true_statement).agree_pct > appraise(false_statement).agree_pct"""
    r_true  = trained_rsvs.appraise(true_stmt)
    r_false = trained_rsvs.appraise(false_stmt)

    gap = r_true.agree_pct - r_false.agree_pct
    assert gap > 0, (
        f"No discriminability: '{true_stmt}' ({r_true.agree_pct:.1f}%) "
        f"<= '{false_stmt}' ({r_false.agree_pct:.1f}%), gap={gap:+.1f}pp"
    )


# ---------------------------------------------------------------------------
# Confidence growth: cross-domain atoms > single-domain atoms
# ---------------------------------------------------------------------------

def test_confidence_growth(trained_rsvs):
    """Atoms appearing across domains have higher confidence than single-domain atoms."""
    cm = trained_rsvs.confidence_map()

    cross = ["energy", "material", "water"]
    single = ["basalt", "polymer"]

    cross_found  = [(a, cm[a]) for a in cross  if a in cm]
    single_found = [(a, cm[a]) for a in single if a in cm]

    if not cross_found:
        pytest.skip("Cross-domain atoms not in graph — corpus too small")

    avg_cross  = sum(c for _, c in cross_found)  / len(cross_found)
    avg_single = sum(c for _, c in single_found) / len(single_found) if single_found else 0.0

    assert avg_cross > avg_single, (
        f"Cross-domain confidence ({avg_cross:.3f}) not higher than "
        f"single-domain ({avg_single:.3f})"
    )
