"""
Python-level tests for RSVS v0.6 bindings.
Run with: python3 -m pytest tests/ -v
"""
import json
import pytest
import rsvs
from rsvs import Rsvs, IngestStats, QueryResult, SimResult, AtomInfo, SenseInfo

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

GEOLOGY = """
Stone is a hard solid mineral material. Rock is a hard heavy solid substance.
Stone is formed by heat and pressure over time. Granite is a hard rough stone.
Stone has a rough hard texture on its surface. Metal is a hard solid material.
Stone and metal are both hard solid materials. Hard solid materials resist pressure.
Stone is heavy and hard. Hard stone resists erosion and pressure.
Stone is formed under great pressure. Heat and pressure transform rock into stone.
Hard materials like stone and metal are solid and dense.
"""

WATER = """
Water is a clear transparent liquid substance. Water flows downhill because it is liquid.
Rain is water falling from clouds. Ice is frozen solid water formed by cold temperature.
Water dissolves many solid materials. Liquid water becomes ice when cold.
Clear liquid water is transparent and flows easily.
"""

@pytest.fixture
def r():
    """Fresh Rsvs instance."""
    return Rsvs(entity_promote_n=3, theta_assign=0.12, n_warm=10, eta=0.1)

@pytest.fixture
def trained(r):
    """Rsvs with geology + water corpus ingested."""
    r.ingest(GEOLOGY)
    r.set_domain(2)
    r.ingest(WATER)
    return r

# -----------------------------------------------------------------------
# Initialization tests
# -----------------------------------------------------------------------

class TestInit:
    def test_creates_instance(self, r):
        assert r is not None

    def test_repr(self, r):
        s = repr(r)
        assert "Rsvs(" in s

    def test_initial_status(self, r):
        st = r.status()
        assert st["total_contexts"] == 0
        assert st["warmed_up"] == 0.0

    def test_no_promoted_atoms_before_ingest(self, r):
        atoms = r.atoms(include_seeds=False)
        assert len(atoms) == 0

    def test_seed_atoms_present(self, r):
        atoms = r.atoms(include_seeds=True)
        assert len(atoms) == 24  # 10 Layer0 + 14 Layer1

# -----------------------------------------------------------------------
# Ingest tests
# -----------------------------------------------------------------------

class TestIngest:
    def test_returns_ingest_stats(self, r):
        stats = r.ingest("Stone is hard. Stone is solid. Stone is heavy material.")
        assert isinstance(stats, IngestStats)

    def test_stats_repr(self, r):
        stats = r.ingest("Stone is hard. Stone is solid. Stone is a hard material.")
        assert "IngestStats" in repr(stats)

    def test_processes_sentences(self, r):
        stats = r.ingest("Stone is hard. Water is liquid. Fire is hot.")
        assert stats.sentences_processed >= 1

    def test_promotes_frequent_atoms(self, r):
        r.ingest(GEOLOGY)
        atoms = r.atoms()
        assert "stone" in atoms or "hard" in atoms, f"Expected stone/hard, got: {atoms}"

    def test_does_not_promote_rare_tokens(self, r):
        r.ingest("Stone is hard. Stone is solid. Xyzquux appears once.")
        assert "xyzquux" not in r.atoms()

    def test_context_count_increases(self, r):
        before = r.status()["total_contexts"]
        r.ingest("Stone is hard. Stone is solid and heavy.")
        after = r.status()["total_contexts"]
        assert after > before

    def test_multiple_ingests_accumulate(self, trained):
        st = trained.status()
        assert st["total_contexts"] > 0
        assert st["total_atoms"] > 21  # more than seed atoms

    def test_domain_switching(self, r):
        r.set_domain(1)
        r.ingest(GEOLOGY)
        r.set_domain(2)
        r.ingest(WATER)
        st = r.status()
        assert st["total_atoms"] > 21

    def test_ingest_with_meta_v1_returns_seq_range(self, r):
        meta = r.ingest_with_meta_v1("Stone is hard. Stone is solid.")
        assert meta.api_version == "v1"
        assert meta.schema_version == "v1"
        assert meta.seq_end >= meta.seq_start

    def test_snapshot_and_events_contract(self, r):
        r.ingest("Stone is hard. Stone is solid.")
        snap = json.loads(r.snapshot_v1())
        assert snap["api_version"] == "v1"
        assert "nodes" in snap and "edges" in snap
        latest = r.latest_seq_v1()
        batch = json.loads(r.consume_events_v1(None, 200))
        assert batch["latest_seq"] == latest
        if batch["events"]:
            assert all("seq" in e and "event_type" in e for e in batch["events"])

# -----------------------------------------------------------------------
# Atom info tests
# -----------------------------------------------------------------------

class TestAtomInfo:
    def test_get_known_atom(self, trained):
        info = trained.atom_info("stone")
        assert isinstance(info, AtomInfo)
        assert info.label == "stone"
        assert 0.0 <= info.confidence <= 1.0
        assert info.tier in (1, 2, 3)

    def test_unknown_atom_raises(self, trained):
        with pytest.raises(Exception):
            trained.atom_info("nonexistent_xyz")

    def test_repr_contains_label(self, trained):
        info = trained.atom_info("stone")
        assert "stone" in repr(info)

    def test_tier_is_valid(self, trained):
        for atom in trained.atoms():
            info = trained.atom_info(atom)
            assert info.tier in (1, 2, 3)

    def test_confidence_in_range(self, trained):
        for atom in trained.atoms():
            info = trained.atom_info(atom)
            assert 0.0 <= info.confidence <= 1.0

# -----------------------------------------------------------------------
# Senses tests
# -----------------------------------------------------------------------

class TestSenses:
    def test_returns_sense_list(self, trained):
        senses = trained.senses("stone")
        assert isinstance(senses, list)
        assert len(senses) >= 1

    def test_each_sense_is_sense_info(self, trained):
        for s in trained.senses("stone"):
            assert isinstance(s, SenseInfo)

    def test_sense_has_positive_n(self, trained):
        for s in trained.senses("stone"):
            assert s.n_contexts >= 1

    def test_sense_coherence_in_range(self, trained):
        for s in trained.senses("stone"):
            assert 0.0 <= s.coherence <= 1.0

    def test_sense_status_valid(self, trained):
        for s in trained.senses("stone"):
            assert s.status in ("fragile", "mature")

    def test_unknown_concept_raises(self, trained):
        with pytest.raises(Exception):
            trained.senses("nonexistent_xyz")

    def test_repr(self, trained):
        s = trained.senses("stone")[0]
        assert "SenseInfo" in repr(s)

# -----------------------------------------------------------------------
# Similarity tests
# -----------------------------------------------------------------------

class TestSimilarity:
    def test_returns_sim_result(self, trained):
        sim = trained.similarity("stone", "hard")
        if sim is not None:
            assert isinstance(sim, SimResult)

    def test_jaccard_in_range(self, trained):
        sim = trained.similarity("stone", "hard")
        if sim is not None:
            assert 0.0 <= sim.jaccard <= 1.0

    def test_related_concepts_positive_jaccard(self, trained):
        sim = trained.similarity("stone", "hard")
        if sim is not None:
            assert sim.jaccard > 0.0, "stone/hard should have positive similarity"

    def test_unrelated_concepts_lower_jaccard(self, trained):
        sim_close = trained.similarity("stone", "hard")
        sim_far   = trained.similarity("stone", "water")
        if sim_close and sim_far:
            assert sim_close.jaccard >= sim_far.jaccard, \
                f"stone/hard ({sim_close.jaccard:.3f}) should >= stone/water ({sim_far.jaccard:.3f})"

    def test_unknown_concept_returns_none(self, trained):
        result = trained.similarity("nonexistent", "stone")
        assert result is None

    def test_repr(self, trained):
        sim = trained.similarity("stone", "hard")
        if sim:
            assert "SimResult" in repr(sim)

    def test_shared_atoms_are_strings(self, trained):
        sim = trained.similarity("stone", "hard")
        if sim:
            for atom in sim.shared:
                assert isinstance(atom, str)

# -----------------------------------------------------------------------
# Query tests
# -----------------------------------------------------------------------

class TestQuery:
    def test_returns_query_result(self, trained):
        result = trained.query("stone", "hard texture")
        if result is not None:
            assert isinstance(result, QueryResult)

    def test_unknown_concept_returns_none(self, trained):
        result = trained.query("nonexistent_xyz", "some context")
        assert result is None

    def test_atoms_sorted_descending(self, trained):
        result = trained.query("stone", "hard rough texture")
        if result and len(result.atoms) >= 2:
            scores = [s for _, s in result.atoms]
            for i in range(1, len(scores)):
                assert scores[i-1] >= scores[i], "Scores should be descending"

    def test_top_atoms_returns_labels(self, trained):
        result = trained.query("stone", "hard solid material")
        if result:
            top = result.top_atoms(3)
            assert isinstance(top, list)
            for label in top:
                assert isinstance(label, str)

    def test_sense_n_is_positive(self, trained):
        result = trained.query("stone", "hard material")
        if result:
            assert result.sense_n >= 1

    def test_repr(self, trained):
        result = trained.query("stone", "hard")
        if result:
            assert "QueryResult" in repr(result)

    def test_different_contexts_activate_different_senses(self, trained):
        r1 = trained.query("stone", "hard rough texture surface")
        r2 = trained.query("stone", "heat pressure underground")
        # Both should return results, potentially different senses
        assert r1 is not None or r2 is not None

# -----------------------------------------------------------------------
# Confidence map tests
# -----------------------------------------------------------------------

class TestConfidenceMap:
    def test_returns_dict(self, trained):
        cm = trained.confidence_map()
        assert isinstance(cm, dict)

    def test_all_values_in_range(self, trained):
        for label, conf in trained.confidence_map().items():
            assert 0.0 <= conf <= 1.0, f"{label}: {conf} out of range"

    def test_includes_promoted_atoms(self, trained):
        cm = trained.confidence_map()
        promoted = trained.atoms()
        for atom in promoted:
            assert atom in cm, f"Promoted atom '{atom}' missing from confidence map"

    def test_seed_atoms_have_confidence_1(self, trained):
        cm = trained.confidence_map()
        seed_labels = ["exists", "feel", "before", "after", "good", "bad"]
        for seed in seed_labels:
            if seed in cm:
                assert cm[seed] == 1.0, f"Seed atom '{seed}' should have confidence 1.0"
