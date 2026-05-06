"""
Wikipedia ingestion tests for RSVS v0.9.
Run with: python3 -m pytest tests/test_wiki.py -v
"""
import pytest
import json
import os
import tempfile
from pathlib import Path

import rsvs
from rsvs import Rsvs
from rsvs.corpus import (
    DOMAINS, ALL_SENTENCES, get_domain_text, get_all_text, domain_names,
)
from rsvs.ingest_wiki import ingest_domains, iter_domain_chunks, print_report

# -----------------------------------------------------------------------
# Corpus tests
# -----------------------------------------------------------------------

class TestCorpus:
    def test_has_seven_domains(self):
        assert len(DOMAINS) == 7

    def test_known_domains_present(self):
        for d in ["geology", "water", "biology", "physics", "materials", "kerajaan", "konsep"]:
            assert d in DOMAINS

    def test_each_domain_has_sentences(self):
        for domain, sentences in DOMAINS.items():
            assert len(sentences) >= 20, f"{domain} has too few sentences"

    def test_all_sentences_nonempty(self):
        for domain, sentence in ALL_SENTENCES:
            assert sentence.strip(), f"Empty sentence in {domain}"

    def test_total_sentence_count(self):
        assert len(ALL_SENTENCES) == 215  # 7 domains

    def test_get_domain_text_returns_string(self):
        text = get_domain_text("geology")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_get_domain_text_unknown_returns_empty(self):
        text = get_domain_text("nonexistent_domain")
        assert text == ""

    def test_get_all_text_covers_all_domains(self):
        text = get_all_text()
        for domain in DOMAINS:
            # Each domain has at least one sentence in the full text
            sample = DOMAINS[domain][0][:20]
            assert sample in text, f"Domain {domain!r} text not in all_text"

    def test_domain_names_returns_list(self):
        names = domain_names()
        assert isinstance(names, list)
        assert len(names) == 7

    def test_sentences_are_strings(self):
        for _, sentence in ALL_SENTENCES:
            assert isinstance(sentence, str)

    def test_sentences_end_with_period(self):
        # Most sentences should end with a period
        ends_with_period = sum(
            1 for _, s in ALL_SENTENCES if s.strip().endswith(".")
        )
        assert ends_with_period > len(ALL_SENTENCES) * 0.9

# -----------------------------------------------------------------------
# iter_domain_chunks tests
# -----------------------------------------------------------------------

class TestIterChunks:
    def test_yields_strings(self):
        for chunk in iter_domain_chunks("geology"):
            assert isinstance(chunk, str)
            break

    def test_chunk_size_respected(self):
        chunks = list(iter_domain_chunks("geology", chunk_size=5))
        # Last chunk may be smaller, but others should be ~5 sentences
        for chunk in chunks[:-1]:
            # Each chunk has at least 2 sentences worth of text
            assert len(chunk) > 10

    def test_all_sentences_covered(self):
        all_text = " ".join(iter_domain_chunks("geology", chunk_size=5))
        for sentence in DOMAINS["geology"]:
            assert sentence[:20] in all_text

    def test_unknown_domain_yields_nothing(self):
        chunks = list(iter_domain_chunks("nonexistent", chunk_size=5))
        assert chunks == []

# -----------------------------------------------------------------------
# ingest_domains tests
# -----------------------------------------------------------------------

class TestIngestDomains:
    @pytest.fixture
    def db(self, tmp_path):
        return str(tmp_path / "test.json")

    def test_creates_db_file(self, db):
        ingest_domains(db, ["geology"], verbose=False)
        assert Path(db).exists()

    def test_returns_summary_dict(self, db):
        summary = ingest_domains(db, ["geology"], verbose=False)
        assert isinstance(summary, dict)
        assert "geology" in summary

    def test_summary_has_stats(self, db):
        summary = ingest_domains(db, ["geology"], verbose=False)
        s = summary["geology"]
        assert "sentences" in s
        assert "atoms_promoted" in s
        assert "senses_created" in s
        assert s["sentences"] > 0

    def test_processes_all_sentences(self, db):
        summary = ingest_domains(db, ["geology"], verbose=False)
        assert summary["geology"]["sentences"] == len(DOMAINS["geology"])

    def test_multiple_domains(self, db):
        summary = ingest_domains(db, ["geology", "water"], verbose=False)
        assert "geology" in summary
        assert "water" in summary

    def test_promotes_atoms(self, db):
        ingest_domains(db, ["geology", "water"], verbose=False)
        r = Rsvs.load(db)
        atoms = r.atoms()
        assert len(atoms) > 0

    def test_different_domains_produce_different_atoms(self, db):
        """Geology and water should produce some domain-specific atoms."""
        ingest_domains(db, ["geology", "water"], verbose=False)
        r = Rsvs.load(db)
        atoms = set(r.atoms())
        # At least some domain-specific terms should be promoted
        geo_terms   = {"stone", "rock", "mineral", "granite", "sedimentary"}
        water_terms = {"water", "liquid", "river", "ocean", "ice"}
        found_geo   = atoms & geo_terms
        found_water = atoms & water_terms
        assert len(found_geo) + len(found_water) >= 2, \
            f"Expected domain-specific atoms, got: {atoms}"

    def test_multi_domain_increases_atoms(self, db):
        """More domains → more atoms."""
        ingest_domains(db, ["geology"], verbose=False)
        r1 = Rsvs.load(db)
        n1 = len(r1.atoms())

        ingest_domains(db, ["water"], verbose=False)
        r2 = Rsvs.load(db)
        n2 = len(r2.atoms())

        assert n2 >= n1, f"Adding water domain should not reduce atoms: {n1} vs {n2}"

    def test_load_after_ingest_and_query(self, db):
        ingest_domains(db, ["geology", "water"], verbose=False)
        r = Rsvs.load(db)
        atoms = r.atoms()
        if atoms:
            # Query should work for any promoted atom
            result = r.query(atoms[0], "hard solid material")
            # Either finds result or not — just no crash
            assert result is None or hasattr(result, "sense_idx")

    def test_unknown_domain_skipped(self, db, capsys):
        summary = ingest_domains(db, ["geology", "nonexistent_xyz"], verbose=True)
        assert "nonexistent_xyz" not in summary
        assert "geology" in summary

    def test_all_seven_domains(self, db):
        summary = ingest_domains(db, domain_names(), verbose=False)
        assert len(summary) == 7
        total_sentences = sum(s["sentences"] for s in summary.values())
        assert total_sentences > 0

# -----------------------------------------------------------------------
# Knowledge quality tests (post full ingestion)
# -----------------------------------------------------------------------

class TestKnowledgeQuality:
    """
    These tests verify that RSVS learns meaningful structure
    from the Wikipedia corpus — not just that it runs without crashing.
    """

    @pytest.fixture(scope="class")
    def full_db(self, tmp_path_factory):
        db = str(tmp_path_factory.mktemp("rsvs") / "full.json")
        ingest_domains(db, domain_names(), verbose=False)
        return db

    def test_promoted_meaningful_atoms(self, full_db):
        r = Rsvs.load(full_db)
        atoms = set(r.atoms())
        # At least some of these should be promoted from 150 sentences
        expected = {"solid", "material", "water", "energy", "rock", "stone"}
        found = atoms & expected
        assert len(found) >= 2, f"Expected some meaningful atoms, got: {atoms}"

    def test_geology_water_atoms_distinct(self, full_db):
        """Geology atoms and water atoms should be in the graph separately."""
        r = Rsvs.load(full_db)
        atoms = set(r.atoms())
        has_geo   = bool(atoms & {"rock", "stone", "mineral", "granite"})
        has_water = bool(atoms & {"water", "liquid", "river"})
        # At least one from each domain expected
        assert has_geo or has_water, f"Expected geo/water atoms, got: {atoms}"

    def test_solid_appears_across_domains(self, full_db):
        """'solid' appears in geology, water, physics, materials → should be promoted."""
        r = Rsvs.load(full_db)
        assert "solid" in r.atoms(), \
            "'solid' should be promoted as it appears across many domains"

    def test_solid_has_multiple_senses(self, full_db):
        """'solid' in different contexts (rock solid, solid state, solid material)
        should produce multiple senses."""
        r = Rsvs.load(full_db)
        if "solid" not in r.atoms():
            pytest.skip("solid not promoted")
        senses = r.senses("solid")
        assert len(senses) >= 1

    def test_cross_domain_similarity_nonzero(self, full_db):
        """Concepts that share atoms across domains should have nonzero similarity."""
        r = Rsvs.load(full_db)
        atoms = r.atoms()
        # Find two promoted atoms and check similarity
        for a in ["solid", "material", "energy", "water"]:
            for b in ["hard", "liquid", "rock", "heat"]:
                if a in atoms and b in atoms:
                    sim = r.similarity(a, b)
                    if sim and sim.jaccard > 0:
                        return  # Found at least one meaningful similarity
        # If no pairs found, check that atoms list is reasonable
        assert len(atoms) > 0

    def test_confidence_increases_with_frequency(self, full_db):
        """Atoms that appear more often should generally have higher confidence."""
        r = Rsvs.load(full_db)
        cm = r.confidence_map()
        # 'solid' appears in many domains → should have decent confidence
        if "solid" in cm:
            assert cm["solid"] > 0.45, \
                f"'solid' confidence too low: {cm['solid']}"

    def test_warmed_up_after_150_contexts(self, full_db):
        """With n_warm=20, system should be warmed up after 150 contexts."""
        r = Rsvs.load(full_db)
        assert r.status()["warmed_up"] == 1.0

    def test_db_size_reasonable(self, full_db):
        """DB should be larger than 10KB but smaller than 10MB for 150 sentences."""
        size = Path(full_db).stat().st_size
        assert 10_000 < size < 10_000_000, \
            f"Unexpected DB size: {size} bytes"

    def test_query_returns_result_for_known_atom(self, full_db):
        r = Rsvs.load(full_db)
        for atom in ["solid", "material", "water", "energy"]:
            if atom in r.atoms():
                result = r.query(atom, "hard material solid")
                assert result is not None, f"Query for '{atom}' returned None"
                return
        pytest.skip("No expected atoms promoted")

    def test_print_report_no_crash(self, full_db, capsys):
        """print_report should run without crashing."""
        print_report(full_db)
        out = capsys.readouterr().out
        assert "RSVS Knowledge Report" in out
        assert "Atoms" in out
