"""
CLI tests for RSVS v0.8.
Run with: python3 -m pytest tests/test_cli.py -v
"""
import pytest
import json
import os
import sys
import tempfile
from pathlib import Path

# Import CLI functions directly (avoids subprocess overhead)
import importlib
import rsvs.cli as cli_module
from rsvs.cli import (
    cmd_init, cmd_ingest, cmd_query, cmd_similarity,
    cmd_status, cmd_atoms, cmd_senses, cmd_info, cmd_ingest_corpus, cmd_eval, cmd_replay_events,
    build_parser, _load_rsvs, _save_rsvs,
)

# -----------------------------------------------------------------------
# Fixture helpers
# -----------------------------------------------------------------------

CORPUS = """
Stone is a hard solid mineral material. Rock is a hard heavy solid substance.
Stone is formed by heat and pressure over time. Granite is a hard rough stone.
Stone has a rough hard texture. Metal is a hard solid material.
Stone and metal are both hard solid materials. Hard solid materials resist pressure.
Stone is heavy and hard. Stone resists erosion and pressure.
"""

CORPUS_WATER = """
Water is a clear transparent liquid. Water flows because it is liquid.
Rain is water falling from clouds. Ice is frozen solid water.
Liquid water is transparent and flows easily.
"""

class MockArgs:
    """Minimal args namespace for testing commands directly."""
    def __init__(self, **kwargs):
        # Defaults
        self.json  = False
        self.seeds = False
        self.force = False
        self.promote_n = 3
        self.theta     = 0.12
        self.n_warm    = 8
        self.eta       = 0.1
        self.domain    = None
        self.top       = 6
        self.all       = False
        self.domains   = None
        self.chunk_size = 5
        self.after_seq = None
        self.limit = 500
        self.json_out = None
        self.baseline_json = None
        for k, v in kwargs.items():
            setattr(self, k, v)

@pytest.fixture
def db(tmp_path):
    """Path to a temp DB file."""
    return str(tmp_path / "test_rsvs.json")

@pytest.fixture
def initialized_db(db):
    """DB that has been init'd."""
    cmd_init(MockArgs(db=db))
    return db

@pytest.fixture
def trained_db(initialized_db):
    """DB with geology + water corpus ingested."""
    cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS))
    cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS_WATER, domain=2))
    return initialized_db

# -----------------------------------------------------------------------
# init tests
# -----------------------------------------------------------------------

class TestInit:
    def test_creates_db_file(self, db):
        cmd_init(MockArgs(db=db))
        assert Path(db).exists()

    def test_db_is_valid_json(self, db):
        cmd_init(MockArgs(db=db))
        with open(db) as f:
            data = json.load(f)
        assert "version" in data

    def test_init_twice_without_force_raises(self, initialized_db, capsys):
        with pytest.raises(SystemExit) as exc:
            cmd_init(MockArgs(db=initialized_db))
        assert exc.value.code != 0

    def test_init_twice_with_force_ok(self, initialized_db):
        # Should not raise
        cmd_init(MockArgs(db=initialized_db, force=True))
        assert Path(initialized_db).exists()

    def test_custom_promote_n(self, db):
        cmd_init(MockArgs(db=db, promote_n=5))
        r = _load_rsvs(db)
        assert r.status()["total_contexts"] == 0  # fresh

    def test_prints_confirmation(self, db, capsys):
        cmd_init(MockArgs(db=db))
        out = capsys.readouterr().out
        assert "Initialized" in out
        assert "seed atoms" in out

# -----------------------------------------------------------------------
# ingest tests
# -----------------------------------------------------------------------

class TestIngest:
    def test_ingest_text(self, initialized_db, capsys):
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS))
        out = capsys.readouterr().out
        assert "Ingested" in out

    def test_ingest_increases_contexts(self, initialized_db):
        r_before = _load_rsvs(initialized_db)
        ctx_before = r_before.status()["total_contexts"]
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS))
        r_after = _load_rsvs(initialized_db)
        assert r_after.status()["total_contexts"] > ctx_before

    def test_ingest_promotes_atoms(self, initialized_db):
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS))
        r = _load_rsvs(initialized_db)
        assert len(r.atoms()) > 0

    def test_ingest_saves_to_db(self, initialized_db):
        size_before = Path(initialized_db).stat().st_size
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS))
        size_after = Path(initialized_db).stat().st_size
        assert size_after > size_before

    def test_ingest_json_output(self, initialized_db, capsys):
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS, json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "sentences_processed" in data
        assert "atoms_promoted" in data

    def test_ingest_from_file(self, initialized_db, tmp_path):
        f = tmp_path / "corpus.txt"
        f.write_text(CORPUS)
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=str(f)))
        r = _load_rsvs(initialized_db)
        assert r.status()["total_contexts"] > 0

    def test_ingest_empty_text_exits(self, initialized_db):
        with pytest.raises(SystemExit) as exc:
            cmd_ingest(MockArgs(db=initialized_db, text_or_file="   "))
        assert exc.value.code != 0

    def test_ingest_with_domain(self, initialized_db):
        # Should not raise
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS, domain=2))
        r = _load_rsvs(initialized_db)
        assert r.status()["total_contexts"] > 0

# -----------------------------------------------------------------------
# query tests
# -----------------------------------------------------------------------

class TestQuery:
    def test_query_known_concept(self, trained_db, capsys):
        if "stone" not in _load_rsvs(trained_db).atoms():
            pytest.skip("stone not promoted in this run")
        cmd_query(MockArgs(db=trained_db, concept="stone", context="hard texture"))
        out = capsys.readouterr().out
        assert "stone" in out

    def test_query_unknown_concept_exits(self, trained_db):
        with pytest.raises(SystemExit) as exc:
            cmd_query(MockArgs(db=trained_db, concept="zzz_unknown", context="foo"))
        assert exc.value.code != 0

    def test_query_json_output(self, trained_db, capsys):
        if "stone" not in _load_rsvs(trained_db).atoms():
            pytest.skip("stone not promoted")
        cmd_query(MockArgs(db=trained_db, concept="stone", context="hard", json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "concept" in data
        assert "sense_idx" in data
        assert "atoms" in data

    def test_query_json_unknown_has_error(self, trained_db, capsys):
        with pytest.raises(SystemExit):
            cmd_query(MockArgs(db=trained_db, concept="zzz", context="foo", json=True))

    def test_query_top_limits_results(self, trained_db, capsys):
        if "stone" not in _load_rsvs(trained_db).atoms():
            pytest.skip("stone not promoted")
        cmd_query(MockArgs(db=trained_db, concept="stone", context="hard solid", json=True, top=2))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data["atoms"]) <= 2

# -----------------------------------------------------------------------
# similarity tests
# -----------------------------------------------------------------------

class TestSimilarity:
    def test_similarity_two_known(self, trained_db, capsys):
        atoms = _load_rsvs(trained_db).atoms()
        if len(atoms) < 2:
            pytest.skip("Need at least 2 atoms")
        a, b = sorted(atoms)[:2]
        cmd_similarity(MockArgs(db=trained_db, a=a, b=b))
        out = capsys.readouterr().out
        assert "jaccard" in out

    def test_similarity_json(self, trained_db, capsys):
        atoms = _load_rsvs(trained_db).atoms()
        if len(atoms) < 2:
            pytest.skip("Need at least 2 atoms")
        a, b = sorted(atoms)[:2]
        cmd_similarity(MockArgs(db=trained_db, a=a, b=b, json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "jaccard" in data
        assert 0.0 <= data["jaccard"] <= 1.0

    def test_similarity_unknown_exits(self, trained_db):
        with pytest.raises(SystemExit) as exc:
            cmd_similarity(MockArgs(db=trained_db, a="zzz_unknown", b="stone"))
        assert exc.value.code != 0

    def test_similarity_json_scores_in_range(self, trained_db, capsys):
        atoms = _load_rsvs(trained_db).atoms()
        if len(atoms) < 2:
            pytest.skip("Need at least 2 atoms")
        a, b = sorted(atoms)[:2]
        cmd_similarity(MockArgs(db=trained_db, a=a, b=b, json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert 0.0 <= data["jaccard"] <= 1.0

# -----------------------------------------------------------------------
# status tests
# -----------------------------------------------------------------------

class TestStatus:
    def test_status_shows_info(self, initialized_db, capsys):
        cmd_status(MockArgs(db=initialized_db))
        out = capsys.readouterr().out
        assert "total nodes" in out
        assert "warmed up" in out

    def test_status_json(self, initialized_db, capsys):
        cmd_status(MockArgs(db=initialized_db, json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "total_nodes" in data
        assert "warmed_up" in data
        assert "theta_assign" in data

    def test_status_nonexistent_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            cmd_status(MockArgs(db=str(tmp_path / "nonexistent.json")))
        assert exc.value.code != 0

    def test_status_includes_db_size(self, initialized_db, capsys):
        cmd_status(MockArgs(db=initialized_db, json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "db_size_bytes" in data
        assert data["db_size_bytes"] > 0

# -----------------------------------------------------------------------
# atoms tests
# -----------------------------------------------------------------------

class TestAtoms:
    def test_atoms_before_ingest(self, initialized_db, capsys):
        cmd_atoms(MockArgs(db=initialized_db))
        out = capsys.readouterr().out
        # No promoted atoms yet
        assert "No atoms yet" in out or "Atoms (0)" in out

    def test_atoms_after_ingest(self, trained_db, capsys):
        cmd_atoms(MockArgs(db=trained_db))
        out = capsys.readouterr().out
        assert "Atoms" in out

    def test_atoms_json(self, trained_db, capsys):
        cmd_atoms(MockArgs(db=trained_db, json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        if data:
            assert "label" in data[0]
            assert "confidence" in data[0]

    def test_atoms_with_seeds(self, initialized_db, capsys):
        cmd_atoms(MockArgs(db=initialized_db, seeds=True))
        out = capsys.readouterr().out
        # Should include seed atoms
        assert "exists" in out or "feel" in out or "Atoms" in out

    def test_atoms_confidence_in_range(self, trained_db, capsys):
        cmd_atoms(MockArgs(db=trained_db, json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        for item in data:
            assert 0.0 <= item["confidence"] <= 1.0

# -----------------------------------------------------------------------
# senses tests
# -----------------------------------------------------------------------

class TestSenses:
    def test_senses_known_concept(self, trained_db, capsys):
        atoms = _load_rsvs(trained_db).atoms()
        if not atoms:
            pytest.skip("No atoms promoted")
        cmd_senses(MockArgs(db=trained_db, concept=atoms[0]))
        out = capsys.readouterr().out
        assert "Senses for" in out

    def test_senses_json(self, trained_db, capsys):
        atoms = _load_rsvs(trained_db).atoms()
        if not atoms:
            pytest.skip("No atoms promoted")
        cmd_senses(MockArgs(db=trained_db, concept=atoms[0], json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        if data:
            assert "coherence" in data[0]
            assert "core_atoms" in data[0]

    def test_senses_unknown_exits(self, trained_db):
        with pytest.raises(SystemExit) as exc:
            cmd_senses(MockArgs(db=trained_db, concept="zzz_unknown"))
        assert exc.value.code != 0

# -----------------------------------------------------------------------
# info tests
# -----------------------------------------------------------------------

class TestInfo:
    def test_info_known_atom(self, trained_db, capsys):
        atoms = _load_rsvs(trained_db).atoms()
        if not atoms:
            pytest.skip("No atoms promoted")
        cmd_info(MockArgs(db=trained_db, atom=atoms[0]))
        out = capsys.readouterr().out
        assert "confidence" in out
        assert "tier" in out

    def test_info_json(self, trained_db, capsys):
        atoms = _load_rsvs(trained_db).atoms()
        if not atoms:
            pytest.skip("No atoms promoted")
        cmd_info(MockArgs(db=trained_db, atom=atoms[0], json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "label" in data
        assert "confidence" in data
        assert "tier" in data

    def test_info_unknown_exits(self, trained_db):
        with pytest.raises(SystemExit) as exc:
            cmd_info(MockArgs(db=trained_db, atom="zzz_unknown"))
        assert exc.value.code != 0

    def test_info_confidence_in_range(self, trained_db, capsys):
        atoms = _load_rsvs(trained_db).atoms()
        if not atoms:
            pytest.skip("No atoms promoted")
        cmd_info(MockArgs(db=trained_db, atom=atoms[0], json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert 0.0 <= data["confidence"] <= 1.0

# -----------------------------------------------------------------------
# Parser tests
# -----------------------------------------------------------------------

class TestParser:
    def test_parser_builds(self):
        p = build_parser()
        assert p is not None

    def test_init_subcommand(self):
        p = build_parser()
        args = p.parse_args(["init", "--db", "/tmp/x.json"])
        assert args.command == "init"
        assert args.db == "/tmp/x.json"

    def test_ingest_subcommand(self):
        p = build_parser()
        args = p.parse_args(["ingest", "some text", "--db", "/tmp/x.json"])
        assert args.command == "ingest"
        assert args.text_or_file == "some text"

    def test_query_subcommand(self):
        p = build_parser()
        args = p.parse_args(["query", "stone", "hard texture", "--top", "3"])
        assert args.command == "query"
        assert args.concept == "stone"
        assert args.top == 3

    def test_similarity_subcommand(self):
        p = build_parser()
        args = p.parse_args(["similarity", "stone", "hard"])
        assert args.a == "stone"
        assert args.b == "hard"

    def test_json_flag(self):
        p = build_parser()
        args = p.parse_args(["status", "--json"])
        assert args.json is True

    def test_default_db(self):
        p = build_parser()
        args = p.parse_args(["status"])
        assert args.db == "rsvs.json"


class TestOperationalCommands:
    def test_ingest_corpus_json(self, initialized_db, capsys):
        cmd_ingest_corpus(MockArgs(db=initialized_db, domains=["geology", "biology"], json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["api_version"] == "v1"
        assert "summary" in data

    def test_eval_json(self, initialized_db, capsys):
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS))
        cmd_eval(MockArgs(db=initialized_db, domains=["geology"], json=True))
        out = capsys.readouterr().out
        data = json.loads(out[out.find("{"):])
        assert data["api_version"] == "v1"
        assert "benchmarks" in data

    def test_replay_events_json(self, initialized_db, capsys):
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS))
        cmd_replay_events(MockArgs(db=initialized_db, after_seq=None, limit=50, json=True))
        out = capsys.readouterr().out
        data = json.loads(out[out.find("{"):])
        assert "events" in data
        assert "latest_seq" in data
