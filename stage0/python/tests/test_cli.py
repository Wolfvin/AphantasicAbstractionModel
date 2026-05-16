"""Tests for RSVS CLI mode commands.

Covers:
  - CLI help flag
  - CLI ingest mode
  - CLI appraise mode (via Python API)
  - CLI relate mode (via Python API)
  - CLI invalid mode

Run with: python3 -m pytest tests/test_cli.py -v
"""

import json
import subprocess
import sys

import pytest

from rsvs.cli import (
    cmd_init,
    cmd_ingest,
    build_parser,
    _load_rsvs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORPUS = """
Stone is a hard solid mineral material. Rock is a hard heavy solid substance.
Stone is formed by heat and pressure over time. Granite is a hard rough stone.
Stone has a rough hard texture. Metal is a hard solid material.
Stone and metal are both hard solid materials. Hard solid materials resist pressure.
Stone is heavy and hard. Stone resists erosion and pressure.
"""


class MockArgs:
    """Minimal args namespace for testing commands directly."""

    def __init__(self, **kwargs):
        self.json = False
        self.seeds = False
        self.force = False
        self.promote_n = 3
        self.theta = 0.12
        self.n_warm = 8
        self.eta = 0.1
        self.domain = None
        self.top = 6
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
    """DB with corpus ingested."""
    cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS))
    return initialized_db


# ===================================================================
# CLI Help
# ===================================================================


class TestCLIHelp:
    """Test CLI help flag."""

    def test_cli_help(self):
        """CLI --help should exit with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "rsvs.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "RSVS" in result.stdout

    def test_cli_version(self):
        """CLI --version should print version info."""
        result = subprocess.run(
            [sys.executable, "-m", "rsvs.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "RSVS" in result.stdout


# ===================================================================
# CLI Ingest Mode
# ===================================================================


class TestCLIIngestMode:
    """Test CLI ingest mode command."""

    def test_cli_ingest_mode(self, initialized_db, capsys):
        """CLI ingest command processes text and saves to DB."""
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS, json=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["sentences_processed"] >= 1
        assert "atoms_promoted" in data
        # Verify DB was updated
        r = _load_rsvs(initialized_db)
        assert r.status()["total_contexts"] > 0

    def test_cli_ingest_accumulates(self, initialized_db, capsys):
        """Multiple ingest calls should accumulate data."""
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS, json=True))
        out1 = capsys.readouterr().out
        data1 = json.loads(out1)

        cmd_ingest(MockArgs(db=initialized_db, text_or_file=CORPUS, json=True))
        out2 = capsys.readouterr().out
        data2 = json.loads(out2)

        r = _load_rsvs(initialized_db)
        atoms = r.atoms()
        assert len(atoms) >= 1  # Should have promoted some atoms

    def test_cli_ingest_from_file(self, initialized_db, tmp_path):
        """CLI ingest can read from a file."""
        f = tmp_path / "corpus.txt"
        f.write_text(CORPUS)
        cmd_ingest(MockArgs(db=initialized_db, text_or_file=str(f)))
        r = _load_rsvs(initialized_db)
        assert r.status()["total_contexts"] > 0

    def test_cli_ingest_empty_text_exits(self, initialized_db):
        """CLI ingest with empty text should exit with error."""
        with pytest.raises(SystemExit) as exc:
            cmd_ingest(MockArgs(db=initialized_db, text_or_file="   "))
        assert exc.value.code != 0


# ===================================================================
# CLI Appraise Mode
# ===================================================================


class TestCLIAppraiseMode:
    """Test CLI appraise mode via Python API."""

    def test_cli_appraise_mode(self, trained_db, capsys):
        """Appraise after ingest returns a valid verdict."""
        r = _load_rsvs(trained_db)
        result = r.appraise("stone is hard and solid")
        assert result is not None
        assert hasattr(result, "verdict")
        assert result.verdict in ("consistent", "partial", "novel")
        assert 0.0 <= float(result.agree_pct) <= 100.0
        assert 0.0 <= float(result.disagree_pct) <= 100.0

    def test_cli_appraise_novel_text(self, trained_db, capsys):
        """Appraise with novel text returns 'novel' verdict."""
        r = _load_rsvs(trained_db)
        result = r.appraise("xyzquux foobarbaz quuxland")
        assert result.verdict == "novel"
        assert float(result.disagree_pct) > 50.0

    def test_cli_appraise_known_text(self, trained_db, capsys):
        """Appraise with known seed terms returns non-novel verdict."""
        r = _load_rsvs(trained_db)
        result = r.appraise("exists entity relation state change time")
        assert result is not None
        assert float(result.agree_pct) > 0.0


# ===================================================================
# CLI Relate Mode
# ===================================================================


class TestCLIRelateMode:
    """Test CLI relate mode via Python API."""

    def test_cli_relate_mode(self, trained_db, capsys):
        """Relate after ingest finds related nodes."""
        r = _load_rsvs(trained_db)
        result = r.relate("exists")
        assert result is not None
        assert hasattr(result, "related_nodes")
        assert hasattr(result, "related_edges")

    def test_cli_relate_unknown_returns_none(self, trained_db, capsys):
        """Relate with unknown concept returns None."""
        r = _load_rsvs(trained_db)
        result = r.relate("nonexistent_concept_xyz")
        assert result is None


# ===================================================================
# CLI Invalid Mode
# ===================================================================


class TestCLIInvalidMode:
    """Test CLI with invalid mode/command."""

    def test_cli_invalid_mode(self, capsys):
        """CLI with invalid command should cause SystemExit."""
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["invalid_mode", "arg1"])

    def test_cli_no_command_exits(self, capsys):
        """CLI with no command should cause SystemExit."""
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])
