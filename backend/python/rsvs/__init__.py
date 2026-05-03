"""
RSVS — Recursive Symbolic Vector Space
Python bindings for the RSVS knowledge graph system.

Quick start::

    from rsvs import Rsvs

    r = Rsvs()
    r.ingest("Stone is a hard solid mineral material.")
    r.ingest("Bone is a hard solid organic structure.")

    sim = r.similarity("stone", "bone")
    print(f"jaccard: {sim.jaccard:.3f}")
    print(f"shared:  {sim.shared}")

    result = r.query("stone", "texture surface")
    print(result.top_atoms(3))
"""

try:
    from ._rsvs import (
        PyRsvs as Rsvs,
        PyIngestStats as IngestStats,
        PyIngestMetaV1 as IngestMetaV1,
        PyQueryResult as QueryResult,
        PySimResult as SimResult,
        PyAtomInfo as AtomInfo,
        PySenseInfo as SenseInfo,
    )
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "Failed to import native module 'rsvs._rsvs'. "
        "Build/install with `maturin develop` or `pip install -e .` first."
    ) from exc

__all__ = ["Rsvs", "IngestStats", "IngestMetaV1", "QueryResult", "SimResult", "AtomInfo", "SenseInfo"]
__version__ = "0.5.0"

from .cli import main as cli_main

__all__ += ["cli_main"]

from .corpus import DOMAINS, get_domain_text, get_all_text, domain_names
from .ingest_wiki import ingest_domains, print_report

__all__ += ["DOMAINS", "get_domain_text", "get_all_text", "domain_names",
            "ingest_domains", "print_report"]

from .eval import run_eval, EvalReport, BenchmarkResult
__all__ += ["run_eval", "EvalReport", "BenchmarkResult"]
