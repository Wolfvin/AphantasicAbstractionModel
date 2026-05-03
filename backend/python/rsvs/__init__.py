"""
RSVS — Recursive Symbolic Vector Space
Python bindings for the RSVS knowledge graph system.

v4.2: Unified node model, appraise/relate methods, PyNodeInfo.

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

    # v4.2: appraise and relate
    appraise = r.appraise("stone is hard")
    print(f"verdict: {appraise.verdict}")

    relate = r.relate("stone")
    print(f"related nodes: {len(relate.related_nodes)}")
"""

try:
    from ._rsvs import (
        PyRsvs as Rsvs,
        PyIngestStats as IngestStats,
        PyIngestMetaV1 as IngestMetaV1,
        PyQueryResult as QueryResult,
        PySimResult as SimResult,
        PyNodeInfo as AtomInfo,  # backward compat alias
        PySenseInfo as SenseInfo,
        # v4.2 types
        PyNodeInfo as NodeInfo,
        PyAppraiseResult as AppraiseResult,
        PyRelateResult as RelateResult,
    )
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "Failed to import native module 'rsvs._rsvs'. "
        "Build/install with `maturin develop` or `pip install -e .` first."
    ) from exc

__all__ = [
    "Rsvs",
    "IngestStats",
    "IngestMetaV1",
    "QueryResult",
    "SimResult",
    "AtomInfo",
    "SenseInfo",
    # v4.2
    "NodeInfo",
    "AppraiseResult",
    "RelateResult",
]
__version__ = "0.5.0"

from .cli import main as cli_main

__all__ += ["cli_main"]

from .corpus import DOMAINS, get_domain_text, get_all_text, domain_names
from .ingest_wiki import ingest_domains, print_report

__all__ += ["DOMAINS", "get_domain_text", "get_all_text", "domain_names",
            "ingest_domains", "print_report"]

from .eval import run_eval, EvalReport, BenchmarkResult
__all__ += ["run_eval", "EvalReport", "BenchmarkResult"]
