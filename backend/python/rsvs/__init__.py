"""RSVS — Relational Symbolic Vocabulary System.

A cognitive symbolic engine with hard attention, multi-sense disambiguation,
and autonomous tiered memory lifecycle.

Architecture:
    Python HTTP layer + artifact persistence
         ↓
    Rust core via PyO3 (rsvs._rsvs.Rsvs)
         ↓
    Graph, Attention, Autonomy, Sense, Pipeline
"""

__version__ = "4.2.0"
__schema_version__ = "v4.2"
__api_version__ = "v1"

# Try to import the Rust native module. If not available (e.g. maturin not built),
# set a flag so downstream code can adapt gracefully.
_rust_core_available = False

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
    _rust_core_available = True
except ImportError:
    # Rust core not built — this is OK for import-time, the bridge server
    # and validation tests can still function without it.
    pass

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

from .cli import main as cli_main

__all__ += ["cli_main"]

from .corpus import DOMAINS, get_domain_text, get_all_text, domain_names

__all__ += ["DOMAINS", "get_domain_text", "get_all_text", "domain_names"]


def __getattr__(name):
    """Lazy imports for modules that depend on Rsvs (avoid circular import)."""
    if name == "ingest_domains":
        from .ingest_wiki import ingest_domains
        return ingest_domains
    if name == "print_report":
        from .ingest_wiki import print_report
        return print_report
    if name == "run_eval":
        from .eval import run_eval
        return run_eval
    if name == "EvalReport":
        from .eval import EvalReport
        return EvalReport
    if name == "BenchmarkResult":
        from .eval import BenchmarkResult
        return BenchmarkResult
    raise AttributeError(f"module 'rsvs' has no attribute {name!r}")
