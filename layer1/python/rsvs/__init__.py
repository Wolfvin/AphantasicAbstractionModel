"""RSVS — Recursive Symbolic Vocabulary System / Aphantasic Abstraction Model.

This package provides the PyO3-compiled Rust extension module ``rsvs._rsvs``
which exposes the v12.0 DAG pipeline engine to Python.

When the Rust extension is not available (e.g., the wheel has not been built),
all classes fall back to stub implementations that raise ``ImportError``.
"""

from rsvs._rsvs import (
    PyV12Pipeline,
    PySemanticAtom,
    PyComposition,
    PyCompositionMember,
    PyKnowledgeGap,
    PyAcquisitionDecision,
    PyInquiryQuestion,
    PyV12IngestResult,
)

__all__ = [
    "PyV12Pipeline",
    "PySemanticAtom",
    "PyComposition",
    "PyCompositionMember",
    "PyKnowledgeGap",
    "PyAcquisitionDecision",
    "PyInquiryQuestion",
    "PyV12IngestResult",
]
