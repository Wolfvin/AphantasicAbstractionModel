"""
stage0 — AAM Rule-Based System (v1.0.0)

This package contains the complete rule-based architecture of the
Aphantasic Abstraction Model, from ingest (Layer 0) through output
and reasoning (Layer 3).

Layout:
    stage0/
    ├── layer0/          Perceptual Front-End (ingest)
    ├── layer1/          Rust Core + PyO3 Bridge
    ├── layer2/          Cognitive Runtime (reasoning)
    ├── layer3/          Deductive Reasoning & Output
    ├── pipeline.py      AamPipeline (wires all layers)
    ├── config.py        Pipeline configuration management
    └── python/          Python rsvs package (API, CLI, server)

Usage:
    All internal imports (e.g. `from layer2.bridge import ...`) work
    automatically when stage0 is imported first:

        import stage0          # adds stage0/ to sys.path
        from layer2.bridge import get_bridge
        from pipeline import AamPipeline

    Or use the convenience re-exports:

        from stage0 import AamPipeline

Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Ensure stage0/ is on sys.path so that `from layer2.bridge import ...`
#    works from ANY working directory, not just the repo root. ──
_STAGE0_DIR = str(Path(__file__).resolve().parent)
if _STAGE0_DIR not in sys.path:
    sys.path.insert(0, _STAGE0_DIR)

__version__ = "1.0.0"


def _lazy_import_pipeline():
    """Lazy-import AamPipeline to avoid circular imports at package load."""
    from pipeline import AamPipeline  # noqa: F401
    return AamPipeline


# Convenience: `from stage0 import AamPipeline` works, but only on first access.
def __getattr__(name):
    if name == "AamPipeline":
        return _lazy_import_pipeline()
    raise AttributeError(f"module 'stage0' has no attribute {name!r}")
