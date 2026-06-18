"""
Structural tests for AGNN skeleton.

These tests verify STRUCTURE (not logic) - they must pass against the
NotImplementedError stubs. Logic tests will be added in downstream PRs
as each module gets implemented.

Tests:
- test_episome_dataclass: Episome can be constructed with required fields.
- test_semesome_dataclass: Semesome can be constructed with required fields.
- test_engram_complex_wraps_agnn_graph: EngramComplex wraps AGNNGraph (has _graph).
- test_agnncore_has_required_methods: AGNNCore exposes all 8 public methods.
"""

import os
import sys
from pathlib import Path

import pytest

# Make AGNN package importable when running from repo root.
# Tests are invoked as: python -m pytest AGNN/tests/ -v
# So we add the AGNN/ directory (parent of tests/) to sys.path.
_AGNN_ROOT = Path(__file__).resolve().parent.parent
if str(_AGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNN_ROOT))

# Also ensure self-ai/src is importable for AGNNGraph in engram_complex.
_SELF_AI_SRC = _AGNN_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))


# ----------------------------------------------------------------------
# Test 1: Episome dataclass
# ----------------------------------------------------------------------

def test_episome_dataclass():
    """Episome can be constructed with required fields and has expected attributes."""
    from engrams.episodic_engram import Episome

    e = Episome(id=1, text="insulin resistance causes T2D", confidence=0.6)
    assert e.id == 1
    assert e.text == "insulin resistance causes T2D"
    assert e.confidence == 0.6
    # Default edge_type and type marker
    assert e.edge_type == "CATEGORICAL"
    assert e.type == "episodic"


# ----------------------------------------------------------------------
# Test 2: Semesome dataclass
# ----------------------------------------------------------------------

def test_semesome_dataclass():
    """Semesome can be constructed with required fields and has expected attributes."""
    from engrams.semantic_engram import Semesome

    s = Semesome(type="CAUSAL", weight=0.7, source="smoking", target="cancer")
    assert s.type == "CAUSAL"
    assert s.weight == 0.7
    assert s.source == "smoking"
    assert s.target == "cancer"
    assert s.type_memory == "semantic"


# ----------------------------------------------------------------------
# Test 3: EngramComplex wraps AGNNGraph
# ----------------------------------------------------------------------

def test_engram_complex_wraps_agnn_graph():
    """
    EngramComplex wraps (does NOT replace) AGNNGraph from self-ai/src/agnn/.

    If self-ai/src/agnn/graph.py is not available, this test is skipped
    (skeleton PR cannot guarantee self-ai layout in all CI contexts).
    """
    try:
        from agnn.graph import AGNNGraph  # noqa: F401
    except ImportError:
        pytest.skip("self-ai/src/agnn/graph.py not available - skipping wrap test")

    from engrams.engram_complex import EngramComplex

    try:
        ec = EngramComplex()
    except ImportError:
        pytest.skip("EngramComplex() could not resolve AGNNGraph at runtime")

    # EngramComplex must delegate to a real AGNNGraph instance.
    assert hasattr(ec, "_graph"), "EngramComplex must expose _graph attribute"
    assert isinstance(ec._graph, AGNNGraph), (
        f"_graph must be an AGNNGraph instance, got {type(ec._graph).__name__}"
    )


# ----------------------------------------------------------------------
# Test 4: AGNNCore has required methods
# ----------------------------------------------------------------------

REQUIRED_METHODS = [
    "learn",
    "process",
    "introspect",
    "traverse",
    "consolidate",
    "reinforce",
    "penalize",
]


def test_agnncore_has_required_methods():
    """AGNNCore exposes all 7 public methods (+ __init__ = 8 total)."""
    from core import AGNNCore

    # __init__ is implicit, but verify class exists and has methods.
    assert callable(AGNNCore), "AGNNCore must be a class"

    for method_name in REQUIRED_METHODS:
        assert hasattr(AGNNCore, method_name), (
            f"AGNNCore missing required method: {method_name}"
        )
        assert callable(getattr(AGNNCore, method_name)), (
            f"AGNNCore.{method_name} must be callable"
        )
