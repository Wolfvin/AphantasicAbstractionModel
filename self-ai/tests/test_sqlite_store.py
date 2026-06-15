# @WHO:   self-ai/tests/test_sqlite_store.py
# @WHAT:  Tests for SQLiteGraphStore backend in UnderstandingGraph
# @PART:  self-ai/tests

"""Tests for SQLite-backed UnderstandingGraph persistence.

Covers:
    1. test_add_and_get_node          — add a node, get by ID → identical
    2. test_save_and_load_roundtrip   — save nodes dict, load again → identical
    3. test_delete_node               — add, delete, get → None
    4. test_fallback_to_json_if_no_db_extension — .json path → no SQLite store
"""

import os
import sys
import tempfile

import pytest

# Ensure the self-ai source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from derivation.understanding_builder import UnderstandingGraph, UnderstandingNode, Transformation
from derivation.sqlite_store import SQLiteGraphStore


# ──────────────── Fixtures ────────────────

def _make_node(id: str = "test_node", name: str = "Test Node",
               concept: str = "Test concept", abstraction: str = "Test abstraction",
               **kwargs) -> UnderstandingNode:
    """Helper to create an UnderstandingNode with sensible defaults."""
    return UnderstandingNode(
        id=id,
        name=name,
        concept=concept,
        abstraction=abstraction,
        **kwargs,
    )


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary .db file path."""
    return str(tmp_path / "test_graph.db")


@pytest.fixture
def tmp_json(tmp_path):
    """Provide a temporary .json file path."""
    return str(tmp_path / "test_graph.json")


@pytest.fixture
def sqlite_store(tmp_db):
    """Provide a SQLiteGraphStore connected to a temp DB."""
    return SQLiteGraphStore(tmp_db)


# ──────────────── Test 1: add_node + get_node ────────────────

def test_add_and_get_node(sqlite_store, tmp_db):
    """Add a node via SQLiteGraphStore, retrieve it by ID → data is identical."""
    node = _make_node(
        id="signal_flip",
        name="Signal Flip",
        concept="Kata pengecualian membalik jawaban",
        abstraction="IF 'kecuali' → jawaban = OPPOSITE",
        conditions=["kecuali", "selain", "terkecuali"],
        confidence=0.9,
        source="self_discovered",
        transformation=Transformation(
            kind="signal_flip",
            trigger={"words": ["kecuali", "selain"]},
            action="Flip the answer to its opposite",
        ),
    )

    sqlite_store.add_node(node)

    # Retrieve and verify
    result = sqlite_store.get_node("signal_flip")
    assert result is not None, "get_node should return the node dict, not None"

    # Reconstruct UnderstandingNode from dict
    loaded_node = UnderstandingNode.from_dict(result)

    assert loaded_node.id == "signal_flip"
    assert loaded_node.name == "Signal Flip"
    assert loaded_node.concept == "Kata pengecualian membalik jawaban"
    assert loaded_node.abstraction == "IF 'kecuali' → jawaban = OPPOSITE"
    assert loaded_node.confidence == 0.9
    assert loaded_node.source == "self_discovered"
    assert loaded_node.conditions == ["kecuali", "selain", "terkecuali"]
    assert loaded_node.transformation is not None
    assert loaded_node.transformation.kind == "signal_flip"


# ──────────────── Test 2: save + load roundtrip ────────────────

def test_save_and_load_roundtrip(sqlite_store, tmp_db):
    """Save a full nodes dict via SQLiteGraphStore, load again → identical."""
    nodes = {
        "node_a": _make_node(id="node_a", name="Node A", concept="Concept A",
                             abstraction="Abstract A", confidence=0.8),
        "node_b": _make_node(id="node_b", name="Node B", concept="Concept B",
                             abstraction="Abstract B", confidence=0.6,
                             conditions=["karena", "sebab"]),
    }

    sqlite_store.save(nodes)

    # Load from a fresh store instance (simulates process restart)
    fresh_store = SQLiteGraphStore(tmp_db)
    loaded = fresh_store.load()

    assert len(loaded) == 2, f"Expected 2 nodes, got {len(loaded)}"

    # Reconstruct and verify node_a
    node_a = UnderstandingNode.from_dict(loaded["node_a"])
    assert node_a.id == "node_a"
    assert node_a.name == "Node A"
    assert node_a.confidence == 0.8

    # Reconstruct and verify node_b
    node_b = UnderstandingNode.from_dict(loaded["node_b"])
    assert node_b.id == "node_b"
    assert node_b.conditions == ["karena", "sebab"]
    assert node_b.confidence == 0.6


# ──────────────── Test 3: delete_node ────────────────

def test_delete_node(sqlite_store, tmp_db):
    """Add a node, delete it, get_node → None."""
    node = _make_node(id="to_delete", name="Delete Me")

    sqlite_store.add_node(node)
    assert sqlite_store.get_node("to_delete") is not None, "Node should exist after add"

    sqlite_store.delete_node("to_delete")
    result = sqlite_store.get_node("to_delete")
    assert result is None, "get_node should return None after delete"


# ──────────────── Test 4: JSON fallback ────────────────

def test_fallback_to_json_if_no_db_extension(tmp_json):
    """store_path ending in .json should NOT use SQLiteGraphStore.

    This ensures backward compatibility: existing code using .json files
    continues to work with the original JSON flat-file persistence.
    """
    graph = UnderstandingGraph(store_path=tmp_json)

    # Verify SQLite is NOT used
    assert not graph._use_sqlite, "JSON path should NOT trigger SQLite backend"
    assert graph._sqlite_store is None, "No SQLiteGraphStore for .json paths"

    # Verify JSON store still works: add a node and check the file
    node = _make_node(id="json_node", name="JSON Node", confidence=0.75)
    graph.add_node(node)

    # The .json file should exist on disk
    assert os.path.exists(tmp_json), "JSON file should be created"

    # Verify the node can be retrieved from a new graph instance
    graph2 = UnderstandingGraph(store_path=tmp_json)
    loaded = graph2.get_node("json_node")
    assert loaded is not None, "Node should be loadable from JSON store"
    assert loaded.id == "json_node"
    assert loaded.confidence == 0.75
