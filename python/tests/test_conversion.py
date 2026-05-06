"""Tests for RSVS Rust→Python conversion module.

Covers:
  - _convert_rust_node: seed, non-seed, compressed, no render key, semantic nesting
  - _convert_rust_edge: basic conversion, no render key
  - _build_bridge_snapshot: schema version, node/edge conversion

Run with: python3 -m pytest tests/test_conversion.py -v
"""

import pytest

from rsvs.conversion import (
    _convert_rust_node,
    _convert_rust_edge,
    _convert_rust_event,
    _build_bridge_snapshot,
)
from rsvs.config import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rust_seed_node(**overrides):
    """Build a minimal Rust seed node dict."""
    node = {
        "id": 1,
        "label": "exists",
        "surface_label": "exists@en",
        "kind": "node",
        "tier": 1,
        "confidence": 1.0,
        "status": "stable",
        "is_seed": True,
        "is_locked": True,
        "compression_state": "raw",
        "derived_from_node_ids": [],
        "sense_count": 0,
        "coherence": None,
    }
    node.update(overrides)
    return node


def _rust_non_seed_node(**overrides):
    """Build a minimal Rust non-seed node dict."""
    node = {
        "id": 10,
        "label": "stone",
        "surface_label": "stone@en",
        "kind": "node",
        "tier": 2,
        "confidence": 0.5,
        "status": "candidate",
        "is_seed": False,
        "is_locked": False,
        "compression_state": "raw",
        "derived_from_node_ids": [],
        "sense_count": 1,
        "coherence": 0.7,
    }
    node.update(overrides)
    return node


def _rust_compressed_node(**overrides):
    """Build a minimal Rust compressed node dict."""
    node = {
        "id": 5,
        "label": "comp",
        "surface_label": "comp@en",
        "kind": "node",
        "tier": 2,
        "confidence": 0.6,
        "status": "candidate",
        "is_seed": False,
        "is_locked": False,
        "compression_state": "compressed",
        "derived_from_node_ids": [1, 2],
        "compression_reason": "co-occurrence aggregation",
        "sense_count": 1,
        "coherence": 0.5,
    }
    node.update(overrides)
    return node


def _rust_edge(**overrides):
    """Build a minimal Rust edge dict."""
    edge = {
        "id": "1->2",
        "source": 1,
        "target": 2,
        "weight": 0.5,
        "source_type": "learned",
    }
    edge.update(overrides)
    return edge


def _rust_event(**overrides):
    """Build a minimal Rust event dict."""
    evt = {
        "api_version": "v1",
        "schema_version": "v4.2",
        "seq": 1,
        "correlation_id": "corr_test",
        "event_type": "node_created",
        "payload": {"id": 1, "label": "test"},
    }
    evt.update(overrides)
    return evt


# ===================================================================
# _convert_rust_node
# ===================================================================


class TestConvertRustNode:
    """Tests for _convert_rust_node."""

    def test_seed_node_conversion(self):
        """Seed node is converted with correct defaults."""
        rn = _rust_seed_node()
        result = _convert_rust_node(rn, "corr_1")
        assert result["is_seed"] is True
        assert result["is_locked"] is True
        assert result["tier"] == 1
        assert result["confidence"] == 1.0
        assert result["status"] == "stable"
        assert result["kind"] == "node"

    def test_non_seed_node_conversion(self):
        """Non-seed node is converted with correct defaults."""
        rn = _rust_non_seed_node()
        result = _convert_rust_node(rn, "corr_2")
        assert result["is_seed"] is False
        assert result["label"] == "stone"
        assert result["surface_label"] == "stone@en"
        assert result["kind"] == "node"

    def test_compressed_node_conversion(self):
        """Compressed node preserves compression metadata."""
        rn = _rust_compressed_node()
        result = _convert_rust_node(rn, "corr_3")
        semantic = result["semantic"]
        assert semantic["compression_state"] == "compressed"
        assert semantic["derived_from_node_ids"] == [1, 2]
        assert semantic["compression_reason"] == "co-occurrence aggregation"

    def test_no_render_key(self):
        """Converted node must NOT contain 'render' metadata."""
        rn = _rust_seed_node()
        result = _convert_rust_node(rn, "corr_4")
        assert "render" not in result
        assert "render_key" not in result
        # Check nested dicts don't have render keys either
        for key in result:
            assert "render" not in key.lower()

    def test_semantic_nested_properly(self):
        """Semantic metadata is nested under 'semantic' key."""
        rn = _rust_non_seed_node()
        result = _convert_rust_node(rn, "corr_5")
        assert "semantic" in result
        assert isinstance(result["semantic"], dict)
        assert "compression_state" in result["semantic"]
        assert "derived_from_node_ids" in result["semantic"]

    def test_policy_meta_present(self):
        """Policy metadata is included in the converted node."""
        rn = _rust_non_seed_node()
        result = _convert_rust_node(rn, "corr_6")
        assert "policy_meta" in result
        pm = result["policy_meta"]
        assert "policy_version" in pm
        assert "governance_score" in pm
        assert "status_flip_count" in pm
        assert "seen_fingerprints" in pm

    def test_seed_node_has_seed_registry(self):
        """Seed node policy_meta includes seed_registry=True."""
        rn = _rust_seed_node()
        result = _convert_rust_node(rn, "corr_7")
        assert result["policy_meta"].get("seed_registry") is True

    def test_non_seed_no_seed_registry(self):
        """Non-seed node policy_meta does NOT include seed_registry."""
        rn = _rust_non_seed_node()
        result = _convert_rust_node(rn, "corr_8")
        assert "seed_registry" not in result["policy_meta"]

    def test_provenance_present(self):
        """Provenance metadata is included in the converted node."""
        rn = _rust_non_seed_node()
        result = _convert_rust_node(rn, "corr_9")
        assert "provenance" in result
        prov = result["provenance"]
        assert "source_batch_id" in prov
        assert "source_domain" in prov
        assert "source_type" in prov

    def test_seed_provenance_is_core_seed(self):
        """Seed node provenance source_domain is 'core_seed'."""
        rn = _rust_seed_node()
        result = _convert_rust_node(rn, "corr_10")
        assert result["provenance"]["source_domain"] == "core_seed"
        assert result["provenance"]["source_type"] == "bootstrap"

    def test_compressed_without_reason_gets_default(self):
        """Compressed node without explicit reason gets default."""
        rn = _rust_compressed_node()
        del rn["compression_reason"]
        result = _convert_rust_node(rn, "corr_11")
        assert result["semantic"]["compression_reason"] == "co-occurrence aggregation"

    def test_raw_without_derived_gets_base_ingest_signal(self):
        """Raw node without derived ids gets 'base_ingest_signal' reason."""
        rn = _rust_non_seed_node()
        result = _convert_rust_node(rn, "corr_12")
        assert result["semantic"]["compression_reason"] == "base_ingest_signal"


# ===================================================================
# _convert_rust_edge
# ===================================================================


class TestConvertRustEdge:
    """Tests for _convert_rust_edge."""

    def test_basic_edge_conversion(self):
        """Edge is converted with correct fields."""
        re = _rust_edge()
        result = _convert_rust_edge(re)
        assert result["source"] == 1
        assert result["target"] == 2
        assert result["weight"] == 0.5
        assert result["direction"] == "undirected"

    def test_no_render_key(self):
        """Converted edge must NOT contain 'render' metadata."""
        re = _rust_edge()
        result = _convert_rust_edge(re)
        assert "render" not in result
        assert "render_key" not in result

    def test_weight_is_rounded(self):
        """Edge weight is rounded to 3 decimal places."""
        re = _rust_edge(weight=0.1234567)
        result = _convert_rust_edge(re)
        assert result["weight"] == round(0.1234567, 3)

    def test_default_source_type(self):
        """Default source_type is 'learned'."""
        re = _rust_edge()
        del re["source_type"]
        result = _convert_rust_edge(re)
        assert result["source_type"] == "learned"

    def test_default_direction_is_undirected(self):
        """Default direction is 'undirected'."""
        re = _rust_edge()
        result = _convert_rust_edge(re)
        assert result["direction"] == "undirected"


# ===================================================================
# _convert_rust_event
# ===================================================================


class TestConvertRustEvent:
    """Tests for _convert_rust_event."""

    def test_basic_event_conversion(self):
        """Event is converted with correct fields."""
        evt = _rust_event()
        result = _convert_rust_event(evt, "corr_evt")
        assert "event_id" in result
        assert "timestamp" in result
        assert result["event_type"] == "node_created"
        # correlation_id comes from the Rust event if present, otherwise from the argument
        assert result["correlation_id"] in ("corr_test", "corr_evt")

    def test_event_has_animation_hint(self):
        """Converted event includes animation_hint."""
        evt = _rust_event()
        result = _convert_rust_event(evt, "corr_anim")
        assert "animation_hint" in result
        hint = result["animation_hint"]
        assert "priority" in hint
        assert "focus_node_id" in hint
        assert "burst_group" in hint

    def test_node_created_is_normal_priority(self):
        """node_created events have 'normal' priority."""
        evt = _rust_event(event_type="node_created")
        result = _convert_rust_event(evt, "corr_pri")
        assert result["animation_hint"]["priority"] == "normal"

    def test_other_events_are_low_priority(self):
        """Non-node_created events have 'low' priority."""
        evt = _rust_event(event_type="confidence_changed")
        result = _convert_rust_event(evt, "corr_low")
        assert result["animation_hint"]["priority"] == "low"

    def test_preserves_rust_metadata(self):
        """Converted event preserves Rust seq and api_version."""
        evt = _rust_event()
        result = _convert_rust_event(evt, "corr_meta")
        assert result["seq"] == 1
        assert result["api_version"] == "v1"
        assert result["schema_version"] == "v4.2"


# ===================================================================
# _build_bridge_snapshot
# ===================================================================


class TestBuildBridgeSnapshot:
    """Tests for _build_bridge_snapshot."""

    def _make_rust_snapshot(self, nodes=None, edges=None):
        """Build a minimal Rust snapshot dict."""
        return {
            "api_version": "v1",
            "schema_version": "v4.2",
            "latest_seq": 1,
            "total_contexts": 5,
            "nodes": nodes if nodes is not None else [_rust_seed_node()],
            "edges": edges if edges is not None else [],
        }

    def test_snapshot_has_schema_version(self):
        """Snapshot must have the correct schema_version."""
        snap = self._make_rust_snapshot()
        result = _build_bridge_snapshot(snap, "corr_snap")
        assert result["schema_version"] == SCHEMA_VERSION

    def test_snapshot_converts_all_nodes(self):
        """All nodes in the Rust snapshot are converted."""
        nodes = [_rust_seed_node(), _rust_non_seed_node()]
        snap = self._make_rust_snapshot(nodes=nodes)
        result = _build_bridge_snapshot(snap, "corr_nodes")
        assert len(result["nodes"]) == 2

    def test_snapshot_converts_all_edges(self):
        """All edges in the Rust snapshot are converted."""
        edges = [_rust_edge(), _rust_edge(id="2->3", source=2, target=3)]
        snap = self._make_rust_snapshot(edges=edges)
        result = _build_bridge_snapshot(snap, "corr_edges")
        assert len(result["edges"]) == 2

    def test_snapshot_has_context(self):
        """Snapshot includes context metadata."""
        snap = self._make_rust_snapshot()
        result = _build_bridge_snapshot(snap, "corr_ctx", lang_code="en")
        assert "context" in result
        assert result["context"]["domain"] == "rsvs-core"
        assert result["context"]["language_code"] == "en"
        assert result["context"]["batch_id"] == "corr_ctx"

    def test_snapshot_has_snapshot_id(self):
        """Snapshot includes a snapshot_id."""
        snap = self._make_rust_snapshot()
        result = _build_bridge_snapshot(snap, "corr_id")
        assert "snapshot_id" in result
        assert result["snapshot_id"].startswith("snapshot_")

    def test_snapshot_has_generated_at(self):
        """Snapshot includes generated_at timestamp."""
        snap = self._make_rust_snapshot()
        result = _build_bridge_snapshot(snap, "corr_ts")
        assert "generated_at" in result
        assert len(result["generated_at"]) > 0

    def test_empty_rust_snapshot(self):
        """Empty Rust snapshot produces valid bridge snapshot with no extra nodes."""
        snap = self._make_rust_snapshot(nodes=[], edges=[])
        result = _build_bridge_snapshot(snap, "corr_empty")
        # When no Rust nodes are provided, conversion should produce empty list
        assert len(result["nodes"]) == 0
        assert len(result["edges"]) == 0
