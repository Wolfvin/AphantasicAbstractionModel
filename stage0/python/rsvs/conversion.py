"""RSVS Bridge format converters — Rust core → Python bridge format."""

from __future__ import annotations

from typing import Any

from .config import SCHEMA_VERSION, iso_now, make_id


__all__ = [
    "_project_node",
    "_convert_rust_node",
    "_convert_rust_edge",
    "_convert_rust_event",
    "_build_bridge_snapshot",
]


# ---------------------------------------------------------------------------
# View projection
# ---------------------------------------------------------------------------


def _project_node(
    node: dict[str, Any],
    view: str,
    node_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Project a full node down to a view-specific subset of fields."""
    semantic = node.get("semantic") or {}
    projected = {
        "id": node.get("id"),
        "label": node.get("label"),
        "surface_label": node.get("surface_label"),
        "language_links": node.get("language_links", []),
        "score": node.get("score"),
        "tier": node.get("tier"),
        "kind": node.get("kind"),
        "status": node.get("status"),
        "confidence": node.get("confidence"),
        "is_seed": bool(node.get("is_seed", False)),
        "is_locked": bool(node.get("is_locked", False)),
        "compression_state": semantic.get("compression_state"),
        "derived_from_node_ids": semantic.get("derived_from_node_ids", []),
        "atoms": semantic.get("atoms"),
        # v8.0: Layer and internal representation
        "layer": node.get("layer", semantic.get("layer", 0)),
        "internal_representation": bool(node.get("internal_representation", semantic.get("internal_representation", False))),
    }
    if view == "detail":
        projected["derived_nodes"] = [
            {
                "id": dep.get("id"),
                "label": dep.get("label"),
                "compression_state": (dep.get("semantic") or {}).get("compression_state"),
            }
            for dep in [node_index.get(dep_id) for dep_id in semantic.get("derived_from_node_ids", [])]
            if dep is not None
        ]
    return projected


# ---------------------------------------------------------------------------
# Rust snapshot → Python bridge format converters
# ---------------------------------------------------------------------------


def _convert_rust_node(rn: dict[str, Any], correlation_id: str) -> dict[str, Any]:
    """Convert a flat Rust RuntimeNode dict to the nested Python bridge node format.

    The Rust core's ``snapshot_v1()`` produces flat nodes like:
        {id, label, surface_label, kind, tier, confidence, status,
         is_seed, is_locked, compression_state, derived_from_node_ids,
         sense_count, coherence}

    The Python bridge / frontend expects:
        {id, label, surface_label, language_links, kind, tier, confidence,
         status, is_seed, is_locked, semantic: {...}, policy_meta: {...},
         provenance: {...}}

    NOTE: Layout/visual properties (position, size, color, glow) are the
    responsibility of the frontend render layer — the bridge does NOT
    generate ``render`` metadata.
    """
    compression_state = rn.get("compression_state", "raw")
    derived_ids = rn.get("derived_from_node_ids", [])
    atoms = rn.get("atoms", None)  # Explicit atom IDs for composed nodes

    # Build compression_reason
    compression_reason: str | None = None
    if compression_state == "composed":
        compression_reason = rn.get("compression_reason") or "composition"
    elif compression_state == "compressed":
        compression_reason = rn.get("compression_reason") or "co-occurrence aggregation"
    elif not derived_ids:
        compression_reason = "base_ingest_signal"

    is_seed = bool(rn.get("is_seed", False))

    # v8.0: Layer and internal_representation from Rust snapshot
    layer = rn.get("layer", 0 if not derived_ids else 1)
    internal_representation = bool(rn.get("internal_representation", False))

    # v8.0: Language links with proper structure (link_type, target_id)
    raw_language_links = rn.get("language_links", [])
    language_links = []
    for ll in raw_language_links:
        if isinstance(ll, dict):
            language_links.append({
                "link_type": ll.get("link_type", "structural_equivalence"),
                "target_id": ll.get("target_id"),
            })
        elif isinstance(ll, (list, tuple)) and len(ll) >= 2:
            # Legacy format: [link_type, target_id]
            language_links.append({
                "link_type": ll[0],
                "target_id": ll[1],
            })

    return {
        "id": rn.get("id"),
        "label": rn.get("label"),
        "surface_label": rn.get("surface_label"),
        "language_links": language_links,
        "kind": rn.get("kind", "node"),
        "tier": rn.get("tier", 1 if is_seed else 3),
        "confidence": rn.get("confidence", 1.0 if is_seed else 0.25),
        "status": rn.get("status", "stable" if is_seed else "new"),
        "is_seed": is_seed,
        "is_locked": bool(rn.get("is_locked", is_seed)),
        "layer": layer,
        "internal_representation": internal_representation,
        "semantic": {
            "compression_state": compression_state,
            "derived_from_node_ids": derived_ids,
            "atoms": atoms,
            "compression_reason": compression_reason,
            "layer": layer,
            "internal_representation": internal_representation,
        },
        "policy_meta": {
            "policy_version": SCHEMA_VERSION,
            "governance_score": rn.get("confidence", 1.0 if is_seed else 0.25),
            "candidate_evidence_pool": 0.0,
            "status_flip_count": 0,
            "seen_fingerprints": [],
            "short_window_hits": 0,
            "long_window_hits": 0,
            "last_seen_at": iso_now(),
            **({"seed_registry": True} if is_seed else {}),
        },
        "provenance": {
            "source_batch_id": rn.get("source_batch_id", correlation_id),
            "source_domain": "core_seed" if is_seed else "rsvs_core",
            "source_type": "bootstrap" if is_seed else "learned",
        },
    }


def _convert_rust_edge(re: dict[str, Any]) -> dict[str, Any]:
    """Convert a Rust RuntimeEdge to the Python bridge edge format.

    NOTE: Visual properties (thickness, color, opacity, pulse) are the
    responsibility of the frontend render layer — the bridge does NOT
    generate ``render`` metadata.
    """
    return {
        "id": re.get("id", ""),
        "source": re.get("source"),
        "target": re.get("target"),
        "direction": "undirected",
        "weight": round(float(re.get("weight", 0.0)), 3),
        "source_type": re.get("source_type", "learned"),
        "status": "new",
    }


def _convert_rust_event(evt: dict[str, Any], correlation_id: str) -> dict[str, Any]:
    """Convert a Rust RuntimeEvent to the Python bridge event format.

    Rust events have: api_version, schema_version, seq, correlation_id,
    event_type, payload.
    Python events expect: event_id, timestamp, correlation_id, event_type,
    payload, animation_hint.
    """
    payload = evt.get("payload", {})
    event_type = evt.get("event_type", "unknown")

    # Derive focus_node_id from payload if present
    focus_node_id = payload.get("id") or payload.get("node_id") or payload.get("source")
    priority = "normal" if event_type == "node_created" else "low"

    return {
        "event_id": make_id("evt"),
        "timestamp": iso_now(),
        "correlation_id": evt.get("correlation_id", correlation_id),
        "event_type": event_type,
        "payload": payload,
        "animation_hint": {
            "priority": priority,
            "focus_node_id": focus_node_id,
            "burst_group": correlation_id,
        },
        # Preserve Rust metadata for consumers that need it
        "seq": evt.get("seq"),
        "api_version": evt.get("api_version"),
        "schema_version": evt.get("schema_version"),
    }


def _build_bridge_snapshot(
    rust_snapshot_raw: dict[str, Any],
    correlation_id: str,
    lang_code: str = "en",
) -> dict[str, Any]:
    """Convert a full Rust RuntimeSnapshot to the Python bridge snapshot format.

    The Rust snapshot has: api_version, schema_version, latest_seq,
    total_contexts, nodes: [RuntimeNode], edges: [RuntimeEdge].

    The Python bridge snapshot has: schema_version, snapshot_id, generated_at,
    context: {...}, nodes: [nested node], edges: [nested edge].
    """
    rust_nodes = rust_snapshot_raw.get("nodes", [])
    rust_edges = rust_snapshot_raw.get("edges", [])

    nodes = [_convert_rust_node(rn, correlation_id) for rn in rust_nodes]
    edges = [_convert_rust_edge(re) for re in rust_edges]

    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": make_id("snapshot"),
        "generated_at": iso_now(),
        "context": {
            "domain": "rsvs-core",
            "batch_id": correlation_id,
            "input_message_id": correlation_id,
            "policy_version": SCHEMA_VERSION,
            "language_code": lang_code,
        },
        "nodes": nodes,
        "edges": edges,
    }
