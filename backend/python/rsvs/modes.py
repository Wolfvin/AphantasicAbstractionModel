"""RSVS Bridge mode implementations — ingest, appraise, relate via Rust core.

Public API:
    run_mode  — unified mode dispatch for FastAPI server
    _run_mode — internal mode dispatch for bridge_server
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .artifacts import (
    _latest_file,
    _read_latest_ingest_bundle,
    _write_ingest_artifacts,
    _write_json,
)
from .config import (
    CONFIG,
    SCHEMA_VERSION,
    API_VERSION,
    iso_now,
    make_id,
)
from .conversion import (
    _build_bridge_snapshot,
    _convert_rust_edge,
    _convert_rust_event,
    _convert_rust_node,
    _project_node,
)
from .exceptions import InvalidModeError, RustCoreUnavailableError
from .protocols import RsvsCoreProtocol
from .rsvs_core import (
    _get_last_ingest_seq,
    _get_rsvs,
    _save_rsvs,
    _set_last_ingest_seq,
    is_rust_core_available,
)
from .validation import _normalize_lang, _normalize_view


__all__ = [
    "_tokenize",
    "_score_overlap",
    "_run_ingest_rust",
    "_run_appraise_rust",
    "_run_relate_rust",
    "_run_compose_rust",
    "_run_mode",
    "_read_latest_mode",
    "run_mode",
]


# ---------------------------------------------------------------------------
# Text utilities (not in Rust — kept here for relate fallback)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Extract alphanumeric tokens from text."""
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


def _score_overlap(input_tokens: list[str], node_label: str) -> float:
    """Simple overlap score — used as fallback for relate mode."""
    lbl = node_label.lower()
    hit = 0
    for t in input_tokens:
        if t == lbl or t in lbl or lbl in t:
            hit += 1
    if not input_tokens:
        return 0.0
    return round(min(1.0, hit / len(input_tokens)), 3)


# ---------------------------------------------------------------------------
# Mode: ingest (Rust core)
# ---------------------------------------------------------------------------


def _run_ingest_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Ingest via the Rust core: ingest_with_meta_v1 → snapshot_v1 → consume_events_v1."""
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for ingest mode")

    _normalize_view((options or {}).get("view"))
    lang_code = _normalize_lang((options or {}).get("language"))

    # Record seq before ingest to consume only new events
    seq_before = r.latest_seq_v1()

    # Ingest via Rust core
    domain_id = (options or {}).get("domain_id")
    if domain_id is not None:
        domain_id = int(domain_id)
    meta = r.ingest_with_meta_v1(text, domain_id)

    # Get the Rust correlation_id from the ingest metadata
    rust_correlation_id = meta.correlation_id

    # Get snapshot and convert
    snapshot_raw = json.loads(r.snapshot_v1())
    snapshot = _build_bridge_snapshot(snapshot_raw, rust_correlation_id, lang_code)

    # Get events and convert
    events_raw = json.loads(r.consume_events_v1(seq_before, 10_000))
    raw_events = events_raw.get("events", [])
    events = [_convert_rust_event(e, rust_correlation_id) for e in raw_events]

    # Update tracked seq
    _set_last_ingest_seq(r.latest_seq_v1())

    # Persist the Rust core state
    _save_rsvs()

    # Build stats from Rust metadata
    tokens = _tokenize(text)
    nodes = snapshot.get("nodes", [])
    edges = snapshot.get("edges", [])
    stats = {
        "batch_id": rust_correlation_id,
        "token_count": len(tokens),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "created_count": meta.atoms_promoted,
        "promoted_count": meta.atoms_promoted,
        "replay_count": 0,
        "sentences_processed": meta.sentences_processed,
        "sense_assigned": meta.sense_assigned,
        "sense_created": meta.sense_created,
        "confidence_updated": meta.confidence_updated,
    }

    # Write artifacts
    files = _write_ingest_artifacts(snapshot, events, stats)

    # Build messages
    promoted_labels = [
        n["label"] for n in nodes
        if not n.get("is_seed", False) and n.get("status") == "stable"
    ][:6]
    promoted = ", ".join(
        f"{n['label']} (T{n['tier']}, c={float(n['confidence']):.2f})"
        for n in nodes if n["label"] in promoted_labels
    ) or "none"

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": (
                f"Ingesting batch {rust_correlation_id} — "
                f"{meta.sentences_processed} sentences, "
                f"{meta.atoms_promoted} atoms promoted, "
                f"{len(edges)} edges (via Rust core)."
            ),
            "timestamp": iso_now(),
            "correlation_id": correlation_id,
        },
        {
            "id": make_id("msg"),
            "type": "system_promoted_nodes",
            "content": f"Promoted nodes: {promoted}",
            "timestamp": iso_now(),
            "correlation_id": correlation_id,
        },
    ]

    result = {
        "snapshot": snapshot,
        "events": events,
        "stats": stats,
    }
    return result, messages, files


# ---------------------------------------------------------------------------
# Mode: appraise (Rust core)
# ---------------------------------------------------------------------------


def _run_appraise_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Appraise via the Rust core: r.appraise(text)."""
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for appraise mode")

    view = _normalize_view((options or {}).get("view"))

    # Call Rust core appraise
    appraise_result = r.appraise(text)

    agree_pct = round(float(appraise_result.agree_pct), 1)
    disagree_pct = round(float(appraise_result.disagree_pct), 1)
    verdict_raw = appraise_result.verdict  # "consistent", "partial", "novel"

    # Map Rust verdicts to Python bridge verdicts for backward compat
    verdict_map = {
        "consistent": "agree",
        "partial": "mixed",
        "novel": "disagree",
    }
    verdict = verdict_map.get(verdict_raw, verdict_raw)

    if verdict == "agree":
        rationale = "Input mostly aligns with existing graph evidence."
    elif verdict == "disagree":
        rationale = "Input weakly supported by current graph; conflicting concepts dominate."
    else:
        rationale = "Input has partial alignment with current graph evidence."

    # Build evidence from Rust result
    evidence_tuples = appraise_result.evidence  # list of (label, confidence)
    support_nodes = [
        {"label": label, "score": round(float(conf), 3)}
        for label, conf in evidence_tuples[:5]
        if float(conf) > 0.0
    ]

    result = {
        "view": view,
        "stance": {"agree": int(agree_pct), "disagree": int(disagree_pct)},
        "confidence": round(min(1.0, 0.4 + len(support_nodes) / 50.0 + abs(agree_pct - disagree_pct) / 200.0), 3),
        "verdict": verdict,
        "rationale": rationale,
        "evidence": {
            "support_nodes": support_nodes,
            "conflict_nodes": [],
            "paths": [],
        },
    }

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": f"Appraise verdict: {verdict} ({int(agree_pct)}% agree / {int(disagree_pct)}% disagree) [Rust core].",
            "timestamp": iso_now(),
            "correlation_id": correlation_id,
        }
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CONFIG.atom_dir / f"appraise-{stamp}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "appraise",
        "timestamp": iso_now(),
        "correlation_id": correlation_id,
        "input": text,
        "result": result,
    }
    _write_json(path, payload)
    return result, messages, {"appraise": str(path)}


# ---------------------------------------------------------------------------
# Mode: relate (Rust core)
# ---------------------------------------------------------------------------


def _run_relate_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Relate via the Rust core: r.relate(concept)."""
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for relate mode")

    view = _normalize_view((options or {}).get("view"))
    top_k = max(1, min(50, int((options or {}).get("top_k", 10))))

    # The Rust core's relate() takes a single concept label.
    # We try each token from the input; fall back to _score_overlap.
    input_tokens = _tokenize(text)

    # Try to use Rust relate for the first known token
    relate_result = None
    used_token = None
    for token in input_tokens:
        try:
            relate_result = r.relate(token)
            if relate_result is not None:
                used_token = token
                break
        except Exception:
            continue

    if relate_result is not None:
        # Convert Rust RelateResult to Python format
        snapshot_raw = json.loads(r.snapshot_v1())
        node_map = {n["id"]: n for n in snapshot_raw.get("nodes", [])}

        related_nodes = []
        for node_id, score in relate_result.related_nodes[:top_k]:
            rn = node_map.get(node_id, {})
            py_node = _convert_rust_node(rn, correlation_id) if rn else {
                "id": node_id, "label": f"#{node_id}", "score": round(float(score), 3),
            }
            py_node["score"] = round(float(score), 3)
            related_nodes.append(_project_node(py_node, view, {}))

        related_edges = []
        for from_id, to_id, weight in relate_result.related_edges[:max(top_k, 10)]:
            related_edges.append({
                "id": f"{from_id}->{to_id}",
                "source": from_id,
                "target": to_id,
                "score": round(float(weight), 3),
            })

        result = {
            "view": view,
            "query_terms": input_tokens,
            "related_nodes": related_nodes,
            "related_edges": related_edges,
            "clusters": [],
        }

        messages = [
            {
                "id": make_id("msg"),
                "type": "system_ingest_status",
                "content": (
                    f"Relate found {len(related_nodes)} nodes and "
                    f"{len(related_edges)} edges [Rust core, token='{used_token}']."
                ),
                "timestamp": iso_now(),
                "correlation_id": correlation_id,
            }
        ]
    else:
        # Fallback: use Python _score_overlap on the current Rust snapshot
        snapshot_raw = json.loads(r.snapshot_v1())
        snapshot = _build_bridge_snapshot(snapshot_raw, correlation_id)
        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])
        node_index = {int(n["id"]): n for n in nodes if isinstance(n.get("id"), int)}

        related_nodes = []
        for n in nodes:
            score = _score_overlap(input_tokens, str(n.get("label", "")))
            if score > 0:
                n_copy = dict(n)
                n_copy["score"] = score
                related_nodes.append(_project_node(n_copy, view, node_index))

        related_nodes = sorted(related_nodes, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        node_ids = {n["id"] for n in related_nodes}

        related_edges = []
        for e in edges:
            s = e.get("source")
            t = e.get("target")
            if s in node_ids or t in node_ids:
                related_edges.append({
                    "id": e.get("id"),
                    "source": s,
                    "target": t,
                    "score": round(float(e.get("weight", 0.0) or 0.0), 3),
                })

        related_edges = sorted(related_edges, key=lambda x: x.get("score", 0), reverse=True)[:max(top_k, 10)]

        result = {
            "view": view,
            "query_terms": input_tokens,
            "related_nodes": related_nodes,
            "related_edges": related_edges,
            "clusters": [],
        }

        messages = [
            {
                "id": make_id("msg"),
                "type": "system_ingest_status",
                "content": (
                    f"Relate found {len(related_nodes)} nodes and "
                    f"{len(related_edges)} edges [Python fallback]."
                ),
                "timestamp": iso_now(),
                "correlation_id": correlation_id,
            }
        ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CONFIG.atom_dir / f"relate-{stamp}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "relate",
        "timestamp": iso_now(),
        "correlation_id": correlation_id,
        "input": text,
        "result": result,
    }
    _write_json(path, payload)
    return result, messages, {"relate": str(path)}


# ---------------------------------------------------------------------------
# Mode: compose (Rust core)
# ---------------------------------------------------------------------------


def _run_compose_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Compose via the Rust core: r.compose(label, atom_ids, lang)."""
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for compose mode")

    label = text  # In compose mode, text carries the label
    atom_ids = (options or {}).get("atom_ids", [])
    lang = (options or {}).get("lang")

    if not atom_ids:
        raise ValueError("atom_ids must be a non-empty list of integer node IDs")

    node_id = r.compose(label, atom_ids, lang)

    # Get snapshot and convert
    snapshot_raw = json.loads(r.snapshot_v1())
    snapshot = _build_bridge_snapshot(snapshot_raw, correlation_id)

    # Get events and convert
    events_raw = json.loads(r.consume_events_v1())
    raw_events = events_raw.get("events", [])
    events = [_convert_rust_event(e, correlation_id) for e in raw_events]

    # Persist the Rust core state
    _save_rsvs()

    result = {
        "mode": "compose",
        "node_id": node_id,
        "label": label,
        "atom_ids": atom_ids,
        "snapshot": snapshot,
        "events": events,
    }

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": (
                f"Composed node '{label}' (id={node_id}) from "
                f"{len(atom_ids)} atoms [Rust core]."
            ),
            "timestamp": iso_now(),
            "correlation_id": correlation_id,
        }
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CONFIG.atom_dir / f"compose-{stamp}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "compose",
        "timestamp": iso_now(),
        "correlation_id": correlation_id,
        "input": label,
        "atom_ids": atom_ids,
        "lang": lang,
        "result": result,
    }
    _write_json(path, payload)
    return result, messages, {"compose": str(path)}


# ---------------------------------------------------------------------------
# Unified mode dispatch
# ---------------------------------------------------------------------------

_MODE_HANDLERS = {
    "ingest": _run_ingest_rust,
    "appraise": _run_appraise_rust,
    "relate": _run_relate_rust,
    "compose": _run_compose_rust,
}


def _run_mode(mode: str, text: str, correlation_id: str, options: dict[str, Any] | None) -> dict[str, Any]:
    """Execute the given mode and return a structured envelope."""
    import time as _time

    started = _time.perf_counter()

    handler = _MODE_HANDLERS.get(mode)
    if handler is None:
        raise InvalidModeError(f"invalid_mode:{mode}")

    result, messages, files = handler(text, correlation_id, options)

    latency_ms = int((_time.perf_counter() - started) * 1000)
    return {
        "ok": True,
        "mode": mode,
        "correlation_id": correlation_id,
        "timestamp": iso_now(),
        "result": result,
        "messages": messages,
        "files": files,
        "meta": {
            "version": API_VERSION,
            "schema_version": SCHEMA_VERSION,
            "atom_dir": str(CONFIG.atom_dir),
            "latency_ms": latency_ms,
            "backend": "rust" if is_rust_core_available() else "unavailable",
        },
    }



def run_mode(
    rsvs: RsvsCoreProtocol,
    mode: str,
    *,
    text: str,
    target: str | None = None,
    source: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Public unified mode dispatch for the FastAPI server.

    This is the primary entry point for the async FastAPI layer.
    It accepts a pre-obtained Rsvs instance and keyword arguments.
    """
    from .config import make_id
    correlation_id = make_id("corr")
    options: dict[str, Any] = {}
    if target is not None:
        options["target"] = target
    if source is not None:
        options["source"] = source
    if top_k != 10:
        options["top_k"] = top_k
    return _run_mode(mode, text, correlation_id, options)


def _read_latest_mode(mode: str) -> dict[str, Any] | None:
    """Read the latest artifact for a given mode."""
    if mode == "ingest":
        bundle = _read_latest_ingest_bundle()
        if bundle is None:
            return None
        return {
            "ok": True,
            "mode": "ingest",
            "timestamp": iso_now(),
            "result": {
                "snapshot": bundle["snapshot"],
                "events": bundle["events"],
                "stats": {
                    "token_count": None,
                    "node_count": len(bundle["snapshot"].get("nodes", [])),
                    "edge_count": len(bundle["snapshot"].get("edges", [])),
                    "batch_id": bundle["snapshot"].get("context", {}).get("batch_id"),
                },
            },
            "messages": bundle.get("messages", []),
            "files": bundle.get("files", {}),
            "meta": {
                "version": API_VERSION,
                "schema_version": SCHEMA_VERSION,
                "atom_dir": str(CONFIG.atom_dir),
                "latency_ms": 0,
            },
        }

    path = _latest_file(f"{mode}-*.json")
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "mode": mode,
        "timestamp": payload.get("timestamp", iso_now()),
        "correlation_id": payload.get("correlation_id"),
        "result": payload.get("result", {}),
        "messages": [],
        "files": {mode: str(path)},
        "meta": {
            "version": API_VERSION,
            "schema_version": SCHEMA_VERSION,
            "atom_dir": str(CONFIG.atom_dir),
            "latency_ms": 0,
        },
    }
