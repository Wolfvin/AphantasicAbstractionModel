#!/usr/bin/env python3
"""RSVS backend bridge server.

Mode-aware HTTP bridge for agent/frontend workflows.
Supports: ingest, appraise, relate.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

API_VERSION = "v1"
SCHEMA_VERSION = "v4.2"
VALID_MODES = {"ingest", "appraise", "relate"}
VALID_VIEWS = {"compact", "detail"}
VALID_STATUSES = {"new", "candidate", "stable", "deprecated", "quarantine"}
PROMOTION_THRESHOLD = 0.75
DEMOTION_THRESHOLD = 0.60
QUARANTINE_FLIP_BUDGET = 3
MAX_CONFIDENCE_DELTA = 0.12
SHORT_WINDOW_BATCH = 3
LONG_WINDOW_BATCH = 10
SEED_LABELS = (
    "exists",
    "entity",
    "relation",
    "state",
    "change",
    "time",
    "space",
    "cause",
    "effect",
    "context",
    "signal",
    "pattern",
    "memory",
    "attention",
    "value",
    "agent",
    "goal",
    "risk",
    "trust",
    "identity",
    "language",
    "meaning",
    "action",
    "feedback",
)
SOURCE_TRUST = {
    "trusted_seed": 1.0,
    "governance_manual": 0.95,
    "verified_runtime": 0.8,
    "user_input": 0.65,
    "unknown_external": 0.4,
}
VALID_LANG_CODES = {"id", "en", "zh", "fr", "python", "javascript"}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id(prefix: str) -> str:
    return f"{prefix}_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{random.randint(1000, 9999)}"


@dataclass
class BridgeConfig:
    host: str
    port: int
    atom_dir: Path


CONFIG = BridgeConfig(
    host=os.environ.get("RSVS_BRIDGE_HOST", "127.0.0.1"),
    port=int(os.environ.get("RSVS_BRIDGE_PORT", "8787")),
    atom_dir=Path(
        os.environ.get(
            "RSVS_ATOM_OUTPUT_DIR",
            "/home/raymond/workspace/projets/skills_and_mcp/RSVS/atom",
        )
    ),
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


def _latest_file(pattern: str) -> Path | None:
    files = sorted(CONFIG.atom_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_events_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")


def _normalize_view(value: Any) -> str:
    view = str(value or "compact").strip().lower()
    if view not in VALID_VIEWS:
        raise ValueError(f"invalid_view:{view}")
    return view


def _normalize_lang(value: Any) -> str:
    lang = str(value or "id").strip().lower()
    if lang not in VALID_LANG_CODES:
        raise ValueError("invalid_language_code")
    return lang


def _is_sense_centric_node(node: dict[str, Any]) -> bool:
    sense_state = node.get("sense_state")
    if not isinstance(sense_state, dict):
        return False
    if "semantic_index" in sense_state:
        return True
    senses = sense_state.get("senses")
    if isinstance(senses, list):
        for s in senses:
            if isinstance(s, dict) and ("layer_1" in s or "layer_2" in s):
                return True
    return False


def _validate_semantic_node(node: dict[str, Any], node_ids: set[int]) -> None:
    kind = node.get("kind")
    if kind != "node":
        raise ValueError("schema_version_mismatch:deprecated_kind")
    if _is_sense_centric_node(node):
        raise ValueError("schema_model_mismatch_sense_centric")

    surface_label = str(node.get("surface_label") or "").strip()
    if "@" not in surface_label:
        raise ValueError("schema_validation_error:surface_label_locale_required")

    semantic = node.get("semantic")
    if not isinstance(semantic, dict):
        raise ValueError("schema_validation_error:semantic_required")

    state = semantic.get("compression_state")
    if state not in {"raw", "compressed"}:
        raise ValueError("schema_validation_error:invalid_compression_state")

    derived = semantic.get("derived_from_node_ids")
    if not isinstance(derived, list):
        raise ValueError("schema_validation_error:derived_from_node_ids_required")

    if len(set(derived)) != len(derived):
        raise ValueError("schema_validation_error:derived_from_node_ids_duplicate")

    node_id = node.get("id")
    if state == "compressed":
        reason = str(semantic.get("compression_reason") or "").strip()
        if not reason:
            raise ValueError("schema_validation_error:compression_reason_required")
        if not derived:
            raise ValueError("schema_validation_error:compressed_requires_derived")

    for dep_id in derived:
        if not isinstance(dep_id, int):
            raise ValueError("schema_validation_error:derived_id_must_be_int")
        if dep_id == node_id:
            raise ValueError("schema_validation_error:self_derived_forbidden")
        if dep_id not in node_ids:
            raise ValueError("schema_validation_error:derived_node_missing")

    status = node.get("status")
    if status not in VALID_STATUSES:
        raise ValueError("schema_validation_error:invalid_status")

    is_seed = bool(node.get("is_seed", False))
    if is_seed:
        if not bool(node.get("is_locked", False)):
            raise ValueError("invariant_violation:seed_requires_lock")
        if int(node.get("tier", 0)) != 1:
            raise ValueError("invariant_violation:seed_tier")
        if float(node.get("confidence", 0.0)) != 1.0:
            raise ValueError("invariant_violation:seed_confidence")
        if status != "stable":
            raise ValueError("invariant_violation:seed_status")


def _validate_snapshot_contract(snapshot: dict[str, Any]) -> None:
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version_mismatch")
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("schema_validation_error:nodes_required")
    node_ids = {int(n["id"]) for n in nodes if isinstance(n, dict) and isinstance(n.get("id"), int)}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("schema_validation_error:node_object_required")
        _validate_semantic_node(node, node_ids)


def _project_node(node: dict[str, Any], view: str, node_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
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


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _norm_text(text: str) -> str:
    return " ".join(_tokenize(text))


def _fingerprint(text: str) -> str:
    return hashlib.sha1(_norm_text(text).encode("utf-8")).hexdigest()


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _build_base_node(
    node_id: int,
    label: str,
    surface_label: str,
    source_batch_id: str,
    source_type: str = "learned",
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "surface_label": surface_label,
        "language_links": [],
        "kind": "node",
        "tier": 3,
        "confidence": 0.25,
        "status": "new",
        "is_seed": False,
        "is_locked": False,
        "semantic": {
            "compression_state": "raw",
            "derived_from_node_ids": [],
            "compression_reason": "base_ingest_signal",
        },
        "policy_meta": {
            "policy_version": SCHEMA_VERSION,
            "governance_score": 0.25,
            "candidate_evidence_pool": 0.0,
            "status_flip_count": 0,
            "seen_fingerprints": [],
            "short_window_hits": 0,
            "long_window_hits": 0,
            "last_seen_at": iso_now(),
        },
        "render": {
            "position": {
                "x": round(random.uniform(-8, 8), 2),
                "y": round(random.uniform(-5, 5), 2),
                "z": round(random.uniform(-8, 8), 2),
            },
            "size": round(random.uniform(0.6, 1.4), 2),
            "color": random.choice(["#00E5FF", "#B388FF", "#69F0AE", "#FFB74D"]),
            "glow": round(random.uniform(0.35, 0.85), 2),
        },
        "provenance": {
            "source_batch_id": source_batch_id,
            "source_domain": "user_input",
            "source_type": source_type,
        },
    }


def _apply_seed_rule(node: dict[str, Any], batch_id: str) -> None:
    node["is_seed"] = True
    node["is_locked"] = True
    node["tier"] = 1
    node["confidence"] = 1.0
    node["status"] = "stable"
    node.setdefault("policy_meta", {})
    node["policy_meta"].update(
        {
            "policy_version": SCHEMA_VERSION,
            "governance_score": 1.0,
            "candidate_evidence_pool": 0.0,
            "status_flip_count": 0,
            "seed_registry": True,
            "last_seen_at": iso_now(),
        }
    )
    node.setdefault("provenance", {})
    node["provenance"].update(
        {
            "source_batch_id": batch_id,
            "source_domain": "core_seed",
            "source_type": "bootstrap",
        }
    )


def _status_transition(current_status: str, score: float, contradiction_penalty: float) -> str:
    if current_status == "quarantine":
        return "candidate" if score >= DEMOTION_THRESHOLD and contradiction_penalty < 0.2 else "quarantine"
    if score >= PROMOTION_THRESHOLD:
        if current_status in {"new", "candidate"}:
            return "stable"
        return current_status
    if current_status == "stable" and score < DEMOTION_THRESHOLD:
        return "candidate"
    if current_status == "new" and score >= 0.4:
        return "candidate"
    return current_status


def _tier_from_score(score: float, is_seed: bool) -> int:
    if is_seed:
        return 1
    if score >= 0.8:
        return 1
    if score >= 0.4:
        return 2
    return 3


def _score_evidence(
    node: dict[str, Any],
    token_count: int,
    source_domain: str,
    fingerprint: str,
    contradiction_penalty: float,
) -> tuple[float, bool]:
    policy_meta = node.setdefault("policy_meta", {})
    seen: list[str] = list(policy_meta.get("seen_fingerprints") or [])
    replay = fingerprint in seen
    strength = _clamp(token_count / 4.0)
    if replay:
        strength *= 0.1
    trust = SOURCE_TRUST.get(source_domain, SOURCE_TRUST["unknown_external"])
    recency = 1.0
    last_seen = _parse_iso(policy_meta.get("last_seen_at"))
    if last_seen is not None:
        hours = max(0.0, (datetime.now(timezone.utc) - last_seen).total_seconds() / 3600.0)
        recency = _clamp(1.0 - (hours / (24.0 * LONG_WINDOW_BATCH)))
    raw_score = (0.4 * strength) + (0.3 * trust) + (0.2 * recency) + (0.1 * (1.0 - contradiction_penalty))
    prev_score = float(policy_meta.get("governance_score", node.get("confidence", 0.25)))
    target = _clamp((0.65 * prev_score) + (0.35 * raw_score))
    delta = max(-MAX_CONFIDENCE_DELTA, min(MAX_CONFIDENCE_DELTA, target - prev_score))
    new_score = _clamp(prev_score + delta)
    if not replay:
        seen.append(fingerprint)
    policy_meta["seen_fingerprints"] = seen[-LONG_WINDOW_BATCH:]
    policy_meta["last_seen_at"] = iso_now()
    policy_meta["short_window_hits"] = int(policy_meta.get("short_window_hits", 0)) + 1
    policy_meta["long_window_hits"] = int(policy_meta.get("long_window_hits", 0)) + 1
    return new_score, replay


def _evaluate_node_policy(
    node: dict[str, Any],
    token_count: int,
    source_domain: str,
    fingerprint: str,
    contradiction_penalty: float,
    batch_id: str,
) -> dict[str, Any]:
    if bool(node.get("is_seed", False)) or str(node.get("label", "")) in SEED_LABELS:
        _apply_seed_rule(node, batch_id)
        return {"rule": "SeedRule", "promoted": True, "replay": False}

    previous_status = str(node.get("status", "new"))
    score, replay = _score_evidence(node, token_count, source_domain, fingerprint, contradiction_penalty)
    target_status = _status_transition(previous_status, score, contradiction_penalty)

    policy_meta = node.setdefault("policy_meta", {})
    if target_status != previous_status:
        policy_meta["status_flip_count"] = int(policy_meta.get("status_flip_count", 0)) + 1
    if int(policy_meta.get("status_flip_count", 0)) >= QUARANTINE_FLIP_BUDGET:
        target_status = "quarantine"

    node["status"] = target_status
    node["tier"] = _tier_from_score(score, False)
    node["confidence"] = round(score, 3)
    node["is_seed"] = False
    node["is_locked"] = bool(node.get("is_locked", False))
    if node["is_locked"] and source_domain != "governance_manual":
        node["is_locked"] = True
    policy_meta["governance_score"] = round(score, 3)
    policy_meta["candidate_evidence_pool"] = round(
        float(policy_meta.get("candidate_evidence_pool", 0.0)) + (0.0 if score >= PROMOTION_THRESHOLD else score),
        3,
    )
    policy_meta["policy_version"] = SCHEMA_VERSION
    promoted = previous_status != "stable" and target_status == "stable"
    return {"rule": "PromotionRule", "promoted": promoted, "replay": replay}


def _ensure_seed_nodes(
    nodes_by_label: dict[str, dict[str, Any]],
    next_id: int,
    batch_id: str,
    lang_code: str,
) -> int:
    for label in SEED_LABELS:
        surface = f"{label}@{lang_code}"
        if surface not in nodes_by_label:
            node = _build_base_node(next_id, label, surface, batch_id, source_type="bootstrap")
            nodes_by_label[surface] = node
            next_id += 1
        _apply_seed_rule(nodes_by_label[surface], batch_id)
    return next_id


def _build_snapshot_and_events(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tokens = _tokenize(text)
    unique = list(dict.fromkeys(tokens))[:24]
    if not unique:
        unique = ["input", "signal", "node"]
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    events: list[dict[str, Any]] = []
    batch_id = make_id("ingest")
    source_domain = "user_input"
    lang_code = _normalize_lang((options or {}).get("language"))
    contradiction_penalty = 0.25 if any(t in {"not", "never", "no"} for t in tokens) else 0.0
    fp = _fingerprint(text)

    previous_bundle = _read_latest_ingest_bundle()
    nodes_by_label: dict[str, dict[str, Any]] = {}
    next_id = 1001
    prev_edges: list[dict[str, Any]] = []
    if previous_bundle is not None:
        prev_nodes = previous_bundle["snapshot"].get("nodes", [])
        prev_edges = previous_bundle["snapshot"].get("edges", [])
        for n in prev_nodes:
            if isinstance(n, dict) and isinstance(n.get("surface_label"), str):
                nodes_by_label[str(n["surface_label"])] = n
                next_id = max(next_id, int(n.get("id", 1000)) + 1)

    next_id = _ensure_seed_nodes(nodes_by_label, next_id, batch_id, lang_code)

    touched_ids: list[int] = []
    promoted_labels: list[str] = []
    replay_count = 0
    created_count = 0

    for label in unique:
        surface_label = f"{label}@{lang_code}"
        node = nodes_by_label.get(surface_label)
        if node is None:
            node = _build_base_node(next_id, label, surface_label, batch_id)
            nodes_by_label[surface_label] = node
            next_id += 1
            created_count += 1
            events.append(
                {
                    "event_id": make_id("evt"),
                    "timestamp": iso_now(),
                    "correlation_id": correlation_id,
                    "event_type": "node_created",
                    "payload": {"node": node},
                    "animation_hint": {
                        "priority": "normal",
                        "focus_node_id": node["id"],
                        "burst_group": batch_id,
                    },
                }
            )

        policy_result = _evaluate_node_policy(
            node=node,
            token_count=counts.get(label, 1),
            source_domain=source_domain,
            fingerprint=fp,
            contradiction_penalty=contradiction_penalty,
            batch_id=batch_id,
        )
        if policy_result.get("replay"):
            replay_count += 1
        if policy_result.get("promoted"):
            promoted_labels.append(label)

        node.setdefault("provenance", {})
        node["provenance"]["source_batch_id"] = batch_id
        node["provenance"]["source_domain"] = source_domain
        node["provenance"]["source_type"] = "learned"
        touched_ids.append(int(node["id"]))
        events.append(
            {
                "event_id": make_id("evt"),
                "timestamp": iso_now(),
                "correlation_id": correlation_id,
                "event_type": "policy_decision",
                "payload": {
                    "node_id": node["id"],
                    "label": label,
                    "status": node.get("status"),
                    "tier": node.get("tier"),
                    "confidence": node.get("confidence"),
                    "rule": policy_result.get("rule"),
                    "replay": bool(policy_result.get("replay", False)),
                },
                "animation_hint": {
                    "priority": "low",
                    "focus_node_id": node["id"],
                    "burst_group": batch_id,
                },
            }
        )

        # Build multilingual relation candidate/final for same surface across locales.
        base = str(node.get("label"))
        for other in nodes_by_label.values():
            if int(other.get("id", -1)) == int(node["id"]):
                continue
            if str(other.get("label")) == base and str(other.get("surface_label")) != surface_label:
                existing = {(lnk.get("type"), int(lnk.get("target_id", -1))) for lnk in node.get("language_links", [])}
                rel = ("same_as", int(other["id"]))
                if rel not in existing:
                    node.setdefault("language_links", []).append({"type": "same_as", "target_id": int(other["id"])})

    edges_by_id: dict[str, dict[str, Any]] = {
        str(e.get("id")): e for e in prev_edges if isinstance(e, dict) and isinstance(e.get("id"), str)
    }
    for i in range(len(touched_ids) - 1):
        source = touched_ids[i]
        target = touched_ids[i + 1]
        edge_id = f"{source}->{target}"
        if edge_id not in edges_by_id:
            edge = {
                "id": edge_id,
                "source": source,
                "target": target,
                "direction": "undirected",
                "weight": round(random.uniform(0.25, 0.9), 2),
                "source_type": "learned",
                "status": "new",
                "render": {
                    "thickness": round(random.uniform(0.4, 1.4), 2),
                    "color": "#80D8FF",
                    "opacity": round(random.uniform(0.4, 0.8), 2),
                    "pulse": round(random.uniform(0.1, 0.5), 2),
                },
            }
            edges_by_id[edge_id] = edge
            events.append(
                {
                    "event_id": make_id("evt"),
                    "timestamp": iso_now(),
                    "correlation_id": correlation_id,
                    "event_type": "edge_created",
                    "payload": {"edge": edge},
                    "animation_hint": {
                        "priority": "low",
                        "focus_node_id": source,
                        "burst_group": batch_id,
                    },
                }
            )

    nodes = sorted(nodes_by_label.values(), key=lambda n: int(n.get("id", 0)))
    edges = list(edges_by_id.values())
    _validate_snapshot_contract({"schema_version": SCHEMA_VERSION, "nodes": nodes})

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": make_id("snapshot"),
        "generated_at": iso_now(),
        "context": {
            "domain": "user-input",
            "batch_id": batch_id,
            "input_message_id": correlation_id,
            "policy_version": SCHEMA_VERSION,
            "language_code": lang_code,
        },
        "nodes": nodes,
        "edges": edges,
    }

    promoted = ", ".join(
        [f"{n['label']} (T{n['tier']}, c={float(n['confidence']):.2f})" for n in nodes if n["label"] in promoted_labels][:6]
    ) or "none"

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": (
                f"Ingesting batch {batch_id} — {len(tokens)} tokens processed, "
                f"{created_count} nodes created, {len(promoted_labels)} promoted, "
                f"{replay_count} replay events, {len(edges)} edges."
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

    stats = {
        "batch_id": batch_id,
        "token_count": len(tokens),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "created_count": created_count,
        "promoted_count": len(promoted_labels),
        "replay_count": replay_count,
    }

    return snapshot, events, messages, stats


def _write_ingest_artifacts(snapshot: dict[str, Any], events: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, str]:
    CONFIG.atom_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    snapshot_path = CONFIG.atom_dir / f"snapshot-{stamp}.json"
    events_path = CONFIG.atom_dir / f"events-{stamp}.jsonl"
    report_path = CONFIG.atom_dir / f"report-{stamp}.json"

    _write_json(snapshot_path, snapshot)
    _write_events_jsonl(events_path, events)
    _write_json(report_path, {**stats, "created_at": iso_now(), "mode": "ingest"})

    return {
        "snapshot": str(snapshot_path),
        "events": str(events_path),
        "report": str(report_path),
    }


def _read_latest_ingest_bundle() -> dict[str, Any] | None:
    if not CONFIG.atom_dir.exists():
        return None

    snapshot_path = _latest_file("snapshot-*.json")
    if snapshot_path is None:
        return None

    events_path = _latest_file("events-*.jsonl")
    report_path = _latest_file("report-*.json")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    _validate_snapshot_contract(snapshot)
    events: list[dict[str, Any]] = []
    if events_path and events_path.exists():
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

    report = None
    if report_path and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    messages: list[dict[str, Any]] = []
    if report:
        messages.append(
            {
                "id": make_id("msg"),
                "type": "system_ingest_status",
                "content": f"Restored latest batch {report.get('batch_id', '-')}: {report.get('node_count', 0)} nodes, {report.get('edge_count', 0)} edges.",
                "timestamp": iso_now(),
            }
        )

    return {
        "snapshot": snapshot,
        "events": events,
        "messages": messages,
        "files": {
            "snapshot": str(snapshot_path),
            "events": str(events_path) if events_path else None,
            "report": str(report_path) if report_path else None,
        },
    }


def _score_overlap(input_tokens: list[str], node_label: str) -> float:
    lbl = node_label.lower()
    hit = 0
    for t in input_tokens:
        if t == lbl or t in lbl or lbl in t:
            hit += 1
    if not input_tokens:
        return 0.0
    return round(min(1.0, hit / len(input_tokens)), 3)


def _run_appraise(text: str, correlation_id: str, options: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    latest = _read_latest_ingest_bundle()
    view = _normalize_view((options or {}).get("view"))
    input_tokens = _tokenize(text)
    nodes = latest["snapshot"].get("nodes", []) if latest else []
    node_index = {int(n["id"]): n for n in nodes if isinstance(n.get("id"), int)}

    scored = []
    for n in nodes:
        score = _score_overlap(input_tokens, str(n.get("label", "")))
        scored.append((n, score))

    support = [
        {
            "id": n["id"],
            "label": n["label"],
            "score": s,
            "compression_state": (n.get("semantic") or {}).get("compression_state"),
            "derived_from_node_ids": (n.get("semantic") or {}).get("derived_from_node_ids", []),
        }
        for n, s in sorted(scored, key=lambda x: x[1], reverse=True)
        if s > 0
    ][:5]

    conflict = [
        {
            "id": n["id"],
            "label": n["label"],
            "score": round(1.0 - s, 3),
            "compression_state": (n.get("semantic") or {}).get("compression_state"),
            "derived_from_node_ids": (n.get("semantic") or {}).get("derived_from_node_ids", []),
        }
        for n, s in sorted(scored, key=lambda x: x[1])
        if s <= 0.05
    ][:5]

    if not nodes:
        agree = 50
    elif not support:
        agree = 30
    else:
        avg = sum(x["score"] for x in support) / len(support)
        agree = int(max(5, min(95, round(avg * 100))))

    disagree = 100 - agree
    confidence = round(min(1.0, 0.4 + (len(nodes) / 50.0) + (abs(agree - disagree) / 200.0)), 3)

    if agree >= 70:
        verdict = "agree"
        rationale = "Input mostly aligns with existing graph evidence."
    elif agree <= 40:
        verdict = "disagree"
        rationale = "Input weakly supported by current graph; conflicting concepts dominate."
    else:
        verdict = "mixed"
        rationale = "Input has partial alignment with current graph evidence."

    result = {
        "view": view,
        "stance": {"agree": agree, "disagree": disagree},
        "confidence": confidence,
        "verdict": verdict,
        "rationale": rationale,
        "evidence": {
            "support_nodes": support,
            "conflict_nodes": conflict,
            "paths": [],
        },
    }

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": f"Appraise verdict: {verdict} ({agree}% agree / {disagree}% disagree).",
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
    CONFIG.atom_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return result, messages, {"appraise": str(path)}


def _run_relate(text: str, correlation_id: str, options: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    latest = _read_latest_ingest_bundle()
    input_tokens = _tokenize(text)
    top_k = int((options or {}).get("top_k", 10)) if options else 10
    view = _normalize_view((options or {}).get("view"))
    top_k = max(1, min(50, top_k))

    nodes = latest["snapshot"].get("nodes", []) if latest else []
    edges = latest["snapshot"].get("edges", []) if latest else []
    node_index = {int(n["id"]): n for n in nodes if isinstance(n.get("id"), int)}

    related_nodes = []
    for n in nodes:
        score = _score_overlap(input_tokens, str(n.get("label", "")))
        if score > 0:
            related_nodes.append(
                _project_node(
                    {
                        "id": n.get("id"),
                        "label": n.get("label"),
                        "score": score,
                        "tier": n.get("tier"),
                        "kind": n.get("kind"),
                        "semantic": n.get("semantic"),
                    },
                    view,
                    node_index,
                )
            )

    related_nodes = sorted(related_nodes, key=lambda x: x["score"], reverse=True)[:top_k]
    node_ids = {n["id"] for n in related_nodes}

    related_edges = []
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s in node_ids or t in node_ids:
            related_edges.append(
                {
                    "id": e.get("id"),
                    "source": s,
                    "target": t,
                    "score": round(float(e.get("weight", 0.0) or 0.0), 3),
                }
            )

    related_edges = sorted(related_edges, key=lambda x: x["score"], reverse=True)[: max(top_k, 10)]

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
            "content": f"Relate found {len(related_nodes)} nodes and {len(related_edges)} edges.",
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
    CONFIG.atom_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    return result, messages, {"relate": str(path)}


def _run_ingest(text: str, correlation_id: str, options: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    _normalize_view((options or {}).get("view"))
    snapshot, events, messages, stats = _build_snapshot_and_events(text, correlation_id, options)
    files = _write_ingest_artifacts(snapshot, events, stats)
    result = {
        "snapshot": snapshot,
        "events": events,
        "stats": stats,
    }
    return result, messages, files


def _run_mode(mode: str, text: str, correlation_id: str, options: dict[str, Any] | None) -> dict[str, Any]:
    started = time.perf_counter()
    if mode == "ingest":
        result, messages, files = _run_ingest(text, correlation_id, options)
    elif mode == "appraise":
        result, messages, files = _run_appraise(text, correlation_id, options)
    elif mode == "relate":
        result, messages, files = _run_relate(text, correlation_id, options)
    else:
        raise ValueError("invalid_mode")

    latency_ms = int((time.perf_counter() - started) * 1000)
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
        },
    }


def _read_latest_mode(mode: str) -> dict[str, Any] | None:
    if mode == "ingest":
        bundle = _read_latest_ingest_bundle()
        if bundle is None:
            return None
        # mode envelope for newer clients
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
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version_mismatch")
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


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send(
                200,
                {
                    "ok": True,
                    "service": "rsvs-bridge",
                    "timestamp": iso_now(),
                    "atom_dir": str(CONFIG.atom_dir),
                    "version": API_VERSION,
                },
            )
            return

        if parsed.path == "/latest":
            qs = parse_qs(parsed.query or "")
            mode = (qs.get("mode", ["ingest"])[0] or "ingest").strip().lower()
            try:
                _normalize_view(qs.get("view", ["compact"])[0] or "compact")
            except ValueError:
                self._send(400, {"ok": False, "error": "invalid_view"})
                return
            if mode not in VALID_MODES:
                self._send(400, {"ok": False, "error": "invalid_mode", "mode": mode})
                return

            if mode == "ingest" and "mode" not in qs:
                # Backward-compatible payload for current frontend restore.
                try:
                    legacy = _read_latest_ingest_bundle()
                except ValueError as exc:
                    self._send(409, {"ok": False, "error": str(exc)})
                    return
                if legacy is None:
                    self._send(404, {"ok": False, "error": "no_artifacts", "mode": "ingest"})
                    return
                self._send(200, {"ok": True, **legacy})
                return

            try:
                envelope = _read_latest_mode(mode)
            except ValueError as exc:
                self._send(409, {"ok": False, "error": str(exc)})
                return
            if envelope is None:
                self._send(404, {"ok": False, "error": "no_artifacts", "mode": mode})
                return
            self._send(200, envelope)
            return

        self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            data = json.loads(raw)
        except Exception:
            self._send(400, {"ok": False, "error": "invalid_json"})
            return

        if parsed.path == "/run":
            mode = (data.get("mode") or "").strip().lower()
            text = (data.get("text") or "").strip()
            correlation_id = (data.get("correlation_id") or make_id("corr")).strip()
            options = data.get("options") if isinstance(data.get("options"), dict) else {}
            incoming_schema = data.get("schema_version")
            if incoming_schema not in (None, SCHEMA_VERSION):
                self._send(
                    409,
                    {
                        "ok": False,
                        "error": "schema_version_mismatch",
                        "expected": SCHEMA_VERSION,
                        "got": incoming_schema,
                    },
                )
                return

            if mode not in VALID_MODES:
                self._send(400, {"ok": False, "error": "invalid_mode", "mode": mode})
                return
            if not text:
                self._send(400, {"ok": False, "error": "text_required", "mode": mode})
                return

            try:
                payload = _run_mode(mode, text, correlation_id, options)
                self._send(200, payload)
            except ValueError as exc:
                self._send(409, {"ok": False, "error": str(exc), "mode": mode})
            except Exception as exc:
                self._send(500, {"ok": False, "error": str(exc), "mode": mode})
            return

        if parsed.path == "/ingest":
            # Backward-compatible endpoint used by current frontend.
            text = (data.get("text") or "").strip()
            correlation_id = (data.get("correlation_id") or make_id("corr")).strip()
            if not text:
                self._send(400, {"ok": False, "error": "text_required"})
                return
            try:
                env = _run_mode("ingest", text, correlation_id, data.get("options") or {})
                self._send(
                    200,
                    {
                        "ok": True,
                        "correlation_id": env.get("correlation_id"),
                        "snapshot": env.get("result", {}).get("snapshot", {}),
                        "events": env.get("result", {}).get("events", []),
                        "messages": env.get("messages", []),
                        "files": env.get("files", {}),
                    },
                )
            except Exception as exc:
                self._send(500, {"ok": False, "error": str(exc)})
            return

        self._send(404, {"ok": False, "error": "not_found"})


def main() -> None:
    server = ThreadingHTTPServer((CONFIG.host, CONFIG.port), Handler)
    print(f"RSVS bridge listening on http://{CONFIG.host}:{CONFIG.port}")
    print(f"Atom output dir: {CONFIG.atom_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
