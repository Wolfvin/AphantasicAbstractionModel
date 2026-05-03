"""RSVS Bridge mode implementations — ingest, appraise, relate via Rust core.

Public API:
    run_mode  — unified mode dispatch for FastAPI server
    _run_mode — internal mode dispatch for bridge_server

Updated for RSVS v6.0 compositional architecture + grounding.
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
    "_run_structural_similarity_rust",
    "_run_substitution_analysis_rust",
    "_run_grounding_info_rust",
    "_run_context_query_rust",
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

    # Build stats from Rust metadata (v5.0: include compositions_induced)
    tokens = _tokenize(text)
    nodes = snapshot.get("nodes", [])
    edges = snapshot.get("edges", [])

    # Extract compositions_induced from PyIngestStats if available
    compositions_induced = getattr(meta, "compositions_induced", 0)

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
        "compositions_induced": compositions_induced,
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
                f"{compositions_induced} compositions induced, "
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
    """Relate via the Rust core: r.relate(concept).

    v5.0: PyRelateResult now includes structural_relations.
    """
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

        # v5.0: Extract structural_relations from PyRelateResult
        structural_relations = getattr(relate_result, "structural_relations", [])

        result = {
            "view": view,
            "query_terms": input_tokens,
            "related_nodes": related_nodes,
            "related_edges": related_edges,
            "clusters": [],
            "structural_relations": structural_relations,
        }

        messages = [
            {
                "id": make_id("msg"),
                "type": "system_ingest_status",
                "content": (
                    f"Relate found {len(related_nodes)} nodes, "
                    f"{len(related_edges)} edges, and "
                    f"{len(structural_relations)} structural relations "
                    f"[Rust core, token='{used_token}']."
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
            "structural_relations": [],
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
# Mode: compose (Rust core) — v5.0 compositional architecture
# ---------------------------------------------------------------------------


def _run_compose_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Compose via the Rust core.

    v5.0 supports two calling patterns:
      - compositions: list of [label, sense_id] pairs → r.compose(label, compositions, lang)
      - atom_ids: list of integer node IDs → r.compose_from_ids(label, atom_ids, lang)
    """
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for compose mode")

    label = text  # In compose mode, text carries the label
    options = options or {}

    lang = options.get("lang")
    compositions = options.get("compositions")  # list of [label, sense_id] pairs
    atom_ids = options.get("atom_ids")  # list of int node IDs (backward compat)

    if compositions:
        # v5.0: New compose with (label, sense_id) pairs
        compositions_tuples = [(c[0], int(c[1])) for c in compositions]
        node_id = r.compose(label, compositions_tuples, lang)
        composed_from = compositions
        composed_type = "compositions"
    elif atom_ids:
        # Backward compat: compose_from_ids with sense_id=0
        node_id = r.compose_from_ids(label, atom_ids, lang)
        composed_from = atom_ids
        composed_type = "atom_ids"
    else:
        raise ValueError(
            "Either 'compositions' (list of [label, sense_id] pairs) or "
            "'atom_ids' (list of integer node IDs) must be provided"
        )

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
        composed_type: composed_from,
        "snapshot": snapshot,
        "events": events,
    }

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": (
                f"Composed node '{label}' (id={node_id}) from "
                f"{len(composed_from)} {composed_type} [Rust core]."
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
        composed_type: composed_from,
        "lang": lang,
        "result": result,
    }
    _write_json(path, payload)
    return result, messages, {"compose": str(path)}


# ---------------------------------------------------------------------------
# Mode: structural_similarity (Rust core) — v5.0
# ---------------------------------------------------------------------------


def _run_structural_similarity_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Structural similarity via the Rust core: r.structural_similarity(a, b).

    Returns PyStructuralSimResult with:
        sense_idx_a, sense_idx_b, structural_similarity,
        shared_compositions, only_a_compositions, only_b_compositions,
        layer_a, layer_b
    """
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for structural_similarity mode")

    options = options or {}
    label_a = options.get("a", text)
    label_b = options.get("b", "")

    if not label_b:
        raise ValueError("Parameter 'b' is required for structural similarity")

    sim_result = r.structural_similarity(label_a, label_b)

    # Extract fields from PyStructuralSimResult
    result = {
        "a": label_a,
        "b": label_b,
        "sense_idx_a": getattr(sim_result, "sense_idx_a", None),
        "sense_idx_b": getattr(sim_result, "sense_idx_b", None),
        "structural_similarity": getattr(sim_result, "structural_similarity", 0.0),
        "shared_compositions": list(getattr(sim_result, "shared_compositions", [])),
        "only_a_compositions": list(getattr(sim_result, "only_a_compositions", [])),
        "only_b_compositions": list(getattr(sim_result, "only_b_compositions", [])),
        "layer_a": getattr(sim_result, "layer_a", None),
        "layer_b": getattr(sim_result, "layer_b", None),
    }

    shared = len(result["shared_compositions"])
    only_a = len(result["only_a_compositions"])
    only_b = len(result["only_b_compositions"])

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": (
                f"Structural similarity '{label_a}' vs '{label_b}': "
                f"{result['structural_similarity']:.3f} "
                f"({shared} shared, {only_a} only-A, {only_b} only-B) [Rust core]."
            ),
            "timestamp": iso_now(),
            "correlation_id": correlation_id,
        }
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CONFIG.atom_dir / f"structural-similarity-{stamp}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "structural_similarity",
        "timestamp": iso_now(),
        "correlation_id": correlation_id,
        "input": f"{label_a} vs {label_b}",
        "result": result,
    }
    _write_json(path, payload)
    return result, messages, {"structural_similarity": str(path)}


# ---------------------------------------------------------------------------
# Mode: substitution_analysis (Rust core) — v5.0
# ---------------------------------------------------------------------------


def _run_substitution_analysis_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Substitution analysis via the Rust core: r.substitution_analysis(a, b).

    Returns PySubstitutionResult with:
        sense_idx_a, sense_idx_b, structural_similarity,
        substitutions, unpaired_only_a, unpaired_only_b
    """
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for substitution_analysis mode")

    options = options or {}
    label_a = options.get("a", text)
    label_b = options.get("b", "")

    if not label_b:
        raise ValueError("Parameter 'b' is required for substitution analysis")

    sub_result = r.substitution_analysis(label_a, label_b)

    # Extract fields from PySubstitutionResult
    # substitutions is a list of (comp_a, comp_b) pairs
    raw_substitutions = list(getattr(sub_result, "substitutions", []))
    substitutions = [
        {"a": pair[0], "b": pair[1]} if isinstance(pair, (list, tuple)) else pair
        for pair in raw_substitutions
    ]

    result = {
        "a": label_a,
        "b": label_b,
        "sense_idx_a": getattr(sub_result, "sense_idx_a", None),
        "sense_idx_b": getattr(sub_result, "sense_idx_b", None),
        "structural_similarity": getattr(sub_result, "structural_similarity", 0.0),
        "substitutions": substitutions,
        "unpaired_only_a": list(getattr(sub_result, "unpaired_only_a", [])),
        "unpaired_only_b": list(getattr(sub_result, "unpaired_only_b", [])),
    }

    n_subs = len(substitutions)
    unpaired_a = len(result["unpaired_only_a"])
    unpaired_b = len(result["unpaired_only_b"])

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": (
                f"Substitution analysis '{label_a}' vs '{label_b}': "
                f"{n_subs} substitution pairs, "
                f"{unpaired_a} unpaired-A, {unpaired_b} unpaired-B "
                f"[Rust core]."
            ),
            "timestamp": iso_now(),
            "correlation_id": correlation_id,
        }
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CONFIG.atom_dir / f"substitution-analysis-{stamp}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "substitution_analysis",
        "timestamp": iso_now(),
        "correlation_id": correlation_id,
        "input": f"{label_a} vs {label_b}",
        "result": result,
    }
    _write_json(path, payload)
    return result, messages, {"substitution_analysis": str(path)}


# ---------------------------------------------------------------------------
# Mode: grounding_info (Rust core) — v6.0
# ---------------------------------------------------------------------------


def _run_grounding_info_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Grounding info via the Rust core: r.grounding_info(label, sense_id).

    Returns detailed grounding evidence for a specific sense, including
    evidence traces, source references, confidence breakdown, and
    composition grounding status.
    """
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for grounding_info mode")

    options = options or {}
    label = options.get("label", text)
    sense_id = int(options.get("sense_id", 0))

    info_result = r.grounding_info(label, sense_id)

    # Extract fields from grounding info result
    result = {
        "label": label,
        "sense_id": sense_id,
        "grounding_score": getattr(info_result, "grounding_score", None),
        "evidence_traces": list(getattr(info_result, "evidence_traces", [])),
        "source_references": list(getattr(info_result, "source_references", [])),
        "confidence_breakdown": getattr(info_result, "confidence_breakdown", {}),
        "composition_grounding": list(getattr(info_result, "composition_grounding", [])),
        "is_grounded": getattr(info_result, "is_grounded", False),
    }

    n_traces = len(result["evidence_traces"])
    n_sources = len(result["source_references"])
    is_grounded = result["is_grounded"]

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_ingest_status",
            "content": (
                f"Grounding info for '{label}' sense #{sense_id}: "
                f"score={result['grounding_score']}, "
                f"{n_traces} evidence traces, {n_sources} sources, "
                f"grounded={'yes' if is_grounded else 'no'} [Rust core]."
            ),
            "timestamp": iso_now(),
            "correlation_id": correlation_id,
        }
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CONFIG.atom_dir / f"grounding-info-{stamp}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "grounding_info",
        "timestamp": iso_now(),
        "correlation_id": correlation_id,
        "input": f"{label} sense #{sense_id}",
        "result": result,
    }
    _write_json(path, payload)
    return result, messages, {"grounding_info": str(path)}


# ---------------------------------------------------------------------------
# Unified mode dispatch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mode: context_query (Rust core) — v6.1
# ---------------------------------------------------------------------------


def _run_context_query_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Context-aware depth-controlled query via Rust core.

    Uses P(a|S,q) scoring, cycle detection, and adaptive halting
    for recursive composition expansion (v6.1).
    """
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for context_query mode")

    options = options or {}
    concept = options.get("concept", text)
    context_atoms = options.get("context_atoms", [])
    if isinstance(context_atoms, str):
        context_atoms = [context_atoms]

    if not context_atoms:
        # Fallback: tokenize the text to get context atoms
        context_atoms = _tokenize(text)

    max_depth = options.get("max_depth")
    gamma = options.get("gamma")
    halt_confidence = options.get("halt_confidence")
    tau_relevance = options.get("tau_relevance")

    result = r.context_query(
        concept,
        context_atoms,
        max_depth,
        gamma,
        halt_confidence,
        tau_relevance,
    )

    if result is None:
        result_dict = {"concept": concept, "result": None}
    else:
        result_dict = {
            "concept": concept,
            "active_sense_idx": getattr(result, "active_sense_idx", None),
            "total_senses": getattr(result, "total_senses", 0),
            "scored_atoms": list(getattr(result, "scored_atoms", [])),
            "depth_reached": getattr(result, "depth_reached", 0),
            "halt_reason": getattr(result, "halt_reason", "unknown"),
            "cycles_detected": getattr(result, "cycles_detected", 0),
            "layer": getattr(result, "layer", 0),
            "grounding_score": getattr(result, "grounding_score", 0.0),
        }

    messages = [
        {
            "id": make_id("msg"),
            "type": "system_context_query",
            "content": (
                f"Context query for '{concept}' with {len(context_atoms)} context atoms: "
                f"depth={result_dict.get('depth_reached', 0)}, "
                f"halt={result_dict.get('halt_reason', 'unknown')}, "
                f"cycles={result_dict.get('cycles_detected', 0)}, "
                f"atoms={len(result_dict.get('scored_atoms', []))} "
                f"[Rust core]."
            ),
            "timestamp": iso_now(),
            "correlation_id": correlation_id,
        }
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = CONFIG.atom_dir / f"context-query-{stamp}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "context_query",
        "timestamp": iso_now(),
        "correlation_id": correlation_id,
        "input": concept,
        "result": result_dict,
    }
    _write_json(path, payload)
    return result_dict, messages, {"context_query": str(path)}


_MODE_HANDLERS = {
    "ingest": _run_ingest_rust,
    "appraise": _run_appraise_rust,
    "relate": _run_relate_rust,
    "compose": _run_compose_rust,
    "structural_similarity": _run_structural_similarity_rust,
    "substitution_analysis": _run_substitution_analysis_rust,
    "grounding_info": _run_grounding_info_rust,
    "context_query": _run_context_query_rust,
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
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Public unified mode dispatch for the FastAPI server.

    This is the primary entry point for the async FastAPI layer.
    It accepts a pre-obtained Rsvs instance and keyword arguments.
    """
    from .config import make_id
    correlation_id = make_id("corr")
    merged_options: dict[str, Any] = dict(options) if options else {}
    if target is not None:
        merged_options["target"] = target
    if source is not None:
        merged_options["source"] = source
    if top_k != 10:
        merged_options["top_k"] = top_k
    return _run_mode(mode, text, correlation_id, merged_options)


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
