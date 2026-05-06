"""Maintenance RSVS API routes — consolidation, reflection, domain attention, etc.

These endpoints manage the lifecycle and health of the knowledge graph:
thinking modes, MCTS queries, verification, and autonomous operations.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from slowapi import Limiter

from ...protocols import RsvsCoreProtocol
from ...rsvs_core import get_rsvs_instance
from ..deps import _verify_api_key, limiter
from ..schemas import (
    ConsolidateRequest,
    MCTSQueryRequest,
    SetDomainAttentionRequest,
    ThinkingModeRequest,
    VerifyRequest,
)

router = APIRouter()


@router.get("/autonomy/pending-removals", tags=["autonomy"])
@limiter.limit("30/minute")
async def pending_removals_endpoint(request: Request, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """v6.2: Get list of nodes that require approval before removal."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    pending = rsvs.pending_removals()
    return {"ok": True, "pending_removals": pending}


@router.get("/entity-candidates", tags=["autonomy"])
@limiter.limit("10/minute")
async def entity_candidates_endpoint(
    request: Request,
    top_k: int = Query(default=10, ge=1, le=100),
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.3: Return entity candidates based on learned centrality + diversity scoring."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    candidates = rsvs.entity_candidates(top_k)
    return {
        "ok": True,
        "candidates": [
            {"label": label, "entity_score": score}
            for label, score in candidates
        ],
    }


@router.post("/set-domain-attention", tags=["domain"])
@limiter.limit("10/minute")
async def set_domain_attention_endpoint(
    request: Request,
    req: SetDomainAttentionRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.3.1: Set per-domain attention weights (α, β, γ)."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    rsvs.set_domain_attention(req.domain_id, req.alpha, req.beta, req.gamma)
    return {
        "ok": True,
        "domain_id": req.domain_id,
        "alpha": req.alpha,
        "beta": req.beta,
        "gamma": req.gamma,
        "message": "Domain attention weights set (auto-normalized). Takes effect after 5 observations.",
    }


@router.post("/thinking-mode", tags=["thinking"])
@limiter.limit("30/minute")
async def thinking_mode_endpoint(
    request: Request,
    req: ThinkingModeRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Set or query the ThinkingToggle mode."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    force_mode = {"auto": -1, "non_thinking": 0, "thinking": 1}.get(req.mode, -1)
    rsvs.set_thinking_mode(force_mode)
    return {
        "ok": True,
        "mode": req.mode,
        "force_mode_code": force_mode,
    }


@router.post("/mcts-query", tags=["query"])
@limiter.limit("10/minute")
async def mcts_query_endpoint(
    request: Request,
    req: MCTSQueryRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: MCTS-style traversal query for complex disambiguation."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    result = rsvs.mcts_query(
        req.concept,
        req.context_atoms,
        req.max_simulations,
        req.max_depth,
    )
    if result is None:
        return {"ok": True, "result": None}
    return {
        "ok": True,
        "concept": req.concept,
        "result": {
            "active_sense_idx": getattr(result, "active_sense_idx", None),
            "total_senses": getattr(result, "total_senses", 0),
            "scored_atoms": list(getattr(result, "scored_atoms", [])),
            "depth_reached": getattr(result, "depth_reached", 0),
            "halt_reason": getattr(result, "halt_reason", "unknown"),
            "simulations_run": getattr(result, "simulations_run", 0),
            "best_path": list(getattr(result, "best_path", [])),
        },
    }


@router.post("/consolidate", tags=["maintenance"])
@limiter.limit("5/minute")
async def consolidate_endpoint(
    request: Request,
    req: ConsolidateRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Trigger manual consolidation of the knowledge graph."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    result = rsvs.consolidate(req.force)
    return {
        "ok": True,
        "senses_merged": getattr(result, "senses_merged", 0),
        "senses_removed": getattr(result, "senses_removed", 0),
        "edges_pruned": getattr(result, "edges_pruned", 0),
        "atoms_compacted": getattr(result, "atoms_compacted", 0),
    }


@router.post("/verify", tags=["verification"])
@limiter.limit("10/minute")
async def verify_endpoint(
    request: Request,
    req: VerifyRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Neuro-symbolic verification of a node's compositions."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    result = rsvs.verify(req.label, req.max_iterations)
    return {
        "ok": True,
        "label": req.label,
        "status": getattr(result, "status", "unknown"),
        "rules_checked": getattr(result, "rules_checked", 0),
        "rules_passed": getattr(result, "rules_passed", 0),
        "rules_failed": getattr(result, "rules_failed", 0),
        "iterations": getattr(result, "iterations", 0),
    }


@router.get("/composition-index/stats", tags=["index"])
@limiter.limit("30/minute")
async def composition_index_stats_endpoint(
    request: Request,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Get statistics about the composition reverse index."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    stats = rsvs.composition_index_stats()
    return {
        "ok": True,
        "total_entries": getattr(stats, "total_entries", 0),
        "total_dependencies": getattr(stats, "total_dependencies", 0),
    }


@router.post("/reflection", tags=["maintenance"])
@limiter.limit("5/minute")
async def reflection_endpoint(
    request: Request,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Trigger a sense reflection cycle."""
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    result = rsvs.run_reflection()
    return {
        "ok": True,
        "actions_total": getattr(result, "actions_total", 0),
        "actions_applied": getattr(result, "actions_applied", 0),
    }
