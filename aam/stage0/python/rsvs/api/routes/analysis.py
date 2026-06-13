"""Analysis RSVS API routes — similarity, structural analysis, context queries.

These endpoints provide structural comparison and context-aware analysis
of the knowledge graph.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...rsvs_core import get_rsvs_instance
from ..deps import _verify_api_key, limiter
from ..schemas import ContextQueryRequest, ContextSimilarityRequest, SimilarityRequest

router = APIRouter()


@router.post("/similarity")
@limiter.limit("30/minute")
async def similarity_endpoint(request: Request, req: SimilarityRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs = get_rsvs_instance()
    sim = await asyncio.to_thread(rsvs.similarity, req.label_a, req.label_b)
    if sim is None:
        raise HTTPException(status_code=404, detail=f"No similarity result for '{req.label_a}' vs '{req.label_b}' — one or both labels not found")
    sim_dict = {
        "jaccard": sim.jaccard,
        "shared": sim.shared,
        "only_a": sim.only_a,
        "only_b": sim.only_b,
    }
    return {"label_a": req.label_a, "label_b": req.label_b, "similarity": sim_dict}


@router.get("/structural-similarity", tags=["structural"])
@limiter.limit("30/minute")
async def structural_similarity_endpoint(
    request: Request,
    a: str = Query(..., min_length=1, description="First label (e.g. 'raja')"),
    b: str = Query(..., min_length=1, description="Second label (e.g. 'ratu')"),
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """Get structural similarity between two labels."""
    rsvs = get_rsvs_instance()
    sim_result = await asyncio.to_thread(rsvs.structural_similarity, a, b)
    return {
        "ok": True,
        "a": a,
        "b": b,
        "sense_idx_a": getattr(sim_result, "sense_idx_a", None),
        "sense_idx_b": getattr(sim_result, "sense_idx_b", None),
        "structural_similarity": getattr(sim_result, "structural_similarity", 0.0),
        "shared_compositions": list(getattr(sim_result, "shared_compositions", [])),
        "only_a_compositions": list(getattr(sim_result, "only_a_compositions", [])),
        "only_b_compositions": list(getattr(sim_result, "only_b_compositions", [])),
        "layer_a": getattr(sim_result, "layer_a", None),
        "layer_b": getattr(sim_result, "layer_b", None),
    }


@router.get("/substitution-analysis", tags=["structural"])
@limiter.limit("30/minute")
async def substitution_analysis_endpoint(
    request: Request,
    a: str = Query(..., min_length=1, description="First label (e.g. 'raja')"),
    b: str = Query(..., min_length=1, description="Second label (e.g. 'ratu')"),
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """Get substitution analysis between two labels."""
    rsvs = get_rsvs_instance()
    sub_result = await asyncio.to_thread(rsvs.substitution_analysis, a, b)

    raw_substitutions = list(getattr(sub_result, "substitutions", []))
    substitutions = [
        {"a": pair[0], "b": pair[1]} if isinstance(pair, (list, tuple)) else pair
        for pair in raw_substitutions
    ]

    return {
        "ok": True,
        "a": a,
        "b": b,
        "sense_idx_a": getattr(sub_result, "sense_idx_a", None),
        "sense_idx_b": getattr(sub_result, "sense_idx_b", None),
        "structural_similarity": getattr(sub_result, "structural_similarity", 0.0),
        "substitutions": substitutions,
        "unpaired_only_a": list(getattr(sub_result, "unpaired_only_a", [])),
        "unpaired_only_b": list(getattr(sub_result, "unpaired_only_b", [])),
    }


@router.post("/context-query", tags=["query"])
@limiter.limit("30/minute")
async def context_query_endpoint(request: Request, req: ContextQueryRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """Context-aware depth-controlled query (v6.1)."""
    rsvs = get_rsvs_instance()
    result = await asyncio.to_thread(
        rsvs.context_query,
        req.concept,
        req.context_atoms,
        req.max_depth,
        req.gamma,
        req.halt_confidence,
        req.tau_relevance,
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
            "cycles_detected": getattr(result, "cycles_detected", 0),
            "layer": getattr(result, "layer", 0),
            "grounding_score": getattr(result, "grounding_score", 0.0),
        },
    }


@router.post("/context-similarity", tags=["similarity"])
@limiter.limit("30/minute")
async def context_similarity_endpoint(
    request: Request,
    req: ContextSimilarityRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.2: Context-weighted similarity between two concepts."""
    rsvs = get_rsvs_instance()
    score = await asyncio.to_thread(rsvs.context_similarity, req.concept_a, req.concept_b, req.context)
    if score is None:
        return {"ok": True, "concept_a": req.concept_a, "concept_b": req.concept_b, "context": req.context, "context_weighted_similarity": None}
    return {
        "ok": True,
        "concept_a": req.concept_a,
        "concept_b": req.concept_b,
        "context": req.context,
        "context_weighted_similarity": score,
    }


@router.get("/detect-convergence", tags=["structural"])
@limiter.limit("10/minute")
async def detect_convergence_endpoint(
    request: Request,
    max_pairs: int = Query(500, ge=1, le=5000, description="Maximum number of convergence pairs to evaluate"),
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """Detect structural convergence across nodes in the graph.

    Finds nodes with structurally similar sense compositions that
    may represent the same concept across different languages or contexts.
    This uses the Rust core's ConvergenceEngine when available,
    otherwise falls back to a Jaccard-based similarity check.
    """
    rsvs = get_rsvs_instance()
    result = await asyncio.to_thread(rsvs.convergence_detect)
    if result is None:
        return {
            "ok": True,
            "pairs_found": 0,
            "convergence_pairs": [],
            "source": "none",
        }

    # Rust core returns a JSON string
    import json as _json
    if isinstance(result, str):
        try:
            parsed = _json.loads(result)
        except _json.JSONDecodeError:
            parsed = {"pairs": []}
    elif isinstance(result, dict):
        parsed = result
    else:
        parsed = {"pairs": []}

    raw_pairs = parsed.get("pairs", [])
    convergence_pairs = []
    for p in raw_pairs[:max_pairs]:
        if isinstance(p, dict):
            convergence_pairs.append({
                "node_a": p.get("a", ""),
                "node_b": p.get("b", ""),
                "similarity": p.get("overlap", 0.0),
                "shared_compositions": [],
                "linked": p.get("linked", False),
            })

    return {
        "ok": True,
        "pairs_found": len(convergence_pairs),
        "convergence_pairs": convergence_pairs,
        "source": "rust_core",
    }
