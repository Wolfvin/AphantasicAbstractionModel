"""Analysis RSVS API routes — similarity, structural analysis, context queries.

These endpoints provide structural comparison and context-aware analysis
of the knowledge graph.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from slowapi import Limiter

from ...protocols import RsvsCoreProtocol
from ...rsvs_core import get_rsvs_instance
from ..deps import _verify_api_key, limiter
from ..schemas import ContextQueryRequest, ContextSimilarityRequest, SimilarityRequest
from fastapi import HTTPException

router = APIRouter()


@router.post("/similarity")
@limiter.limit("30/minute")
async def similarity_endpoint(request: Request, req: SimilarityRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    sim = rsvs.similarity(req.label_a, req.label_b)
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
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    sim_result = rsvs.structural_similarity(a, b)
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
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    sub_result = rsvs.substitution_analysis(a, b)

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
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    result = rsvs.context_query(
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
    rsvs: RsvsCoreProtocol = get_rsvs_instance()
    score = rsvs.context_similarity(req.concept_a, req.concept_b, req.context)
    if score is None:
        return {"ok": True, "concept_a": req.concept_a, "concept_b": req.concept_b, "context": req.context, "context_weighted_similarity": None}
    return {
        "ok": True,
        "concept_a": req.concept_a,
        "concept_b": req.concept_b,
        "context": req.context,
        "context_weighted_similarity": score,
    }
