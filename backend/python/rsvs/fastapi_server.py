"""RSVS FastAPI server — async, OpenAPI docs, production-grade.

Updated for RSVS v6.0 with:
  - Structural similarity and substitution analysis endpoints
  - Compose with (label, sense_id) pairs
  - Enriched return types from Rust core (layer, grounding_score, compositions, etc.)
  - Grounding info and sense revision endpoints
  - Indonesian corpus (kerajaan, konsep domains)
"""

from __future__ import annotations

import json
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .exceptions import (
    CompositionError,
    GroundingError,
    InvariantViolationError,
    InvalidModeError,
    NodeNotFoundError,
    RsvsError,
    RustCoreUnavailableError,
    SchemaValidationError,
    SchemaVersionMismatchError,
    SenseError,
)
from .modes import run_mode
from .protocols import RsvsCoreProtocol
from .rsvs_core import get_rsvs_instance


# --- Rate limiter ---

limiter = Limiter(key_func=get_remote_address)


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the Rust core
    try:
        get_rsvs_instance()
    except RustCoreUnavailableError:
        pass
    yield


# --- App ---

app = FastAPI(
    title="RSVS — Relational Symbolic Vocabulary System",
    version="6.2.0",
    description="Hard-attention symbolic knowledge engine with Rust core (v6.2 context-similarity + adaptive weights + impact counting + condition labels)",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS: configurable whitelist via environment variable ---
# Set RSVS_ALLOWED_ORIGINS to a comma-separated list of allowed origins.
# Example: RSVS_ALLOWED_ORIGINS=http://localhost:3000,https://rsvs.example.com
# If not set, defaults to localhost:3000 (development-safe).
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("RSVS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# --- Request size limit middleware ---

MAX_REQUEST_SIZE = 1_000_000  # 1MB


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.headers.get("content-length"):
        if int(request.headers["content-length"]) > MAX_REQUEST_SIZE:
            return JSONResponse(status_code=413, content={"error": "request_too_large"})
    response = await call_next(request)
    return response


# --- Request/Response Models ---

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000, description="Text to ingest (max 100K characters)")
    source: str | None = Field(None, max_length=500, description="Source identifier")


class RunRequest(BaseModel):
    mode: str = Field(..., description="Mode: ingest | query | appraise | relate | compose | structural_similarity | substitution_analysis | context_similarity")
    text: str = Field(..., min_length=1, max_length=100_000, description="Input text (max 100K characters)")
    target: str | None = Field(None, max_length=500)
    source: str | None = Field(None, max_length=500)


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)


class SimilarityRequest(BaseModel):
    label_a: str
    label_b: str


class AppraiseRequest(BaseModel):
    target: str = Field(..., min_length=1)


class RelateRequest(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)


class CompositionPair(BaseModel):
    """A (label, sense_id) pair for compositional node creation."""
    label: str = Field(..., min_length=1, description="Atom label")
    sense_id: int = Field(0, ge=0, description="Sense ID within the atom (0 = default)")


class ComposeRequest(BaseModel):
    label: str = Field(..., min_length=1, description="Label for the composite node")
    compositions: Optional[List[CompositionPair]] = Field(
        None,
        description="List of (label, sense_id) pairs for v5.0 compose. Mutually exclusive with atom_ids.",
    )
    atom_ids: Optional[List[int]] = Field(
        None,
        description="List of atom node IDs to compose from (backward compat). Mutually exclusive with compositions.",
    )
    lang: Optional[str] = Field(None, description="Language code (e.g. 'id', 'en')")
    condition_label: Optional[str] = Field(None, max_length=100, description="v6.2: Optional condition label for the new sense (e.g. 'via_api.partial_burn')")


class NodeInfoRequest(BaseModel):
    label: str = Field(..., min_length=1)


class SensesRequest(BaseModel):
    label: str = Field(..., min_length=1)


class ContextQueryRequest(BaseModel):
    """Request for context-aware depth-controlled query (v6.1)."""
    concept: str = Field(..., min_length=1, max_length=500, description="Concept label to query (e.g. 'raja')")
    context_atoms: List[str] = Field(..., min_length=1, description="Context atom labels for disambiguation (e.g. ['kerajaan', 'tahta'])")
    max_depth: Optional[int] = Field(None, ge=1, le=10, description="Maximum traversal depth (default: from pipeline config)")
    gamma: Optional[float] = Field(None, ge=0.001, le=1.0, description="Stability halting threshold")
    halt_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence halting threshold")
    tau_relevance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance gating threshold")


class ContextSimilarityRequest(BaseModel):
    """v6.2: Request for context-weighted similarity between two concepts."""
    concept_a: str = Field(..., min_length=1, max_length=200, description="First concept label (e.g. 'batu')")
    concept_b: str = Field(..., min_length=1, max_length=200, description="Second concept label (e.g. 'tulang')")
    context: list[str] = Field(default_factory=list, max_length=50, description="Context atom labels (e.g. ['kekerasan'])")


# --- Exception mapping ---

# --- API Key authentication ---
# Set RSVS_API_KEY to enable authentication. If not set, auth is skipped (dev mode).
_API_KEY = os.environ.get("RSVS_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """Verify the API key if RSVS_API_KEY is configured.

    If RSVS_API_KEY is not set (empty), authentication is skipped for development.
    In production, always set RSVS_API_KEY to a strong random value.
    """
    if not _API_KEY:
        # Dev mode: no API key configured, skip auth
        return
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Exception mapping ---

_EXCEPTION_STATUS_MAP: dict[str, int] = {
    "SchemaVersionMismatchError": 409,
    "SchemaValidationError": 422,
    "InvariantViolationError": 422,
    "InvalidModeError": 400,
    "RustCoreUnavailableError": 503,
    "NodeNotFoundError": 404,
    "CompositionError": 400,
    "SenseError": 400,
    "GroundingError": 422,
}


@app.exception_handler(RsvsError)
async def rsvs_error_handler(request: Request, exc: RsvsError) -> JSONResponse:
    status = _EXCEPTION_STATUS_MAP.get(type(exc).__name__, 500)
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Helper: extract enriched fields from PyO3 result objects
# ---------------------------------------------------------------------------

def _enriched_node_info(raw: Any) -> dict[str, Any]:
    """Convert a PyNodeInfo (v5.0) to a dict, including layer."""
    result: dict[str, Any] = {}
    for attr in ("id", "label", "surface_label", "tier", "confidence",
                 "status", "is_seed", "is_locked", "layer"):
        val = getattr(raw, attr, None)
        if val is not None:
            result[attr] = val
    return result


def _enriched_sense_info(raw: Any) -> dict[str, Any]:
    """Convert a PySenseInfo (v5.0) to a dict, including layer, grounding_score, compositions."""
    result: dict[str, Any] = {}
    for attr in ("sense_idx", "label", "tier", "confidence",
                 "status", "layer", "grounding_score", "compositions"):
        val = getattr(raw, attr, None)
        if val is not None:
            result[attr] = val
    return result


def _enriched_query_result(raw: Any) -> dict[str, Any]:
    """Convert a PyQueryResult (v5.0) to a dict, including layer, grounding_score, compositions."""
    result: dict[str, Any] = {}
    for attr in ("label", "tier", "confidence", "status",
                 "layer", "grounding_score", "compositions"):
        val = getattr(raw, attr, None)
        if val is not None:
            result[attr] = val
    return result


# --- Endpoints ---

@app.post("/run")
@limiter.limit("30/minute")
async def run_endpoint(request: Request, req: RunRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, req.mode, text=req.text, target=req.target, source=req.source)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
@limiter.limit("30/minute")
async def ingest_endpoint(request: Request, req: IngestRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "ingest", text=req.text, source=req.source)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
@limiter.limit("30/minute")
async def query_endpoint(request: Request, req: QueryRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        result = rsvs.query(req.text, "")
        if result is None:
            return {"ok": True, "result": None}
        # v5.0: PyQueryResult with layer, grounding_score, compositions
        enriched = _enriched_query_result(result)
        return {"ok": True, "result": enriched}
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity")
@limiter.limit("30/minute")
async def similarity_endpoint(request: Request, req: SimilarityRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        sim = rsvs.similarity(req.label_a, req.label_b)
        # Convert PySimResult to a JSON-serializable dict
        sim_dict = {
            "jaccard": sim.jaccard,
            "shared": sim.shared,
            "only_a": sim.only_a,
            "only_b": sim.only_b,
        }
        return {"label_a": req.label_a, "label_b": req.label_b, "similarity": sim_dict}
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/structural-similarity", tags=["structural"])
@limiter.limit("30/minute")
async def structural_similarity_endpoint(
    request: Request,
    a: str = Query(..., min_length=1, description="First label (e.g. 'raja')"),
    b: str = Query(..., min_length=1, description="Second label (e.g. 'ratu')"),
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """Get structural similarity between two labels.

    Returns PyStructuralSimResult with shared/differing compositions,
    layer information, and structural similarity score.
    """
    try:
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
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/substitution-analysis", tags=["structural"])
@limiter.limit("30/minute")
async def substitution_analysis_endpoint(
    request: Request,
    a: str = Query(..., min_length=1, description="First label (e.g. 'raja')"),
    b: str = Query(..., min_length=1, description="Second label (e.g. 'ratu')"),
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """Get substitution analysis between two labels.

    Returns PySubstitutionResult with substitution pairs and unpaired
    compositions, revealing how two concepts differ structurally.
    """
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        sub_result = rsvs.substitution_analysis(a, b)

        # Convert substitution pairs to serializable format
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
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/appraise")
@limiter.limit("30/minute")
async def appraise_endpoint(request: Request, req: AppraiseRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "appraise", text=req.target)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/relate")
@limiter.limit("30/minute")
async def relate_endpoint(request: Request, req: RelateRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "relate", text=req.source, target=req.target)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compose", tags=["compose"])
@limiter.limit("30/minute")
async def compose_node(request: Request, body: ComposeRequest, _auth: None = Depends(_verify_api_key)):
    """
    Create a composite node from explicit compositions or atom IDs.

    v5.0 Compositional Architecture:
    - **compositions**: List of (label, sense_id) pairs, e.g.
      `[{"label": "tahta_tertinggi", "sense_id": 0}, {"label": "laki_laki", "sense_id": 0}]`
    - **atom_ids**: Legacy list of integer node IDs (backward compat, uses sense_id=0)

    The compositional mechanism builds higher-level concepts from lower-level atoms:
    - "raja" = tahta_tertinggi + laki_laki + kerajaan
    - "ratu" = tahta_tertinggi + perempuan + kerajaan

    Shared atoms (tahta_tertinggi, kerajaan) create semantic relationships.
    """
    rsvs = get_rsvs_instance()
    if rsvs is None:
        raise RustCoreUnavailableError()

    req = body

    # Validate mutually exclusive fields
    if req.compositions and req.atom_ids:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'compositions' or 'atom_ids', not both.",
        )

    if not req.compositions and not req.atom_ids:
        raise HTTPException(
            status_code=400,
            detail="Either 'compositions' or 'atom_ids' must be provided.",
        )

    try:
        if req.compositions:
            # v6.0: compose with (label, sense_id) pairs
            compositions_tuples = [(c.label, c.sense_id) for c in req.compositions]
            node_id = rsvs.compose(req.label, compositions_tuples, req.lang)
            composed_from = [{"label": c.label, "sense_id": c.sense_id} for c in req.compositions]
            composed_type = "compositions"
        else:
            # Backward compat: compose_from_ids
            node_id = rsvs.compose_from_ids(req.label, req.atom_ids, req.lang)
            composed_from = req.atom_ids
            composed_type = "atom_ids"

        # v6.2: Set condition label if provided
        if req.condition_label:
            try:
                rsvs.set_sense_label(req.label, 0, req.condition_label)
            except Exception:
                pass  # Non-critical annotation — don't fail the compose

        snapshot = rsvs.snapshot_v1()
        events = rsvs.consume_events_v1()
        return {
            "ok": True,
            "status": "ok",
            "node_id": node_id,
            "label": req.label,
            composed_type: composed_from,
            "lang": req.lang,
            "snapshot": snapshot,
            "events": events,
        }
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/node-info", tags=["node"])
@limiter.limit("30/minute")
async def node_info_endpoint(request: Request, req: NodeInfoRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """Get enriched node info (v5.0: includes layer field)."""
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        info = rsvs.node_info(req.label)
        if info is None:
            return {"ok": True, "result": None}
        enriched = _enriched_node_info(info)
        return {"ok": True, "result": enriched}
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/senses", tags=["node"])
@limiter.limit("30/minute")
async def senses_endpoint(request: Request, req: SensesRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """Get sense info for a label (v5.0: includes layer, grounding_score, compositions)."""
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        senses = rsvs.senses(req.label)
        enriched_senses = [_enriched_sense_info(s) for s in senses]
        return {"ok": True, "label": req.label, "senses": enriched_senses}
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context-query", tags=["query"])
@limiter.limit("30/minute")
async def context_query_endpoint(request: Request, req: ContextQueryRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """Context-aware depth-controlled query (v6.1).

    Uses P(a|S,q) scoring, cycle detection, and adaptive halting
    for recursive composition expansion. Returns scored atoms with
    traversal metadata.

    Depth presets:
    - Shallow (max_depth=1): Fast appraise-style lookup
    - Medium (max_depth=2): Relate-style one-hop expansion
    - Deep (max_depth=5): Full grounding verification
    """
    try:
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
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context-similarity", tags=["similarity"])
@limiter.limit("30/minute")
async def context_similarity_endpoint(
    request: Request,
    req: ContextSimilarityRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.2: Context-weighted similarity between two concepts.

    Unlike structural similarity which compares compositions equally,
    this endpoint weighs each composition based on its relevance to
    the provided context. Two concepts may have low structural similarity
    but high context-weighted similarity when the context highlights
    their shared aspects.

    Example: context_similarity("batu", "tulang", ["kekerasan"]) → high
    because both score high for "hard" in the context of "kekerasan".
    """
    try:
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
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snapshot")
@limiter.limit("60/minute")
async def snapshot_endpoint(request: Request, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return json.loads(rsvs.snapshot_v1())
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events")
@limiter.limit("60/minute")
async def events_endpoint(request: Request, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return json.loads(rsvs.consume_events_v1())
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "version": "6.2.0"}


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request) -> dict[str, str]:
    return {"name": "RSVS", "version": "6.2.0", "docs": "/docs"}


@app.get("/autonomy/pending-removals", tags=["autonomy"])
@limiter.limit("30/minute")
async def pending_removals_endpoint(request: Request, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """v6.2: Get list of nodes that require approval before removal.

    Nodes on this list have low confidence but high impact (many dependents),
    so they cannot be automatically removed. Manual review is required.
    """
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        pending = rsvs.pending_removals()
        return {"ok": True, "pending_removals": pending}
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    """Run the FastAPI server with uvicorn."""
    import os
    import uvicorn
    reload = os.environ.get("RSVS_DEV_RELOAD", "0") == "1"
    uvicorn.run(
        "rsvs.fastapi_server:app",
        host=os.environ.get("RSVS_HOST", "0.0.0.0"),
        port=int(os.environ.get("RSVS_PORT", "8000")),
        reload=reload,
    )


if __name__ == "__main__":
    main()
