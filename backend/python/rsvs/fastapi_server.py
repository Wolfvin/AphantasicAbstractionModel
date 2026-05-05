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
import logging
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

logger = logging.getLogger(__name__)

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


# --- Rate limiter (keyed by API key when available, fallback to IP) ---

def _rate_limit_key(request: Request) -> str:
    """Rate limit by API key when available, otherwise by IP address.

    IP-based limits are trivially bypassed via proxy rotation.
    Keying by API key ensures per-tenant isolation.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


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
    version="7.2.0",
    description="Hard-attention symbolic knowledge engine with Rust core (v7.2 — Security hardening: API key proxy, centralized error handling, HTTPS, atomic persist)",
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
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_REQUEST_SIZE:
                return JSONResponse(status_code=413, content={"error": "request_too_large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "invalid_content_length"})
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
    options: dict[str, Any] | None = Field(None, description="Mode-specific options (e.g. atom_ids, compositions, lang for compose)")


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


class SetDomainAttentionRequest(BaseModel):
    """v6.3.1: Set per-domain attention weights for adaptive α/β/γ."""
    domain_id: int = Field(..., ge=0, description="Domain identifier")
    alpha: float = Field(..., ge=0.0, le=1.0, description="Weight for NPMI term (will be normalized)")
    beta: float = Field(..., ge=0.0, le=1.0, description="Weight for Jaccard term (will be normalized)")
    gamma: float = Field(..., ge=0.0, le=1.0, description="Weight for co-occurrence term (will be normalized)")


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
    if not api_key or not secrets.compare_digest(api_key, _API_KEY):
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
    # Log full detail server-side, but only send error class name to client
    if status >= 500:
        logger.error("RSVS error (%s): %s", type(exc).__name__, exc)
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log internal details server-side, return generic response to client.

    Prevents stack traces, file paths, and Rust module names from leaking
    to the browser — a common information disclosure vulnerability.
    """
    logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=True)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


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
        return run_mode(rsvs, req.mode, text=req.text, target=req.target, source=req.source, options=req.options)
    except RsvsError:
        raise
    except Exception:
        raise


@app.post("/ingest")
@limiter.limit("30/minute")
async def ingest_endpoint(request: Request, req: IngestRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "ingest", text=req.text, source=req.source)
    except RsvsError:
        raise
    except Exception:
        raise


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
    except Exception:
        raise


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
    except Exception:
        raise


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
    except Exception:
        raise


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
    except Exception:
        raise


@app.post("/appraise")
@limiter.limit("30/minute")
async def appraise_endpoint(request: Request, req: AppraiseRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "appraise", text=req.target)
    except RsvsError:
        raise
    except Exception:
        raise


@app.post("/relate")
@limiter.limit("30/minute")
async def relate_endpoint(request: Request, req: RelateRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "relate", text=req.source, target=req.target)
    except RsvsError:
        raise
    except Exception:
        raise


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
        # Count senses before compose to determine new sense index for condition_label
        sense_count_before = 0
        try:
            sense_count_before = len(rsvs.senses(req.label))
        except Exception:
            pass

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
        # Fix: determine the correct sense index (not hardcoded 0)
        if req.condition_label:
            try:
                sense_count_after = len(rsvs.senses(req.label))
                # New sense is at last index if count increased, else index 0
                target_idx = sense_count_after - 1 if sense_count_after > sense_count_before else 0
                rsvs.set_sense_label(req.label, target_idx, req.condition_label)
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
        if "CompositionRejected" in str(e):
            raise HTTPException(status_code=422, detail="composition_rejected")
        raise HTTPException(status_code=400, detail="bad_request")


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
    except Exception:
        raise


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
    except Exception:
        raise


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
    except Exception:
        raise


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
    except Exception:
        raise


@app.get("/snapshot")
@limiter.limit("60/minute")
async def snapshot_endpoint(request: Request, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return json.loads(rsvs.snapshot_v1())
    except RsvsError:
        raise
    except Exception:
        raise


@app.get("/events")
@limiter.limit("60/minute")
async def events_endpoint(request: Request, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return json.loads(rsvs.consume_events_v1())
    except RsvsError:
        raise
    except Exception:
        raise


@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "version": "7.2.0"}


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request) -> dict[str, str]:
    return {"name": "RSVS", "version": "7.2.0", "docs": "/docs"}


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
    except Exception:
        raise


@app.get("/entity-candidates", tags=["autonomy"])
@limiter.limit("10/minute")
async def entity_candidates_endpoint(
    request: Request,
    top_k: int = Query(default=10, ge=1, le=100),
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.3: Return entity candidates based on learned centrality + diversity scoring.

    These tokens appear structurally significant in the attention graph
    but have not yet been promoted to nodes. This is a suggestion
    mechanism — promotion must be done explicitly via /compose or /ingest.
    """
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        candidates = rsvs.entity_candidates(top_k)
        return {
            "ok": True,
            "candidates": [
                {"label": label, "entity_score": score}
                for label, score in candidates
            ],
        }
    except RsvsError:
        raise
    except Exception:
        raise


@app.post("/set-domain-attention", tags=["domain"])
@limiter.limit("10/minute")
async def set_domain_attention_endpoint(
    request: Request,
    req: SetDomainAttentionRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.3.1: Set per-domain attention weights (α, β, γ).

    Creates or updates the attention weight configuration for a specific domain.
    After at least 5 ingest observations, these domain-specific weights override
    the global attention config when ingesting text tagged with that domain.

    The weights are automatically normalized to sum to 1.0.

    **When to use:**
    - Domain A is mostly about spatial relationships → set β (Jaccard) higher
    - Domain B is about co-occurrence patterns → set γ (Cooc) higher
    - Domain C requires strong statistical significance → set α (NPMI) higher

    **Adaptive nudging:** The system also auto-nudges domain weights during ingest
    based on coherence improvements. This endpoint sets the initial/override values.
    """
    try:
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
    except RsvsError:
        raise
    except Exception:
        raise


# ---------------------------------------------------------------------------
# v6.5: New endpoints for Losion Cross-Pollination features
# ---------------------------------------------------------------------------

class ThinkingModeRequest(BaseModel):
    """v6.5: Set or query the ThinkingToggle mode."""
    mode: str = Field("auto", description="Mode: 'auto', 'thinking', or 'non_thinking'")


class MCTSQueryRequest(BaseModel):
    """v6.5: MCTS-style traversal query."""
    concept: str = Field(..., min_length=1, max_length=500, description="Concept label to query")
    context_atoms: List[str] = Field(default_factory=list, description="Context atom labels")
    max_simulations: int = Field(10, ge=1, le=100, description="Number of MCTS simulations")
    max_depth: int = Field(4, ge=1, le=10, description="Max depth per simulation")


class ConsolidateRequest(BaseModel):
    """v6.5: Trigger manual consolidation."""
    force: bool = Field(False, description="Force consolidation regardless of interval")


class VerifyRequest(BaseModel):
    """v6.5: Neuro-symbolic verification of a node's compositions."""
    label: str = Field(..., min_length=1, max_length=500, description="Node label to verify")
    max_iterations: int = Field(3, ge=1, le=10, description="Max verification-revision iterations")


@app.post("/thinking-mode", tags=["thinking"])
@limiter.limit("30/minute")
async def thinking_mode_endpoint(
    request: Request,
    req: ThinkingModeRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Set or query the ThinkingToggle mode.

    Controls whether queries use shallow (NON_THINKING) or deep (THINKING)
    traversal. In 'auto' mode, the system classifies each query's complexity
    and selects the appropriate mode automatically.
    """
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        force_mode = {"auto": -1, "non_thinking": 0, "thinking": 1}.get(req.mode, -1)
        rsvs.set_thinking_mode(force_mode)
        return {
            "ok": True,
            "mode": req.mode,
            "force_mode_code": force_mode,
        }
    except RsvsError:
        raise
    except Exception:
        raise


@app.post("/mcts-query", tags=["query"])
@limiter.limit("10/minute")
async def mcts_query_endpoint(
    request: Request,
    req: MCTSQueryRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: MCTS-style traversal query for complex disambiguation.

    Uses Monte Carlo Tree Search with UCB1 selection and structural
    value evaluation (grounding × coherence) for deeper exploration
    of compositional structures. Best for multi-sense, high-layer queries.
    """
    try:
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
    except RsvsError:
        raise
    except Exception:
        raise


@app.post("/consolidate", tags=["maintenance"])
@limiter.limit("5/minute")
async def consolidate_endpoint(
    request: Request,
    req: ConsolidateRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Trigger manual consolidation of the knowledge graph.

    Consolidation performs thorough cleanup:
    - Remove dead senses (fragile + ungrounded + very inactive)
    - Merge similar senses across nodes (Jaccard ≥ 0.8)
    - Prune weak edges (weight below threshold after decay)
    - Compact atom records (remove nodes below tau_remove)

    Normally runs automatically at intervals, but this endpoint
    allows manual triggering for maintenance.
    """
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        result = rsvs.consolidate(req.force)
        return {
            "ok": True,
            "senses_merged": getattr(result, "senses_merged", 0),
            "senses_removed": getattr(result, "senses_removed", 0),
            "edges_pruned": getattr(result, "edges_pruned", 0),
            "atoms_compacted": getattr(result, "atoms_compacted", 0),
        }
    except RsvsError:
        raise
    except Exception:
        raise


@app.post("/verify", tags=["verification"])
@limiter.limit("10/minute")
async def verify_endpoint(
    request: Request,
    req: VerifyRequest,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Neuro-symbolic verification of a node's compositions.

    Verifies that a node's compositional senses satisfy structural invariants:
    - No self-reference
    - Layer consistency (compositions reference lower layers)
    - Grounding threshold (targets are grounded)
    - Frequency threshold (targets have sufficient freq)
    - No circular chains

    Returns verification status and per-rule results.
    """
    try:
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
    except RsvsError:
        raise
    except Exception:
        raise


@app.get("/composition-index/stats", tags=["index"])
@limiter.limit("30/minute")
async def composition_index_stats_endpoint(
    request: Request,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Get statistics about the composition reverse index.

    Returns counts of indexed composition references, which is useful
    for monitoring the O(1) reverse lookup system's size.
    """
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        stats = rsvs.composition_index_stats()
        return {
            "ok": True,
            "total_entries": getattr(stats, "total_entries", 0),
            "total_dependencies": getattr(stats, "total_dependencies", 0),
        }
    except RsvsError:
        raise
    except Exception:
        raise


@app.post("/reflection", tags=["maintenance"])
@limiter.limit("5/minute")
async def reflection_endpoint(
    request: Request,
    _auth: None = Depends(_verify_api_key),
) -> dict[str, Any]:
    """v6.5: Trigger a sense reflection cycle.

    Reflection evaluates each sense and produces actions:
    - CONFIRM: sense is well-grounded
    - REVIEW: sense has some contradictions (monitor)
    - REVISE: sense needs composition pruning
    - RETIRE: sense is fragile + ungrounded + inactive

    Returns the number of actions produced and applied.
    """
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        result = rsvs.run_reflection()
        return {
            "ok": True,
            "actions_total": getattr(result, "actions_total", 0),
            "actions_applied": getattr(result, "actions_applied", 0),
        }
    except RsvsError:
        raise
    except Exception:
        raise


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
