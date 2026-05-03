"""RSVS FastAPI server — async, OpenAPI docs, production-grade."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .exceptions import RsvsError, RustCoreUnavailableError
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
    version="4.2.0",
    description="Hard-attention symbolic knowledge engine with Rust core",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    text: str = Field(..., min_length=1, description="Text to ingest")
    source: str | None = Field(None, description="Source identifier")


class RunRequest(BaseModel):
    mode: str = Field(..., description="Mode: ingest | query | appraise | relate")
    text: str = Field(..., min_length=1)
    target: str | None = None
    source: str | None = None


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


class ComposeRequest(BaseModel):
    label: str = Field(..., min_length=1, description="Label for the composite node")
    atom_ids: List[int] = Field(..., min_length=1, description="List of atom node IDs to compose from")
    lang: Optional[str] = Field(None, description="Language code (e.g. 'id', 'en')")


# --- Exception mapping ---

_EXCEPTION_STATUS_MAP: dict[str, int] = {
    "SchemaVersionMismatchError": 409,
    "SchemaValidationError": 422,
    "InvariantViolationError": 422,
    "InvalidModeError": 400,
    "RustCoreUnavailableError": 503,
}


@app.exception_handler(RsvsError)
async def rsvs_error_handler(request: Request, exc: RsvsError) -> JSONResponse:
    status = _EXCEPTION_STATUS_MAP.get(type(exc).__name__, 500)
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


# --- Endpoints ---

@app.post("/run")
@limiter.limit("30/minute")
async def run_endpoint(request: Request, req: RunRequest) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, req.mode, text=req.text, target=req.target, source=req.source)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
@limiter.limit("30/minute")
async def ingest_endpoint(request: Request, req: IngestRequest) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "ingest", text=req.text, source=req.source)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
@limiter.limit("30/minute")
async def query_endpoint(request: Request, req: QueryRequest) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "query", text=req.text, top_k=req.top_k)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity")
@limiter.limit("30/minute")
async def similarity_endpoint(request: Request, req: SimilarityRequest) -> dict[str, Any]:
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


@app.post("/appraise")
@limiter.limit("30/minute")
async def appraise_endpoint(request: Request, req: AppraiseRequest) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "appraise", text=req.target)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/relate")
@limiter.limit("30/minute")
async def relate_endpoint(request: Request, req: RelateRequest) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return run_mode(rsvs, "relate", text=req.source, target=req.target)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compose", tags=["compose"])
async def compose_node(request: ComposeRequest):
    """
    Create a composite node from explicit atom IDs.

    This is the core compositional mechanism of RSVS: higher-level concepts
    are built from lower-level atoms. For example:
    - "raja" = tahta_tertinggi + laki_laki + kerajaan
    - "ratu" = tahta_tertinggi + perempuan + kerajaan

    The shared atoms (tahta_tertinggi, kerajaan) create a semantic
    relationship between "raja" and "ratu" with Jaccard similarity = 0.5.
    """
    rsvs = get_rsvs_instance()
    if rsvs is None:
        raise RustCoreUnavailableError()

    try:
        node_id = rsvs.compose(request.label, request.atom_ids, request.lang)
        snapshot = rsvs.snapshot_v1()
        events = rsvs.consume_events_v1()
        return {
            "status": "ok",
            "node_id": node_id,
            "label": request.label,
            "atom_ids": request.atom_ids,
            "snapshot": snapshot,
            "events": events,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/snapshot")
@limiter.limit("60/minute")
async def snapshot_endpoint(request: Request) -> dict[str, Any]:
    try:
        rsvs: RsvsCoreProtocol = get_rsvs_instance()
        return json.loads(rsvs.snapshot_v1())
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events")
@limiter.limit("60/minute")
async def events_endpoint(request: Request) -> dict[str, Any]:
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
    return {"status": "ok", "version": "4.2.0"}


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request) -> dict[str, str]:
    return {"name": "RSVS", "version": "4.2.0", "docs": "/docs"}


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
