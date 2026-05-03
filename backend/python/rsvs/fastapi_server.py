"""RSVS FastAPI server — async, OpenAPI docs, production-grade."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .exceptions import RsvsError, RustCoreUnavailableError
from .modes import run_mode
from .rsvs_core import get_rsvs_instance


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def run_endpoint(req: RunRequest) -> dict[str, Any]:
    try:
        rsvs = get_rsvs_instance()
        return run_mode(rsvs, req.mode, text=req.text, target=req.target, source=req.source)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest) -> dict[str, Any]:
    try:
        rsvs = get_rsvs_instance()
        return run_mode(rsvs, "ingest", text=req.text, source=req.source)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_endpoint(req: QueryRequest) -> dict[str, Any]:
    try:
        rsvs = get_rsvs_instance()
        return run_mode(rsvs, "query", text=req.text, top_k=req.top_k)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity")
async def similarity_endpoint(req: SimilarityRequest) -> dict[str, Any]:
    try:
        rsvs = get_rsvs_instance()
        sim = rsvs.similarity(req.label_a, req.label_b)
        return {"label_a": req.label_a, "label_b": req.label_b, "similarity": sim}
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/appraise")
async def appraise_endpoint(req: AppraiseRequest) -> dict[str, Any]:
    try:
        rsvs = get_rsvs_instance()
        return run_mode(rsvs, "appraise", text=req.target)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/relate")
async def relate_endpoint(req: RelateRequest) -> dict[str, Any]:
    try:
        rsvs = get_rsvs_instance()
        return run_mode(rsvs, "relate", text=req.source, target=req.target)
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snapshot")
async def snapshot_endpoint() -> dict[str, Any]:
    try:
        rsvs = get_rsvs_instance()
        return json.loads(rsvs.snapshot_v1())
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events")
async def events_endpoint() -> dict[str, Any]:
    try:
        rsvs = get_rsvs_instance()
        return json.loads(rsvs.consume_events_v1())
    except RsvsError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "4.2.0"}


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "RSVS", "version": "4.2.0", "docs": "/docs"}


def main() -> None:
    """Run the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run("rsvs.fastapi_server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
