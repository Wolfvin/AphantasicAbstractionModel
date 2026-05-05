"""RSVS FastAPI server — thin wiring module.

The actual route logic is in the ``rsvs.api`` package:
  - api.schemas    — Pydantic request/response models
  - api.deps       — Auth, rate limiter, exception handlers, enriched-field helpers
  - api.middleware — Request size limit
  - api.routes.core         — Core CRUD endpoints (run, ingest, query, compose, etc.)
  - api.routes.analysis     — Structural similarity & context queries
  - api.routes.maintenance  — Consolidation, reflection, domain attention, etc.

This file creates the FastAPI app, wires middleware and exception handlers,
includes the route modules, and provides the ``main()`` entrypoint.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from .api.deps import (
    limiter,
    rsvs_error_handler,
    global_exception_handler,
    _rate_limit_exceeded_handler,
)
from .api.middleware import limit_request_size
from .api.routes.core import router as core_router
from .api.routes.analysis import router as analysis_router
from .api.routes.maintenance import router as maintenance_router
from .exceptions import RsvsError, RustCoreUnavailableError
from .rsvs_core import get_rsvs_instance


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_rsvs_instance()
    except RustCoreUnavailableError:
        pass
    yield


# --- App ---

app = FastAPI(
    title="RSVS — Relational Symbolic Vocabulary System",
    version="7.2.1",
    description="Hard-attention symbolic knowledge engine with Rust core (v7.2.1 — Session-auth proxy, constant-time compare, modular routes)",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(RsvsError, rsvs_error_handler)
app.add_exception_handler(Exception, global_exception_handler)

# --- CORS ---
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

# --- Middleware ---
app.middleware("http")(limit_request_size)

# --- Routes ---
app.include_router(core_router)
app.include_router(analysis_router)
app.include_router(maintenance_router)


def main() -> None:
    """Run the FastAPI server with uvicorn."""
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
