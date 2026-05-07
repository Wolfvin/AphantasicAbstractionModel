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
import sys
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
from ._version import __version__


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # v8.3: Production fail-fast — refuse to boot without required secrets.
    # In production (NODE_ENV=production), RSVS_API_KEY and RSVS_SESSION_SECRET
    # MUST be set. Empty values are a security risk.
    _node_env = os.environ.get("NODE_ENV", "development")
    _api_key = os.environ.get("RSVS_API_KEY", "")
    _session_secret = os.environ.get("RSVS_SESSION_SECRET", "")

    if _node_env == "production":
        if not _api_key:
            print(
                "FATAL: RSVS_API_KEY is required in production but not set. "
                "Set it via environment variable or .env file.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not _session_secret:
            print(
                "FATAL: RSVS_SESSION_SECRET is required in production but not set. "
                "Set it via environment variable or .env file.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Development mode: warn if secrets are empty
        if not _api_key:
            import logging
            logging.getLogger(__name__).warning(
                "RSVS_API_KEY not set — API authentication is DISABLED. "
                "Set RSVS_API_KEY in production!"
            )
        if not _session_secret and not _api_key:
            import logging
            logging.getLogger(__name__).warning(
                "RSVS_SESSION_SECRET not set — using RSVS_API_KEY as fallback. "
                "Set RSVS_SESSION_SECRET explicitly in production!"
            )

    try:
        get_rsvs_instance()
    except RustCoreUnavailableError:
        pass
    yield


# --- App ---

app = FastAPI(
    title="RSVS — Recursive Symbolic Vocabulary System",
    version=__version__,
    description="Hard-attention symbolic knowledge engine with Rust core (v8.3 — Language-agnostic, convergence detection, production-ready)",
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
