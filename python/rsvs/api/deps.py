"""Shared dependencies for RSVS API routes.

Includes: API key authentication, rate limiter, RSVS instance accessor,
and enriched-field helpers.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from typing import Any

from fastapi import Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..exceptions import RsvsError, RustCoreUnavailableError
from ..protocols import RsvsCoreProtocol
from ..rsvs_core import get_rsvs_instance

logger = logging.getLogger(__name__)

# --- API Key authentication ---

_API_KEY = os.environ.get("RSVS_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    """Verify the API key if RSVS_API_KEY is configured.

    If RSVS_API_KEY is not set (empty), authentication is skipped for development.
    In production, always set RSVS_API_KEY to a strong random value.
    """
    if not _API_KEY:
        return
    if not api_key or not secrets.compare_digest(api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Rate limiter ---

def _rate_limit_key(request: Request) -> str:
    """Rate limit by API key when available, otherwise by IP address."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


# --- Exception mapping ---

EXCEPTION_STATUS_MAP: dict[str, int] = {
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


async def rsvs_error_handler(request: Request, exc: RsvsError) -> JSONResponse:
    status = EXCEPTION_STATUS_MAP.get(type(exc).__name__, 500)
    if status >= 500:
        logger.error("RSVS error (%s): %s", type(exc).__name__, exc)
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__},
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log internal details server-side, return generic response to client."""
    logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=True)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


# --- Enriched field helpers ---

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
    """Convert a PyQueryResult (v8.2) to a dict, including layer, grounding_score, compositions, convergence_contributors."""
    result: dict[str, Any] = {}
    for attr in ("label", "tier", "confidence", "status",
                 "layer", "grounding_score", "compositions",
                 "convergence_contributors"):
        val = getattr(raw, attr, None)
        if val is not None:
            result[attr] = val
    return result
