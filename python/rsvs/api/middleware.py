"""Request size limit middleware for RSVS API."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

MAX_REQUEST_SIZE = 1_000_000  # 1MB


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
