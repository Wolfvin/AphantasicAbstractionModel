"""Request size limit middleware for RSVS API.

Reads the actual request body (including chunked transfers) and enforces
a maximum size limit. This prevents the chunked-transfer-encoding bypass
where Content-Length is absent and the body exceeds the limit.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response

MAX_REQUEST_SIZE = 1_000_000  # 1MB


async def limit_request_size(request: Request, call_next):
    # Read and accumulate the request body, enforcing size limit
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_REQUEST_SIZE:
            return Response(content="Request body too large", status_code=413)

    # Re-create the request with the consumed body
    async def receive():
        return {"type": "http.request", "body": body}

    request._receive = receive
    response = await call_next(request)
    return response
