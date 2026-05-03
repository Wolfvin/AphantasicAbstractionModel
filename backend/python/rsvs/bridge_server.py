#!/usr/bin/env python3
"""DEPRECATED: Use fastapi_server.py instead. This module will be removed in v5.0.

RSVS backend bridge server — Rust-core integration (v4.2).

Mode-aware HTTP bridge for agent/frontend workflows.
Delegates all computational heavy lifting to the Rust core via PyO3,
while keeping HTTP infrastructure and artifact persistence in Python.

Architecture:
    HTTP Request → bridge_server.py (thin HTTP layer)
                       ↓
                  rsvs.modes (mode dispatch)
                       ↓
                  rsvs.Rsvs (Rust core via PyO3)
                       ↓
                  Rust: pipeline, attention, autonomy, sense, graph
"""

from __future__ import annotations

import json
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import Any

from .config import (
    API_VERSION,
    CONFIG,
    SCHEMA_VERSION,
    VALID_MODES,
    iso_now,
    make_id,
)
from .exceptions import (
    InvalidModeError,
    RsvsError,
    RustCoreUnavailableError,
    SchemaValidationError,
    SchemaVersionMismatchError,
)
from .modes import _read_latest_ingest_bundle, _read_latest_mode, _run_mode
from .rsvs_core import _get_rsvs, _save_rsvs, is_rust_core_available
from .validation import _normalize_view


# Emit deprecation warning when this module is imported
warnings.warn(
    "bridge_server is deprecated, use fastapi_server instead",
    DeprecationWarning,
    stacklevel=2,
)


# ---------------------------------------------------------------------------
# HTTP status code mapping for RSVS exceptions
# ---------------------------------------------------------------------------

_EXCEPTION_STATUS_MAP: dict[type[RsvsError], int] = {
    SchemaVersionMismatchError: 409,
    SchemaValidationError: 422,
    InvalidModeError: 400,
    RustCoreUnavailableError: 503,
}


def _error_status(exc: RsvsError) -> int:
    """Return the most specific HTTP status code for an RSVS exception."""
    for exc_type in type(exc).__mro__:
        if exc_type in _EXCEPTION_STATUS_MAP:
            return _EXCEPTION_STATUS_MAP[exc_type]
    return 500


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    """Thin HTTP handler that routes to the mode implementations."""

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        # GET /health
        if parsed.path == "/health":
            self._send(200, {
                "status": "ok",
                "rust_core_available": is_rust_core_available(),
                "service": "rsvs-bridge",
                "timestamp": iso_now(),
                "atom_dir": str(CONFIG.atom_dir),
                "version": API_VERSION,
                "schema_version": SCHEMA_VERSION,
            })
            return

        # GET /latest
        if parsed.path == "/latest":
            qs = parse_qs(parsed.query or "")
            mode = (qs.get("mode", ["ingest"])[0] or "ingest").strip().lower()
            try:
                _normalize_view(qs.get("view", ["compact"])[0] or "compact")
            except SchemaValidationError:
                self._send(400, {"ok": False, "error": "invalid_view"})
                return
            if mode not in VALID_MODES:
                self._send(400, {"ok": False, "error": "invalid_mode", "mode": mode})
                return

            if mode == "ingest" and "mode" not in qs:
                # Backward-compatible payload for current frontend restore.
                try:
                    legacy = _read_latest_ingest_bundle()
                except RsvsError as exc:
                    self._send(_error_status(exc), {"ok": False, "error": str(exc)})
                    return
                if legacy is None:
                    self._send(404, {"ok": False, "error": "no_artifacts", "mode": "ingest"})
                    return
                self._send(200, {"ok": True, **legacy})
                return

            try:
                envelope = _read_latest_mode(mode)
            except RsvsError as exc:
                self._send(_error_status(exc), {"ok": False, "error": str(exc)})
                return
            if envelope is None:
                self._send(404, {"ok": False, "error": "no_artifacts", "mode": mode})
                return
            self._send(200, envelope)
            return

        # GET /status
        if parsed.path == "/status":
            r = _get_rsvs()
            if r is not None:
                try:
                    status = r.status()
                    self._send(200, {"ok": True, "status": status, "backend": "rust"})
                except Exception as exc:
                    self._send(500, {"ok": False, "error": str(exc)})
            else:
                self._send(200, {"ok": True, "status": {}, "backend": "unavailable"})
            return

        self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            data = json.loads(raw)
        except Exception:
            self._send(400, {"ok": False, "error": "invalid_json"})
            return

        # POST /run
        if parsed.path == "/run":
            mode = (data.get("mode") or "").strip().lower()
            text = (data.get("text") or "").strip()
            correlation_id = (data.get("correlation_id") or make_id("corr")).strip()
            options = data.get("options") if isinstance(data.get("options"), dict) else {}
            incoming_schema = data.get("schema_version")
            if incoming_schema not in (None, SCHEMA_VERSION):
                self._send(409, {
                    "ok": False,
                    "error": "schema_version_mismatch",
                    "expected": SCHEMA_VERSION,
                    "got": incoming_schema,
                })
                return

            if mode not in VALID_MODES:
                self._send(400, {"ok": False, "error": "invalid_mode", "mode": mode})
                return
            if not text:
                self._send(400, {"ok": False, "error": "text_required", "mode": mode})
                return

            try:
                payload = _run_mode(mode, text, correlation_id, options)
                self._send(200, payload)
            except RsvsError as exc:
                self._send(_error_status(exc), {"ok": False, "error": str(exc), "mode": mode})
            except Exception as exc:
                self._send(500, {"ok": False, "error": str(exc), "mode": mode})
            return

        # POST /ingest (backward-compatible endpoint)
        if parsed.path == "/ingest":
            text = (data.get("text") or "").strip()
            correlation_id = (data.get("correlation_id") or make_id("corr")).strip()
            if not text:
                self._send(400, {"ok": False, "error": "text_required"})
                return
            try:
                env = _run_mode("ingest", text, correlation_id, data.get("options") or {})
                self._send(200, {
                    "ok": True,
                    "correlation_id": env.get("correlation_id"),
                    "snapshot": env.get("result", {}).get("snapshot", {}),
                    "events": env.get("result", {}).get("events", []),
                    "messages": env.get("messages", []),
                    "files": env.get("files", {}),
                })
            except RsvsError as exc:
                self._send(_error_status(exc), {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send(500, {"ok": False, "error": str(exc)})
            return

        self._send(404, {"ok": False, "error": "not_found"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the RSVS bridge HTTP server."""
    backend = "rust" if is_rust_core_available() else "unavailable"
    print(f"[bridge] RSVS bridge server starting")
    print(f"[bridge] Backend: {backend}")
    print(f"[bridge] Listening on http://{CONFIG.host}:{CONFIG.port}")
    print(f"[bridge] Atom output dir: {CONFIG.atom_dir}")

    # Initialize the Rsvs singleton on startup
    r = _get_rsvs()
    if r is not None:
        status = r.status()
        print(f"[bridge] Rsvs instance ready: {status}")

    server = ThreadingHTTPServer((CONFIG.host, CONFIG.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] Shutting down...")
        _save_rsvs()
        print("[bridge] State saved. Goodbye.")


if __name__ == "__main__":
    main()
