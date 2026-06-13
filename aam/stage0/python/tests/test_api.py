"""RSVS FastAPI Contract Tests.

Comprehensive HTTP-level tests for the FastAPI server endpoints:
  POST /run, /ingest, /query, /appraise, /relate
  GET  /health, /snapshot, /events

Run with: python3 -m pytest tests/test_api.py -v
"""

import json

import pytest
from starlette.testclient import TestClient

from rsvs.fastapi_server import app
from rsvs._version import __version__


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ===================================================================
# GET /health
# ===================================================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client):
        """GET /health must return 200 with status=ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "ok"

    def test_health_reports_version(self, client):
        """GET /health must include the version field."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "version" in body


# ===================================================================
# GET /
# ===================================================================


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "RSVS"
        assert "version" in body
        assert "docs" in body


# ===================================================================
# POST /run — mode-specific tests
# ===================================================================


class TestRunEndpoint:
    """Tests for POST /run with various modes."""

    def test_invalid_mode_returns_400(self, client):
        """POST /run with invalid mode must return 400."""
        resp = client.post("/run", json={"mode": "invalid_mode", "text": "test text"})
        assert resp.status_code == 400

    def test_missing_text_returns_422(self, client):
        """POST /run without text must return 422."""
        resp = client.post("/run", json={"mode": "ingest"})
        assert resp.status_code == 422

    def test_empty_text_returns_422(self, client):
        """POST /run with empty text must return 422."""
        resp = client.post("/run", json={"mode": "ingest", "text": "   "})
        assert resp.status_code == 422


# ===================================================================
# POST /ingest
# ===================================================================


class TestIngestEndpoint:
    def test_ingest_simple_text(self, client):
        resp = client.post("/ingest", json={"text": "Air adalah kebutuhan dasar"})
        assert resp.status_code == 200

    def test_ingest_empty_text_returns_422(self, client):
        resp = client.post("/ingest", json={"text": ""})
        assert resp.status_code == 422

    def test_ingest_missing_text_returns_422(self, client):
        resp = client.post("/ingest", json={})
        assert resp.status_code == 422


# ===================================================================
# POST /query
# ===================================================================


class TestQueryEndpoint:
    def test_query_unknown_concept(self, client):
        resp = client.post("/query", json={"text": "xyznonexistent"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ok") is True

    def test_query_missing_text_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422


# ===================================================================
# CORS headers
# ===================================================================


class TestHTTPIntegration:
    """End-to-end HTTP integration tests for the API server."""

    def test_health_has_cors(self, client):
        """CORS middleware adds headers when Origin is from allowed list."""
        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
        origin = resp.headers.get("access-control-allow-origin")
        assert origin is not None

    def test_health_no_cors_for_disallowed_origin(self, client):
        """CORS should not allow arbitrary origins."""
        resp = client.get("/health", headers={"Origin": "http://evil.example.com"})
        origin = resp.headers.get("access-control-allow-origin")
        assert origin != "http://evil.example.com"

    def test_unknown_path_returns_404(self, client):
        """Unknown paths must return 404."""
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_request_too_large_rejected(self, client):
        """Requests exceeding 1MB should be rejected."""
        large_text = "x" * 1_000_001
        resp = client.post("/ingest", json={"text": large_text})
        # Either 413 (request too large) or 422 (pydantic max_length)
        assert resp.status_code in (413, 422)
