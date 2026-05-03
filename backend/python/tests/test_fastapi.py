"""Tests for the FastAPI server using Starlette TestClient."""
import pytest
from starlette.testclient import TestClient
from rsvs.fastapi_server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "4.2.0"

    def test_health_has_cors(self, client):
        # CORS middleware only adds headers when Origin is present
        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert resp.headers.get("access-control-allow-origin") == "*"


class TestRootEndpoint:
    def test_root_returns_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "RSVS"
        assert body["version"] == "4.2.0"
        assert "docs" in body


class TestRunEndpoint:
    def test_invalid_mode_returns_400(self, client):
        resp = client.post("/run", json={"mode": "invalid", "text": "test"})
        assert resp.status_code == 400

    def test_missing_text_returns_422(self, client):
        # Pydantic will validate min_length=1
        resp = client.post("/run", json={"mode": "ingest", "text": ""})
        assert resp.status_code == 422
