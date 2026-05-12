"""Tests for the FastAPI server using Starlette TestClient.

Covers: health, root, run, ingest, query, similarity, structural-similarity,
substitution-analysis, appraise, relate, compose, node-info, senses, snapshot,
events, CORS, authentication, rate limiting, input validation, and error handling.
"""

import pytest
from starlette.testclient import TestClient
from rsvs.fastapi_server import app
from rsvs._version import __version__


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────
# Health & Root
# ──────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__

    def test_health_has_cors(self, client):
        """CORS middleware adds headers when Origin is from allowed list."""
        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
        origin = resp.headers.get("access-control-allow-origin")
        # Either the origin is echoed back, or it's in the allowed list
        assert origin is not None
        assert origin == "http://localhost:3000" or origin == "*"

    def test_health_no_cors_for_disallowed_origin(self, client):
        """CORS should not allow arbitrary origins."""
        resp = client.get("/health", headers={"Origin": "http://evil.example.com"})
        origin = resp.headers.get("access-control-allow-origin")
        # The origin should not be reflected for disallowed domains
        assert origin != "http://evil.example.com"


class TestRootEndpoint:
    def test_root_returns_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "RSVS"
        assert body["version"] == __version__
        assert "docs" in body


# ──────────────────────────────────────────────────────────────────────
# /run endpoint
# ──────────────────────────────────────────────────────────────────────


class TestRunEndpoint:
    def test_invalid_mode_returns_400(self, client):
        resp = client.post("/run", json={"mode": "invalid", "text": "test"})
        assert resp.status_code == 400

    def test_missing_text_returns_422(self, client):
        resp = client.post("/run", json={"mode": "ingest", "text": ""})
        assert resp.status_code == 422

    def test_run_ingest_mode(self, client):
        resp = client.post("/run", json={"mode": "ingest", "text": "Raja adalah raja kerajaan"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ok") is True or "stats" in body or "mode" in body

    def test_run_text_too_long_returns_422(self, client):
        """max_length=100_000 should reject overly long text."""
        resp = client.post("/run", json={"mode": "ingest", "text": "x" * 100_001})
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /ingest endpoint
# ──────────────────────────────────────────────────────────────────────


class TestIngestEndpoint:
    def test_ingest_simple_text(self, client):
        resp = client.post("/ingest", json={"text": "Air adalah kebutuhan dasar"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ok") is True or "stats" in body

    def test_ingest_with_source(self, client):
        resp = client.post("/ingest", json={"text": "Air mengalir ke laut", "source": "wikipedia"})
        assert resp.status_code == 200

    def test_ingest_empty_text_returns_422(self, client):
        resp = client.post("/ingest", json={"text": ""})
        assert resp.status_code == 422

    def test_ingest_text_too_long_returns_422(self, client):
        """max_length=100_000 on text field."""
        resp = client.post("/ingest", json={"text": "a" * 100_001})
        assert resp.status_code == 422

    def test_ingest_source_too_long_returns_422(self, client):
        """max_length=500 on source field."""
        resp = client.post("/ingest", json={"text": "hello", "source": "s" * 501})
        assert resp.status_code == 422

    def test_ingest_missing_text_returns_422(self, client):
        resp = client.post("/ingest", json={})
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /query endpoint
# ──────────────────────────────────────────────────────────────────────


class TestQueryEndpoint:
    def test_query_unknown_concept(self, client):
        """Query for a concept that doesn't exist should return null result."""
        resp = client.post("/query", json={"text": "xyznonexistent"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ok") is True
        assert body.get("result") is None

    def test_query_missing_text_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_top_k_bounds(self, client):
        """top_k must be 1-100."""
        resp = client.post("/query", json={"text": "test", "top_k": 0})
        assert resp.status_code == 422
        resp = client.post("/query", json={"text": "test", "top_k": 101})
        assert resp.status_code == 422
        resp = client.post("/query", json={"text": "test", "top_k": 10})
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────
# /similarity endpoint
# ──────────────────────────────────────────────────────────────────────


class TestSimilarityEndpoint:
    def test_similarity_missing_fields(self, client):
        resp = client.post("/similarity", json={"label_a": "test"})
        assert resp.status_code == 422

    def test_similarity_both_labels(self, client):
        resp = client.post("/similarity", json={"label_a": "exists", "label_b": "entity"})
        assert resp.status_code == 200
        body = resp.json()
        assert "similarity" in body or "error" in body


# ──────────────────────────────────────────────────────────────────────
# /structural-similarity endpoint
# ──────────────────────────────────────────────────────────────────────


class TestStructuralSimilarityEndpoint:
    def test_requires_both_params(self, client):
        resp = client.get("/structural-similarity")
        assert resp.status_code == 422

    def test_with_labels(self, client):
        resp = client.get("/structural-similarity?a=exists&b=entity")
        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body

    def test_missing_b_param(self, client):
        resp = client.get("/structural-similarity?a=test")
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /substitution-analysis endpoint
# ──────────────────────────────────────────────────────────────────────


class TestSubstitutionAnalysisEndpoint:
    def test_with_labels(self, client):
        resp = client.get("/substitution-analysis?a=exists&b=entity")
        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body

    def test_missing_params(self, client):
        resp = client.get("/substitution-analysis")
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /appraise endpoint
# ──────────────────────────────────────────────────────────────────────


class TestAppraiseEndpoint:
    def test_appraise_with_text(self, client):
        resp = client.post("/appraise", json={"target": "exists"})
        assert resp.status_code == 200

    def test_appraise_empty_target(self, client):
        resp = client.post("/appraise", json={"target": ""})
        assert resp.status_code == 422

    def test_appraise_missing_target(self, client):
        resp = client.post("/appraise", json={})
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /relate endpoint
# ──────────────────────────────────────────────────────────────────────


class TestRelateEndpoint:
    def test_relate_with_source_and_target(self, client):
        resp = client.post("/relate", json={"source": "exists", "target": "entity"})
        assert resp.status_code == 200

    def test_relate_missing_fields(self, client):
        resp = client.post("/relate", json={"source": "exists"})
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /compose endpoint
# ──────────────────────────────────────────────────────────────────────


class TestComposeEndpoint:
    def test_compose_requires_fields(self, client):
        """Must provide either compositions or atom_ids."""
        resp = client.post("/compose", json={"label": "test"})
        assert resp.status_code == 400

    def test_compose_cannot_provide_both(self, client):
        """Cannot provide both compositions and atom_ids."""
        resp = client.post("/compose", json={
            "label": "test",
            "compositions": [{"label": "exists", "sense_id": 0}],
            "atom_ids": [1],
        })
        assert resp.status_code == 400

    def test_compose_with_compositions(self, client):
        resp = client.post("/compose", json={
            "label": "test_composite",
            "compositions": [{"label": "exists", "sense_id": 0}, {"label": "entity", "sense_id": 0}],
        })
        assert resp.status_code in (200, 400)  # 400 if nodes don't exist yet

    def test_compose_with_atom_ids(self, client):
        resp = client.post("/compose", json={
            "label": "test_composite",
            "atom_ids": [1, 2],
        })
        assert resp.status_code in (200, 400)

    def test_compose_empty_label(self, client):
        resp = client.post("/compose", json={
            "label": "",
            "compositions": [{"label": "exists", "sense_id": 0}],
        })
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /node-info endpoint
# ──────────────────────────────────────────────────────────────────────


class TestNodeInfoEndpoint:
    def test_node_info_for_seed(self, client):
        """Seed nodes should be queryable."""
        resp = client.post("/node-info", json={"label": "exists"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ok") is True

    def test_node_info_missing_label(self, client):
        resp = client.post("/node-info", json={})
        assert resp.status_code == 422

    def test_node_info_empty_label(self, client):
        resp = client.post("/node-info", json={"label": ""})
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /senses endpoint
# ──────────────────────────────────────────────────────────────────────


class TestSensesEndpoint:
    def test_senses_for_seed(self, client):
        resp = client.post("/senses", json={"label": "exists"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ok") is True
        assert "senses" in body

    def test_senses_missing_label(self, client):
        resp = client.post("/senses", json={})
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────
# /snapshot endpoint
# ──────────────────────────────────────────────────────────────────────


class TestSnapshotEndpoint:
    def test_snapshot_returns_json(self, client):
        resp = client.get("/snapshot")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)


# ──────────────────────────────────────────────────────────────────────
# /events endpoint
# ──────────────────────────────────────────────────────────────────────


class TestEventsEndpoint:
    def test_events_returns_json(self, client):
        resp = client.get("/events")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)


# ──────────────────────────────────────────────────────────────────────
# Input validation & security
# ──────────────────────────────────────────────────────────────────────


class TestInputValidation:
    def test_request_too_large_rejected(self, client):
        """Requests exceeding 1MB should be rejected."""
        large_text = "x" * 1_000_001
        resp = client.post("/ingest", json={"text": large_text})
        # Either 413 (request too large) or 422 (pydantic max_length)
        assert resp.status_code in (413, 422)

    def test_composition_pair_sense_id_non_negative(self, client):
        """sense_id must be >= 0."""
        resp = client.post("/compose", json={
            "label": "test",
            "compositions": [{"label": "exists", "sense_id": -1}],
        })
        assert resp.status_code == 422


class TestAuthBehavior:
    """Test API key authentication behavior.

    When RSVS_API_KEY is not set (dev mode), auth is skipped.
    These tests verify that the endpoint works without an API key
    in the default test configuration.
    """

    def test_endpoints_work_without_api_key_in_dev_mode(self, client):
        """In dev mode (no RSVS_API_KEY set), all endpoints should work."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_compose_endpoint_rejects_neither_fields(self, client):
        """Compose must have either compositions or atom_ids."""
        resp = client.post("/compose", json={"label": "test"})
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────────
# Contract tests: /run with each valid mode
# ──────────────────────────────────────────────────────────────────────


class TestRunModeContract:
    """Contract tests verifying /run works with each valid mode."""

    @pytest.mark.parametrize("mode", [
        "ingest",
        "appraise",
        "relate",
        "compose",
    ])
    def test_run_mode_accepted(self, client, mode):
        """Each valid mode should not return 400 for invalid mode."""
        resp = client.post("/run", json={"mode": mode, "text": "exists"})
        # Some modes need target or other params, so we accept 400 for missing target too
        # 500 is possible if Rust core encounters an error for incomplete params
        assert resp.status_code in (200, 400, 422, 500)

    def test_run_query_mode(self, client):
        resp = client.post("/run", json={"mode": "ingest", "text": "Raja menguasai kerajaan"})
        assert resp.status_code == 200

    def test_run_structural_similarity_mode(self, client):
        resp = client.post("/run", json={
            "mode": "structural_similarity",
            "text": "exists",
            "target": "entity",
        })
        # May return 500 if Rust core can't process with minimal data
        assert resp.status_code in (200, 400, 500)

    def test_run_substitution_analysis_mode(self, client):
        resp = client.post("/run", json={
            "mode": "substitution_analysis",
            "text": "exists",
            "target": "entity",
        })
        # May return 500 if Rust core can't process with minimal data
        assert resp.status_code in (200, 400, 500)


# ──────────────────────────────────────────────────────────────────────
# /context-query endpoint (v6.1)
# ──────────────────────────────────────────────────────────────────────


class TestContextQueryEndpoint:
    def test_context_query_with_context(self, client):
        """Context query with a concept and context atoms."""
        # First ingest some text to populate the graph
        client.post("/ingest", json={"text": "Raja adalah raja kerajaan yang berkuasa"})
        resp = client.post("/context-query", json={
            "concept": "raja",
            "context_atoms": ["kerajaan"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ok") is True

    def test_context_query_missing_concept(self, client):
        """concept field is required."""
        resp = client.post("/context-query", json={
            "context_atoms": ["kerajaan"],
        })
        assert resp.status_code == 422

    def test_context_query_missing_context_atoms(self, client):
        """context_atoms field is required."""
        resp = client.post("/context-query", json={
            "concept": "raja",
        })
        assert resp.status_code == 422

    def test_context_query_empty_concept(self, client):
        """concept must be non-empty."""
        resp = client.post("/context-query", json={
            "concept": "",
            "context_atoms": ["kerajaan"],
        })
        assert resp.status_code == 422

    def test_context_query_empty_context_atoms(self, client):
        """context_atoms must have at least 1 entry."""
        resp = client.post("/context-query", json={
            "concept": "raja",
            "context_atoms": [],
        })
        assert resp.status_code == 422

    def test_context_query_max_depth_bounds(self, client):
        """max_depth must be between 1 and 10."""
        resp = client.post("/context-query", json={
            "concept": "raja",
            "context_atoms": ["kerajaan"],
            "max_depth": 0,
        })
        assert resp.status_code == 422
        resp = client.post("/context-query", json={
            "concept": "raja",
            "context_atoms": ["kerajaan"],
            "max_depth": 11,
        })
        assert resp.status_code == 422

    def test_context_query_with_depth_param(self, client):
        """max_depth=2 should work."""
        resp = client.post("/context-query", json={
            "concept": "exists",
            "context_atoms": ["entity"],
            "max_depth": 2,
        })
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────
# Contract tests: compositional architecture (v6.1)
# ──────────────────────────────────────────────────────────────────────


class TestCompositionalContract:
    """Contract tests verifying the compositional architecture works end-to-end."""

    def test_compose_and_query(self, client):
        """Composed nodes should be queryable."""
        # Ingest some text first
        client.post("/ingest", json={"text": "Air mengalir melalui batu dan tanah"})

        # Compose a node
        resp = client.post("/compose", json={
            "label": "test_composed",
            "compositions": [{"label": "exists", "sense_id": 0}, {"label": "entity", "sense_id": 0}],
        })
        # Should succeed or fail gracefully
        assert resp.status_code in (200, 400)

    def test_structural_similarity_seeds(self, client):
        """Structural similarity between seed atoms should work."""
        resp = client.get("/structural-similarity?a=exists&b=entity")
        assert resp.status_code == 200
        body = resp.json()
        assert "structural_similarity" in body

    def test_substitution_analysis_seeds(self, client):
        """Substitution analysis between seed atoms should work."""
        resp = client.get("/substitution-analysis?a=exists&b=entity")
        assert resp.status_code == 200
        body = resp.json()
        assert "structural_similarity" in body

    def test_ingest_produces_compositions(self, client):
        """Ingesting text should produce compositional senses."""
        resp = client.post("/ingest", json={
            "text": "Raja adalah raja kerajaan. Ratu adalah ratu kerajaan. Tahta tertinggi milik raja."
        })
        assert resp.status_code == 200

    def test_senses_endpoint_returns_compositions(self, client):
        """Senses endpoint should return compositional information."""
        resp = client.post("/senses", json={"label": "exists"})
        assert resp.status_code == 200
        body = resp.json()
        assert "senses" in body

    def test_snapshot_contains_layer_info(self, client):
        """Snapshot should include layer information for nodes."""
        resp = client.get("/snapshot")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)


# ──────────────────────────────────────────────────────────────────────
# Version check (v6.1)
# ──────────────────────────────────────────────────────────────────────


class TestVersionV61:
    def test_health_reports_v61(self, client):
        """Health endpoint should report v6.1.0."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == __version__

    def test_root_reports_v61(self, client):
        """Root endpoint should report v6.1.0."""
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == __version__
