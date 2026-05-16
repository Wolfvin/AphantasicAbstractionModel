"""RSVS Bridge API Contract Tests — v4.2.

Comprehensive HTTP-level tests for the bridge server endpoints:
  POST /run  — mode=ingest|appraise|relate
  GET  /latest?mode=ingest|appraise|relate
  GET  /health
  GET  /status

Run with: python3 -m pytest tests/test_api.py -v
"""

import json
import threading
import time
import http.client

import pytest

from rsvs.config import BridgeConfig, SCHEMA_VERSION
from rsvs.modes import _run_mode, _read_latest_ingest_bundle, _read_latest_mode
from rsvs.fastapi_server import app
import rsvs.rsvs_core as _rsvs_core
import rsvs.config as _config_mod
import rsvs.modes as _modes_mod
import rsvs.artifacts as _artifacts_mod

# Modules that bind CONFIG from rsvs.config — all must be patched for temp-dir isolation
_CONFIG_MODULES = (_config_mod, _modes_mod, _artifacts_mod, _rsvs_core)


# ---------------------------------------------------------------------------
# Sample corpora
# ---------------------------------------------------------------------------

GEOLOGY = """
Stone is a hard solid mineral material. Rock is a hard heavy solid substance.
Stone is formed by heat and pressure over time. Granite is a hard rough stone.
Stone has a rough hard texture. Metal is a hard solid material.
Stone and metal are both hard solid materials. Hard solid materials resist pressure.
Stone is heavy and hard. Hard stone resists erosion and pressure.
"""

WATER = """
Water is a clear transparent liquid. Water flows because it is liquid.
Rain is water falling from clouds. Ice is frozen solid water.
Water dissolves many solid materials. Liquid water becomes ice when cold.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge_tmp_config(tmp_path, monkeypatch):
    """Patch CONFIG so artifact I/O goes to a temp dir."""
    cfg = BridgeConfig(host="127.0.0.1", port=0, atom_dir=tmp_path / "atom")
    for mod in _CONFIG_MODULES:
        monkeypatch.setattr(mod, "CONFIG", cfg)
    return cfg


@pytest.fixture
def fresh_bridge(bridge_tmp_config, monkeypatch):
    """Reset the module-level singleton so each test starts fresh."""
    monkeypatch.setattr(_rsvs_core, "_instance", None)
    monkeypatch.setattr(_rsvs_core, "_last_ingest_seq", 0)
    return bridge_tmp_config


# ---------------------------------------------------------------------------
# Helper: start the bridge server on a random port in a thread
# ---------------------------------------------------------------------------


class _ServerCtx:
    """Holds a running ThreadingHTTPServer + its port."""

    def __init__(self, atom_dir):
        self.server = None
        self.port = None
        self.thread = None
        self.atom_dir = atom_dir

    def start(self, monkeypatch):
        """Start the server on a random port, patch CONFIG accordingly."""
        cfg = BridgeConfig(host="127.0.0.1", port=0, atom_dir=self.atom_dir)
        for mod in _CONFIG_MODULES:
            monkeypatch.setattr(mod, "CONFIG", cfg)
        monkeypatch.setattr(_rsvs_core, "_instance", None)
        monkeypatch.setattr(_rsvs_core, "_last_ingest_seq", 0)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def stop(self):
        if self.server:
            self.server.shutdown()

    def request(self, method, path, body=None):
        """Make an HTTP request and return (status, headers_dict, body_str)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"} if body else {}
        conn.request(
            method,
            path,
            body=json.dumps(body) if body else None,
            headers=headers,
        )
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        hdrs = dict(resp.getheaders())
        conn.close()
        return resp.status, hdrs, data


@pytest.fixture
def server_ctx(tmp_path, monkeypatch):
    ctx = _ServerCtx(tmp_path / "atom")
    ctx.start(monkeypatch)
    yield ctx
    ctx.stop()


# ===================================================================
# GET /health
# ===================================================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, server_ctx):
        """GET /health must return 200 with status=ok."""
        status, _, data = server_ctx.request("GET", "/health")
        assert status == 200
        body = json.loads(data)
        assert body.get("status") == "ok"

    def test_health_reports_rust_core_status(self, server_ctx):
        """GET /health must include rust_core_available boolean."""
        status, _, data = server_ctx.request("GET", "/health")
        assert status == 200
        body = json.loads(data)
        assert "rust_core_available" in body
        assert isinstance(body["rust_core_available"], bool)

    def test_health_includes_schema_version(self, server_ctx):
        """GET /health must include the current schema_version."""
        status, _, data = server_ctx.request("GET", "/health")
        body = json.loads(data)
        assert body.get("schema_version") == SCHEMA_VERSION

    def test_health_includes_service_name(self, server_ctx):
        """GET /health must identify the service."""
        status, _, data = server_ctx.request("GET", "/health")
        body = json.loads(data)
        assert body.get("service") == "rsvs-bridge"


# ===================================================================
# GET /status
# ===================================================================


class TestStatusEndpoint:
    """Tests for GET /status."""

    def test_status_returns_200(self, server_ctx):
        """GET /status must return 200."""
        status, _, data = server_ctx.request("GET", "/status")
        assert status == 200
        body = json.loads(data)
        assert isinstance(body, dict)

    def test_status_includes_backend(self, server_ctx):
        """GET /status must report which backend is in use."""
        status, _, data = server_ctx.request("GET", "/status")
        body = json.loads(data)
        assert "backend" in body
        assert body["backend"] in ("rust", "unavailable")


# ===================================================================
# POST /run — mode-specific tests
# ===================================================================


class TestRunEndpoint:
    """Tests for POST /run with various modes."""

    def test_ingest_mode_returns_snapshot(self, fresh_bridge):
        """POST /run mode=ingest returns a snapshot in the envelope."""
        env = _run_mode("ingest", GEOLOGY, "corr_ingest_1", {"view": "compact"})
        assert env["ok"] is True
        assert "result" in env
        snapshot = env["result"]["snapshot"]
        assert snapshot["schema_version"] == SCHEMA_VERSION
        assert len(snapshot["nodes"]) > 0

    def test_ingest_mode_creates_artifacts(self, fresh_bridge):
        """POST /run mode=ingest writes snapshot, events, report files."""
        env = _run_mode("ingest", GEOLOGY, "corr_artifacts", {"view": "compact"})
        files = env["files"]
        assert "snapshot" in files
        assert "events" in files
        assert "report" in files
        from pathlib import Path

        for key in ("snapshot", "events", "report"):
            assert Path(files[key]).exists(), f"{key} file should exist"

    def test_appraise_mode_returns_verdict(self, fresh_bridge):
        """POST /run mode=appraise returns a verdict after ingest."""
        _run_mode("ingest", GEOLOGY, "corr_prep", {"view": "compact"})
        env = _run_mode(
            "appraise", "stone is hard and solid", "corr_appraise_1", {"view": "compact"}
        )
        result = env["result"]
        assert "verdict" in result
        assert result["verdict"] in ("agree", "mixed", "disagree")
        assert "stance" in result
        assert "agree" in result["stance"]
        assert "disagree" in result["stance"]
        assert "evidence" in result

    def test_relate_mode_returns_related_nodes(self, fresh_bridge):
        """POST /run mode=relate returns related_nodes and related_edges."""
        _run_mode("ingest", GEOLOGY, "corr_prep_r", {"view": "compact"})
        env = _run_mode("relate", "stone", "corr_relate_1", {"view": "compact"})
        result = env["result"]
        assert "related_nodes" in result
        assert "related_edges" in result
        assert "query_terms" in result

    def test_invalid_mode_returns_400(self, server_ctx):
        """POST /run with invalid mode must return 400."""
        status, _, data = server_ctx.request(
            "POST",
            "/run",
            body={"mode": "invalid_mode", "text": "test text"},
        )
        assert status == 400
        body = json.loads(data)
        assert body.get("error") == "invalid_mode"

    def test_missing_text_returns_400(self, server_ctx):
        """POST /run without text must return 400."""
        status, _, data = server_ctx.request(
            "POST",
            "/run",
            body={"mode": "ingest"},
        )
        assert status == 400
        body = json.loads(data)
        assert body.get("error") == "text_required"

    def test_empty_text_returns_400(self, server_ctx):
        """POST /run with empty text must return 400."""
        status, _, data = server_ctx.request(
            "POST",
            "/run",
            body={"mode": "ingest", "text": "   "},
        )
        assert status == 400
        body = json.loads(data)
        assert body.get("error") == "text_required"

    def test_ingest_snapshot_has_v6_schema(self, fresh_bridge):
        """Ingest snapshot must use schema_version v6.0."""
        env = _run_mode("ingest", GEOLOGY, "corr_v6", {"view": "compact"})
        snapshot = env["result"]["snapshot"]
        assert snapshot["schema_version"] == "v6.0"

    def test_ingest_nodes_have_kind_node(self, fresh_bridge):
        """All nodes in an ingest snapshot must have kind='node'."""
        env = _run_mode("ingest", GEOLOGY, "corr_kind", {"view": "compact"})
        nodes = env["result"]["snapshot"]["nodes"]
        for node in nodes:
            assert node["kind"] == "node", f"Node {node.get('id')} has kind={node['kind']}"

    def test_ingest_seed_nodes_have_correct_invariants(self, fresh_bridge):
        """Seed nodes must be locked, tier=1, confidence=1.0, status=stable."""
        env = _run_mode("ingest", GEOLOGY, "corr_seeds", {"view": "compact"})
        nodes = env["result"]["snapshot"]["nodes"]
        seed_nodes = [n for n in nodes if n.get("is_seed") is True]
        assert len(seed_nodes) > 0, "Should have seed nodes"
        for seed in seed_nodes:
            assert seed["is_locked"] is True
            assert seed["tier"] == 1
            assert float(seed["confidence"]) == 1.0
            assert seed["status"] == "stable"
            assert seed["kind"] == "node"


# ===================================================================
# GET /latest
# ===================================================================


class TestLatestEndpoint:
    """Tests for GET /latest."""

    def test_latest_returns_most_recent_snapshot(self, fresh_bridge):
        """GET /latest returns the most recently ingested snapshot."""
        _run_mode("ingest", GEOLOGY, "corr_latest", {"view": "compact"})
        result = _read_latest_ingest_bundle()
        assert result is not None
        assert "snapshot" in result
        snapshot = result["snapshot"]
        assert snapshot["schema_version"] == SCHEMA_VERSION
        assert len(snapshot["nodes"]) > 0

    def test_latest_with_mode_filter(self, fresh_bridge):
        """GET /latest?mode=appraise returns appraise artifacts."""
        _run_mode("ingest", GEOLOGY, "corr_prep_m", {"view": "compact"})
        _run_mode("appraise", "stone is hard", "corr_appraise_m", {"view": "compact"})
        # _read_latest_mode should return appraise artifacts
        envelope = _read_latest_mode("appraise")
        assert envelope is not None
        assert envelope["mode"] == "appraise"

    def test_latest_no_artifacts_returns_none(self, fresh_bridge):
        """GET /latest with no artifacts returns None / 404."""
        result = _read_latest_ingest_bundle()
        assert result is None

    def test_latest_ingest_after_ingest(self, fresh_bridge):
        """GET /latest?mode=ingest returns ingest artifacts after ingest."""
        _run_mode("ingest", GEOLOGY, "corr_ing_latest", {"view": "compact"})
        envelope = _read_latest_mode("ingest")
        assert envelope is not None
        assert envelope["mode"] == "ingest"
        assert "result" in envelope


# ===================================================================
# Schema validation (snapshot contract)
# ===================================================================


class TestSchemaValidation:
    """Tests for snapshot schema validation through the bridge."""

    def test_snapshot_has_schema_version(self, fresh_bridge):
        """Snapshot must contain schema_version field."""
        env = _run_mode("ingest", GEOLOGY, "corr_schema_v", {"view": "compact"})
        snapshot = env["result"]["snapshot"]
        assert "schema_version" in snapshot
        assert snapshot["schema_version"] == SCHEMA_VERSION

    def test_nodes_have_surface_label_with_locale(self, fresh_bridge):
        """All nodes must have surface_label with @locale."""
        env = _run_mode("ingest", GEOLOGY, "corr_locale", {"view": "compact"})
        nodes = env["result"]["snapshot"]["nodes"]
        for node in nodes:
            assert "@" in node.get("surface_label", ""), (
                f"Node {node.get('id')} surface_label missing locale"
            )

    def test_compressed_nodes_have_derived_ids(self, fresh_bridge):
        """Compressed nodes must have non-empty derived_from_node_ids."""
        env = _run_mode("ingest", GEOLOGY, "corr_comp", {"view": "compact"})
        nodes = env["result"]["snapshot"]["nodes"]
        compressed = [n for n in nodes if n.get("semantic", {}).get("compression_state") == "compressed"]
        for node in compressed:
            derived = node["semantic"].get("derived_from_node_ids", [])
            assert len(derived) > 0, (
                f"Compressed node {node.get('id')} must have derived_from_node_ids"
            )

    def test_seed_nodes_are_locked_tier1_stable(self, fresh_bridge):
        """Seed nodes must have is_locked=True, tier=1, confidence=1.0, status=stable."""
        env = _run_mode("ingest", GEOLOGY, "corr_seed_inv", {"view": "compact"})
        nodes = env["result"]["snapshot"]["nodes"]
        for node in nodes:
            if node.get("is_seed") is True:
                assert node["is_locked"] is True
                assert node["tier"] == 1
                assert float(node["confidence"]) == 1.0
                assert node["status"] == "stable"


# ===================================================================
# HTTP-level integration tests
# ===================================================================


class TestHTTPIntegration:
    """End-to-end HTTP integration tests for the bridge server."""

    def test_cors_headers(self, server_ctx):
        """Responses must include CORS headers."""
        status, hdrs, _ = server_ctx.request("GET", "/health")
        assert status == 200
        assert hdrs.get("Access-Control-Allow-Origin") == "*"

    def test_options_returns_ok(self, server_ctx):
        """OPTIONS request must return 200."""
        status, _, data = server_ctx.request("OPTIONS", "/run")
        assert status == 200
        body = json.loads(data)
        assert body.get("ok") is True

    def test_unknown_path_returns_404(self, server_ctx):
        """Unknown paths must return 404."""
        status, _, data = server_ctx.request("GET", "/nonexistent")
        assert status == 404

    def test_post_unknown_path_returns_404(self, server_ctx):
        """POST to unknown paths must return 404."""
        status, _, data = server_ctx.request("POST", "/unknown", body={"mode": "ingest"})
        assert status == 404

    def test_schema_version_mismatch_returns_409(self, server_ctx):
        """POST /run with wrong schema_version must return 409."""
        status, _, data = server_ctx.request(
            "POST",
            "/run",
            body={"mode": "ingest", "text": "test", "schema_version": "v3.0"},
        )
        assert status == 409
        body = json.loads(data)
        assert body.get("error") == "schema_version_mismatch"

    def test_multiple_ingests_accumulate(self, fresh_bridge):
        """Multiple sequential ingests must accumulate nodes."""
        env1 = _run_mode("ingest", GEOLOGY, "corr_multi_1", {"view": "compact"})
        n1 = env1["result"]["stats"]["node_count"]
        env2 = _run_mode("ingest", WATER, "corr_multi_2", {"view": "compact"})
        n2 = env2["result"]["stats"]["node_count"]
        assert n2 >= n1, f"Second ingest should not reduce node count: {n1} -> {n2}"

    def test_appraise_after_ingest_has_verdict(self, fresh_bridge):
        """Appraise after ingest must produce a valid verdict."""
        _run_mode("ingest", GEOLOGY, "corr_prep_ai", {"view": "compact"})
        env = _run_mode(
            "appraise", "stone is hard solid material", "corr_appraise_ai", {"view": "compact"}
        )
        result = env["result"]
        assert result["verdict"] in ("agree", "mixed", "disagree")
        assert 0 <= result["stance"]["agree"] <= 100
        assert 0 <= result["stance"]["disagree"] <= 100

    def test_relate_after_ingest_finds_nodes(self, fresh_bridge):
        """Relate after ingest must return list results."""
        _run_mode("ingest", GEOLOGY, "corr_prep_ri", {"view": "compact"})
        env = _run_mode("relate", "exists", "corr_relate_ri", {"view": "compact"})
        result = env["result"]
        assert isinstance(result["related_nodes"], list)
        assert isinstance(result["related_edges"], list)
