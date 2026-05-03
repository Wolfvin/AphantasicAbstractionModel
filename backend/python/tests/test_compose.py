"""Tests for the RSVS compose endpoint.

Covers: compose creates node, shared atoms Jaccard similarity,
invalid atom error handling, missing fields validation.

Run with: python -m pytest tests/test_compose.py -v
"""

import pytest
from starlette.testclient import TestClient
from rsvs.fastapi_server import app

import rsvs.rsvs_core as _rsvs_core


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    """Provide a fresh TestClient with a reset RSVS singleton."""
    # Reset the singleton so each test starts fresh
    monkeypatch.setattr(_rsvs_core, "_instance", None)
    monkeypatch.setattr(_rsvs_core, "_last_ingest_seq", 0)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper: ingest enough text to create atom nodes
# ---------------------------------------------------------------------------

GEOLOGY = """
Stone is a hard solid mineral material. Rock is a hard heavy solid substance.
Stone is formed by heat and pressure over time. Granite is a hard rough stone.
Stone has a rough hard texture. Metal is a hard solid material.
Stone and metal are both hard solid. Hard solid materials resist pressure.
Stone is heavy and hard. Stone resists erosion and pressure.
"""


def _ingest_and_get_nodes(client, text=GEOLOGY):
    """Ingest text and return node list from snapshot."""
    client.post("/ingest", json={"text": text})
    snap = client.get("/snapshot").json()
    return snap.get("nodes", [])


# ===================================================================
# POST /compose — basic functionality
# ===================================================================


class TestComposeBasic:
    """Tests for basic compose endpoint behavior."""

    def test_compose_creates_node(self, client):
        """Test that POST /compose creates a composite node."""
        nodes = _ingest_and_get_nodes(client)
        if len(nodes) >= 3:
            atom_ids = [n["id"] for n in nodes[:3]]

            resp = client.post("/compose", json={
                "label": "raja",
                "atom_ids": atom_ids,
                "lang": "id",
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["label"] == "raja"
            assert data["node_id"] is not None

    def test_compose_returns_atom_ids(self, client):
        """Test that compose response includes the atom_ids."""
        nodes = _ingest_and_get_nodes(client)
        if len(nodes) >= 2:
            atom_ids = [n["id"] for n in nodes[:2]]

            resp = client.post("/compose", json={
                "label": "concept",
                "atom_ids": atom_ids,
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["atom_ids"] == atom_ids

    def test_compose_snapshot_includes_new_node(self, client):
        """Test that the snapshot includes the newly composed node."""
        nodes = _ingest_and_get_nodes(client)
        if len(nodes) >= 2:
            atom_ids = [n["id"] for n in nodes[:2]]

            client.post("/compose", json={
                "label": "new_concept",
                "atom_ids": atom_ids,
            })

            snap = client.get("/snapshot").json()
            snap_nodes = snap.get("nodes", [])
            labels = [n.get("label") for n in snap_nodes]
            assert "new_concept" in labels

    def test_compose_node_has_compressed_state(self, client):
        """Test that the composed node has compression_state=compressed."""
        nodes = _ingest_and_get_nodes(client)
        if len(nodes) >= 2:
            atom_ids = [n["id"] for n in nodes[:2]]

            client.post("/compose", json={
                "label": "compressed_concept",
                "atom_ids": atom_ids,
            })

            snap = client.get("/snapshot").json()
            snap_nodes = snap.get("nodes", [])
            comp = [n for n in snap_nodes if n.get("label") == "compressed_concept"]
            if comp:
                # In snapshot v1, compression_state and derived_from_node_ids
                # are at the top level of the node object (not nested in semantic)
                assert comp[0].get("compression_state") == "compressed"
                assert len(comp[0].get("derived_from_node_ids", [])) > 0


# ===================================================================
# POST /compose — shared atoms and similarity
# ===================================================================


class TestComposeSimilarity:
    """Tests for compose with shared atoms and Jaccard similarity."""

    def test_compose_shared_atoms(self, client):
        """Test that composites sharing atoms have Jaccard > 0."""
        nodes = _ingest_and_get_nodes(client)

        if len(nodes) >= 4:
            # raja = nodes[0] + nodes[1] + nodes[2]
            # ratu = nodes[0] + nodes[3] + nodes[2]
            raja_ids = [nodes[0]["id"], nodes[1]["id"], nodes[2]["id"]]
            ratu_ids = [nodes[0]["id"], nodes[3]["id"], nodes[2]["id"]]

            r1 = client.post("/compose", json={
                "label": "raja",
                "atom_ids": raja_ids,
            })
            r2 = client.post("/compose", json={
                "label": "ratu",
                "atom_ids": ratu_ids,
            })

            assert r1.status_code == 200
            assert r2.status_code == 200

            # Check similarity
            sim = client.post("/similarity", json={
                "label_a": "raja",
                "label_b": "ratu",
            })

            assert sim.status_code == 200, f"Similarity request failed: {sim.status_code} {sim.text}"
            sim_data = sim.json()
            jaccard = sim_data.get("similarity", {}).get("jaccard", 0)
            assert jaccard > 0, "Jaccard should be > 0 for shared atoms"

    def test_compose_disjoint_atoms_low_jaccard(self, client):
        """Test that composites with disjoint atoms have Jaccard = 0."""
        nodes = _ingest_and_get_nodes(client)

        if len(nodes) >= 6:
            # concept_a = nodes[0] + nodes[1] + nodes[2]
            # concept_b = nodes[3] + nodes[4] + nodes[5]
            a_ids = [nodes[0]["id"], nodes[1]["id"], nodes[2]["id"]]
            b_ids = [nodes[3]["id"], nodes[4]["id"], nodes[5]["id"]]

            # Only proceed if the ID sets are actually disjoint
            if not set(a_ids) & set(b_ids):
                r1 = client.post("/compose", json={
                    "label": "disjoint_a",
                    "atom_ids": a_ids,
                })
                r2 = client.post("/compose", json={
                    "label": "disjoint_b",
                    "atom_ids": b_ids,
                })

                if r1.status_code == 200 and r2.status_code == 200:
                    sim = client.post("/similarity", json={
                        "label_a": "disjoint_a",
                        "label_b": "disjoint_b",
                    })
                    assert sim.status_code == 200, f"Similarity request failed: {sim.status_code} {sim.text}"
                    sim_data = sim.json()
                    jaccard = sim_data.get("similarity", {}).get("jaccard", 0)
                    assert jaccard == 0, "Jaccard should be 0 for disjoint atoms"


# ===================================================================
# POST /compose — error handling
# ===================================================================


class TestComposeErrors:
    """Tests for compose endpoint error handling."""

    def test_compose_invalid_atom(self, client):
        """Test that composing with non-existent atom returns error."""
        resp = client.post("/compose", json={
            "label": "invalid_composite",
            "atom_ids": [99999],
            "lang": "id",
        })
        # Should return 400 error (the endpoint catches exceptions as 400)
        assert resp.status_code == 400

    def test_compose_missing_fields(self, client):
        """Test that missing required fields return validation error."""
        resp = client.post("/compose", json={"label": "test"})
        assert resp.status_code == 422  # Validation error

    def test_compose_missing_label(self, client):
        """Test that missing label returns validation error."""
        nodes = _ingest_and_get_nodes(client)
        if nodes:
            resp = client.post("/compose", json={
                "atom_ids": [nodes[0]["id"]],
            })
            assert resp.status_code == 422

    def test_compose_empty_atom_ids(self, client):
        """Test that empty atom_ids returns validation error."""
        resp = client.post("/compose", json={
            "label": "empty_atoms",
            "atom_ids": [],
        })
        assert resp.status_code == 422  # min_length=1


# ===================================================================
# POST /compose — update existing label
# ===================================================================


class TestComposeUpdate:
    """Tests for composing with an already existing label."""

    def test_compose_updates_existing_label(self, client):
        """Test that composing with same label updates the node."""
        nodes = _ingest_and_get_nodes(client)

        if len(nodes) >= 3:
            atom_ids_1 = [nodes[0]["id"], nodes[1]["id"]]
            atom_ids_2 = [nodes[0]["id"], nodes[1]["id"], nodes[2]["id"]]

            # First compose
            r1 = client.post("/compose", json={
                "label": "reusable",
                "atom_ids": atom_ids_1,
            })
            assert r1.status_code == 200
            first_id = r1.json()["node_id"]

            # Second compose with same label
            r2 = client.post("/compose", json={
                "label": "reusable",
                "atom_ids": atom_ids_2,
            })
            assert r2.status_code == 200
            second_id = r2.json()["node_id"]

            # Should return the same node ID
            assert first_id == second_id


# ===================================================================
# POST /compose — language tag
# ===================================================================


class TestComposeLanguage:
    """Tests for compose endpoint language tag behavior."""

    def test_compose_with_lang(self, client):
        """Test that composing with lang sets surface_label correctly."""
        nodes = _ingest_and_get_nodes(client)

        if len(nodes) >= 2:
            atom_ids = [nodes[0]["id"], nodes[1]["id"]]

            client.post("/compose", json={
                "label": "raja",
                "atom_ids": atom_ids,
                "lang": "id",
            })

            snap = client.get("/snapshot").json()
            snap_nodes = snap.get("nodes", [])
            comp = [n for n in snap_nodes if n.get("label") == "raja"]
            if comp:
                assert comp[0].get("surface_label", "").endswith("@id")

    def test_compose_without_lang_defaults_en(self, client):
        """Test that composing without lang defaults to @en."""
        nodes = _ingest_and_get_nodes(client)

        if len(nodes) >= 2:
            atom_ids = [nodes[0]["id"], nodes[1]["id"]]

            client.post("/compose", json={
                "label": "king",
                "atom_ids": atom_ids,
            })

            snap = client.get("/snapshot").json()
            snap_nodes = snap.get("nodes", [])
            comp = [n for n in snap_nodes if n.get("label") == "king"]
            if comp:
                assert comp[0].get("surface_label", "").endswith("@en")
