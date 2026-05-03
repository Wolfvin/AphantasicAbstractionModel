import json

import pytest

from rsvs import bridge_server as bs


@pytest.fixture
def bridge_tmp_config(tmp_path, monkeypatch):
    cfg = bs.BridgeConfig(host="127.0.0.1", port=0, atom_dir=tmp_path / "atom")
    monkeypatch.setattr(bs, "CONFIG", cfg)
    return cfg


def test_ingest_writes_v42_schema_and_node_model(bridge_tmp_config):
    env = bs._run_mode("ingest", "water stone flow", "corr_test", {"view": "compact"})
    snapshot = env["result"]["snapshot"]
    assert snapshot["schema_version"] == bs.SCHEMA_VERSION
    assert snapshot["nodes"]
    for node in snapshot["nodes"]:
        assert node["kind"] == "node"
        semantic = node["semantic"]
        assert semantic["compression_state"] in {"raw", "compressed"}
        assert isinstance(semantic["derived_from_node_ids"], list)


def test_relate_detail_includes_compression_fields(bridge_tmp_config):
    bs._run_mode("ingest", "water stone flow", "corr_a", {"view": "compact"})
    env = bs._run_mode("relate", "water", "corr_b", {"view": "detail", "top_k": 5})
    result = env["result"]
    assert result["view"] == "detail"
    for node in result["related_nodes"]:
        assert "compression_state" in node
        assert "derived_from_node_ids" in node
        assert "derived_nodes" in node


def test_seed_nodes_are_locked_and_stable(bridge_tmp_config):
    env = bs._run_mode("ingest", "water stone flow", "corr_seed", {"view": "compact"})
    nodes = env["result"]["snapshot"]["nodes"]
    seed_nodes = [n for n in nodes if n.get("is_seed") is True]
    assert seed_nodes
    for seed in seed_nodes:
        assert seed["is_locked"] is True
        assert seed["status"] == "stable"
        assert seed["tier"] == 1
        assert float(seed["confidence"]) == 1.0


def test_confidence_accumulates_across_batches_for_repeated_signal(bridge_tmp_config):
    env1 = bs._run_mode("ingest", "voltage signal repeat", "corr_1", {"view": "compact"})
    node1 = next(n for n in env1["result"]["snapshot"]["nodes"] if n["label"] == "voltage")
    c1 = float(node1["confidence"])

    env2 = bs._run_mode("ingest", "voltage voltage signal repeat", "corr_2", {"view": "compact"})
    node2 = next(n for n in env2["result"]["snapshot"]["nodes"] if n["label"] == "voltage")
    c2 = float(node2["confidence"])
    assert c2 >= c1


def test_old_snapshot_is_rejected_hard_break(bridge_tmp_config):
    bridge_tmp_config.atom_dir.mkdir(parents=True, exist_ok=True)
    old_payload = {
        "snapshot_id": "old_1",
        "nodes": [{"id": 1001, "label": "legacy", "kind": "composite"}],
        "edges": [],
    }
    (bridge_tmp_config.atom_dir / "snapshot-20990101T000000Z.json").write_text(
        json.dumps(old_payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version_mismatch"):
        bs._read_latest_ingest_bundle()


def test_invalid_view_rejected():
    with pytest.raises(ValueError, match="invalid_view"):
        bs._normalize_view("graph")
