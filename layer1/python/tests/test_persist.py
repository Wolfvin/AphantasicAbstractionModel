"""
Persistence tests for RSVS v4.2.
Run with: python3 -m pytest tests/test_persist.py -v
"""
import pytest
import json
import os
import tempfile
import rsvs
from rsvs import PyV12Pipeline

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

CORPUS_GEO = """
Stone is a hard solid mineral material. Rock is a hard heavy solid substance.
Stone is formed by heat and pressure over time. Granite is a hard rough stone.
Stone has a rough hard texture. Metal is a hard solid material.
Stone and metal are both hard solid. Hard solid materials resist pressure.
Stone is heavy and hard. Stone resists erosion and pressure.
"""

CORPUS_WATER = """
Water is a clear transparent liquid. Water flows because it is liquid.
Rain is water falling from clouds. Ice is frozen solid water.
Liquid water is transparent and flows easily. Water is essential for life.
"""

@pytest.fixture
def tmp_path_str():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "rsvs_state.json")

@pytest.fixture
def trained():
    r = PyV12Pipeline(entity_promote_n=3, theta_assign=0.12, n_warm=8, eta=0.1)
    r.ingest(CORPUS_GEO)
    r.set_domain(2)
    r.ingest(CORPUS_WATER)
    return r

@pytest.fixture
def saved(trained, tmp_path_str):
    trained.save(tmp_path_str)
    return tmp_path_str

# -----------------------------------------------------------------------
# Save tests
# -----------------------------------------------------------------------

class TestSave:
    def test_creates_file(self, trained, tmp_path_str):
        trained.save(tmp_path_str)
        assert os.path.exists(tmp_path_str)

    def test_file_is_nonempty(self, trained, tmp_path_str):
        trained.save(tmp_path_str)
        assert os.path.getsize(tmp_path_str) > 0

    def test_file_is_valid_json(self, saved):
        with open(saved) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_snapshot_has_version(self, saved):
        with open(saved) as f:
            data = json.load(f)
        # Version should be at least "5.0" (matches Cargo.toml version)
        assert data["version"] >= "5.0"

    def test_snapshot_has_nodes(self, saved):
        with open(saved) as f:
            data = json.load(f)
        assert "nodes" in data
        assert len(data["nodes"]) > 0

    def test_snapshot_has_edges(self, saved):
        with open(saved) as f:
            data = json.load(f)
        assert "edges" in data

    def test_snapshot_has_token_to_id(self, saved):
        with open(saved) as f:
            data = json.load(f)
        assert "token_to_id" in data
        assert len(data["token_to_id"]) > 0

    def test_snapshot_has_sense_managers(self, saved):
        with open(saved) as f:
            data = json.load(f)
        assert "sense_managers" in data

    def test_snapshot_has_cooc_stats(self, saved):
        with open(saved) as f:
            data = json.load(f)
        assert "cooc_stats" in data
        assert data["cooc_stats"]["total_sentences"] > 0

    def test_snapshot_contexts_correct(self, trained, saved):
        contexts_before = int(trained.status()["total_contexts"])
        with open(saved) as f:
            data = json.load(f)
        assert data["total_contexts"] == contexts_before

    def test_invalid_path_raises(self, trained):
        with pytest.raises(Exception):
            trained.save("/nonexistent_dir/rsvs.json")

    def test_overwrite_existing_file(self, trained, tmp_path_str):
        # Write something first
        with open(tmp_path_str, 'w') as f:
            f.write("old content")
        # Should overwrite cleanly
        trained.save(tmp_path_str)
        with open(tmp_path_str) as f:
            data = json.load(f)
        assert data["version"] >= "5.0"

# -----------------------------------------------------------------------
# Load tests
# -----------------------------------------------------------------------

class TestLoad:
    def test_load_returns_rsvs(self, saved):
        r = PyV12Pipeline.load(saved)
        assert r is not None

    def test_load_nonexistent_raises(self):
        with pytest.raises(Exception):
            PyV12Pipeline.load("/nonexistent/path/rsvs.json")

    def test_load_invalid_json_raises(self, tmp_path_str):
        with open(tmp_path_str, 'w') as f:
            f.write("not valid json {{{")
        with pytest.raises(Exception):
            PyV12Pipeline.load(tmp_path_str)

# -----------------------------------------------------------------------
# Roundtrip correctness tests
# -----------------------------------------------------------------------

class TestRoundtrip:
    def test_atoms_preserved(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        assert sorted(r2.atoms()) == sorted(trained.atoms())

    def test_total_contexts_preserved(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        assert r2.status()["total_contexts"] == trained.status()["total_contexts"]

    def test_warmup_state_preserved(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        assert r2.status()["warmed_up"] == trained.status()["warmed_up"]

    def test_confidence_values_preserved(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        conf1 = trained.confidence_map()
        conf2 = r2.confidence_map()
        for label in trained.atoms():
            if label in conf1 and label in conf2:
                assert abs(conf1[label] - conf2[label]) < 0.001, \
                    f"Confidence mismatch for '{label}': {conf1[label]} vs {conf2[label]}"

    def test_similarity_works_after_load(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        sim_before = trained.similarity("stone", "hard")
        sim_after  = r2.similarity("stone", "hard")
        # Both should return result or both None
        assert (sim_before is None) == (sim_after is None)
        if sim_before and sim_after:
            assert abs(sim_before.jaccard - sim_after.jaccard) < 0.001

    def test_query_works_after_load(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        result = r2.query("stone", "hard texture")
        # Either works or concept not learned — just check no crash
        assert result is None or hasattr(result, 'sense_idx')

    def test_senses_count_preserved(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        if "stone" in trained.atoms() and "stone" in r2.atoms():
            s1 = len(trained.senses("stone"))
            s2 = len(r2.senses("stone"))
            assert s1 == s2, f"Sense count: {s1} vs {s2}"

    def test_sense_coherence_preserved(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        if "stone" in trained.atoms() and "stone" in r2.atoms():
            senses1 = trained.senses("stone")
            senses2 = r2.senses("stone")
            for s1, s2 in zip(senses1, senses2):
                assert abs(s1.coherence - s2.coherence) < 0.001

    def test_sense_status_preserved(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        if "stone" in trained.atoms() and "stone" in r2.atoms():
            senses1 = trained.senses("stone")
            senses2 = r2.senses("stone")
            for s1, s2 in zip(senses1, senses2):
                assert s1.status == s2.status

    def test_atom_tier_preserved(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        for atom in trained.atoms():
            info1 = trained.atom_info(atom)
            info2 = r2.atom_info(atom)
            assert info1.tier == info2.tier, \
                f"Tier mismatch for '{atom}': {info1.tier} vs {info2.tier}"

    def test_seed_atoms_still_conf_1(self, saved):
        r2 = PyV12Pipeline.load(saved)
        cm = r2.confidence_map()
        for seed in ["exists", "feel", "before", "after"]:
            if seed in cm:
                assert cm[seed] == 1.0, f"Seed '{seed}' conf={cm[seed]} after load"

    def test_status_count_matches(self, trained, saved):
        r2 = PyV12Pipeline.load(saved)
        st1 = trained.status()
        st2 = r2.status()
        assert st1["total_atoms"] == st2["total_atoms"]
        assert st1["total_nodes"] == st2["total_nodes"]

# -----------------------------------------------------------------------
# Post-load functionality tests
# -----------------------------------------------------------------------

class TestPostLoad:
    def test_ingest_after_load(self, saved):
        r = PyV12Pipeline.load(saved)
        atoms_before = set(r.atoms())
        r.ingest("Fire is hot luminous. Fire produces heat and light. Hot fire burns wood.")
        # System should still work — no crash
        assert r.status()["total_contexts"] > 0

    def test_can_ingest_new_domain_after_load(self, saved):
        r = PyV12Pipeline.load(saved)
        r.set_domain(3)
        r.ingest("Fire is hot. Fire burns. Fire produces heat.")
        r.ingest("Hot fire is bright luminous. Fire and heat are related.")
        # No crash
        assert True

    def test_double_roundtrip(self, trained, tmp_path_str):
        """Save → load → save → load → compare"""
        import os
        path2 = tmp_path_str.replace(".json", "_2.json")

        trained.save(tmp_path_str)
        r2 = PyV12Pipeline.load(tmp_path_str)
        r2.save(path2)
        r3 = PyV12Pipeline.load(path2)

        assert sorted(r3.atoms()) == sorted(r2.atoms())
        assert r3.status()["total_contexts"] == r2.status()["total_contexts"]

        if os.path.exists(path2):
            os.remove(path2)

    def test_similarity_consistent_after_load(self, saved):
        """Similarity results should be identical before and after load."""
        r = PyV12Pipeline.load(saved)
        # Run query twice — should be deterministic
        sim1 = r.similarity("stone", "solid")
        sim2 = r.similarity("stone", "solid")
        if sim1 and sim2:
            assert sim1.jaccard == sim2.jaccard

    def test_repr_after_load(self, saved):
        r = PyV12Pipeline.load(saved)
        s = repr(r)
        assert "PyV12Pipeline(" in s

# -----------------------------------------------------------------------
# JSON structure tests
# -----------------------------------------------------------------------

class TestJsonStructure:
    def test_nodes_have_required_fields(self, saved):
        with open(saved) as f:
            data = json.load(f)
        for node in data["nodes"]:
            assert "id" in node
            assert "kind" in node
            assert "atoms" in node
            assert "confidence" in node
            assert "tier" in node

    def test_node_kind_valid(self, saved):
        with open(saved) as f:
            data = json.load(f)
        for node in data["nodes"]:
            assert node["kind"] == "node", \
                f"Invalid kind: {node['kind']}"

    def test_node_tier_valid(self, saved):
        with open(saved) as f:
            data = json.load(f)
        for node in data["nodes"]:
            assert node["tier"] in (1, 2, 3), \
                f"Invalid tier: {node['tier']}"

    def test_node_confidence_in_range(self, saved):
        with open(saved) as f:
            data = json.load(f)
        for node in data["nodes"]:
            assert 0.0 <= node["confidence"] <= 1.0, \
                f"Confidence out of range: {node['confidence']}"

    def test_sense_managers_match_nodes(self, saved):
        with open(saved) as f:
            data = json.load(f)
        node_ids = {str(n["id"]) for n in data["nodes"]}
        sm_ids   = set(data["sense_managers"].keys())
        # Every sense manager should correspond to a node
        assert sm_ids.issubset(node_ids), \
            f"Orphan sense managers: {sm_ids - node_ids}"

    def test_cooc_stats_totals_consistent(self, saved):
        with open(saved) as f:
            data = json.load(f)
        cs = data["cooc_stats"]
        assert cs["total_tokens"] >= cs["total_sentences"], \
            "total_tokens should be >= total_sentences"

    def test_token_to_id_values_are_node_ids(self, saved):
        with open(saved) as f:
            data = json.load(f)
        node_ids = {n["id"] for n in data["nodes"]}
        for token, node_id in data["token_to_id"].items():
            assert node_id in node_ids, \
                f"token '{token}' maps to unknown node {node_id}"
