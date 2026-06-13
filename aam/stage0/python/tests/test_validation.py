"""Tests for RSVS schema validation module.

Covers:
  - Semantic node validation (raw, compressed, seed, error cases)
  - Snapshot contract validation
  - View and language normalizers

Run with: python3 -m pytest tests/test_validation.py -v
"""

import pytest

from rsvs.validation import (
    _validate_semantic_node,
    _validate_snapshot_contract,
    _normalize_view,
    _normalize_lang,
)
from rsvs.config import SCHEMA_VERSION
from rsvs.exceptions import (
    SchemaVersionMismatchError,
    SchemaValidationError,
    InvariantViolationError,
)


# ---------------------------------------------------------------------------
# Helpers: node builders
# ---------------------------------------------------------------------------


def _raw_node(**overrides):
    """Build a minimal valid raw (non-seed) node."""
    node = {
        "id": 10,
        "label": "test",
        "surface_label": "test@en",
        "kind": "node",
        "tier": 3,
        "confidence": 0.25,
        "status": "new",
        "is_seed": False,
        "is_locked": False,
        "semantic": {
            "compression_state": "raw",
            "derived_from_node_ids": [],
        },
    }
    node.update(overrides)
    return node


def _seed_node(**overrides):
    """Build a minimal valid seed node."""
    node = {
        "id": 1,
        "label": "exists",
        "surface_label": "exists@en",
        "kind": "node",
        "tier": 1,
        "confidence": 1.0,
        "status": "stable",
        "is_seed": True,
        "is_locked": True,
        "semantic": {
            "compression_state": "raw",
            "derived_from_node_ids": [],
        },
    }
    node.update(overrides)
    return node


def _compressed_node(**overrides):
    """Build a minimal valid compressed node."""
    node = {
        "id": 5,
        "label": "comp",
        "surface_label": "comp@en",
        "kind": "node",
        "tier": 2,
        "confidence": 0.5,
        "status": "candidate",
        "is_seed": False,
        "is_locked": False,
        "semantic": {
            "compression_state": "compressed",
            "derived_from_node_ids": [1, 2],
            "compression_reason": "co-occurrence aggregation",
        },
    }
    node.update(overrides)
    return node


# ===================================================================
# Semantic node validation
# ===================================================================


class TestSemanticNodeValidation:
    """Tests for _validate_semantic_node."""

    def test_valid_raw_node(self):
        """A properly formed raw node passes validation."""
        node = _raw_node()
        _validate_semantic_node(node, {10})  # should not raise

    def test_valid_compressed_node(self):
        """A properly formed compressed node passes validation."""
        node = _compressed_node()
        _validate_semantic_node(node, {1, 2, 5})  # should not raise

    def test_invalid_kind_composite_rejected(self):
        """Kind 'composite' is rejected (deprecated in v4.2)."""
        node = _raw_node(kind="composite")
        with pytest.raises(SchemaVersionMismatchError, match="deprecated_kind"):
            _validate_semantic_node(node, {10})

    def test_invalid_kind_atom_rejected(self):
        """Kind 'atom' is rejected (deprecated in v4.2)."""
        node = _raw_node(kind="atom")
        with pytest.raises(SchemaVersionMismatchError, match="deprecated_kind"):
            _validate_semantic_node(node, {10})

    def test_missing_surface_label_locale_rejected(self):
        """surface_label without @locale is rejected."""
        node = _raw_node(surface_label="test_no_locale")
        with pytest.raises(SchemaValidationError, match="surface_label_locale_required"):
            _validate_semantic_node(node, {10})

    def test_missing_semantic_rejected(self):
        """Missing semantic dict is rejected."""
        node = _raw_node()
        del node["semantic"]
        with pytest.raises(SchemaValidationError, match="semantic_required"):
            _validate_semantic_node(node, {10})

    def test_invalid_compression_state_rejected(self):
        """Invalid compression_state is rejected."""
        node = _raw_node()
        node["semantic"]["compression_state"] = "invalid"
        with pytest.raises(SchemaValidationError, match="invalid_compression_state"):
            _validate_semantic_node(node, {10})

    def test_self_derived_forbidden(self):
        """Node cannot derive from itself."""
        node = _compressed_node(semantic={
            "compression_state": "compressed",
            "derived_from_node_ids": [5],  # same as node id
            "compression_reason": "test",
        })
        with pytest.raises(SchemaValidationError, match="self_derived_forbidden"):
            _validate_semantic_node(node, {5})

    def test_compressed_without_reason_rejected(self):
        """Compressed node without compression_reason is rejected."""
        node = _compressed_node(semantic={
            "compression_state": "compressed",
            "derived_from_node_ids": [1, 2],
            "compression_reason": "",
        })
        with pytest.raises(SchemaValidationError, match="compression_reason_required"):
            _validate_semantic_node(node, {1, 2, 5})

    def test_compressed_without_derived_rejected(self):
        """Compressed node without derived_from_node_ids is rejected."""
        node = _compressed_node(semantic={
            "compression_state": "compressed",
            "derived_from_node_ids": [],
            "compression_reason": "test",
        })
        with pytest.raises(SchemaValidationError, match="compressed_requires_derived"):
            _validate_semantic_node(node, {5})

    def test_seed_without_lock_rejected(self):
        """Seed node without is_locked is rejected."""
        node = _seed_node(is_locked=False)
        with pytest.raises(InvariantViolationError, match="seed_requires_lock"):
            _validate_semantic_node(node, {1})

    def test_seed_wrong_tier_rejected(self):
        """Seed node with tier != 1 is rejected."""
        node = _seed_node(tier=2)
        with pytest.raises(InvariantViolationError, match="seed_tier"):
            _validate_semantic_node(node, {1})

    def test_seed_wrong_confidence_rejected(self):
        """Seed node with confidence != 1.0 is rejected."""
        node = _seed_node(confidence=0.8)
        with pytest.raises(InvariantViolationError, match="seed_confidence"):
            _validate_semantic_node(node, {1})

    def test_seed_wrong_status_rejected(self):
        """Seed node with status != stable is rejected."""
        node = _seed_node(status="candidate")
        with pytest.raises(InvariantViolationError, match="seed_status"):
            _validate_semantic_node(node, {1})

    def test_sense_centric_schema_rejected(self):
        """Sense-centric node schema (deprecated) is rejected."""
        node = _raw_node(sense_state={"semantic_index": 0})
        with pytest.raises(SchemaValidationError, match="schema_model_mismatch_sense_centric"):
            _validate_semantic_node(node, {10})

    def test_invalid_status_rejected(self):
        """Invalid node status is rejected."""
        node = _raw_node(status="invalid_status")
        with pytest.raises(SchemaValidationError, match="invalid_status"):
            _validate_semantic_node(node, {10})

    def test_derived_from_missing_node_rejected(self):
        """derived_from_node_ids referencing missing nodes is rejected."""
        node = _compressed_node(semantic={
            "compression_state": "compressed",
            "derived_from_node_ids": [999],  # not in node_ids
            "compression_reason": "test",
        })
        with pytest.raises(SchemaValidationError, match="derived_node_missing"):
            _validate_semantic_node(node, {5})

    def test_duplicate_derived_ids_rejected(self):
        """Duplicate entries in derived_from_node_ids are rejected."""
        node = _compressed_node(semantic={
            "compression_state": "compressed",
            "derived_from_node_ids": [1, 1],  # duplicate
            "compression_reason": "test",
        })
        with pytest.raises(SchemaValidationError, match="derived_from_node_ids_duplicate"):
            _validate_semantic_node(node, {1, 5})


# ===================================================================
# Snapshot contract validation
# ===================================================================


class TestSnapshotValidation:
    """Tests for _validate_snapshot_contract."""

    def _make_valid_snapshot(self):
        """Create a minimal valid v4.2 snapshot."""
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": "test_snap",
            "generated_at": "2024-01-01T00:00:00Z",
            "nodes": [
                _seed_node(),
            ],
            "edges": [],
        }

    def test_valid_snapshot(self):
        """A properly formed v4.2 snapshot passes validation."""
        snap = self._make_valid_snapshot()
        _validate_snapshot_contract(snap)  # should not raise

    def test_wrong_schema_version_rejected(self):
        """Wrong schema_version is rejected."""
        snap = self._make_valid_snapshot()
        snap["schema_version"] = "v3.0"
        with pytest.raises(SchemaVersionMismatchError):
            _validate_snapshot_contract(snap)

    def test_missing_nodes_rejected(self):
        """Missing nodes list is rejected."""
        snap = self._make_valid_snapshot()
        del snap["nodes"]
        with pytest.raises(SchemaValidationError, match="nodes_required"):
            _validate_snapshot_contract(snap)

    def test_nodes_not_list_rejected(self):
        """nodes field must be a list."""
        snap = self._make_valid_snapshot()
        snap["nodes"] = "not a list"
        with pytest.raises(SchemaValidationError, match="nodes_required"):
            _validate_snapshot_contract(snap)

    def test_node_not_dict_rejected(self):
        """Each node must be a dict."""
        snap = self._make_valid_snapshot()
        snap["nodes"] = ["not a dict"]
        with pytest.raises(SchemaValidationError, match="node_object_required"):
            _validate_snapshot_contract(snap)

    def test_invalid_node_in_snapshot_rejected(self):
        """Invalid node inside snapshot is rejected."""
        snap = self._make_valid_snapshot()
        snap["nodes"] = [
            _raw_node(kind="composite"),  # deprecated kind
        ]
        with pytest.raises(SchemaVersionMismatchError, match="deprecated_kind"):
            _validate_snapshot_contract(snap)

    def test_multiple_nodes_snapshot(self):
        """Snapshot with multiple valid nodes passes validation."""
        snap = self._make_valid_snapshot()
        # Create a compressed node that references the seed node (id=1)
        comp_node = _compressed_node(id=5, semantic={
            "compression_state": "compressed",
            "derived_from_node_ids": [1],  # references seed node which exists
            "compression_reason": "co-occurrence aggregation",
        })
        snap["nodes"] = [
            _seed_node(),
            _raw_node(id=10),
            comp_node,
        ]
        _validate_snapshot_contract(snap)  # should not raise


# ===================================================================
# Normalizers
# ===================================================================


class TestNormalizers:
    """Tests for _normalize_view and _normalize_lang."""

    # --- view normalizer ---

    def test_normalize_view_compact(self):
        """'compact' view is valid."""
        assert _normalize_view("compact") == "compact"

    def test_normalize_view_detail(self):
        """'detail' view is valid."""
        assert _normalize_view("detail") == "detail"

    def test_normalize_view_invalid(self):
        """Invalid view is rejected."""
        with pytest.raises(SchemaValidationError, match="invalid_view"):
            _normalize_view("graph")

    def test_normalize_view_case_insensitive(self):
        """View is case-insensitive."""
        assert _normalize_view("Compact") == "compact"
        assert _normalize_view("DETAIL") == "detail"

    def test_normalize_view_none_defaults_to_compact(self):
        """None defaults to 'compact'."""
        assert _normalize_view(None) == "compact"

    # --- language normalizer ---

    def test_normalize_lang_valid(self):
        """Valid language codes are accepted."""
        assert _normalize_lang("en") == "en"
        assert _normalize_lang("id") == "id"
        assert _normalize_lang("zh") == "zh"
        assert _normalize_lang("fr") == "fr"
        assert _normalize_lang("ja") == "ja"
        assert _normalize_lang("ko") == "ko"

    def test_normalize_lang_invalid(self):
        """Invalid language code is rejected."""
        with pytest.raises(SchemaValidationError, match="invalid_language_code"):
            _normalize_lang("xx")

    def test_normalize_lang_runtime_codes_separate(self):
        """Runtime codes (python, javascript) are NOT language codes."""
        with pytest.raises(SchemaValidationError, match="invalid_language_code"):
            _normalize_lang("python")

    def test_normalize_lang_case_insensitive(self):
        """Language code is case-insensitive."""
        assert _normalize_lang("EN") == "en"
        assert _normalize_lang("ID") == "id"
        assert _normalize_lang("ZH") == "zh"

    def test_normalize_lang_none_defaults_to_id(self):
        """None defaults to 'id'."""
        assert _normalize_lang(None) == "id"
