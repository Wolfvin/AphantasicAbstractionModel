"""RSVS Bridge schema validation."""

from __future__ import annotations

from typing import Any

from .config import SCHEMA_VERSION
from .exceptions import (
    InvariantViolationError,
    SchemaValidationError,
    SchemaVersionMismatchError,
)


__all__ = [
    "VALID_VIEWS",
    "VALID_STATUSES",
    "VALID_LANG_CODES",
    "_normalize_view",
    "_normalize_lang",
    "_is_sense_centric_node",
    "_validate_semantic_node",
    "_validate_snapshot_contract",
]

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

VALID_VIEWS = {"compact", "detail"}
VALID_STATUSES = {"new", "candidate", "stable", "deprecated", "quarantine"}
VALID_LANG_CODES = {"id", "id-ID", "jv", "su", "en", "en-US", "en-GB", "zh", "zh-CN", "zh-TW", "fr", "ja", "ko", "de", "es", "pt", "ar", "hi", "ru"}

VALID_RUNTIME_CODES = {"python", "javascript", "typescript", "rust", "go", "java"}


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


def _normalize_view(value: Any) -> str:
    """Normalize and validate a view parameter."""
    view = str(value or "compact").strip().lower()
    if view not in VALID_VIEWS:
        raise SchemaValidationError(f"invalid_view:{view}")
    return view


def _normalize_lang(value: Any) -> str:
    """Normalize and validate a language code parameter."""
    lang = str(value or "id").strip().lower()
    if lang not in VALID_LANG_CODES:
        raise SchemaValidationError("invalid_language_code")
    return lang


# ---------------------------------------------------------------------------
# Node / snapshot validation
# ---------------------------------------------------------------------------


def _is_sense_centric_node(node: dict[str, Any]) -> bool:
    """Return True if the node uses the deprecated sense-centric schema."""
    sense_state = node.get("sense_state")
    if not isinstance(sense_state, dict):
        return False
    if "semantic_index" in sense_state:
        return True
    senses = sense_state.get("senses")
    if isinstance(senses, list):
        for s in senses:
            if isinstance(s, dict) and ("layer_1" in s or "layer_2" in s):
                return True
    return False


def _validate_semantic_node(node: dict[str, Any], node_ids: set[int]) -> None:
    """Validate a single semantic node against the v5.0 schema contract."""
    kind = node.get("kind")
    if kind != "node":
        raise SchemaVersionMismatchError("deprecated_kind")

    if _is_sense_centric_node(node):
        raise SchemaValidationError("schema_model_mismatch_sense_centric")

    surface_label = str(node.get("surface_label") or "").strip()
    if "@" not in surface_label:
        raise SchemaValidationError("surface_label_locale_required")

    semantic = node.get("semantic")
    if not isinstance(semantic, dict):
        raise SchemaValidationError("semantic_required")

    state = semantic.get("compression_state")
    if state not in {"raw", "compressed", "composed"}:
        raise SchemaValidationError("invalid_compression_state")

    derived = semantic.get("derived_from_node_ids")
    if not isinstance(derived, list):
        raise SchemaValidationError("derived_from_node_ids_required")

    if len(set(derived)) != len(derived):
        raise SchemaValidationError("derived_from_node_ids_duplicate")

    node_id = node.get("id")
    if state == "compressed":
        reason = str(semantic.get("compression_reason") or "").strip()
        if not reason:
            raise SchemaValidationError("compression_reason_required")
        if not derived:
            raise SchemaValidationError("compressed_requires_derived")

    for dep_id in derived:
        if not isinstance(dep_id, int):
            raise SchemaValidationError("derived_id_must_be_int")
        if dep_id == node_id:
            raise SchemaValidationError("self_derived_forbidden")
        if dep_id not in node_ids:
            raise SchemaValidationError("derived_node_missing")

    status = node.get("status")
    if status not in VALID_STATUSES:
        raise SchemaValidationError("invalid_status")

    is_seed = bool(node.get("is_seed", False))
    if is_seed:
        if not bool(node.get("is_locked", False)):
            raise InvariantViolationError("seed_requires_lock")
        if int(node.get("tier", 0)) != 1:
            raise InvariantViolationError("seed_tier")
        if float(node.get("confidence", 0.0)) != 1.0:
            raise InvariantViolationError("seed_confidence")
        if status != "stable":
            raise InvariantViolationError("seed_status")


def _validate_snapshot_contract(snapshot: dict[str, Any]) -> None:
    """Validate a full snapshot against the v5.0 schema contract."""
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise SchemaVersionMismatchError(
            f"expected={SCHEMA_VERSION}, got={snapshot.get('schema_version')}"
        )
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list):
        raise SchemaValidationError("nodes_required")
    node_ids = {int(n["id"]) for n in nodes if isinstance(n, dict) and isinstance(n.get("id"), int)}
    for node in nodes:
        if not isinstance(node, dict):
            raise SchemaValidationError("node_object_required")
        _validate_semantic_node(node, node_ids)
