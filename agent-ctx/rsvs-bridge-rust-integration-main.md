# Task: Fix Rust↔Python Integration Gap in RSVS Bridge Server

## Task ID: rsvs-bridge-rust-integration

## Agent: main

## Summary

Rewrote `bridge_server.py` to use the Rust core via PyO3 for all computational heavy lifting, while keeping HTTP infrastructure and artifact persistence in Python. Also updated `__init__.py` to export new v4.2 types.

## Files Modified

1. **`/home/z/my-project/SymbolicPuzzle3D/backend/python/rsvs/bridge_server.py`** — Complete rewrite
2. **`/home/z/my-project/SymbolicPuzzle3D/backend/python/rsvs/__init__.py`** — Added v4.2 exports

## Architecture After Fix

```
HTTP Request → bridge_server.py (thin HTTP layer + artifact I/O)
                   ↓
              rsvs.Rsvs (Rust core via PyO3)
                   ↓
              Rust: pipeline, attention, autonomy, sense, graph
```

## Key Changes in bridge_server.py

### Rust Core Integration
- **Import pattern**: `try: from rsvs import Rsvs ... except: _RSVS_AVAILABLE = False`
- **Singleton management**: `_get_rsvs()` creates/loads a single `Rsvs` instance
- **Persistence**: `_save_rsvs()` calls `r.save()` to `atom/rsvs-state.json`
- **Startup**: Attempts to load from `rsvs-state.json`, creates fresh instance on failure

### Mode Implementations (Rust path)
- **`_run_ingest_rust()`**: Calls `r.ingest_with_meta_v1()`, then `r.snapshot_v1()` + `r.consume_events_v1()`, converts to Python format, validates, writes artifacts
- **`_run_appraise_rust()`**: Calls `r.appraise(text)`, maps Rust verdicts (`consistent`/`partial`/`novel`) to Python verdicts (`agree`/`mixed`/`disagree`)
- **`_run_relate_rust()`**: Calls `r.relate(concept)` for first known token, falls back to Python `_score_overlap` on Rust snapshot

### Format Conversion Layer
- **`_convert_rust_node()`**: Converts flat Rust RuntimeNode → nested Python node (adds `semantic`, `policy_meta`, `render`, `provenance`)
- **`_convert_rust_edge()`**: Converts Rust RuntimeEdge → Python edge (adds `render`, `direction`, `status`)
- **`_convert_rust_event()`**: Converts Rust RuntimeEvent → Python event (adds `event_id`, `timestamp`, `animation_hint`)
- **`_build_bridge_snapshot()`**: Converts full Rust RuntimeSnapshot → Python bridge snapshot format

### Legacy Fallback (DEPRECATED)
- All old Python reimplementations preserved as `_legacy_*` functions
- Used only when `rsvs` native module is not available
- Emits `DeprecationWarning` on use
- Functions: `_legacy_build_base_node`, `_legacy_apply_seed_rule`, `_legacy_status_transition`, `_legacy_tier_from_score`, `_legacy_score_evidence`, `_legacy_evaluate_node_policy`, `_legacy_ensure_seed_nodes`, `_legacy_build_snapshot_and_events`, `_legacy_run_ingest`, `_legacy_run_appraise`, `_legacy_run_relate`

### Preserved Python Utilities (not in Rust)
- Schema validation: `_validate_snapshot_contract`, `_validate_semantic_node`, `_is_sense_centric_node`
- View projection: `_project_node`
- File I/O: `_write_json`, `_write_events_jsonl`, `_latest_file`, `_read_latest_ingest_bundle`
- Token/overlap: `_tokenize`, `_score_overlap`, `_norm_text`, `_fingerprint`
- Normalization: `_normalize_view`, `_normalize_lang`
- HTTP handler: Complete `Handler` class with all endpoints

### New Endpoints
- **`GET /status`**: Returns Rust core status info
- **`GET /health`**: Now includes `backend` field ("rust" or "python_legacy")

### Backward Compatibility
- All existing endpoints preserved: `POST /run`, `POST /ingest`, `GET /latest`, `GET /health`
- Response payloads maintain same structure
- `meta` field now includes `backend` indicator
- Graceful shutdown saves Rust state via `KeyboardInterrupt` handler

## Key Changes in __init__.py

- Added `PyNodeInfo as NodeInfo` export
- Added `PyAppraiseResult as AppraiseResult` export
- Added `PyRelateResult as RelateResult` export
- Updated `__all__` to include all v4.2 types
