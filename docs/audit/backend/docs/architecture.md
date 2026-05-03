# RSVS Architecture

## Core Semantics (`crates/rsvs-core/src`)
- `types.rs`: primitive IDs, tiers, node/edge contracts.
- `graph.rs`: DAG storage, similarity primitives, edge lookup.
- `seed.rs`: deterministic seed bootstrap.
- `attention.rs`: tokenizer, co-occurrence stats, attention candidate scoring.
- `sense.rs`: sense clustering and coherence lifecycle.
- `autonomy.rs`: confidence/tier/memory governance and stability gate.
- `pipeline.rs`: end-to-end ingestion/query orchestration.

## Representation vs Ontology Boundary (V4.2)
- Ontology runtime memakai **single node model** (`kind=node`).
- `composite` tidak lagi dianggap tipe ontologis.
- Kompresi dinyatakan di layer representasi semantik:
  - `semantic.compression_state`: `raw|compressed`
  - `semantic.derived_from_node_ids`: provenance kompresi
  - `semantic.compression_reason`: alasan kompresi
- Query wajib mendukung dua tampilan:
  - `compact_view`: canonical node output
  - `detail_view`: include expansion/provenance node asal

## Persistence Boundary
- `persist.rs` is the only owner for JSON snapshot schema and roundtrip behavior.
- Runtime modules do not perform direct file I/O; they expose state for `persist.rs`.
- Bridge artifacts harus membawa `schema_version` eksplisit (`v4.2`).
- Hard break policy: payload schema lama ditolak, tidak ada fallback parse ke model `composite`.

## Python Adapter Boundary
- `bindings.rs` is the only adapter from Rust core to Python (`rsvs._rsvs`).
- Python package (`python/rsvs`) calls stable binding surface only.
- Python CLI and corpus/eval tooling must not bypass adapter with custom Rust FFI logic.

## Dependency Direction
- `types/graph/seed/sense/attention/autonomy` -> `pipeline`
- `pipeline` -> `persist` and `bindings` (via crate surface)
- Python package depends on compiled extension + pure Python helpers.

## Non-runtime Artifacts
- Skill artifacts are archived in `docs/skills/`.
- Operational notes live under `docs/` and do not participate in runtime imports.

## Accepted Node Invariants (V4.2)
- `kind` harus `node`.
- `semantic.compression_state` harus `raw` atau `compressed`.
- Jika `compressed`, maka:
  - `derived_from_node_ids` wajib non-empty,
  - tidak boleh self-reference,
  - tidak boleh duplicate id,
  - semua id turunan harus tersedia di snapshot yang sama,
  - `compression_reason` wajib non-empty.
