# RSVS Modes Spec V1 (Schema v4.2)

Mode final:
- `ingest`
- `appraise`
- `relate`

## Unified API

### POST `/run`
Request:
```json
{
  "mode": "ingest|appraise|relate",
  "text": "input text",
  "correlation_id": "optional",
  "schema_version": "v4.2",
  "options": {}
}
```

`options` supports:
- `view`: `compact|detail` (default `compact`)
- `top_k`: integer (relate mode)

Shared response envelope:
```json
{
  "ok": true,
  "mode": "ingest|appraise|relate",
  "correlation_id": "...",
  "timestamp": "ISO-8601",
  "result": {},
  "messages": [],
  "files": {},
  "meta": {
    "version": "v1",
    "schema_version": "v4.2",
    "atom_dir": "...",
    "latency_ms": 0
  }
}
```

Errors:
- `400 invalid_mode`
- `400 text_required`
- `400 invalid_json`
- `400 invalid_view`
- `409 schema_version_mismatch`
- `500 <runtime error>`

## Mode Contracts

### 1) `ingest`
`result`:
- `snapshot`
- `events[]`
- `stats`: `token_count`, `node_count`, `edge_count`, `batch_id`

Artifacts:
- `snapshot-*.json`
- `events-*.jsonl`
- `report-*.json`

### 2) `appraise`
Basis: evaluate against current graph evidence.

`result`:
- `stance`: `{ "agree": int, "disagree": int }` (`sum=100`)
- `confidence`: `0..1`
- `verdict`: `agree|mixed|disagree`
- `rationale`: string
- `view`: `compact|detail`
- `evidence`:
  - `support_nodes[]` (`id,label,score,compression_state,derived_from_node_ids`)
  - `conflict_nodes[]` (`id,label,score,compression_state,derived_from_node_ids`)
  - `paths[]` (v1 may be empty)

Artifact:
- `appraise-*.json`

### 3) `relate`
`result`:
- `query_terms[]`
- `view`: `compact|detail`
- `related_nodes[]` (`id,label,score,tier,kind,compression_state,derived_from_node_ids[,derived_nodes]`) sorted desc
- `related_edges[]` (`id,source,target,score`) sorted desc
- `clusters[]` (v1 optional, default empty)

Artifact:
- `relate-*.json`

## Latest Retrieval

### GET `/latest`
- default (no query): legacy ingest bundle for frontend restore (`snapshot/events/messages/files`)

### GET `/latest?mode=ingest|appraise|relate`
- return mode envelope (`ok, mode, result, files, meta`)
- supports `view=compact|detail` query param

Error when no artifact:
```json
{ "ok": false, "error": "no_artifacts", "mode": "..." }
```

Error for legacy artifact/schema:
```json
{ "ok": false, "error": "schema_version_mismatch" }
```

## CLI Mapping
- `run --mode ingest|appraise|relate --view compact|detail`
- `ingest` = alias to `run --mode ingest`
- `latest --mode ... --view compact|detail`
- `atom-show appraise|relate`
