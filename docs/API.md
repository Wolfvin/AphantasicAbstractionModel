# RSVS API Reference

> Complete HTTP API documentation for the RSVS Bridge Server (Schema v4.2, API v1)

**Base URL**: `http://127.0.0.1:8787` (configurable via `RSVS_BRIDGE_HOST` and `RSVS_BRIDGE_PORT`)

**Content-Type**: All requests and responses use `application/json; charset=utf-8`

**CORS**: Enabled for all origins (`Access-Control-Allow-Origin: *`)

---

## Table of Contents

- [POST /run](#post-run)
- [POST /ingest](#post-ingest)
- [POST /query](#post-query)
- [POST /similarity](#post-similarity)
- [POST /appraise](#post-appraise)
- [POST /relate](#post-relate)
- [GET /snapshot](#get-snapshot)
- [GET /events](#get-events)
- [GET /health](#get-health)
- [GET /latest](#get-latest)
- [GET /status](#get-status)
- [Error Handling](#error-handling)
- [Type Reference](#type-reference)

---

## POST /run

General mode dispatch endpoint. Executes the specified mode (`ingest`, `appraise`, or `relate`) on the provided text.

### Request

```http
POST /run HTTP/1.1
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `mode` | `string` | Yes | — | One of: `ingest`, `appraise`, `relate` |
| `text` | `string` | Yes | — | Input text to process |
| `correlation_id` | `string` | No | Auto-generated | Client-provided correlation ID for tracing |
| `schema_version` | `string` | No | `null` | Must be `"v4.2"` if provided; mismatch returns 409 |
| `options` | `object` | No | `{}` | Mode-specific options (see below) |

#### Options by Mode

**Ingest Options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `view` | `string` | `"compact"` | Output view: `compact` or `detail` |
| `language` | `string` | `"en"` | ISO 639-1 language code |
| `domain_id` | `integer` | `null` | Domain identifier for multi-domain graphs |

**Appraise Options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `view` | `string` | `"compact"` | Output view: `compact` or `detail` |

**Relate Options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `view` | `string` | `"compact"` | Output view: `compact` or `detail` |
| `top_k` | `integer` | `10` | Maximum number of related results (1–50) |

### Example Request

```json
{
  "mode": "ingest",
  "text": "Water flows through porous stone. Erosion shapes the landscape over millennia.",
  "correlation_id": "corr_abc123",
  "schema_version": "v4.2",
  "options": {
    "view": "compact",
    "language": "en"
  }
}
```

### Response

#### Ingest Mode Response (200 OK)

```json
{
  "ok": true,
  "mode": "ingest",
  "correlation_id": "corr_abc123",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "result": {
    "snapshot": {
      "schema_version": "v4.2",
      "snapshot_id": "snapshot_1714737600000_1234",
      "generated_at": "2025-05-03T12:00:00.000000+00:00",
      "context": {
        "domain": "rsvs-core",
        "batch_id": "corr_abc123",
        "input_message_id": "corr_abc123",
        "policy_version": "v4.2",
        "language_code": "en"
      },
      "nodes": [
        {
          "id": 0,
          "label": "exists",
          "surface_label": "exists@en",
          "language_links": [],
          "kind": "node",
          "tier": 1,
          "confidence": 1.0,
          "status": "stable",
          "is_seed": true,
          "is_locked": true,
          "semantic": {
            "compression_state": "raw",
            "derived_from_node_ids": [],
            "compression_reason": "base_ingest_signal"
          },
          "policy_meta": {
            "policy_version": "v4.2",
            "governance_score": 1.0,
            "candidate_evidence_pool": 0.0,
            "status_flip_count": 0,
            "seen_fingerprints": [],
            "seed_registry": true,
            "short_window_hits": 0,
            "long_window_hits": 0,
            "last_seen_at": "2025-05-03T12:00:00.000000+00:00"
          },
          "provenance": {
            "source_batch_id": "corr_abc123",
            "source_domain": "core_seed",
            "source_type": "bootstrap"
          }
        }
      ],
      "edges": [
        {
          "id": "",
          "source": 0,
          "target": 5,
          "direction": "undirected",
          "weight": 0.35,
          "source_type": "learned",
          "status": "new"
        }
      ]
    },
    "events": [
      {
        "event_id": "evt_1714737600000_1234",
        "timestamp": "2025-05-03T12:00:00.000000+00:00",
        "correlation_id": "corr_abc123",
        "event_type": "node_created",
        "payload": {
          "id": 25,
          "label": "water"
        },
        "animation_hint": {
          "priority": "normal",
          "focus_node_id": 25,
          "burst_group": "corr_abc123"
        },
        "seq": 1,
        "api_version": "v1",
        "schema_version": "v4.2"
      }
    ],
    "stats": {
      "batch_id": "corr_abc123",
      "token_count": 8,
      "node_count": 30,
      "edge_count": 42,
      "created_count": 3,
      "promoted_count": 3,
      "replay_count": 0,
      "sentences_processed": 2,
      "sense_assigned": 2,
      "sense_created": 1,
      "confidence_updated": 5
    }
  },
  "messages": [
    {
      "id": "msg_1714737600000_1234",
      "type": "system_ingest_status",
      "content": "Ingesting batch corr_abc123 — 2 sentences, 3 atoms promoted, 42 edges (via Rust core).",
      "timestamp": "2025-05-03T12:00:00.000000+00:00",
      "correlation_id": "corr_abc123"
    }
  ],
  "files": {
    "snapshot": "atom/snapshot-20250503T120000Z.json",
    "events": "atom/events-20250503T120000Z.jsonl",
    "report": "atom/report-20250503T120000Z.json"
  },
  "meta": {
    "version": "v1",
    "schema_version": "v4.2",
    "atom_dir": "/path/to/atom",
    "latency_ms": 42,
    "backend": "rust"
  }
}
```

#### Appraise Mode Response (200 OK)

```json
{
  "ok": true,
  "mode": "appraise",
  "correlation_id": "corr_def456",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "result": {
    "view": "compact",
    "stance": {
      "agree": 75,
      "disagree": 25
    },
    "confidence": 0.485,
    "verdict": "mixed",
    "rationale": "Input has partial alignment with current graph evidence.",
    "evidence": {
      "support_nodes": [
        { "label": "water", "score": 0.85 },
        { "label": "stone", "score": 0.72 }
      ],
      "conflict_nodes": [],
      "paths": []
    }
  },
  "messages": [
    {
      "id": "msg_1714737600000_5678",
      "type": "system_ingest_status",
      "content": "Appraise verdict: mixed (75% agree / 25% disagree) [Rust core].",
      "timestamp": "2025-05-03T12:00:00.000000+00:00",
      "correlation_id": "corr_def456"
    }
  ],
  "files": {
    "appraise": "atom/appraise-20250503T120000Z.json"
  },
  "meta": {
    "version": "v1",
    "schema_version": "v4.2",
    "atom_dir": "/path/to/atom",
    "latency_ms": 15,
    "backend": "rust"
  }
}
```

#### Relate Mode Response (200 OK)

```json
{
  "ok": true,
  "mode": "relate",
  "correlation_id": "corr_ghi789",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "result": {
    "view": "compact",
    "query_terms": ["water", "erosion"],
    "related_nodes": [
      {
        "id": 25,
        "label": "water",
        "score": 0.92,
        "tier": 2,
        "kind": "node",
        "status": "stable",
        "confidence": 0.78
      },
      {
        "id": 8,
        "label": "erosion",
        "score": 0.85,
        "tier": 2,
        "kind": "node",
        "status": "candidate",
        "confidence": 0.65
      }
    ],
    "related_edges": [
      {
        "id": "5->8",
        "source": 5,
        "target": 8,
        "score": 0.35
      }
    ],
    "clusters": []
  },
  "messages": [
    {
      "id": "msg_1714737600000_9012",
      "type": "system_ingest_status",
      "content": "Relate found 2 nodes and 1 edges [Rust core, token='water'].",
      "timestamp": "2025-05-03T12:00:00.000000+00:00",
      "correlation_id": "corr_ghi789"
    }
  ],
  "files": {
    "relate": "atom/relate-20250503T120000Z.json"
  },
  "meta": {
    "version": "v1",
    "schema_version": "v4.2",
    "atom_dir": "/path/to/atom",
    "latency_ms": 8,
    "backend": "rust"
  }
}
```

### Error Responses

| Status | Condition | Body |
|--------|-----------|------|
| 400 | Invalid or missing `mode` | `{"ok": false, "error": "invalid_mode", "mode": "..."}` |
| 400 | Empty `text` | `{"ok": false, "error": "text_required", "mode": "..."}` |
| 400 | Malformed JSON body | `{"ok": false, "error": "invalid_json"}` |
| 409 | Schema version mismatch | `{"ok": false, "error": "schema_version_mismatch", "expected": "v4.2", "got": "..."}` |
| 422 | Schema validation error | `{"ok": false, "error": "...", "mode": "..."}` |
| 503 | Rust core unavailable | `{"ok": false, "error": "...", "mode": "..."}` |
| 500 | Internal server error | `{"ok": false, "error": "...", "mode": "..."}` |

---

## POST /ingest

Backward-compatible shorthand for `POST /run` with `mode: "ingest"`. Ingests text into the knowledge graph and returns a flattened response.

### Request

```http
POST /ingest HTTP/1.1
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | `string` | Yes | — | Text to ingest |
| `correlation_id` | `string` | No | Auto-generated | Correlation ID for tracing |
| `options` | `object` | No | `{}` | Ingest options (same as `/run` ingest options) |

### Example Request

```json
{
  "text": "Water flows through porous stone.",
  "correlation_id": "corr_ingest_001"
}
```

### Response (200 OK)

```json
{
  "ok": true,
  "correlation_id": "corr_ingest_001",
  "snapshot": {
    "schema_version": "v4.2",
    "snapshot_id": "snapshot_1714737600000_1234",
    "generated_at": "2025-05-03T12:00:00.000000+00:00",
    "context": { "..." : "..." },
    "nodes": ["..."],
    "edges": ["..."]
  },
  "events": ["..."],
  "messages": ["..."],
  "files": {
    "snapshot": "atom/snapshot-20250503T120000Z.json"
  }
}
```

### Error Responses

| Status | Condition | Body |
|--------|-----------|------|
| 400 | Empty `text` | `{"ok": false, "error": "text_required"}` |
| 400 | Malformed JSON | `{"ok": false, "error": "invalid_json"}` |
| 503 | Rust core unavailable | `{"ok": false, "error": "..."}` |
| 500 | Internal error | `{"ok": false, "error": "..."}` |

---

## POST /query

Semantic query endpoint. Searches the knowledge graph for nodes matching the provided query text.

> **Note**: This endpoint delegates to the `relate` mode internally.

### Request

```http
POST /query HTTP/1.1
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | `string` | Yes | — | Query text |
| `top_k` | `integer` | No | `10` | Maximum results (1–50) |
| `view` | `string` | No | `"compact"` | Output view: `compact` or `detail` |
| `correlation_id` | `string` | No | Auto-generated | Correlation ID |

### Example Request

```json
{
  "text": "geological processes",
  "top_k": 5,
  "view": "compact"
}
```

### Response (200 OK)

```json
{
  "ok": true,
  "mode": "relate",
  "correlation_id": "corr_query_001",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "result": {
    "view": "compact",
    "query_terms": ["geological", "processes"],
    "related_nodes": ["..."],
    "related_edges": ["..."],
    "clusters": []
  },
  "messages": ["..."],
  "files": {},
  "meta": {
    "version": "v1",
    "schema_version": "v4.2",
    "atom_dir": "/path/to/atom",
    "latency_ms": 8,
    "backend": "rust"
  }
}
```

---

## POST /similarity

Pairwise similarity computation between two concepts in the knowledge graph.

> **Note**: This endpoint computes Jaccard similarity between the atom sets of two nodes.

### Request

```http
POST /similarity HTTP/1.1
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source` | `string` | Yes | — | Source concept label |
| `target` | `string` | Yes | — | Target concept label |
| `correlation_id` | `string` | No | Auto-generated | Correlation ID |

### Example Request

```json
{
  "source": "water",
  "target": "erosion"
}
```

### Response (200 OK)

```json
{
  "ok": true,
  "correlation_id": "corr_sim_001",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "result": {
    "source": "water",
    "target": "erosion",
    "jaccard": 0.35,
    "npmi": 0.42,
    "cooc": 0.18,
    "combined_score": 0.344
  }
}
```

### Error Responses

| Status | Condition | Body |
|--------|-----------|------|
| 400 | Missing `source` or `target` | `{"ok": false, "error": "source_and_target_required"}` |
| 404 | Node not found | `{"ok": false, "error": "node_not_found", "label": "..."}` |

---

## POST /appraise

Node quality assessment. Evaluates how well the provided text aligns with the current knowledge graph.

### Request

```http
POST /appraise HTTP/1.1
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | `string` | Yes | — | Text to appraise |
| `correlation_id` | `string` | No | Auto-generated | Correlation ID |
| `options` | `object` | No | `{}` | Appraise options |

### Example Request

```json
{
  "text": "Water erodes stone over geological time",
  "correlation_id": "corr_appraise_001",
  "options": {
    "view": "compact"
  }
}
```

### Response (200 OK)

```json
{
  "ok": true,
  "mode": "appraise",
  "correlation_id": "corr_appraise_001",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "result": {
    "view": "compact",
    "stance": {
      "agree": 75,
      "disagree": 25
    },
    "confidence": 0.485,
    "verdict": "mixed",
    "rationale": "Input has partial alignment with current graph evidence.",
    "evidence": {
      "support_nodes": [
        { "label": "water", "score": 0.85 },
        { "label": "stone", "score": 0.72 }
      ],
      "conflict_nodes": [],
      "paths": []
    }
  },
  "messages": ["..."],
  "files": {
    "appraise": "atom/appraise-20250503T120000Z.json"
  },
  "meta": {
    "version": "v1",
    "schema_version": "v4.2",
    "atom_dir": "/path/to/atom",
    "latency_ms": 15,
    "backend": "rust"
  }
}
```

### Verdict Values

| Verdict | Meaning |
|---------|---------|
| `agree` | Input is mostly consistent with graph evidence (>70% agreement) |
| `mixed` | Input has partial alignment with graph evidence |
| `disagree` | Input is poorly supported by current graph |

### Error Responses

| Status | Condition | Body |
|--------|-----------|------|
| 400 | Empty `text` | `{"ok": false, "error": "text_required", "mode": "appraise"}` |
| 503 | Rust core unavailable | `{"ok": false, "error": "..."}` |

---

## POST /relate

Relationship analysis. Finds nodes and edges related to the provided concept text.

### Request

```http
POST /relate HTTP/1.1
Content-Type: application/json
```

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | `string` | Yes | — | Concept text to find relations for |
| `correlation_id` | `string` | No | Auto-generated | Correlation ID |
| `options` | `object` | No | `{}` | Relate options |

#### Relate Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `view` | `string` | `"compact"` | Output view: `compact` or `detail` |
| `top_k` | `integer` | `10` | Maximum results (1–50) |

### Example Request

```json
{
  "text": "volcanic activity",
  "correlation_id": "corr_relate_001",
  "options": {
    "top_k": 5,
    "view": "compact"
  }
}
```

### Response (200 OK)

```json
{
  "ok": true,
  "mode": "relate",
  "correlation_id": "corr_relate_001",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "result": {
    "view": "compact",
    "query_terms": ["volcanic", "activity"],
    "related_nodes": [
      {
        "id": 12,
        "label": "volcano",
        "surface_label": "volcano@en",
        "language_links": [],
        "score": 0.92,
        "tier": 2,
        "kind": "node",
        "status": "stable",
        "confidence": 0.78,
        "is_seed": false,
        "is_locked": false,
        "compression_state": "raw",
        "derived_from_node_ids": []
      }
    ],
    "related_edges": [
      {
        "id": "12->15",
        "source": 12,
        "target": 15,
        "score": 0.45
      }
    ],
    "clusters": []
  },
  "messages": ["..."],
  "files": {
    "relate": "atom/relate-20250503T120000Z.json"
  },
  "meta": {
    "version": "v1",
    "schema_version": "v4.2",
    "atom_dir": "/path/to/atom",
    "latency_ms": 8,
    "backend": "rust"
  }
}
```

### Error Responses

| Status | Condition | Body |
|--------|-----------|------|
| 400 | Empty `text` | `{"ok": false, "error": "text_required", "mode": "relate"}` |
| 503 | Rust core unavailable | `{"ok": false, "error": "..."}` |

---

## GET /snapshot

Retrieve a full snapshot of the current knowledge graph state.

### Request

```http
GET /latest?mode=ingest&view=compact HTTP/1.1
```

#### Query Parameters

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `mode` | `string` | `"ingest"` | `ingest`, `appraise`, `relate` | Artifact mode to retrieve |
| `view` | `string` | `"compact"` | `compact`, `detail` | Output view projection |

### Response (200 OK)

Returns the latest stored snapshot artifact. When `mode=ingest`, the response includes the full snapshot, events, and stats bundle:

```json
{
  "ok": true,
  "mode": "ingest",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "result": {
    "snapshot": {
      "schema_version": "v4.2",
      "snapshot_id": "snapshot_1714737600000_1234",
      "generated_at": "2025-05-03T12:00:00.000000+00:00",
      "context": { "..." : "..." },
      "nodes": ["..."],
      "edges": ["..."]
    },
    "events": ["..."],
    "stats": {
      "token_count": null,
      "node_count": 30,
      "edge_count": 42,
      "batch_id": "corr_abc123"
    }
  },
  "messages": ["..."],
  "files": { "..." : "..." },
  "meta": {
    "version": "v1",
    "schema_version": "v4.2",
    "atom_dir": "/path/to/atom",
    "latency_ms": 0
  }
}
```

### Error Responses

| Status | Condition | Body |
|--------|-----------|------|
| 400 | Invalid `mode` | `{"ok": false, "error": "invalid_mode", "mode": "..."}` |
| 400 | Invalid `view` | `{"ok": false, "error": "invalid_view"}` |
| 404 | No artifacts found | `{"ok": false, "error": "no_artifacts", "mode": "..."}` |

---

## GET /events

Event stream consumption endpoint. Retrieves events that have been generated since the last consumption.

> **Note**: Events are consumed (one-time read) from the Rust core's event buffer via `consume_events_v1()`.

### Request

```http
GET /latest?mode=ingest HTTP/1.1
```

Events are included within the ingest response under the `result.events` key.

### Event Schema

```json
{
  "event_id": "evt_1714737600000_1234",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "correlation_id": "corr_abc123",
  "event_type": "node_created",
  "payload": {
    "id": 25,
    "label": "water"
  },
  "animation_hint": {
    "priority": "normal",
    "focus_node_id": 25,
    "burst_group": "corr_abc123"
  },
  "seq": 1,
  "api_version": "v1",
  "schema_version": "v4.2"
}
```

### Event Types

| Event Type | Trigger | Priority |
|------------|---------|----------|
| `node_created` | New node inserted into graph | `normal` |
| `confidence_changed` | Node confidence updated | `low` |
| `tier_changed` | Node tier promoted/demoted | `low` |
| `status_changed` | Node status transition | `low` |
| `edge_created` | New edge added | `low` |

---

## GET /health

Health check endpoint. Returns system status including Rust core availability.

### Request

```http
GET /health HTTP/1.1
```

### Response (200 OK)

```json
{
  "status": "ok",
  "rust_core_available": true,
  "service": "rsvs-bridge",
  "timestamp": "2025-05-03T12:00:00.000000+00:00",
  "atom_dir": "/path/to/atom",
  "version": "v1",
  "schema_version": "v4.2"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"ok"` when the server is running |
| `rust_core_available` | `boolean` | Whether the Rust core (via PyO3) is loaded and functional |
| `service` | `string` | Service identifier: `"rsvs-bridge"` |
| `timestamp` | `string` | ISO 8601 UTC timestamp |
| `atom_dir` | `string` | Path to the artifact output directory |
| `version` | `string` | API version (`"v1"`) |
| `schema_version` | `string` | Data schema version (`"v4.2"`) |

---

## GET /latest

Retrieve the latest stored artifact for a given mode. This is the primary read endpoint used by the frontend to restore state on page load.

### Request

```http
GET /latest?mode=ingest&view=compact HTTP/1.1
```

#### Query Parameters

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `mode` | `string` | `"ingest"` | `ingest`, `appraise`, `relate` | Mode artifact to retrieve |
| `view` | `string` | `"compact"` | `compact`, `detail` | Output view projection |

### Response

See [GET /snapshot](#get-snapshot) for response schema. The response structure varies by mode:

- **ingest**: Returns full snapshot + events + stats bundle
- **appraise**: Returns appraise result with verdict and evidence
- **relate**: Returns related nodes and edges

---

## GET /status

Returns runtime statistics from the Rust core engine.

### Request

```http
GET /status HTTP/1.1
```

### Response (200 OK)

```json
{
  "ok": true,
  "status": {
    "total_nodes": 30,
    "total_atoms": 28,
    "total_contexts": 42,
    "warmed_up": true,
    "watchlist_count": 0,
    "theta_assign": 0.25,
    "theta_merge": 0.55
  },
  "backend": "rust"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_nodes` | `integer` | Total nodes in the knowledge graph |
| `total_atoms` | `integer` | Total atom entries |
| `total_contexts` | `integer` | Total processed contexts |
| `warmed_up` | `boolean` | Whether the system has passed warm-up phase |
| `watchlist_count` | `integer` | Number of nodes on the governance watchlist |
| `theta_assign` | `float` | Current adaptive sense assignment threshold |
| `theta_merge` | `float` | Current adaptive sense merge threshold |

---

## Error Handling

All error responses follow a consistent format:

```json
{
  "ok": false,
  "error": "<error_code>",
  "mode": "<mode_if_applicable>"
}
```

### HTTP Status Code Mapping

| Status | RSVS Exception | Condition |
|--------|---------------|-----------|
| 400 | `InvalidModeError` | Unsupported mode specified |
| 400 | — | Malformed JSON, missing required fields |
| 404 | — | Endpoint not found, no artifacts available |
| 409 | `SchemaVersionMismatchError` | Client schema version doesn't match server |
| 422 | `SchemaValidationError` | Request data fails schema validation |
| 500 | — | Unhandled internal error |
| 503 | `RustCoreUnavailableError` | Rust core (PyO3) is not loaded |

### Exception Hierarchy

```
RsvsError (base)
├── SchemaVersionMismatchError    → 409
├── SchemaValidationError         → 422
├── InvalidModeError              → 400
└── RustCoreUnavailableError      → 503
```

---

## Type Reference

### Node

| Field | Type | Description |
|-------|------|-------------|
| `id` | `integer` | Unique node identifier |
| `label` | `string` | Canonical label (e.g. `"water"`) |
| `surface_label` | `string` | Display label with language (e.g. `"water@en"`) |
| `language_links` | `LanguageLink[]` | Cross-language references |
| `kind` | `string` | Always `"node"` in v4.2 |
| `tier` | `1 \| 2 \| 3` | Node tier (1=Autonomous, 2=Flagged, 3=Blocked) |
| `confidence` | `float` | Confidence score [0.0, 1.0] |
| `status` | `string` | Node status: `new`, `candidate`, `stable`, `deprecated`, `quarantine` |
| `is_seed` | `boolean` | Whether this is an immutable seed atom |
| `is_locked` | `boolean` | Whether the node is protected from deletion |
| `semantic` | `SemanticMeta` | Compression and provenance metadata |
| `policy_meta` | `PolicyMeta` | Governance metadata |
| `provenance` | `Provenance` | Source tracking metadata |

### SemanticMeta

| Field | Type | Description |
|-------|------|-------------|
| `compression_state` | `string` | `"raw"` or `"compressed"` |
| `derived_from_node_ids` | `integer[]` | IDs of nodes this was compressed from |
| `compression_reason` | `string \| null` | Reason for compression |

### PolicyMeta

| Field | Type | Description |
|-------|------|-------------|
| `policy_version` | `string` | Policy schema version (`"v4.2"`) |
| `governance_score` | `float` | Composite governance score [0.0, 1.0] |
| `candidate_evidence_pool` | `float` | Accumulated evidence |
| `status_flip_count` | `integer` | Number of status transitions |
| `seen_fingerprints` | `string[]` | Dedup tracking |
| `short_window_hits` | `integer` | Recent observation count (short window) |
| `long_window_hits` | `integer` | Recent observation count (long window) |
| `last_seen_at` | `string` | ISO 8601 timestamp of last observation |

### Edge

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Edge identifier |
| `source` | `integer` | Source node ID |
| `target` | `integer` | Target node ID |
| `direction` | `string` | Always `"undirected"` |
| `weight` | `float` | Edge weight [0.0, 1.0] |
| `source_type` | `string` | `"bootstrap"` or `"learned"` |
| `status` | `string` | Edge status |

### Event

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | `string` | Unique event identifier |
| `timestamp` | `string` | ISO 8601 timestamp |
| `correlation_id` | `string` | Correlation ID for tracing |
| `event_type` | `string` | Event type (see table above) |
| `payload` | `object` | Event-specific data |
| `animation_hint` | `object` | Frontend animation hints |
| `seq` | `integer` | Sequence number |
| `api_version` | `string` | API version |
| `schema_version` | `string` | Schema version |

### LanguageLink

| Field | Type | Description |
|-------|------|-------------|
| `lang` | `string` | ISO 639-1 language code |
| `label` | `string` | Label in the target language |
| `confidence` | `float` | Translation confidence [0.0, 1.0] |

### Provenance

| Field | Type | Description |
|-------|------|-------------|
| `source_batch_id` | `string` | Batch correlation ID |
| `source_domain` | `string` | Domain: `core_seed` or `rsvs_core` |
| `source_type` | `string` | `bootstrap` or `learned` |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RSVS_BRIDGE_HOST` | `127.0.0.1` | Server bind address |
| `RSVS_BRIDGE_PORT` | `8787` | Server bind port |
| `RSVS_ATOM_OUTPUT_DIR` | `../atom` | Artifact output directory |
| `RSVS_ATTENTION_CONFIG` | — | Path to JSON config for attention weights override |
