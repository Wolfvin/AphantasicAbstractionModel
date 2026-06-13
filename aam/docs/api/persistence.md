# Persistence Operations

Methods for saving, loading, and streaming knowledge graph state.

---

## `save()`

Serialize the entire knowledge graph to a JSON file.

```python
r.save(path: str) -> None
```

The JSON file contains the complete graph state: all nodes, edges, senses, autonomy records, co-occurrence statistics, and configuration parameters. The file is self-contained and can be loaded in a different session.

---

## `load()`

Class method. Deserialize a knowledge graph from a JSON file.

```python
r = Rsvs.load(path: str) -> Rsvs
```

Reconstructs the full graph state including all Rust-side data structures. The loaded instance is identical to the one that was saved.

---

## `snapshot_v1()`

Runtime snapshot for UI consumption.

```python
json_string = r.snapshot_v1() -> str
```

Returns a JSON string containing the current graph state in the v1 snapshot format. This is used by the frontend for 3D visualization and is designed for low-latency consumption.

---

## `consume_events_v1()`

Incremental event stream. Returns events with sequence numbers after the specified position.

```python
events_json = r.consume_events_v1(
    after_seq: int | None = None,
    limit: int = 100,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `after_seq` | `int | None` | None | Return events after this sequence number |
| `limit` | `int` | 100 | Maximum number of events to return |

Events are consumed (one-time read) from the Rust core's event buffer. Each event includes a monotonic sequence number, correlation ID, timestamp, event type, and payload.

---

## `latest_seq_v1()`

Current monotonic event sequence number.

```python
seq = r.latest_seq_v1() -> int
```

Use this to track the event stream position. Pass the result as `after_seq` in subsequent calls to `consume_events_v1()` to get only new events.

---

## Event Schema

```json
{
    "api_version": "v8.3",
    "schema_version": "v8.3",
    "seq": 42,
    "correlation_id": "corr_abc123",
    "event_type": "node_created",
    "payload": { "id": 25, "label": "batu" }
}
```

### Event Types

| Type | Trigger | Priority |
|------|---------|----------|
| `node_created` | New node inserted | normal |
| `confidence_changed` | Confidence updated | low |
| `tier_changed` | Tier promoted/demoted | low |
| `status_changed` | Status transition | low |
| `edge_created` | New edge added | low |
