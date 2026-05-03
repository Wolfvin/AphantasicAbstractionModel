# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 6.1.x   | :white_check_mark: |
| 6.0.x   | :white_check_mark: |
| < 5.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability:

1. **Do NOT** open a public issue
2. Email the maintainer directly or use GitHub's private vulnerability reporting
3. Include: description, reproduction steps, affected versions, potential impact

We will:
- Acknowledge within 48 hours
- Provide an initial assessment within 7 days
- Release a fix as soon as possible
- Credit you in the security advisory (unless you prefer to remain anonymous)

## Security Features

### Core Protections

- **Seed atom immutability**: 24 primitive atoms are locked at tier 1 with confidence 1.0 and cannot be modified or deleted
- **Policy engine**: Single-owner governance with hysteresis anti-flip-flop protection
- **Quarantine system**: Suspicious nodes are automatically quarantined before affecting the graph
- **Stability gate**: Global rollback mechanism prevents cascading corruption
- **DAG invariant**: The graph maintains a directed acyclic structure with circular reference detection
- **Inactivity TTL** (v6.1): Atoms not seen within 50 contexts are aggressively decayed and moved to Tier3, preventing zombie atom accumulation

### Traversal Safety (v6.1)

- **Cycle detection**: `HashSet<(NodeId, SenseId)>` tracks visited nodes during recursive composition expansion, preventing infinite loops
- **Depth safety net**: `max_depth` parameter prevents unbounded recursion
- **Adaptive halting**: Stability, confidence, and relevance-gating criteria automatically stop traversal when sufficient evidence is gathered
- **Relevance gating**: Only nodes with `similarity(node, query_context) >= tau_relevance` are expanded, preventing exponential blowup in dense graphs

### API Security (v6.1)

- **API Key authentication**: Set `RSVS_API_KEY` environment variable to enable. All endpoints require `X-API-Key` header when configured. In development (no key set), auth is skipped.
- **CORS whitelist**: Origins are configured via `RSVS_ALLOWED_ORIGINS` (comma-separated). Defaults to `http://localhost:3000`. **Never use `*` in production.**
- **Rate limiting**: All endpoints are rate-limited (30/min for mutations, 60/min for reads). Uses `slowapi` with per-IP tracking.
- **Request size limit**: Maximum 1MB request body enforced via middleware.
- **Input validation**: All text fields have `max_length` constraints (100K characters for text, 500 for source/target). Schema validation via Pydantic.
- **Context query validation** (v6.1): `max_depth` bounded [1, 10], thresholds bounded [0, 1], context atoms require at least 1 entry.

### Cryptographic Notes

- **Fingerprint hasher** (v6.1): Uses `XxHash64` — a fast, deterministic, cross-version-stable hash algorithm. Suitable for both in-session deduplication and persistent content-addressable storage across restarts and Rust compiler versions.
- **ID generation**: Uses `uuid4` for collision-resistant unique identifiers in the Python bridge.
- **Note**: Prior versions used `DefaultHasher` which was NOT stable across Rust compiler versions. The v6.1 migration to XxHash64 resolves this.
