# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
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

### API Security (v6.0)

- **API Key authentication**: Set `RSVS_API_KEY` environment variable to enable. All endpoints require `X-API-Key` header when configured. In development (no key set), auth is skipped.
- **CORS whitelist**: Origins are configured via `RSVS_ALLOWED_ORIGINS` (comma-separated). Defaults to `http://localhost:3000`. **Never use `*` in production.**
- **Rate limiting**: All endpoints are rate-limited (30/min for mutations, 60/min for reads). Uses `slowapi` with per-IP tracking.
- **Request size limit**: Maximum 1MB request body enforced via middleware.
- **Input validation**: All text fields have `max_length` constraints (100K characters for text, 500 for source/target). Schema validation via Pydantic.

### Cryptographic Notes

- **Fingerprint hasher**: Uses `std::collections::hash_map::DefaultHasher` (labeled `"std_default_hasher"`). This is NOT a stable hash across Rust compiler versions — suitable only for in-session deduplication, not persistent content-addressable storage.
- **ID generation**: Uses `uuid4` for collision-resistant unique identifiers in the Python bridge.
