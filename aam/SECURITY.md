# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 7.2.x   | :white_check_mark: |
| 7.1.x   | :white_check_mark: |
| 7.0.x   | :white_check_mark: |
| 6.x     | :x: (upgrade required) |
| < 6.0   | :x:                |

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
- **Neuro-symbolic verification** (v7.2): Every composition is automatically verified for structural invariants (no self-reference, layer consistency, grounding, frequency, no circular chains) upon creation

### Traversal Safety (v6.1+)

- **Cycle detection**: `HashSet<(NodeId, SenseId)>` tracks visited nodes during recursive composition expansion, preventing infinite loops
- **Depth safety net**: `max_depth` parameter prevents unbounded recursion
- **Adaptive halting**: Stability, confidence, and relevance-gating criteria automatically stop traversal when sufficient evidence is gathered
- **Relevance gating**: Only nodes with `similarity(node, query_context) >= tau_relevance` are expanded, preventing exponential blowup in dense graphs
- **Paradigm routing** (v7.2): Queries are routed through ParadigmRouter (Direct → Shallow → Standard → Deep → MCTS) before ThinkingToggle fine-tunes depth, ensuring computation is proportional to query complexity

### API Security (v7.2)

- **API Key proxy**: All frontend-backend calls go through `/api/proxy/[...path]` Next.js API route. The API key is injected server-side — the browser NEVER sees it. No `NEXT_PUBLIC_` prefixed secrets exist in the frontend bundle.
- **API Key authentication**: Set `RSVS_API_KEY` environment variable to enable. All endpoints require `X-API-Key` header when configured. In development (no key set), auth is skipped.
- **Centralized error handling**: A global `@app.exception_handler(Exception)` returns generic `{"error": "internal_error"}` to clients. Full stack traces and error details are logged server-side only. Custom `RsvsError` handler returns only the error class name (e.g., `{"error": "NodeNotFoundError"}`), never internal details like Rust module names or file paths.
- **CORS whitelist**: Origins are configured via `RSVS_ALLOWED_ORIGINS` (comma-separated). Defaults to `http://localhost:3000`. **Never use `*` in production.**
- **Rate limiting — API-key based**: All endpoints are rate-limited. When an `X-API-Key` header is present, rate limits key by that key rather than IP address. This prevents proxy-rotation bypass. Limits: 30/min for mutations, 60/min for reads, 5/min for heavy operations (consolidation, reflection), 10/min for MCTS queries.
- **Request size limit**: Maximum 1MB request body enforced at both FastAPI middleware and nginx (`client_max_body_size 1m`).
- **Input validation**: All text fields have `max_length` constraints (100K characters for text, 500 for source/target). Schema validation via Pydantic with strict type checking.
- **Context query validation** (v6.1): `max_depth` bounded [1, 10], thresholds bounded [0, 1], context atoms require at least 1 entry.

### Transport Security (v7.1+)

- **HTTPS support**: Certbot service included in docker-compose.yml for automatic Let's Encrypt certificate management. ACME challenge path configured in nginx.
- **TLS configuration**: When HTTPS is enabled, nginx is configured with TLSv1.2/1.3 only, `HIGH:!aNULL:!MD5` cipher suite, and server cipher preference enabled.
- **HTTP→HTTPS redirect**: Configure by uncommenting the HTTPS server block in nginx.conf after obtaining certificates.
- **HSTS**: `Strict-Transport-Security: max-age=63072000; includeSubDomains` header set when HTTPS is active.

### Security Headers (v7.1+)

- **Content-Security-Policy**: `default-src 'self'; script-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'` — restricts all resource loading to same-origin by default
- **X-Content-Type-Options**: `nosniff` — prevents MIME type sniffing
- **X-Frame-Options**: `DENY` — prevents clickjacking via iframe embedding
- **X-XSS-Protection**: `1; mode=block` — enables browser XSS filter
- **Referrer-Policy**: `strict-origin-when-cross-origin` — limits referrer information leakage

### Data Integrity (v7.1+)

- **Atomic persistence**: State files are written to a `.tmp` file first, then atomically renamed via `std::fs::rename`. On crash, the state file is always either the complete old version or the complete new version — never a partial write.
- **Fingerprint hasher** (v6.1): Uses `XxHash64` — a fast, deterministic, cross-version-stable hash algorithm. Suitable for both in-session deduplication and persistent content-addressable storage across restarts and Rust compiler versions.
- **ID generation**: Uses `uuid4` for collision-resistant unique identifiers in the Python bridge.

### Infrastructure Security

- **No direct backend exposure**: Backend (port 8000) and frontend (port 3000) are only exposed internally within the Docker network. Only nginx (ports 80/443) is exposed to the host.
- **Docker health checks**: All services have health checks with appropriate start periods and retry counts.
- **Environment-based configuration**: All secrets (API keys, allowed origins) are configured via environment variables, never hardcoded.
