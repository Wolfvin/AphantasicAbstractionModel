# FastAPI Server

RSVS includes an optional FastAPI server for HTTP access to all operations. Install with the `server` extra:

```bash
pip install rsvs[server]
```

---

## Starting the Server

```bash
# Development mode (auto-reload)
RSVS_DEV_RELOAD=1 python -m rsvs.fastapi_server

# Production mode
RSVS_API_KEY=your-secret-key \
RSVS_SESSION_SECRET=your-session-secret \
python -m rsvs.fastapi_server
```

The server runs on `0.0.0.0:8000` by default.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RSVS_HOST` | `0.0.0.0` | Server bind address (alias: `RSVS_BRIDGE_HOST`) |
| `RSVS_PORT` | `8000` | Server bind port (alias: `RSVS_BRIDGE_PORT`) |
| `RSVS_API_KEY` | — | API key for authentication |
| `RSVS_SESSION_SECRET` | — | Session secret for production |
| `RSVS_ALLOWED_ORIGINS` | — | CORS allowed origins (comma-separated) |
| `RSVS_DEV_RELOAD` | — | Enable auto-reload for development |

---

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Ingest text into the knowledge graph |
| `POST` | `/compose` | Create a compositional node |
| `GET` | `/structural-similarity` | Structural similarity between two concepts |
| `GET` | `/substitution-analysis` | Substitution analysis between two concepts |
| `POST` | `/mcts-query` | MCTS reasoning query |
| `POST` | `/consolidate` | Run graph consolidation |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | OpenAPI documentation |

---

## Authentication

All endpoints require an API key when `RSVS_API_KEY` is set. Include it as a header:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/ingest
```

---

## Rate Limiting

Requests are rate-limited per API key (or per IP if no key is present). Default: 100 requests/minute. Configurable via environment variable.

---

## Example Requests

```bash
# Ingest text
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Raja adalah pemimpin kerajaan."}'

# Structural similarity
curl "http://localhost:8000/structural-similarity?a=raja&b=ratu"

# Substitution analysis
curl "http://localhost:8000/substitution-analysis?a=raja&b=ratu"

# MCTS query
curl -X POST http://localhost:8000/mcts-query \
  -H "Content-Type: application/json" \
  -d '{"label": "batu", "simulations": 100}'
```
